"""Answer the 206 test questions with one generator, the way the served path does.

    python -m finetune.evaluate retrieve                   # once: the excerpts per question
    python -m finetune.evaluate answer hf-base             # Qwen2.5-3B-Instruct, NF4, no adapter
    python -m finetune.evaluate answer hf-tuned            # the same weights plus the adapter
    python -m finetune.evaluate answer hf-tuned-d20        # a variant's adapter (train --tag d20)
    python -m finetune.evaluate answer ollama:qwen2.5:7b-instruct
    python -m finetune.evaluate judge hf-tuned             # the fixed 14B judge

Every generator sees the same five excerpts per question, retrieved once and cached,
and the served system prompt. The reply is greedy and capped at 256 tokens. Recorded
beside it:

  p_decline  probability mass on a first reply token that begins INSUFFICIENT
             EVIDENCE: the generator's own continuous signal for declining
  p_yes      the generator's own P(YES) on the support question, computed by
             scores.support_score_logprob, the score the served gate uses

Both read the top of the first-token distribution (20 tokens for p_decline, 5 for
p_yes), the same way for a local transformers model and for Ollama, so the
two runtimes produce comparable numbers. Output goes to runs/finetune/answers_<name>.json
and is resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

from conformal_rag.answer import _SYSTEM as ANSWER_SYSTEM  # the served prompt, verbatim
from conformal_rag.answer import build_prompt
from conformal_rag.config import DEFAULT
from conformal_rag.judge import judge_answer, loss_from_verdict
from conformal_rag.llm import LLMResult, TokenDist
from conformal_rag.retrieve import Hit
from conformal_rag.scores import support_score_logprob
from conformal_rag.store import Store

from .common import (
    DATA,
    JUDGE,
    OLLAMA,
    RUNS,
    Ollama,
    cap_gpu,
    is_refusal,
    load_base,
    load_test_pool,
    run_dir,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

HITS = DATA / "test_hits.json"
MAX_NEW = 256


def decline_mass(dist: TokenDist) -> float:
    """Probability that the reply starts spelling INSUFFICIENT EVIDENCE.

    Case-sensitive and at least two characters, so "I" and "In" (ordinary openings)
    do not count as the start of a decline.
    """
    return sum(
        p
        for t, p in dist.probs.items()
        if len(t.strip()) >= 2 and "INSUFFICIENT".startswith(t.strip())
    )


# ----------------------------------------------------------------- generators


class HFGenerator:
    """Qwen2.5-3B-Instruct in NF4 through transformers, with or without the adapter."""

    def __init__(self, adapter: Path | None, name: str):
        import torch
        from transformers import AutoTokenizer

        from .common import BASE

        cap_gpu()
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(BASE)
        model = load_base()
        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.eval()
        self.name = name
        self._first: tuple[str, str, TokenDist] | None = None
        torch.cuda.reset_peak_memory_stats()

    def _ids(self, system: str, user: str):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        return self.torch.tensor([ids], device="cuda")

    def _dist(self, logits, top_k: int) -> TokenDist:
        p = logits.float().softmax(-1)
        vals, idx = p.topk(top_k)
        return TokenDist(
            {self.tok.decode([int(i)]): float(v) for v, i in zip(vals, idx, strict=True)}
        )

    def reply(self, system: str, user: str) -> tuple[LLMResult, float]:
        x = self._ids(system, user)
        t = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(
                input_ids=x,
                attention_mask=self.torch.ones_like(x),
                max_new_tokens=MAX_NEW,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.0,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tok.eos_token_id,
            )
        self.torch.cuda.synchronize()
        secs = time.perf_counter() - t
        new = out.sequences[0][x.shape[1] :]
        self._first = (system, user, self._dist(out.scores[0][0], 20))
        text = self.tok.decode(new, skip_special_tokens=True).strip()
        return LLMResult(text, self.name, x.shape[1], len(new)), secs

    def top_logprobs(self, system: str, user: str, top_k: int = 5) -> TokenDist:
        if self._first and self._first[:2] == (system, user) and top_k <= 20:
            return TokenDist(
                dict(sorted(self._first[2].probs.items(), key=lambda kv: -kv[1])[:top_k])
            )
        x = self._ids(system, user)
        with self.torch.no_grad():
            logits = self.model(
                input_ids=x, attention_mask=self.torch.ones_like(x), logits_to_keep=1
            ).logits[0, -1]
        return self._dist(logits, top_k)

    def memory(self) -> dict:
        return {"peak_gib": round(self.torch.cuda.max_memory_allocated() / 2**30, 2)}


class OllamaGenerator:
    """Any Ollama model, through the same memory-mapped client as the teacher and judge."""

    def __init__(self, model: str):
        self.model = model
        self.name = f"ollama-{model.replace(':', '-')}"
        self.llm = Ollama(model)

    def reply(self, system: str, user: str) -> tuple[LLMResult, float]:
        t = time.perf_counter()
        res = self.llm.complete(system, user)
        return res, time.perf_counter() - t

    def top_logprobs(self, system: str, user: str, top_k: int = 5) -> TokenDist:
        return self.llm.top_logprobs(system, user, top_k)

    def memory(self) -> dict:
        ps = httpx.get(f"{OLLAMA}/api/ps", timeout=30).json().get("models", [])
        m = next((m for m in ps if m["name"] == self.model), None)
        if m is None:
            return {}
        return {
            "size_gib": round(m["size"] / 2**30, 2),
            "vram_gib": round(m["size_vram"] / 2**30, 2),
            "context": m.get("context_length"),
        }


def adapter_for(name: str) -> Path | None:
    """hf-base has no adapter; hf-tuned is the main run's, hf-tuned-<tag> a variant's."""
    if name == "hf-base":
        return None
    return run_dir(name.removeprefix("hf-tuned").removeprefix("-")) / "adapter"


def make(name: str):
    if name == "hf-base" or name.startswith("hf-tuned"):
        return HFGenerator(adapter_for(name), name)
    if name.startswith("ollama:"):
        return OllamaGenerator(name.split(":", 1)[1])
    raise SystemExit(f"unknown generator {name!r}")


def out_path(name: str) -> Path:
    return RUNS / f"answers_{name.replace(':', '-')}.json"


# --------------------------------------------------------------------- stages


def stage_retrieve() -> None:
    from conformal_rag.embed import get_embedder
    from conformal_rag.retrieve import retrieve

    store, emb = Store(DEFAULT.db_path), get_embedder("bge")
    hits = {}
    for g in load_test_pool():
        found = retrieve(
            store, emb, g["question"], DEFAULT.k_bm25, DEFAULT.k_vec, DEFAULT.k_final, DEFAULT.rrf_k
        )
        hits[g["id"]] = [h.chunk_id for h in found]
    HITS.parent.mkdir(parents=True, exist_ok=True)
    HITS.write_text(json.dumps(hits, indent=1))
    print(f"{len(hits)} questions -> {HITS}")


def stage_answer(name: str) -> None:
    gen = make(name)
    store = Store(DEFAULT.db_path)
    cached = json.loads(HITS.read_text())
    path = out_path(name)
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {r["id"] for r in rows}
    pool = [g for g in load_test_pool() if g["id"] not in done]
    print(f"{gen.name}: {len(pool)} to answer, {len(done)} cached", flush=True)
    if pool:  # load the model and fill caches before anything is timed
        gen.reply(
            ANSWER_SYSTEM, "Excerpts:\n\n(none)\n\nQuestion: warm-up\n\nAnswer with citations:"
        )

    t0 = time.perf_counter()
    for n, g in enumerate(pool, start=1):
        by_id = {c["id"]: c for c in store.get_chunks(cached[g["id"]])}
        hits = [
            Hit(c, by_id[c]["doc"], by_id[c]["page"], by_id[c]["text"], 0.0, ())
            for c in cached[g["id"]]
        ]
        user = build_prompt(g["question"], hits)
        res, secs = gen.reply(ANSWER_SYSTEM, user)
        rows.append(
            {
                "id": g["id"],
                "answerable": g["answerable"],
                "type": g.get("type", "?"),
                "answer": res.text,
                "refused": is_refusal(res.text),
                "p_decline": round(decline_mass(gen.top_logprobs(ANSWER_SYSTEM, user, 20)), 6),
                "p_yes": round(support_score_logprob(g["question"], hits, gen), 6),
                "seconds": round(secs, 3),
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
            }
        )
        if n % 10 == 0 or n == len(pool):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows, indent=1))
            print(
                f"  {len(rows)}/{len(rows) + len(pool) - n}  ({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
    meta = RUNS / f"memory_{name.replace(':', '-')}.json"
    if pool:
        meta.write_text(json.dumps(gen.memory(), indent=1))


def stage_judge(name: str) -> None:
    path = out_path(name)
    rows = json.loads(path.read_text())
    golden = {g["id"]: g for g in load_test_pool()}
    judge = Ollama(JUDGE)  # mapped: most of the 14B sits in system RAM
    todo = [r for r in rows if "verdict" not in r]
    print(f"judging {len(todo)} of {len(rows)} with {JUDGE}", flush=True)
    t0 = time.perf_counter()
    for n, r in enumerate(todo, start=1):
        g = golden[r["id"]]
        if g["answerable"]:
            r["verdict"] = judge_answer(g["question"], r["answer"], g.get("note", ""), judge)
            r["loss"] = loss_from_verdict(r["verdict"])
        else:
            r["verdict"], r["loss"] = "UNANSWERABLE", 1.0
        r["judge"] = JUDGE
        if n % 10 == 0 or n == len(todo):
            path.write_text(json.dumps(rows, indent=1))
            print(f"  {n}/{len(todo)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    n_ans = sum(r["answerable"] for r in rows)
    correct = sum(r["answerable"] and r["loss"] == 0 for r in rows)
    print(
        f"{name}: {correct}/{n_ans} answerable correct, base error "
        f"{sum(r['loss'] for r in rows) / len(rows):.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["retrieve", "answer", "judge"])
    ap.add_argument("generator", nargs="?")
    a = ap.parse_args(argv)
    if a.stage == "retrieve":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # nothing else needs the GPU here
        stage_retrieve()
    elif a.generator is None:
        ap.error(f"{a.stage} needs a generator")
    elif a.stage == "answer":
        stage_answer(a.generator)
    else:
        stage_judge(a.generator)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
