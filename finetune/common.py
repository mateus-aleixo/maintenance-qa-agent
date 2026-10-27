"""Paths, models, and the line between the training data and the test pool.

The test pool is the 206 questions the gate results in docs/results.md are measured
on. The training data comes from the same two manuals, so the line has to be drawn on
purpose: no training question is written from a page within one page of a test
question's gold page, and none is a near-paraphrase of a test question (cosine 0.85
or more under the serving encoder). Both filters are counted in runs/finetune/data.json.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("FINETUNE_DATA", ROOT / "data" / "finetune"))
RUNS = ROOT / "runs" / "finetune"

BASE = "Qwen/Qwen2.5-3B-Instruct"  # the weights Ollama's qwen2.5:3b-instruct is quantised from
TEACHER = "qwen2.5:7b-instruct"  # writes the training questions and checks the labels
JUDGE = "qwen2.5:14b-instruct"  # the fixed judge behind every comparison in docs/results.md
OLLAMA = os.environ.get("CRAG_OLLAMA_URL", "http://localhost:11434")

TEST_SETS = ("golden.jsonl", "golden_generated.jsonl", "golden_v2.jsonl", "golden_v3.jsonl")
REFUSAL = "INSUFFICIENT EVIDENCE"
PAGE_MARGIN = 1
PARAPHRASE_COS = 0.85


def run_dir(tag: str = "") -> Path:
    """One training run's examples and adapter: DATA for the main run, DATA/<tag> for a variant."""
    return DATA / tag if tag else DATA


def tagged(name: str, tag: str = "") -> str:
    """A runs/finetune file name for one run: train.json for the main run, train_d20.json
    for d20."""
    stem, ext = name.rsplit(".", 1)
    return f"{stem}_{tag}.{ext}" if tag else name


def load_test_pool(root: Path = ROOT) -> list[dict]:
    rows = []
    for name in TEST_SETS:
        for line in (root / "evals" / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def excluded_pages(test: list[dict], margin: int = PAGE_MARGIN) -> dict[str, set[int]]:
    """Pages no training question may come from: every gold page and its neighbours."""
    out: dict[str, set[int]] = {}
    for r in test:
        if r.get("answerable") and r.get("gold_page") is not None:
            p = int(r["gold_page"])
            out.setdefault(r["gold_doc"], set()).update(range(p - margin, p + margin + 1))
    return out


def is_refusal(text: str) -> bool:
    return REFUSAL in (text or "").upper()


def with_citation(answer: str, n: int) -> str:
    """The teacher's answer, citing excerpt n the way the system prompt asks."""
    a = answer.strip()
    if a.endswith("."):
        a = a[:-1]
    return f"{a} [{n}]."


def overlaps_verbatim(question: str, source: str, n: int = 8) -> bool:
    """True if the question copies an n-word span straight from its chunk."""
    qw = re.findall(r"[a-z0-9]+", question.lower())
    sw = " ".join(re.findall(r"[a-z0-9]+", source.lower()))
    return any(" ".join(qw[i : i + n]) in sw for i in range(max(0, len(qw) - n + 1)))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


class Ollama:
    """One Ollama model over the native API, with its weights memory-mapped.

    On Windows with CUDA, Ollama loads without mmap by default. For a model split
    between the GPU and system RAM that doubles what its runner holds: the 7B took
    9.7 GB of commit that way and 4.7 GB mapped, on a machine with 13.7 GB. Every call
    sends the same load options, so the runner is never reloaded mid-run, and the
    native route serves the first-token logprobs the scores need.

    The runner also keeps every prompt's KV state in RAM, up to 8 GiB by default
    (llama.cpp's --cache-ram), and Ollama passes no limit. With thousands of distinct
    prompts that grew the 7B's runner from 4.8 to 8.6 GB in under an hour, so the
    experiment talks to its own server started with LLAMA_ARG_CACHE_RAM=0 (see
    finetune/README.md). No prompt repeats here, so the cache never saved any work.
    """

    def __init__(self, model: str, url: str = OLLAMA):
        self.model, self.url = model, url.rstrip("/")

    def _chat(self, system: str, user: str, timeout: float = 600, **extra) -> dict:
        import httpx

        options = {"temperature": 0.0, "use_mmap": True, **extra.pop("options", {})}
        resp = httpx.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "options": options,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **extra,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(data["error"])
        return data

    def complete(self, system: str, user: str, temperature: float = 0.0, seed: int = 0):
        from conformal_rag.llm import LLMResult

        data = self._chat(system, user, options={"temperature": temperature, "seed": seed})
        return LLMResult(
            text=data["message"]["content"],
            model=self.model,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )

    def top_logprobs(self, system: str, user: str, top_k: int = 5):
        """Distribution over the first reply token, as OpenAICompatClient returns it."""
        import math

        from conformal_rag.llm import TokenDist

        data = self._chat(
            system, user, options={"num_predict": 1}, logprobs=True, top_logprobs=top_k
        )
        first = (data.get("logprobs") or [{}])[0]
        return TokenDist(
            {t["token"]: math.exp(t["logprob"]) for t in first.get("top_logprobs", [])}
        )


def linear_memory_attention() -> None:
    """Keep attention memory linear in the sequence length.

    Qwen2.5-3B has 16 query heads and 2 key/value heads. transformers hands SDPA the 2
    heads with enable_gqa=True, but this Windows build of torch has no flash kernel and
    the memory-efficient kernel does not take grouped heads, so SDPA fell back to its
    math kernel, which holds the whole sequence-by-sequence attention matrix: training
    ran out of memory from about 2,350 tokens. Expanding the key/value heads first, as
    transformers already does whenever a mask is passed, lets the memory-efficient
    kernel run. With the math kernel switched off, any other fallback fails loudly
    instead of quietly needing more memory.
    """
    import torch
    from transformers.integrations import sdpa_attention

    sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False
    torch.backends.cuda.enable_math_sdp(False)


def load_base():
    """Qwen2.5-3B-Instruct in 4-bit NF4 with bf16 compute: the training and test setup."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    linear_memory_attention()
    return AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )


def cap_gpu(margin_gib: float = 0.25) -> None:
    """Keep torch inside the card's own memory.

    On Windows the NVIDIA driver backs an over-allocation with shared system RAM
    instead of failing it. An uncapped run reached 9.97 GiB on this 6 GiB card and
    hung the terminal it ran in. Capping the caching allocator at what the device had
    free at start-up turns that into an ordinary out-of-memory error.
    """
    import torch

    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(max(0.1, (free - margin_gib * 2**30) / total))
