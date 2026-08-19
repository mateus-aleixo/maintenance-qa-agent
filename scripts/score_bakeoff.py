"""Compare candidate nonconformity scores on one criterion: do they see correctness?

M4's gate failed because `support_score` separated answerable from unanswerable
questions but not correct from incorrect answers. This scores the same 100
questions with every candidate and reports, for each, the gap between correct
and incorrect answers — plus AUC, which is the threshold-free version of the
same question and the one that decides whether a gate can work at all.

Reuses the verdicts already cached in runs/gate_scores.json, so nothing is
re-judged and the comparison is like-for-like.

    python scripts/score_bakeoff.py --k 3
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from conformal_rag.answer import _SYSTEM, build_prompt
from conformal_rag.config import DEFAULT
from conformal_rag.embed import get_embedder
from conformal_rag.llm import get_llm
from conformal_rag.retrieve import retrieve
from conformal_rag.scores import (
    groundedness_score,
    self_consistency_score,
    support_score_v2,
)
from conformal_rag.store import Store

ROOT = Path(__file__).parent.parent


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """P(score of a correct answer > score of an incorrect one). 0.5 = blind."""
    pos, neg = scores[positive == 1], scores[positive == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3, help="self-consistency samples")
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--embedder", default="bge")
    ap.add_argument("--cached", type=Path, default=ROOT / "runs" / "gate_scores.json")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "bakeoff.json")
    a = ap.parse_args(argv)

    cfg = replace(DEFAULT, llm_provider=a.provider)
    store, emb, llm = Store(cfg.db_path), get_embedder(a.embedder), get_llm(cfg)

    cached = {r["id"]: r for r in json.loads(a.cached.read_text())}
    rows = []
    for p in ["golden.jsonl", "golden_generated.jsonl"]:
        rows += [json.loads(line) for line in (ROOT / "evals" / p).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r["id"] in cached]

    # Resume: this run costs ~6 model calls per question, so losing it to an
    # interruption is expensive. Partial results are flushed as we go and reloaded
    # on restart. (Learned the hard way — an earlier run died at 80/100 having
    # written nothing.)
    out = []
    if a.out.exists():
        out = json.loads(a.out.read_text())
        print(f"resuming: {len(out)} already scored in {a.out}")
    done = {x["id"] for x in out}
    todo = [r for r in rows if r["id"] not in done]

    t0 = time.perf_counter()
    for n, r in enumerate(todo, start=1):
        prev = cached[r["id"]]
        hits = retrieve(store, emb, r["question"], cfg.k_bm25, cfg.k_vec, cfg.k_final, cfg.rrf_k)
        prompt = build_prompt(r["question"], hits)
        # Reuse the answer the scoring run already produced when it stored one.
        # Groundedness only needs *an* answer from this generator, and at 14 B a
        # regenerated one costs ~19 s to say the same thing.
        ans = prev.get("answer") or (llm.complete(_SYSTEM, prompt).text or "")

        sc, _ = self_consistency_score(_SYSTEM, prompt, llm, k=a.k)
        out.append({
            "id": r["id"], "answerable": r["answerable"], "loss": prev["loss"],
            "verdict": prev["verdict"],
            "support_v1": prev["score"],
            "support_v2": support_score_v2(r["question"], hits, llm),
            "groundedness": groundedness_score(ans, hits, llm),
            "self_consistency": sc,
        })
        if n % 5 == 0 or n == len(todo):
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(out, indent=2))
            print(f"  {len(out)}/{len(rows)}  ({time.perf_counter()-t0:.0f}s)  [saved]")

    a.out.write_text(json.dumps(out, indent=2))

    keys = ["support_v1", "support_v2", "groundedness", "self_consistency"]
    ansd = [x for x in out if x["answerable"]]
    correct = np.array([1 - x["loss"] for x in ansd])

    print(f"\n{'='*74}\nSeparation on the {len(ansd)} ANSWERABLE questions "
          f"({int(correct.sum())} correct / {int((1-correct).sum())} not)\n")
    print(f"  {'score':<18} {'correct':>8} {'incorrect':>10} {'gap':>7} {'AUC':>7} {'distinct':>9}")
    results = {}
    for k in keys:
        v = np.array([x[k] for x in ansd], dtype=float)
        c, i = v[correct == 1].mean(), v[correct == 0].mean()
        A = auc(v, correct)
        d = len(set(np.round(v, 3)))
        results[k] = {"correct": c, "incorrect": i, "gap": c - i, "auc": A, "distinct": d}
        print(f"  {k:<18} {c:>8.3f} {i:>10.3f} {c-i:>+7.3f} {A:>7.3f} {d:>9}")

    print(f"\nAnswerability separation (all {len(out)} questions), for reference\n")
    for k in keys:
        v = np.array([x[k] for x in out], dtype=float)
        m = np.array([x["answerable"] for x in out])
        print(f"  {k:<18} answerable {v[m].mean():.3f}  unanswerable {v[~m].mean():.3f}"
              f"  gap {v[m].mean()-v[~m].mean():+.3f}")

    best = max(keys, key=lambda k: (results[k]["auc"] if results[k]["auc"] == results[k]["auc"] else 0))
    print(f"\n  best correctness-ranker: {best}  (AUC {results[best]['auc']:.3f})")
    print(f"  written -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
