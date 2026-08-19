"""Before/after table for the cross-encoder reranker.

Retrieval fetches a wide candidate list (default 20 fused), the reranker scores
each question/chunk pair jointly and keeps the top k. Identical scoring rule to
eval_retrieval.py — page-level with +/-1 slack — so the two tables are directly
comparable.

    python scripts/eval_rerank.py --golden evals/golden_generated.jsonl
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from eval_retrieval import load_golden  # noqa: E402  (same directory)

from conformal_rag.config import DEFAULT
from conformal_rag.embed import get_embedder
from conformal_rag.rerank import CrossEncoderReranker
from conformal_rag.retrieve import retrieve
from conformal_rag.store import Store


def hit_rank(hits, row, slack):
    for i, h in enumerate(hits, start=1):
        if h.doc == row["gold_doc"] and abs(h.page - row["gold_page"]) <= slack:
            return i
    return None


def summarise(name, ranks, k, elapsed):
    found = [r for r in ranks if r]
    at1 = sum(1 for r in ranks if r == 1)
    mrr = statistics.mean(1 / r for r in found) if found else 0.0
    print(f"  {name:<22} recall@{k} {len(found)}/{len(ranks)} = {len(found)/len(ranks):.2f}"
          f"   recall@1 {at1/len(ranks):.2f}   MRR {mrr:.3f}   {elapsed:.0f}s")
    return {"recall_at_k": len(found) / len(ranks), "recall_at_1": at1 / len(ranks), "mrr": mrr}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, nargs="+", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--candidates", type=int, default=20, help="fused list handed to the reranker")
    ap.add_argument("--slack", type=int, default=1)
    ap.add_argument("--embedder", default="bge")
    a = ap.parse_args(argv)

    store, emb = Store(DEFAULT.db_path), get_embedder(a.embedder)
    rows = [r for r in load_golden(a.golden) if r["answerable"]]
    print(f"corpus {store.count()} chunks · {len(rows)} answerable questions · "
          f"candidates {a.candidates} -> top {a.k}\n")

    t0 = time.perf_counter()
    base_ranks, pools = [], []
    for r in rows:
        wide = retrieve(store, emb, r["question"], DEFAULT.k_bm25, DEFAULT.k_vec,
                        a.candidates, DEFAULT.rrf_k)
        pools.append(wide)
        base_ranks.append(hit_rank(wide[: a.k], r, a.slack))
    t_base = time.perf_counter() - t0
    base = summarise("hybrid (RRF)", base_ranks, a.k, t_base)

    rr = CrossEncoderReranker()
    t0 = time.perf_counter()
    re_ranks = [hit_rank(rr.rerank(r["question"], pool, a.k), r, a.slack)
                for r, pool in zip(rows, pools, strict=True)]
    t_re = time.perf_counter() - t0
    re = summarise("+ cross-encoder", re_ranks, a.k, t_base + t_re)

    d_k = re["recall_at_k"] - base["recall_at_k"]
    d_1 = re["recall_at_1"] - base["recall_at_1"]
    print(f"\n  delta   recall@{a.k} {d_k:+.2f}   recall@1 {d_1:+.2f}   MRR {re['mrr']-base['mrr']:+.3f}")
    print(f"  cost    +{t_re/len(rows)*1000:.0f} ms per question ({rr.name})")

    rescued = [r["id"] for r, b, x in zip(rows, base_ranks, re_ranks, strict=True) if not b and x]
    broken = [r["id"] for r, b, x in zip(rows, base_ranks, re_ranks, strict=True) if b and not x]
    if rescued:
        print(f"\n  rescued (missed before, found after): {', '.join(rescued)}")
    if broken:
        print(f"  BROKEN  (found before, missed after): {', '.join(broken)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
