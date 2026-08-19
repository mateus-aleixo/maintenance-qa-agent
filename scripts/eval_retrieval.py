"""Retrieval evaluation against the manually grounded golden set.

Answerable questions carry a gold document + page, written by reading the source
chunk rather than by asking the retriever where it would look — otherwise the
metric only measures its own tie-breaking.

A hit means some retrieved chunk lands on the gold page, within `--slack` pages.
Page-level rather than chunk-level, because chunks overlap and a boundary can
split an answer; scoring on chunk identity would penalise a retrieval that
returned the right text.

Unanswerable questions have no gold. What they measure is *separation*: the
confidence the pipeline assigns them versus answerable ones. That gap is what
the conformal gate turns into a calibrated abstention rule in M4.

    python scripts/eval_retrieval.py [--k 5] [--slack 1] [--embedder hash|bge]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from conformal_rag.answer import confidence_from_hits
from conformal_rag.config import DEFAULT
from conformal_rag.embed import get_embedder
from conformal_rag.retrieve import retrieve
from conformal_rag.store import Store

GOLDEN = Path(__file__).parent.parent / "evals" / "golden.jsonl"


def load_golden(paths: list[Path] | None = None) -> list[dict]:
    rows = []
    for p in paths or [GOLDEN]:
        rows += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--slack", type=int, default=1, help="page tolerance either side")
    ap.add_argument("--embedder", default="hash", choices=["hash", "bge"])
    ap.add_argument("--json", type=Path, default=None, help="write raw per-question results")
    ap.add_argument("--golden", type=Path, nargs="+", default=None,
                    help="golden set file(s); defaults to the hand-written evals/golden.jsonl")
    a = ap.parse_args(argv)

    store = Store(DEFAULT.db_path)
    emb = get_embedder(a.embedder)
    rows = load_golden(a.golden)

    results, hits_by_source = [], {"bm25": 0, "vec": 0, "both": 0}
    for r in rows:
        hs = retrieve(store, emb, r["question"], DEFAULT.k_bm25, DEFAULT.k_vec, a.k, DEFAULT.rrf_k)
        conf = confidence_from_hits(hs)
        rank = None
        if r["answerable"]:
            for i, h in enumerate(hs, start=1):
                if h.doc == r["gold_doc"] and abs(h.page - r["gold_page"]) <= a.slack:
                    rank = i
                    break
        if hs:
            top = hs[0]
            key = "both" if len(top.sources) > 1 else top.sources[0]
            hits_by_source[key] = hits_by_source.get(key, 0) + 1
        results.append({**{k: r[k] for k in ("id", "type", "answerable")},
                        "rank": rank, "confidence": conf,
                        "top": f"{hs[0].doc[:18]} p.{hs[0].page}" if hs else None})

    ans = [x for x in results if x["answerable"]]
    una = [x for x in results if not x["answerable"]]
    found = [x for x in ans if x["rank"]]

    print(f"corpus: {store.count()} chunks | embedder: {a.embedder} | k={a.k} | page slack ±{a.slack}")
    print(f"golden set: {len(ans)} answerable, {len(una)} unanswerable\n")
    print(f"  recall@{a.k}          {len(found)}/{len(ans)}  = {len(found)/len(ans):.2f}")
    r1 = sum(1 for x in ans if x["rank"] == 1)
    print(f"  recall@1           {r1}/{len(ans)}  = {r1/len(ans):.2f}")
    if found:
        print(f"  MRR                {statistics.mean(1/x['rank'] for x in found):.3f} (over found)")
    print(f"  top-hit source     {hits_by_source}")
    print()
    print(f"  mean confidence, answerable    {statistics.mean(x['confidence'] for x in ans):.4f}")
    print(f"  mean confidence, unanswerable  {statistics.mean(x['confidence'] for x in una):.4f}")
    print("  ^ the gap here is what the conformal gate calibrates in M4\n")

    misses = [x for x in ans if not x["rank"]]
    if misses:
        print("  missed:")
        for m in misses:
            print(f"    {m['id']} ({m['type']}) conf={m['confidence']:.3f} top={m['top']}")

    if a.json:
        a.json.write_text(json.dumps(results, indent=2))
        print(f"\n  raw results -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
