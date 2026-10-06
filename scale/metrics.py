"""Ranking metrics over MS MARCO's sparse judgements, and latency percentiles.

A run is an (n_queries, depth) array of pids in rank order, padded with -1. Every
metric averages over all judged queries, so a query with nothing relevant retrieved
counts as a zero rather than dropping out.

MRR@10 is MS MARCO's official passage metric. nDCG@10 is reported as well because it
is what MTEB and BEIR publish, which gives the dense arm an outside number to check
against.
"""

from __future__ import annotations

import numpy as np


def ranks_of_relevant(run: np.ndarray, qids: np.ndarray, qrels: dict[int, set[int]]):
    """For each judged query: a boolean (depth,) hit row and its relevant count."""
    for row, qid in zip(run, qids, strict=True):
        rel = qrels.get(int(qid))
        if rel:
            yield np.fromiter((int(p) in rel for p in row), dtype=bool, count=len(row)), len(rel)


def evaluate(run: np.ndarray, qids: np.ndarray, qrels: dict[int, set[int]]) -> dict:
    mrr10, r10, r100, ndcg10 = [], [], [], []
    discount = 1.0 / np.log2(np.arange(2, 12))
    for hits, n_rel in ranks_of_relevant(run, qids, qrels):
        first = np.flatnonzero(hits[:10])
        mrr10.append(1.0 / (first[0] + 1) if len(first) else 0.0)
        r10.append(hits[:10].sum() / n_rel)
        r100.append(hits[:100].sum() / n_rel)
        dcg = float(discount[: len(hits[:10])] @ hits[:10])
        ndcg10.append(dcg / discount[: min(n_rel, 10)].sum())
    return {
        "queries": len(mrr10),
        "mrr@10": round(float(np.mean(mrr10)), 4),
        "recall@10": round(float(np.mean(r10)), 4),
        "recall@100": round(float(np.mean(r100)), 4),
        "ndcg@10": round(float(np.mean(ndcg10)), 4),
    }


def overlap_at(run: np.ndarray, exact: np.ndarray, k: int = 10) -> float:
    """Share of the exact search's top k that an approximate index also returns.

    This isolates what the index costs from what the embedding model costs: a recall
    drop against the judgements could be either, an overlap drop is only the index.
    """
    shared = [len(np.intersect1d(a[:k], b[:k])) for a, b in zip(run, exact, strict=True)]
    return round(float(np.mean(shared)) / k, 4)


def latency(seconds: np.ndarray | list[float]) -> dict:
    ms = np.asarray(seconds, dtype=np.float64) * 1000.0
    return {
        "n": int(ms.size),
        "p50_ms": round(float(np.percentile(ms, 50)), 2),
        "p95_ms": round(float(np.percentile(ms, 95)), 2),
        "p99_ms": round(float(np.percentile(ms, 99)), 2),
        "mean_ms": round(float(ms.mean()), 2),
    }
