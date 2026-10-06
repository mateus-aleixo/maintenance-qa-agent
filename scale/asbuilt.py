"""The serving code path, unchanged, pointed at a bigger corpus.

    python -m scale.asbuilt served     # the deployed index, 1,752 chunks
    python -m scale.asbuilt 100k
    python -m scale.asbuilt 1m

This is retrieve.retrieve() over a Store, exactly as /retrieve calls it: FTS5 BM25 in
SQLite, then every stored vector read back out of SQLite (Store.all_embeddings), one
matmul and a full argsort, then RRF. Nothing here is a reimplementation; the corpus is
the only change. Queries are encoded by the served ONNX encoder.

The full collection is not run: Store.all_embeddings would assemble an 8.8M x 384
float32 matrix, 12.6 GB, on every query, on a machine with 15.2 GB of RAM.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from conformal_rag.embed import OnnxEmbedder
from conformal_rag.ingest import Chunk
from conformal_rag.retrieve import retrieve, vector_search
from conformal_rag.store import Store

from .common import (
    ROOT,
    Passages,
    dir_bytes,
    latency_sample,
    load_queries,
    peak_rss_gb,
    size_dir,
    subset,
    timed,
    write_json,
)
from .embed import open_vectors
from .metrics import latency

QUERIES_PER_SIZE = {"served": 200, "100k": 100, "1m": 20}


def build(size: str, rows: int = 50_000) -> dict:
    """A Store holding one subset, filled through the Store's own write path."""
    path = size_dir(size) / "asbuilt.db"
    if path.exists():
        return {"db_bytes": dir_bytes(path)}
    pids = subset(size)
    passages, vecs = Passages(), open_vectors()
    store = Store(path)
    with timed() as t:
        for a in range(0, len(pids), rows):
            p = pids[a : a + rows]
            texts = passages.get(p)
            store.add_chunks(
                [Chunk("msmarco", int(i), int(i), s) for i, s in zip(p, texts, strict=True)]
            )
            # chunk ids are SQLite rowids, assigned 1, 2, 3, ... in insertion order
            ids = np.arange(a + 1, a + 1 + len(p))
            store.add_embeddings(ids, np.asarray(vecs[p], dtype=np.float32))
        store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    store.close()
    return {"build_s": round(t["s"], 1), "db_bytes": dir_bytes(path)}


def measure(size: str) -> dict:
    if size == "served":
        store = Store(ROOT / "models" / "index.db", read_only=True)
        info = {"chunks": store.count()}
    else:
        info = build(size)
        store = Store(size_dir(size) / "asbuilt.db")
        info["chunks"] = store.count()
    enc = OnnxEmbedder(ROOT / "models" / "embedder")
    _, texts = load_queries()
    sample = latency_sample()[: QUERIES_PER_SIZE[size]]
    retrieve(store, enc, texts[sample[0]])  # warm: first FTS5 page reads, ORT session
    total, bm25, vec = [], [], []
    for qi in sample:
        q = texts[qi]
        t0 = time.perf_counter()
        retrieve(store, enc, q)  # k_bm25=20, k_vec=20, k_final=5: the served defaults
        t1 = time.perf_counter()
        store.bm25(q, 20)
        t2 = time.perf_counter()
        vector_search(store, enc, q, 20)
        t3 = time.perf_counter()
        total.append(t1 - t0)
        bm25.append(t2 - t1)
        vec.append(t3 - t2)
    out = {
        "size": size,
        **info,
        "retrieve": latency(total),
        "fts5_bm25": latency(bm25),
        "vector_search": latency(vec),
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(size_dir(size) / "asbuilt.json", out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("size", choices=list(QUERIES_PER_SIZE))
    a = ap.parse_args(argv)
    print(measure(a.size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
