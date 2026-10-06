"""Dense retrieval at scale: exact search and two FAISS indexes over the same vectors.

    python -m scale.dense exact 1m    # GPU, every dev query: the ceiling and the reference
    python -m scale.dense brute 1m    # the serving design: a NumPy matmul per query
    python -m scale.dense hnsw 1m     # build, then search at several efSearch
    python -m scale.dense ivfpq 1m    # build, then search at several nprobe, with a re-rank

exact   Every query against every passage on the GPU, in float32 tiles. Nothing is
        approximated, so this is the quality ceiling for the encoder and the reference
        that the indexes' overlap is measured against.
brute   What retrieve.py does today: the whole matrix in RAM, one matmul and a full
        argsort per query, on the CPU. Its ranking is exact, so only its latency and
        memory are measured.
hnsw    FAISS HNSW, M = 32, over 8-bit scalar-quantised vectors: a graph walk that
        scores a few thousand passages per query instead of all of them.
ivfpq   FAISS IVF with product quantisation: about 4 sqrt(N) clusters and 48 bytes per
        vector, so the full collection fits in about half a gigabyte. The price is
        recall; re-ranking its top 100 exactly, from vectors left on disk, buys most of
        it back.

Quality is computed from batched searches over all 6,980 queries. Latency is timed
separately, one query at a time on one thread, over the same 1,000 queries for every
arm (brute force is the exception: NumPy's BLAS uses every core it finds).
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from .common import (
    DEPTH,
    dir_bytes,
    latency_sample,
    load_queries,
    peak_rss_gb,
    read_json,
    save_run,
    size_dir,
    subset,
    timed,
    write_json,
)
from .embed import DIM, QUERIES, done_mask, open_vectors
from .metrics import latency

EF_SEARCH = (32, 64, 128, 256)
NPROBE = (8, 16, 32, 64, 128)
RERANK_NPROBE = (32, 64)


def _pids(size: str) -> np.ndarray:
    pids = subset(size)
    if not done_mask()[pids].all():
        raise SystemExit(
            f"not every passage of {size} is embedded yet: python -m scale.embed {size}"
        )
    return pids


def batches(pids: np.ndarray, rows: int = 131_072):
    """float32 vectors for `pids`, a block at a time, never the whole matrix."""
    vecs = open_vectors()
    for a in range(0, len(pids), rows):
        p = pids[a : a + rows]
        block = vecs[p[0] : p[-1] + 1] if p[-1] - p[0] + 1 == len(p) else vecs[p]
        yield a, np.asarray(block, dtype=np.float32)


def _sample(pids: np.ndarray, n: int) -> np.ndarray:
    pick = np.sort(np.random.default_rng(0).choice(len(pids), min(n, len(pids)), replace=False))
    return np.asarray(open_vectors()[pids[pick]], dtype=np.float32)


def _save(size: str, name: str, pids: np.ndarray, pos: np.ndarray, scores: np.ndarray, **extra):
    out = np.where(pos >= 0, pids[np.maximum(pos, 0)], -1)
    qids, _ = load_queries()
    save_run(size, name, out, scores, qids=qids, **extra)


# -- exact -------------------------------------------------------------------------


def exact(size: str, k: int = DEPTH, rows: int = 65_536, qtile: int = 2048) -> dict:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False  # full float32: this is the reference
    pids = _pids(size)
    dev = torch.device("cuda")
    q = torch.from_numpy(np.load(QUERIES)).to(dev)
    nq = len(q)
    best_s = torch.full((nq, k), -math.inf, device=dev)
    best_i = torch.full((nq, k), -1, dtype=torch.int64, device=dev)
    vecs = open_vectors()
    torch.cuda.synchronize()
    with timed() as t:
        for a in range(0, len(pids), rows):
            p = pids[a : a + rows]
            block = vecs[p[0] : p[-1] + 1] if p[-1] - p[0] + 1 == len(p) else vecs[p]
            x = torch.from_numpy(np.array(block)).to(dev).float()  # a copy: the memmap is read-only
            for qa in range(0, nq, qtile):
                s, i = torch.topk(q[qa : qa + qtile] @ x.T, min(k, len(p)), dim=1)
                cat_s = torch.cat([best_s[qa : qa + qtile], s], dim=1)
                cat_i = torch.cat([best_i[qa : qa + qtile], i + a], dim=1)
                top_s, j = torch.topk(cat_s, k, dim=1)
                best_s[qa : qa + qtile] = top_s
                best_i[qa : qa + qtile] = torch.gather(cat_i, 1, j)
        torch.cuda.synchronize()
    _save(size, "exact", pids, best_i.cpu().numpy(), best_s.cpu().numpy())
    return {
        "arm": "exact",
        "batch_s_all_queries": round(t["s"], 1),
        "device": torch.cuda.get_device_name(0),
    }


# -- brute force, as served ----------------------------------------------------------


def brute(size: str, k: int = DEPTH, max_gb: float = 4.0) -> dict:
    """retrieve.vector_search's arithmetic, timed per query with the matrix in RAM.

    The serving code also re-reads every vector from SQLite on every query; that cost
    is left out here and measured in scale.asbuilt. This is the best case for brute
    force: the matrix already cached."""
    pids = _pids(size)
    need_gb = len(pids) * DIM * 4 / 2**30
    if need_gb > max_gb:
        return {
            "arm": "brute",
            "skipped": f"float32 matrix needs {need_gb:.1f} GB, over the {max_gb} GB budget",
        }
    m = np.empty((len(pids), DIM), dtype=np.float32)
    for a, x in batches(pids):
        m[a : a + len(x)] = x
    q = np.load(QUERIES)
    seconds = []
    for qi in latency_sample():
        t0 = time.perf_counter()
        sims = m @ q[qi]
        top = np.argsort(-sims)[:k]  # as retrieve.vector_search does
        seconds.append(time.perf_counter() - t0)
        del top
    d = size_dir(size) / "runs"
    d.mkdir(exist_ok=True)
    np.savez(d / "brute.npz", lat_seconds=np.array(seconds))  # ranking is exact's
    return {"arm": "brute", "matrix_gb": round(need_gb, 2), "latency": latency(seconds)}


# -- FAISS ---------------------------------------------------------------------------


def _faiss():
    import faiss

    return faiss


def _search_and_time(index, size: str, name: str, pids: np.ndarray, rerank: bool = False) -> dict:
    faiss = _faiss()
    q = np.load(QUERIES)
    threads = faiss.omp_get_max_threads()
    d, i = index.search(q, DEPTH)  # quality: batched, every thread
    if rerank:
        d, i = _rerank(q, pids, i)
    faiss.omp_set_num_threads(1)
    try:
        seconds = []
        for qi in latency_sample():
            t0 = time.perf_counter()
            dd, ii = index.search(q[qi : qi + 1], DEPTH)
            if rerank:
                _rerank(q[qi : qi + 1], pids, ii)
            seconds.append(time.perf_counter() - t0)
    finally:
        faiss.omp_set_num_threads(threads)
    _save(size, name, pids, i, d, lat_seconds=np.array(seconds))
    return {"arm": name, "latency": latency(seconds)}


def _rerank(q: np.ndarray, pids: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact inner products for each query's candidates, read from the vectors on disk."""
    vecs = open_vectors()
    out_s = np.full(pos.shape, -np.inf, dtype=np.float32)
    out_i = np.full(pos.shape, -1, dtype=np.int64)
    for r in range(len(pos)):
        cand = pos[r][pos[r] >= 0]
        order = np.argsort(pids[cand])  # ascending pid: friendlier to the disk
        cand = cand[order]
        s = np.asarray(vecs[pids[cand]], dtype=np.float32) @ q[r]
        o = np.argsort(-s)
        out_s[r, : len(o)] = s[o]
        out_i[r, : len(o)] = cand[o]
    return out_s, out_i


def _build_report(size: str, name: str, index, path, seconds: float, **extra) -> dict:
    info = {
        "arm": name,
        "size": size,
        "passages": int(index.ntotal),
        "build_s": round(seconds, 1),
        "index_bytes": dir_bytes(path),
        "build_peak_rss_gb": peak_rss_gb(),
        "threads": _faiss().omp_get_max_threads(),
        **extra,
    }
    write_json(path.with_suffix(".json"), info)
    return info


def build_hnsw(size: str, m: int = 32, ef_construction: int = 100) -> dict:
    faiss = _faiss()
    pids = _pids(size)
    path = size_dir(size) / "hnsw.faiss"
    with timed() as t:
        index = faiss.IndexHNSWSQ(DIM, faiss.ScalarQuantizer.QT_8bit, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.train(_sample(pids, 200_000))
        for _, x in batches(pids):
            index.add(x)
    faiss.write_index(index, str(path))
    return _build_report(size, "hnsw", index, path, t["s"], m=m, ef_construction=ef_construction)


def search_hnsw(size: str) -> list[dict]:
    faiss = _faiss()
    pids = _pids(size)
    index = faiss.read_index(str(size_dir(size) / "hnsw.faiss"))
    out = []
    for ef in EF_SEARCH:
        index.hnsw.efSearch = ef
        out.append(_search_and_time(index, size, f"hnsw_ef{ef}", pids))
    return out


def nlist_for(n: int) -> int:
    """About 4 sqrt(N) clusters, to the nearest power of two (FAISS's usual guidance)."""
    return 2 ** round(math.log2(4 * math.sqrt(n)))


def build_ivfpq(size: str, m: int = 48) -> dict:
    faiss = _faiss()
    pids = _pids(size)
    nlist = nlist_for(len(pids))
    path = size_dir(size) / "ivfpq.faiss"
    with timed() as t:
        index = faiss.index_factory(DIM, f"IVF{nlist},PQ{m}", faiss.METRIC_INNER_PRODUCT)
        # 32 training vectors per cluster: inside FAISS's guidance of 30 to 256, though its
        # clustering code prints a warning below 39.
        index.train(_sample(pids, min(len(pids), max(100_000, 32 * nlist))))
        for _, x in batches(pids):
            index.add(x)
    faiss.write_index(index, str(path))
    return _build_report(size, "ivfpq", index, path, t["s"], nlist=nlist, pq_bytes=m)


def search_ivfpq(size: str) -> list[dict]:
    faiss = _faiss()
    pids = _pids(size)
    index = faiss.read_index(str(size_dir(size) / "ivfpq.faiss"))
    out = []
    for nprobe in NPROBE:
        index.nprobe = nprobe
        out.append(_search_and_time(index, size, f"ivfpq_np{nprobe}", pids))
        if nprobe in RERANK_NPROBE:
            out.append(_search_and_time(index, size, f"ivfpq_np{nprobe}_rerank", pids, rerank=True))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["exact", "brute", "hnsw", "ivfpq"])
    ap.add_argument("size")
    ap.add_argument("--search-only", action="store_true", help="reuse the built index")
    a = ap.parse_args(argv)
    d = size_dir(a.size)
    if a.arm == "exact":
        result = exact(a.size)
    elif a.arm == "brute":
        result = brute(a.size)
    elif a.arm == "hnsw":
        build = read_json(d / "hnsw.json") if a.search_only else build_hnsw(a.size)
        result = {"build": build, "search": search_hnsw(a.size)}
    else:
        build = read_json(d / "ivfpq.json") if a.search_only else build_ivfpq(a.size)
        result = {"build": build, "search": search_ivfpq(a.size)}
    write_json(d / f"dense_{a.arm}.json", result)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
