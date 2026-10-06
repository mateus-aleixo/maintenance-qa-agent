"""Score every run of one corpus size, fuse the hybrids, write runs/scale_<size>.json.

    python -m scale.report 1m

Hybrids are fused with conformal_rag.retrieve.rrf_fuse, the function the serving
system uses, over each arm's top 100. A hybrid's latency is the sum, query by query,
of encoding the query, both searches and the fusion: what one request costs when the
two searches run one after the other.

The query encoder is timed the way it is served, the ONNX export on the CPU, and is
checked against the GPU float16 vectors the quality numbers were computed with.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from conformal_rag.retrieve import rrf_fuse

from .common import (
    DATA,
    DEPTH,
    ROOT,
    RUNS,
    SIZES,
    latency_sample,
    load_qrels,
    load_queries,
    load_run,
    machine,
    read_json,
    runs_available,
    size_dir,
    write_json,
)
from .embed import MODEL, QUERIES
from .metrics import evaluate, latency, overlap_at

HYBRIDS = ("exact", "hnsw_ef128", "ivfpq_np64_rerank")
ENCODER = DATA / "query_encoder.json"


def query_encoder() -> dict:
    """Per-query encode time with the served ONNX encoder, measured once per machine."""
    if ENCODER.exists():
        return read_json(ENCODER)
    from conformal_rag.embed import OnnxEmbedder

    enc = OnnxEmbedder(ROOT / "models" / "embedder")
    _, texts = load_queries()
    gpu = np.load(QUERIES)
    sample = latency_sample()
    enc.encode(["warm up"])
    seconds, cos = [], []
    for qi in sample:
        t0 = time.perf_counter()
        v = enc.encode([texts[qi]])[0]
        seconds.append(time.perf_counter() - t0)
        cos.append(float(v @ gpu[qi]) / float(np.linalg.norm(v) * np.linalg.norm(gpu[qi])))
    info = {
        "model": MODEL,
        "runtime": "onnxruntime, CPU",
        "latency": latency(seconds),
        "min_cosine_vs_gpu_float16": round(min(cos), 6),
        "lat_seconds": seconds,
    }
    write_json(ENCODER, info)
    return info


def fuse(
    a: np.ndarray, sa: np.ndarray, b: np.ndarray, sb: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    out = np.full((len(a), DEPTH), -1, dtype=np.int32)
    seconds = np.empty(len(a))
    for r in range(len(a)):
        t0 = time.perf_counter()
        fused = rrf_fuse(
            {
                "bm25": [(int(p), float(s)) for p, s in zip(a[r], sa[r], strict=True) if p >= 0],
                "vec": [(int(p), float(s)) for p, s in zip(b[r], sb[r], strict=True) if p >= 0],
            }
        )[:DEPTH]
        seconds[r] = time.perf_counter() - t0
        out[r, : len(fused)] = [cid for cid, _, _ in fused]
    return out, seconds


def report(size: str) -> dict:
    qrels = load_qrels()
    names = runs_available(size)
    runs = {n: load_run(size, n) for n in names}
    exact = runs.get("exact")
    d = size_dir(size)
    builds = {}
    for name, path in (
        ("bm25", d / "bm25" / "build.json"),
        ("hnsw", d / "hnsw.json"),
        ("ivfpq", d / "ivfpq.json"),
    ):
        if path.exists():
            builds[name] = read_json(path)
    enc = query_encoder()
    enc_s = np.asarray(enc["lat_seconds"])
    sample = latency_sample()

    arms = {}
    for name, run in runs.items():
        arm: dict = {}
        if "pids" in run:
            arm["quality"] = evaluate(run["pids"], run["qids"], qrels)
            if exact is not None and name not in ("exact", "bm25"):
                arm["overlap@10_vs_exact"] = overlap_at(run["pids"], exact["pids"])
        if "lat_seconds" in run:
            arm["search_latency"] = latency(run["lat_seconds"])
        base = name.split("_")[0]
        if base in builds:
            arm["index"] = {k: v for k, v in builds[base].items() if k not in ("arm", "size")}
        arms[name] = arm
    if (d / "dense_exact.json").exists() and "exact" in arms:
        batch = read_json(d / "dense_exact.json")["batch_s_all_queries"]
        arms["exact"]["batch_s_all_queries"] = batch
    if (d / "dense_brute.json").exists():
        brute = read_json(d / "dense_brute.json")
        arm = arms.setdefault("brute", {})
        arm.update({k: v for k, v in brute.items() if k in ("matrix_gb", "skipped")})
        if exact is not None and "skipped" not in brute:
            arm["quality"] = arms["exact"]["quality"]  # same arithmetic, same ranking

    if "bm25" in runs:
        bm = runs["bm25"]
        for dense in HYBRIDS:
            if dense not in runs:
                continue
            dn = runs[dense]
            fused, fuse_s = fuse(bm["pids"], bm["scores"], dn["pids"], dn["scores"])
            arm = {"quality": evaluate(fused, bm["qids"], qrels)}
            timed_dense = "brute" if dense == "exact" and "brute" in runs else dense
            if "lat_seconds" in runs.get(timed_dense, {}):
                total = (
                    enc_s + bm["lat_seconds"] + runs[timed_dense]["lat_seconds"] + fuse_s[sample]
                )
                arm["end_to_end_latency"] = latency(total)
                arm["timed_as"] = f"onnx encode + bm25 + {timed_dense} + rrf"
            arms[f"hybrid_bm25+{dense}"] = arm

    out = {
        "size": size,
        "passages": SIZES[size],
        "queries": "MS MARCO dev.small",
        "query_encoder": {k: v for k, v in enc.items() if k != "lat_seconds"},
        "arms": arms,
        "as_built": read_json(d / "asbuilt.json") if (d / "asbuilt.json").exists() else None,
        "machine": machine(),
    }
    write_json(RUNS / f"scale_{size}.json", out)
    return out


def table(out: dict) -> str:
    rows = [
        "| arm | MRR@10 | R@10 | R@100 | nDCG@10 | overlap@10 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, arm in out["arms"].items():
        q = arm.get("quality", {})
        lat = arm.get("end_to_end_latency") or arm.get("search_latency") or {}
        cells = [
            q.get("mrr@10"),
            q.get("recall@10"),
            q.get("recall@100"),
            q.get("ndcg@10"),
            arm.get("overlap@10_vs_exact"),
            lat.get("p50_ms"),
            lat.get("p95_ms"),
        ]
        rows.append(f"| {name} | " + " | ".join("" if c is None else str(c) for c in cells) + " |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("size", choices=list(SIZES))
    a = ap.parse_args(argv)
    out = report(a.size)
    print(table(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
