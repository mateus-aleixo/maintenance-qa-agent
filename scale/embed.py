"""Embed the collection with bge-small-en-v1.5 on the GPU, resumably.

    python -m scale.embed 1m       # only the passages the 1m subset needs
    python -m scale.embed full     # all 8.8 million

The encoder and its settings are the serving index's: CLS pooling, L2 normalised, no
query instruction. The forward pass runs in float16, and the vectors are stored in
float16 too: 8.8M x 384 x 2 bytes is 6.3 GB on disk, where float32 would be 12.6 GB,
more than this machine's free RAM. That is the first wall the brute-force design
runs into, before any question of speed.

Vectors go into one memmap indexed by pid, and a done-mask is saved after every
block, so an interrupted run (a closed terminal, a reboot) resumes where it stopped.
Smaller subsets are embedded first, so 100k and 1m can be measured while the rest
of the collection is still running.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .common import DATA, N_PASSAGES, SIZES, Passages, load_queries, subset, write_json

MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384
VECTORS = DATA / "vectors_bge_small_f16.npy"
DONE = DATA / "vectors_done.npy"
QUERIES = DATA / "queries_bge_small.npy"


def open_vectors(mode: str = "r") -> np.ndarray:
    if not VECTORS.exists():
        np.lib.format.open_memmap(VECTORS, mode="w+", dtype=np.float16, shape=(N_PASSAGES, DIM))
    return np.load(VECTORS, mmap_mode=mode)


def done_mask() -> np.ndarray:
    return np.load(DONE) if DONE.exists() else np.zeros(N_PASSAGES, dtype=bool)


def _save_done(done: np.ndarray) -> None:
    tmp = DONE.with_name("vectors_done.tmp.npy")
    np.save(tmp, done)
    tmp.replace(DONE)


def todo(size: str, done: np.ndarray) -> np.ndarray:
    """Pids still missing for `size`, smallest subset first."""
    order = []
    seen = np.zeros(N_PASSAGES, dtype=bool)
    for s in SIZES:
        pids = subset(s)
        pids = pids[~seen[pids]]
        seen[pids] = True
        order.append(pids)
        if s == size:
            break
    pids = np.concatenate(order)
    return pids[~done[pids]]


def load_model():
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device: install a CUDA build of torch (scale/README.md). "
            "On CPU, the full collection would take days."
        )
    model = SentenceTransformer(MODEL, device="cuda")
    model.half()
    return model


def embed_queries(model) -> np.ndarray:
    _, texts = load_queries()
    q = model.encode(texts, batch_size=512, normalize_embeddings=True, convert_to_numpy=True)
    q = q.astype(np.float32)
    np.save(QUERIES, q)
    return q


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("size", choices=list(SIZES))
    ap.add_argument("--block", type=int, default=65_536)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args(argv)

    model = load_model()
    if not QUERIES.exists():
        embed_queries(model)
    done = done_mask()
    pending = todo(a.size, done)
    print(
        f"{a.size}: {len(pending):,} passages to embed, {int(done.sum()):,} already done",
        flush=True,
    )
    if len(pending) == 0:
        return 0
    vecs = open_vectors("r+")
    passages = Passages()
    t0, n = time.perf_counter(), 0
    for start in range(0, len(pending), a.block):
        block = np.sort(pending[start : start + a.block])  # sorted: sequential disk access
        emb = model.encode(
            passages.get(block),
            batch_size=a.batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        norms = np.linalg.norm(emb.astype(np.float32), axis=1)
        if not np.allclose(norms, 1.0, atol=1e-2):
            raise RuntimeError(f"vectors not unit length (min {norms.min():.4f}): check the model")
        vecs[block] = emb.astype(np.float16)
        vecs.flush()
        done[block] = True
        _save_done(done)
        n += len(block)
        el = time.perf_counter() - t0
        rate = n / el
        eta = (len(pending) - n) / rate
        print(
            f"  {n:>9,}/{len(pending):,}  {rate:,.0f} passages/s  eta {eta / 60:.0f} min",
            flush=True,
        )
    stats = {
        "model": MODEL,
        "precision": "float16",
        "passages_embedded_this_run": n,
        "seconds_this_run": round(time.perf_counter() - t0, 1),
        "passages_per_s": round(n / (time.perf_counter() - t0), 1),
        "done_total": int(done.sum()),
    }
    write_json(DATA / f"embed_{a.size}.json", stats)
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
