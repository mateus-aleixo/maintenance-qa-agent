"""Layout, loaders and nested subsets shared by every stage of the benchmark.

Everything heavy lives under data/msmarco/, which is gitignored. MS MARCO is licensed
for non-commercial research, so neither its text nor anything derived from the text
(vectors, indexes, run files) is committed. Only the measured numbers in runs/ are.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("SCALE_DATA", ROOT / "data" / "msmarco"))
RUNS = ROOT / "runs"

N_PASSAGES = 8_841_823
N_DEV_QUERIES = 6_980
N_DEV_QRELS = 7_437
SIZES = {"100k": 100_000, "1m": 1_000_000, "full": N_PASSAGES}
DEPTH = 100  # every arm keeps its top 100: enough for recall@100 and for fusion


class Passages:
    """The collection by pid, read from disk on demand.

    2.9 GB of text held as Python strings is about 3.5 GB of RAM, a quarter of the
    machine this was measured on. So the text sits in one file, an offsets array maps
    pid to bytes, and nothing ever loads the collection whole.
    """

    def __init__(self, data: Path = DATA):
        self.offsets = np.load(data / "offsets.npy", mmap_mode="r")
        self.blob = np.memmap(data / "passages.bin", dtype=np.uint8, mode="r")

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def get(self, pids: Sequence[int] | np.ndarray) -> list[str]:
        out = []
        for p in pids:
            a, b = self.offsets[p], self.offsets[p + 1]
            out.append(self.blob[a:b].tobytes().decode("utf-8"))
        return out

    def span(self, start: int, stop: int) -> list[str]:
        """Passages start..stop-1 in one read, for the sequential passes."""
        offs = np.asarray(self.offsets[start : stop + 1])
        raw = self.blob[offs[0] : offs[-1]].tobytes()
        rel = offs - offs[0]
        return [raw[rel[i] : rel[i + 1]].decode("utf-8") for i in range(stop - start)]


def load_queries() -> tuple[np.ndarray, list[str]]:
    """The dev.small queries, the set MS MARCO's own leaderboard reports MRR@10 on."""
    qids, texts = [], []
    with open(DATA / "queries.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            qids.append(int(qid))
            texts.append(text)
    return np.array(qids, dtype=np.int64), texts


def load_qrels() -> dict[int, set[int]]:
    rel: dict[int, set[int]] = {}
    with open(DATA / "qrels.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            qid, _, pid, _ = line.split()
            rel.setdefault(int(qid), set()).add(int(pid))
    return rel


def subset(size: str) -> np.ndarray:
    """Sorted pids of one corpus size.

    Every subset holds every passage judged relevant to a dev query, plus random
    distractors. The distractors come from one seeded permutation, so 100k sits
    inside 1m and 1m inside full: moving along the curve only ever adds distractors,
    and every query stays answerable at every size.
    """
    n = SIZES[size]
    if n == N_PASSAGES:
        return np.arange(N_PASSAGES, dtype=np.int32)
    path = DATA / f"subset_{size}.npy"
    if path.exists():
        return np.load(path)
    positives = np.array(sorted({p for ps in load_qrels().values() for p in ps}), dtype=np.int32)
    rest = np.setdiff1d(np.arange(N_PASSAGES, dtype=np.int32), positives, assume_unique=True)
    order = np.random.default_rng(0).permutation(len(rest))
    pids = np.sort(np.concatenate([positives, rest[order[: n - len(positives)]]]))
    np.save(path, pids)
    return pids


def latency_sample(n: int = 1000) -> np.ndarray:
    """The dev queries every arm is timed on, one at a time: the same 1,000 for all."""
    return np.sort(np.random.default_rng(1).choice(N_DEV_QUERIES, n, replace=False))


def size_dir(size: str) -> Path:
    d = DATA / size
    d.mkdir(parents=True, exist_ok=True)
    return d


@contextmanager
def timed() -> Iterator[dict]:
    """`with timed() as t: ...` then t["s"] holds the wall time in seconds."""
    t: dict = {}
    start = time.perf_counter()
    try:
        yield t
    finally:
        t["s"] = time.perf_counter() - start


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dir_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def save_run(size: str, name: str, pids: np.ndarray, scores: np.ndarray, **extra) -> None:
    """One arm's top-DEPTH pids and scores per dev query, padded with -1."""
    d = size_dir(size) / "runs"
    d.mkdir(exist_ok=True)
    np.savez(
        d / f"{name}.npz", pids=pids.astype(np.int32), scores=scores.astype(np.float32), **extra
    )


def load_run(size: str, name: str) -> dict:
    with np.load(size_dir(size) / "runs" / f"{name}.npz") as z:
        return {k: z[k] for k in z.files}


def runs_available(size: str) -> list[str]:
    d = size_dir(size) / "runs"
    return sorted(p.stem for p in d.glob("*.npz")) if d.exists() else []


def peak_rss_gb() -> float | None:
    """This process's peak resident memory so far, for the index build reports."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (n, ctypes.c_size_t)
                for n in ("peak_ws", "ws", "qpp", "qp", "qpnp", "qnp", "pagefile", "peak_pagefile")
            ]

        # A pseudo-handle is pointer sized; without these declarations ctypes passes
        # it as a 32-bit int and the call fails.
        current = ctypes.windll.kernel32.GetCurrentProcess
        current.restype = wintypes.HANDLE
        info = ctypes.windll.psapi.GetProcessMemoryInfo
        info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
        c = _Counters()
        c.cb = ctypes.sizeof(_Counters)
        if not info(current(), ctypes.byref(c), c.cb):
            return None
        return round(c.peak_ws / 2**30, 2)
    try:
        import resource
    except ImportError:
        return None
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)  # KiB on Linux


def machine() -> dict:
    """What the numbers were measured on, recorded beside them."""
    cpu = platform.processor()
    ram = None
    if sys.platform == "win32":
        import ctypes
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()

        class _Mem(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong)
                for n in ("total", "avail", "pf_total", "pf_avail", "v_total", "v_avail", "x")
            ]

        m = _Mem()
        m.len = ctypes.sizeof(_Mem)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        ram = m.total
    elif hasattr(os, "sysconf"):
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            cpu = next(
                (ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")), cpu
            )
    info = {
        "cpu": cpu,
        "threads": os.cpu_count(),
        "ram_gb": round(ram / 2**30, 1) if ram else None,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info
