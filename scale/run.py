"""Every stage for one corpus size, in order, skipping what is already on disk.

    python -m scale.run full

Meant for the long unattended run: each stage is a fresh process, so a failure or
an out-of-memory kill in one stage leaves the finished ones intact, and running the
same command again resumes from the first stage without output.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from .common import DATA, SIZES, size_dir


def stages(size: str) -> list[tuple[list[str], str | None]]:
    """(command, the file that marks it done) in dependency order."""
    d = size_dir(size)
    runs = d / "runs"
    out = [
        (["scale.fetch"], str(DATA / "layout.json")),
        (["scale.embed", size], None),  # resumable on its own; a no-op when complete
        (["scale.bm25", "tokens"], str(DATA / "bm25_tokens" / "meta.json")),
        (["scale.bm25", "build", size], str(d / "bm25" / "build.json")),
        (["scale.bm25", "search", size], str(runs / "bm25.npz")),
        (["scale.dense", "exact", size], str(runs / "exact.npz")),
        (["scale.dense", "brute", size], str(d / "dense_brute.json")),
        (["scale.dense", "ivfpq", size], str(d / "dense_ivfpq.json")),
        (["scale.dense", "hnsw", size], str(d / "dense_hnsw.json")),
    ]
    if size != "full":
        out.append((["scale.asbuilt", size], str(d / "asbuilt.json")))
    out.append((["scale.report", size], None))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("size", choices=list(SIZES))
    ap.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="stage names to leave for later, e.g. hnsw: about 6 GB of RAM at 8.8M passages",
    )
    a = ap.parse_args(argv)
    for cmd, marker in stages(a.size):
        if any(s in cmd for s in a.skip):
            print(f"later {' '.join(cmd)}", flush=True)
            continue
        if marker and Path(marker).exists():
            print(f"skip  {' '.join(cmd)}", flush=True)
            continue
        print(f"{time.strftime('%H:%M:%S')}  {' '.join(cmd)}", flush=True)
        t0 = time.perf_counter()
        rc = subprocess.call([sys.executable, "-u", "-m", *cmd])
        print(f"          exit {rc} after {time.perf_counter() - t0:.0f} s", flush=True)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
