"""Download MS MARCO passage ranking and lay it out for random access.

    python -m scale.fetch

collection.tsv is streamed straight out of the 1 GB tarball into passages.bin plus an
offsets array, so the 2.9 GB TSV is never written to disk. Every count is checked
against the published one before anything downstream is allowed to trust the files.
"""

from __future__ import annotations

import io
import tarfile
import time
from pathlib import Path

import httpx
import numpy as np

from .common import DATA, N_DEV_QRELS, N_DEV_QUERIES, N_PASSAGES, timed, write_json

URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz"
TARBALL_BYTES = 1_057_717_952
KEEP = {"queries.dev.small.tsv": N_DEV_QUERIES, "qrels.dev.small.tsv": N_DEV_QRELS}


def download(dest: Path, attempts: int = 8) -> None:
    """Resumable: a dropped connection continues from the bytes already on disk."""
    if dest.exists() and dest.stat().st_size == TARBALL_BYTES:
        return
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with httpx.stream("GET", URL, headers=headers, timeout=60, follow_redirects=True) as r:
                r.raise_for_status()
                if have and r.status_code != 206:  # server ignored the range: start over
                    have = 0
                with open(part, "ab" if have else "wb") as f:
                    mark = have
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        if have - mark >= 100 << 20:
                            print(
                                f"  {have / TARBALL_BYTES:5.1%} of {TARBALL_BYTES >> 20} MB",
                                flush=True,
                            )
                            mark = have
            break
        except httpx.HTTPError as e:
            print(f"  attempt {attempt}: {type(e).__name__}: {e}; resuming", flush=True)
            time.sleep(5 * attempt)
    else:
        raise RuntimeError(f"download failed after {attempts} attempts")
    if part.stat().st_size != TARBALL_BYTES:
        raise RuntimeError(f"tarball is {part.stat().st_size} bytes, expected {TARBALL_BYTES}")
    part.replace(dest)


def write_collection(src: io.BufferedIOBase) -> None:
    """pid<TAB>text lines become one blob plus an int64 offsets array.

    The bytes are copied as they are, without decoding: MS MARCO carries some
    double-encoded UTF-8 ("â€™" for an apostrophe), and every published BM25 number
    is computed on that text, so cleaning it would make the numbers incomparable.
    """
    offsets = np.empty(N_PASSAGES + 1, dtype=np.int64)
    offsets[0] = 0
    pos, n = 0, 0
    blob = DATA / "passages.bin.part"
    with open(blob, "wb") as out:
        for line in src:
            pid, text = line.rstrip(b"\r\n").split(b"\t", 1)
            if int(pid) != n:
                raise ValueError(f"pid {int(pid)} at line {n}: the offsets array assumes pid order")
            out.write(text)
            pos += len(text)
            n += 1
            offsets[n] = pos
    if n != N_PASSAGES:
        raise ValueError(f"{n} passages, expected {N_PASSAGES}")
    np.save(DATA / "offsets.npy", offsets)
    blob.replace(DATA / "passages.bin")


def unpack(tarball: Path) -> None:
    # Members are matched by base name against a fixed list and written to fixed
    # paths, so a hostile member name cannot place a file anywhere.
    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if name == "collection.tsv":
                print("  collection.tsv -> passages.bin + offsets.npy", flush=True)
                write_collection(tar.extractfile(member))
            elif name in KEEP:
                (DATA / name).write_bytes(tar.extractfile(member).read())


def check() -> dict:
    counts = {}
    for name, expected in KEEP.items():
        with open(DATA / name, encoding="utf-8") as f:
            counts[name] = sum(1 for _ in f)
        if counts[name] != expected:
            raise ValueError(f"{name}: {counts[name]} lines, expected {expected}")
    from .common import Passages

    p = Passages()
    counts["passages"] = len(p)
    if len(p) != N_PASSAGES:
        raise ValueError(f"{len(p)} passages, expected {N_PASSAGES}")
    rng = np.random.default_rng(0)
    p.get(rng.integers(0, N_PASSAGES, 1000))  # raises if any sampled passage is not UTF-8
    return counts


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    tarball = DATA / "collectionandqueries.tar.gz"
    if not (DATA / "passages.bin").exists():
        with timed() as t:
            download(tarball)
        print(f"downloaded in {t['s']:.0f} s", flush=True)
        with timed() as t:
            unpack(tarball)
        print(f"unpacked in {t['s']:.0f} s", flush=True)
    counts = check()
    write_json(DATA / "layout.json", {"source": URL, **counts})
    tarball.unlink(missing_ok=True)  # 1 GB, and the script fetches it again if needed
    print(f"ok: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
