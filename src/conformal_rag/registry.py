"""Assemble a serving registry from experiment artifacts.

`runs/` is a lab notebook: thirty JSON files from generator sweeps, score bakeoffs
and nested calibration. `models/` is the serving contract: the query encoder, the
corpus, and the one calibrated gate the service actually applies, with the numbers
that justify it.

    python -m conformal_rag.registry --gate gate_v3_14b

The encoder is produced separately by `scripts/export_embedder.py`, because it needs
torch and this does not.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

GATE_MANIFEST = "gate.json"


def _checkpoint_wal(db: Path) -> None:
    """Fold any WAL back into the file and leave it in a non-WAL journal mode.

    The served copy is read-only and immutable. A database still marked WAL in its
    header drags the `-wal`/`-shm` files along as a requirement, which a read-only
    container cannot satisfy, so normalise the artifact at build time rather than
    depending on the reader to work around it.
    """
    import sqlite3

    conn = sqlite3.connect(str(db))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.commit()
    finally:
        conn.close()
    for suffix in ("-wal", "-shm"):
        leftover = db.with_name(db.name + suffix)
        if leftover.exists():
            leftover.unlink()


def build(
    runs_root: Path,
    gate_name: str,
    index_db: Path,
    out_dir: Path,
    generator: str = "qwen2.5:14b-instruct",
    score: str = "support_v1",
) -> Path:
    src = runs_root / f"{gate_name}.json"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing; run scripts/calibrate_gate.py first")
    if not index_db.exists():
        raise FileNotFoundError(f"{index_db} missing; run ingest first")

    art = json.loads(src.read_text())
    gate, held = art["gate"], art["held_out"]

    out_dir.mkdir(parents=True, exist_ok=True)
    served_index = out_dir / "index.db"
    shutil.copy2(index_db, served_index)
    _checkpoint_wal(served_index)

    manifest = {
        "source_artifact": f"{gate_name}.json",
        "generator": generator,
        "score": score,
        "gate": gate,
        "held_out": {
            "risk": held["risk"],
            "answer_rate": held["answer_rate"],
            "n_answered": held["n_answered"],
            "n_total": held["n_total"],
        },
        "ungated_risk": art["ungated_risk"],
        "n_calibration": art["n_cal"],
    }
    (out_dir / GATE_MANIFEST).write_text(json.dumps(manifest, indent=2))
    return out_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gate", default="gate_v3_14b", help="artifact stem under runs/")
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument("--index", type=Path, default=Path("data/index.db"))
    p.add_argument("--models-root", type=Path, default=Path("models"))
    p.add_argument("--generator", default="qwen2.5:14b-instruct")
    p.add_argument("--score", default="support_v1")
    a = p.parse_args(argv)

    out = build(a.runs_root, a.gate, a.index, a.models_root, a.generator, a.score)
    m = json.loads((out / GATE_MANIFEST).read_text())
    print(f"registry: {out}")
    print(f"  gate from {m['source_artifact']}: alpha {m['gate']['alpha']}, "
          f"threshold {m['gate']['global_threshold']}")
    print(f"  held out: risk {m['held_out']['risk']:.3f} at "
          f"{m['held_out']['answer_rate']:.0%} coverage "
          f"(ungated {m['ungated_risk']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
