"""Does more calibration data buy coverage? Held-out test set held FIXED.

Scoring the new hand-written questions on their own gave 65% coverage at
alpha = 0.2 against the mixed set's 30%, which looks like a large win for
better calibration data. It is not directly comparable: the two sets differ in
difficulty as well as size (ungated risk 0.423 vs 0.540), so an easier set
would show higher coverage with no improvement in calibration at all.

This separates the two. One test set, drawn once and never touched; several
calibration sets varying in size and provenance; every gate evaluated on the
same questions. Any difference is then attributable to the calibration data,
because nothing else moved.

    python scripts/calibration_size.py --alpha 0.2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from conformal_rag.conformal import calibrate_threshold, selective_risk

ROOT = Path(__file__).parent.parent


def load(*paths):
    rows = []
    for p in paths:
        rows += json.loads(Path(p).read_text())
    return rows


def fit_and_test(cal, test, alpha):
    cs = np.array([r["score"] for r in cal], float)
    cl = np.array([r["loss"] for r in cal], float)
    ts = np.array([r["score"] for r in test], float)
    tl = np.array([r["loss"] for r in test], float)
    thr = calibrate_threshold(cs, cl, alpha)
    h = selective_risk(ts, tl, thr)
    return {"n_cal": len(cal), "threshold": thr, "risk": h["risk"],
            "coverage": h["answer_rate"], "n_answered": h["n_answered"],
            "met": h["risk"] <= alpha and h["n_answered"] > 0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--test-size", type=int, default=45)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "calibration_size.json")
    a = ap.parse_args(argv)

    mixed = load(ROOT / "runs" / "gate_scores_14b.json")        # 20 hand + 80 generated
    new = load(ROOT / "runs" / "gate_scores_v2_14b.json")       # 52 hand-written
    hand_old = [r for r in mixed if r["id"].startswith("g-")]
    generated = [r for r in mixed if r["id"].startswith("gen-")]
    all_rows = mixed + new

    rng = random.Random(a.seed)
    pool = list(all_rows)
    rng.shuffle(pool)
    test = pool[: a.test_size]
    test_ids = {r["id"] for r in test}

    def avail(rows):
        return [r for r in rows if r["id"] not in test_ids]

    ungated = float(np.mean([r["loss"] for r in test]))
    print(f"alpha = {a.alpha}   fixed test set of {len(test)} questions "
          f"(ungated risk {ungated:.3f})\n")

    sets = {
        "20 hand-written (v1)": avail(hand_old),
        "52 hand-written (v2)": avail(new),
        "72 hand-written (all)": avail(hand_old + new),
        "80 generated": avail(generated),
        f"{len(avail(all_rows))} everything": avail(all_rows),
    }
    # size sweep drawn from the full pool, to separate size from provenance
    full = avail(all_rows)
    rng.shuffle(full)
    for n in (25, 50, 75):
        if n <= len(full):
            sets[f"{n} random (mixed)"] = full[:n]

    results = {}
    print(f"  {'calibration set':<24} {'n':>4} {'thr':>6} {'risk':>7} {'coverage':>9}  met?")
    for name, cal in sets.items():
        if not cal:
            continue
        r = fit_and_test(cal, test, a.alpha)
        results[name] = r
        print(f"  {name:<24} {r['n_cal']:>4} {r['threshold']:>6.2f} {r['risk']:>7.3f}"
              f" {r['coverage']:>8.0%}  {'yes' if r['met'] else 'no'}")

    a.out.write_text(json.dumps({"alpha": a.alpha, "ungated": ungated,
                                 "n_test": len(test), "results": results}, indent=2))
    print(f"\n  written -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
