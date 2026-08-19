"""Honest coverage estimate: pick the safety margin on validation, report on test.

Fitting the threshold at exactly alpha maximises coverage and sits right where
the estimate is noisiest — on the logprob score it produced a threshold of 0.000
and a test risk of 0.217 against a target of 0.20. Fitting at a slightly stricter
alpha buys a margin, but the margin has to be *chosen*, and choosing it by
looking at the test set is the same selection bias this repo already documented
once.

So: three splits. **cal** fits the threshold, **val** chooses the margin, **test**
is looked at exactly once, at the end. Repeated over many random splits, because
a single 50-item split of a 152-item set is mostly noise.

    python scripts/nested_gate.py --alpha 0.2 --repeats 200
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
MARGINS = [0.20, 0.175, 0.15, 0.125, 0.10, 0.075, 0.05]


def one_trial(rows, key, alpha, rng):
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n = len(idx) // 3
    cal = [rows[i] for i in idx[:n]]
    val = [rows[i] for i in idx[n:2 * n]]
    test = [rows[i] for i in idx[2 * n:]]

    def arr(part):
        return (np.array([r[key] for r in part], float),
                np.array([r["loss"] for r in part], float))

    cs, cl = arr(cal)
    vs, vl = arr(val)
    ts, tl = arr(test)

    # choose the margin on VALIDATION: most coverage among margins that hold alpha
    best, best_cov = None, -1.0
    for m in MARGINS:
        thr = calibrate_threshold(cs, cl, m)
        h = selective_risk(vs, vl, thr)
        if h["n_answered"] > 0 and h["risk"] <= alpha and h["answer_rate"] > best_cov:
            best, best_cov = m, h["answer_rate"]
    if best is None:
        return None

    thr = calibrate_threshold(cs, cl, best)
    h = selective_risk(ts, tl, thr)          # looked at once
    if h["n_answered"] == 0:
        return None
    return {"margin": best, "threshold": thr, "risk": h["risk"],
            "coverage": h["answer_rate"], "met": h["risk"] <= alpha}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=ROOT / "runs" / "logprob_scores.json")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "nested_gate.json")
    a = ap.parse_args(argv)

    rows = json.loads(a.scores.read_text())
    rng = random.Random(0)
    print(f"alpha = {a.alpha}   {len(rows)} questions   {a.repeats} random 3-way splits")
    print("  (cal fits the threshold, val picks the margin, test is seen once)\n")
    print(f"  {'score':<14}{'met alpha':>11}{'coverage':>20}{'risk':>18}")

    summary = {}
    for key, label in [("support_v1", "support_v1"), ("logprob_score", "logprob")]:
        trials = [t for t in (one_trial(rows, key, a.alpha, rng) for _ in range(a.repeats)) if t]
        if not trials:
            print(f"  {label:<14}{'never':>11}")
            continue
        cov = np.array([t["coverage"] for t in trials])
        risk = np.array([t["risk"] for t in trials])
        met = np.mean([t["met"] for t in trials])
        summary[key] = {"met_rate": float(met), "coverage_mean": float(cov.mean()),
                        "coverage_p25": float(np.percentile(cov, 25)),
                        "coverage_p75": float(np.percentile(cov, 75)),
                        "risk_mean": float(risk.mean()), "n_trials": len(trials)}
        print(f"  {label:<14}{met:>10.0%}"
              f"{cov.mean():>13.0%} (IQR {np.percentile(cov,25):.0%}-{np.percentile(cov,75):.0%})"
              f"{risk.mean():>10.3f}")

    a.out.write_text(json.dumps({"alpha": a.alpha, "repeats": a.repeats,
                                 "summary": summary}, indent=2))
    print(f"\n  written -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
