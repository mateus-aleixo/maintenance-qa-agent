"""Fit and verify the conformal abstention gate.

Pipeline per question: retrieve → support score → answer → (if answerable) judge
against the reference. That yields, for every item, the pair conformal risk
control needs:

    score   support_score in [0, 1]  — the generation-side signal (see support.py)
    loss    1 if ANSWERING this item would be a mistake, else 0
              unanswerable question  -> answering is always a mistake
              answerable question    -> mistake unless the judge says CORRECT

The gate then answers only when score >= threshold, and the threshold is chosen
so that the selective risk — the mistake rate *among answered questions* —
respects alpha with a finite-sample correction.

Calibration and test are split by a seeded shuffle and the reported risk comes
from the test half only. Scoring on the calibration half would report the
threshold's training error, which is the whole failure this method exists to
avoid.

    python scripts/calibrate_gate.py --alpha 0.2 --plot docs/figures/risk_curve.png
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from conformal_rag.answer import answer
from conformal_rag.config import DEFAULT
from conformal_rag.conformal import ConformalGate, calibrate_threshold, selective_risk
from conformal_rag.embed import get_embedder
from conformal_rag.judge import judge_answer, loss_from_verdict
from conformal_rag.llm import get_llm
from conformal_rag.store import Store
from conformal_rag.support import support_score

ROOT = Path(__file__).parent.parent
REFUSAL = "INSUFFICIENT EVIDENCE"


def load_sets(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        if p.exists():
            rows += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def score_all(rows, store, emb, llm, cfg, verbose=True) -> list[dict]:
    out = []
    t0 = time.perf_counter()
    for n, r in enumerate(rows, start=1):
        res = answer(r["question"], store, emb, llm, cfg)          # no gate yet
        text = res.text or ""
        s = support_score(r["question"], list(res.hits), llm)
        if r["answerable"]:
            verdict = judge_answer(r["question"], text, r.get("note", ""), llm)
            loss = loss_from_verdict(verdict)
        else:
            verdict = "UNANSWERABLE"
            loss = 1.0            # answering an unanswerable question is the mistake
        out.append({"id": r["id"], "answerable": r["answerable"],
                    "type": r.get("type", "?"), "score": s,
                    "loss": loss, "verdict": verdict,
                    "refused_by_prompt": REFUSAL in text.upper() or res.abstained,
                    # Stored so a later run can re-judge these answers with a
                    # different judge. Comparing two generators is only clean if
                    # the judge is held fixed, and the first run did not keep them.
                    "answer": text[:1200],
                    "model": cfg.ollama_model if cfg.llm_provider == "ollama" else cfg.openai_model})
        if verbose and (n % 10 == 0 or n == len(rows)):
            print(f"  scored {n}/{len(rows)}  ({time.perf_counter() - t0:.0f}s)")
        if verbose and n % 10 == 0:          # checkpoint: these runs are expensive
            _flush(out)
    return out


_FLUSH_PATH: Path | None = None


def _flush(rows):
    if _FLUSH_PATH:
        _FLUSH_PATH.write_text(json.dumps(rows, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--embedder", default="bge")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cache", type=Path, default=ROOT / "runs" / "gate_scores.json")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--golden", type=Path, nargs="+", default=None,
                    help="golden set file(s); defaults to the hand-written + generated pair")
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "gate.json")
    a = ap.parse_args(argv)

    cfg = replace(DEFAULT, llm_provider=a.provider, alpha=a.alpha)

    if a.cache.exists() and not a.rescore:
        scored = json.loads(a.cache.read_text())
        print(f"loaded {len(scored)} cached scores from {a.cache} (--rescore to redo)")
    else:
        rows = load_sets(a.golden or [ROOT / "evals" / "golden.jsonl",
                                      ROOT / "evals" / "golden_generated.jsonl"])
        print(f"scoring {len(rows)} questions with {a.provider} ...")
        global _FLUSH_PATH
        a.cache.parent.mkdir(parents=True, exist_ok=True)
        _FLUSH_PATH = a.cache
        scored = score_all(rows, Store(cfg.db_path), get_embedder(a.embedder),
                           get_llm(cfg), cfg)
        a.cache.write_text(json.dumps(scored, indent=2))

    idx = list(range(len(scored)))
    random.Random(a.seed).shuffle(idx)
    half = len(idx) // 2
    cal = [scored[i] for i in idx[:half]]
    test = [scored[i] for i in idx[half:]]

    cs, cl = np.array([x["score"] for x in cal]), np.array([x["loss"] for x in cal])
    ts, tl = np.array([x["score"] for x in test]), np.array([x["loss"] for x in test])

    thr = calibrate_threshold(cs, cl, a.alpha)
    held = selective_risk(ts, tl, thr)

    n_ans = sum(1 for x in scored if x["answerable"])
    print(f"\n{'=' * 64}\nitems {len(scored)}  ({n_ans} answerable, {len(scored) - n_ans} not)"
          f"   calibration {len(cal)} / test {len(test)}")
    print(f"\n  mean support score   answerable {np.mean([x['score'] for x in scored if x['answerable']]):.3f}"
          f"   unanswerable {np.mean([x['score'] for x in scored if not x['answerable']]):.3f}")
    print(f"\n  alpha (target risk)  {a.alpha}")
    print(f"  fitted threshold     {thr:.3f}")
    print(f"\n  HELD-OUT selective risk  {held['risk']:.3f}   "
          f"{'OK  <= alpha' if held['risk'] <= a.alpha else 'EXCEEDS alpha'}")
    print(f"  answered              {held['n_answered']}/{held['n_total']}"
          f"  = {held['answer_rate']:.2f}")

    ungated = float(np.mean(tl))
    print(f"\n  for comparison, answering EVERYTHING on the test half: risk {ungated:.3f}")

    gate = ConformalGate(alpha=a.alpha, min_group=cfg.min_group)
    gate.global_threshold = thr
    gate.group_thresholds = {}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"gate": gate.to_dict(), "held_out": held,
                                 "ungated_risk": ungated,
                                 "n_cal": len(cal), "n_test": len(test)}, indent=2))
    print(f"\n  gate -> {a.out}")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grid = np.linspace(0, 1, 101)
        risk = [selective_risk(ts, tl, t)["risk"] for t in grid]
        rate = [selective_risk(ts, tl, t)["answer_rate"] for t in grid]
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        ax.plot(grid, risk, color="#b0342c", lw=2, label="selective risk (test half)")
        ax.plot(grid, rate, color="#24548f", lw=1.6, ls="--", label="answer rate")
        ax.axhline(a.alpha, color="#888", lw=1, ls=":", label=f"α = {a.alpha}")
        ax.axvline(thr, color="#1a7a4a", lw=1.6,
                   label=f"threshold {thr:.2f} (fitted on the other half)")
        ax.set_xlabel("support score threshold")
        ax.set_ylabel("rate")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Conformal abstention: risk falls as the gate tightens, so does coverage",
                     fontsize=10.5)
        ax.legend(fontsize=8.4, frameon=False)
        ax.grid(alpha=0.25)
        a.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=120, facecolor="white")
        print(f"  plot -> {a.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
