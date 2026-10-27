"""Compare the generators behind the same conformal gate.

    python -m finetune.report

Every generator answered the same 206 questions from the same excerpts, and the
fixed 14B judge labelled every answer. Per generator:

  base error     the mistake rate with every question answered, counted as in
                 runs/*.json: answering an unanswerable question is a mistake, and
                 so is a refusal of an answerable one
  correct        share of the 145 answerable questions judged CORRECT
  declined       refusal rate on the answerable and on the unanswerable questions
  gate           nested 3-way splits at alpha, by scripts/nested_gate.py's own
                 one_trial (cal fits the threshold, val picks the margin, test is
                 seen once), with the same split sequence for every generator, so
                 the comparison is paired
  risk at 50%    the mistake rate among the half of the questions the score ranks
                 highest, on all 206

Three scores go through the gate, each with the refusal veto (a reply that declines
scores 0, so the gate never counts a refusal as an answer):

  fixed     the 14B's P(YES) from runs/logprob_scores.json, the same number for every
            generator, so only the answers differ
  own       the generator's own P(YES) on the same support question
  decline   1 - P(decline), the generator's own probability of not declining
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys

import numpy as np

from .common import ROOT, RUNS, is_refusal, load_test_pool, tagged

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

GENERATORS = [
    ("hf-base", "3B, prompted (NF4, transformers)"),
    ("hf-tuned", "3B, fine-tuned, 40% declines (NF4 + LoRA)"),
    ("hf-tuned-d20", "3B, fine-tuned, 20% declines (NF4 + LoRA)"),
    ("hf-tuned-d0", "3B, fine-tuned, no declines (NF4 + LoRA)"),
    ("ollama-qwen2.5-3b-instruct", "3B, prompted (Q4_K_M, Ollama)"),
    ("ollama-qwen2.5-7b-instruct", "7B, prompted (Q4_K_M, Ollama)"),
    ("14b-cached", "14B, prompted (Q4_K_M, Ollama; answers from runs/)"),
]
ALPHAS = (0.1, 0.2, 0.3)
REPEATS = 300


def _nested_gate():
    """scripts/nested_gate.py, imported by path so the protocol is literally the same."""
    spec = importlib.util.spec_from_file_location(
        "nested_gate", ROOT / "scripts" / "nested_gate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_rows(name: str) -> list[dict] | None:
    """One row per test question with answer, loss, refusal and the three scores."""
    fixed = {
        r["id"]: r["logprob_score"]
        for r in json.loads((ROOT / "runs" / "logprob_scores.json").read_text())
    }
    if name == "14b-cached":
        rows = []
        for f in ("gate_scores_14b.json", "gate_scores_v2_14b.json", "gate_scores_v3_14b.json"):
            for r in json.loads((ROOT / "runs" / f).read_text()):
                rows.append(
                    {
                        **r,
                        "refused": is_refusal(r["answer"]),
                        "p_yes": fixed[r["id"]],
                        "p_decline": None,
                    }
                )
    else:
        path = RUNS / f"answers_{name}.json"
        if not path.exists():
            return None
        rows = json.loads(path.read_text())
        if any("loss" not in r for r in rows):
            return None
    order = {g["id"]: i for i, g in enumerate(load_test_pool())}
    rows = sorted(rows, key=lambda r: order[r["id"]])
    for r in rows:
        veto = 0.0 if r["refused"] else 1.0
        r["s_fixed"] = veto * fixed[r["id"]]
        r["s_fixed_noveto"] = fixed[r["id"]]
        r["s_own"] = veto * r["p_yes"]
        r["s_decline"] = None if r["p_decline"] is None else veto * (1.0 - r["p_decline"])
    return rows


def risk_at_coverage(rows: list[dict], key: str, coverage: float) -> float:
    """Mistake rate among the top `coverage` share by score; ties keep pool order."""
    ranked = sorted(range(len(rows)), key=lambda i: -rows[i][key])
    top = ranked[: round(coverage * len(rows))]
    return float(np.mean([rows[i]["loss"] for i in top]))


def gate(rows: list[dict], key: str, alpha: float, ng) -> dict:
    rng = random.Random(0)  # one split sequence, shared by every generator and score
    trials = [t for t in (ng.one_trial(rows, key, alpha, rng) for _ in range(REPEATS)) if t]
    if not trials:
        return {"met_rate": 0.0, "coverage_mean": 0.0, "n_trials": 0}
    cov = np.array([t["coverage"] for t in trials])
    return {
        "met_rate": round(float(np.mean([t["met"] for t in trials])), 3),
        "coverage_mean": round(float(cov.mean()), 3),
        "coverage_p25": round(float(np.percentile(cov, 25)), 3),
        "coverage_p75": round(float(np.percentile(cov, 75)), 3),
        "risk_mean": round(float(np.mean([t["risk"] for t in trials])), 3),
        "n_trials": len(trials),
    }


def category(row: dict) -> str:
    """Where a test question came from, which decides how hard it is.

    Generated questions share vocabulary with their chunk and retrieve easily; the
    hand-written ones do not. Unanswerable ones are either plainly outside the corpus
    (g- and gen- ids) or in-domain specifics the manuals never give (h2- and h3-).
    """
    if row["answerable"]:
        return "generated" if row["id"].startswith("gen-") else "hand-written"
    return "outside corpus" if row["id"].startswith(("g-", "gen-")) else "in-domain, absent"


def paired_diff(a: list[dict], b: list[dict], stat, n_boot: int = 2000) -> dict:
    """stat(a) - stat(b) with a 95% interval, resampling the same questions for both."""
    rng = np.random.default_rng(0)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(stat([a[i] for i in idx]) - stat([b[i] for i in idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": round(stat(a) - stat(b), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)]}


def correct_rate(rows: list[dict]) -> float:
    ans = [r for r in rows if r["answerable"]]
    return sum(r["loss"] == 0 for r in ans) / len(ans) if ans else 0.0


def base_error(rows: list[dict]) -> float:
    return float(np.mean([r["loss"] for r in rows]))


def summarise(name: str, label: str, rows: list[dict], ng) -> dict:
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    out = {
        "label": label,
        "n": len(rows),
        "base_error": round(float(np.mean([r["loss"] for r in rows])), 3),
        "correct": round(sum(r["loss"] == 0 for r in ans) / len(ans), 3),
        "declined_answerable": round(sum(r["refused"] for r in ans) / len(ans), 3),
        "declined_unanswerable": round(sum(r["refused"] for r in una) / len(una), 3),
        "by_category": {},
        "gate": {},
        "risk_at_50": {},
    }
    for cat in ("generated", "hand-written", "outside corpus", "in-domain, absent"):
        part = [r for r in rows if category(r) == cat]
        key = "correct" if part[0]["answerable"] else "declined"
        val = (
            correct_rate(part) if key == "correct" else sum(r["refused"] for r in part) / len(part)
        )
        out["by_category"][cat] = {"n": len(part), key: round(val, 3)}
    keys = ["s_fixed", "s_own", "s_decline"] + (["s_fixed_noveto"] if name == "14b-cached" else [])
    for key in keys:
        if rows[0][key] is None:
            continue
        out["risk_at_50"][key] = round(risk_at_coverage(rows, key, 0.5), 3)
        out["gate"][key] = {str(a): gate(rows, key, a, ng) for a in ALPHAS}
    secs = [r["seconds"] for r in rows if "seconds" in r]
    if secs:
        toks = sum(r["completion_tokens"] for r in rows)
        out["latency"] = {
            "mean_s": round(float(np.mean(secs)), 2),
            "median_s": round(float(np.median(secs)), 2),
            "completion_tokens_mean": round(toks / len(rows), 1),
            "tokens_per_s": round(toks / sum(secs), 1),
        }
    mem = RUNS / f"memory_{name}.json"
    if mem.exists():
        out["memory"] = json.loads(mem.read_text())
    return out


def main(argv: list[str] | None = None) -> int:
    ng = _nested_gate()
    report = {"alphas": ALPHAS, "repeats": REPEATS, "generators": {}}
    for name, label in GENERATORS:
        rows = load_rows(name)
        if rows is None:
            print(f"  (no judged answers for {name} yet)")
            continue
        report["generators"][name] = summarise(name, label, rows, ng)
    # Each fine-tuned run minus every prompted generator, paired on the same questions.
    report["tuned_minus"], report["training"] = {}, {}
    for t_name, _ in GENERATORS:
        if not t_name.startswith("hf-tuned"):
            continue
        tuned = load_rows(t_name)
        if tuned is None:
            continue
        report["tuned_minus"][t_name] = {}
        for name, _ in GENERATORS:
            other = None if name.startswith("hf-tuned") else load_rows(name)
            if other is None:
                continue
            report["tuned_minus"][t_name][name] = {
                "correct": paired_diff(tuned, other, correct_rate),
                "base_error": paired_diff(tuned, other, base_error),
                "risk_at_50_fixed": paired_diff(
                    tuned, other, lambda r: risk_at_coverage(r, "s_fixed", 0.5)
                ),
            }
        train = RUNS / tagged("train.json", t_name.removeprefix("hf-tuned").removeprefix("-"))
        if train.exists():
            t = json.loads(train.read_text())
            report["training"][t_name] = {
                k: t.get(k)
                for k in (
                    "n_train",
                    "n_val",
                    "optimizer_steps",
                    "best_step",
                    "trainable_params",
                    "peak_gib",
                    "minutes",
                    "train_tokens_per_s",
                )
            }
    (RUNS / "report.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'generator':<52}{'base err':>9}{'correct':>9}{'decl A':>8}{'decl U':>8}")
    for g in report["generators"].values():
        print(
            f"{g['label']:<52}{g['base_error']:>9.3f}{g['correct']:>9.3f}"
            f"{g['declined_answerable']:>8.3f}{g['declined_unanswerable']:>8.3f}"
        )
    for key in ("s_fixed", "s_own", "s_decline", "s_fixed_noveto"):
        print(f"\ngate at alpha = 0.2, score {key}: met rate, coverage (IQR), risk; risk at 50%")
        for g in report["generators"].values():
            if key not in g["gate"]:
                continue
            r = g["gate"][key]["0.2"]
            iqr = f"({r.get('coverage_p25', 0):.0%}-{r.get('coverage_p75', 0):.0%})"
            print(
                f"  {g['label']:<52}{r['met_rate']:>6.0%}{r['coverage_mean']:>7.0%} {iqr:<11}"
                f"{r.get('risk_mean', 0):>6.3f}{g['risk_at_50'][key]:>8.3f}"
            )
    print("\nfine-tuned minus prompted 3B (NF4): difference [95% interval]")
    for t_name, comps in report["tuned_minus"].items():
        d = comps.get("hf-base")
        if d:
            c, r = d["correct"], d["risk_at_50_fixed"]
            print(
                f"  {t_name:<14} correct {c['diff']:+.3f} {c['ci95']}"
                f"   risk at 50% {r['diff']:+.3f} {r['ci95']}"
            )
    print(f"\n  -> {RUNS / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
