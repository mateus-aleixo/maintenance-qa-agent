"""3B vs 7B, on everything the gate depends on.

M5 concluded the binding constraint was the generator rather than the score.
This tests that directly.

**The confound, and how it is handled.** Switching model changes the *judge* as
well as the generator, so a lower error rate could just be a more lenient
grader. `--rejudge` regenerates the 3B answers and scores them with the **7B**
judge, giving a like-for-like row where only the generator differs. Without
that flag the comparison is still informative but not clean, and the output
says so.

    python scripts/compare_models.py
    python scripts/compare_models.py --rejudge      # isolates the generator
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

# A Windows console is cp1252 and these scripts print arrows and Greek letters.
# Same class of bug that killed the ONNX export in conformal-seg.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import numpy as np

from conformal_rag.answer import answer
from conformal_rag.config import DEFAULT
from conformal_rag.conformal import calibrate_threshold, selective_risk
from conformal_rag.embed import get_embedder
from conformal_rag.judge import judge_answer, loss_from_verdict
from conformal_rag.llm import get_llm
from conformal_rag.store import Store

ROOT = Path(__file__).parent.parent


def summarise(rows, label):
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    err = float(np.mean([r["loss"] for r in ans]))
    refused = sum(1 for r in una if r.get("refused_by_prompt"))
    s_ans = float(np.mean([r["score"] for r in ans]))
    s_una = float(np.mean([r["score"] for r in una]))
    return {"label": label, "n": len(rows), "base_error": err,
            "refusal_rate": refused / len(una) if una else float("nan"),
            "support_answerable": s_ans, "support_unanswerable": s_una,
            "distinct_scores": len({round(r["score"], 3) for r in rows})}


def gate_table(rows, alphas, seed=7):
    import random
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    h = len(idx) // 2
    cal, test = [rows[i] for i in idx[:h]], [rows[i] for i in idx[h:]]
    cs = np.array([r["score"] for r in cal], float)
    cl = np.array([r["loss"] for r in cal], float)
    ts = np.array([r["score"] for r in test], float)
    tl = np.array([r["loss"] for r in test], float)
    out = []
    for a in alphas:
        thr = calibrate_threshold(cs, cl, a)
        hh = selective_risk(ts, tl, thr)
        out.append({"alpha": a, "threshold": thr, **hh,
                    "met": hh["risk"] <= a and hh["n_answered"] > 0})
    return float(tl.mean()), out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--three", type=Path, default=ROOT / "runs" / "gate_scores_3b.json")
    ap.add_argument("--seven", type=Path, default=ROOT / "runs" / "gate_scores_7b.json")
    ap.add_argument("--rejudge", action="store_true",
                    help="regenerate 3B answers and judge them with the 7B judge")
    ap.add_argument("--judge-model", default="qwen2.5:7b-instruct")
    ap.add_argument("--gen-model", default="qwen2.5:3b-instruct")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "model_comparison.json")
    a = ap.parse_args(argv)

    three = json.loads(a.three.read_text())
    seven = json.loads(a.seven.read_text())
    rows = [summarise(three, "qwen2.5:3b"), summarise(seven, "qwen2.5:7b")]

    if a.rejudge:
        print("re-judging the 3B answers with the 7B judge (isolates the generator)...")
        gen_cfg = replace(DEFAULT, llm_provider="ollama", ollama_model=a.gen_model)
        jud_cfg = replace(DEFAULT, llm_provider="ollama", ollama_model=a.judge_model)
        store, emb = Store(DEFAULT.db_path), get_embedder("bge")
        gen_llm, jud_llm = get_llm(gen_cfg), get_llm(jud_cfg)
        golden = {}
        for p in ["golden.jsonl", "golden_generated.jsonl"]:
            for line in (ROOT / "evals" / p).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    golden[d["id"]] = d
        fixed = []
        for n, r in enumerate(three, start=1):
            g = golden[r["id"]]
            if r["answerable"]:
                text = answer(g["question"], store, emb, gen_llm, gen_cfg).text or ""
                v = judge_answer(g["question"], text, g.get("note", ""), jud_llm)
                loss = loss_from_verdict(v)
            else:
                v, loss = "UNANSWERABLE", 1.0
            fixed.append({**r, "verdict": v, "loss": loss})
            if n % 20 == 0:
                print(f"  {n}/{len(three)}")
        (ROOT / "runs" / "gate_scores_3b_rejudged.json").write_text(json.dumps(fixed, indent=2))
        rows.append(summarise(fixed, "qwen2.5:3b (7B judge)"))
        three = fixed

    print(f"\n{'='*78}")
    print(f"  {'':<24} {'base err':>9} {'refusal':>8} {'support a/u':>14} {'distinct':>9}")
    for r in rows:
        print(f"  {r['label']:<24} {r['base_error']:>9.3f} {r['refusal_rate']:>8.2f}"
              f" {r['support_answerable']:>6.3f}/{r['support_unanswerable']:<7.3f}"
              f" {r['distinct_scores']:>9}")

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    print("\n  gate on the support score (any-mistake loss)")
    gates = {}
    for name, data in [("3B", three), ("7B", seven)]:
        ung, tab = gate_table(data, alphas)
        gates[name] = {"ungated": ung, "rows": tab}
        best = next((t for t in tab if t["met"]), None)
        print(f"    {name}: ungated {ung:.3f}"
              + (f" | best alpha met {best['alpha']:.2f} -> risk {best['risk']:.3f}"
                 f" at {best['answer_rate']:.0%} coverage" if best else " | no alpha met"))

    a.out.write_text(json.dumps({"summary": rows, "gates": gates}, indent=2))
    print(f"\n  written -> {a.out}")
    if not a.rejudge:
        print("\n  NOTE: generator and judge both changed. Re-run with --rejudge to"
              "\n  isolate the generator effect before quoting these numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
