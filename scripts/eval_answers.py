"""Answer-level evaluation: does it cite honestly, and does it refuse?

Retrieval eval (eval_retrieval.py) asks whether the right text was found.
This asks what the model then did with it. Three things are measured exactly,
without a judge model, because they are decidable:

  refusal        on the 5 unanswerable questions, did it say INSUFFICIENT
                 EVIDENCE rather than improvise? This is the headline number —
                 a RAG system's worst failure is fluent invention.
  answer rate    on the 15 answerable ones, did it actually answer?
  citation       every [n] it emits must index a excerpt that was really
  validity       supplied (1..k). A citation pointing at nothing is a
                 hallucinated source, and it is checkable arithmetic.

Semantic correctness — "is this answer *right*" — is deliberately NOT scored
here. It needs a judge model and a rubric, which is M4/M5 work; guessing at it
with string overlap would produce a number that looks like accuracy and isn't.

    python scripts/eval_answers.py --provider ollama [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from dataclasses import replace
from pathlib import Path

from conformal_rag.answer import answer
from conformal_rag.config import DEFAULT
from conformal_rag.embed import get_embedder
from conformal_rag.llm import get_llm
from conformal_rag.store import Store

GOLDEN = Path(__file__).parent.parent / "evals" / "golden.jsonl"
CITE = re.compile(r"\[(\d+)\]")
REFUSAL = "INSUFFICIENT EVIDENCE"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="ollama", choices=["ollama", "openai", "stub"])
    ap.add_argument("--embedder", default="bge", choices=["hash", "bge"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    cfg = replace(DEFAULT, llm_provider=a.provider)
    store, emb, llm = Store(cfg.db_path), get_embedder(a.embedder), get_llm(cfg)
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    if a.limit:
        rows = rows[: a.limit]

    out = []
    for r in rows:
        t0 = time.perf_counter()
        res = answer(r["question"], store, emb, llm, cfg)
        dt = time.perf_counter() - t0
        text = res.text or ""
        refused = REFUSAL in text.upper() or res.abstained
        cites = [int(c) for c in CITE.findall(text)]
        n_ex = len(res.hits)
        bad = [c for c in cites if c < 1 or c > n_ex]
        out.append({
            "id": r["id"], "answerable": r["answerable"], "type": r["type"],
            "refused": refused, "n_citations": len(cites), "bad_citations": bad,
            "chars": len(text), "seconds": round(dt, 1),
            "text": text[:300],
        })
        flag = "REFUSED" if refused else f"answered, {len(cites)} cite(s)"
        bad_s = f"  BAD CITES {bad}" if bad else ""
        print(f"  {r['id']:5s} {'ans' if r['answerable'] else 'UNANS'}  {flag:22s} {dt:5.1f}s{bad_s}")

    ans = [x for x in out if x["answerable"]]
    una = [x for x in out if not x["answerable"]]
    answered = [x for x in ans if not x["refused"]]
    correctly_refused = [x for x in una if x["refused"]]
    bad_cite_rows = [x for x in out if x["bad_citations"]]

    print("\n" + "=" * 62)
    print(f"provider={a.provider} embedder={a.embedder} corpus={store.count()} chunks")
    print(f"\n  REFUSAL on unanswerable   {len(correctly_refused)}/{len(una)}"
          f"  = {len(correctly_refused)/len(una):.2f}   <- the headline")
    print(f"  answer rate on answerable {len(answered)}/{len(ans)}  = {len(answered)/len(ans):.2f}")
    print(f"  invalid citations         {len(bad_cite_rows)}/{len(out)} responses")
    if answered:
        cited = [x for x in answered if x["n_citations"]]
        print(f"  answers carrying a citation {len(cited)}/{len(answered)}"
              f"  = {len(cited)/len(answered):.2f}")
    print(f"  median latency            {statistics.median(x['seconds'] for x in out):.1f}s")

    leaks = [x for x in una if not x["refused"]]
    if leaks:
        print("\n  INVENTED an answer to an unanswerable question:")
        for x in leaks:
            print(f"    {x['id']}: {x['text'][:150]}")
    over_refused = [x for x in ans if x["refused"]]
    if over_refused:
        print("\n  refused something it could have answered:")
        for x in over_refused:
            print(f"    {x['id']} ({x['type']})")

    if a.out:
        a.out.write_text(json.dumps(out, indent=2))
        print(f"\n  raw -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
