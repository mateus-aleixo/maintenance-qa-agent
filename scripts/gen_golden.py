"""Grow the golden set: generate candidate questions from real chunks.

M2 said the calibration set had to get bigger before a gate could mean anything,
so this generates candidates the only way that scales — the local model reads a
real chunk and writes a question a technician would ask about it. The source
chunk's page is the gold by construction.

**The bias this introduces, stated up front.** A question written *from* a chunk
shares vocabulary with it and is therefore easier to retrieve back to it than a
question a real user would type. So retrieval numbers on this set are optimistic
and should not be compared with the 20 hand-written questions in golden.jsonl.
What the set is *for* is calibrating the abstention gate, where the quantity that
matters is the answerable/unanswerable separation — much less affected by that
bias, since the unanswerable half is not generated from the corpus at all.

Mitigations applied: the prompt demands the technician's own words rather than the
chunk's phrasing, questions echoing a long span of the source verbatim are dropped,
and every generated question is written to a separate file so the hand-written set
stays untouched and citable on its own.

    python scripts/gen_golden.py --n 60 --out evals/golden_generated.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import replace
from pathlib import Path

from conformal_rag.config import DEFAULT
from conformal_rag.llm import get_llm
from conformal_rag.store import Store

SYSTEM = (
    "You write evaluation questions for a maintenance-manual assistant. "
    "Given an excerpt, write ONE question a technician would actually ask, whose "
    "answer is contained in the excerpt. Use the technician's own words, not the "
    "excerpt's phrasing. No preamble. Reply as exactly one JSON object: "
    '{"question": "...", "answer": "...", "type": "factual|procedure|reasoning"}'
)

# Out-of-corpus questions. Plausible for a maintenance assistant, absent from a
# 1985 US Army automotive manual — the hard kind, not absurd ones.
UNANSWERABLE = [
    "What is the torque spec for the cylinder head bolts on a Volvo D13?",
    "How do I reset the service light on a 2023 Ford Transit?",
    "What is the recommended cold tyre pressure for a 2024 Toyota Corolla?",
    "How much does a replacement alternator cost?",
    "What is the warranty period on a Tesla Model 3 traction battery?",
    "Which OBD-II code corresponds to a lean bank 1 condition?",
    "How do I pair a Bluetooth phone with the infotainment system?",
    "What viscosity of synthetic oil does the manufacturer specify for -30 C?",
    "How do I configure the Kubernetes ingress for the maintenance portal?",
    "What is the lead time for ordering a replacement turbocharger?",
    "Which lithium-ion cell chemistry is used in the hybrid battery pack?",
    "How do I calibrate the lane-keeping camera after a windscreen change?",
    "What is the EPA fuel economy rating for this vehicle?",
    "How do I update the ECU firmware over the air?",
    "What does the manufacturer charge for a scheduled 60,000 km service?",
    "Which grade of DEF fluid is required for the SCR system?",
    "How do I claim warranty on a failed water pump?",
    "What is the part number for the cabin air filter?",
    "How many kilowatt-hours does a full charge take?",
    "What is the maximum towing capacity in kilograms?",
]

JSON_RE = re.compile(r"\{.*\}", re.S)


def overlaps_verbatim(question: str, source: str, n: int = 8) -> bool:
    """True if the question copies an n-word span straight from the chunk."""
    qw = re.findall(r"[a-z0-9]+", question.lower())
    sw = " ".join(re.findall(r"[a-z0-9]+", source.lower()))
    return any(" ".join(qw[i:i + n]) in sw for i in range(max(0, len(qw) - n + 1)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="answerable questions to generate")
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", type=Path, default=Path("evals/golden_generated.jsonl"))
    a = ap.parse_args(argv)

    cfg = replace(DEFAULT, llm_provider=a.provider)
    store, llm = Store(cfg.db_path), get_llm(cfg)

    rows = store.conn.execute("SELECT id, doc, page, text FROM chunks").fetchall()
    good = [r for r in rows if len(r[3]) > 700 and r[3].count(" ") > 110]
    random.Random(a.seed).shuffle(good)

    out, seen_pages, tried = [], set(), 0
    for _cid, doc, page, text in good:
        if len(out) >= a.n:
            break
        if page in seen_pages:      # spread across the corpus
            continue
        tried += 1
        res = llm.complete(SYSTEM, f"Excerpt (from {doc}, page {page}):\n\n{text[:1400]}")
        m = JSON_RE.search(res.text)
        if not m:
            continue
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        q = str(obj.get("question", "")).strip()
        ans = str(obj.get("answer", "")).strip()
        if len(q) < 20 or not ans or overlaps_verbatim(q, text):
            continue
        seen_pages.add(page)
        out.append({
            "id": f"gen-{len(out) + 1:03d}",
            "type": obj.get("type", "factual"),
            "question": q,
            "answerable": True,
            "gold_doc": doc,
            "gold_page": page,
            "note": ans[:200],
            "source": "generated-from-chunk",
        })
        print(f"  [{len(out):3d}/{a.n}] p.{page:4d}  {q[:74]}")

    for i, q in enumerate(UNANSWERABLE, start=1):
        out.append({"id": f"gen-u{i:03d}", "type": "unanswerable", "question": q,
                    "answerable": False, "note": "out of corpus by construction",
                    "source": "hand-written"})

    a.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n",
                     encoding="utf-8")
    ans_n = sum(1 for r in out if r["answerable"])
    print(f"\nwrote {a.out}: {ans_n} answerable (from {tried} chunks tried), "
          f"{len(out) - ans_n} unanswerable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
