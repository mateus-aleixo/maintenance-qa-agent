"""Check that every answerable golden question really is supported by the page it cites.

A wrong `gold_page` is invisible at runtime and silently corrupts the retrieval
metric: recall@k is measured against the cited page, so a mislabelled question
counts as a retrieval miss no matter what the retriever does. Writing 38
questions by hand produced exactly one such slip (an auxiliary-receptacle answer
cited to page 404 when the text is on 403).

This needs the corpus index, which is gitignored, so it is a local check and
exits 0 when `data/index.db` is absent. The structural half of the same contract
(ids unique, unanswerable rows citing no source, answerable rows carrying a doc
and an integer page) is in `tests/test_golden_sets.py`, which does run in CI.

The test is deliberately crude: take the content words of the recorded answer
and require most of them to appear on the cited page. It catches transposed and
off-by-one pages, not paraphrase.

    python scripts/verify_golden.py
    python scripts/verify_golden.py evals/golden_v3.jsonl --min-overlap 0.5
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

ROOT = Path(__file__).parent.parent
DEFAULT_FILES = ["golden.jsonl", "golden_v2.jsonl", "golden_v3.jsonl"]

STOP = set(
    ["the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "was", "were", "be", "been", "being", "for", "on", "at", "by", "with", "that", "this", "it", "its", "as", "from", "not", "no", "than", "then", "so", "which", "when", "what", "where", "how", "why", "can", "may", "must", "into", "out", "over", "under", "also", "both", "each", "other", "more", "most", "such", "only", "same", "they", "them", "their", "there", "these", "those", "your", "with", "within", "while"]
)


def content_words(text: str) -> list[str]:
    words = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
    return [w for w in words if w not in STOP and len(w) > 3]


def page_text(con: sqlite3.Connection, doc: str, page: int) -> str:
    rows = con.execute(
        "select text from chunks where doc = ? and page = ?", (doc, page)
    ).fetchall()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", t.lower()) for (t,) in rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--db", type=Path, default=ROOT / "data" / "index.db")
    ap.add_argument("--min-overlap", type=float, default=0.5)
    a = ap.parse_args(argv)

    files = a.files or [ROOT / "evals" / f for f in DEFAULT_FILES]
    if not a.db.exists():
        print(f"no index at {a.db}; run the ingest first")
        return 0

    con = sqlite3.connect(a.db)
    known_docs = {d for (d,) in con.execute("select distinct doc from chunks")}
    problems = 0

    for f in files:
        if not f.exists():
            continue
        lines = f.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(ln) for ln in lines if ln.strip()]
        n_ans = sum(1 for r in rows if r.get("answerable"))
        weak = []
        for r in rows:
            if not r.get("answerable"):
                # An unanswerable item must not point at a source, or the gate is
                # being handed the answer it is supposed to refuse.
                if r.get("gold_page") is not None or r.get("gold_doc") is not None:
                    weak.append((r["id"], "unanswerable but cites a source"))
                continue
            doc, page = r.get("gold_doc"), r.get("gold_page")
            if doc not in known_docs:
                weak.append((r["id"], f"unknown doc {doc!r}"))
                continue
            body = " ".join(page_text(con, doc, page).split())
            if not body:
                weak.append((r["id"], f"page {page} has no text"))
                continue
            kws = content_words(r.get("note") or "")
            if not kws:
                continue
            hit = sum(1 for w in kws if w in body)
            frac = hit / len(kws)
            if frac < a.min_overlap:
                missing = [w for w in kws if w not in body][:6]
                weak.append(
                    (r["id"], f"p{page} overlap {hit}/{len(kws)}, missing {missing}")
                )

        problems += len(weak)
        status = "ok" if not weak else f"{len(weak)} PROBLEM(S)"
        print(f"{f.name:<22} {len(rows):>3} questions ({n_ans} answerable)   {status}")
        for qid, why in weak:
            print(f"    {qid}: {why}")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
