"""Build the training set: questions from non-test pages, answered or declined.

Every example is what the served path would show the model: the system prompt from
conformal_rag.answer, the five excerpts `retrieve()` returns for the question, and the
question. The target is a short answer citing the excerpt it came from, or INSUFFICIENT
EVIDENCE when those five excerpts do not contain it. Labels are about the excerpts,
not the corpus, because the excerpts are all the model sees.

Four kinds of example:

  answer   the teacher wrote the question from a chunk, and retrieval brought that
           chunk back. Target: the teacher's answer, citing the chunk's [n].
  removed  the same question with the source page and its neighbours taken out of the
           results, so the next five are on the same topic but do not say it. Target:
           decline. Paired with `answer`, it teaches the model to read the excerpts
           rather than the question.
  missed   a question whose source chunk retrieval did not bring back. Target: decline.
  absent   questions the manuals do not answer: a value or step the source chunk's
           topic leaves out, and questions from outside the corpus. Target: decline.

The teacher's P(YES) on the support question (scores.support_score_logprob) checks
every label: an answer needs at least 0.5, a decline at most 0.2, and anything in
between is dropped rather than trained on.

    python -m finetune.data questions   # the teacher writes candidates (GPU)
    python -m finetune.data label       # retrieve and check each one (GPU)
    python -m finetune.data split       # paraphrase filter, mixture, train/val (CPU)
    python -m finetune.data all
    python -m finetune.data split --decline 0.2 --tag d20   # a variant with fewer declines

A variant keeps every answer example of the main split and the first part of the same
shuffled declines, so its declines are a subset of the main run's.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter

from conformal_rag.answer import _SYSTEM as ANSWER_SYSTEM  # the served prompt, verbatim
from conformal_rag.answer import build_prompt
from conformal_rag.config import DEFAULT
from conformal_rag.embed import get_embedder
from conformal_rag.retrieve import Hit, retrieve
from conformal_rag.scores import support_score_logprob
from conformal_rag.store import Store

from .common import (
    DATA,
    PAGE_MARGIN,
    PARAPHRASE_COS,
    REFUSAL,
    RUNS,
    TEACHER,
    Ollama,
    append_jsonl,
    excluded_pages,
    load_test_pool,
    overlaps_verbatim,
    read_jsonl,
    run_dir,
    tagged,
    with_citation,
)

GEN_SYSTEM = (
    "You write evaluation questions for a maintenance-manual assistant. "
    "Given an excerpt, write ONE question a technician would actually ask, whose "
    "answer is contained in the excerpt. Use the technician's own words, not the "
    "excerpt's phrasing. The question must stand on its own, because whoever asks it "
    "has not seen the excerpt: name the component or system it is about. Then answer "
    "it in one or two complete sentences, using only the excerpt. No preamble. Reply "
    "as exactly one JSON object: "
    '{"question": "...", "answer": "...", "type": "factual|procedure|reasoning"}'
)

NEAR_SYSTEM = (
    "You write evaluation questions for a maintenance-manual assistant. Given an "
    "excerpt, write ONE question a technician might ask on the same topic that the "
    "excerpt does NOT answer, such as a specific value, part number, interval or step "
    "it does not give. It should sound as if it belongs in the same manual, and stand "
    "on its own: name the component or system it is about. No preamble. Reply as "
    'exactly one JSON object: {"question": "..."}'
)

OUTSIDE_SYSTEM = (
    "You write evaluation questions for a maintenance assistant whose only sources are "
    "a 1985 US Army manual on the principles of automotive vehicles and a parts manual "
    "for a 5 kW military generator set. Write 10 different questions a user might "
    "plausibly ask this assistant that those sources cannot answer, on the topic given. "
    "One question per line, no numbering, no preamble."
)

OUTSIDE_TOPICS = [
    "electric and hybrid vehicles",
    "modern engine electronics and diagnostics",
    "prices, ordering and warranty",
    "a specific current car or truck model",
    "software, apps and IT systems",
    "aircraft and helicopters",
    "boats and marine engines",
    "heating and air conditioning in buildings",
    "industrial robots and PLCs",
    "agricultural and construction machinery",
    "motorcycles",
    "emissions rules and modern exhaust after-treatment",
    "workshop safety rules and certifications",
    "tyres and wheels on current vehicles",
    "batteries and chargers for consumer devices",
    "railway equipment",
    "wind turbines and solar inverters",
    "forklifts, lifts and cranes",
    "company policy and paperwork",
    "medical and laboratory equipment",
]

ANSWER_MIN_P, DECLINE_MAX_P = 0.5, 0.2
JSON_RE = re.compile(r"\{.*\}", re.S)
QUESTIONS = DATA / "questions.jsonl"
LABELLED = DATA / "labelled.jsonl"


TEACHER_LLM = Ollama(TEACHER)


def chat(system: str, user: str, temperature: float = 0.0, seed: int = 0) -> str:
    return TEACHER_LLM.complete(system, user, temperature=temperature, seed=seed).text


def parse_object(text: str) -> dict | None:
    m = JSON_RE.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def outside_lines(text: str) -> list[str]:
    """One question per line, with any numbering or bullets the teacher added anyway."""
    out = []
    for line in (text or "").splitlines():
        q = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip()
        if len(q) >= 20 and q.endswith("?"):
            out.append(q)
    return out


def eligible_chunks(store: Store, excluded: dict[str, set[int]]) -> list[tuple]:
    """Prose chunks (gen_golden's filter) on pages no test question was written from."""
    rows = store.conn.execute("SELECT id, doc, page, text FROM chunks ORDER BY id").fetchall()
    return [
        r
        for r in rows
        if len(r[3]) > 700 and r[3].count(" ") > 110 and r[2] not in excluded.get(r[1], set())
    ]


# ------------------------------------------------------------------ questions


def stage_questions(n_near: int, seed: int, limit: int | None = None) -> None:
    store = Store(DEFAULT.db_path)
    chunks = eligible_chunks(store, excluded_pages(load_test_pool()))
    random.Random(seed).shuffle(chunks)
    chunks = chunks[:limit]
    done = {r["id"] for r in read_jsonl(QUESTIONS)}
    print(f"{len(chunks)} eligible chunks, {len(done)} questions already written", flush=True)

    t0 = time.perf_counter()
    for n, (cid, doc, page, text) in enumerate(chunks, start=1):
        src = {"src_chunk": cid, "src_doc": doc, "src_page": page}
        excerpt = f"Excerpt (from {doc}, page {page}):\n\n{text[:1400]}"
        if f"c{cid}" not in done:
            obj = parse_object(chat(GEN_SYSTEM, excerpt)) or {}
            q, a = str(obj.get("question", "")).strip(), str(obj.get("answer", "")).strip()
            drop = (
                "unparseable"
                if not obj
                else "short"
                if len(q) < 20 or not a
                else "verbatim"
                if overlaps_verbatim(q, text)
                else None
            )
            append_jsonl(
                QUESTIONS,
                {
                    "id": f"c{cid}",
                    "kind": "chunk",
                    "question": q,
                    "answer": a,
                    "type": str(obj.get("type", "factual")),
                    "drop": drop,
                    **src,
                },
            )
        if n <= n_near and f"n{cid}" not in done:
            obj = parse_object(chat(NEAR_SYSTEM, excerpt)) or {}
            q = str(obj.get("question", "")).strip()
            drop = "unparseable" if not obj else "short" if len(q) < 20 else None
            append_jsonl(
                QUESTIONS,
                {
                    "id": f"n{cid}",
                    "kind": "near",
                    "question": q,
                    "type": "unanswerable",
                    "drop": drop,
                    **src,
                },
            )
        if n % 25 == 0:
            print(f"  {n}/{len(chunks)} chunks  ({time.perf_counter() - t0:.0f}s)", flush=True)

    for t, topic in enumerate(OUTSIDE_TOPICS[:limit]):
        if any(i.startswith(f"o{t:02d}-") for i in done):
            continue
        qs = outside_lines(chat(OUTSIDE_SYSTEM, f"Topic: {topic}", temperature=0.7, seed=t))
        for i, q in enumerate(qs):
            append_jsonl(
                QUESTIONS,
                {
                    "id": f"o{t:02d}-{i:02d}",
                    "kind": "outside",
                    "question": q,
                    "type": "unanswerable",
                    "topic": topic,
                    "drop": None,
                },
            )
    print(f"questions done ({time.perf_counter() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------- label


def stage_label() -> None:
    store = Store(DEFAULT.db_path)
    emb = get_embedder("bge")
    teacher = TEACHER_LLM
    qs = [q for q in read_jsonl(QUESTIONS) if not q.get("drop")]
    done = {r["qid"] for r in read_jsonl(LABELLED)}
    todo = [q for q in qs if q["id"] not in done]
    print(f"labelling {len(todo)} questions ({len(done)} done)", flush=True)

    def row(q: dict, kind: str, hits: list[Hit], target: str) -> dict:
        return {
            "id": f"{q['id']}-{kind}",
            "qid": q["id"],
            "kind": kind,
            "question": q["question"],
            "target": target,
            "chunks": [h.chunk_id for h in hits],
            "p_yes": round(support_score_logprob(q["question"], hits, teacher), 6),
            "type": q.get("type"),
            "src_doc": q.get("src_doc"),
            "src_page": q.get("src_page"),
        }

    t0 = time.perf_counter()
    for n, q in enumerate(todo, start=1):
        hits = retrieve(store, emb, q["question"], DEFAULT.k_bm25, DEFAULT.k_vec, 20, DEFAULT.rrf_k)
        top, ids = hits[: DEFAULT.k_final], [h.chunk_id for h in hits[: DEFAULT.k_final]]
        rows = []
        if q["kind"] == "chunk" and q["src_chunk"] in ids:
            n_cite = ids.index(q["src_chunk"]) + 1
            rows.append(row(q, "answer", top, with_citation(q["answer"], n_cite)))
            rest = [
                h
                for h in hits
                if not (h.doc == q["src_doc"] and abs(h.page - q["src_page"]) <= PAGE_MARGIN)
            ]
            if len(rest) >= DEFAULT.k_final:
                rows.append(row(q, "removed", rest[: DEFAULT.k_final], REFUSAL))
        elif q["kind"] == "chunk":
            rows.append(row(q, "missed", top, REFUSAL))
        else:
            rows.append(row(q, "absent", top, REFUSAL))
        for r in rows:
            append_jsonl(LABELLED, r)
        if n % 25 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}  ({time.perf_counter() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------- split


def kept(r: dict) -> bool:
    if r["kind"] == "answer":
        return r["p_yes"] >= ANSWER_MIN_P
    return r["p_yes"] <= DECLINE_MAX_P


def stage_split(seed: int, val_frac: float, decline_frac: float, tag: str = "") -> dict:
    questions = read_jsonl(QUESTIONS)
    labelled = read_jsonl(LABELLED)
    checked = [r for r in labelled if kept(r)]

    # Near-paraphrases of test questions go, whatever page they came from.
    test = load_test_pool()
    emb = get_embedder("bge")
    texts = sorted({r["question"] for r in checked})
    sims = emb.encode(texts) @ emb.encode([t["question"] for t in test]).T
    nearest = dict(zip(texts, sims.max(axis=1).tolist(), strict=True))
    clean = [r for r in checked if nearest[r["question"]] < PARAPHRASE_COS]

    # Validation holds out whole source pages, so no page is on both sides.
    def group(r: dict) -> str:
        return f"{r['src_doc']}#{r['src_page']}" if r.get("src_doc") else r["qid"]

    groups = sorted({group(r) for r in clean})
    random.Random(seed).shuffle(groups)
    val_groups = set(groups[: round(len(groups) * val_frac)])

    store = Store(DEFAULT.db_path)
    dest = run_dir(tag)
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("train", "val"):
        part = [r for r in clean if (group(r) in val_groups) == (split == "val")]
        answers = [r for r in part if r["kind"] == "answer"]
        declines = [r for r in part if r["kind"] != "answer"]
        random.Random(seed).shuffle(declines)
        cap = round(len(answers) * decline_frac / (1 - decline_frac))
        rows = answers + declines[:cap]
        random.Random(seed + 1).shuffle(rows)
        with (dest / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                by_id = {c["id"]: c for c in store.get_chunks(r["chunks"])}
                hits = [
                    Hit(c, by_id[c]["doc"], by_id[c]["page"], by_id[c]["text"], 0.0, ())
                    for c in r["chunks"]
                ]
                f.write(
                    json.dumps(
                        {
                            "id": r["id"],
                            "kind": r["kind"],
                            "system": ANSWER_SYSTEM,
                            "prompt": build_prompt(r["question"], hits),
                            "target": r["target"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        out[split] = dict(Counter(r["kind"] for r in rows))

    stats = {
        "teacher": TEACHER,
        "questions": dict(Counter(q["kind"] for q in questions)),
        "dropped_at_writing": dict(Counter(q["drop"] for q in questions if q.get("drop"))),
        "labelled": dict(Counter(r["kind"] for r in labelled)),
        "dropped_by_teacher_check": dict(Counter(r["kind"] for r in labelled if not kept(r))),
        "dropped_as_test_paraphrase": len(checked) - len(clean),
        "paraphrase_cosine": PARAPHRASE_COS,
        "max_cosine_kept": round(max((nearest[r["question"]] for r in clean), default=0.0), 4),
        "decline_fraction": decline_frac,
        "split": out,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / tagged("data.json", tag)).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["questions", "label", "split", "all"])
    ap.add_argument("--near", type=int, default=400, help="chunks that also get a near-miss")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--decline", type=float, default=0.4, help="share of declines in each split")
    ap.add_argument("--limit", type=int, default=None, help="first n chunks and topics only")
    ap.add_argument("--tag", default="", help="write the split to data/finetune/<tag>")
    a = ap.parse_args(argv)
    # The teacher holds the GPU, so the encoder stays on the CPU and the two never
    # share the card. Set before sentence-transformers first imports torch.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if a.stage in ("questions", "all"):
        stage_questions(a.near, a.seed, a.limit)
    if a.stage in ("label", "all"):
        stage_label()
    if a.stage in ("split", "all"):
        stage_split(a.seed, a.val, a.decline, a.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
