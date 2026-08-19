"""CLI: ingest · ask · agent · calibrate. argparse, no extra dependency."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .agent import Agent
from .answer import answer
from .config import DEFAULT, Config
from .conformal import ConformalGate, selective_risk
from .embed import get_embedder
from .ingest import ingest_pdf
from .llm import get_llm
from .store import Store
from .tools import make_calculator_tool, make_rul_tool, make_search_tool
from .trace import Tracer


def cmd_ingest(args: argparse.Namespace, cfg: Config) -> int:
    store = Store(cfg.db_path)
    embedder = get_embedder(args.embedder)
    total = 0
    for pattern in args.pdfs:
        for path in sorted(Path().glob(pattern)) or [Path(pattern)]:
            if not path.exists():
                print(f"skip (missing): {path}", file=sys.stderr)
                continue
            chunks = ingest_pdf(path, cfg.chunk_chars, cfg.chunk_overlap)
            store.add_chunks(chunks)
            ids = [
                r[0] for r in store.conn.execute(
                    "SELECT id FROM chunks WHERE doc=? ORDER BY ordinal", (path.name,)
                )
            ]
            texts = [c["text"] for c in store.get_chunks(ids)]
            store.add_embeddings(ids, embedder.encode(texts))
            total += len(chunks)
            print(f"ingested {path.name}: {len(chunks)} chunks")
    print(f"total chunks in store: {store.count()} (+{total} this run)")
    return 0


def _load_gate(path: Path) -> ConformalGate | None:
    if path.exists():
        return ConformalGate.from_dict(json.loads(path.read_text()))
    return None


def cmd_ask(args: argparse.Namespace, cfg: Config) -> int:
    store = Store(cfg.db_path)
    tracer = Tracer(cfg.trace_dir)
    gate = _load_gate(Path(args.gate)) if args.gate else None
    with tracer.span("ask", question=args.question, provider=cfg.llm_provider):
        ans = answer(
            args.question, store, get_embedder(args.embedder), get_llm(cfg), cfg, gate,
            use_support=args.support,
        )
    tracer.emit(
        "gate",
        abstained=ans.abstained,
        reason=ans.reason,
        confidence=ans.confidence,
        guard_flags=list(ans.guard_flags),
    )
    if ans.abstained:
        print(f"ABSTAIN ({ans.reason}; confidence={ans.confidence})")
        if ans.gate:
            print(f"  threshold={ans.gate.threshold} (α={cfg.alpha}) — "
                  "the corpus does not support a sufficiently reliable answer.")
    else:
        print(ans.text)
        print("\nSources:")
        for i, h in enumerate(ans.hits, start=1):
            print(f"  [{i}] {h.doc} p.{h.page} (score {h.score:.4f}, {'+'.join(h.sources)})")
    return 0


def cmd_agent(args: argparse.Namespace, cfg: Config) -> int:
    store = Store(cfg.db_path)
    embedder = get_embedder(args.embedder)
    agent = Agent.build(
        get_llm(cfg),
        [make_search_tool(store, embedder, cfg), make_calculator_tool(), make_rul_tool(cfg)],
        max_steps=cfg.max_steps,
    )
    run = agent.run(args.question)
    for i, step in enumerate(run.steps, start=1):
        print(f"step {i}: {step.tool}({json.dumps(step.args)})")
        if step.guard_flags:
            print(f"  ⚠ injection patterns in output: {list(step.guard_flags)}")
    print(f"\n{run.final if run.final else f'(no answer: {run.stopped})'}")
    return 0


def cmd_calibrate(args: argparse.Namespace, cfg: Config) -> int:
    """Fit the gate from a JSONL of {"score": float, "loss": 0|1, "group": str}."""
    rows = [
        json.loads(line)
        for line in Path(args.records).read_text().splitlines()
        if line.strip()
    ]
    if len(rows) < 20:
        print(f"refusing to calibrate on {len(rows)} records (<20): "
              "a guarantee fitted on nothing is a lie", file=sys.stderr)
        return 1
    scores = np.array([r["score"] for r in rows])
    losses = np.array([r["loss"] for r in rows])
    groups = [r.get("group", "_global") for r in rows]
    gate = ConformalGate(alpha=cfg.alpha, min_group=cfg.min_group).fit(scores, losses, groups)
    Path(args.out).write_text(json.dumps(gate.to_dict(), indent=2))
    held = selective_risk(scores, losses, gate.global_threshold)
    print(f"gate written to {args.out}")
    print(f"α={cfg.alpha} → global threshold {gate.global_threshold:.4f}; "
          f"on calibration data: risk={held['risk']:.3f}, answer_rate={held['answer_rate']:.2%}")
    print("NOTE: report held-out risk from a separate split, never this number.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="conformal_rag")
    p.add_argument("--embedder", default="hash", choices=["hash", "bge"])
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("ingest", help="ingest PDFs into the store")
    sp.add_argument("pdfs", nargs="+")
    sp.set_defaults(fn=cmd_ingest)

    sp = sub.add_parser("ask", help="answer one question (with abstention if gated)")
    sp.add_argument("question")
    sp.add_argument("--gate", default="data/gate.json")
    sp.add_argument("--support", action="store_true",
                    help="gate on the support score (one extra model call) rather "
                         "than on retrieval agreement — this is what the gate is "
                         "calibrated against")
    sp.set_defaults(fn=cmd_ask)

    sp = sub.add_parser("agent", help="tool-using agent")
    sp.add_argument("question")
    sp.set_defaults(fn=cmd_agent)

    sp = sub.add_parser("calibrate", help="fit the conformal gate from scored records")
    sp.add_argument("records")
    sp.add_argument("--out", default="data/gate.json")
    sp.set_defaults(fn=cmd_calibrate)

    args = p.parse_args(argv)
    return args.fn(args, DEFAULT)


if __name__ == "__main__":
    raise SystemExit(main())
