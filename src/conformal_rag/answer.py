"""Grounded answering with citations and a confidence signal for the gate.

The confidence score is deliberately model-free in v0: it comes from retrieval
agreement (how strongly the fused retrievers concur) so the conformal gate can be
calibrated before any LLM-judge machinery exists. M4 adds judge-based signals; the
gate's contract does not change.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .conformal import ConformalGate, GateDecision
from .embed import Embedder
from .guard import SYSTEM_RULES, fence, flag
from .llm import LLMClient
from .retrieve import Hit, retrieve
from .store import Store

_SYSTEM = (
    "You answer maintenance questions strictly from the provided excerpts. "
    "Cite excerpts by their bracketed number, e.g. [2]. If the excerpts do not "
    "contain the answer, say exactly: INSUFFICIENT EVIDENCE. Never invent facts. "
    + SYSTEM_RULES
)


@dataclass(frozen=True)
class Answer:
    question: str
    text: str | None          # None when abstaining
    abstained: bool
    reason: str               # "gate" | "no-hits" | "answered"
    hits: tuple[Hit, ...]
    confidence: float
    gate: GateDecision | None
    guard_flags: tuple[str, ...]


def confidence_from_hits(hits: list[Hit]) -> float:
    """Retrieval-agreement confidence in [0, 1].

    Max fused score, scaled by the theoretical RRF ceiling for two retrievers
    (2/(k+1) with k=60), boosted when the top hit was found by both retrievers.
    Crude, monotone, and calibratable — which is all the conformal gate needs.
    """
    if not hits:
        return 0.0
    ceiling = 2.0 / 61.0
    base = min(hits[0].score / ceiling, 1.0)
    both = 1.0 if len(hits[0].sources) > 1 else 0.6
    return round(base * both, 4)


def build_prompt(question: str, hits: list[Hit]) -> str:
    parts = []
    for i, h in enumerate(hits, start=1):
        parts.append(f"[{i}] ({h.doc}, p.{h.page})\n{fence(h.text)}")
    excerpts = "\n\n".join(parts)
    return f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}\n\nAnswer with citations:"


def answer(
    question: str,
    store: Store,
    embedder: Embedder,
    llm: LLMClient,
    cfg: Config,
    gate: ConformalGate | None = None,
    group: str = "_global",
    use_support: bool = False,
) -> Answer:
    """Answer, or abstain if a fitted `gate` says the evidence is too weak.

    `use_support` picks which signal the gate sees. Retrieval agreement (the
    default) is cheap but weak — M1 measured it separating answerable from
    unanswerable questions *worse* as retrieval improved. The support score
    (support.py) costs one extra model call and is what the gate is actually
    calibrated on; see docs/results.md.
    """
    hits = retrieve(
        store, embedder, question,
        k_bm25=cfg.k_bm25, k_vec=cfg.k_vec, k_final=cfg.k_final, rrf_k=cfg.rrf_k,
    )
    guard_flags = tuple(f for h in hits for f in flag(h.text).flags)

    if not hits:
        return Answer(question, None, True, "no-hits", (), 0.0, None, guard_flags)

    if use_support:
        from .support import support_score  # local: keeps the import cycle out

        conf = support_score(question, hits, llm)
    else:
        conf = confidence_from_hits(hits)
    decision = gate.decide(conf, group) if gate is not None else None
    if decision is not None and not decision.answer:
        return Answer(question, None, True, "gate", tuple(hits), conf, decision, guard_flags)

    result = llm.complete(_SYSTEM, build_prompt(question, hits))
    return Answer(
        question, result.text, False, "answered", tuple(hits), conf, decision, guard_flags
    )
