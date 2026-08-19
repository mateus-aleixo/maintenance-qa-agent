"""Candidate nonconformity scores, and the one thing they are judged on.

M4 established the criterion: a score is only useful to the gate if it separates
**correct from incorrect answers**. `support_score` failed that test — 0.651 vs
0.625 — because it inspects the excerpts and the question, and never looks at
what the model actually said. Both scores here look at the answer.

  groundedness   asks whether the answer's claims are present in the excerpts it
                 cited. Wrong answers are usually wrong by asserting something
                 the excerpt does not say, which is visible without knowing the
                 truth. One extra call.

  self_consistency  samples the answer k times at temperature > 0 and measures
                 how much the samples agree. A model that knows tends to repeat
                 itself; a model that is guessing wanders. k extra calls, and no
                 judge — agreement is computed lexically, so it adds no second
                 stochastic component to the score.

Both are deliberately measured before either is adopted; M4's lesson was that a
plausible-sounding signal can be blind to the thing it is supposed to rank.
"""

from __future__ import annotations

import re

from .guard import SYSTEM_RULES, fence
from .llm import LLMClient
from .retrieve import Hit

_INT = re.compile(r"-?\d+")
_WORD = re.compile(r"[a-z0-9]+")

# Words carrying no claim; keeping them would let two unrelated answers look
# similar merely for being English.
_STOP = frozenset(["a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those", "is", "are", "was", "were", "be", "been", "being", "of", "in", "on", "at", "to", "from", "by", "for", "with", "without", "into", "over", "under", "as", "it", "its"])


_GROUNDED_SYSTEM = (
    "You check whether an answer is supported by the excerpts it was drawn from. "
    "You do NOT judge whether the answer is true in general, only whether these "
    "excerpts state it. Reply with a single integer 0-100 and nothing else, where "
    "the number is the percentage of the answer's factual claims that the excerpts "
    "actually support. " + SYSTEM_RULES
)


def groundedness_score(answer_text: str, hits: list[Hit], llm: LLMClient) -> float:
    """How much of the answer is actually stated by its excerpts, in [0, 1]."""
    if not answer_text.strip() or not hits:
        return 0.0
    excerpts = "\n\n".join(
        f"[{i}] {fence(h.text)}" for i, h in enumerate(hits, start=1)
    )
    out = llm.complete(
        _GROUNDED_SYSTEM,
        f"Excerpts:\n\n{excerpts}\n\nAnswer to check:\n{answer_text}\n\nScore:",
    )
    m = _INT.search(out.text or "")
    if not m:
        return 0.0
    return max(0.0, min(100.0, float(m.group(0)))) / 100.0


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def agreement(samples: list[str]) -> float:
    """Mean pairwise Jaccard over content words, in [0, 1].

    Lexical rather than model-judged on purpose: the score feeds a calibration
    whose whole promise is a bound, and a second sampled judgement inside it
    would make that bound harder to reason about, not easier.
    """
    sets = [content_words(s) for s in samples if s.strip()]
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return 0.0
    pairs = [(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    return sum(len(a & b) / len(a | b) for a, b in pairs) / len(pairs)


def self_consistency_score(
    system: str, prompt: str, llm: LLMClient, k: int = 3, temperature: float = 0.7
) -> tuple[float, list[str]]:
    """Sample the answer k times and measure how much the samples agree."""
    samples = [llm.complete(system, prompt, temperature=temperature).text or "" for _ in range(k)]
    return agreement(samples), samples


# ------------------------------------------------------------ logprob score

_YES_NO_SYSTEM = (
    "You decide whether a set of excerpts contains the information needed to "
    "answer a question. Reply with exactly one word: YES or NO. Nothing else. "
    "Judge only what the excerpts state; do not use outside knowledge and do not "
    "answer the question itself. " + SYSTEM_RULES
)

_YES = ("YES", "Yes", "yes", " YES", " Yes", " yes")
_NO = ("NO", "No", "no", " NO", " No", " no")


def support_score_logprob(question: str, hits: list[Hit], llm) -> float:
    """P(YES) over the first reply token — a genuinely continuous score.

    Every prompt-written score tried here collapsed: asking for 0-100 with
    anchors gave three distinct values across 152 questions, and removing the
    anchors bought granularity only by wrecking the ranking. The cause is that
    the model *writes* the number, and models write round numbers.

    Reading the probability of the YES token sidesteps that entirely. The model
    emits one token; the quantity scored is the distribution behind it, which is
    continuous by construction and not something the model can round off. The
    two-way normalisation over YES and NO discards mass on irrelevant tokens, so
    the result is a proper conditional probability rather than a raw softmax
    reading that drifts with how chatty the model feels.

    Returns 0.0 when neither token appears in the top-k — no evidence of support
    means abstain, consistent with every other score here.
    """
    if not hits:
        return 0.0
    excerpts = "\n\n".join(
        f"[{i}] ({h.doc}, p.{h.page})\n{fence(h.text)}" for i, h in enumerate(hits, start=1)
    )
    dist = llm.top_logprobs(
        _YES_NO_SYSTEM,
        f"Question: {question}\n\nExcerpts:\n\n{excerpts}\n\nDo the excerpts answer it?",
    )
    yes, no = dist.mass(*_YES), dist.mass(*_NO)
    total = yes + no
    return (yes / total) if total > 0 else 0.0


# --------------------------------------------------- head-purity combination

def _citation_ok(answer_text: str, n_excerpts: int) -> bool:
    """Every [n] in the answer indexes an excerpt that was actually supplied."""
    cites = [int(c) for c in re.findall(r"\[(\d+)\]", answer_text or "")]
    return all(1 <= c <= n_excerpts for c in cites)


def head_score(
    signals: dict[str, float],
    answer_text: str = "",
    n_excerpts: int = 0,
    veto: bool = True,
) -> float:
    """Conjunctive score built for a **pure head**, not for ranking.

    A conformal gate draws one line and keeps everything above it, so the only
    property that matters is whether the top bucket is nearly free of mistakes.
    AUC does not measure that — it describes the whole ordering, and a score can
    rank well on average while its top bucket stays contaminated. On 14 B the
    geometric-mean `combined` score had the best AUC (0.845) of anything tried
    and could not meet alpha = 0.2 at any threshold.

    The fix is to stop averaging. `min` is a conjunction: an answer scores high
    only if **every** check scores high, so reaching the head means surviving all
    of them, and a wrong answer has to fool all of them at once. Averaging lets
    one confident signal carry a doubtful one into the head, which is exactly how
    a head gets polluted.

    `veto` adds the free deterministic checks. A refusal or a citation pointing
    at an excerpt that was never supplied is direct evidence against the answer,
    costs no model call, and zeroes the score outright.
    """
    if veto and answer_text:
        if "INSUFFICIENT EVIDENCE" in answer_text.upper():
            return 0.0
        if n_excerpts and not _citation_ok(answer_text, n_excerpts):
            return 0.0
    vals = [max(0.0, min(1.0, v)) for v in signals.values()]
    return min(vals) if vals else 0.0


def consistency_excluding_refusals(samples: list[str]) -> float:
    """Agreement, but refusals score 0 instead of 1.

    Plain agreement is actively harmful at the head: three identical refusals are
    perfect agreement, so `self_consistency` on 14 B scored unanswerable questions
    a flat 1.000 and its own top bucket carried a risk of 0.789. Treating a
    consistent refusal as *no* support removes that contamination — the model
    declining three times is evidence against answering, not for it.
    """
    usable = [s for s in samples if s.strip() and "INSUFFICIENT EVIDENCE" not in s.upper()]
    if len(usable) < len(samples):          # any refusal at all is disqualifying
        return 0.0
    return agreement(usable)


# ---------------------------------------------------------------- support v2

# M4 found the original support prompt collapsed to four distinct values across
# 100 questions, because it offered 0 / 50 / 100 as anchors and the model treated
# them as the whole scale. This version gives no anchors and demands granularity.
_SUPPORT_V2_SYSTEM = (
    "You rate how completely a set of excerpts answers a question. "
    "Reply with a single integer from 0 to 100 and nothing else. "
    "Use the full range and avoid round numbers: 43 and 78 are better answers "
    "than 50 or 100. Judge only what the excerpts state; do not use outside "
    "knowledge and do not answer the question. " + SYSTEM_RULES
)


def support_score_v2(question: str, hits: list[Hit], llm: LLMClient) -> float:
    if not hits:
        return 0.0
    excerpts = "\n\n".join(
        f"[{i}] ({h.doc}, p.{h.page})\n{fence(h.text)}" for i, h in enumerate(hits, start=1)
    )
    out = llm.complete(_SUPPORT_V2_SYSTEM,
                       f"Question: {question}\n\nExcerpts:\n\n{excerpts}\n\nScore:")
    m = _INT.search(out.text or "")
    if not m:
        return 0.0
    return max(0.0, min(100.0, float(m.group(0)))) / 100.0
