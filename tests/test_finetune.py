"""The fine-tuning experiment's bookkeeping, checked without a model or a GPU.

Training and evaluation need the 3B weights and a CUDA card. What decides whether the
comparison is fair does not: which pages training may draw from, how targets cite
their excerpt, which tokens carry the loss, what counts as a decline, and how the
gate is scored. Those are pinned here.
"""

from __future__ import annotations

import pytest

from conformal_rag.llm import TokenDist
from finetune import data, train
from finetune.common import excluded_pages, is_refusal, overlaps_verbatim, with_citation
from finetune.evaluate import decline_mass
from finetune.report import _nested_gate, risk_at_coverage

# -- the line between training data and the test pool --------------------------


def test_excluded_pages_cover_each_gold_page_and_its_neighbours():
    test = [
        {"answerable": True, "gold_doc": "a.pdf", "gold_page": 10},
        {"answerable": True, "gold_doc": "b.pdf", "gold_page": 3},
        {"answerable": False, "note": "out of corpus"},  # no page to protect
        {"answerable": True, "gold_doc": "a.pdf", "gold_page": None},
    ]
    assert excluded_pages(test) == {"a.pdf": {9, 10, 11}, "b.pdf": {2, 3, 4}}
    assert excluded_pages(test, margin=0) == {"a.pdf": {10}, "b.pdf": {3}}


def test_eligible_chunks_skip_excluded_pages_and_short_text():
    prose = " ".join(["word"] * 150)

    class FakeStore:
        class conn:  # stands in for sqlite3.Connection
            @staticmethod
            def execute(_sql):
                class Rows:
                    @staticmethod
                    def fetchall():
                        return [
                            (1, "a.pdf", 10, prose),  # a test question's gold page
                            (2, "a.pdf", 12, prose),  # two pages away: allowed
                            (3, "a.pdf", 13, "too short"),
                            (4, "b.pdf", 10, prose),  # same page number, other manual
                        ]

                return Rows

    kept = data.eligible_chunks(FakeStore, {"a.pdf": {9, 10, 11}})
    assert [r[0] for r in kept] == [2, 4]


def test_verbatim_overlap_drops_copied_questions_only():
    src = "The thermostat remains closed until the coolant reaches operating temperature."
    copied = (
        "Is it true the thermostat remains closed until the coolant reaches operating temperature?"
    )
    assert overlaps_verbatim(copied, src)
    assert not overlaps_verbatim("When does the cooling system thermostat open?", src)


# -- targets --------------------------------------------------------------------


def test_citation_goes_inside_the_final_full_stop():
    assert with_citation("It fails open.", 2) == "It fails open [2]."
    assert with_citation("Expansion valve", 4) == "Expansion valve [4]."
    assert with_citation("  1.28  ", 1) == "1.28 [1]."


def test_refusal_is_the_system_prompts_exact_phrase_in_any_case():
    assert is_refusal("INSUFFICIENT EVIDENCE")
    assert is_refusal("Insufficient evidence: the excerpts do not say.")
    assert not is_refusal("The evidence is sufficient [1].")


def test_labels_keep_confident_answers_and_confident_declines_only():
    assert data.kept({"kind": "answer", "p_yes": 0.5})
    assert not data.kept({"kind": "answer", "p_yes": 0.49})
    assert data.kept({"kind": "removed", "p_yes": 0.2})
    assert not data.kept({"kind": "absent", "p_yes": 0.21})
    assert not data.kept({"kind": "missed", "p_yes": 0.9})


def test_teacher_output_parsing():
    assert data.parse_object('Sure!\n{"question": "Why?", "answer": "Because."}') == {
        "question": "Why?",
        "answer": "Because.",
    }
    assert data.parse_object("no json here") is None
    assert data.parse_object("{not: valid}") is None
    lines = (
        "1. How do I pair a phone?\n- What does a new one cost?\nHere are ten questions:\n\nShort?"
    )
    assert data.outside_lines(lines) == ["How do I pair a phone?", "What does a new one cost?"]


# -- training -------------------------------------------------------------------


class FakeTokenizer:
    """Chat template = one token per character of the user message, plus 3 markers."""

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert add_generation_prompt and tokenize
        return [1] + [ord(c) for c in messages[1]["content"]] + [2, 3]

    def __call__(self, text, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [ord(c) for c in text]}

    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return 99


def test_encode_puts_the_reply_and_end_token_last_and_counts_them():
    row = {"system": "s", "prompt": "ab", "target": "xyz"}
    ids, n_reply = train.encode(FakeTokenizer(), row, max_len=100)
    assert ids == [1, ord("a"), ord("b"), 2, 3, ord("x"), ord("y"), ord("z"), 99]
    assert n_reply == 4  # x, y, z and <|im_end|>
    assert train.encode(FakeTokenizer(), row, max_len=8) is None  # 9 tokens: dropped, not cut


def test_reply_loss_scores_only_the_reply_positions():
    torch = pytest.importorskip("torch")

    vocab = 7

    class Out:
        def __init__(self, logits):
            self.logits = logits

    class Model:
        """Logits that favour token (position % vocab); returns only the kept positions."""

        def __call__(self, input_ids, logits_to_keep, attention_mask=None):
            seq = input_ids.shape[1]
            logits = torch.zeros(1, seq, vocab)
            for pos in range(seq):
                logits[0, pos, (pos + 1) % vocab] = 5.0
            return Out(logits[:, -logits_to_keep:, :])

    ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
    loss = train.reply_loss(Model(), ids, n_reply=2)
    # the reply is tokens 4 and 5, predicted from positions 3 and 4, which favour 4 and 5
    full = torch.zeros(2, vocab)
    full[0, 4] = full[1, 5] = 5.0
    expected = torch.nn.functional.cross_entropy(full, torch.tensor([4, 5]))
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)


def test_attention_fix_expands_grouped_heads_and_turns_the_math_kernel_off():
    torch = pytest.importorskip("torch")
    sdpa = pytest.importorskip("transformers.integrations.sdpa_attention")
    from finetune.common import linear_memory_attention

    original, math_on = sdpa.use_gqa_in_sdpa, torch.backends.cuda.math_sdp_enabled()
    try:
        linear_memory_attention()
        kv = torch.zeros(1, 2, 4, 8)
        assert sdpa.use_gqa_in_sdpa(None, kv, kv) is False
        assert not torch.backends.cuda.math_sdp_enabled()
    finally:
        sdpa.use_gqa_in_sdpa = original
        torch.backends.cuda.enable_math_sdp(math_on)


def test_variant_runs_get_their_own_files_and_adapters():
    from finetune.common import DATA, run_dir, tagged
    from finetune.evaluate import adapter_for

    assert run_dir("") == DATA and run_dir("d20") == DATA / "d20"
    assert tagged("train.json") == "train.json" and tagged("train.json", "d0") == "train_d0.json"
    assert adapter_for("hf-base") is None
    assert adapter_for("hf-tuned") == DATA / "adapter"
    assert adapter_for("hf-tuned-d20") == DATA / "d20" / "adapter"


def test_learning_rate_warms_up_then_decays_to_the_floor():
    peak, total, warmup = 2e-4, 100, 5
    assert train.lr_at(0, total, warmup, peak) == pytest.approx(peak / 5)
    assert train.lr_at(4, total, warmup, peak) == pytest.approx(peak)
    assert train.lr_at(5, total, warmup, peak) == pytest.approx(peak)
    assert train.lr_at(100, total, warmup, peak) == pytest.approx(0.1 * peak)
    mids = [train.lr_at(s, total, warmup, peak) for s in range(5, 101)]
    assert all(a >= b for a, b in zip(mids, mids[1:], strict=False))


# -- evaluation -----------------------------------------------------------------


def test_decline_mass_counts_the_start_of_the_refusal_and_nothing_else():
    dist = TokenDist({"INS": 0.5, "IN": 0.1, "In": 0.2, "I": 0.1, " The": 0.1})
    assert decline_mass(dist) == pytest.approx(0.6)


def test_risk_at_coverage_ranks_by_score_and_keeps_order_on_ties():
    rows = [
        {"s": 0.9, "loss": 0.0},
        {"s": 0.8, "loss": 1.0},
        {"s": 0.8, "loss": 0.0},
        {"s": 0.1, "loss": 1.0},
    ]
    assert risk_at_coverage(rows, "s", 0.25) == 0.0
    assert risk_at_coverage(rows, "s", 0.5) == 0.5  # the tie resolves to pool order
    assert risk_at_coverage(rows, "s", 1.0) == 0.5


def test_a_score_that_ranks_mistakes_last_lets_the_gate_hold():
    import random

    ng = _nested_gate()
    # 60 of 120 answers are right and all of them outrank the 60 mistakes
    rows = [{"s": 1.0 - i / 120, "loss": float(i >= 60)} for i in range(120)]
    rng = random.Random(0)
    trials = [t for t in (ng.one_trial(rows, "s", 0.2, rng) for _ in range(50)) if t]
    cov = [t["coverage"] for t in trials]
    assert sum(t["met"] for t in trials) / len(trials) >= 0.8
    assert max(cov) <= 0.75  # never much past the 50% that is right
    assert sum(cov) / len(cov) >= 0.45
