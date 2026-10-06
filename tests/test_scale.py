"""The scale benchmark's arithmetic, on corpora small enough to check by hand.

The benchmark itself needs MS MARCO and a GPU. These tests need neither: they pin the
metrics and the BM25 index against formulas written out here, so a wrong number at
8.8 million passages cannot come from either.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scale import common
from scale.bm25 import BM25, TokenCache, Tokenizer, doc_term_counts, write_index
from scale.metrics import evaluate, overlap_at

# -- metrics -------------------------------------------------------------------


def test_mrr_recall_ndcg_by_hand():
    qrels = {1: {10}, 2: {20, 21}, 3: {30}}
    run = np.array(
        [
            [10, 11, 12],  # relevant at rank 1
            [22, 21, 20],  # both relevant, at ranks 2 and 3
            [31, 32, 33],  # nothing relevant retrieved
        ]
    )
    m = evaluate(run, np.array([1, 2, 3]), qrels)
    assert m["queries"] == 3
    assert m["mrr@10"] == pytest.approx((1 + 1 / 2 + 0) / 3, abs=1e-4)
    assert m["recall@10"] == pytest.approx((1 + 1 + 0) / 3, abs=1e-4)
    ndcg_q2 = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert m["ndcg@10"] == pytest.approx((1 + ndcg_q2 + 0) / 3, abs=1e-4)


def test_relevant_past_rank_ten_counts_for_recall_at_100_only():
    run = np.arange(100)[None, :]  # pid 50 sits at rank 51
    m = evaluate(run, np.array([7]), {7: {50}})
    assert m["mrr@10"] == 0.0
    assert m["recall@10"] == 0.0
    assert m["recall@100"] == 1.0


def test_unjudged_queries_are_skipped_and_padding_never_matches():
    run = np.array([[-1, -1], [5, -1]])
    m = evaluate(run, np.array([1, 99]), {1: {5}})  # query 99 has no judgements
    assert m["queries"] == 1
    assert m["mrr@10"] == 0.0


def test_overlap_counts_shared_ids_regardless_of_order():
    assert overlap_at(np.array([[1, 2, 3, 4]]), np.array([[4, 3, 9, 8]]), k=4) == 0.5


# -- BM25 ----------------------------------------------------------------------

DOCS = [
    "the pump seal leaks oil under pressure",
    "oil pressure drops at idle when the pump wears",
    "check the seal before the pump",
    "brake fluid and brake pads",
    "oil oil oil",
]
QUERIES = ["oil pressure", "pump seal", "brake brake", "nothing matches here", "oil"]


def _bm25_by_hand(query: list[str], docs: list[list[str]], k1: float, b: float) -> np.ndarray:
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n
    scores = np.zeros(n)
    for t in query:  # repeated terms count once per occurrence
        df = sum(t in d for d in docs)
        if df == 0:
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for i, d in enumerate(docs):
            tf = d.count(t)
            if tf:
                scores[i] += idf * tf / (tf + k1 * (1 - b + b * len(d) / avgdl))
    return scores


@pytest.fixture
def tiny_index(tmp_path):
    tok = Tokenizer(
        stopwords=frozenset({"the", "at", "and", "when", "under", "before", "here"}), stem=False
    )
    docs = [tok(d) for d in DOCS]
    vocab, terms, tf, indptr = doc_term_counts(docs)

    def rows(a, z):
        s, e = indptr[a], indptr[z]
        row = np.repeat(np.arange(z - a), np.diff(indptr[a : z + 1]))
        return terms[s:e], tf[s:e], row

    lens = np.array([len(d) for d in docs])
    write_index(rows, lens, len(vocab), tmp_path, k1=0.9, b=0.4, chunk=2)  # chunk 2: crosses chunks
    index = BM25(
        np.load(tmp_path / "indptr.npy"),
        np.load(tmp_path / "indices.npy"),
        np.load(tmp_path / "data.npy"),
        np.arange(100, 100 + len(docs)),  # pids differ from positions on purpose
        {w: i for i, w in enumerate(vocab)},
        tok,
    )
    return index, docs, tok


@pytest.mark.parametrize("query", QUERIES)
def test_bm25_matches_the_formula(tiny_index, query):
    index, docs, tok = tiny_index
    expected = _bm25_by_hand(tok(query), docs, k1=0.9, b=0.4)
    pids, scores = index.search(query, k=10)
    hits = np.flatnonzero(expected)
    assert sorted(pids - 100) == sorted(hits)
    np.testing.assert_allclose(scores, expected[pids - 100], rtol=1e-5)
    assert list(scores) == sorted(scores, reverse=True)


def test_bm25_buffer_is_clean_between_queries(tiny_index):
    index, _, _ = tiny_index
    first = index.search("oil pressure")
    index.search("brake")
    again = index.search("oil pressure")
    np.testing.assert_array_equal(first[0], again[0])
    np.testing.assert_array_equal(first[1], again[1])


def test_bm25_top_k_truncates_to_the_best(tiny_index):
    index, docs, tok = tiny_index
    expected = _bm25_by_hand(tok("oil"), docs, k1=0.9, b=0.4)
    pids, _ = index.search("oil", k=2)
    assert list(pids - 100) == list(np.argsort(-expected)[:2])


def test_doc_term_counts_is_a_bag_of_words():
    vocab, terms, tf, indptr = doc_term_counts([["a", "b", "a"], [], ["b"]])
    assert list(indptr) == [0, 2, 2, 3]
    first = {vocab[t]: int(c) for t, c in zip(terms[0:2], tf[0:2], strict=True)}
    assert first == {"a": 2, "b": 1}
    assert vocab[terms[2]] == "b"


def test_token_cache_gathers_scattered_rows_like_slicing_does():
    rng = np.random.default_rng(0)
    lens = rng.integers(0, 6, size=50)
    cache = object.__new__(TokenCache)  # the arrays, without the files behind them
    cache.indptr = np.concatenate([[0], np.cumsum(lens)])
    cache.terms = rng.integers(0, 1000, size=int(lens.sum())).astype(np.int32)
    cache.tf = rng.integers(1, 9, size=int(lens.sum())).astype(np.uint16)
    for pids in (np.arange(10, 30), np.array([1, 4, 5, 17, 33, 49])):
        terms, tf, row = cache.rows(pids)
        spans = [slice(cache.indptr[p], cache.indptr[p + 1]) for p in pids]
        np.testing.assert_array_equal(terms, np.concatenate([cache.terms[s] for s in spans]))
        np.testing.assert_array_equal(tf, np.concatenate([cache.tf[s] for s in spans]))
        np.testing.assert_array_equal(row, np.repeat(np.arange(len(pids)), lens[pids]))


# -- corpus sizes ----------------------------------------------------------------


def test_subsets_nest_and_keep_every_judged_passage(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "DATA", tmp_path)
    sizes = {"small": 1_000, "medium": 5_000, "full": common.N_PASSAGES}
    monkeypatch.setattr(common, "SIZES", sizes)
    judged = {7, 8_000_000, 123_456}
    lines = [f"{q}\t0\t{p}\t1\n" for q, p in enumerate(sorted(judged))]
    (tmp_path / "qrels.dev.small.tsv").write_text("".join(lines), encoding="utf-8")
    small, medium = common.subset("small"), common.subset("medium")
    assert len(small) == 1_000 and len(medium) == 5_000
    assert judged <= set(small.tolist())
    assert set(small.tolist()) <= set(medium.tolist())  # only distractors are added
    assert np.array_equal(small, common.subset("small"))  # cached and deterministic
