"""BM25 over millions of passages, built in a few gigabytes of RAM.

    python -m scale.bm25 tokens          # once: tokenize all 8.8M passages, in parallel
    python -m scale.bm25 build 1m        # one corpus size's index, from the token cache
    python -m scale.bm25 search 1m       # every dev query, timed one at a time
    python -m scale.bm25 check 100k      # scores against bm25s, the reference

The score is Lucene's BM25 (bm25s calls it "lucene"), the variant Anserini publishes
its MS MARCO baselines with:

    idf(t)  = ln(1 + (N - df + 0.5) / (df + 0.5))
    w(t, d) = idf(t) * tf / (tf + k1 * (1 - b + b * |d| / avgdl))

with Anserini's MS MARCO passage parameters k1 = 0.9 and b = 0.4, so the BM25 row has
a published number to land near: MRR@10 0.184 on dev.small.

bm25s is the reference implementation, and `check` compares the two. The build here
differs in one respect: it never holds the corpus as Python objects. Every passage is
tokenized once, in parallel, into a compact cache on disk (int32 term ids, uint16
counts), and each corpus size's term-major index is then assembled from that cache a
chunk at a time, so peak memory is the finished index plus one chunk.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import time
from pathlib import Path

import numpy as np

from .common import (
    DATA,
    DEPTH,
    N_PASSAGES,
    Passages,
    dir_bytes,
    latency_sample,
    load_queries,
    peak_rss_gb,
    read_json,
    save_run,
    size_dir,
    subset,
    timed,
    write_json,
)

K1, B = 0.9, 0.4
TOKENS = DATA / "bm25_tokens"
_SPLIT = re.compile(r"(?u)\b\w\w+\b")  # bm25s's default splitter


class Tokenizer:
    """Lowercase, split, drop stopwords, stem: bm25s's order, so `check` compares like
    with like. Stopwords are matched before stemming, as in bm25s."""

    def __init__(self, stopwords: frozenset[str] | None = None, stem: bool = True):
        if stopwords is None:
            from bm25s.stopwords import STOPWORDS_EN

            stopwords = frozenset(STOPWORDS_EN)
        self.stop = stopwords
        self._stemmer = None
        if stem:
            import Stemmer

            self._stemmer = Stemmer.Stemmer("english")
        self._cache: dict[str, str] = {}

    def words(self, text: str) -> list[str]:
        return [w for w in _SPLIT.findall(text.lower()) if w not in self.stop]

    def stems(self, words: list[str]) -> list[str]:
        if self._stemmer is None:
            return words
        out = []
        for w in words:
            s = self._cache.get(w)
            if s is None:
                s = self._cache[w] = self._stemmer.stemWord(w)
            out.append(s)
        return out

    def __call__(self, text: str) -> list[str]:
        return self.stems(self.words(text))


def doc_term_counts(docs: list[list[str]]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Bag of words per document, as (vocab, term ids, counts, indptr) in CSR order.

    Vectorised: one integer key per (doc, term) occurrence, then np.unique, instead
    of a Counter per document."""
    vocab: dict[str, int] = {}
    flat = [vocab.setdefault(t, len(vocab)) for d in docs for t in d]
    lens = np.fromiter((len(d) for d in docs), dtype=np.int64, count=len(docs))
    v = max(len(vocab), 1)
    key = np.repeat(np.arange(len(docs), dtype=np.int64), lens) * v + np.asarray(
        flat, dtype=np.int64
    )
    uniq, counts = np.unique(key, return_counts=True)
    indptr = np.zeros(len(docs) + 1, dtype=np.int64)
    np.cumsum(np.bincount(uniq // v, minlength=len(docs)), out=indptr[1:])
    tf = np.minimum(counts, np.iinfo(np.uint16).max).astype(np.uint16)
    return list(vocab), (uniq % v).astype(np.int32), tf, indptr


# -- the token cache -------------------------------------------------------------

_worker_tok: Tokenizer | None = None


def _tokenize_span(span: tuple[int, int]):
    global _worker_tok
    if _worker_tok is None:
        _worker_tok = Tokenizer()
    start, stop = span
    docs = [_worker_tok(t) for t in Passages().span(start, stop)]
    lens = np.fromiter((len(d) for d in docs), dtype=np.int32, count=len(docs))
    return (*doc_term_counts(docs), lens)


def build_tokens(workers: int | None = None, chunk: int = 20_000) -> dict:
    """Tokenize the whole collection once; every corpus size's index reads from this.

    Workers return chunk-local vocabularies, merged here into one global id space, so
    no process ever holds more than one chunk of text."""
    TOKENS.mkdir(parents=True, exist_ok=True)
    spans = [(a, min(a + chunk, N_PASSAGES)) for a in range(0, N_PASSAGES, chunk)]
    gvocab: dict[str, int] = {}
    indptr = np.zeros(N_PASSAGES + 1, dtype=np.int64)
    doclen = np.empty(N_PASSAGES, dtype=np.int32)
    pos = 0
    workers = workers or max(1, (mp.cpu_count() or 2) - 2)
    t0 = time.perf_counter()
    with (
        open(TOKENS / "terms.i32", "wb") as ft,
        open(TOKENS / "tf.u16", "wb") as ff,
        mp.get_context("spawn").Pool(workers) as pool,
    ):
        for i, ((a, b), (stems, terms, tf, ip, lens)) in enumerate(
            zip(spans, pool.imap(_tokenize_span, spans), strict=True)
        ):
            g = np.fromiter(
                (gvocab.setdefault(s, len(gvocab)) for s in stems), dtype=np.int32, count=len(stems)
            )
            ft.write(g[terms].tobytes())
            ff.write(tf.tobytes())
            indptr[a + 1 : b + 1] = pos + ip[1:]
            doclen[a:b] = lens
            pos += len(terms)
            if (i + 1) % 20 == 0 or b == N_PASSAGES:
                el = time.perf_counter() - t0
                print(
                    f"  tokenized {b:>9,} passages, {len(gvocab):,} terms, {el:.0f} s", flush=True
                )
    np.save(TOKENS / "indptr.npy", indptr)
    np.save(TOKENS / "doclen.npy", doclen)
    (TOKENS / "vocab.txt").write_text("\n".join(gvocab), encoding="utf-8")
    meta = {
        "passages": N_PASSAGES,
        "nnz": pos,
        "vocab": len(gvocab),
        "tokens": int(doclen.sum(dtype=np.int64)),
        "seconds": round(time.perf_counter() - t0, 1),
        "workers": workers,
    }
    write_json(TOKENS / "meta.json", meta)
    return meta


class TokenCache:
    def __init__(self):
        meta = read_json(TOKENS / "meta.json")
        self.n_vocab = meta["vocab"]
        self.indptr = np.load(TOKENS / "indptr.npy", mmap_mode="r")
        self.doclen = np.load(TOKENS / "doclen.npy", mmap_mode="r")
        self.terms = np.memmap(TOKENS / "terms.i32", dtype=np.int32, mode="r", shape=(meta["nnz"],))
        self.tf = np.memmap(TOKENS / "tf.u16", dtype=np.uint16, mode="r", shape=(meta["nnz"],))

    def rows(self, pids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(term ids, counts, row within `pids`) for every entry of the given passages."""
        starts = np.asarray(self.indptr[pids])
        lens = np.asarray(self.indptr[pids + 1]) - starts
        row = np.repeat(np.arange(len(pids), dtype=np.int64), lens)
        if pids[-1] - pids[0] + 1 == len(pids):  # contiguous: one slice, no gather
            sl = slice(int(starts[0]), int(starts[0] + lens.sum()))
            return np.asarray(self.terms[sl]), np.asarray(self.tf[sl]), row
        idx = np.arange(int(lens.sum()), dtype=np.int64) + np.repeat(
            starts - (np.cumsum(lens) - lens), lens
        )
        return self.terms[idx], self.tf[idx], row


def vocabulary() -> dict[str, int]:
    words = (TOKENS / "vocab.txt").read_text(encoding="utf-8").split("\n")
    return {w: i for i, w in enumerate(words)}


# -- one corpus size's index ------------------------------------------------------


def write_index(
    rows,
    doclen: np.ndarray,
    n_vocab: int,
    out: Path,
    k1: float = K1,
    b: float = B,
    chunk: int = 500_000,
) -> dict:
    """Term-major (CSC) BM25 weights, written straight to `out`.

    `rows(a, b)` returns (term ids, counts, row) for documents a..b-1. Two passes:
    document frequencies first, then each chunk's weights are counting-sorted by term
    into place, so no step ever holds a coordinate copy of the whole matrix."""
    n = len(doclen)
    lens = np.asarray(doclen, dtype=np.float32)
    avgdl = float(lens.mean())
    norm = (k1 * (1 - b + b * lens / avgdl)).astype(np.float32)
    df = np.zeros(n_vocab, dtype=np.int64)
    for a in range(0, n, chunk):
        terms, _, _ = rows(a, min(a + chunk, n))
        df += np.bincount(terms, minlength=n_vocab)
    idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
    indptr = np.zeros(n_vocab + 1, dtype=np.int64)
    np.cumsum(df, out=indptr[1:])
    nnz = int(indptr[-1])
    indices = np.lib.format.open_memmap(out / "indices.npy", "w+", np.int32, (nnz,))
    data = np.lib.format.open_memmap(out / "data.npy", "w+", np.float32, (nnz,))
    fill = indptr[:-1].copy()
    for a in range(0, n, chunk):
        terms, tf, row = rows(a, min(a + chunk, n))
        row = row + a
        tf = tf.astype(np.float32)
        w = idf[terms] * tf / (tf + norm[row])
        order = np.argsort(terms, kind="stable")  # stable: rows stay ascending per term
        t_sorted = terms[order]
        uniq, first, counts = np.unique(t_sorted, return_index=True, return_counts=True)
        pos = fill[t_sorted] + (np.arange(len(t_sorted)) - np.repeat(first, counts))
        indices[pos] = row[order]
        data[pos] = w[order]
        fill[uniq] += counts
    if not np.array_equal(fill, indptr[1:]):
        raise RuntimeError("posting lists were not filled exactly; the index is corrupt")
    indices.flush()
    data.flush()
    del indices, data
    np.save(out / "indptr.npy", indptr)
    return {"postings": nnz, "avgdl": round(avgdl, 3), "k1": k1, "b": b}


def build_index(size: str, k1: float = K1, b: float = B) -> dict:
    pids = subset(size)
    cache = TokenCache()
    out = size_dir(size) / "bm25"
    out.mkdir(exist_ok=True)
    with timed() as t:
        info = write_index(
            lambda a, z: cache.rows(pids[a:z]),
            np.asarray(cache.doclen[pids]),
            cache.n_vocab,
            out,
            k1,
            b,
        )
    info = {
        "size": size,
        "passages": len(pids),
        **info,
        "build_s": round(t["s"], 1),
        "index_bytes": sum(dir_bytes(out / f) for f in ("indptr.npy", "indices.npy", "data.npy")),
        "build_peak_rss_gb": peak_rss_gb(),
    }
    write_json(out / "build.json", info)
    return info


class BM25:
    """Query side, over an index held in RAM as a server would hold it."""

    def __init__(self, indptr, indices, data, pids, vocab: dict[str, int], tok: Tokenizer):
        self.indptr, self.indices, self.data, self.pids = indptr, indices, data, pids
        self.vocab, self.tok = vocab, tok
        self._acc = np.zeros(len(pids), dtype=np.float32)

    @classmethod
    def load(cls, size: str, vocab: dict[str, int] | None = None) -> BM25:
        d = size_dir(size) / "bm25"
        return cls(
            np.load(d / "indptr.npy"),
            np.load(d / "indices.npy"),
            np.load(d / "data.npy"),
            subset(size),
            vocab if vocab is not None else vocabulary(),
            Tokenizer(),
        )

    def term_ids(self, text: str) -> list[int]:
        return [i for s in self.tok(text) if (i := self.vocab.get(s)) is not None]

    def search_positions(self, text: str, k: int = DEPTH) -> tuple[np.ndarray, np.ndarray]:
        """Top k as (positions within the subset, scores), best first.

        Repeated query terms count once per occurrence, as in bm25s and Anserini.
        Every BM25 weight is positive, so the non-zero accumulators are exactly the
        passages touched, and resetting them leaves the buffer clean for the next
        query."""
        acc = self._acc
        for t in self.term_ids(text):
            a, b = self.indptr[t], self.indptr[t + 1]
            acc[self.indices[a:b]] += self.data[a:b]  # one posting per passage: no lost adds
        touched = np.flatnonzero(acc)
        cand = touched
        if len(cand) > k:
            cand = cand[np.argpartition(acc[cand], -k)[-k:]]
        scores = acc[cand]
        order = np.lexsort((cand, -scores))  # score, then position: deterministic ties
        top, top_scores = cand[order], scores[order].copy()
        acc[touched] = 0.0
        return top, top_scores

    def search(self, text: str, k: int = DEPTH) -> tuple[np.ndarray, np.ndarray]:
        pos, scores = self.search_positions(text, k)
        return self.pids[pos], scores


def search_all(size: str) -> dict:
    qids, texts = load_queries()
    with timed() as t_load:
        index = BM25.load(size)
    pids = np.full((len(texts), DEPTH), -1, dtype=np.int32)
    scores = np.zeros((len(texts), DEPTH), dtype=np.float32)
    seconds = np.empty(len(texts))
    for i, q in enumerate(texts):
        t0 = time.perf_counter()
        p, s = index.search(q)
        seconds[i] = time.perf_counter() - t0
        pids[i, : len(p)] = p
        scores[i, : len(s)] = s
    save_run(size, "bm25", pids, scores, qids=qids, lat_seconds=seconds[latency_sample()])
    return {"load_s": round(t_load["s"], 1), "mean_ms": round(float(1000 * seconds.mean()), 2)}


def check(size: str = "100k", n_queries: int = 300) -> dict:
    """Score parity with bm25s on the same tokens: every top-10 score, and the
    reference's own top-10 scores, must agree to float32 precision."""
    import bm25s

    pids = subset(size)
    tok = Tokenizer()
    texts = Passages().get(pids)
    ref = bm25s.BM25(method="lucene", k1=K1, b=B)
    ref.index([tok(t) for t in texts], show_progress=False)
    mine = BM25.load(size)
    _, queries = load_queries()
    worst, compared = 0.0, 0
    for q in queries[:n_queries]:
        qt = [s for s in tok(q) if s in ref.vocab_dict]
        if not qt:
            continue
        ref_scores = ref.get_scores(qt)
        pos, s = mine.search_positions(q, k=10)
        ref_top = np.sort(ref_scores)[::-1][: len(s)]
        worst = max(
            worst, float(np.abs(ref_scores[pos] - s).max()), float(np.abs(ref_top - s).max())
        )
        compared += 1
    if worst > 1e-4:
        raise AssertionError(f"BM25 disagrees with bm25s by up to {worst:.2e}")
    return {"queries_compared": compared, "max_abs_diff": worst}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["tokens", "build", "search", "check"])
    ap.add_argument("size", nargs="?", default="100k")
    a = ap.parse_args(argv)
    if a.step == "tokens":
        print(build_tokens())
    elif a.step == "build":
        print(build_index(a.size))
    elif a.step == "search":
        print(search_all(a.size))
    else:
        print(check(a.size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
