# Retrieval at 8.8 million passages

The served index holds 1,752 chunks, and the [architecture notes](architecture.md)
defend searching them by brute force on that basis: a few thousand vectors are
milliseconds of NumPy. This page measures where that stops being true, and what each
replacement costs, on MS MARCO passage ranking at 100 thousand, 1 million and all
8.8 million passages.

Code and instructions are in [`scale/`](../scale/README.md); the raw numbers are in
`runs/scale_100k.json`, `runs/scale_1m.json` and `runs/scale_full.json`.

## Setup

- **Corpus.** The MS MARCO passage collection, 8,841,823 passages. The two smaller
  sizes are nested subsets: each holds all 7,433 passages judged relevant to a dev
  query, plus random distractors from one seeded permutation, so 100k sits inside 1M
  and 1M inside the full collection. Moving along the curve only ever adds
  distractors.
- **Queries.** dev.small: 6,980 queries with 7,437 judgements, about one relevant
  passage per query. MRR@10 is MS MARCO's official metric; recall@100 is what a
  re-ranking stage would get to see.
- **Encoder.** The served `bge-small-en-v1.5`, CLS pooling, no query instruction. The
  collection was embedded on the GPU in float16, at about 1,800 passages a second.
  Queries are timed with the served ONNX export on the CPU, whose vectors match the
  GPU ones to a cosine of 0.999999. Encoding one query takes 4.4 ms at the median,
  and every hybrid latency below includes it.
- **Timing.** Every arm answers the same 1,000 queries, one at a time, and every
  latency on this page comes from one session with no other heavy job running. FAISS
  runs on one thread; brute force uses whatever NumPy's BLAS takes, which is all 16.
  One laptop: Ryzen 7 6800H, 15.2 GB of RAM, RTX 3060 Laptop GPU with 6 GB.

## 1. The served design takes 4.9 seconds a query at a million passages

`retrieve.retrieve()` over a `Store` is exactly what `/retrieve` runs: FTS5 BM25
inside SQLite, every stored vector read back out of SQLite, one matmul and a full
sort, then reciprocal rank fusion. Only the corpus changes. Because a query takes
seconds at 1M, this arm is timed on the first 200, 100 and 20 of the 1,000 queries.

| corpus | retrieve, p50 | p95 | FTS5 BM25, p50 | vector search, p50 | peak RAM |
|---|---:|---:|---:|---:|---:|
| 1,752 chunks (served) | 15 ms | 21 ms | 3.5 ms | 11 ms | 0.2 GB |
| 100k | 446 ms | 552 ms | 131 ms | 306 ms | 0.5 GB |
| 1M | 4.9 s | 5.8 s | 1.2 s | 3.6 s | 3.3 GB |

Neither half slows down for an arithmetic reason. The vector path reads the whole
vector table back out of SQLite on every query: the matmul and sort alone take 65 ms
at a million (section 2), so about 3.5 of its 3.6 seconds is that reload. FTS5 joins
the query's words with OR and keeps stopwords, so before it can return its top 20 it
scores every passage containing "what" or "the". At 8.8 million passages the vector
path would assemble a 12.6 GB float32 matrix per query on a 15.2 GB machine, so it
was not run.

## 2. Brute force stops fitting before it stops being fast

With the matrix held in RAM, the reload is gone and the arithmetic is what remains:
5.8 ms a query at 100k and 65 ms at 1M. At 8.8M the float32 matrix is 12.6 GB, and
even in float16 it is 6.3 GB, more than the 6 GB GPU. The GPU can still search
exactly in batches, streaming vectors from disk: all 6,980 queries against all
8.8 million passages take 18 seconds, which is how the exact rows below were
computed, but that answers a batch, not a request.

## 3. HNSW answers in about a millisecond at every size

FAISS HNSW (M = 32, vectors quantised to 8 bits) at efSearch 128 answers in 1.3 ms
at the median and 2.1 ms at p95 on all 8.8 million passages, returning 94.2% of the
exact search's top 10 (96.4% at 1M, 98.3% at 100k), with MRR@10 0.327 against
exact's 0.344. Its latency does not grow with the corpus, 1.2, 1.4 and 1.3 ms at the
three sizes, while brute force grows elevenfold between 100k and 1M alone. Raising
efSearch to 256 keeps 95.6% at 2.2 ms. The cost is memory and build time: the 8.8M
index is 5.40 GB and took 24 minutes to build on 16 threads, against 497 MB for
IVF-PQ.

## 4. IVF-PQ holds all 8.8 million passages in 497 MB

IVF with 16,384 clusters and 48-byte product-quantisation codes stores the whole
collection in 497 MB, against 6.3 GB for the float16 vectors. Quantisation alone is
expensive: at nprobe 64 the index returns 56.5% of the exact top 10. Scoring its top
100 again with the exact vectors, read from the float16 file on disk, raises that to
86.8% and MRR@10 from 0.264 to 0.317, against exact's 0.344, at 4.2 ms median and
4.7 ms p95. The file stays on disk; the re-rank was timed with it partly in the
operating system's cache, as it would be on a server that has been running.

## 5. Fusion lowers the ranking it is fused into

Reciprocal rank fusion of BM25 with the exact dense ranking scores below the dense
ranking alone at every size: MRR@10 0.794 against 0.860 at 100k, 0.567 against 0.643
at 1M, 0.288 against 0.344 at 8.8M. Recall@100 rises by 0.002 at 100k and 1M, and at
8.8M it falls as well, 0.866 against 0.871.

RRF gives both rankings an equal say, which suits retrievers of similar strength.
Here the encoder is well ahead of BM25 at every size (0.344 against 0.184 at 8.8M),
and an equal say pulls the better list down. This matters beyond MS MARCO because the
served system fuses by default, and its golden set has never been scored with each
retriever on its own, so whether fusion earns its place on the manuals is untested.

## 6. Both baselines reproduce published numbers

- **BM25:** MRR@10 0.1839 at 8.8M, against
  [Anserini's](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md)
  0.1840 with the same k1 = 0.9 and b = 0.4. The tokenizer produces 340,859,891
  tokens for the collection, the count
  [bm25-benchmarks](https://github.com/xhluca/bm25-benchmarks) lists for bm25s, and
  at 100k the scores match bm25s to within 3.8e-6.
- **Dense:** MRR@10 0.3435 and nDCG@10 0.4053, against the 0.3459 and 0.4083 that
  MTEB records for `bge-small-en-v1.5` on the same queries and collection.

## All the numbers

Both retrievers lose ground as distractors are added, BM25 faster: its MRR@10 falls
from 0.674 at 100k to 0.184 at 8.8M, the encoder's from 0.860 to 0.344.

**MRR@10**

| arm | 100k | 1M | 8.8M |
|---|---:|---:|---:|
| BM25 | 0.674 | 0.421 | 0.184 |
| exact, GPU | 0.860 | 0.643 | 0.344 |
| HNSW, efSearch 128 | 0.855 | 0.629 | 0.327 |
| IVF-PQ, nprobe 64 | 0.787 | 0.537 | 0.264 |
| IVF-PQ, nprobe 64, re-ranked | 0.835 | 0.605 | 0.317 |
| RRF of BM25 and exact | 0.794 | 0.567 | 0.288 |
| RRF of BM25 and HNSW | 0.792 | 0.559 | 0.280 |
| RRF of BM25 and IVF-PQ re-ranked | 0.780 | 0.547 | 0.275 |

**Recall@100**

| arm | 100k | 1M | 8.8M |
|---|---:|---:|---:|
| BM25 | 0.943 | 0.845 | 0.660 |
| exact, GPU | 0.993 | 0.968 | 0.871 |
| HNSW, efSearch 128 | 0.986 | 0.944 | 0.830 |
| IVF-PQ, nprobe 64 | 0.955 | 0.883 | 0.743 |
| IVF-PQ, nprobe 64, re-ranked | 0.955 | 0.883 | 0.743 |
| RRF of BM25 and exact | 0.995 | 0.970 | 0.866 |
| RRF of BM25 and HNSW | 0.995 | 0.968 | 0.859 |
| RRF of BM25 and IVF-PQ re-ranked | 0.992 | 0.957 | 0.835 |

**Share of the exact top 10 that each index returns**

| index | 100k | 1M | 8.8M |
|---|---:|---:|---:|
| HNSW, efSearch 32 | 0.949 | 0.917 | 0.887 |
| HNSW, efSearch 64 | 0.972 | 0.946 | 0.920 |
| HNSW, efSearch 128 | 0.983 | 0.964 | 0.942 |
| HNSW, efSearch 256 | 0.988 | 0.974 | 0.956 |
| IVF-PQ, nprobe 8 | 0.570 | 0.528 | 0.502 |
| IVF-PQ, nprobe 16 | 0.598 | 0.561 | 0.531 |
| IVF-PQ, nprobe 32 | 0.616 | 0.581 | 0.551 |
| IVF-PQ, nprobe 64 | 0.626 | 0.594 | 0.565 |
| IVF-PQ, nprobe 128 | 0.630 | 0.603 | 0.575 |
| IVF-PQ, nprobe 32, re-ranked | 0.898 | 0.850 | 0.834 |
| IVF-PQ, nprobe 64, re-ranked | 0.936 | 0.889 | 0.868 |

**Latency, p50 / p95.** Search only, except the fused rows, which run one request
end to end: encode the query, both searches, fuse.

| arm | 100k | 1M | 8.8M |
|---|---:|---:|---:|
| BM25 | 0.5 ms / 0.9 ms | 4.7 ms / 9.4 ms | 50 ms / 97 ms |
| brute force, matrix in RAM | 5.8 ms / 6.3 ms | 65 ms / 74 ms | does not fit |
| HNSW, efSearch 128 | 1.2 ms / 1.6 ms | 1.4 ms / 2.1 ms | 1.3 ms / 2.1 ms |
| IVF-PQ, nprobe 64 | 0.3 ms / 0.4 ms | 0.9 ms / 1.1 ms | 2.6 ms / 3.0 ms |
| IVF-PQ, nprobe 64, re-ranked | 2.2 ms / 2.4 ms | 2.6 ms / 2.9 ms | 4.2 ms / 4.7 ms |
| RRF of BM25 and brute force | 11 ms / 12 ms | 75 ms / 85 ms | does not fit |
| RRF of BM25 and HNSW | 6.2 ms / 7.7 ms | 11 ms / 16 ms | 56 ms / 103 ms |
| RRF of BM25 and IVF-PQ re-ranked | 7.2 ms / 8.8 ms | 12 ms / 17 ms | 59 ms / 105 ms |

At 8.8M a fused request is mostly BM25: 50 of the 56 ms with HNSW.

**Index size and build time**

| index | 100k | 1M | 8.8M |
|---|---:|---:|---:|
| BM25 | 33 MB, 4 s | 233 MB, 31 s | 1.93 GB, 50 s |
| HNSW | 63 MB, 21 s | 626 MB, 236 s | 5.40 GB, 1,467 s |
| IVF-PQ | 7 MB, 92 s | 60 MB, 106 s | 497 MB, 521 s |
| float32 matrix for brute force | 0.14 GB | 1.43 GB | 12.6 GB, does not fit |

The BM25 builds start from a token cache made once for the whole collection, in
83 seconds on six processes. About a minute of every IVF-PQ build is training the 48
product quantisers, which does not grow with the collection. Build times at 100k and
1M were taken while the GPU embedding ran beside them, so they are upper bounds.

## What this does not show

- One laptop answering one query at a time: no concurrency and no network.
  Medians and 95th percentiles are over the same 1,000 queries.
- MS MARCO judges about one passage per query and leaves many relevant ones
  unjudged, so absolute recall is understated. Comparisons between arms on the same
  queries are what carry.
- One small encoder, and BM25 at Anserini's default parameters, not tuned further.
- Peak RAM for the builds, in `runs/`, includes pages of the memory-mapped vector
  file, which the operating system can drop. The index sizes above are what a
  server would hold.
