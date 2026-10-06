# Retrieval at millions of passages

The serving system searches 1,752 chunks by brute force, and the architecture notes
defend that choice by the corpus size. This benchmark measures where the choice stops
holding, and what the replacements cost, on MS MARCO passage ranking: 100 thousand,
1 million and all 8.8 million passages, scored on the 6,980 dev.small queries.

Results are in [`docs/scale.md`](../docs/scale.md), and the raw numbers in
`runs/scale_*.json`.

## What is compared

| arm | what it is |
|---|---|
| as built | `retrieve.retrieve()` over a `Store`, unchanged: FTS5 BM25, every vector re-read from SQLite per query, RRF |
| bm25 | Lucene BM25 (k1 0.9, b 0.4) over a term-major sparse index held in RAM |
| exact | every query against every passage on the GPU, float32: the ceiling for the encoder |
| brute | the serving arithmetic with the matrix cached in RAM: one matmul and a full sort per query |
| hnsw | FAISS HNSW, M 32, over 8-bit quantised vectors, at several `efSearch` |
| ivfpq | FAISS IVF with 48-byte product quantisation, at several `nprobe`, with and without an exact re-rank of its top 100 |
| hybrid | RRF of bm25 with a dense arm, through the same `rrf_fuse` the system serves |

The encoder is the serving one, `bge-small-en-v1.5`, with the same pooling and no
query instruction. Each corpus size contains every passage judged relevant to a dev
query, plus random distractors from one seeded permutation, so 100k sits inside 1m and
1m inside the full collection: moving along the curve only adds distractors.

Quality comes from batched searches over all 6,980 queries. Latency is timed one query
at a time, over the same 1,000 queries for every arm, with FAISS on one thread.

## Running it

```bash
uv pip install -e ".[scale]"
# a CUDA build of torch, for the embedding run (pick the wheel matching your driver):
uv pip install --reinstall-package torch torch --index-url https://download.pytorch.org/whl/cu126

python -m scale.fetch                 # 1 GB download, checked against the published counts
python -m scale.embed 1m              # GPU; `full` embeds all 8.8M, resumably
python -m scale.bm25 tokens           # tokenize the collection once, in parallel
python -m scale.bm25 build 1m
python -m scale.bm25 search 1m
python -m scale.dense exact 1m
python -m scale.dense brute 1m
python -m scale.dense hnsw 1m
python -m scale.dense ivfpq 1m
python -m scale.asbuilt 1m
python -m scale.report 1m             # scores every run, fuses the hybrids, writes runs/scale_1m.json
```

Or every stage in that order, skipping whatever is already on disk, so an interrupted
run resumes: `python -m scale.run 1m`. At 8.8M the HNSW index is 5.4 GB and its build
peaked at 8.6 GB of working set; `python -m scale.run full --skip hnsw` leaves it for
a quiet machine.

`python -m scale.bm25 check 100k` compares the BM25 index with
[bm25s](https://github.com/xhluca/bm25s) on the same tokens.

Everything lands under `data/msmarco/`, about 23 GB with every index built, and none
of it is committed. MS MARCO is released for non-commercial research use only, so the
text and anything derived from it (vectors, indexes, run files) stay local; only the
measured numbers are in the repository.
