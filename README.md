# maintenance-qa-agent

[![ci](https://github.com/mateus-aleixo/maintenance-qa-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/mateus-aleixo/maintenance-qa-agent/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Question answering over industrial maintenance manuals that declines the questions it
cannot answer. It uses retrieval-augmented generation (RAG), and a confidence gate
calibrated with conformal risk control decides whether to answer. The gate gives a
distribution-free bound on the error rate among the questions it does answer.

Retrieval and the calibrated gate are live on AWS Lambda (see [Serving](#serving)). The
agent can also call a live remaining-useful-life API, served by
[turbofan-rul](https://github.com/mateus-aleixo/turbofan-rul), as one of its tools.

## Results

Full numbers are in [`docs/results.md`](docs/results.md). Thresholds are fitted on one half
of the question pool, and every figure is reported on the other half.

On the 100-question pool:

| risk being bounded | without the gate | with the gate | target |
|---|---|---|---|
| answering a question the corpus cannot answer | 0.260 | 0.031 at 64% coverage | met at α = 0.1 |
| any mistake (an unanswerable question or a wrong answer) | 0.540 | 0.312 at 64% coverage | met at α = 0.4, not at 0.2 |

A conformal gate can only bound a risk that its score can rank. The first score only
judged the retrieved excerpts. The current score looks at what the model actually said:
it combines agreement across sampled answers with whether the answer is supported by its
own citations. This raised the AUC for ranking correct answers from 0.51, no better than
chance, to 0.75, and lowered the any-mistake risk from 0.540 to 0.312 while still
answering 64% of questions.

## Findings

### 1. The larger model was better calibrated, but not more accurate

At 14B, α = 0.2 becomes reachable, but for an unexpected reason. Every generator, re-judged
by the same judge:

| generator | base error | support on unanswerable questions | best α met | coverage |
|---|---|---|---|---|
| 3B | 0.427 | 0.080 | none | |
| 7B | 0.347 | 0.060 | 0.40 | 74% |
| 14B | 0.373 | 0.020 | 0.20 | 30% |

The 14B model answers slightly worse than the 7B model, but its gate works far better.
What scale improved was the model's judgement about its own evidence. It is much better at
recognising when the excerpts cannot answer the question, and the conformal gate turns
exactly that into a guarantee.

The cost is coverage: to meet its target, the gate declines seven questions in ten.
Whether that is a good trade depends on whether a wrong maintenance answer is worse than
no answer, which it is here.

### 2. Coverage was limited by how few values the score could take

Coverage stayed at 30% because a score that the model writes as text takes only 9 distinct
values across 152 questions, and 143 of them fall on just three values. The risk-coverage
curve was therefore a cliff, with nothing between 33% and 70% coverage. This one fact
explains what four earlier experiments had put down to other causes: too little
calibration data, an impure score head, a smaller generator, and a worse combination
rule.

The fix was to stop asking the model to write the number. The gate now asks one yes/no
question and reads P(YES) from the token probabilities, which is continuous and cannot be
rounded off by the model. The number of distinct values goes from 9 to 23, the AUC from
0.697 to 0.821, and the mean coverage at α = 0.2 from 29% to 45% (300 nested splits,
margin chosen on validation, test set used once).

This has a cost. The logprob gate holds α on 76% of splits, against 90% for the written
score, because it operates near the top of its range, where only a few points fix the
threshold. Coverage that no threshold could reach before becomes reachable, but with a
less stable threshold.

On an expanded pool of 206 questions, most of that cost goes away: coverage is 63%, the
interquartile range narrows from 31 points to 9, and the hold rate stays at 77%. The gate
had been too conservative because it had too little calibration data, running at a risk
of 0.107 against a budget of 0.20. With more data it can use the budget it has.

### 3. A score with a better AUC can be worse for the gate

A combined groundedness × self-consistency score ranks correct answers far better (AUC
0.845 against 0.697), yet it cannot meet α = 0.2 at any threshold, while the score with
the lower AUC can.

AUC describes the whole ordering. A conformal gate draws one line and keeps what is above
it, so what matters is whether some top bucket of the score is nearly pure. Choosing a
score by AUC can therefore pick one that cannot deliver the guarantee. At a looser
α = 0.30, the combined score is the better choice, with 60% coverage against 34%.

### 4. The judge has to stay fixed when comparing generators

Changing the model also changes the judge, and judges vary in ways that are hard to
predict: the 7B judge is stricter than both the 3B and the 14B. This led to a wrong
conclusion before it was noticed, so every generator comparison above is re-judged by one
fixed judge.

## Why abstain, and why conformal

RAG systems fail worst on the questions they cannot answer. Retrieval returns something
loosely related, the model writes a fluent paragraph, and the reader has no way to tell
that it is wrong. Confidence heuristics, such as the model seeming unsure, come with no
guarantee.

Conformal risk control does give one. Given a calibration set of questions labelled as
answered correctly or wrongly, it chooses the confidence threshold

λ̂ = inf { λ : (n/(n+1)) · R̂(λ) + 1/(n+1) ≤ α }

and answers only above it. Under exchangeability, the wrong-answer rate among answered
questions is then at most α, in finite samples and with no assumptions about the
distribution (Angelopoulos et al., 2022; Mohri and Hashimoto, 2024). A Mondrian split gives
a separate threshold for each question type, with a fallback for small groups, the same
construction turbofan-rul uses for operating regimes.

## Architecture

```
PDFs ──ingest──> chunks ──> SQLite (FTS5 BM25 + embedded vectors)
                                    │
question ──> hybrid retrieval (RRF) ──> [rerank: measured, not adopted]
                                    │
                             LLM with citations ──> JSONL trace (tokens, latency, cost)
                                    │
                  groundedness × self-consistency  =  nonconformity score
                                    │
                             conformal gate ──abstain──> "cannot answer, and why"
                                    ▲
        agent loop (from scratch): search_docs · predict_rul (live API) · calculator
```

Design choices:

- **No framework.** The agent loop, tool protocol, guardrails and conformal gate are a few
  hundred lines of plain Python, without LangChain, so when the behaviour is wrong the bug
  is in this repository and can be found.
- **SQLite for all stored data.** FTS5 provides BM25, and embeddings are stored as blobs
  and searched with NumPy. The corpus has thousands of chunks, so brute-force search takes
  milliseconds and has little that can go wrong. It is a single file and deploys anywhere.
- **Any LLM provider.** Local [Ollama](https://ollama.com) by default, any
  OpenAI-compatible endpoint through environment variables, and a deterministic stub for
  tests, so CI runs the full suite with no model downloads and no API costs.
- **Prompt-injection tests.** `evals/injection_cases.jsonl` embeds adversarial
  instructions in corpus documents, and CI checks that the system treats retrieved text as
  data and does not follow instructions found in it.

## Corpus

Public-domain US Army technical manuals (works of the US federal government, 17 U.S.C.
§105), for example TM 9-8000 *Principles of Automotive Vehicles*. They are real industrial
PDFs, with all the mess that comes with them. `scripts/fetch_corpus.py` downloads them, and
the raw PDFs are kept out of git.

## Quickstart

```bash
uv sync --all-extras          # or: pip install -e ".[dev]"
uv run pytest                 # full suite, no network, no models
uv run python -m conformal_rag ingest data/raw/*.pdf
uv run python -m conformal_rag ask "What does low oil pressure at idle indicate?"
uv run python -m conformal_rag agent "Remaining life for these engine readings: ..."
```

## Serving

Live: `https://245evfkghe.execute-api.eu-west-1.amazonaws.com`

```bash
curl "$API/gates"
curl "$API/retrieve?q=What+does+low+oil+pressure+at+idle+indicate%3F&k=3"
curl "$API/gate?score=0.05"     # -> abstain, below the calibrated threshold
```

The service does not need PyTorch: it is FastAPI over onnxruntime, in one image that runs
under uvicorn locally and unchanged on Lambda. `bge-small-en-v1.5` is exported to ONNX with
CLS pooling and L2 normalisation built into the graph, and it is checked against
sentence-transformers (maximum absolute difference 2.3e-07) before it can ship. A pooling
mismatch would not raise an error. It would return vectors that point somewhere else and
quietly lower recall against an index built the other way.

The generator is not hosted. It is a 14B model, which does not fit in a Lambda function
and would be expensive to run per request for a free demo. The service therefore hosts the
parts that carry the guarantee and can run serverless: retrieval and the calibrated gate.
`/ask` returns 503 with that explanation, and it works as soon as `LLM_BASE_URL` points at
any OpenAI-compatible endpoint, with no change to the code.

`/gate` is separate from generation on purpose. The gate depends only on the score, so
anyone can post their own generator's score and get the decision this system would make,
together with the threshold and the measured risk behind it.

Deployment details, including three different ways SQLite fails on Lambda's read-only file
system, are in [docs/deploy.md](docs/deploy.md).

## Retrieval at scale

Brute-force search is the right choice at 1,752 chunks. [docs/scale.md](docs/scale.md)
measures where it stops being right, with the same retrieval code pointed at MS MARCO's 8.8
million passages. Unchanged, the served path takes 4.9 s per query at a million passages,
almost all of it spent re-reading every vector from SQLite and running FTS5 queries that
match any passage containing "what" or "the". An HNSW index returns 94% of the exact top 10
in 1.3 ms on all 8.8 million passages, using 5.4 GB of RAM, and IVF with product
quantisation fits them in 497 MB. Fusing BM25 into the dense ranking lowers MRR@10 at every
size, so it is still an open question whether fusion helps on the manuals. BM25 and the
encoder both reproduce their published MS MARCO scores to within 0.003.

## Documentation

[Results](docs/results.md) ·
[Architecture](docs/architecture.md) ·
[Retrieval at scale](docs/scale.md) ·
[Model card](docs/model-card.md) ·
[Deployment](docs/deploy.md)

## Measured and dropped

Two planned components were built, measured and then dropped. They are documented here
because the measurements are useful:

- A trained reranker, which added 0.02 to recall@5 and 899 ms of latency.
- A support prompt without anchors, which fixed the score's granularity but still could not
  tell correct answers from wrong ones.

## Limits and next steps

Every hand-written question set has a base error between 0.30 and 0.42, while the
generated set has 0.588, so the gate has not yet been tested where it is weakest. The next
step is a harder hand-written batch.

Not in scope for a first version: a user interface, retrieval across several corpora, a
fine-tuned generator, Kubernetes, and streaming.

## References

- Angelopoulos, Bates, Fisch, Lei, Schuster, *Conformal Risk Control*, 2022.
- Mohri, Hashimoto, *Language Models with Conformal Factuality Guarantees*, 2024.
- Barber, Candès, Ramdas, Tibshirani, *Predictive inference with the jackknife+*,
  2021.

## License

MIT. Built by [Mateus Aleixo](https://github.com/mateus-aleixo).
