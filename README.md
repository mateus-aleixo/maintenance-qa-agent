# conformal-rag

[![ci](https://github.com/mateus-aleixo/conformal-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/mateus-aleixo/conformal-rag/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

**Retrieval-augmented question answering over industrial maintenance manuals that
declines the questions it cannot answer.** Rather than producing a fluent answer to a
question the corpus does not cover, the system **abstains with a calibrated,
distribution-free guarantee** on its error rate: conformal risk control applied to
selective question answering.

Third of a series applying a single principle, *a prediction without a trustworthy
confidence statement is not a decision aid*, to three different kinds of data:

| repo | modality | the guarantee |
|---|---|---|
| [conformal-rul](https://github.com/mateus-aleixo/conformal-rul) | sensor sequences | RUL intervals with verified coverage, live on AWS Lambda |
| [conformal-seg](https://github.com/mateus-aleixo/conformal-seg) | vision | defect masks bounding the missed-defect rate |
| **conformal-rag** | language | selective QA that abstains at a calibrated error rate, retrieval and gate live on AWS Lambda |

The agent in this repo calls the **live conformal-rul API** as one of its tools, so
the series composes rather than merely rhyming.

## Results

Full numbers in [`docs/results.md`](docs/results.md). Thresholds are fitted on one
half of the question pool and every figure is reported from the other half.

On the 100-question pool:

| risk being bounded | ungated | gated | |
|---|---|---|---|
| answered a question the corpus **cannot** answer | 0.260 | **0.031** at 64% coverage | ✅ α = 0.1 |
| **any** mistake (unanswerable *or* wrong answer) | 0.540 | **0.312** at 64% coverage | ⚠️ α = 0.4, not 0.2 |

A conformal gate can only bound a risk that its score can *rank*, and the first score
judged the retrieved excerpts rather than the answer. Replacing it with a score that
looks at what the model actually said, namely agreement across sampled answers
combined with whether the answer is grounded in its own citations, raises the ability
to rank correctness from **AUC 0.51, a coin flip, to 0.75**, and pulls the
any-mistake risk from 0.540 to **0.312 while still answering 64% of questions**.

## Four findings

### 1. Scaling bought calibration, not correctness

α = 0.2 becomes reachable at 14 B, and not for the reason predicted. Every
generator re-judged by the same judge:

| generator | base error | support on *unanswerable* | best α met | coverage |
|---|---|---|---|---|
| 3 B | 0.427 | 0.080 | none | |
| 7 B | **0.347** | 0.060 | 0.40 | 74% |
| 14 B | 0.373 | **0.020** | **0.20** | 30% |

The 14 B model answers *slightly worse* than the 7 B and gates far better. What scale
improved was not accuracy but the model's judgement about its own evidence: it is
much better at recognising when the excerpts cannot answer the question, and a
conformal gate converts exactly that into a guarantee.

The price is coverage. The gate meets its target by declining seven questions in ten.
Whether that is the right trade depends on whether a wrong maintenance answer is
worse than no answer. Here it is.

### 2. The binding constraint was score quantisation, not data volume

Coverage sat at 30% because a score the model *writes* takes only 9 distinct values
across 152 questions, 143 of them on just three values. The risk-coverage curve was
therefore a cliff, with nothing available between 33% and 70% coverage. That single
fact explains what four earlier experiments had blamed on other causes: too little
calibration data, an impure score head, a smaller generator, a worse combination
rule.

The fix was to stop letting the model write the number. Ask one YES/NO question and
read **P(YES)** from the token distribution, which is continuous by construction and
not something a model can round off. Granularity goes from 9 to **23** distinct
values, AUC from 0.697 to **0.821**, and mean coverage at α = 0.2 from **29% to
45%** (300 nested splits, margin chosen on validation, test seen once).

It is a trade rather than a free win: the logprob gate holds α on **76%** of
splits against the written score's 90%, because it operates near the top of its range
where few points fix the threshold. Coverage that was *unreachable at any threshold*
becomes reachable at the cost of a less stable one.

Rescored on an expanded 206-question pool, that trade largely dissolves: **63%
coverage**, with the interquartile range down from 31 points wide to 9, and the hold
rate unchanged at 77%. The gate had been over-conservative for want of calibration
data, running at risk 0.107 against a budget of 0.20. What the extra data buys is
permission to spend the budget.

### 3. Choose a nonconformity score by the purity of its head, not by AUC

This is the sharpest methodological result here. A combined groundedness ×
self-consistency signal ranks correctness far better, **AUC 0.845 against 0.697**,
and *cannot* meet α = 0.2 at any threshold, while the weaker-ranking score can.

AUC describes the whole ordering. A conformal gate draws one line and keeps what sits
above it, so the only property that matters is whether some **top bucket is nearly
pure**. Selecting a score on AUC picks the score that cannot deliver the guarantee.
At a looser α = 0.30 the combined score is the right pick after all, at 60%
coverage against 34%.

### 4. Hold the judge fixed when comparing generators

Swapping the model swaps the **judge** as well, and judges vary unpredictably: the
7 B judge is stricter than both the 3 B and the 14 B. This cost a wrong conclusion
before it was caught, and every generator comparison above is re-judged by one fixed
judge for that reason.

## Why abstention, and why conformal

RAG systems fail worst on the questions they cannot answer. Retrieval returns
something vaguely related, the model writes a fluent paragraph, and the reader has no
way to tell it is wrong. Confidence heuristics ("the model seemed unsure") carry no
guarantee.

Conformal risk control does. Given a calibration set of questions labelled
answered-correctly or answered-wrongly, choose the confidence threshold

λ̂ = inf { λ : (n/(n+1)) · R̂(λ) + 1/(n+1) ≤ α }

and answer only above it. Under exchangeability the result is that **the wrong-answer
rate among answered questions is ≤ α**, finite-sample, with no
distributional assumptions (Angelopoulos et al., 2022; Mohri and Hashimoto, 2024). A
Mondrian split gives per-question-type thresholds with a small-group fallback, the
same construction used in conformal-rul for operating regimes.

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

Design choices, deliberately boring where boring is right:

- **Framework-free.** The agent loop, tool protocol, guardrails and conformal gate
  are a few hundred lines of plain Python. No LangChain. When the behaviour is wrong,
  the bug is in this repository, and it is findable.
- **SQLite for everything at rest.** FTS5 provides BM25; embeddings live as blobs and
  are searched with NumPy. The corpus is thousands of chunks rather than billions, so
  brute force takes milliseconds and has no failure modes. One file, deploys
  anywhere.
- **Provider-agnostic LLM client.** Local [Ollama](https://ollama.com) by default,
  any OpenAI-compatible endpoint via environment variables, and a deterministic stub
  for tests, so CI runs the full suite with no model downloads and no API spend.
- **Security is tested, not claimed.** `evals/injection_cases.jsonl` embeds
  adversarial instructions inside corpus documents, and CI asserts that the system
  treats retrieved text as data rather than as orders.

## Corpus

Public-domain **US Army technical manuals** (works of the US federal government,
17 U.S.C. §105), for example TM 9-8000 *Principles of Automotive Vehicles*. Real,
messy, industrial PDFs. `scripts/fetch_corpus.py` downloads them, and raw PDFs stay
out of git.

## Quickstart

```bash
uv sync --all-extras          # or: pip install -e ".[dev]"
uv run pytest                 # full suite, no network, no models
uv run python -m conformal_rag ingest data/raw/*.pdf
uv run python -m conformal_rag ask "What does low oil pressure at idle indicate?"
uv run python -m conformal_rag agent "Remaining life for these engine readings: ..."
```

## Serving

**Live**: `https://245evfkghe.execute-api.eu-west-1.amazonaws.com`

```bash
curl "$API/gates"
curl "$API/retrieve?q=What+does+low+oil+pressure+at+idle+indicate%3F&k=3"
curl "$API/gate?score=0.05"     # -> abstain, below the calibrated threshold
```

Torch-free, on the conformal-rul and conformal-seg container pattern: FastAPI over
onnxruntime, one image that runs under uvicorn locally and unchanged on Lambda.
`bge-small-en-v1.5` is exported to ONNX with CLS pooling and L2 normalisation baked
into the graph, checked against sentence-transformers at max |diff| 2.3e-07 before it
is allowed to ship, because a pooling mismatch does not raise: it returns a vector
pointing somewhere else and quietly degrades recall against an index built the other
way.

**The generator is deliberately not hosted.** The sibling repos ship their own
networks, a few MB of ONNX apiece. This one's is a 14B model, which does not fit in a
Lambda and is not something a free demo endpoint should pay for per request. So what
is served is the part that carries the guarantee and is genuinely serverless:
retrieval, and the calibrated gate. `/ask` returns 503 explaining exactly that, and
becomes available by pointing `LLM_BASE_URL` at any OpenAI-compatible endpoint, which
is configuration rather than a different code path.

Splitting `/gate` out from generation is not a workaround. The gate is a total
function of the score, so anyone can post their own generator's number and get the
decision this system would make, with the threshold and the measured risk it bought
returned alongside.

Deployment details, and the three separate ways SQLite fails on Lambda's read-only
filesystem, are in [docs/deploy.md](docs/deploy.md).

## Documentation

[Results](docs/results.md) ·
[Architecture](docs/architecture.md) ·
[Model card](docs/model-card.md) ·
[Deployment](docs/deploy.md)

## Measured and rejected

Two planned components were built, measured and then dropped. They are written up
rather than quietly removed, because the measurement is the useful part:

- **A trained reranker**, which bought +0.02 recall@5 for +899 ms of latency.
- **An anchor-free support prompt**, which fixed the score's granularity but not its
  blindness to answer correctness.

## Limits and next steps

Every hand-written question set sits at 0.30 to 0.42 base error while the generated
set sits at 0.588, so the gate is not currently being tested where it is weakest. The
next step is a deliberately harder hand-written batch.

Out of scope for a first version: a UI, multi-corpus retrieval, a fine-tuned
generator, Kubernetes, and streaming.

## References

- Angelopoulos, Bates, Fisch, Lei, Schuster, *Conformal Risk Control*, 2022.
- Mohri, Hashimoto, *Language Models with Conformal Factuality Guarantees*, 2024.
- Barber, Candès, Ramdas, Tibshirani, *Predictive inference with the jackknife+*,
  2021.

## License

MIT. Built by [Mateus Aleixo](https://github.com/mateus-aleixo).
