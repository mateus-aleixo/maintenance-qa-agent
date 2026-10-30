# maintenance-qa-agent

[![ci](https://github.com/mateus-aleixo/maintenance-qa-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/mateus-aleixo/maintenance-qa-agent/actions/workflows/ci.yml)

Question answering over industrial maintenance manuals that declines the questions it
can't answer. Retrieval finds the excerpts, a Qwen2.5 model answers with citations, and a
confidence gate calibrated with conformal risk control decides whether the answer goes
out. Retrieval and the gate run live on AWS Lambda.

## Results

Thresholds are fitted on one half of a 100-question pool, and every number below is
measured on the other half.

| risk | without the gate | with the gate | target |
|---|---|---|---|
| answering a question the manuals can't answer | 0.260 | 0.031 at 64% coverage | met at α = 0.1 |
| any mistake (unanswerable or wrong) | 0.540 | 0.312 at 64% coverage | met at α = 0.4, not at 0.2 |

The gate is only as good as the score it ranks answers by. A score that the model writes as
text took just 9 distinct values, so coverage could only move in big steps. Reading P(YES)
from the token probabilities instead raised the ranking AUC from 0.697 to 0.821.

Model size helped the gate more than the answers. Under one fixed judge, the 14B generator
answered slightly worse than the 7B (base error 0.373 against 0.347), but it was much better
at spotting questions its excerpts couldn't answer, and it is the only one that reaches
α = 0.2, at 30% coverage. The judge has to stay fixed in a comparison like this, because
the 7B judge is stricter than the 3B and 14B ones.

[docs/results.md](docs/results.md) has the full tables and the experiments behind them.

## How it works

- Hybrid retrieval: BM25 from SQLite FTS5 plus dense vectors from bge-small-en-v1.5, merged
  with reciprocal rank fusion. Everything lives in one SQLite file.
- The agent loop, tools and guardrails are plain Python, without LangChain. One tool calls
  the live remaining-useful-life API from
  [turbofan-rul](https://github.com/mateus-aleixo/turbofan-rul).
- CI runs prompt-injection cases (`evals/injection_cases.jsonl`): instructions hidden in
  corpus documents have to be treated as text.
- I trained a cross-encoder reranker and left it out. It added 0.02 to recall@5 for 899 ms
  per query.
- The corpus is public-domain US Army technical manuals, downloaded by
  `scripts/fetch_corpus.py`.

More in [docs/architecture.md](docs/architecture.md) and the
[model card](docs/model-card.md).

## Live API

`https://245evfkghe.execute-api.eu-west-1.amazonaws.com`

    curl "$API/retrieve?q=What+does+low+oil+pressure+at+idle+indicate%3F&k=3"
    curl "$API/gate?score=0.05"     # -> abstain, below the calibrated threshold

The service is FastAPI over onnxruntime, with no PyTorch. The embedding model is exported to
ONNX and checked against sentence-transformers (largest difference 2.3e-07) before it
ships. The 14B generator isn't hosted, since it doesn't fit in Lambda, so `/ask` returns 503
unless `LLM_BASE_URL` points at an OpenAI-compatible endpoint. See
[docs/deploy.md](docs/deploy.md).

## Two follow-up studies

- [Retrieval at scale](docs/scale.md): the same code on MS MARCO's 8.8 million passages. An
  HNSW index returns 94% of the exact top 10 in 1.3 ms.
- [Fine-tuning against prompting](docs/finetune.md): QLoRA on Qwen2.5-3B, trained on a 6 GB
  GPU. Prompting stayed ahead.

## Running it

    uv sync --all-extras
    uv run pytest                    # no network, no models
    uv run python -m conformal_rag ingest data/raw/*.pdf
    uv run python -m conformal_rag ask "What does low oil pressure at idle indicate?"

It uses a local Ollama model by default, or any OpenAI-compatible endpoint set through
environment variables. The tests use a deterministic stub, so CI needs no model.

MIT licence.
