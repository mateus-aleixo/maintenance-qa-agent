# Model card — conformal-rag

## What it is

A retrieval-augmented question-answering system over industrial maintenance
manuals, with a **conformal abstention gate**: below a calibrated confidence
threshold it declines to answer rather than guessing, and the wrong-answer rate
among the questions it *does* answer is bounded by α, finite-sample and
distribution-free.

The system is not a single model. It is four separable pieces, and the card
below treats them separately because they fail differently:

| piece | what it is |
|---|---|
| retriever | FTS5 BM25 + brute-force cosine over `bge-small-en-v1.5` vectors, fused by reciprocal rank |
| generator | an external instruction-tuned LLM, not shipped |
| judge | an LLM scoring answers as correct / wrong / refused, used only offline |
| gate | one calibrated scalar threshold, the only component carrying a guarantee |

## Intended use

Benchmarking, teaching and demonstration of selective prediction over RAG, and
as the reference implementation behind the public API in this repository.

**Not** a maintenance authority. It answers questions about a corpus of
public-domain manuals; it has no knowledge of any specific machine, its service
history, or its current state. A person acting on its output is responsible for
verifying the citation it returns.

## Deployment scope, and what is deliberately absent

The public API serves **retrieval and the gate only**. Generation is not hosted:
the served threshold was calibrated against a 14B model, which does not fit in a
Lambda and is not something a free endpoint should pay for per request. `/ask`
answers 503 explaining this. Pointing `LLM_BASE_URL` at any OpenAI-compatible
endpoint enables it, and that is a configuration change rather than a different
code path.

This matters for the card because **the guarantee travels with the generator**.
A threshold fitted against one model says nothing about another, so `/gates` and
every gate response report the generator and score they were calibrated against.

## Data

**Corpus.** Public-domain US Army technical manuals (works of the US federal
government, 17 U.S.C. §105), for example TM 9-8000 *Principles of Automotive
Vehicles*. Real scanned-era industrial PDFs with running headers and page
furniture. 1,752 chunks in the served index. `scripts/fetch_corpus.py` downloads
them; raw PDFs are not committed.

**Question sets.** Hand-written and generated questions with a manually grounded
answer key, held in `evals/`. Sets were expanded over the project from 50 to 100
to 152 to 206 questions; every reported figure names the pool it came from,
because several conclusions changed when the pool grew.

**Known imbalance.** Hand-written sets sit at 0.30–0.42 base error while the
generated set sits at 0.588. The gate is therefore not currently being stressed
where it is weakest. This is the repository's main open limitation.

## The served gate

From `runs/gate_v3_14b.json`, calibrated against `qwen2.5:14b-instruct` with the
`support_v1` score:

| | |
|---|---|
| α | 0.20 |
| threshold | 0.17 |
| calibration size | 27 |
| ungated error rate | 0.296 |
| gated error rate, held out | **0.050** |
| coverage (questions answered) | **74%** |

## Metrics

Full tables in [results.md](results.md). On the 100-question pool:

| risk bounded | ungated | gated | verdict |
|---|---|---|---|
| answered a question the corpus **cannot** answer | 0.260 | 0.031 at 64% coverage | α = 0.1 met |
| **any** mistake (unanswerable or wrong) | 0.540 | 0.312 at 64% coverage | α = 0.4, not 0.2 |

Retrieval reaches recall@5 of 1.00 on the manually grounded set with real
embeddings, so retrieval is not the bottleneck; answer correctness is.

Four results that shaped the design:

1. **Scale bought calibration, not correctness.** With the judge held fixed, the
   14B generator answers slightly *worse* than the 7B (0.373 vs 0.347 base
   error) and gates far better (support on unanswerable questions 0.020 vs
   0.060). What improved with scale was the model's judgement about its own
   evidence, which is exactly what a conformal gate converts into a guarantee.
2. **The binding constraint was score quantisation.** A score the model *writes*
   took 9 distinct values across 152 questions, 143 of them on three values, so
   the risk-coverage curve was a cliff. Reading **P(YES)** from the token
   distribution instead moved granularity to 23 values, AUC 0.697 → 0.821 and
   mean coverage at α = 0.2 from 29% → 45%.
3. **Pick a score by the purity of its head, not by AUC.** A combined
   groundedness × self-consistency signal ranks correctness far better (AUC
   0.845 vs 0.697) and *cannot* meet α = 0.2 at any threshold, while the
   weaker-ranking score can. A gate draws one line and keeps what is above it,
   so only the top bucket matters.
4. **Judges are not interchangeable.** Swapping the generator swaps the judge,
   and the 7B judge is stricter than both the 3B and the 14B. Every generator
   comparison here is re-judged by one fixed judge; an earlier conclusion was
   wrong until it was.

## Limitations

- **The guarantee is conditional on a great deal.** It holds for the generator,
  the score, the corpus and the question distribution it was calibrated on.
  Change any of them and the threshold must be refitted. n = 27 for the served
  gate, so the finite-sample correction is loose.
- **α = 0.2 is bought with coverage.** The gate meets its target by declining
  roughly a quarter to three-quarters of questions depending on configuration.
  Whether that trade is right depends on whether a wrong maintenance answer is
  worse than no answer.
- **The judge is an LLM.** Correctness labels come from a model, not a human, so
  the measured risk inherits the judge's biases. Holding the judge fixed
  controls comparisons between generators; it does not make the labels ground
  truth.
- **Exchangeability is assumed** between the calibration questions and the
  questions a user actually asks. A user whose questions are systematically
  harder than the pool gets no guarantee.
- **Retrieval quality is corpus-specific.** recall@5 of 1.00 reflects a small,
  clean, manually grounded set over one corpus.
- **The reranker was measured and rejected** (+0.02 recall@5 for +899 ms), and
  the anchor-free support prompt fixed granularity but not blindness to answer
  correctness. Both are written up rather than quietly removed.

## Security

Retrieved text is treated as data, never as instructions.
`evals/injection_cases.jsonl` embeds adversarial instructions inside corpus
documents and CI asserts the system does not follow them. This is a tested
property, not a claimed one, but the suite covers the injection patterns
enumerated in it and cannot cover the ones nobody has thought of.

## Safety framing

The abstention is the safety feature. A RAG system's worst failure is a fluent,
well-cited, wrong answer about a procedure someone is about to perform, and no
confidence heuristic carries a guarantee. What this system provides is a bound
on that failure rate among answered questions, at a stated and measured cost in
coverage, with the citation attached so a human can check.

Any real deployment would additionally need: recalibration whenever the
generator, prompt or corpus changes, monitoring of the abstention rate as a
drift signal, and a route for the declined questions, since a system that
abstains without an escalation path has moved the problem rather than solved it.
