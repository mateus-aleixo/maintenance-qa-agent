# Results

**No benchmark numbers yet: honestly.** Retrieval recall, answer correctness and
abstention risk require the manually verified golden set (M1/M4); until then this
file records only what has actually been run, which is the pipeline working
end to end.

## Verified end to end

Corpus: 1,752 chunks from two public-domain US Army technical manuals.
Model: `qwen2.5:3b-instruct` via local Ollama. Retrieval: hybrid BM25 + vectors,
RRF-fused.

**Grounded answer with a citation.** *"What is the purpose of the cooling system
thermostat in an engine?"* →

> The thermostat in an engine's cooling system regulates engine temperature by
> controlling the amount of coolant flowing from the engine block to the radiator
> core. It operates based on heat, and if it fails, it will be in the opened
> position so as to allow the free circulation of coolant through the engine **[4]**.

with `[4]` resolving to TM-9-8000 p.241. All five hits found by **both** retrievers.

**Correct refusal.** *"What should be inspected before starting the generator
set?"* → `INSUFFICIENT EVIDENCE`. Retrieval surfaced only the automotive manual
(the generator document in the corpus is a parts catalogue with no pre-start prose),
and the system declined rather than improvising from adjacent text. This is the
intended behaviour and it happened without the conformal gate even being fitted:
the gate makes the refusal rate *calibrated*, the prompt makes it *possible*.

**Agent tool loop.** `calculator` → `17 * 23` → `391`, one step, clean JSON protocol.

**Cross-project composition.** `predict_rul` called the live
[turbofan-rul](https://github.com/mateus-aleixo/turbofan-rul) service on AWS
Lambda with a real C-MAPSS cycle:

```json
{"rul_cycles": 118.9,
 "interval": {"lower": 87.5, "upper": 125.0, "coverage": 90},
 "risk_band": "healthy", "model": "transformer"}
```

## Retrieval

Golden set: 20 questions (15 answerable, 5 unanswerable), written by **reading the
source chunks**, not by asking the retriever where it would look: grounding gold on
the retriever's own top hit would only measure its tie-breaking. Scored at page level
with ±1 slack, because chunks overlap and a boundary can split an answer.
Corpus: 1,752 chunks. `python scripts/eval_retrieval.py --embedder {hash,bge}`

| embedder | recall@5 | recall@1 | MRR | top hit found by both retrievers |
|---|---|---|---|---|
| `hash` (deterministic placeholder, CI default) | 0.93 | 0.73 | 0.851 | 18/20 |
| **`bge-small-en-v1.5`** | **1.00** | **0.93** | **0.950** | 19/20 |

Real embeddings fix the one miss: a "what keeps the brakes working if the power
steering fails?" question that the lexical path answered with the wrong page while
reporting **high** confidence (0.969). Hybrid retrieval is doing real work either
way: the top hit was found by *both* BM25 and the vector search on 18–19 of 20
questions, so this is not BM25 with extra steps.

### The finding that matters, and it is not the recall

| embedder | mean confidence, answerable | unanswerable | **gap** |
|---|---|---|---|
| `hash` | 0.946 | 0.643 | **0.303** |
| `bge` | 0.992 | 0.831 | **0.161** |

**Better retrieval made the abstention signal worse.** Recall went up and the gap
between answerable and unanswerable questions roughly halved, because a stronger
retriever confidently finds *something* topically plausible even when the corpus
cannot answer the question. Retrieval agreement measures "did my retrievers concur",
which is not the same as "is the answer in here".

That is a problem for M4, and a useful one: it means the conformal gate cannot be
built on retrieval confidence alone, and the nonconformity score needs a
judge-based signal. Better to learn that from a 20-question eval than from a
calibration that quietly certifies confident nonsense. The unanswerable questions
were written to be *plausible* (a torque spec for an engine the corpus does not
cover) rather than absurd, which is precisely why they are hard to separate.

## Answers

Same 20 questions through the full pipeline. `qwen2.5:3b-instruct` via local
Ollama, bge retrieval. `python scripts/eval_answers.py --provider ollama`

Three things are scored, all decidable without a judge model:

| | | |
|---|---|---|
| **Refusal on unanswerable** | **5 / 5 = 1.00** | said `INSUFFICIENT EVIDENCE` rather than improvising |
| Answer rate on answerable | 14 / 15 = 0.93 | |
| **Invalid citations** | **0 / 20 responses** | every `[n]` indexed a excerpt that was actually supplied |
| Answers carrying a citation | 14 / 14 = 1.00 | |
| Median latency | 3.7 s | 3 B model, CPU-class hardware |

Semantic correctness ("is the answer *right*") is deliberately **not** scored
here. That needs a judge model and a rubric (M4/M5). Approximating it with string
overlap would produce a number that looks like accuracy and isn't.

### This answers the question M1 raised

M1 found that better retrieval *shrank* the confidence gap between answerable and
unanswerable questions (0.303 → 0.161), which made retrieval agreement look like a
poor basis for abstention. M2 shows where the signal actually lives: **the model
reading the excerpts separated them perfectly, 5 out of 5, where retrieval
confidence could not.**

So the M4 nonconformity score should be built on the generation step, not the
retrieval step. That is a design decision now supported by evidence rather than by
taste, and it is the opposite of what the v0 confidence heuristic assumed.

**Sample-size caveat, stated plainly:** 5/5 on five questions is not a 100% refusal
rate. It is "no failures observed in five attempts", whose 95% upper bound is
roughly 45%. The number to trust is the *direction*, not the value. Expanding the
unanswerable set is the first task of M4, precisely because the gate will be
calibrated against it.

### The one over-refusal

`g-05`: *"Why does damping on rebound only force the use of stiffer springs?"*:
was refused despite the answer being on the retrieved page. It is the only
**reasoning**-type question that required combining two sentences rather than
quoting one. The system errs conservative, which is the right direction for a
maintenance assistant, but it marks a real ceiling: strict grounding prompts
suppress inference, and questions needing a short chain of reasoning are where
that costs recall.

## The conformal gate

Golden set grown to **100 questions** (75 answerable, 25 not): the 20 hand-written
ones plus 80 generated from real chunks (`scripts/gen_golden.py`; its sampling bias is
documented in that file). Each question scored: retrieve → `support_score` → answer →
judge against the reference. Threshold fitted on a random half, every number below
from the other half.

![Risk against the support threshold](figures/risk_curve.png)

### The gate does not work for the risk it was designed for

| loss | ungated risk | best achievable | verdict |
|---|---|---|---|
| **any mistake** (unanswerable *or* wrong answer) | 0.540 | 0.471 at any threshold; α=0.5 is the first that is met | **fails α ≤ 0.4** |
| **answered an unanswerable question** | 0.260 | **0.031** at threshold 0.50, still answering **64%** | **works at α = 0.1** |

For the second loss this is a real result: the rate of answering questions the corpus
cannot answer falls from **26% to 3.1%**, while two thirds of questions still get
answered. That is what a working abstention gate looks like.

For the first, no threshold helps. The reason is one line of arithmetic:

```
support score, answerable vs unanswerable    0.640  vs  0.080   <- separated
support score, correct vs incorrect answers  0.651  vs  0.625   <- NOT separated
```

**The score judges the excerpts, so it can see answerability and is nearly blind to
correctness.** Conformal risk control bounds a risk its nonconformity score can rank;
it cannot bound one the score cannot see. With 43% of *answerable* questions answered
wrongly by a 3 B model (and the judge counting INCOMPLETE as a failure) the selective
risk floor sits near that base rate regardless of where the threshold goes.

This is the honest ceiling of the design: **the gate controls "should I have answered
at all", not "is this answer right".** Those are different guarantees and conflating
them is exactly the mistake the repo exists to argue against.

### A self-inflicted problem: the score is quantised

| value | count |
|---|---|
| 0.00 | 32 |
| 0.01 | 1 |
| 0.50 | 34 |
| 1.00 | 33 |

**Four distinct values across 100 questions**, so only four usable thresholds. The
cause is my own prompt: it offered 0 / 50 / 100 as anchors and the model treated them
as the entire scale. Conformal calibration wants a continuous score to place a tight
threshold; this gives it a three-way switch, which is why the risk curves are step
functions with long plateaus. Fixes, in order of honesty: read token logprobs for the
score, drop the anchors and ask for a bare integer, or sample the judgement several
times and average. Worth stating plainly: the limitation is in the prompt, not in the
method.

### What would actually reach α on total error

1. **A stronger generator.** 43% wrong on answerable questions is the dominant term;
   no gate fixes a base rate that high.
2. **A nonconformity score that predicts correctness**: self-consistency across
   samples, or an entailment check between answer and cited excerpt. Both cost more
   calls, which is the trade to measure next.
3. Not: a bigger calibration set. That tightens the estimate, it does not move the
   floor.

## A correctness-aware score

M4 left one question: can a different nonconformity score bound *total* error, not
just answerability? Four candidates, scored on the same 100 questions, ranked by
**AUC over correct-vs-incorrect answers**: the threshold-free version of "can a gate
built on this work at all". 0.5 is a coin flip.

| score | what it inspects | correct | incorrect | **AUC** | distinct values |
|---|---|---|---|---|---|
| `support_v1` | question + excerpts | 0.651 | 0.625 | **0.511** | 4 |
| `support_v2` | same, no anchors in the prompt | 0.483 | 0.466 | **0.519** | 16 |
| `groundedness` | **the answer** vs its excerpts | 0.928 | 0.722 | **0.582** | 7 |
| `self_consistency` | agreement across 3 sampled answers | 0.566 | 0.443 | **0.680** | 70 |
| **`combined`** | √(groundedness × self-consistency) | n/a | n/a | **0.754** | n/a |

### My quantisation hypothesis was wrong

M4 blamed the flat risk curves on the support prompt's 0/50/100 anchors, and predicted
that removing them would help. `support_v2` removed them: distinct values went **4 →
16**, and AUC moved **0.511 → 0.519**. Essentially nothing.

So the anchors caused the *granularity* problem and not the *blindness*. The real
cause is structural: `support_score` never sees the answer, so it cannot rank whether
the answer is right, no matter how finely it is expressed. Worth recording as a wrong
call: the fix I proposed would not have worked, and only measuring it showed that.

### The two useful signals are complementary, not redundant

| | ranks correctness | ranks answerability |
|---|---|---|
| `groundedness` | weakly (0.582) | **superbly**, 0.840 answerable vs 0.080 not |
| `self_consistency` | **best single** (0.680) | **backwards**, 0.513 vs 0.612 |

Their geometric mean beats both (**AUC 0.754**), which is what "complementary" means
in practice: one asks whether the excerpts support the claim, the other whether the
model is stable in making it, and the failures are different.

**`self_consistency` ranking answerability backwards is a genuine artefact worth
naming.** Unanswerable questions score *higher* agreement, because the model
consistently refuses them, and three identical refusals are perfect agreement. Used
alone the signal is also non-monotone (see the plot): above a threshold of ~0.45 the
risk *rises*, as the high-agreement bucket fills with confident repeated errors.

![Which signal lets the gate bound total error](figures/gate_v2.png)

### The gate, recalibrated

| | best α met | held-out risk | answered |
|---|---|---|---|
| M4 (`support_v1`) | 0.50 | 0.375 | 64% |
| **M5 (`combined`)** | **0.40** | **0.312** | **64%** |

Ungated risk is 0.540. The combined gate cuts it to **0.312 while still answering 64%
of questions**: real progress, and still short of the α = 0.2 the project wants.

**The binding constraint is now unambiguous, and it is not the score.** 43% of
*answerable* questions are answered wrongly by a 3 B model. A gate can only decline to
answer; it cannot make a wrong answer right. Reaching α = 0.2 needs a better
generator, and no amount of calibration substitutes for one. That is the honest end of
this line of work, and it is worth more than a tuned number would have been.

## Does a bigger generator move the ceiling?

M5 concluded the binding constraint was the generator, not the score. Testing that
directly: same 100 questions, same corpus, same retrieval, **`qwen2.5:7b-instruct`**
in place of the 3 B model.

### Controlling the confound first

Swapping the model swaps the **judge** as well as the generator, so a lower error rate
could just be a softer grader. Re-running the 3 B answers through the **7 B judge**
isolates the generator (`scripts/compare_models.py --rejudge`):

| generator | judge | base error on answerable |
|---|---|---|
| 3 B | 3 B | 0.427 |
| **3 B** | **7 B** | **0.493** |
| 7 B | 7 B | 0.360 |

**The 7 B judge is stricter, not softer**, it fails 3 B answers 49.3% of the time
where the 3 B judge failed them 42.7%. That is the opposite of the bias I expected,
and it means the naive comparison *understated* what the bigger model bought:

- naive (each judged by itself): 0.427 → 0.360, a **16%** relative reduction
- like-for-like (both judged by 7 B): **0.493 → 0.360, a 27% relative reduction**

Worth stating plainly: without the control I would have published the smaller number
and been wrong about the size of the effect, in the direction that flatters the
smaller model.

### What it bought the gate

| | ungated risk | best α met | risk | coverage |
|---|---|---|---|---|
| 3 B (7 B-judged) | 0.620 | **none** | n/a | n/a |
| **7 B** | 0.540 | **0.40** | 0.378 | **74%** |

Coverage at the met threshold rises from 64% (M5's combined score on 3 B) to **74%**,
on the plain support score alone. The support score's own separation barely moved
(0.640/0.080 → 0.685/0.060), which is the expected result, it measures the excerpts,
and the excerpts did not change.

### The conclusion, unchanged in direction and sharper in size

A 7 B generator is a **real** improvement: a quarter of the errors gone, like-for-like
, and **still not enough for α = 0.2**. At a 36% base error rate on answerable
questions, a gate that can only decline to answer cannot get selective risk to 0.2
without refusing most of the corpus.

The honest reading is that this pipeline needs a generator in a different class, not
one size step up; and that the calibration machinery has been correct throughout,
it reported an unreachable target rather than quietly hitting it.

## 14 B: the gate finally meets α = 0.2, for a reason I did not predict

`qwen2.5:14b-instruct` (9 GB, 57%/43% CPU/GPU on a 6 GB card, ~28 s/question).
Every generator re-judged by the **same 14 B judge**, so the rows are comparable.

| generator | base error (14 B judge) | support: answerable / unanswerable |
|---|---|---|
| 3 B | 0.427 | 0.640 / 0.080 |
| **7 B** | **0.347** | 0.685 / 0.060 |
| 14 B | 0.373 | 0.687 / **0.020** |

| generator | ungated | best α met | risk | coverage |
|---|---|---|---|---|
| 3 B | 0.580 | none | n/a | n/a |
| 7 B | 0.540 | 0.40 | 0.378 | 74% |
| **14 B** | 0.540 | **0.20** | **0.133** | 30% |

### The 14 B answers slightly *worse* than the 7 B, and gates far better

Base error goes **up** from 0.347 to 0.373 between 7 B and 14 B, judged identically.
Yet 14 B is the first configuration to meet α = 0.2: a target three earlier
configurations could not reach at any threshold.

The reason is in the last column of the first table. What improved with scale was not
answer accuracy but **the model's judgement about the evidence**: mean support score on
questions the corpus cannot answer falls **0.080 → 0.060 → 0.020**. The 14 B is
markedly better at recognising when the excerpts do not contain the answer, and a
conformal gate converts exactly that into a guarantee.

**So scaling bought calibration, not correctness.** That is not what "the binding
constraint is the generator" predicted: the prediction was right about the *outcome*
and wrong about the *mechanism*.

The cost is coverage: 30% at α = 0.2, against 74% at α = 0.4. The gate reaches the
target by declining seven questions in ten. That is a real guarantee and a real price,
and which one matters depends on whether a wrong maintenance answer is worse than no
answer: for this domain, it is.

### Correction: "a bigger judge is stricter" was wrong

The 7 B write-up above concluded the 7 B judge was stricter than the 3 B one and
inferred a trend. The 14 B judge breaks it: on the *same* 3 B answers:

| judge | base error on 3 B answers |
|---|---|
| 3 B | 0.427 |
| 7 B | **0.493** |
| 14 B | 0.427 |

The 7 B judge is stricter than **both** its neighbours. There is no monotone
relationship between judge size and severity; I drew a line through two points and the
third disqualified it. The practical lesson stands and is in fact strengthened:
**hold the judge fixed when comparing generators**, but the reason is that judges vary
unpredictably, not that they get harsher with scale.

## The combined score on 14 B: best AUC, worse gate

The obvious next move was to put M5's combined score on the 14 B and buy back the
coverage the α = 0.2 gate gives up. It does not work, and *why* is the most useful
thing measured so far.

| score | AUC (correct vs incorrect) | best α met | risk | coverage |
|---|---|---|---|---|
| **`support_v1`** | 0.697 | **0.15** | **0.133** | 30% |
| `support_v2` | 0.570 | none | n/a | n/a |
| `groundedness` | 0.785 | none | n/a | n/a |
| `self_consistency` | 0.796 | none | n/a | n/a |
| **`combined`** | **0.845** | 0.30 | 0.300 | **60%** |

![Which signal bounds total error on 14 B](figures/gate_14b.png)

**The best-ranking score gives the worst guarantee at α = 0.2.** `combined` ranks
correctness better than everything else (AUC 0.845 vs 0.697) and cannot meet α = 0.2 at
any threshold, while `support_v1` can.

### AUC and conformal gating measure different things

AUC asks: *pick a correct and an incorrect answer at random: is the correct one scored
higher?* That is a statement about the **whole ordering**. A conformal gate never uses
the whole ordering. It draws one line and keeps what is above it, so all that matters is
whether some **top bucket is nearly pure**.

`support_v1` has a pure head. Its top bucket (score = 1.0) carries a risk of 0.133,
which is why it meets a tight α while ranking worse overall. `combined` spreads its
correct answers more evenly (better on average, no clean top) and bottoms out at 0.30.

`self_consistency` is the extreme case: **AUC 0.796 and a risk of 0.789 at its own top
threshold.** Its high-agreement bucket is where the model repeats itself, which includes
repeating a refusal and repeating a mistake: on 14 B its answerability separation is
**−0.392**, with unanswerable questions scoring a *perfect* 1.000 agreement. Mixing that
into `combined` is what pollutes the head.

**The practical rule:** choose a nonconformity score by the purity of its top bucket at
the α you need, not by AUC. Selecting on AUC would have picked the score that cannot
deliver the guarantee.

### Where it does help

At **α = 0.30**, `combined` answers **60%** of questions against `support_v1`'s 34%:
nearly double the coverage for the same guarantee. So the combined score is the right
choice for a looser target and the wrong one for a tight target, which is a more precise
statement than "better score".

Also worth noting: `support_v1`'s AUC rose from **0.511 on 3 B to 0.697 on 14 B**. The
score did not change: the model's ability to judge its own evidence did, which is the
same effect that let 14 B reach α = 0.2 in the first place.

## A score designed for head purity: it did not beat the baseline

The previous section said to select a score by the purity of its top bucket. So: build
one for that property and see. The design followed from the diagnosis:
**conjunction instead of averaging** (`min`, so an answer must clear *every* check
rather than letting one confident signal carry a doubtful one), plus free
deterministic **vetoes** (a refusal, or a citation pointing at an excerpt never
supplied, zeroes the score), plus a **refusal-aware consistency** that scores three
identical refusals 0 rather than 1.

Thirteen rules, evaluated on the objective that actually matters: *maximise coverage
subject to held-out risk ≤ α*: at α = 0.2, no new model calls.

| rule | threshold | test risk | test coverage | meets α | head purity (top 30%) |
|---|---|---|---|---|---|
| `support_v1` (baseline) | 1.00 | **0.133** | 30% | ✅ | **0.133** |
| `su + veto` | 1.00 | 0.133 | 30% | ✅ | 0.133 |
| `min(su, gr)` | 0.80 | 0.188 | **32%** | ✅ | 0.200 |
| `min(su, gr) + veto` | 0.80 | 0.188 | 32% | ✅ | 0.200 |
| `mean(su, gr)` | 0.65 | 0.281 | 64% | ❌ | 0.133 |
| `min(su, gr, sc)` | 0.41 | 0.217 | 46% | ❌ | 0.267 |
| `geo(gr, sc)` (M5's) | 1.00 | 0.556 | 18% | ❌ | 0.400 |

**The designed score did not beat the plain support score.** The best conjunction that
meets α buys **32% coverage against 30%**: two points at n = 50, which is noise. The
conjunction hypothesis is not vindicated.

### What it did establish

**The vetoes are redundant, not wrong.** `su + veto` is *identical* to `support_v1`,
and `min(su,gr) + veto` identical to `min(su,gr)`. On 14 B a refused answer already
scores support ≈ 0, and citation errors are already at zero (M2 measured 0/20 invalid
citations). The veto fires on nothing. It is free insurance against a failure mode this
model does not have: worth keeping for a weaker generator, worth knowing it is inert
here.

**Conjunction helps exactly where contamination lives.** `min(gr, sc)` and `geo(gr, sc)`
score identically badly (0.556, head 0.400), because `self_consistency`'s head is
polluted, and neither combining rule can clean a signal that is confidently wrong. You
cannot fix a bad component by how you combine it.

### The rule-selection trap, demonstrated

Choosing among thirteen rules on the same test half would be selection bias, so the
script picks on the **calibration** half and reports the test number. It picked
`mean(su, gr)` (58% calibration coverage, met α there) and on held-out data it
**misses**, risk 0.281 against α = 0.2.

That is the honest headline of this experiment: **the selection procedure itself
overfits at n = 50.** Had the table been scanned for the best test row, `mean(su,gr)`'s
64% coverage would have looked like a large win over the baseline's 30%. It is not a
win; it is a rule that fails its guarantee.

**Conclusion:** `support_v1` on the 14 B remains the gate: risk 0.133 at 30% coverage,
α = 0.2. Recovering coverage needs more calibration data or a better generator, not a
cleverer combination of the signals already in hand.

## More hand-written questions: and the real constraint, found at last

52 new questions written by reading 20 prose chunks across pages 65–694, none of which
the first batch touched (32 answerable, 20 hard negatives). Hand-written total: **72**;
whole set **152**.

Two things surfaced while writing them.

**The corpus is smaller than 1,752 chunks suggests.** The second "manual",
TM 9-6115-641-24P, is a **parts catalogue**: NSN tables, part numbers, figure indexes,
essentially no prose. Pages 720+ of TM 9-8000 are a glossary. The usable prose is one
manual, roughly pages 40–720. That is exactly why 11 of 60 auto-generated questions came
out as *"what is the part number for the felt gasket"*.

**TM 9-8000 is a *principles* manual**, so it explains how things work and never gives a
repair step, a torque figure, a capacity or a service interval. That makes for much
harder negatives than out-of-domain questions: *"how do I bleed the brakes"*, *"what
torque for the main bearing caps"*, *"what is the firing order"* all sound like they
belong in a vehicle manual and are genuinely absent from this one.

On the new set alone the gate looked transformed: **65% coverage at α = 0.2**, against
30% on the mixed set. That comparison is invalid, and checking it is where the real
answer came from.

### More calibration data does not buy coverage

One test set of 45 questions, drawn once and never touched. Several calibration sets.
Every gate evaluated on the same questions, so only the calibration data varies.

| calibration set | n | threshold | risk | coverage | meets α = 0.2 |
|---|---|---|---|---|---|
| 20 hand-written | 15 | 0.50 | 0.321 | 62% | ❌ |
| 52 hand-written | 38 | 0.50 | 0.321 | 62% | ❌ |
| **72 hand-written** | 53 | 0.50 | 0.321 | 62% | ❌ |
| 80 generated | 54 | 1.00 | 0.167 | 13% | ✅ |
| 25 random | 25 | 0.85 | 0.143 | 16% | ✅ |
| 75 random | 75 | 0.58 | 0.250 | 18% | ❌ |

Size does nothing. 25 questions meet α; 75 do not. The apparent 65% was the new set
being **easier** (ungated risk 0.423 vs 0.540), exactly as suspected, not better
calibration.

### The actual constraint: the score has a cliff

`support_v1` on 14 B takes **9 distinct values across 152 questions**, and 143 of them
sit on just three: 0.00 (45), 0.50 (56), 1.00 (42). Only nine questions land anywhere
else. So the entire achievable risk–coverage curve is:

| threshold | coverage | risk |
|---|---|---|
| ≤ 0.50 | **70%** | 0.274 |
| 0.58 | 33% | 0.180 |
| 0.85 | 30% | 0.089 |
| 1.00 | 28% | 0.095 |

**There is no operating point between 33% and 70% coverage.** The score is a three-way
switch, so the curve is a cliff, and α = 0.2 falls just under the 0.274 plateau: which
forces the gate off the cliff to the 30% side. At α = 0.3 the same gate would answer
**70%**.

That explains every result in this file that has been blamed on something else. More
calibration data cannot help: the threshold can only land in a handful of places. A
purer head cannot help: the head is already at risk 0.089. A better generator moved the
plateau but not the cliff.

**The open problem, stated precisely:** granularity *and* ranking, together.
`support_v1` has a pure head and no granularity; `support_v2` fixed the granularity (25
distinct values) and lost the ranking (AUC 0.570). Neither has both. Token logprobs over
the score are the obvious route: a continuous quantity by construction, and unlike a
prompt, one the model cannot round off.

## Token logprobs: the cliff is gone, and a new cost appears

The open problem was granularity *and* ranking together. Instead of asking the model to
write a number, ask it a single YES/NO question and read **P(YES)** from the token
distribution, normalised over the YES and NO spellings. Ollama serves logprobs on its
OpenAI-compatible route (`/v1`), not on `/api/chat`.

| | distinct values | AUC (correctness) | answerable | unanswerable |
|---|---|---|---|---|
| `support_v1` (written number) | 9 | 0.697 | 0.692 | 0.044 |
| **`logprob`** | **23** | **0.821** | 0.834 | **0.022** |

**Both goals met.** Granularity nearly tripled, ranking improved substantially, and the
answerable/unanswerable separation is the widest measured. The model cannot round off a
quantity it never writes.

### The cliff is gone, but the first attempt still failed

Fitted at exactly α = 0.2 the gate chose threshold **0.000** and scored test risk
**0.217**: a miss. Not because the score is bad: its risk curve is nearly flat at
0.15–0.19 across the whole range, so calibration at the boundary lands on the most
permissive threshold, which is precisely where the estimate is noisiest.

Fitting at a slightly stricter α buys a margin, but the margin must be *chosen*, and
choosing it by looking at the test set is the selection bias documented earlier in this
file. So: three splits. **cal** fits the threshold, **val** picks the margin, **test** is
seen once. Repeated over 300 random splits, because one 50-item split of 152 is mostly
noise.

| score | holds α | coverage | risk |
|---|---|---|---|
| `support_v1` | **90%** of splits | 29% (IQR 27–33%) | 0.104 |
| `logprob` | 76% of splits | **45%** (IQR 27–58%) | 0.148 |

**The logprob score buys about 16 points of coverage and holds the guarantee less
reliably.** That is the honest summary: not a clean win, a trade with a price.

Two things worth saying about the 90% and 76%. First, conformal risk control promises
E[risk] ≤ α *in expectation over the calibration draw*, not on every split: neither
number is evidence of a broken method. Second, the gap between them is real: the logprob
gate operates near the top of its range where few points fix the threshold, so it is
more fragile. Its coverage IQR (27–58%) says the same thing.

**What actually changed:** the quantised score made ~45% coverage *unreachable at any
threshold*. It is now reachable, at the cost of a less stable threshold. That is
progress of the useful kind: the constraint moved from "impossible" to "expensive".

## More calibration data does not help, and the reason retires a metric

The line above used to end "and the next lever is more calibration data, which for
the first time in this project would genuinely help." That was tested, and it is
wrong. `scripts/calibration_curve.py` holds the validation and test sets at a fixed
size and sweeps only `n_cal`, over 1,500 draws per point:

| n_cal | support_v1 holds α | coverage | logprob holds α | coverage |
|---:|---:|---:|---:|---:|
| 10 | 93% ± 1 | 28% | 70% ± 2 | 44% |
| 30 | 92% ± 1 | 29% | 70% ± 1 | 42% |
| 62 | 87% ± 1 | 30% | 73% ± 1 | 46% |

Six times the calibration data moves the logprob hold rate by about three points,
roughly two standard errors, and moves coverage not at all. On `support_v1` it is
mildly *negative*.

**Why: the hold rate is mostly test-set sampling noise.** If the gate's true
selective risk is R and a test draw answers k questions, the observed risk exceeds
α some of the time no matter how well the threshold was fitted. Predicted from that
alone against observed:

| score | n_cal | true R | answered | predicted miss | observed miss |
|---|---:|---:|---:|---:|---:|
| logprob | 10 | 0.164 | 22 | 29% | 30% |
| logprob | 30 | 0.147 | 21 | 19% | 30% |
| logprob | 62 | 0.147 | 23 | 24% | 27% |

So "holds α on 76% of splits" was largely a statement about a 50-question test set,
not about the score. Conformal risk control promises E[risk] ≤ α over the
calibration draw; mean risk is **0.147 ≤ 0.20**, so the guarantee is being met. The
per-split hold rate is a stricter quantity that no amount of calibration data can
buy, because the noise is on the other side of the split.

Two dead ends closed on the way, both worth not repeating:

- **The score is saturated**: median 0.9998, and p75 through p99 all read 1.0000,
  with the entire useful threshold range in the last decimal places. This is a true
  description and is *not* the problem: `calibrate_threshold` takes its candidates
  from the observed scores with no interpolation, so the pipeline is invariant to
  any monotone transform. A logit rescaling selects the identical question set. Only
  the ordering ever mattered.
- **The first version of the sweep let the test set absorb the remainder**
  (`n_test = N - n_cal - n_val`), so the test set shrank from 102 to 50 as
  calibration grew, and manufactured an "extra data hurts" trend out of a noisier
  estimate. Hold the evaluation fixed and discard the surplus.

The real lever is the generator: base error is **0.493** on this set, so the ceiling
on a gate that must not exceed 0.20 is set by how often the model is simply wrong.

## 54 more hand-written questions, written for measurement

Following the finding above, the third batch was written to tighten the
*measurement* rather than to improve the gate. **126 hand-written questions now**
(38 answerable and 16 unanswerable added), holding the unanswerable share at 33%.

Two deliberate choices, both reacting to how the second batch went wrong:

- **Aimed at the thin bands.** Page coverage before and after: 0-99 went 2 → 11,
  400-499 went 2 → 25, 500-599 went 3 → 9. The second batch had clustered on
  already-covered material and turned out simply easier than the first
  (ungated 0.423 against 0.540), which made the two sets non-comparable.
- **Reasoning over lookup.** `reasoning` is now the largest answerable type (42 of
  85). The generated majority skews to catalogue lookups (11 of 60 ask for a part
  or figure number), and questions of that shape flatter every metric.

`scripts/verify_golden.py` checks each answerable question against the page it
cites, by requiring the content words of the recorded answer to appear there.
Writing 38 by hand produced exactly one mislabelled page, caught that way. It is
a local check because the corpus index is gitignored; the structural half of the
contract runs in CI as `tests/test_golden_sets.py` (ids unique, no question asked
twice, unanswerable rows citing no source, answerable rows carrying an integer
page).

## Rescored on 206, and the "more data does not help" finding is half wrong

All 206 questions judged and scored by the same 14B. The prediction above was
that this would narrow the hold-rate estimate **without moving the gate**. The
first half held. The second did not, and the section before this one is corrected
by what follows rather than deleted.

Nested gate, 300 splits, same protocol:

| | 152 questions | 206 questions |
|---|---|---|
| support_v1 | 90% hold, 29% coverage (IQR 27-33) | 92% hold, 33% coverage (IQR 29-34) |
| logprob | 76% hold, 45% coverage (IQR 27-58) | 77% hold, **63%** coverage (IQR **60-69**) |

Hold rate barely moved, exactly as the test-noise argument predicts. But logprob
coverage rose 18 points and its interquartile range collapsed from 31 points wide
to 9. The gate did move.

The size sweep shows why. On 206, under the **identical** split geometry used for
the flat result above (val 40, test 50):

| n_cal | coverage | risk | holds α |
|---:|---:|---:|---:|
| 10 | 45% | 0.107 | 92% |
| 30 | 55% | 0.131 | 85% |
| 62 | 61% | 0.146 | 77% |

Sixteen points of coverage for the same six-fold increase that bought one point on
the 152-question pool. Geometry is not the explanation; it was held fixed
deliberately, because changing the pool and the split at once is how the first
version of this experiment went wrong.

**What was actually happening: with too little calibration data the gate is
over-conservative.** At n_cal = 10 it runs at risk 0.107 against a budget of 0.20
and under-answers to stay safe. More calibration data lets it spend the budget it
is entitled to, so coverage rises and risk climbs toward α from below. That is
correct behaviour, and the old 152-question pool was simply too saturated to show
it.

So the honest statement is narrower than the heading above: **more calibration
data did not help *that* pool.** It helps this one. The part that survives intact
is the mechanism for the hold rate, which is still dominated by test-set sampling
noise and still cannot be bought with calibration data.

Base error fell 0.493 → 0.451 because the new batch is easier than the mix, so
some of the coverage gain is the questions and not the data. Per batch:
v1 hand 0.300, v2 hand 0.423, **v3 hand 0.333**, generated 0.588. The v3 batch sits
between the two earlier hand-written sets, so it did not repeat the v2 skew.

**One incidental correction.** This file claimed the generated questions "skew
toward catalogue lookups, which are easier than real maintenance questions". They
are the *hardest* subset by a wide margin: error 0.588 against 0.300-0.423 for the
hand-written batches. A parts catalogue has no prose for the retriever to work
with, so a part-number lookup is hard for a RAG system even though it is trivial
for a human with the book open.

## Still to come
- **A harder hand-written batch.** Every hand-written set lands at 0.30-0.42 error
  while the generated set sits at 0.588. Questions spanning two sections, or
  requiring a figure, would test the gate where it is weakest.
- **M4**: reranker before/after: recall@5 without vs with the trained cross-encoder.
- **M4**: abstention: held-out selective risk vs α, answer rate, Mondrian breakdown.
- **M5**: answer correctness (judge + exact-match subset), cost and latency by provider.

Planned tables (see README roadmap):

- **M1**: retrieval recall@5 on the manually verified golden set (not the seeds).
- **M4**: reranker before/after: recall@5 without vs with the trained
  cross-encoder, same split, same seed.
- **M4**: abstention: held-out selective risk vs α, answer rate, per-group
  (Mondrian) breakdown, and the risk plot.
- **M5**: end-to-end answer correctness (LLM-judge + exact-match subset), cost
  and latency per question by provider.

Rule carried over from turbofan-rul: the headline is whatever the data says,
including when the boring baseline wins.
