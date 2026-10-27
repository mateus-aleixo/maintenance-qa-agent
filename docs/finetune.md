# Fine-tuning against prompting

The served generator is prompted. Its system prompt tells it to answer from the five
retrieved excerpts, cite them, and reply INSUFFICIENT EVIDENCE when they do not contain
the answer. This page measures whether QLoRA fine-tuning teaches Qwen2.5-3B-Instruct that
behaviour better than the prompt does. The tuned model is compared with the same weights
prompted, and with the 7B and 14B prompted, on the 206 test questions behind
[the results](results.md), with one fixed judge and the same conformal gate.

Fine-tuning lowered the share of correct answers. The planned run, trained with 40%
declines, answered 11 points fewer questions correctly than the prompted 3B, mostly
because it declined a quarter of the answerable ones. Two runs with fewer declines, added
after that result, declined less but did not answer more questions correctly. No 3B row
came near the prompted 7B behind the gate.

Code and instructions are in [`finetune/`](../finetune/README.md), and the raw numbers
in `runs/finetune/`.

## Setup

- **Model.** Qwen2.5-3B-Instruct, the weights Ollama's `qwen2.5:3b-instruct` is quantised
  from, loaded in 4-bit NF4 with double quantisation and bf16 compute. LoRA adapters of
  rank 16 (alpha 32, dropout 0.05) sit on every linear layer of every block, 29.9 million
  trainable parameters in all.
- **Training.** The loss covers the reply tokens only, up to and including the token that
  stops generation, so the model is never trained to reproduce the excerpts. AdamW at
  2e-4 with a 3% warm-up and cosine decay to a tenth of that, batch 1 with 8-step
  gradient accumulation, two epochs, gradient checkpointing. Validation loss is measured
  every 25 optimiser steps, and the adapter with the lowest one is kept.
- **Training data.** Written by the 7B (`qwen2.5:7b-instruct`) from the same two manuals,
  on pages no test question was written from: every test question's gold page and the
  page on either side are excluded. Details in the next section.
- **Test questions.** The 206 questions of the expanded pool: 145 answerable (85
  hand-written, 60 generated) and 61 unanswerable (25 from outside the corpus, 36
  in-domain details the manuals do not give). Every generator run here sees the same five
  excerpts per question, retrieved once by the served `retrieve()` and cached, with the
  served system prompt, and replies greedily, up to 256 tokens. Prompts are about 1,700 tokens
  at the median and 4,971 at the longest.
- **Judge.** The 14B (`qwen2.5:14b-instruct`, Q4_K_M) judges every answer against its
  reference, as in every generator comparison in the results. An answer judged
  incomplete counts as a mistake.
- **Gate.** The trial from `scripts/nested_gate.py`, imported unchanged: 300 random
  three-way splits, where the calibration third fits the threshold, the validation third
  picks the margin, and the test third is looked at once. Every generator gets the same
  sequence of splits, so the comparison is paired. The main score is the 14B's P(YES)
  that the excerpts support an answer (`runs/logprob_scores.json`). It is the same number
  for every generator, so only the answers differ. A reply that declines scores 0, so the
  gate never counts a refusal as an answer.
- **Hardware.** One laptop with an RTX 3060 Laptop GPU (6 GB) and 13.7 GB of RAM. The
  14B judge ran with 20 of its 49 layers on the GPU, at about 1.2 seconds a verdict.

## Training data

The 7B wrote one question and its answer from each of 1,006 prose chunks, a question on
the same topic that the chunk does not answer for 400 of them, and 200 questions from
outside the corpus on 20 topics. 119 questions that copied an eight-word span from their
chunk were dropped. Every other question went through the served retrieval, and each
example was labelled by what its five excerpts contain:

| kind | example | target | labelled | dropped by the check | planned run |
|---|---|---|---:|---:|---:|
| answer | a question whose source chunk is among the five excerpts | the 7B's answer, citing that excerpt | 799 | 125 | 600 |
| removed | the same question, with its source page and the pages either side taken out of the results | decline | 799 | 169 | 221 |
| missed | a question whose source chunk retrieval did not return | decline | 88 | 45 | 14 |
| absent | a near-miss question on the chunk's topic, or one from outside the corpus | decline | 600 | 50 | 165 |

The check is the 7B's own P(YES) on the support question: an answer is kept at 0.5 or
more, a decline at 0.2 or less, and anything in between is dropped. Six questions within
cosine 0.85 of a test question under the serving encoder were dropped as well. The
closest one kept is at 0.836. Validation holds out a tenth of the source pages, so no
page is on both sides.

The planned run takes all 600 answer examples and 400 of the declines, drawn at random,
so 40% of its 1,000 training examples are declines. Its validation set, mixed the same
way, has 118. The answer targets are one or two sentences, 19 tokens at the median.
Training examples are about 1,700 tokens at the median and 3,277 at the longest.

## Results

| generator | correct | declined, answerable | declined, unanswerable | risk at 50% coverage | gate holds α = 0.2 | coverage |
|---|---:|---:|---:|---:|---:|---:|
| 3B prompted (NF4, transformers) | 71.0% | 7.6% | 93.4% | 0.165 | 63.6% | 37.3% |
| 3B fine-tuned, 40% declines (planned) | 60.0% | 26.2% | 96.7% | 0.184 | 64.8% | 34.1% |
| 3B fine-tuned, 20% declines | 62.1% | 18.6% | 96.7% | 0.194 | 49.0% | 27.8% |
| 3B fine-tuned, no declines | 69.0% | 0.0% | 0.0% | 0.214 | 54.1% | 24.2% |
| 3B prompted (Q4_K_M, Ollama) | 69.7% | 6.2% | 95.1% | 0.184 | 58.5% | 26.9% |
| 7B prompted (Q4_K_M, Ollama) | 76.6% | 11.7% | 96.7% | **0.068** | 94.0% | 59.5% |
| 14B prompted (Q4_K_M, Ollama) | **77.9%** | 11.0% | 98.4% | 0.087 | **100%** | **59.8%** |

Correct is the share of the 145 answerable questions the judge marks correct. Risk at 50%
coverage is the mistake rate among the half of the 206 questions that the fixed score
ranks highest. The last two columns are for the gate at α = 0.2 with the fixed score: the
share of splits whose test third holds α, and the mean share of test questions answered.
Splits where validation finds no threshold that holds α are left out of both, as in
`nested_gate.py`.

### 1. The planned run declines a quarter of the answerable questions

The run trained with 40% declines answers 60.0% of the answerable questions correctly,
against 71.0% for the same weights prompted. That is 11.0 points fewer, with a 95%
interval of 2.9 to 19.3 points (paired bootstrap over the 206 questions, 2,000
resamples). It declines 38 of the 145 answerable questions, where the prompted model
declines 11. For 29 of those 38 the gold page was among the five excerpts, and the
prompted model answered 18 of the 29 correctly. On the unanswerable questions it declines
slightly more often, 96.7% against 93.4%.

When it does answer, it is right more often: 87 of its 107 answers to answerable
questions, against 103 of 134. At a fixed coverage that precision does not carry over.
After its declines it has 109 replies left to rank, so its top 103, half the pool,
include nearly all of them. The prompted model picks its top 103 from 138. The tuned
model's risk at 50% coverage is 0.184 against 0.165, a difference of +0.019 (interval
-0.068 to 0.126). At α = 0.2 the two hold the target on about the same share of splits, 64.8% and
63.6%, and the tuned model answers 34.1% of the questions against 37.3%.

### 2. Fewer declines in training gave fewer declines and no more correct answers

The two other runs were added after the planned run's result, to test whether its decline
share caused the over-declining. Both keep the planned run's 600 answer examples. The 20%
run adds the first 150 of the same shuffled declines, the no-declines run adds none, and
their validation sets are cut the same way (89 and 71 examples).

| training mix | correct | declined, answerable | declined, unanswerable | correct, tuned minus prompted 3B |
|---|---:|---:|---:|---|
| 40% declines (planned) | 60.0% | 26.2% | 96.7% | -11.0 points (-19.3 to -2.9) |
| 20% declines | 62.1% | 18.6% | 96.7% | -9.0 points (-16.9 to -1.4) |
| no declines | 69.0% | 0.0% | 0.0% | -2.1 points (-9.3 to 5.3) |
| prompted 3B | 71.0% | 7.6% | 93.4% | |

Halving the decline share cut the declined answerable questions from 38 to 27, and 26 of
the 27 are questions the planned run also declined. The 20% run still answers 9.0 points
fewer correctly than the prompted model. With no declines in training the model never
declines, not even on the 61 unanswerable questions, although its system prompt still
asks it to. It answers all 145 answerable questions and gets 100 right, against 103 for
the prompted model, which declined 11. The two disagree on 29 questions: the tuned model
alone is right on 13 and the prompted model alone on 16.

### 3. The tuned models answer at the length of the training answers

The answer targets are 19 tokens at the median. On the answerable questions they answer,
the three tuned runs write 21 to 23 tokens on average, against 59 for the prompted model.
With no declines in training, twice as many answers are judged incomplete: 16 of 145,
against 8 of the prompted model's 134. The 40% and 20% runs have 6 of 107 and 7 of 118.

### 4. No 3B row comes near the prompted 7B

The 7B, prompted, answers 76.6% of the answerable questions correctly, 16.6 points more
than the planned run (interval 9.4 to 24.0). Behind the gate at α = 0.2 it holds the
target on 94.0% of splits and answers 59.5% of the questions. The 3B rows hold it on 49.0
to 64.8% of splits and answer 24.2 to 37.3%. At 50% coverage the 7B's risk is 0.068, and
every 3B row's is between 0.165 and 0.214.

## Training

| run | train / validation examples | optimiser steps | adapter kept at step | validation loss, start / kept | minutes | peak GPU memory |
|---|---|---:|---:|---|---:|---:|
| 40% declines | 1,000 / 118 | 250 | 250 | 1.303 / 0.252 | 100.7 | 3.77 GiB |
| 20% declines | 750 / 89 | 187 | 100 | 1.500 / 0.305 | 73.1 | 3.77 GiB |
| no declines | 600 / 71 | 150 | 75 | 1.693 / 0.397 | 56.6 | 3.59 GiB |

The three validation sets differ, so their losses do not compare directly. On the 71
answer examples all three share, the kept adapters reach 0.412, 0.375 and 0.397, from
1.693 for the base model. Every run trained at about 680 tokens a second, and no example
ran out of memory. The 20% and no-declines adapters were kept about one epoch in, while
the planned run's lowest validation loss came at its last step.

### Attention memory on this machine

The first attempt at the planned run was stopped at step 85 of 250. Eleven examples
between 2,347 and 3,277 tokens had run out of memory, and each took its accumulation
window with it. This Windows build of PyTorch has no flash attention kernel. Qwen2.5-3B
has 16 query heads and 2 key/value heads, and transformers passes the grouped heads to
PyTorch's scaled dot-product attention, whose memory-efficient kernel does not accept
them. PyTorch then used its math kernel, which holds the whole attention matrix, so
memory grew with the square of the length.

`linear_memory_attention()` in `finetune/common.py` expands the key/value heads before the
call, as transformers already does whenever it passes a mask, and switches the math kernel
off, so any other fallback fails with an error. Measured once on this machine, for one
training step:

| example length | grouped heads (default) | heads expanded |
|---|---|---|
| 1,700 tokens | 3.34 s, 3.09 GiB | 2.47 s, 2.78 GiB |
| 2,700 tokens | 6.07 s, 4.34 GiB | 3.99 s, 3.20 GiB |
| 5,000 tokens | out of memory | 8.22 s, 4.17 GiB |

Answering had the same limit: with grouped heads a 5,000-token prompt ran out of memory
before the first token, and the longest test prompt is 4,971 tokens. With the heads
expanded the prefill takes 2.07 s and 2.78 GiB.

## What this does not show

- One base model, one seed, and one setting each for the learning rate, the rank and the
  number of epochs, on 1,000 training examples at most. None of these was tuned.
- The 20% and no-declines runs were designed after the planned run's results were known.
- Every answer target was written by the 7B, which itself answers 76.6% of the answerable
  test questions correctly.
- The judge is a 14B model, and its verdicts on these answers were not checked by hand. A
  short answer may be judged incomplete where a person would accept it.
- The training questions come from the same two manuals as the test questions. Pages near
  a test question's gold page are excluded and near-paraphrases are dropped, but the
  topics overlap.
- Ollama cannot load a LoRA adapter for Qwen2 without llama.cpp's conversion tools, so
  the tuned models only ran in transformers. The two prompted 3B rows, NF4 in
  transformers and Q4_K_M in Ollama, differ by 1.3 points in correct answers.
- Two test prompts are longer than the 4,096-token context of the Ollama rows, and Ollama
  cut both to 2,050 tokens. The transformers rows read them whole.
- The 14B row uses the answers already in `runs/` from the earlier experiments. They were
  not generated again for this page.

## All the numbers

**By question type.** Correct for the answerable questions, declined for the
unanswerable ones.

| generator | generated (60) | hand-written (85) | outside the corpus (25) | in-domain, absent (36) |
|---|---:|---:|---:|---:|
| 3B prompted (NF4) | 56.7% | 81.2% | 100% | 88.9% |
| 3B fine-tuned, 40% declines | 50.0% | 67.1% | 100% | 94.4% |
| 3B fine-tuned, 20% declines | 48.3% | 71.8% | 100% | 94.4% |
| 3B fine-tuned, no declines | 58.3% | 76.5% | 0% | 0% |
| 3B prompted (Ollama) | 50.0% | 83.5% | 100% | 91.7% |
| 7B prompted | 56.7% | 90.6% | 100% | 94.4% |
| 14B prompted | 55.0% | 94.1% | 100% | 97.2% |

**Gate at α = 0.2 with each score.** Share of splits that hold α, mean coverage, and risk
at 50% coverage. Each score has the same refusal veto. The generator's own P(YES) is its
answer to the same support question the 14B answers for the fixed score, and P(not
declining) is one minus the probability its first reply token starts INSUFFICIENT
EVIDENCE. For the 14B, its own P(YES) is the fixed score.

| generator | fixed 14B P(YES) | own P(YES) | own P(not declining) |
|---|---|---|---|
| 3B prompted (NF4) | 63.6%, 37.3%, 0.165 | 50.8%, 32.8%, 0.233 | 27.2%, 37.5%, 0.184 |
| 3B fine-tuned, 40% declines | 64.8%, 34.1%, 0.184 | 60.7%, 34.9%, 0.204 | 49.7%, 36.0%, 0.194 |
| 3B fine-tuned, 20% declines | 49.0%, 27.8%, 0.194 | 67.1%, 30.2%, 0.243 | 23.2%, 20.3%, 0.233 |
| 3B fine-tuned, no declines | 54.1%, 24.2%, 0.214 | 74.6%, 11.1%, 0.272 | no threshold, 0.485 |
| 3B prompted (Ollama) | 58.5%, 26.9%, 0.184 | 0.0%, 15.3%, 0.272 | 37.3%, 32.4%, 0.204 |
| 7B prompted | 94.0%, 59.5%, 0.068 | 90.3%, 56.8%, 0.117 | 83.8%, 59.3%, 0.126 |
| 14B prompted | 100%, 59.8%, 0.087 | 100%, 59.8%, 0.087 | |

With the fixed score, validation found a threshold on 247, 233, 210, 181, 234, 299 and
299 of the 300 splits, in the order of the rows. The gate results at α = 0.1 and 0.3 are
in `runs/finetune/report.json`.

**Speed and memory while answering.** Median seconds per question, mean reply tokens over
all 206, and GPU memory. The adapters run unmerged on top of the NF4 weights.

| generator | median seconds | mean reply tokens | GPU memory |
|---|---:|---:|---|
| 3B prompted (NF4) | 2.75 | 44.6 | 2.49 GiB peak |
| 3B fine-tuned, 40% declines | 2.14 | 15.3 | 2.94 GiB peak |
| 3B fine-tuned, 20% declines | 2.40 | 15.7 | 2.94 GiB peak |
| 3B fine-tuned, no declines | 3.33 | 20.6 | 2.94 GiB peak |
| 3B prompted (Ollama) | 1.16 | 61.4 | 2.01 GiB, whole model |
| 7B prompted (Ollama) | 3.30 | 71.5 | 3.90 of 4.77 GiB |
