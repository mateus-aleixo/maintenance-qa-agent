# Fine-tuning against prompting

The served generator is a prompted model: the system prompt tells it to answer from the
excerpts, cite them, and reply INSUFFICIENT EVIDENCE when they do not contain the answer.
This experiment trains that behaviour into Qwen2.5-3B-Instruct with QLoRA and measures it
against the same model prompted, and against the 7B and 14B prompted, all behind the
conformal gate the repository already uses.

Results are in [`docs/finetune.md`](../docs/finetune.md), and the raw numbers in
`runs/finetune/`.

## What is compared

| generator | weights | runtime |
|---|---|---|
| 3B prompted | Qwen2.5-3B-Instruct, 4-bit NF4 | transformers |
| 3B fine-tuned | the same, plus a LoRA adapter (r 16, every linear layer) trained with 40% declines | transformers |
| 3B fine-tuned, 20% and no declines | the same, with adapters trained on fewer declines | transformers |
| 3B prompted | qwen2.5:3b-instruct, Q4_K_M | Ollama |
| 7B prompted | qwen2.5:7b-instruct, Q4_K_M | Ollama |
| 14B prompted | qwen2.5:14b-instruct, Q4_K_M (answers already in `runs/`) | Ollama |

Every generator answers the same 206 test questions from the same five excerpts, with the
served system prompt, greedily. The 14B judges every answer against its reference, as it
did for every comparison in [`docs/results.md`](../docs/results.md). The 3B rows in
transformers differ only by the adapter. The Ollama 3B row shows what the change of
runtime and quantisation does on its own. The 20% and no-declines runs were added after
the 40% run had been evaluated.

## Training data

Questions come from the two manuals, written by the 7B from chunks on pages that no test
question was written from (a margin of one page either side of every gold page). Any
question within cosine 0.85 of a test question under the serving encoder is dropped as
well. Each example is the served prompt for that question, with the excerpts `retrieve()`
actually returns, and its target is either the answer citing its excerpt or a decline:

- **answer**: retrieval brought back the chunk the question was written from.
- **removed**: the same question, with that page and its neighbours taken out of the
  results, so the excerpts are on the topic but do not say it. Decline.
- **missed**: retrieval did not bring the source chunk back. Decline.
- **absent**: a value or step the chunk's topic leaves out, or a question from outside the
  corpus. Decline.

The 7B's P(YES) on the support question checks every label before it is kept (at least
0.5 for an answer, at most 0.2 for a decline). The plan named the 14B as the question
writer. When the data was written, the 14B failed to load ("unable to allocate CUDA_Host
buffer") with most of this machine's 13.7 GB of RAM in use, so the 7B wrote the questions.
The 14B later ran as the judge without trouble, memory-mapped, on a server with the prompt
cache off, with 20 of its 49 layers on the GPU.

The 40% run mixes all the answer examples with as many declines as make 40% of the
split. `python -m finetune.data split --decline 0.2 --tag d20` writes a variant to
`data/finetune/d20/` with the same answer examples and the first part of the same
shuffled declines, and `--tag` on the later stages points them at it.

## Running it

```bash
uv pip install -e ".[finetune]"
# a CUDA build of torch (pick the wheel matching your driver):
uv pip install --reinstall-package torch torch --index-url https://download.pytorch.org/whl/cu126

# Ollama's runner caches every prompt's KV state in RAM, up to 8 GiB, unless told not to.
# A separate server for the experiment, with the cache off and the same model files:
OLLAMA_HOST=127.0.0.1:11435 LLAMA_ARG_CACHE_RAM=0 ollama serve &
export CRAG_OLLAMA_URL=http://127.0.0.1:11435

python -m finetune.data all                     # Ollama with qwen2.5:7b-instruct
python -m finetune.train                        # QLoRA, 2.8 to 4.2 GiB of GPU memory
python -m finetune.evaluate retrieve
python -m finetune.evaluate answer hf-base      # and hf-tuned, ollama:qwen2.5:3b-instruct, ...
python -m finetune.evaluate judge hf-base       # Ollama with qwen2.5:14b-instruct
python -m finetune.report

# a variant with 20% declines
python -m finetune.data split --decline 0.2 --tag d20
python -m finetune.train --tag d20
python -m finetune.evaluate answer hf-tuned-d20
python -m finetune.evaluate judge hf-tuned-d20

# or every stage above in order, both variants included, as one unattended run
python -m finetune.run
```

Every stage resumes from what is already on disk. Torch is capped at the memory the card
has free when a stage starts: on Windows the NVIDIA driver otherwise backs an
over-allocation with shared system RAM, which hung the terminal once during development.
