"""QLoRA on Qwen2.5-3B-Instruct: the served prompt in, the answer or the decline out.

    python -m finetune.train
    python -m finetune.train --tag d20     # a variant split written by finetune.data

The base is quantised to 4-bit NF4 with double quantisation and computes in bf16;
LoRA r=16 sits on every linear layer of every block, and the loss covers the reply
tokens only, so the model is never trained to reproduce the excerpts. Batch 1 with
8-step accumulation: with gradient checkpointing, one 1,700-token example takes
2.8 GiB on this 6 GB card and a 5,000-token one 4.2 GiB. Validation loss is measured
every `--eval-every` optimizer steps and the adapter with the lowest one is kept.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

from .common import BASE, RUNS, cap_gpu, load_base, read_jsonl, run_dir, tagged

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tok, row: dict, max_len: int) -> tuple[list[int], int] | None:
    """Token ids for one example and how many of them, at the end, are the reply.

    The reply ends with <|im_end|>, the token that stops generation, so the model
    learns where an answer finishes as well as what it says.
    """
    messages = [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["prompt"]},
    ]
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if not isinstance(prompt, list):  # newer tokenizers return a BatchEncoding
        prompt = prompt["input_ids"]
    reply = tok(row["target"], add_special_tokens=False)["input_ids"] + [
        tok.convert_tokens_to_ids("<|im_end|>")
    ]
    if len(prompt) + len(reply) > max_len:
        return None
    return list(prompt) + reply, len(reply)


def reply_loss(model, ids, n_reply: int):
    """Mean cross-entropy over the reply tokens, computing logits for those alone.

    Logits for a whole 1,700-token example over a 151,936-token vocabulary take 1 GiB
    in float32. `logits_to_keep` returns only the positions that predict the reply.
    """
    import torch
    import torch.nn.functional as F

    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=n_reply + 1)
    logits = out.logits[:, :-1, :].float()
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, -n_reply:].reshape(-1))


def lr_at(step: int, total: int, warmup: int, peak: float, floor: float = 0.1) -> float:
    """Linear warm-up, then cosine decay to `floor` x peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--eval-every", type=int, default=25, help="optimizer steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="first n training examples only")
    ap.add_argument("--tag", default="", help="train on data/finetune/<tag> instead")
    a = ap.parse_args(argv)

    import torch

    cap_gpu()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    data = {}
    for split in ("train", "val"):
        rows = read_jsonl(run_dir(a.tag) / f"{split}.jsonl")
        rows = rows[: a.limit if split == "train" else None]
        enc = [(r, encode(tok, r, a.max_len)) for r in rows]
        data[split] = [(r["kind"], e) for r, e in enc if e is not None]
        print(
            f"{split}: {len(data[split])} examples ({len(rows) - len(data[split])} over "
            f"{a.max_len} tokens dropped), longest {max(len(e[0]) for _, e in data[split])}"
        )

    model = load_base()
    model.config.use_cache = False
    # Train mode matters here: transformers only checkpoints in training, and without
    # it the 3B model needed 10 GiB for one example.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(
        r=a.rank,
        lora_alpha=2 * a.rank,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)

    def validate() -> dict:
        model.eval()
        sums: dict[str, list[float]] = {}
        with torch.no_grad():
            for kind, (ids, n) in data["val"]:
                x = torch.tensor([ids], device="cuda")
                loss = reply_loss(model, x, n).item()
                sums.setdefault("all", []).append(loss)
                sums.setdefault("answer" if kind == "answer" else "decline", []).append(loss)
        model.train()
        return {k: round(sum(v) / len(v), 4) for k, v in sums.items()}

    # One flat schedule over every epoch, cut to whole accumulation windows, so the
    # last optimizer step is a full one and the step count is exact.
    rng = random.Random(a.seed)
    schedule = []
    for _ in range(a.epochs):
        order = list(range(len(data["train"])))
        rng.shuffle(order)
        schedule += order
    total = len(schedule) // a.accum
    schedule = schedule[: total * a.accum]
    warmup = max(1, round(0.03 * total))
    log = {
        "base": BASE,
        "args": vars(a),
        "trainable_params": trainable,
        "n_train": len(data["train"]),
        "n_val": len(data["val"]),
        "optimizer_steps": total,
        "train": [],
        "val": [],
    }
    best = float("inf")
    adapter_dir = run_dir(a.tag) / "adapter"
    RUNS.mkdir(parents=True, exist_ok=True)

    model.train()
    torch.cuda.reset_peak_memory_stats()
    step, window, tokens, val_s = 0, [], 0, 0.0
    t0 = time.perf_counter()

    def checkpoint() -> None:
        nonlocal best, val_s
        tv = time.perf_counter()
        v = validate()
        val_s += time.perf_counter() - tv
        log["val"].append({"step": step, **v})
        saved = step > 0 and v["all"] < best
        if saved:
            best = v["all"]
            model.save_pretrained(adapter_dir)
            log["best_step"] = step
        print(f"step {step}  val {v}{'  saved' if saved else ''}", flush=True)
        log["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        log["minutes"] = round((time.perf_counter() - t0) / 60, 1)
        log["train_tokens_per_s"] = round(tokens / max(1e-9, time.perf_counter() - t0 - val_s))
        (RUNS / tagged("train.json", a.tag)).write_text(json.dumps(log, indent=2))

    checkpoint()
    skipped = []
    for micro, i in enumerate(schedule, start=1):
        _, (ids, n) = data["train"][i]
        x = torch.tensor([ids], device="cuda")
        try:
            loss = reply_loss(model, x, n)
            (loss / a.accum).backward()
            window.append(loss.item())
            tokens += len(ids)
        except torch.OutOfMemoryError:
            # One example too long for the card. Its backward may have written part of
            # a gradient, so the whole accumulation window is dropped with it.
            del x
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            skipped.append(len(ids))
            log["skipped_out_of_memory"] = skipped
            print(f"  out of memory at {len(ids)} tokens: window dropped", flush=True)
        if micro % a.accum:
            continue
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, warmup, a.lr)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 5 == 0:
            elapsed = time.perf_counter() - t0
            log["train"].append(
                {
                    "step": step,
                    "loss": round(sum(window) / max(1, len(window)), 4),
                    "lr": round(opt.param_groups[0]["lr"], 7),
                    "elapsed_s": round(elapsed),
                }
            )
            print(
                f"step {step}/{total}  epoch {micro / len(data['train']):.2f}"
                f"  loss {log['train'][-1]['loss']:.4f}  {elapsed / 60:.1f} min",
                flush=True,
            )
            window = []
        if step % a.eval_every == 0 or step == total:
            checkpoint()
    print(f"done: best val {best:.4f} at step {log.get('best_step')}, adapter -> {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
