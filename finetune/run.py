"""Every stage of the experiment, in order, skipping what is already on disk.

    python -m finetune.run

Meant for the long unattended run, like scale.run: each stage is a fresh process, so a
failure in one leaves the finished ones intact, and running the same command again
resumes from the first unfinished stage. Before a stage that loads torch onto the GPU,
every model Ollama holds is unloaded, because the two cannot share a 6 GB card.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import httpx

from .common import DATA, JUDGE, OLLAMA, RUNS, run_dir

GENERATORS = ["hf-base", "hf-tuned", "ollama:qwen2.5:3b-instruct", "ollama:qwen2.5:7b-instruct"]
# Variants of the main run (40% declines): the same answer examples with fewer declines.
# Added after the main run's results, which declined 26% of the answerable questions.
VARIANTS = {"d20": 0.2, "d0": 0.0}


def _rows(name: str) -> list[dict] | None:
    path = RUNS / f"answers_{name.replace(':', '-')}.json"
    return json.loads(path.read_text()) if path.exists() else None


def answered(name: str) -> bool:
    rows = _rows(name)
    return rows is not None and len(rows) == 206


def judged(name: str) -> bool:
    rows = _rows(name)
    return rows is not None and len(rows) == 206 and all("loss" in r for r in rows)


def free_gpu(keep: str | None = None) -> None:
    """Unload whatever Ollama holds except `keep`; a torch stage needs the whole card.

    Both servers are cleared: the experiment's own and the default one on 11434, which
    anything else on the machine may have used in the meantime.
    """
    for url in dict.fromkeys([OLLAMA, "http://localhost:11434"]):
        try:
            loaded = httpx.get(f"{url}/api/ps", timeout=30).json().get("models", [])
        except httpx.HTTPError:
            continue
        for m in [m for m in loaded if not (url == OLLAMA and m["name"] == keep)]:
            body = {"model": m["name"], "keep_alive": 0}
            httpx.post(f"{url}/api/generate", json=body, timeout=120)
            print(f"  unloaded {m['name']} from {url}", flush=True)
    time.sleep(5)  # the driver returns the memory shortly after the runner exits


def stages() -> list[tuple[list[str], object, str]]:
    """(module and args, done-check, what to clear first) in dependency order.

    "gpu" unloads every Ollama model before a torch stage, and "judge" every model but
    the 14B, so the 7B and the 14B never sit on the card together.
    """
    out = [
        (["finetune.data", "all"], lambda: (DATA / "train.jsonl").exists(), ""),
        (["finetune.train"], lambda: (DATA / "adapter" / "adapter_config.json").exists(), "gpu"),
        (["finetune.evaluate", "retrieve"], lambda: (DATA / "test_hits.json").exists(), ""),
    ]
    for g in GENERATORS:
        clear = "gpu" if g.startswith("hf-") else ""
        out.append((["finetune.evaluate", "answer", g], lambda g=g: answered(g), clear))
    for g in GENERATORS:
        out.append((["finetune.evaluate", "judge", g], lambda g=g: judged(g), "judge"))
    for tag, share in VARIANTS.items():
        g = f"hf-tuned-{tag}"
        split = ["finetune.data", "split", "--decline", str(share), "--tag", tag]
        out += [
            (split, lambda tag=tag: (run_dir(tag) / "train.jsonl").exists(), ""),
            (
                ["finetune.train", "--tag", tag],
                lambda tag=tag: (run_dir(tag) / "adapter" / "adapter_config.json").exists(),
                "gpu",
            ),
            (["finetune.evaluate", "answer", g], lambda g=g: answered(g), "gpu"),
            (["finetune.evaluate", "judge", g], lambda g=g: judged(g), "judge"),
        ]
    out.append((["finetune.report"], lambda: False, ""))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--skip", nargs="*", default=[], help="stage words to leave for later, e.g. judge"
    )
    a = ap.parse_args(argv)
    for cmd, done, clear in stages():
        if any(s in cmd for s in a.skip):
            print(f"later {' '.join(cmd)}", flush=True)
            continue
        if done():
            print(f"skip  {' '.join(cmd)}", flush=True)
            continue
        if clear:
            free_gpu(keep=JUDGE if clear == "judge" else None)
        print(f"{time.strftime('%H:%M:%S')}  {' '.join(cmd)}", flush=True)
        t0 = time.perf_counter()
        rc = subprocess.call([sys.executable, "-m", *cmd])
        print(f"          exit {rc} after {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
