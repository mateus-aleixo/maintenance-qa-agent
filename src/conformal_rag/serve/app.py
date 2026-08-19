"""FastAPI service over the retrieval index and the calibrated gate.

Torch-free: the query encoder is the ONNX export from `scripts/export_embedder.py`
run on onnxruntime, the corpus is SQLite, and the gate is arithmetic. Same container
arrangement as conformal-rul and conformal-seg, and the module exposes `handler` for
Lambda via Mangum.

**What is deliberately not hosted: the generator.** conformal-rul and conformal-seg
ship their own networks, a few MB of ONNX apiece. This system's generator is a 14B
model, which does not fit in a Lambda and is not a thing a free demo endpoint should
be paying for per request. So the deployed surface is everything that is genuinely
this repo's contribution and genuinely serverless:

  /retrieve    hybrid BM25 + vector search over the real corpus
  /gate        the calibrated conformal decision, given a score
  /gates       the thresholds and the measured risk they bought

and /ask is wired but returns 503 until an operator points LLM_BASE_URL at an
OpenAI-compatible endpoint. That is the same provider-agnostic client the CLI uses,
so it is a configuration difference rather than a different code path.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from conformal_rag import __version__
from conformal_rag.conformal import ConformalGate
from conformal_rag.registry import GATE_MANIFEST

from .schemas import (
    AskResponse,
    GateDecisionResponse,
    GateInfo,
    Hit,
    RetrieveResponse,
)

MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "models"))
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")

app = FastAPI(
    title="conformal-rag",
    version=__version__,
    description="Retrieval and a distribution-free abstention gate over industrial "
    "maintenance manuals. Retrieval and the gate are served; generation is not "
    "hosted (see /ask).",
)


class Bundle:
    def __init__(self, root: Path):
        manifest_path = root / GATE_MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(f"no gate manifest under {root}")

        from conformal_rag.embed import OnnxEmbedder
        from conformal_rag.store import Store

        self.manifest = json.loads(manifest_path.read_text())
        self.gate = ConformalGate.from_dict(self.manifest["gate"])
        # read_only: the image ships a built index and Lambda mounts it on a
        # read-only filesystem, where WAL setup alone fails to open the file.
        self.store = Store(str(root / "index.db"), read_only=True)
        self.embedder = OnnxEmbedder(root / "embedder")


@lru_cache(maxsize=1)
def bundle() -> Bundle:
    try:
        return Bundle(MODEL_ROOT)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/gates", response_model=GateInfo)
def gates() -> GateInfo:
    m = bundle().manifest
    held = m["held_out"]
    return GateInfo(
        alpha=m["gate"]["alpha"],
        threshold=m["gate"]["global_threshold"],
        score=m["score"],
        generator=m["generator"],
        n_calibration=m["n_calibration"],
        ungated_risk=m["ungated_risk"],
        held_out_risk=held["risk"],
        coverage=held["answer_rate"],
        n_test=held["n_total"],
        source_artifact=m["source_artifact"],
    )


@app.get("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(
    q: str = Query(..., min_length=2, max_length=512, description="The question."),
    k: int = Query(5, ge=1, le=20, description="How many chunks to return."),
) -> RetrieveResponse:
    from conformal_rag.retrieve import retrieve as run_retrieval

    b = bundle()
    hits = run_retrieval(b.store, b.embedder, q, k_final=k)
    return RetrieveResponse(
        query=q,
        hits=[
            Hit(chunk_id=h.chunk_id, doc=h.doc, page=h.page, text=h.text,
                score=round(h.score, 6), sources=list(h.sources))
            for h in hits
        ],
    )


@app.get("/gate", response_model=GateDecisionResponse)
def gate_decide(
    score: float = Query(..., ge=0.0, le=1.0,
                         description="Confidence score for a candidate answer."),
) -> GateDecisionResponse:
    """Apply the calibrated threshold to a score.

    Split out from generation on purpose: the gate is the part with the guarantee,
    and it is a total function of the score. Anyone can post their own generator's
    score and get the same decision this system would make.
    """
    b = bundle()
    d = b.gate.decide(score)
    held = b.manifest["held_out"]
    return GateDecisionResponse(
        answer=d.answer,
        decision="answer" if d.answer else "abstain",
        score=d.score,
        threshold=d.threshold,
        alpha=b.manifest["gate"]["alpha"],
        held_out_risk=held["risk"],
        coverage=held["answer_rate"],
    )


@app.post("/ask", response_model=AskResponse)
def ask(q: str = Query(..., min_length=2, max_length=512)) -> AskResponse:
    if not LLM_BASE_URL:
        raise HTTPException(
            status_code=503,
            detail=(
                "generation is not hosted. The gate's calibrated threshold was fitted "
                "against qwen2.5:14b-instruct, which does not fit in a Lambda and is "
                "not something a free demo should pay for per request. Set LLM_BASE_URL "
                "(and LLM_API_KEY) to any OpenAI-compatible endpoint to enable this "
                "route, or run the CLI locally against Ollama. /retrieve and /gate are "
                "the parts that are served."
            ),
        )
    raise HTTPException(
        status_code=501,
        detail="LLM_BASE_URL is set but hosted generation is not implemented in this "
               "deployment; use the CLI.",
    )


try:  # Lambda entrypoint; absent locally unless the serve extra is installed
    from mangum import Mangum

    handler = Mangum(app)
except ImportError:  # pragma: no cover
    handler = None
