"""API contract.

Every gate response carries the measured numbers next to the decision: the risk the
threshold actually bought on held-out data, and the coverage it cost. A guarantee
quoted without its coverage is half a result, which is the same discipline the
sibling repos apply to alpha and the false-alarm rate.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Hit(BaseModel):
    chunk_id: int
    doc: str
    page: int
    text: str
    score: float = Field(..., description="Fused reciprocal-rank score; bigger is better.")
    sources: list[str] = Field(..., description="Which retrievers found it: bm25, vec.")


class RetrieveResponse(BaseModel):
    query: str
    hits: list[Hit]


class GateInfo(BaseModel):
    alpha: float = Field(..., description="Bound on the wrong-answer rate among answered questions.")
    threshold: float
    score: str = Field(..., description="Which nonconformity score the threshold applies to.")
    generator: str = Field(..., description="The gate is only valid for the generator it was calibrated against.")
    n_calibration: int
    ungated_risk: float = Field(..., description="Error rate with no gate at all.")
    held_out_risk: float = Field(..., description="Error rate among answered questions, on held-out data.")
    coverage: float = Field(..., description="Fraction of questions answered rather than declined.")
    n_test: int
    source_artifact: str


class GateDecisionResponse(BaseModel):
    answer: bool
    decision: str = Field(..., description="'answer' or 'abstain'.")
    score: float
    threshold: float
    alpha: float
    held_out_risk: float
    coverage: float


class AskResponse(BaseModel):
    """Reserved for a deployment that hosts a generator; see /ask."""

    query: str
    decision: str
    answer: str | None = None
    citations: list[Hit] = []
