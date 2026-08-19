"""Conformal risk control for selective answering.

Setting: each calibration item is a question the pipeline answered, with a
confidence score s ∈ [0, 1] and a binary loss (1 = the answer was wrong). At
deployment we answer only when s ≥ λ̂ and abstain below.

λ̂ is chosen by conformal risk control (Angelopoulos et al., 2022):

    λ̂ = inf { λ : (n/(n+1)) · R̂(λ) + B/(n+1) ≤ α }

with R̂(λ) the empirical mean loss among calibration items with s ≥ λ and B = 1
the loss bound. Guarantee, under exchangeability of calibration and test items:
E[loss | answered] ≤ α. Distribution-free, finite-sample.

The Mondrian variant fits one threshold per group (question type), falling back
to the global threshold when a group has fewer than `min_group` items — the same
small-group fallback used in conformal-rul for operating regimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class GateDecision:
    answer: bool
    score: float
    threshold: float
    group: str


def _risk_at(scores: np.ndarray, losses: np.ndarray, lam: float) -> tuple[float, int]:
    kept = scores >= lam
    n_kept = int(kept.sum())
    if n_kept == 0:
        return 0.0, 0
    return float(losses[kept].mean()), n_kept


def calibrate_threshold(
    scores: np.ndarray, losses: np.ndarray, alpha: float = 0.10
) -> float:
    """Smallest λ whose corrected selective risk is ≤ α.

    Candidate thresholds are the observed scores (plus 0 and 1): risk only
    changes there. If even the strictest threshold cannot meet α, return a
    threshold of 1.0 — the gate then abstains on everything, which is the only
    honest behaviour.
    """
    scores = np.asarray(scores, dtype=float)
    losses = np.asarray(losses, dtype=float)
    if scores.shape != losses.shape or scores.ndim != 1:
        raise ValueError("scores and losses must be 1-D and the same length")
    if not len(scores):
        return 1.0
    if losses.min() < 0 or losses.max() > 1:
        raise ValueError("losses must be in [0, 1]")

    candidates = np.unique(np.concatenate([scores, [0.0, 1.0]]))
    for lam in candidates:  # ascending: first feasible λ is the smallest
        risk, n_kept = _risk_at(scores, losses, lam)
        if n_kept == 0:
            continue
        corrected = (n_kept / (n_kept + 1)) * risk + 1.0 / (n_kept + 1)
        if corrected <= alpha:
            return float(lam)
    return 1.0


@dataclass
class ConformalGate:
    alpha: float = 0.10
    min_group: int = 30
    global_threshold: float = 1.0
    group_thresholds: dict[str, float] = field(default_factory=dict)

    def fit(
        self,
        scores: np.ndarray,
        losses: np.ndarray,
        groups: list[str] | None = None,
    ) -> ConformalGate:
        scores = np.asarray(scores, dtype=float)
        losses = np.asarray(losses, dtype=float)
        self.global_threshold = calibrate_threshold(scores, losses, self.alpha)
        self.group_thresholds = {}
        if groups is not None:
            arr = np.asarray(groups)
            for g in np.unique(arr):
                mask = arr == g
                if int(mask.sum()) >= self.min_group:
                    self.group_thresholds[str(g)] = calibrate_threshold(
                        scores[mask], losses[mask], self.alpha
                    )
        return self

    def decide(self, score: float, group: str = "_global") -> GateDecision:
        thr = self.group_thresholds.get(group, self.global_threshold)
        return GateDecision(
            answer=bool(score >= thr), score=float(score), threshold=float(thr), group=group
        )

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "min_group": self.min_group,
            "global_threshold": self.global_threshold,
            "group_thresholds": dict(self.group_thresholds),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ConformalGate:
        gate = cls(alpha=d["alpha"], min_group=d["min_group"])
        gate.global_threshold = d["global_threshold"]
        gate.group_thresholds = dict(d["group_thresholds"])
        return gate


def selective_risk(
    scores: np.ndarray, losses: np.ndarray, threshold: float
) -> dict[str, float]:
    """Held-out evaluation of a fitted gate: risk among answered + answer rate."""
    scores = np.asarray(scores, dtype=float)
    losses = np.asarray(losses, dtype=float)
    kept = scores >= threshold
    n_kept = int(kept.sum())
    return {
        "risk": float(losses[kept].mean()) if n_kept else 0.0,
        "answer_rate": n_kept / len(scores) if len(scores) else 0.0,
        "n_answered": n_kept,
        "n_total": len(scores),
    }
