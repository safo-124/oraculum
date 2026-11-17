"""Calibrator fitting and selective-prediction metrics (NumPy only).

Used at runtime (Calibrator.predict) and by the study scripts
(scripts/fit_calibrator.py). Kept dependency-free beyond NumPy so the
server never imports scikit-learn.

Terminology (selective prediction):
- coverage: the fraction of questions the system answers;
- risk: the fraction of answered questions whose answer is wrong;
- a confidence score is useful when sorting by it gives low risk at high
  coverage (area under the risk-coverage curve, AURC: lower is better);
- it is calibrated when, among answers given confidence p, a fraction p is
  correct (expected calibration error, ECE: lower is better).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- the model


@dataclass
class Calibrator:
    """Standardize -> logistic regression -> P(answer is correct)."""

    features: list[str]
    mean: list[float]
    std: list[float]
    weights: list[float]
    bias: float
    threshold: float = 0.5
    meta: dict = field(default_factory=dict)

    def vector(self, values: Mapping[str, float | None]) -> np.ndarray:
        """Feature row; a missing or non-finite value is imputed with the mean."""
        row = []
        for name, mean in zip(self.features, self.mean):
            value = values.get(name)
            if value is None or not math.isfinite(float(value)):
                value = mean
            row.append(float(value))
        return np.asarray(row, dtype=float)

    def predict_matrix(self, matrix: np.ndarray) -> np.ndarray:
        z = (matrix - np.asarray(self.mean)) / np.asarray(self.std)
        return _sigmoid(z @ np.asarray(self.weights) + self.bias)

    def predict(self, values: Mapping[str, float | None]) -> float:
        return float(self.predict_matrix(self.vector(values)[None, :])[0])

    def to_json(self) -> dict:
        return {
            "features": self.features,
            "mean": self.mean,
            "std": self.std,
            "weights": self.weights,
            "bias": self.bias,
            "threshold": self.threshold,
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, payload: Mapping) -> "Calibrator":
        return cls(
            features=list(payload["features"]),
            mean=[float(v) for v in payload["mean"]],
            std=[float(v) for v in payload["std"]],
            weights=[float(v) for v in payload["weights"]],
            bias=float(payload["bias"]),
            threshold=float(payload.get("threshold", 0.5)),
            meta=dict(payload.get("meta", {})),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Calibrator":
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def impute_matrix(
    rows: Sequence[Mapping[str, float | None]], features: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Rows -> matrix with NaN for missing values, plus per-column means."""
    matrix = np.array(
        [
            [
                float(row[name])
                if row.get(name) is not None and math.isfinite(float(row[name]))
                else np.nan
                for name in features
            ]
            for row in rows
        ],
        dtype=float,
    ).reshape(len(rows), len(features))
    with np.errstate(all="ignore"):
        means = np.nanmean(matrix, axis=0) if len(rows) else np.zeros(len(features))
    means = np.where(np.isfinite(means), means, 0.0)
    filled = np.where(np.isnan(matrix), means, matrix)
    return filled, means


def fit_logistic(
    rows: Sequence[Mapping[str, float | None]],
    labels: Sequence[int],
    features: Sequence[str],
    *,
    l2: float = 1.0,
    iterations: int = 50,
) -> Calibrator:
    """L2-regularized logistic regression by Newton's method (IRLS).

    Features are standardized with the training mean/std, so the penalty
    treats them equally and the weights are comparable as importances.
    The intercept is not penalized.
    """
    x, means = impute_matrix(rows, features)
    y = np.asarray(labels, dtype=float)
    std = x.std(axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    z = (x - means) / std
    design = np.hstack([z, np.ones((len(z), 1))])
    beta = np.zeros(design.shape[1])
    penalty = np.full(design.shape[1], l2)
    penalty[-1] = 0.0
    for _ in range(iterations):
        p = _sigmoid(design @ beta)
        gradient = design.T @ (p - y) + penalty * beta
        weight = p * (1.0 - p)
        hessian = (design * weight[:, None]).T @ design + np.diag(penalty)
        step = np.linalg.solve(hessian + 1e-9 * np.eye(len(beta)), gradient)
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return Calibrator(
        features=list(features),
        mean=[float(v) for v in means],
        std=[float(v) for v in std],
        weights=[float(v) for v in beta[:-1]],
        bias=float(beta[-1]),
    )


# ---------------------------------------------------------------- metrics


def _equal_mass_bins(conf: np.ndarray, bins: int) -> np.ndarray:
    """Bin index per value; edges at quantiles, and tied values share a bin."""
    if len(conf) == 0:
        return np.zeros(0, dtype=int)
    edges = np.unique(np.quantile(conf, np.linspace(0.0, 1.0, bins + 1))[1:-1])
    return np.searchsorted(edges, conf, side="right")


def expected_calibration_error(
    confidence: Sequence[float], correct: Sequence[int], bins: int = 10
) -> float:
    """ECE with equal-mass bins (robust when confidences cluster)."""
    conf = np.asarray(confidence, dtype=float)
    hit = np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return float("nan")
    index = _equal_mass_bins(conf, bins)
    error = 0.0
    for b in np.unique(index):
        mask = index == b
        error += mask.sum() / len(conf) * abs(conf[mask].mean() - hit[mask].mean())
    return float(error)


def reliability_table(
    confidence: Sequence[float], correct: Sequence[int], bins: int = 10
) -> list[dict]:
    conf = np.asarray(confidence, dtype=float)
    hit = np.asarray(correct, dtype=float)
    index = _equal_mass_bins(conf, bins)
    return [
        {
            "n": int((index == b).sum()),
            "mean_confidence": float(conf[index == b].mean()),
            "accuracy": float(hit[index == b].mean()),
        }
        for b in np.unique(index)
    ]


def brier_score(confidence: Sequence[float], correct: Sequence[int]) -> float:
    conf = np.asarray(confidence, dtype=float)
    hit = np.asarray(correct, dtype=float)
    return float(np.mean((conf - hit) ** 2)) if len(conf) else float("nan")


def auroc(score: Sequence[float], correct: Sequence[int]) -> float:
    """P(score of a correct answer > score of a wrong one); ties count half."""
    s = np.asarray(score, dtype=float)
    y = np.asarray(correct, dtype=int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = _average_ranks(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def risk_coverage(
    score: Sequence[float], correct: Sequence[int], answered: Sequence[bool] | None = None
) -> dict:
    """Risk-coverage curve over ALL questions.

    Questions the system never answers (answered=False: retrieval found
    nothing, or the model declined) sit outside coverage at every threshold.
    The rest are admitted in order of decreasing score.
    """
    s = np.asarray(score, dtype=float)
    y = np.asarray(correct, dtype=float)
    mask = np.ones(len(s), dtype=bool) if answered is None else np.asarray(answered, dtype=bool)
    total = len(s)
    idx = np.where(mask)[0]
    order = idx[np.argsort(-s[idx], kind="stable")]
    coverage, risk = [], []
    errors = 0.0
    for count, i in enumerate(order, start=1):
        errors += 1.0 - y[i]
        coverage.append(count / total)
        risk.append(errors / count)
    aurc = float(np.mean(risk)) if risk else float("nan")
    return {"coverage": coverage, "risk": risk, "aurc": aurc, "max_coverage": len(order) / max(1, total)}


def selective_accuracy_at(curve: dict, target_coverage: float) -> float | None:
    """Accuracy of answered questions at the largest coverage <= target."""
    best = None
    for coverage, risk in zip(curve["coverage"], curve["risk"]):
        if coverage <= target_coverage + 1e-12:
            best = 1.0 - risk
    return best


def threshold_for_risk(
    score: Sequence[float], correct: Sequence[int], target_risk: float
) -> float:
    """Lowest threshold whose admitted set has risk <= target_risk.

    Scans thresholds from the top; returns 1.0 (answer nothing) when even
    the most confident answers exceed the target.
    """
    s = np.asarray(score, dtype=float)
    y = np.asarray(correct, dtype=float)
    order = np.argsort(-s, kind="stable")
    best = 1.0
    errors = 0.0
    for count, i in enumerate(order, start=1):
        errors += 1.0 - y[i]
        if errors / count <= target_risk:
            best = float(s[i])
    return best
