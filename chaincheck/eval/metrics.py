"""Evaluation metric computation for hallucination detection benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EvalMetrics:
    """Precision, recall, F1, accuracy, latency, and calibration for one eval run."""

    precision: float
    recall: float
    f1: float
    accuracy: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    n_samples: int
    ece: float = 0.0  # Expected Calibration Error — lower is better


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    latencies_ms: list[float],
    scores: list[float] | None = None,
    positive_label: str = "yes",
    n_bins: int = 10,
) -> EvalMetrics:
    """
    Compute classification, latency, and calibration metrics.

    Args:
        y_true: Ground-truth labels ("yes" = hallucinated, "no" = clean).
        y_pred: Predicted labels from the detection method.
        latencies_ms: Per-sample detection latency in milliseconds.
        scores: Raw continuous scores in [0, 1] (aggregate_score). When provided,
                ECE is computed. ECE = 0 means confidence equals accuracy everywhere.
        positive_label: Label treated as positive class (default "yes").
        n_bins: Number of equal-width bins for ECE computation.

    Returns:
        EvalMetrics dataclass with all computed statistics.
    """
    n = len(y_true)
    pairs = list(zip(y_true, y_pred, strict=True))
    pos = positive_label
    tp = sum(1 for t, p in pairs if t == pos and p == pos)
    fp = sum(1 for t, p in pairs if t != pos and p == pos)
    fn = sum(1 for t, p in pairs if t == pos and p != pos)
    tn = sum(1 for t, p in pairs if t != pos and p != pos)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    )
    accuracy = (tp + tn) / n if n > 0 else 0.0

    lat = np.array(latencies_ms, dtype=np.float64)
    ece = _compute_ece(y_true, scores, positive_label, n_bins) if scores is not None else 0.0

    return EvalMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        avg_latency_ms=float(np.mean(lat)),
        p50_latency_ms=float(np.percentile(lat, 50)),
        p95_latency_ms=float(np.percentile(lat, 95)),
        n_samples=n,
        ece=ece,
    )


def _compute_ece(
    y_true: list[str],
    scores: list[float],
    positive_label: str,
    n_bins: int,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Bins predictions by confidence score, then measures the weighted mean
    absolute difference between average confidence and fraction of true positives
    within each bin. ECE of 0 means the model's confidence is perfectly calibrated.
    """
    if not scores:
        return 0.0
    s = np.array(scores, dtype=np.float64)
    y = np.array([1 if t == positive_label else 0 for t in y_true], dtype=np.float64)
    n = len(s)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (s >= lo) & (s < hi)
        if not mask.any():
            continue
        b_conf = float(s[mask].mean())
        b_acc = float(y[mask].mean())
        ece += (mask.sum() / n) * abs(b_conf - b_acc)
    return float(ece)
