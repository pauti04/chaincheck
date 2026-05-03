"""Evaluation metric computation for hallucination detection benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EvalMetrics:
    """Precision, recall, F1, accuracy, and latency statistics for one eval run."""

    precision: float
    recall: float
    f1: float
    accuracy: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    n_samples: int


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    latencies_ms: list[float],
    positive_label: str = "yes",
) -> EvalMetrics:
    """
    Compute classification and latency metrics.

    Args:
        y_true: Ground-truth labels ("yes" = hallucinated, "no" = clean).
        y_pred: Predicted labels from the detection method.
        latencies_ms: Per-sample detection latency in milliseconds.
        positive_label: Label treated as positive class (default "yes").

    Returns:
        EvalMetrics dataclass with all computed statistics.
    """
    n = len(y_true)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p == positive_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_label and p == positive_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p != positive_label)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t != positive_label and p != positive_label)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    )
    accuracy = (tp + tn) / n if n > 0 else 0.0

    lat = np.array(latencies_ms, dtype=np.float64)
    return EvalMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        avg_latency_ms=float(np.mean(lat)),
        p50_latency_ms=float(np.percentile(lat, 50)),
        p95_latency_ms=float(np.percentile(lat, 95)),
        n_samples=n,
    )
