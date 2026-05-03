"""Classification and latency metrics for ChainCheck benchmarks."""

from __future__ import annotations

import statistics


def classification_metrics(predictions: list[bool], labels: list[bool]) -> dict[str, float]:
    """Compute precision, recall, F1, and accuracy for binary predictions."""
    paired = list(zip(predictions, labels, strict=False))
    tp = sum(pred and label for pred, label in paired)
    fp = sum(pred and not label for pred, label in paired)
    fn = sum(not pred and label for pred, label in paired)
    correct = sum(pred == label for pred, label in paired)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = correct / len(labels) if labels else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def latency_metrics(latencies_ms: list[float]) -> dict[str, float]:
    """Compute average, p50, and p95 latency."""
    if not latencies_ms:
        return {"avg_latency_ms": 0.0, "p50_latency_ms": 0.0, "p95_latency_ms": 0.0}
    sorted_latencies = sorted(latencies_ms)
    p95_index = min(len(sorted_latencies) - 1, int(len(sorted_latencies) * 0.95))
    return {
        "avg_latency_ms": statistics.fmean(latencies_ms),
        "p50_latency_ms": statistics.median(latencies_ms),
        "p95_latency_ms": sorted_latencies[p95_index],
    }
