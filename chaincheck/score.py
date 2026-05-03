"""Score aggregation utilities for ChainCheck."""

from __future__ import annotations

from collections.abc import Mapping

from chaincheck.config import env_float, method_weights


def risk_level(score: float, low_threshold: float = 0.3, high_threshold: float = 0.7) -> str:
    """Map an aggregate hallucination risk score to a risk level."""
    if score < low_threshold:
        return "low"
    if score < high_threshold:
        return "medium"
    return "high"


def aggregate_score(method_scores: Mapping[str, float], selected: list[str]) -> float:
    """Aggregate method hallucination scores with normalized configured weights."""
    weights = method_weights(selected)
    return sum(method_scores.get(name, 0.0) * weight for name, weight in weights.items())


def nli_risk(label: str, confidence: float) -> float:
    """Convert an NLI label and confidence into hallucination risk."""
    if label == "entailed":
        return 1.0 - confidence
    if label == "contradicted":
        return confidence
    return max(0.5, confidence)


def judge_risk(label: str, confidence: float) -> float:
    """Convert a judge label and confidence into hallucination risk."""
    if label == "supported":
        return 1.0 - confidence
    if label == "contradicted":
        return confidence
    return max(0.5, confidence)


def mean(values: list[float]) -> float:
    """Return the arithmetic mean of a list, or zero for an empty list."""
    return sum(values) / len(values) if values else 0.0


def configured_risk_level(score: float) -> str:
    """Map score to risk level using environment thresholds."""
    return risk_level(
        score,
        low_threshold=env_float("RISK_LOW_THRESHOLD", 0.3),
        high_threshold=env_float("RISK_HIGH_THRESHOLD", 0.7),
    )
