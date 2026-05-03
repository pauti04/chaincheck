"""Tests for ChainCheck score aggregation."""

from chaincheck.score import (
    aggregate_score,
    configured_risk_level,
    judge_risk,
    nli_risk,
    risk_level,
)


def test_risk_level_thresholds() -> None:
    """Risk levels follow configured thresholds."""
    assert risk_level(0.1) == "low"
    assert risk_level(0.5) == "medium"
    assert risk_level(0.8) == "high"


def test_method_risk_conversions() -> None:
    """Method-specific labels convert into hallucination risk."""
    assert nli_risk("entailed", 0.9) < 0.2
    assert judge_risk("contradicted", 0.9) > 0.8


def test_aggregate_score_selected_methods() -> None:
    """Aggregate score normalizes over selected methods."""
    score = aggregate_score({"nli": 1.0}, ["nli"])
    assert score == 1.0


def test_configured_risk_level(monkeypatch) -> None:
    """Risk thresholds are environment configurable."""
    monkeypatch.setenv("RISK_LOW_THRESHOLD", "0.2")
    monkeypatch.setenv("RISK_HIGH_THRESHOLD", "0.4")
    assert configured_risk_level(0.3) == "medium"
