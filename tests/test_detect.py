"""Tests for chaincheck.detect — orchestration and weighted aggregation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from chaincheck.detect import _compute_risk_level, _weighted_aggregate
from chaincheck.models import DetectionResult, MethodResult


def _mr(method: str, score: float) -> MethodResult:
    return MethodResult(method=method, raw_score=score, latency_ms=10.0)


class TestComputeRiskLevel:
    def test_low(self):
        assert _compute_risk_level(0.0) == "low"

    def test_low_boundary(self):
        assert _compute_risk_level(0.29) == "low"

    def test_medium(self):
        assert _compute_risk_level(0.5) == "medium"

    def test_high_boundary(self):
        assert _compute_risk_level(0.7) == "high"

    def test_high(self):
        assert _compute_risk_level(1.0) == "high"


class TestWeightedAggregate:
    def test_empty_returns_zero(self):
        assert _weighted_aggregate({}, []) == 0.0

    def test_single_method_equals_score(self):
        results = {"nli": _mr("nli", 0.8)}
        score = _weighted_aggregate(results, ["nli"])
        assert score == pytest.approx(0.8, abs=1e-6)

    def test_all_three_default_weights(self):
        results = {
            "nli": _mr("nli", 1.0),
            "consistency": _mr("consistency", 1.0),
            "judge": _mr("judge", 1.0),
        }
        score = _weighted_aggregate(results, ["nli", "consistency", "judge"])
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_two_methods_normalises_weights(self):
        results = {
            "nli": _mr("nli", 0.0),
            "judge": _mr("judge", 1.0),
        }
        score = _weighted_aggregate(results, ["nli", "judge"])
        # nli weight=0.35, judge weight=0.25 → normalised sum = 0.6
        expected = (0.35 * 0.0 + 0.25 * 1.0) / (0.35 + 0.25)
        assert score == pytest.approx(expected, abs=1e-6)

    def test_unknown_method_ignored(self):
        results = {"nli": _mr("nli", 0.6), "unknown_method": _mr("unknown_method", 1.0)}
        score = _weighted_aggregate(results, ["nli", "unknown_method"])
        assert score == pytest.approx(0.6, abs=1e-6)

    def test_clamps_to_one(self):
        results = {"nli": _mr("nli", 1.0)}
        assert _weighted_aggregate(results, ["nli"]) <= 1.0

    def test_clamps_to_zero(self):
        results = {"nli": _mr("nli", 0.0)}
        assert _weighted_aggregate(results, ["nli"]) >= 0.0


@pytest.mark.asyncio
async def test_detect_returns_detection_result():
    """detect() should return a DetectionResult for a simple response."""
    from chaincheck.detect import detect

    mock_claims = ["The sky is blue."]
    mock_nli = MethodResult(method="nli", raw_score=0.1, latency_ms=50.0)

    with (
        patch("chaincheck.detect.decompose", new=AsyncMock(return_value=mock_claims)),
        patch("chaincheck.detect.check_nli", new=AsyncMock(return_value=mock_nli)),
        patch("chaincheck.detect.check_judge", new=AsyncMock(return_value=mock_nli)),
    ):
        result = await detect("The sky is blue.", context="The sky appears blue.", methods=["nli"])

    assert isinstance(result, DetectionResult)
    assert result.claims == mock_claims
    assert result.request_id is not None


@pytest.mark.asyncio
async def test_detect_skips_nli_without_context():
    """detect() should not call check_nli when context is empty."""
    from chaincheck.detect import detect

    mock_claims = ["Some claim."]
    mock_judge = MethodResult(method="judge", raw_score=0.5, latency_ms=100.0)

    with (
        patch("chaincheck.detect.decompose", new=AsyncMock(return_value=mock_claims)),
        patch("chaincheck.detect.check_judge", new=AsyncMock(return_value=mock_judge)) as mock_j,
        patch("chaincheck.detect.check_nli", new=AsyncMock()) as mock_n,
    ):
        result = await detect("Some claim.", context="", methods=["nli", "judge"])

    mock_n.assert_not_called()
    mock_j.assert_called_once()
    assert "nli" not in result.method_results


@pytest.mark.asyncio
async def test_detect_handles_method_exception_gracefully():
    """A failing method should produce an error MethodResult, not crash detect()."""
    from chaincheck.detect import detect

    mock_claims = ["Some claim."]

    async def _raise(*a, **kw):
        raise RuntimeError("model timeout")

    with (
        patch("chaincheck.detect.decompose", new=AsyncMock(return_value=mock_claims)),
        patch("chaincheck.detect.check_judge", new=AsyncMock(side_effect=_raise)),
    ):
        result = await detect("Some claim.", context="ctx", methods=["judge"])

    assert result.method_results["judge"].error is not None
