"""Tests for chaincheck.score — claim-level aggregation."""

from __future__ import annotations

import pytest

from chaincheck.models import ClaimResult, MethodResult
from chaincheck.score import aggregate_claim_scores, method_score


def _cr(label: str, confidence: float = 0.9) -> ClaimResult:
    return ClaimResult(claim="c", label=label, confidence=confidence, evidence="e", method="nli")


class TestAggregateClaimScores:
    def test_empty_returns_zero(self):
        assert aggregate_claim_scores([]) == 0.0

    def test_all_supported(self):
        claims = [_cr("supported", 0.9), _cr("supported", 0.8)]
        assert aggregate_claim_scores(claims) == pytest.approx(0.0, abs=1e-6)

    def test_all_contradicted(self):
        claims = [_cr("contradicted", 1.0)]
        assert aggregate_claim_scores(claims) == pytest.approx(1.0, abs=1e-6)

    def test_all_unsupported(self):
        claims = [_cr("unsupported", 1.0)]
        assert aggregate_claim_scores(claims) == pytest.approx(1.0, abs=1e-6)

    def test_mixed_50_50(self):
        claims = [_cr("supported", 1.0), _cr("contradicted", 1.0)]
        assert aggregate_claim_scores(claims) == pytest.approx(0.5, abs=1e-6)

    def test_weighted_by_confidence(self):
        # contradicted with confidence 0.2, supported with confidence 0.8
        claims = [_cr("contradicted", 0.2), _cr("supported", 0.8)]
        score = aggregate_claim_scores(claims)
        expected = 0.2 / (0.2 + 0.8)
        assert score == pytest.approx(expected, abs=1e-6)

    def test_zero_confidence_falls_back_to_unweighted(self):
        claims = [_cr("contradicted", 0.0), _cr("supported", 0.0)]
        assert aggregate_claim_scores(claims) == pytest.approx(0.5, abs=1e-6)

    def test_unknown_not_penalised(self):
        claims = [_cr("unknown", 0.9)]
        assert aggregate_claim_scores(claims) == pytest.approx(0.0, abs=1e-6)

    def test_clamps_to_one(self):
        # Feed raw fractions that could exceed 1.0 via the sum path (all zero-confidence)
        claims = [_cr("contradicted", 0.0)] * 3 + [_cr("supported", 0.0)]
        assert aggregate_claim_scores(claims) <= 1.0

    def test_returns_float(self):
        assert isinstance(aggregate_claim_scores([_cr("supported")]), float)


class TestMethodScore:
    def test_uses_claims_when_present(self):
        mr = MethodResult(
            method="nli",
            claims=[_cr("contradicted", 1.0)],
            raw_score=0.0,
            latency_ms=10.0,
        )
        assert method_score(mr) == pytest.approx(1.0, abs=1e-6)

    def test_falls_back_to_raw_score_when_no_claims(self):
        mr = MethodResult(method="nli", raw_score=0.7, latency_ms=10.0)
        assert method_score(mr) == pytest.approx(0.7, abs=1e-6)

    def test_prefers_computed_over_raw(self):
        # raw_score=0.0 but claims say 1.0 — claims win
        mr = MethodResult(
            method="nli",
            claims=[_cr("contradicted", 1.0)],
            raw_score=0.0,
            latency_ms=10.0,
        )
        assert method_score(mr) != pytest.approx(0.0, abs=1e-6)
