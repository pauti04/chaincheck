"""Tests for chaincheck.eval.claimlevel — claim-level discrimination metrics."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from chaincheck.eval.claimlevel import (
    ClaimLevelMetrics,
    ClaimLevelRun,
    _compute_claimlevel_metrics,
    _eval_pair,
    _roc_auc,
    run_claimlevel,
)
from chaincheck.models import ClaimResult, DetectionResult, MethodResult


class TestRocAuc:
    def test_perfect_separation(self):
        labels = [1, 1, 0, 0]
        scores = [1.0, 0.9, 0.1, 0.0]
        assert _roc_auc(labels, scores) == pytest.approx(1.0, abs=1e-6)

    def test_inverse_separation(self):
        labels = [1, 1, 0, 0]
        scores = [0.0, 0.1, 0.9, 1.0]
        assert _roc_auc(labels, scores) == pytest.approx(0.0, abs=1e-6)

    def test_random_is_near_half(self):
        import random
        random.seed(42)
        labels = [random.randint(0, 1) for _ in range(200)]
        scores = [random.random() for _ in range(200)]
        auc = _roc_auc(labels, scores)
        assert 0.3 < auc < 0.7

    def test_all_positive_returns_half(self):
        assert _roc_auc([1, 1, 1], [0.5, 0.6, 0.7]) == pytest.approx(0.5, abs=1e-6)

    def test_all_negative_returns_half(self):
        assert _roc_auc([0, 0, 0], [0.5, 0.6, 0.7]) == pytest.approx(0.5, abs=1e-6)

    def test_empty_returns_half(self):
        assert _roc_auc([], []) == pytest.approx(0.5, abs=1e-6)


class TestComputeClaimlevelMetrics:
    def _make_raw(
        self,
        correct_flagged: int,
        correct_total: int,
        halluc_flagged: int,
        halluc_total: int,
    ) -> list[dict]:
        return [{
            "question": "q",
            "correct_score": 0.0,
            "halluc_score": 1.0,
            "correct_claims": correct_total,
            "halluc_claims": halluc_total,
            "correct_flagged": correct_flagged,
            "halluc_flagged": halluc_flagged,
            "correct_claim_scores": [0.0] * correct_total,
            "halluc_claim_scores": [1.0] * halluc_total,
        }]

    def test_perfect_discrimination(self):
        raw = self._make_raw(0, 5, 5, 5)
        m = _compute_claimlevel_metrics(raw, 1.0)
        assert m.clean_flagging_rate == pytest.approx(0.0, abs=1e-6)
        assert m.halluc_flagging_rate == pytest.approx(1.0, abs=1e-6)
        assert m.discrimination_ratio == float("inf")

    def test_no_discrimination(self):
        raw = self._make_raw(5, 5, 5, 5)
        m = _compute_claimlevel_metrics(raw, 1.0)
        assert m.discrimination_ratio == pytest.approx(1.0, abs=1e-6)

    def test_empty_raw(self):
        m = _compute_claimlevel_metrics([], 0.0)
        assert m.clean_flagging_rate == 0.0
        assert m.halluc_flagging_rate == 0.0
        assert m.n_pairs == 0

    def test_n_pairs_counted(self):
        raw = self._make_raw(1, 4, 3, 4) * 3
        m = _compute_claimlevel_metrics(raw, 1.0)
        assert m.n_pairs == 3

    def test_avg_claims_per_response(self):
        raw = self._make_raw(0, 4, 2, 4)
        m = _compute_claimlevel_metrics(raw, 1.0)
        # (4 clean + 4 halluc) / (2 * 1 pair) = 4.0
        assert m.avg_claims_per_response == pytest.approx(4.0, abs=1e-6)

    def test_claim_auc_range(self):
        raw = self._make_raw(0, 5, 5, 5)
        m = _compute_claimlevel_metrics(raw, 1.0)
        assert 0.0 <= m.claim_auc <= 1.0


def _fake_detect_result(label: str) -> DetectionResult:
    claim_result = ClaimResult(
        claim="test claim", label=label, confidence=0.9, evidence="", method="nli"
    )
    method_result = MethodResult(
        method="nli", claims=[claim_result], raw_score=0.8, latency_ms=50.0
    )
    return DetectionResult(
        response="test response",
        claims=["test claim"],
        method_results={"nli": method_result},
        aggregate_score=0.8 if label == "contradicted" else 0.1,
        risk_level="high" if label == "contradicted" else "low",
        latency_ms={"nli": 50.0},
        request_id="test-id",
    )


class TestEvalPair:
    @pytest.mark.asyncio
    async def test_returns_expected_keys(self):
        pair = {
            "question": "What is X?",
            "context": "X is a thing.",
            "correct_answer": "X is a thing.",
            "hallucinated_answer": "X is not real.",
        }
        fake_correct = _fake_detect_result("supported")
        fake_halluc = _fake_detect_result("contradicted")

        with patch("chaincheck.detect.detect", new=AsyncMock(
            side_effect=[fake_correct, fake_halluc]
        )):
            result = await _eval_pair(pair, "nli")

        assert "correct_score" in result
        assert "halluc_score" in result
        assert "correct_flagged" in result
        assert "halluc_flagged" in result
        assert "correct_claim_scores" in result
        assert "halluc_claim_scores" in result

    @pytest.mark.asyncio
    async def test_flagging_counts_correctly(self):
        pair = {
            "question": "q?",
            "context": "ctx",
            "correct_answer": "correct",
            "hallucinated_answer": "hallucinated",
        }
        fake_correct = _fake_detect_result("supported")   # 0 flagged
        fake_halluc = _fake_detect_result("contradicted")  # 1 flagged

        with patch("chaincheck.detect.detect", new=AsyncMock(
            side_effect=[fake_correct, fake_halluc]
        )):
            result = await _eval_pair(pair, "nli")

        assert result["correct_flagged"] == 0
        assert result["halluc_flagged"] == 1


class TestRunClaimlevel:
    @pytest.mark.asyncio
    async def test_returns_claimlevel_run(self):
        fake_pairs = [{
            "question": "q?",
            "context": "ctx",
            "correct_answer": "right answer",
            "hallucinated_answer": "wrong answer",
        }]

        fake_correct = _fake_detect_result("supported")
        fake_halluc = _fake_detect_result("contradicted")

        with (
            patch("chaincheck.eval.claimlevel._load_pairs", return_value=fake_pairs),
            patch("chaincheck.detect.detect", new=AsyncMock(
                side_effect=[fake_correct, fake_halluc]
            )),
        ):
            result = await run_claimlevel(method="nli", n_pairs=1)

        assert isinstance(result, ClaimLevelRun)
        assert result.method == "nli"
        assert result.pairs == 1
        assert isinstance(result.metrics, ClaimLevelMetrics)

    @pytest.mark.asyncio
    async def test_metrics_populated(self):
        fake_pairs = [{
            "question": "q?",
            "context": "ctx",
            "correct_answer": "right answer",
            "hallucinated_answer": "wrong answer",
        }]

        fake_correct = _fake_detect_result("supported")
        fake_halluc = _fake_detect_result("contradicted")

        with (
            patch("chaincheck.eval.claimlevel._load_pairs", return_value=fake_pairs),
            patch("chaincheck.detect.detect", new=AsyncMock(
                side_effect=[fake_correct, fake_halluc]
            )),
        ):
            result = await run_claimlevel(method="nli", n_pairs=1)

        assert result.metrics.n_pairs == 1
        assert result.metrics.latency_ms >= 0
