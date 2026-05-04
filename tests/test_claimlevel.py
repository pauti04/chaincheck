"""Tests for chaincheck.eval.claimlevel — claim-level discrimination metrics."""

from __future__ import annotations

import pytest

from chaincheck.eval.claimlevel import (
    ClaimLevelMetrics,
    _compute_claimlevel_metrics,
    _roc_auc,
)


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
