"""Tests for chaincheck.methods.nli — NLI entailment detection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chaincheck.methods.nli import (
    _build_label_map,
    _claim_context_hash,
    _find_best_evidence,
    _score_from_claims,
)
from chaincheck.models import ClaimResult


class TestClaimContextHash:
    def test_deterministic(self):
        assert _claim_context_hash("claim", "ctx") == _claim_context_hash("claim", "ctx")

    def test_different_claim(self):
        assert _claim_context_hash("A", "ctx") != _claim_context_hash("B", "ctx")

    def test_different_context(self):
        assert _claim_context_hash("claim", "A") != _claim_context_hash("claim", "B")

    def test_returns_64_char_hex(self):
        h = _claim_context_hash("claim", "ctx")
        assert len(h) == 64
        int(h, 16)


class TestBuildLabelMap:
    def test_standard_labels(self):
        cfg = MagicMock()
        cfg.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
        model = MagicMock()
        model.config = cfg
        mapping = _build_label_map(model)
        assert mapping[0] == "contradicted"
        assert mapping[1] == "supported"
        assert mapping[2] == "unknown"

    def test_fallback_when_no_config(self):
        model = MagicMock()
        model.config = None
        mapping = _build_label_map(model)
        assert set(mapping.values()) <= {"supported", "contradicted", "unknown"}


class TestFindBestEvidence:
    def test_returns_best_matching_sentence(self):
        context = "Paris is the capital of France. Berlin is in Germany."
        claim = "Paris is a capital city."
        evidence = _find_best_evidence(claim, context)
        assert "Paris" in evidence

    def test_empty_context_returns_fallback(self):
        result = _find_best_evidence("Some claim.", "")
        assert result == "no relevant context found"

    def test_no_overlap_returns_fallback(self):
        result = _find_best_evidence("zzz yyy xxx", "abc def ghi")
        assert result == "no relevant context found"


class TestScoreFromClaims:
    def test_all_supported_returns_zero(self):
        claims = [
            ClaimResult(
                claim="c", label="supported", confidence=0.9, evidence="e", method="nli"
            )
        ]
        assert _score_from_claims(claims) == pytest.approx(0.0, abs=1e-6)

    def test_all_contradicted_returns_one(self):
        claims = [
            ClaimResult(
                claim="c", label="contradicted", confidence=1.0, evidence="e", method="nli"
            )
        ]
        assert _score_from_claims(claims) == pytest.approx(1.0, abs=1e-6)

    def test_empty_returns_zero(self):
        assert _score_from_claims([]) == 0.0

    def test_zero_confidence_uses_unweighted(self):
        claims = [
            ClaimResult(
                claim="c1", label="contradicted", confidence=0.0, evidence="e", method="nli"
            ),
            ClaimResult(
                claim="c2", label="supported", confidence=0.0, evidence="e", method="nli"
            ),
        ]
        score = _score_from_claims(claims)
        assert score == pytest.approx(0.5, abs=1e-6)


@pytest.mark.asyncio
async def test_check_nli_empty_claims():
    """check_nli() with empty claims should return raw_score=0.0 without calling the model."""
    from chaincheck.methods.nli import check_nli

    with patch("chaincheck.methods.nli._get_model") as mock_get:
        result = await check_nli([], "some context")
    mock_get.assert_not_called()
    assert result.raw_score == 0.0


@pytest.mark.asyncio
async def test_check_nli_empty_context():
    """check_nli() with empty context should label all claims unknown."""
    from chaincheck.methods.nli import check_nli

    result = await check_nli(["The sky is blue."], "")
    assert result.claims[0].label == "unknown"
    assert result.raw_score == 0.0


@pytest.mark.asyncio
async def test_check_nli_returns_one_result_per_claim():
    """check_nli() should return exactly one ClaimResult per input claim."""
    import chaincheck.methods.nli as nli_mod
    from chaincheck.methods import check_nli

    fake_scores = np.array([[0.05, 0.9, 0.05]])  # entailment dominant (index 1)
    nli_mod._label_map = {0: "contradicted", 1: "supported", 2: "unknown"}

    mock_model = MagicMock()
    mock_model.predict.return_value = fake_scores

    with patch("chaincheck.methods.nli._get_model", return_value=mock_model):
        result = await check_nli(["Paris is in France."], "Paris is the capital of France.")

    assert len(result.claims) == 1
    assert result.claims[0].label == "supported"
