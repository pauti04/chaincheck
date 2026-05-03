"""Tests for chaincheck.methods.judge — LLM-as-judge detection."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from chaincheck.methods.judge import (
    JudgeVerdict,
    _build_judge_prompt,
    _score_from_claims,
    _truncate_context,
)
from chaincheck.models import ClaimResult


class TestJudgeVerdict:
    def test_valid(self):
        v = JudgeVerdict(label="supported", confidence=0.9, evidence="e")
        assert v.confidence == 0.9

    def test_confidence_too_high_raises(self):
        with pytest.raises(ValidationError):
            JudgeVerdict(label="supported", confidence=1.5, evidence="e")

    def test_confidence_negative_raises(self):
        with pytest.raises(ValidationError):
            JudgeVerdict(label="supported", confidence=-0.1, evidence="e")

    def test_confidence_boundary_values(self):
        JudgeVerdict(label="supported", confidence=0.0, evidence="e")
        JudgeVerdict(label="supported", confidence=1.0, evidence="e")


class TestTruncateContext:
    def test_short_passes_through(self):
        short = "A short context."
        assert _truncate_context(short, max_tokens=800) == short

    def test_long_is_truncated(self):
        long = " ".join(["word"] * 1200)
        result = _truncate_context(long, max_tokens=800)
        assert len(result.split()) < 1200

    def test_empty_returns_empty(self):
        assert _truncate_context("", max_tokens=800) == ""

    def test_exact_limit_passes_through(self):
        # 800 tokens * 0.75 = 600 words
        text = " ".join(["word"] * 600)
        assert _truncate_context(text, max_tokens=800) == text


class TestBuildJudgePrompt:
    def test_contains_claim(self):
        p = _build_judge_prompt("The sky is blue.", "Some context.")
        assert "The sky is blue." in p

    def test_contains_context(self):
        p = _build_judge_prompt("Some claim.", "The sky is blue.")
        assert "The sky is blue." in p

    def test_empty_context_handled(self):
        p = _build_judge_prompt("Some claim.", "")
        assert "none provided" in p.lower() or "no context" in p.lower() or "none" in p.lower()


class TestScoreFromClaims:
    def test_all_supported(self):
        claims = [
            ClaimResult(claim="c", label="supported", confidence=0.9, evidence="e", method="judge")
        ]
        assert _score_from_claims(claims) == pytest.approx(0.0, abs=1e-6)

    def test_all_contradicted(self):
        claims = [
            ClaimResult(
                claim="c", label="contradicted", confidence=1.0, evidence="e", method="judge"
            )
        ]
        assert _score_from_claims(claims) == pytest.approx(1.0, abs=1e-6)

    def test_empty(self):
        assert _score_from_claims([]) == 0.0


@pytest.mark.asyncio
async def test_check_judge_empty_claims():
    """check_judge() with empty claims returns score=0.0 without API calls."""
    from chaincheck.methods.judge import check_judge

    with patch("chaincheck.llm.complete", new=AsyncMock()) as mock_complete:
        result = await check_judge([], "context")
    mock_complete.assert_not_called()
    assert result.raw_score == 0.0


@pytest.mark.asyncio
async def test_check_judge_preserves_claim_order():
    """Results must be in original claim order despite internal shuffling."""
    from chaincheck.methods.judge import check_judge

    claims = ["Claim A is here.", "Claim B is here.", "Claim C is here."]

    async def _fake_verify(claim, context, model, retries=3):
        return JudgeVerdict(
            label="supported" if "A" in claim else "contradicted",
            confidence=0.9,
            evidence="test",
        )

    with patch("chaincheck.methods.judge._verify_claim", side_effect=_fake_verify):
        result = await check_judge(claims, "some context")

    assert result.claims[0].claim == "Claim A is here."
    assert result.claims[0].label == "supported"
    assert result.claims[1].claim == "Claim B is here."
    assert result.claims[1].label == "contradicted"


@pytest.mark.asyncio
async def test_check_judge_returns_correct_count():
    """check_judge() should return exactly one ClaimResult per input claim."""
    from chaincheck.methods.judge import check_judge

    claims = ["Claim one.", "Claim two.", "Claim three."]

    async def _fake_verify(claim, context, model, retries=3):
        return JudgeVerdict(label="supported", confidence=0.8, evidence="ok")

    with patch("chaincheck.methods.judge._verify_claim", side_effect=_fake_verify):
        result = await check_judge(claims, "ctx")

    assert len(result.claims) == 3
