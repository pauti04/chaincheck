"""Tests for chaincheck.methods.logprobs — token-level uncertainty detection."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chaincheck.methods.logprobs import (
    _extract_token_logprobs,
    _logprob_to_label,
    _score_from_claims,
)
from chaincheck.models import ClaimResult


class TestLogprobToLabel:
    def test_high_logprob_is_supported(self):
        label, conf = _logprob_to_label(-0.1)
        assert label == "supported"
        assert conf > 0.9

    def test_low_logprob_is_unsupported(self):
        label, conf = _logprob_to_label(-5.0)
        assert label == "unsupported"

    def test_at_threshold_is_unsupported(self):
        from chaincheck.methods.logprobs import _LOGPROB_THRESHOLD

        label, _ = _logprob_to_label(_LOGPROB_THRESHOLD - 0.01)
        assert label == "unsupported"

    def test_confidence_is_probability(self):
        _, conf = _logprob_to_label(-0.5)
        assert 0.0 <= conf <= 1.0

    def test_zero_logprob_is_certain(self):
        label, conf = _logprob_to_label(0.0)
        assert label == "supported"
        assert conf == pytest.approx(1.0, abs=1e-6)


class TestExtractTokenLogprobs:
    def test_extracts_pairs(self):
        tok = MagicMock()
        tok.token = "hello"
        tok.logprob = -0.5
        content = MagicMock()
        content.content = [tok]
        choice = MagicMock()
        choice.logprobs = content
        response = MagicMock()
        response.choices = [choice]
        pairs = _extract_token_logprobs(response)
        assert pairs == [("hello", -0.5)]

    def test_returns_empty_on_missing_logprobs(self):
        response = MagicMock()
        response.choices = []
        assert _extract_token_logprobs(response) == []


class TestScoreFromClaims:
    def test_empty(self):
        assert _score_from_claims([]) == 0.0

    def test_all_supported(self):
        claims = [
            ClaimResult(
                claim="c", label="supported", confidence=0.9, evidence="e", method="logprobs"
            )
        ]
        assert _score_from_claims(claims) == pytest.approx(0.0, abs=1e-6)

    def test_all_unsupported(self):
        claims = [
            ClaimResult(
                claim="c", label="unsupported", confidence=0.8, evidence="e", method="logprobs"
            )
        ]
        assert _score_from_claims(claims) == pytest.approx(1.0, abs=1e-6)


class TestClaimAvgLogprob:
    def test_empty_token_logprobs_returns_below_threshold(self):
        from chaincheck.methods.logprobs import _LOGPROB_THRESHOLD, _claim_avg_logprob

        result = _claim_avg_logprob("The sky is blue.", "The sky is blue.", [])
        assert result < _LOGPROB_THRESHOLD

    def test_exact_match_returns_avg_of_span(self):
        from chaincheck.methods.logprobs import _claim_avg_logprob

        token_logprobs = [("The", -0.1), (" sky", -0.2), (" is", -0.1), (" blue", -0.15)]
        result = _claim_avg_logprob("the sky is blue", "The sky is blue.", token_logprobs)
        assert isinstance(result, float)

    def test_no_match_uses_word_fallback(self):
        from chaincheck.methods.logprobs import _claim_avg_logprob

        token_logprobs = [("Paris", -0.1), (" is", -0.2), (" beautiful", -0.1)]
        result = _claim_avg_logprob("Paris is beautiful", "Paris is beautiful", token_logprobs)
        assert isinstance(result, float)

    def test_short_words_fallback_returns_below_threshold(self):
        from chaincheck.methods.logprobs import _LOGPROB_THRESHOLD, _claim_avg_logprob

        # All claim words are < 4 chars, no match possible
        result = _claim_avg_logprob("no way", "something else here", [("x", -0.1)])
        assert result < _LOGPROB_THRESHOLD


@pytest.mark.asyncio
async def test_check_logprobs_no_api_key_returns_error():
    """check_logprobs() returns error MethodResult when OPENAI_API_KEY is absent."""
    from chaincheck.methods.logprobs import check_logprobs

    with patch.dict("os.environ", {}, clear=True):
        # Ensure OPENAI_API_KEY is unset
        import os

        os.environ.pop("OPENAI_API_KEY", None)
        result = await check_logprobs("prompt", ["some claim"])

    assert result.error is not None
    assert result.raw_score == 0.0


@pytest.mark.asyncio
async def test_check_logprobs_empty_claims_skips_api():
    """check_logprobs() with empty claims list should not call OpenAI."""
    import sys

    from chaincheck.methods.logprobs import check_logprobs

    mock_openai = MagicMock()
    with patch.dict(sys.modules, {"openai": mock_openai}):
        result = await check_logprobs("prompt", [])
    mock_openai.AsyncOpenAI.assert_not_called()
    assert result.raw_score == 0.0


@pytest.mark.asyncio
async def test_check_logprobs_returns_one_result_per_claim():
    """check_logprobs() should return exactly one ClaimResult per input claim."""
    import sys

    from chaincheck.methods.logprobs import check_logprobs

    tok1 = MagicMock()
    tok1.token = "The"
    tok1.logprob = -0.1
    tok2 = MagicMock()
    tok2.token = " sky"
    tok2.logprob = -0.2
    tok3 = MagicMock()
    tok3.token = " is"
    tok3.logprob = -0.15
    tok4 = MagicMock()
    tok4.token = " blue"
    tok4.logprob = -0.1

    lp_content = MagicMock()
    lp_content.content = [tok1, tok2, tok3, tok4]
    choice = MagicMock()
    choice.logprobs = lp_content
    choice.message.content = "The sky is blue."

    fake_response = MagicMock()
    fake_response.choices = [choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=fake_response)

    mock_openai_module = MagicMock()
    mock_openai_module.AsyncOpenAI.return_value = mock_client

    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        patch.dict(sys.modules, {"openai": mock_openai_module}),
    ):
        result = await check_logprobs("What colour is the sky?", ["The sky is blue."])

    assert len(result.claims) == 1
    assert result.claims[0].method == "logprobs"
    assert 0.0 <= result.raw_score <= 1.0
