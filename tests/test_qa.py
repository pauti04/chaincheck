"""Tests for chaincheck.methods.qa — QA-based claim verification."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from chaincheck.models import ClaimResult


class TestScoreFromClaims:
    def test_all_supported(self):
        from chaincheck.methods.qa import _score_from_claims

        claims = [ClaimResult(claim="c", label="supported", confidence=0.9,
                              evidence="", method="qa")]
        assert _score_from_claims(claims) == pytest.approx(0.0, abs=1e-6)

    def test_all_contradicted(self):
        from chaincheck.methods.qa import _score_from_claims

        claims = [ClaimResult(claim="c", label="contradicted", confidence=0.9,
                              evidence="", method="qa")]
        assert _score_from_claims(claims) == pytest.approx(1.0, abs=1e-6)

    def test_empty(self):
        from chaincheck.methods.qa import _score_from_claims

        assert _score_from_claims([]) == 0.0

    def test_mixed(self):
        from chaincheck.methods.qa import _score_from_claims

        claims = [
            ClaimResult(claim="a", label="supported", confidence=0.9, evidence="", method="qa"),
            ClaimResult(claim="b", label="contradicted", confidence=0.9, evidence="", method="qa"),
        ]
        score = _score_from_claims(claims)
        assert 0.0 < score < 1.0


@pytest.mark.asyncio
async def test_check_qa_empty_claims():
    from chaincheck.methods.qa import check_qa

    result = await check_qa([], "some context")
    assert result.raw_score == 0.0
    assert result.claims == []


@pytest.mark.asyncio
async def test_check_qa_no_context_returns_unknown():
    from chaincheck.methods.qa import check_qa

    result = await check_qa(["The sky is blue."], "")
    assert all(c.label == "unknown" for c in result.claims)
    assert result.raw_score == 0.0


@pytest.mark.asyncio
async def test_check_qa_yes_answer_gives_supported():
    from chaincheck.methods.qa import check_qa

    with patch("chaincheck.methods.qa._ask_yn", new=AsyncMock(return_value=("supported", 0.9))):
        result = await check_qa(["The sky is blue."], "The sky is blue.")

    assert result.claims[0].label == "supported"
    assert result.raw_score == pytest.approx(0.0, abs=1e-6)


@pytest.mark.asyncio
async def test_check_qa_no_answer_gives_contradicted():
    from chaincheck.methods.qa import check_qa

    with patch("chaincheck.methods.qa._ask_yn", new=AsyncMock(return_value=("contradicted", 0.9))):
        result = await check_qa(["Water boils at 50C."], "Water boils at 100C.")

    assert result.claims[0].label == "contradicted"
    assert result.raw_score == pytest.approx(1.0, abs=1e-6)


@pytest.mark.asyncio
async def test_check_qa_preserves_claim_order():
    from chaincheck.methods.qa import check_qa

    claims = ["Claim A.", "Claim B.", "Claim C."]
    labels = {"Claim A.": "supported", "Claim B.": "contradicted", "Claim C.": "supported"}

    async def _fake_ask(claim, context, model):
        return labels[claim], 0.9

    with patch("chaincheck.methods.qa._ask_yn", side_effect=_fake_ask):
        result = await check_qa(claims, "context")

    assert [c.claim for c in result.claims] == claims
    assert result.claims[0].label == "supported"
    assert result.claims[1].label == "contradicted"


@pytest.mark.asyncio
async def test_check_qa_returns_correct_count():
    from chaincheck.methods.qa import check_qa

    with patch("chaincheck.methods.qa._ask_yn", new=AsyncMock(return_value=("supported", 0.9))):
        result = await check_qa(["A.", "B.", "C."], "ctx")

    assert len(result.claims) == 3


@pytest.mark.asyncio
async def test_ask_yn_yes():
    from chaincheck.methods.qa import _ask_yn

    with patch("chaincheck.llm.complete", new=AsyncMock(return_value="yes")):
        label, conf = await _ask_yn("claim", "context", "gpt-4o-mini")

    assert label == "supported"
    assert conf == 0.9


@pytest.mark.asyncio
async def test_ask_yn_no():
    from chaincheck.methods.qa import _ask_yn

    with patch("chaincheck.llm.complete", new=AsyncMock(return_value="no")):
        label, conf = await _ask_yn("claim", "context", "gpt-4o-mini")

    assert label == "contradicted"
    assert conf == 0.9


@pytest.mark.asyncio
async def test_ask_yn_unknown_on_llm_error():
    from chaincheck.llm import LLMError
    from chaincheck.methods.qa import _ask_yn

    with patch("chaincheck.llm.complete", side_effect=LLMError("fail")):
        label, conf = await _ask_yn("claim", "context", "gpt-4o-mini")

    assert label == "unknown"
    assert conf == 0.0
