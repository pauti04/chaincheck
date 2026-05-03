"""Tests for ChainCheck detection orchestration."""

import pytest

import chaincheck.detect as detect_module
from chaincheck.detect import DetectionResult, detect
from chaincheck.methods.consistency import ConsistencyResult
from chaincheck.methods.judge import JudgeClaimResult
from chaincheck.methods.nli import NLIClaimResult


def test_detection_result_imports() -> None:
    """DetectionResult can be imported."""

    assert DetectionResult(response="x").response == "x"


@pytest.mark.asyncio
async def test_detect_aggregates_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detection orchestrates methods and returns a risk score."""
    monkeypatch.setattr(detect_module, "decompose", _fake_decompose)
    monkeypatch.setattr(detect_module, "check_claims_nli", _fake_nli)
    monkeypatch.setattr(detect_module, "judge_claims", _fake_judge)
    monkeypatch.setattr(detect_module, "check_consistency", _fake_consistency)
    result = await detect("Response.", "Context.", "Prompt.")
    assert result.claims == ["Claim one."]
    assert result.aggregate_score < 0.5


@pytest.mark.asyncio
async def test_detect_rejects_unknown_method() -> None:
    """Detection rejects unknown method names."""
    with pytest.raises(ValueError, match="Unknown detection method"):
        await detect("Response.", methods=["unknown"])


async def _fake_decompose(response: str) -> list[str]:
    del response
    return ["Claim one."]


async def _fake_nli(claims: list[str], context: str) -> list[NLIClaimResult]:
    del context
    return [NLIClaimResult(claim=claims[0], label="entailed", confidence=0.9, evidence="e")]


async def _fake_judge(
    claims: list[str], context: str, model: str = "claude-haiku-4-5"
) -> list[JudgeClaimResult]:
    del context, model
    return [JudgeClaimResult(claim=claims[0], label="supported", confidence=0.8, evidence="e")]


async def _fake_consistency(
    response: str, prompt: str, sample_count: int, threshold: float, model: str | None = None
) -> ConsistencyResult:
    del response, prompt, sample_count, threshold, model
    return ConsistencyResult(
        consistency_score=0.9,
        similarity_matrix=[[1.0]],
        sampled_response_count=0,
        inconsistent=False,
    )
