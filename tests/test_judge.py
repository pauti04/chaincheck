"""Tests for judge detection."""

import pytest

from chaincheck.methods import judge
from chaincheck.methods.judge import default_model, judge_claims


@pytest.mark.asyncio
async def test_judge_returns_claims_from_structured_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Judge parses structured JSON and restores claim order."""
    monkeypatch.setattr(judge, "complete", _fake_complete)
    results = await judge_claims(["A claim."], "Context.")
    assert results[0].claim == "A claim."
    assert results[0].label == "supported"


@pytest.mark.asyncio
async def test_judge_fallback_after_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Judge falls back after malformed provider output."""
    monkeypatch.setattr(judge, "complete", _bad_complete)
    results = await judge_claims(["Context."], "Context.")
    assert results[0].label == "supported"


def test_default_model_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Judge model default is environment configurable."""
    monkeypatch.setenv("JUDGE_MODEL", "ollama:llama3")
    assert default_model() == "ollama:llama3"


async def _fake_complete(prompt: str, model: str, max_tokens: int) -> str:
    del prompt, model, max_tokens
    return '{"label":"supported","confidence":0.9,"evidence":"Context."}'


async def _bad_complete(prompt: str, model: str, max_tokens: int) -> str:
    del prompt, model, max_tokens
    return "not json"
