"""Tests for claim decomposition."""

import pytest

import chaincheck.decompose as decompose_module
from chaincheck.decompose import decompose, fallback_sentence_split, postprocess_claims


@pytest.mark.asyncio
async def test_decompose_falls_back_when_llm_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decomposition falls back to sentence splitting on LLM errors."""
    monkeypatch.setattr(decompose_module, "_cache", _MemoryCache)
    claims = await decompose("Paris is in France. Rome is in Italy.")
    assert claims == ["Paris is in France.", "Rome is in Italy."]


def test_postprocess_claims_deduplicates_and_strips_numbering() -> None:
    """Claim post-processing strips numbering and removes duplicates."""
    claims = postprocess_claims(["1. Paris is in France.", "Paris is in France.", "tiny"])
    assert claims == ["Paris is in France."]


def test_fallback_sentence_split() -> None:
    """Fallback splitting returns sentence candidates."""
    assert fallback_sentence_split("One fact. Another fact!") == ["One fact.", "Another fact!"]


class _MemoryCache(dict):
    def set(self, key: str, value: list[str], expire: int) -> None:
        del expire
        self[key] = value
