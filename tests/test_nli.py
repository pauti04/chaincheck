"""Tests for NLI detection."""

import pytest

from chaincheck.methods import nli
from chaincheck.methods.nli import check_claims_nli, context_sentences


@pytest.mark.asyncio
async def test_nli_lexical_fallback_returns_claims(monkeypatch: pytest.MonkeyPatch) -> None:
    """NLI returns one result per claim when model loading is disabled."""
    monkeypatch.setenv("CHAINCHECK_DISABLE_MODELS", "1")
    results = await check_claims_nli(["A claim."], "Context.")
    assert results[0].claim == "A claim."
    assert results[0].label in {"entailed", "neutral", "contradicted"}


def test_context_sentences_splits_context() -> None:
    """Context splitting keeps sentence-sized evidence."""
    assert context_sentences("A. B!") == ["A.", "B!"]


def test_nli_cache_roundtrip(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """NLI cache keys preserve stored result dictionaries."""
    monkeypatch.setenv("CACHE_PATH", str(tmp_path))
    result = nli._lexical_result("Paris is in France", ["Paris is in France."], "")
    nli._write_cached([result], "Paris is in France.")
    cached, missing = nli._read_cached(["Paris is in France"], "Paris is in France.")
    assert not missing
    assert cached["Paris is in France"].claim == "Paris is in France"
