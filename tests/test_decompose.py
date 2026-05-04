"""Tests for chaincheck.decompose — claim extraction, caching, and postprocessing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chaincheck.decompose import _postprocess_claims, _sentence_split_fallback


class TestPostprocessClaims:
    def test_removes_short_claims(self):
        claims = ["short", "This is a proper verifiable claim.", "ok"]
        result = _postprocess_claims(claims)
        assert all(len(c) >= 10 for c in result)
        assert "short" not in result

    def test_deduplicates(self):
        claims = ["Same claim here.", "Same claim here.", "Different claim here."]
        result = _postprocess_claims(claims)
        assert len(result) == 2

    def test_strips_leading_numbering_period(self):
        claims = ["1. The sky is blue.", "2. Water is wet."]
        result = _postprocess_claims(claims)
        assert all(not c[0].isdigit() for c in result)

    def test_strips_leading_numbering_paren(self):
        claims = ["1) First claim here.", "2) Second claim here."]
        result = _postprocess_claims(claims)
        assert all(not c[0].isdigit() for c in result)

    def test_strips_bullet_dash(self):
        claims = ["- The Earth is round.", "* Stars are hot."]
        result = _postprocess_claims(claims)
        assert all(c[0] not in "-*" for c in result)

    def test_empty_input(self):
        assert _postprocess_claims([]) == []

    def test_all_short_returns_empty(self):
        assert _postprocess_claims(["a", "bb", "ccc"]) == []

    def test_preserves_order(self):
        claims = ["Alpha claim is here.", "Beta claim is here.", "Gamma claim is here."]
        result = _postprocess_claims(claims)
        assert result == claims


class TestSentenceSplitFallback:
    def test_splits_on_periods(self):
        text = "The sky is blue. Water is wet. The sun is a star."
        result = _sentence_split_fallback(text)
        assert len(result) >= 3

    def test_splits_on_exclamation(self):
        text = "The Earth orbits the Sun! Mars is red."
        result = _sentence_split_fallback(text)
        assert len(result) >= 2

    def test_filters_short_fragments(self):
        text = "A. The Earth orbits the Sun. B."
        result = _sentence_split_fallback(text)
        assert all(len(c) >= 10 for c in result)

    def test_empty_string(self):
        assert _sentence_split_fallback("") == []

    def test_whitespace_only(self):
        assert _sentence_split_fallback("   ") == []


@pytest.mark.asyncio
async def test_decompose_returns_list_of_strings():
    """decompose() should return a list of strings when the API returns valid JSON."""
    from chaincheck.decompose import decompose

    fake_json = '["The Eiffel Tower is in Paris.", "It was built in 1889."]'
    with (
        patch("chaincheck.decompose._get_cache") as mock_cache_fn,
        patch("chaincheck.llm.complete", new=AsyncMock(return_value=fake_json)),
    ):
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache.set = MagicMock()
        mock_cache_fn.return_value = mock_cache

        result = await decompose("The Eiffel Tower is in Paris, built in 1889.")

    assert isinstance(result, list)
    assert all(isinstance(c, str) for c in result)
    assert len(result) >= 1


@pytest.mark.asyncio
async def test_decompose_falls_back_on_bad_json():
    """decompose() should use sentence_split_fallback when JSON is malformed."""
    from chaincheck.decompose import decompose

    with (
        patch("chaincheck.decompose._get_cache") as mock_cache_fn,
        patch("chaincheck.llm.complete", new=AsyncMock(return_value="not json at all")),
    ):
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache.set = MagicMock()
        mock_cache_fn.return_value = mock_cache

        result = await decompose("The sky is blue. Water is wet.")

    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_decompose_uses_cache_on_second_call():
    """Second call with identical input must not make an API call."""
    from chaincheck.decompose import decompose

    cached_claims = ["The sky is blue."]

    with patch("chaincheck.decompose._get_cache") as mock_cache_fn:
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached_claims
        mock_cache_fn.return_value = mock_cache

        with patch("chaincheck.llm.complete", new=AsyncMock()) as mock_complete:
            result = await decompose("The sky is blue.")
            mock_complete.assert_not_called()

    assert result == cached_claims


@pytest.mark.asyncio
async def test_decompose_secondary_fallback_on_empty_llm_result():
    """When the LLM returns [] for a long-enough response, sentence split is used."""
    from chaincheck.decompose import decompose

    with (
        patch("chaincheck.decompose._get_cache") as mock_cache_fn,
        patch("chaincheck.llm.complete", new=AsyncMock(return_value="[]")),
    ):
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache.set = MagicMock()
        mock_cache_fn.return_value = mock_cache

        result = await decompose("The sky is blue. Water is wet.")

    assert len(result) >= 1


@pytest.mark.asyncio
async def test_decompose_strips_markdown_code_fence():
    """decompose() strips ```json ... ``` wrappers before parsing."""
    from chaincheck.decompose import decompose

    fenced = '```json\n["The sun is a star."]\n```'

    with (
        patch("chaincheck.decompose._get_cache") as mock_cache_fn,
        patch("chaincheck.llm.complete", new=AsyncMock(return_value=fenced)),
    ):
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache.set = MagicMock()
        mock_cache_fn.return_value = mock_cache

        result = await decompose("The sun is our nearest star.")

    assert len(result) >= 1
    assert all(isinstance(c, str) for c in result)


@pytest.mark.asyncio
async def test_decompose_raises_on_non_list_json():
    """decompose() falls back to sentence split when JSON is not an array."""
    from chaincheck.decompose import decompose

    with (
        patch("chaincheck.decompose._get_cache") as mock_cache_fn,
        patch("chaincheck.llm.complete", new=AsyncMock(return_value='{"claim": "x"}')),
    ):
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache.set = MagicMock()
        mock_cache_fn.return_value = mock_cache

        result = await decompose("The sky is blue. Water is wet.")

    assert isinstance(result, list)
