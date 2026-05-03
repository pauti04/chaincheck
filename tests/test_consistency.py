"""Tests for chaincheck.methods.consistency — self-consistency detection."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from chaincheck.methods.consistency import (
    _cosine_similarity_matrix,
    _text_hash,
)


class TestCosineSimilarityMatrix:
    def test_shape(self):
        embeddings = np.random.rand(4, 64).astype(np.float32)
        m = _cosine_similarity_matrix(embeddings)
        assert m.shape == (4, 4)

    def test_diagonal_is_one(self):
        embeddings = np.random.rand(4, 64).astype(np.float32)
        m = _cosine_similarity_matrix(embeddings)
        np.testing.assert_allclose(np.diag(m), 1.0, atol=1e-5)

    def test_identical_vectors(self):
        v = np.ones((2, 8), dtype=np.float32)
        m = _cosine_similarity_matrix(v)
        assert m[0, 1] == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        m = _cosine_similarity_matrix(a)
        assert m[0, 1] == pytest.approx(0.0, abs=1e-5)

    def test_symmetric(self):
        embeddings = np.random.rand(5, 32).astype(np.float32)
        m = _cosine_similarity_matrix(embeddings)
        np.testing.assert_allclose(m, m.T, atol=1e-5)

    def test_values_in_range(self):
        embeddings = np.random.rand(6, 16).astype(np.float32)
        m = _cosine_similarity_matrix(embeddings)
        assert np.all(m >= -1.0 - 1e-5)
        assert np.all(m <= 1.0 + 1e-5)


class TestTextHash:
    def test_deterministic(self):
        assert _text_hash("hello") == _text_hash("hello")

    def test_different_inputs_differ(self):
        assert _text_hash("hello") != _text_hash("world")

    def test_returns_hex_string(self):
        h = _text_hash("test")
        assert len(h) == 64
        int(h, 16)


@pytest.mark.asyncio
async def test_check_consistency_returns_result():
    """check_consistency() should return a ConsistencyResult with correct shape."""
    from chaincheck.methods.consistency import check_consistency

    fake_embs = np.eye(3, dtype=np.float32)  # 3 texts, 3-dim embeddings

    _target = "chaincheck.methods.consistency._sample_responses"
    with (
        patch(_target, new=AsyncMock(return_value=["r1", "r2"])),
        patch("chaincheck.methods.consistency._embed_texts", return_value=fake_embs),
    ):
        result = await check_consistency("What is X?", "X is Y.", n_samples=2)

    assert 0.0 <= result.consistency_score <= 1.0
    assert result.sample_count == 2
    assert len(result.similarity_matrix) == 3


@pytest.mark.asyncio
async def test_check_consistency_identical_responses_score_one():
    """Identical response and samples should yield consistency_score ≈ 1.0."""
    from chaincheck.methods.consistency import check_consistency

    text = "The answer is 42."
    fake_embs = np.ones((3, 4), dtype=np.float32)

    _target = "chaincheck.methods.consistency._sample_responses"
    with (
        patch(_target, new=AsyncMock(return_value=[text, text])),
        patch("chaincheck.methods.consistency._embed_texts", return_value=fake_embs),
    ):
        result = await check_consistency("prompt", text, n_samples=2)

    assert result.consistency_score == pytest.approx(1.0, abs=1e-4)
