"""Tests for consistency detection."""

import numpy as np
import pytest

from chaincheck.methods import consistency
from chaincheck.methods.consistency import check_consistency, cosine_matrix, embedding_for_text


@pytest.mark.asyncio
async def test_consistency_counts_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    """Consistency preserves sample count and computes a matrix."""
    monkeypatch.setenv("CHAINCHECK_DISABLE_MODELS", "1")
    monkeypatch.setattr(consistency, "sample_responses", _fake_samples)
    result = await check_consistency("Response.", "Prompt.", sample_count=3)
    assert result.sampled_response_count == 3
    assert len(result.similarity_matrix) == 4


def test_cosine_matrix_identity() -> None:
    """Cosine matrix has ones on the diagonal for nonzero vectors."""
    matrix = cosine_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert matrix[0, 0] == 1.0


def test_embedding_for_text_lexical_fallback(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedding fallback returns deterministic dense vectors."""
    monkeypatch.setenv("CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("CHAINCHECK_DISABLE_MODELS", "1")
    assert embedding_for_text("hello") == embedding_for_text("hello")


async def _fake_samples(prompt: str, sample_count: int, model: str) -> list[str]:
    del model
    return [prompt for _ in range(sample_count)]
