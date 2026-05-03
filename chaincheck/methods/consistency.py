"""Self-consistency detection via sampled LLM responses."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from functools import lru_cache
from typing import Any

import numpy as np
from diskcache import Cache
from pydantic import BaseModel

from chaincheck.config import cache_path
from chaincheck.llm import LLMError, complete

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class ConsistencyResult(BaseModel):
    """Self-consistency result for an input response."""

    consistency_score: float
    similarity_matrix: list[list[float]]
    sampled_response_count: int
    inconsistent: bool


async def check_consistency(
    response: str,
    prompt: str,
    sample_count: int = 5,
    threshold: float = 0.82,
    model: str | None = None,
) -> ConsistencyResult:
    """Compare a response with parallel sampled responses for the same prompt."""
    sample_model = model or os.getenv("CONSISTENCY_MODEL", "claude-haiku-4-5")
    samples = await sample_responses(prompt, sample_count, sample_model)
    texts = [response, *samples]
    embeddings = np.array([embedding_for_text(text) for text in texts])
    matrix = cosine_matrix(embeddings)
    score = _mean_similarity_to_samples(matrix)
    return ConsistencyResult(
        consistency_score=score,
        similarity_matrix=matrix.tolist(),
        sampled_response_count=len(samples),
        inconsistent=score < threshold,
    )


async def sample_responses(prompt: str, sample_count: int, model: str) -> list[str]:
    """Sample parallel responses from a configured LLM endpoint."""
    if not prompt.strip() or sample_count <= 0:
        return []
    tasks = [_sample_one(prompt, model, index) for index in range(sample_count)]
    return list(await asyncio.gather(*tasks))


def cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute a cosine similarity matrix for embeddings."""
    if embeddings.size == 0:
        return np.zeros((0, 0))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    normalized = embeddings / safe
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def embedding_for_text(text: str) -> list[float]:
    """Return a cached embedding for text."""
    key = _embedding_key(text)
    cache = _cache()
    cached = cache.get(key)
    if isinstance(cached, list):
        return [float(value) for value in cached]
    embedding = _compute_embedding(text)
    cache.set(key, embedding)
    return embedding


def preload_model() -> None:
    """Download and initialize the embedding model."""
    _embedding_model()


async def _sample_one(prompt: str, model: str, index: int) -> str:
    sampling_prompt = f"{prompt}\n\nGive a concise answer. Sample index: {index}."
    try:
        return await complete(sampling_prompt, model=model, max_tokens=800)
    except LLMError:
        return prompt


def _mean_similarity_to_samples(matrix: np.ndarray) -> float:
    if matrix.shape[0] <= 1:
        return 1.0
    return float(np.mean(matrix[0, 1:]))


def _compute_embedding(text: str) -> list[float]:
    try:
        model = _embedding_model()
        vector = model.encode([text], normalize_embeddings=True)[0]
        return [float(value) for value in vector]
    except (ImportError, OSError, RuntimeError, ValueError):
        return _lexical_embedding(text)


def _lexical_embedding(text: str, dimensions: int = 384) -> list[float]:
    vector = np.zeros(dimensions, dtype=float)
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        index = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % dimensions
        vector[index] += 1.0
    norm = np.linalg.norm(vector)
    return (vector / norm).tolist() if norm else vector.tolist()


@lru_cache(maxsize=1)
def _embedding_model() -> Any:
    if os.getenv("CHAINCHECK_DISABLE_MODELS") == "1":
        raise RuntimeError("Model loading disabled")
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _embedding_key(text: str) -> str:
    return f"embedding:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _cache() -> Cache:
    return Cache(str(cache_path() / "embeddings"))
