"""
Self-consistency detection method.

Samples N responses from a configurable LLM endpoint in parallel via
asyncio + httpx, embeds them with all-MiniLM-L6-v2, and computes a pairwise
cosine similarity matrix. Low mean similarity to the input response signals
likely hallucination.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from typing import TYPE_CHECKING

import numpy as np

from chaincheck.models import ConsistencyResult

if TYPE_CHECKING:
    import diskcache

_CONSISTENCY_MODEL = os.getenv("CONSISTENCY_MODEL", "claude-haiku-4-5-20251001")
_CONSISTENCY_SAMPLES = int(os.getenv("CONSISTENCY_SAMPLES", "5"))
_CONSISTENCY_THRESHOLD = float(os.getenv("CONSISTENCY_THRESHOLD", "0.82"))
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_embed_model = None
_embed_cache: diskcache.Cache | None = None


def _get_embed_model():
    """Lazily load the sentence-transformer embedding model (once per process)."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model


def _get_embed_cache() -> diskcache.Cache:
    """Lazily initialise the embedding disk cache."""
    global _embed_cache
    if _embed_cache is None:
        import diskcache

        base = os.getenv("CACHE_PATH", ".chaincheck_cache")
        _embed_cache = diskcache.Cache(f"{base}_embeddings")
    return _embed_cache


async def check_consistency(
    prompt: str,
    response: str,
    n_samples: int = _CONSISTENCY_SAMPLES,
) -> ConsistencyResult:
    """
    Measure self-consistency of a response by sampling alternatives.

    Args:
        prompt: The original prompt that generated the response.
        response: The LLM output to evaluate.
        n_samples: Number of additional LLM samples to draw.

    Returns:
        ConsistencyResult with consistency score and full similarity matrix.
        consistency_score < _CONSISTENCY_THRESHOLD signals likely inconsistency.
    """
    start = time.time()
    samples = await _sample_responses(prompt, n_samples, _CONSISTENCY_MODEL)
    all_texts = [response] + samples

    embeddings = _embed_texts(all_texts)
    sim_matrix = _cosine_similarity_matrix(embeddings)

    # Row 0 is the input response; columns 1..n are the samples
    response_sims = sim_matrix[0, 1:]
    consistency_score = float(np.mean(response_sims)) if len(response_sims) > 0 else 1.0

    return ConsistencyResult(
        consistency_score=consistency_score,
        similarity_matrix=sim_matrix.tolist(),
        sample_count=n_samples,
        latency_ms=(time.time() - start) * 1000,
    )


async def _sample_responses(
    prompt: str,
    n: int,
    model: str,
) -> list[str]:
    """
    Draw n independent LLM responses for the given prompt in parallel.

    Routes through llm.complete() so any provider (Anthropic, OpenAI, Ollama)
    can be used by setting CONSISTENCY_MODEL to the appropriate model ID.
    """
    from chaincheck.llm import complete

    return list(await asyncio.gather(*[complete(prompt, model, max_tokens=512) for _ in range(n)]))


def _embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts using the sentence-transformer model.

    Caches embeddings by SHA-256 hash of each text to avoid redundant
    computation across calls.

    Returns:
        Float32 array of shape (len(texts), embedding_dim).
    """
    model = _get_embed_model()
    cache = _get_embed_cache()

    result: list[tuple[int, np.ndarray]] = []
    to_encode: list[str] = []
    to_encode_indices: list[int] = []

    for i, text in enumerate(texts):
        key = _text_hash(text)
        cached = cache.get(key)
        if cached is not None:
            result.append((i, np.array(cached, dtype=np.float32)))
        else:
            to_encode.append(text)
            to_encode_indices.append(i)

    if to_encode:
        new_embeddings = model.encode(to_encode, convert_to_numpy=True)
        for idx, text, emb in zip(to_encode_indices, to_encode, new_embeddings, strict=True):
            cache.set(_text_hash(text), emb.tolist())
            result.append((idx, emb.astype(np.float32)))

    result.sort(key=lambda x: x[0])
    return np.array([r[1] for r in result], dtype=np.float32)


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute the full pairwise cosine similarity matrix.

    Args:
        embeddings: 2-D float32 array of shape (n, d).

    Returns:
        Square float32 array of shape (n, n) with 1s on the diagonal.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.clip(norms, 1e-8, None)
    return (normalized @ normalized.T).astype(np.float32)


def _text_hash(text: str) -> str:
    """Return SHA-256 hex digest for use as an embedding cache key."""
    return hashlib.sha256(text.encode()).hexdigest()
