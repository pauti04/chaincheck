"""
Atomic claim extraction from LLM responses.

Uses a Claude model to decompose free-text into self-contained, verifiable
factual assertions. Results are cached by SHA-256 hash of the input with a
24-hour TTL to avoid redundant API calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import diskcache

_CACHE_PATH = os.getenv("CACHE_PATH", ".chaincheck_cache")
_CACHE: diskcache.Cache | None = None
_DECOMPOSE_MODEL = os.getenv("DECOMPOSE_MODEL", "gpt-4o-mini")
_CACHE_TTL = 60 * 60 * 24  # 24 hours


def _get_cache() -> diskcache.Cache:
    """Lazily initialise the disk cache singleton."""
    global _CACHE
    if _CACHE is None:
        import diskcache

        _CACHE = diskcache.Cache(_CACHE_PATH)
    return _CACHE


def _response_hash(response: str) -> str:
    """Return SHA-256 hex digest of a response string for use as a cache key."""
    return hashlib.sha256(response.encode()).hexdigest()


async def decompose(
    response: str,
    model: str = _DECOMPOSE_MODEL,
) -> list[str]:
    """
    Extract atomic factual claims from a free-text LLM response.

    Each claim is self-contained (no pronouns), verifiable, and represents
    a single assertion. Results are cached for 24 h by response hash.

    Args:
        response: The LLM output to decompose.
        model: Model identifier passed to llm.complete() — supports Anthropic,
               OpenAI (gpt-*), and Ollama (ollama:<name>).

    Returns:
        Deduplicated list of atomic claim strings, each at least 10 chars.
    """
    from chaincheck.llm import LLMError, complete

    cache = _get_cache()
    key = _response_hash(response)
    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    prompt = _build_decompose_prompt(response)

    try:
        text = await complete(prompt, model, max_tokens=1024)
        text = text.strip()
        # Strip markdown code fences if the model wraps output
        if text.startswith("```"):
            text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        raw = json.loads(text)
        if not isinstance(raw, list):
            raise ValueError("Expected JSON array")
        claims = _postprocess_claims([str(c) for c in raw])
    except (json.JSONDecodeError, ValueError, IndexError, AttributeError, LLMError):
        claims = _sentence_split_fallback(response)

    # If the model returned nothing but the response has content, use sentence split.
    # This handles terse answers (e.g. "Nothing happens") that the model skips over.
    if not claims and len(response.strip()) >= 15:
        claims = _sentence_split_fallback(response)

    cache.set(key, claims, expire=_CACHE_TTL)
    return claims


def _sentence_split_fallback(response: str) -> list[str]:
    """
    Fallback claim extractor using simple sentence splitting.

    Used when the LLM returns malformed JSON. Splits on sentence-ending
    punctuation and filters fragments shorter than 10 characters.
    """
    if not response.strip():
        return []
    sentences = re.split(r"(?<=[.!?])\s+", response.strip())
    return _postprocess_claims(sentences)


def _postprocess_claims(raw: list[str]) -> list[str]:
    """
    Deduplicate, filter short fragments, and strip numbering from raw claims.

    Args:
        raw: Unprocessed claim strings from the LLM or fallback splitter.

    Returns:
        Cleaned, deduplicated list of claim strings.
    """
    seen: set[str] = set()
    result: list[str] = []
    for claim in raw:
        # Strip leading "1.", "2)", "- ", "• ", etc.
        claim = re.sub(r"^\s*\d+[.)]\s*", "", claim).strip()
        claim = re.sub(r"^\s*[-*•]\s*", "", claim).strip()
        if len(claim) < 10:
            continue
        if claim in seen:
            continue
        seen.add(claim)
        result.append(claim)
    return result


def _build_decompose_prompt(response: str) -> str:
    """Return the user prompt for atomic claim extraction."""
    return (
        "Extract every atomic factual assertion from the text below.\n\n"
        "Rules:\n"
        "- Replace all pronouns with the noun they refer to (each claim must stand alone)\n"
        "- Each claim is ONE verifiable fact — no compound claims joined by 'and'\n"
        "- Keep contrastive statements as ONE claim: 'X is A, not B' → 'X is A, not B'\n"
        "- Do NOT split on negations: 'A not B' is a single assertion about A\n"
        "- Exclude opinions, hedges (might/could/seems), and meta-commentary\n"
        "- If the entire input is a single short assertion, return it as one claim\n"
        "- Respond with ONLY a valid JSON array of strings, no other text\n\n"
        f"Text:\n{response}\n\n"
        "JSON array:"
    )
