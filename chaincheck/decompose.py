"""Atomic claim decomposition for ChainCheck."""

from __future__ import annotations

import hashlib
import json
import re

from diskcache import Cache
from pydantic import TypeAdapter, ValidationError

from chaincheck.config import cache_path
from chaincheck.llm import LLMError, complete

_CLAIMS_ADAPTER = TypeAdapter(list[str])
_CACHE_TTL_SECONDS = 60 * 60 * 24


async def decompose(response: str, model: str = "claude-haiku-4-5") -> list[str]:
    """Extract atomic factual claims from an LLM response."""
    if not response.strip():
        return []
    cache = _cache()
    key = _cache_key(response, model)
    cached = cache.get(key)
    if isinstance(cached, list):
        return [str(item) for item in cached]
    claims = await _extract_claims(response, model)
    cache.set(key, claims, expire=_CACHE_TTL_SECONDS)
    return claims


def postprocess_claims(claims: list[str]) -> list[str]:
    """Normalize, filter, and deduplicate extracted claims."""
    seen: set[str] = set()
    processed: list[str] = []
    for claim in claims:
        cleaned = _strip_numbering(claim)
        if len(cleaned) < 10 or cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        processed.append(cleaned)
    return processed


async def _extract_claims(response: str, model: str) -> list[str]:
    prompt = _decomposition_prompt(response)
    try:
        raw = await complete(prompt, model=model, max_tokens=1200)
        parsed = _parse_claim_json(raw)
    except (LLMError, json.JSONDecodeError, ValidationError):
        parsed = fallback_sentence_split(response)
    return postprocess_claims(parsed)


def fallback_sentence_split(response: str) -> list[str]:
    """Split response into simple sentence-like factual claim candidates."""
    normalized = re.sub(r"\s+", " ", response.strip())
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part.strip(" -\t\n") for part in parts if part.strip()]


def _parse_claim_json(raw: str) -> list[str]:
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("JSON array not found", raw, 0)
    return _CLAIMS_ADAPTER.validate_python(json.loads(raw[start : end + 1]))


def _decomposition_prompt(response: str) -> str:
    return (
        "Extract every atomic factual assertion from the response as a JSON array of strings.\n"
        "Each claim must be self-contained with no pronouns, verifiable, and a single assertion.\n"
        "Return only JSON, with no markdown.\n\n"
        f"Response:\n{response}"
    )


def _strip_numbering(claim: str) -> str:
    cleaned = re.sub(r"^\s*(?:[-*]|\d+[\).:-])\s*", "", claim.strip())
    return cleaned.strip().strip('"').strip("'").strip()


def _cache_key(response: str, model: str) -> str:
    digest = hashlib.sha256(f"{model}\0{response}".encode()).hexdigest()
    return f"decompose:{digest}"


def _cache() -> Cache:
    return Cache(str(cache_path() / "decompose"))
