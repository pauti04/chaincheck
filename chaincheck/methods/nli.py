"""NLI entailment detection using a cross-encoder model."""

from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from typing import Any

from diskcache import Cache
from pydantic import BaseModel

from chaincheck.config import cache_path, env_float, env_int

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
_LABELS = ("contradicted", "entailed", "neutral")


class NLIClaimResult(BaseModel):
    """NLI result for one atomic claim."""

    claim: str
    label: str
    confidence: float
    evidence: str


async def check_claims_nli(claims: list[str], context: str) -> list[NLIClaimResult]:
    """Score claims against context with NLI entailment."""
    if not claims:
        return []
    sentences = context_sentences(context)
    cached, missing = _read_cached(claims, context)
    if missing:
        computed = _score_missing(missing, sentences, context)
        _write_cached(computed, context)
        cached.update({result.claim: result for result in computed})
    return [cached[claim] for claim in claims]


def context_sentences(context: str) -> list[str]:
    """Split context into evidence-sized sentences."""
    normalized = re.sub(r"\s+", " ", context.strip())
    if not normalized:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]


def preload_model() -> None:
    """Download and initialize the NLI cross-encoder model."""
    _model()


def _score_missing(claims: list[str], sentences: list[str], context: str) -> list[NLIClaimResult]:
    try:
        return _score_with_cross_encoder(claims, sentences, context)
    except (ImportError, OSError, RuntimeError, ValueError):
        return [_lexical_result(claim, sentences, context) for claim in claims]


def _score_with_cross_encoder(
    claims: list[str], sentences: list[str], context: str
) -> list[NLIClaimResult]:
    model = _model()
    pairs = [(_best_evidence(claim, sentences, context), claim) for claim in claims]
    raw_scores = model.predict(
        pairs, batch_size=env_int("NLI_BATCH_SIZE", 16), convert_to_numpy=True
    )
    return [
        _result_from_scores(claim, evidence, scores)
        for (evidence, claim), scores in zip(pairs, raw_scores, strict=False)
    ]


def _result_from_scores(claim: str, evidence: str, scores: Any) -> NLIClaimResult:
    probs = _softmax([float(score) for score in scores])
    index = max(range(len(probs)), key=probs.__getitem__)
    label = _LABELS[index] if probs[index] >= env_float("NLI_THRESHOLD", 0.5) else "neutral"
    return NLIClaimResult(claim=claim, label=label, confidence=probs[index], evidence=evidence)


def _lexical_result(claim: str, sentences: list[str], context: str) -> NLIClaimResult:
    evidence = _best_evidence(claim, sentences, context)
    overlap = _token_overlap(claim, evidence)
    label = "entailed" if overlap >= 0.6 else "neutral" if overlap >= 0.25 else "contradicted"
    confidence = min(0.95, max(0.35, overlap))
    return NLIClaimResult(claim=claim, label=label, confidence=confidence, evidence=evidence)


def _best_evidence(claim: str, sentences: list[str], context: str) -> str:
    if not sentences:
        return context[:500]
    return max(sentences, key=lambda sentence: _token_overlap(claim, sentence))


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]+", left.lower()))
    right_tokens = set(re.findall(r"[a-z0-9]+", right.lower()))
    if not left_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _softmax(scores: list[float]) -> list[float]:
    import math

    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]


@lru_cache(maxsize=1)
def _model() -> Any:
    if os.getenv("CHAINCHECK_DISABLE_MODELS") == "1":
        raise RuntimeError("Model loading disabled")
    from sentence_transformers import CrossEncoder

    return CrossEncoder(NLI_MODEL_NAME)


def _read_cached(
    claims: list[str], context: str
) -> tuple[dict[str, NLIClaimResult], list[str]]:
    cache = _cache()
    cached: dict[str, NLIClaimResult] = {}
    missing: list[str] = []
    for claim in claims:
        value = cache.get(_cache_key(claim, context))
        if isinstance(value, dict):
            cached[claim] = NLIClaimResult.model_validate(value)
        else:
            missing.append(claim)
    return cached, missing


def _write_cached(results: list[NLIClaimResult], context: str) -> None:
    cache = _cache()
    for result in results:
        cache.set(_cache_key(result.claim, context), result.model_dump())


def _cache_key(claim: str, context: str) -> str:
    claim_hash = hashlib.sha256(claim.encode("utf-8")).hexdigest()
    context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
    return f"nli:{claim_hash}:{context_hash}"


def _cache() -> Cache:
    return Cache(str(cache_path() / "nli"))
