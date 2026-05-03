"""
Token-level uncertainty detection via log-probabilities.

Re-generates the response from the original prompt using OpenAI's API (which
exposes token logprobs) and flags low-confidence spans. Claims that overlap
with low-logprob spans are marked as uncertain.

Requires OPENAI_API_KEY. Falls back gracefully if the key is absent.

Why OpenAI here: the Anthropic API does not currently expose token log-
probabilities in its public API; OpenAI's Chat Completions API has stable
logprob support via `logprobs=True`.
"""

from __future__ import annotations

import os
import re
import time

from chaincheck.models import ClaimResult, MethodResult

_LOGPROB_MODEL = os.getenv("LOGPROB_MODEL", "gpt-4o-mini")
_LOGPROB_THRESHOLD = float(os.getenv("LOGPROB_THRESHOLD", "-1.5"))  # nats; tune against eval


async def check_logprobs(
    prompt: str,
    claims: list[str],
    model: str = _LOGPROB_MODEL,
) -> MethodResult:
    """
    Detect uncertain claims by analysing token-level log-probabilities.

    Re-generates a response from the prompt with logprobs enabled and maps
    low-confidence tokens back to the input claims via substring overlap.
    Claims whose tokens average below _LOGPROB_THRESHOLD are flagged as
    unsupported.

    Args:
        prompt: The original user prompt that generated the response.
        claims: Atomic claims to score (from decompose()).
        model: OpenAI model with logprobs support.

    Returns:
        MethodResult with per-claim confidence derived from logprobs.
        Returns an error MethodResult if OPENAI_API_KEY is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return MethodResult(
            method="logprobs",
            raw_score=0.0,
            latency_ms=0.0,
            error="OPENAI_API_KEY not set — logprobs method unavailable",
        )

    if not claims:
        return MethodResult(method="logprobs", raw_score=0.0, latency_ms=0.0)

    start = time.time()

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            logprobs=True,
            top_logprobs=1,
        )
    except Exception as exc:
        return MethodResult(
            method="logprobs",
            raw_score=0.0,
            latency_ms=(time.time() - start) * 1000,
            error=f"OpenAI API error: {exc}",
        )

    token_logprobs = _extract_token_logprobs(response)
    generated_text = response.choices[0].message.content or ""

    claim_results: list[ClaimResult] = []
    for claim in claims:
        avg_lp = _claim_avg_logprob(claim, generated_text, token_logprobs)
        label, confidence = _logprob_to_label(avg_lp)
        claim_results.append(
            ClaimResult(
                claim=claim,
                label=label,
                confidence=confidence,
                evidence=f"avg token log-prob: {avg_lp:.3f} (threshold: {_LOGPROB_THRESHOLD})",
                method="logprobs",
            )
        )

    raw_score = _score_from_claims(claim_results)
    return MethodResult(
        method="logprobs",
        claims=claim_results,
        raw_score=raw_score,
        latency_ms=(time.time() - start) * 1000,
    )


def _extract_token_logprobs(response) -> list[tuple[str, float]]:
    """
    Extract (token_text, logprob) pairs from an OpenAI completion response.

    Returns an empty list if logprob data is unavailable.
    """
    try:
        content = response.choices[0].logprobs.content or []
        return [(t.token, t.logprob) for t in content]
    except (AttributeError, IndexError, TypeError):
        return []


def _claim_avg_logprob(
    claim: str,
    generated_text: str,
    token_logprobs: list[tuple[str, float]],
) -> float:
    """
    Compute the average log-probability of tokens that match the claim text.

    Uses a sliding-window search to find the contiguous token span that best
    reconstructs the claim, then averages their log-probabilities.
    Returns _LOGPROB_THRESHOLD - 1 (signals uncertain) when no match is found.
    """
    if not token_logprobs:
        return _LOGPROB_THRESHOLD - 1.0

    # Build cumulative token string to find claim span
    tokens = [t for t, _ in token_logprobs]
    logprobs = [lp for _, lp in token_logprobs]
    cum = ""
    token_starts: list[int] = []
    for tok in tokens:
        token_starts.append(len(cum))
        cum += tok

    # Find claim in generated_text (case-insensitive, strip punctuation variation)
    claim_clean = re.sub(r"\s+", " ", claim.lower().strip())
    gen_clean = re.sub(r"\s+", " ", generated_text.lower())

    idx = gen_clean.find(claim_clean)
    if idx == -1:
        # Try partial match on key words (>= 4 chars) as fallback
        words = [w for w in claim_clean.split() if len(w) >= 4]
        if not words:
            return _LOGPROB_THRESHOLD - 1.0
        matched_lps: list[float] = []
        for word in words:
            word_idx = gen_clean.find(word)
            if word_idx != -1:
                # Find which token(s) contain this position
                for ti, ts in enumerate(token_starts):
                    te = token_starts[ti + 1] if ti + 1 < len(token_starts) else len(cum)
                    if ts <= word_idx < te:
                        matched_lps.append(logprobs[ti])
                        break
        return sum(matched_lps) / len(matched_lps) if matched_lps else _LOGPROB_THRESHOLD - 1.0

    # Exact match: find token indices that cover [idx, idx+len(claim_clean))
    end_idx = idx + len(claim_clean)
    def _in_span(ti: int, ts: int) -> bool:
        next_start = token_starts[ti + 1] if ti + 1 < len(token_starts) else len(cum)
        return ts >= idx and (ts < end_idx or next_start <= end_idx)

    span_lps = [logprobs[ti] for ti, ts in enumerate(token_starts) if _in_span(ti, ts)]
    return sum(span_lps) / len(span_lps) if span_lps else _LOGPROB_THRESHOLD - 1.0


def _logprob_to_label(avg_lp: float) -> tuple[str, float]:
    """
    Convert an average log-probability to a (label, confidence) pair.

    Logprob of 0 = probability 1.0 (certain). More negative = less certain.
    Maps to [0, 1] confidence and a verdict label.
    """
    import math

    prob = math.exp(min(0.0, avg_lp))  # clamp to avoid >1
    if avg_lp >= _LOGPROB_THRESHOLD:
        return "supported", prob
    return "unsupported", 1.0 - prob


def _score_from_claims(claims: list[ClaimResult]) -> float:
    """Confidence-weighted fraction of claims that are unsupported or contradicted."""
    if not claims:
        return 0.0
    bad = {"unsupported", "contradicted"}
    total_w = sum(c.confidence for c in claims)
    if total_w == 0:
        return float(sum(1 for c in claims if c.label in bad) / len(claims))
    return min(1.0, sum(c.confidence for c in claims if c.label in bad) / total_w)
