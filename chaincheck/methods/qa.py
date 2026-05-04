"""
QA-based claim verification.

For each atomic claim, asks the LLM a direct yes/no question against the
context. No chain-of-thought, no rubric, temperature=0. Faster than judge
(no explanation required) and complementary to NLI (LLM-based rather than
model-based, catches semantic errors NLI misses).

Distinction from judge:
  - Judge: complex rubric, asks for label + confidence + evidence, CoT
  - QA:    single yes/no question, temperature=0, ~3× fewer output tokens
  - NLI:   no LLM at all, pure DeBERTa cross-encoder, ~30× faster than QA

Use QA when you want an LLM check but judge is too slow or too expensive,
or as a fast second opinion alongside NLI.
"""

from __future__ import annotations

import asyncio
import os
import time

from chaincheck.models import ClaimResult, MethodResult

_QA_MODEL = os.getenv("QA_MODEL", "gpt-4o-mini")
_QA_CONCURRENCY = int(os.getenv("QA_CONCURRENCY", "8"))

_SYSTEM_PROMPT = (
    "You are a fact-checker. Given a context and a claim, answer only 'yes' if the "
    "context supports the claim, or 'no' if the context contradicts or does not support "
    "the claim. No other output."
)


async def check_qa(
    claims: list[str],
    context: str,
    model: str = _QA_MODEL,
) -> MethodResult:
    """
    Verify claims against context using structured yes/no LLM queries.

    Args:
        claims: Atomic claim strings to verify.
        context: Reference text to verify claims against.
        model: Model identifier passed to llm.complete().

    Returns:
        MethodResult with a ClaimResult for each input claim.
    """
    if not claims:
        return MethodResult(method="qa", raw_score=0.0, latency_ms=0.0)

    if not context.strip():
        return MethodResult(
            method="qa",
            claims=[
                ClaimResult(claim=c, label="unknown", confidence=0.0,
                            evidence="no context provided", method="qa")
                for c in claims
            ],
            raw_score=0.0,
            latency_ms=0.0,
        )

    start = time.time()
    sem = asyncio.Semaphore(_QA_CONCURRENCY)

    async def _verify(claim: str) -> ClaimResult:
        async with sem:
            label, confidence = await _ask_yn(claim, context, model)
        return ClaimResult(
            claim=claim,
            label=label,
            confidence=confidence,
            evidence=context[:200] if label == "contradicted" else "",
            method="qa",
        )

    claim_results = list(await asyncio.gather(*[_verify(c) for c in claims]))
    raw_score = _score_from_claims(claim_results)
    return MethodResult(
        method="qa",
        claims=claim_results,
        raw_score=raw_score,
        latency_ms=(time.time() - start) * 1000,
    )


async def _ask_yn(claim: str, context: str, model: str) -> tuple[str, float]:
    """Ask a yes/no question and map the answer to a (label, confidence) pair."""
    from chaincheck.llm import LLMError, complete

    prompt = (
        f"Context:\n{context[:1500]}\n\n"
        f"Claim: {claim}\n\n"
        "Does the context support this claim? Answer only 'yes' or 'no'."
    )
    try:
        answer = (await complete(prompt, model, max_tokens=5)).strip().lower()
    except LLMError:
        return "unknown", 0.0

    if answer.startswith("yes"):
        return "supported", 0.9
    if answer.startswith("no"):
        return "contradicted", 0.9
    return "unknown", 0.5


def _score_from_claims(claims: list[ClaimResult]) -> float:
    """Confidence-weighted fraction of contradicted/unsupported claims."""
    if not claims:
        return 0.0
    bad = {"unsupported", "contradicted"}
    total_w = sum(c.confidence for c in claims)
    if total_w == 0:
        return float(sum(1 for c in claims if c.label in bad) / len(claims))
    return min(1.0, sum(c.confidence for c in claims if c.label in bad) / total_w)
