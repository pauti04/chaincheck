"""
LLM-as-judge detection method.

Prompts a configurable judge model (Claude Haiku, gpt-4o-mini, or local Ollama)
to verify each atomic claim against the context using a structured rubric.
Randomises claim order to mitigate position bias; caps context length to
mitigate verbosity bias.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time

from pydantic import BaseModel, Field

from chaincheck.models import ClaimResult, MethodResult

def _default_judge_model() -> str:
    if os.getenv("JUDGE_MODEL"):
        return os.getenv("JUDGE_MODEL")  # type: ignore[return-value]
    if os.getenv("OPENAI_API_KEY"):
        return "gpt-4o-mini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude-haiku-4-5-20251001"
    return "gpt-4o-mini"

_JUDGE_MODEL = _default_judge_model()
_MAX_CONTEXT_TOKENS = 800
_MAX_RETRIES = 3
_VALID_LABELS = {"supported", "unsupported", "contradicted"}


class JudgeVerdict(BaseModel):
    """Structured output from the judge LLM for a single claim."""

    label: str = Field(description="supported | unsupported | contradicted")
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str


async def check_judge(
    claims: list[str],
    context: str,
    model: str = _JUDGE_MODEL,
    fact_check: bool = False,
) -> MethodResult:
    """
    Verify claims using an LLM judge with a structured rubric.

    Randomises claim order before sending to mitigate position bias.
    Caps context at _MAX_CONTEXT_TOKENS tokens to mitigate verbosity bias.
    Results are restored to original claim order before returning.

    Args:
        claims: Atomic claim strings to verify.
        context: Reference text to verify claims against.
        model: Judge model identifier (Claude, OpenAI, or Ollama).

    Returns:
        MethodResult with a ClaimResult per claim in original order.
    """
    if not claims:
        return MethodResult(method="judge", raw_score=0.0, latency_ms=0.0)

    start = time.time()
    truncated_ctx = _truncate_context(context)

    # Shuffle to mitigate position bias; track original indices to restore order
    indexed = list(enumerate(claims))
    random.shuffle(indexed)
    original_indices, shuffled_claims = zip(*indexed, strict=False)

    verdicts = list(
        await asyncio.gather(*[_verify_claim(c, truncated_ctx, model, fact_check=fact_check) for c in shuffled_claims])
    )

    # Restore original claim order
    ordered: list[JudgeVerdict | None] = [None] * len(claims)
    for orig_idx, verdict in zip(original_indices, verdicts, strict=True):
        ordered[orig_idx] = verdict

    conf_cap = 0.7 if fact_check else 1.0
    claim_results: list[ClaimResult] = []
    for claim, verdict in zip(claims, ordered, strict=True):
        raw_label = verdict.label.lower() if verdict else "unknown"
        label = raw_label if raw_label in _VALID_LABELS else "unknown"
        claim_results.append(
            ClaimResult(
                claim=claim,
                label=label,
                confidence=min(verdict.confidence, conf_cap) if verdict else 0.0,
                evidence=verdict.evidence if verdict else "parse error",
                method="judge",
            )
        )

    raw_score = _score_from_claims(claim_results)
    latency = (time.time() - start) * 1000
    return MethodResult(
        method="judge", claims=claim_results, raw_score=raw_score, latency_ms=latency
    )


async def _verify_claim(
    claim: str,
    context: str,
    model: str,
    retries: int = _MAX_RETRIES,
    fact_check: bool = False,
) -> JudgeVerdict:
    """
    Send a single claim to the judge model and parse structured JSON output.

    Retries with exponential backoff on malformed JSON (up to retries attempts).
    Supports any provider via llm.complete() — Anthropic, OpenAI, or Ollama.
    """
    from chaincheck.llm import complete

    prompt = _build_factcheck_prompt(claim) if fact_check else _build_judge_prompt(claim, context)

    last_err: str = ""
    for attempt in range(retries):
        try:
            raw = await complete(prompt, model, max_tokens=256)
            raw = raw.strip()
            # Strip markdown code fences if present
            match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
            if match:
                raw = match.group(1)
            data = json.loads(raw)
            return JudgeVerdict(**data)
        except Exception as exc:
            last_err = str(exc)
            if attempt < retries - 1:
                await asyncio.sleep(2**attempt)

    return JudgeVerdict(
        label="unknown",
        confidence=0.0,
        evidence=f"failed after {retries} retries: {last_err[:120]}",
    )


def _truncate_context(context: str, max_tokens: int = _MAX_CONTEXT_TOKENS) -> str:
    """
    Truncate context to approximately max_tokens tokens.

    Uses a word-count heuristic (1 token ≈ 0.75 words) to avoid importing
    a full tokenizer.
    """
    if not context:
        return context
    words = context.split()
    max_words = int(max_tokens * 0.75)
    if len(words) <= max_words:
        return context
    return " ".join(words[:max_words])


def _build_factcheck_prompt(claim: str) -> str:
    """Return a world-knowledge fact-check prompt (no source document)."""
    from datetime import date
    today = date.today().strftime("%B %d, %Y")
    return (
        f"You are a careful fact-checker using general world knowledge. Today's date is {today}.\n"
        "No source document is available — assess based on widely-accepted facts only.\n"
        "When evaluating age or time-sensitive claims, use today's date for your calculations.\n\n"
        f"Claim: {claim}\n\n"
        "Is this claim factually accurate based on common knowledge?\n"
        "For specific numeric facts (ages, dates, statistics): if the number is clearly wrong, mark 'unsupported'.\n"
        "Reserve 'supported' for claims that are actually correct or very close to correct.\n"
        "Only be conservative for genuinely ambiguous or hard-to-verify claims.\n"
        'Respond with ONLY valid JSON:\n'
        '{"label": "supported" | "unsupported" | "contradicted", '
        '"confidence": <float 0.0-0.7>, '
        '"evidence": "<brief explanation from general knowledge, citing today\'s date if relevant>"}'
    )


def _build_judge_prompt(claim: str, context: str) -> str:
    """Return the structured verification prompt for a single claim."""
    ctx_section = f"Context:\n{context}\n\n" if context else "Context: (none provided)\n\n"
    return (
        "You are a precise fact-checking assistant.\n\n"
        + ctx_section
        + f"Claim: {claim}\n\n"
        "Is this claim supported by the context above?\n"
        'Respond with ONLY valid JSON (no markdown fences):\n'
        '{"label": "supported" | "unsupported" | "contradicted", '
        '"confidence": <float 0.0-1.0>, '
        '"evidence": "<relevant quote or \'no relevant context found\'>"}'
    )


def _score_from_claims(claims: list[ClaimResult]) -> float:
    """Compute hallucination risk as mean confidence of bad claims across all claims."""
    if not claims:
        return 0.0
    bad = {"unsupported", "contradicted"}
    return min(1.0, sum(c.confidence for c in claims if c.label in bad) / len(claims))
