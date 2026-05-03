"""LLM-as-judge hallucination detection."""

from __future__ import annotations

import asyncio
import json
import os
import random
from typing import Literal

from pydantic import BaseModel, ValidationError

from chaincheck.llm import LLMError, complete


class JudgeClaimResult(BaseModel):
    """Judge result for one atomic claim."""

    claim: str
    label: Literal["supported", "unsupported", "contradicted"]
    confidence: float
    evidence: str


async def judge_claims(
    claims: list[str],
    context: str,
    model: str = "claude-haiku-4-5",
) -> list[JudgeClaimResult]:
    """Ask a judge LLM whether each claim is supported by context."""
    if not claims:
        return []
    capped_context = _cap_context(context)
    indexed = list(enumerate(claims))
    random.shuffle(indexed)
    tasks = [_judge_one(index, claim, capped_context, model) for index, claim in indexed]
    judged = await asyncio.gather(*tasks)
    return [result for _, result in sorted(judged, key=lambda item: item[0])]


async def _judge_one(
    index: int, claim: str, context: str, model: str
) -> tuple[int, JudgeClaimResult]:
    for attempt in range(3):
        try:
            raw = await complete(_judge_prompt(claim, context), model=model, max_tokens=500)
            return index, _parse_judge(raw, claim)
        except (LLMError, json.JSONDecodeError, ValidationError):
            await asyncio.sleep(0.2 * (2**attempt))
    return index, _fallback_judge(claim, context)


def _parse_judge(raw: str, claim: str) -> JudgeClaimResult:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("JSON object not found", raw, 0)
    payload = json.loads(raw[start : end + 1])
    payload["claim"] = claim
    return JudgeClaimResult.model_validate(payload)


def _fallback_judge(claim: str, context: str) -> JudgeClaimResult:
    label = "supported" if claim.lower() in context.lower() else "unsupported"
    evidence = claim if label == "supported" else "no relevant context found"
    return JudgeClaimResult(claim=claim, label=label, confidence=0.35, evidence=evidence)


def _judge_prompt(claim: str, context: str) -> str:
    return (
        "You are auditing whether one factual claim is supported by context.\n"
        "Return strict JSON with keys: label, confidence, evidence.\n"
        "label must be one of: supported, unsupported, contradicted.\n"
        "confidence must be a number from 0.0 to 1.0.\n"
        'evidence must quote relevant context or say "no relevant context found".\n\n'
        f"Context:\n{context}\n\nClaim:\n{claim}"
    )


def _cap_context(context: str, max_tokens: int = 800) -> str:
    tokens = context.split()
    if len(tokens) <= max_tokens:
        return context
    head = " ".join(tokens[: max_tokens // 2])
    tail = " ".join(tokens[-max_tokens // 2 :])
    return f"{head}\n\n[...context truncated for judge...]\n\n{tail}"


def default_model() -> str:
    """Return the configured default judge model."""
    return os.getenv("JUDGE_MODEL", "claude-haiku-4-5")
