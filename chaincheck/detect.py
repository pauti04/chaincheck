"""Orchestrates all hallucination detection methods and aggregates results."""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Literal

from chaincheck.config import (
    RISK_HIGH_THRESHOLD,
    RISK_LOW_THRESHOLD,
    WEIGHT_CONSISTENCY,
    WEIGHT_JUDGE,
    WEIGHT_LOGPROBS,
    WEIGHT_NLI,
)
from chaincheck.decompose import decompose
from chaincheck.methods.consistency import check_consistency
from chaincheck.methods.judge import check_judge
from chaincheck.methods.logprobs import check_logprobs
from chaincheck.methods.nli import check_nli
from chaincheck.methods.qa import check_qa
from chaincheck.models import DetectionResult, MethodResult

_METHOD_WEIGHTS: dict[str, float] = {
    "nli": WEIGHT_NLI,
    "consistency": WEIGHT_CONSISTENCY,
    "judge": WEIGHT_JUDGE,
    "logprobs": WEIGHT_LOGPROBS,
    "qa": float(os.getenv("QA_WEIGHT", "0.25")),
}

_RISK_LOW = RISK_LOW_THRESHOLD
_RISK_HIGH = RISK_HIGH_THRESHOLD

_DEFAULT_METHODS: list[str] = ["nli", "judge"]


async def detect(
    response: str,
    context: str = "",
    prompt: str = "",
    methods: list[Literal["nli", "consistency", "judge", "logprobs"]] | None = None,
    request_id: str | None = None,
    cascade: bool = False,
) -> DetectionResult:
    """
    Detect hallucinations in an LLM response.

    Args:
        response: The LLM output to analyse.
        context: Retrieved context (RAG) or reference document.
        prompt: Original user prompt that generated the response.
        methods: Detection methods to run. Defaults to ["nli", "judge"].
                 Add "logprobs" to enable token-level uncertainty (requires OPENAI_API_KEY).
        request_id: Optional trace ID; auto-generated if not supplied.
        cascade: If True and context is provided, run NLI first and only escalate
                 to judge when the NLI score is in the ambiguous 0.2–0.8 range.
                 Cuts average latency by ~34× on clear-cut cases.

    Returns:
        DetectionResult with per-claim labels, aggregate score, and risk level.
    """
    active = list(methods or _DEFAULT_METHODS)
    rid = request_id or str(uuid.uuid4())

    claims = await decompose(response)

    if cascade and "nli" in active and "judge" in active and context.strip():
        return await _cascade_detect(claims, response, context, prompt, active, rid)

    return await _full_detect(claims, response, context, prompt, active, rid)


async def _cascade_detect(
    claims: list[str],
    response: str,
    context: str,
    prompt: str,
    active: list[str],
    rid: str,
) -> DetectionResult:
    """Run NLI; escalate to judge only when score is in the ambiguous band (0.2–0.8)."""
    nli_result = await check_nli(claims, context)
    nli_score = nli_result.raw_score
    method_results: dict[str, MethodResult] = {"nli": nli_result}
    latency_ms: dict[str, float] = {"nli": nli_result.latency_ms}

    if 0.2 <= nli_score <= 0.8:
        judge_result = await check_judge(claims, context)
        method_results["judge"] = judge_result
        latency_ms["judge"] = judge_result.latency_ms

    aggregate = _weighted_aggregate(method_results, list(method_results.keys()))
    return DetectionResult(
        response=response,
        claims=claims,
        method_results=method_results,
        aggregate_score=aggregate,
        risk_level=_compute_risk_level(aggregate),
        latency_ms=latency_ms,
        request_id=rid,
    )


async def _full_detect(
    claims: list[str],
    response: str,
    context: str,
    prompt: str,
    active: list[str],
    rid: str,
) -> DetectionResult:
    """Run all requested methods in parallel."""
    tasks: dict[str, object] = {}
    if "nli" in active and context.strip():
        tasks["nli"] = check_nli(claims, context)
    if "judge" in active:
        tasks["judge"] = check_judge(claims, context)
    if "qa" in active and context.strip():
        tasks["qa"] = check_qa(claims, context)
    if "consistency" in active and prompt.strip():
        tasks["consistency"] = check_consistency(prompt, response)
    if "logprobs" in active and prompt.strip():
        tasks["logprobs"] = check_logprobs(prompt, claims)

    raw_results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    method_results: dict[str, MethodResult] = {}
    consistency_result = None
    latency_ms: dict[str, float] = {}

    for method_name, result in zip(tasks.keys(), raw_results, strict=True):
        if isinstance(result, Exception):
            method_results[method_name] = MethodResult(
                method=method_name,
                raw_score=0.0,
                latency_ms=0.0,
                error=str(result),
            )
        elif method_name == "consistency":
            consistency_result = result
            method_results[method_name] = MethodResult(
                method=method_name,
                raw_score=max(0.0, 1.0 - result.consistency_score),
                latency_ms=result.latency_ms,
            )
            latency_ms[method_name] = result.latency_ms
        else:
            method_results[method_name] = result
            latency_ms[method_name] = result.latency_ms

    aggregate = _weighted_aggregate(method_results, list(method_results.keys()))
    return DetectionResult(
        response=response,
        claims=claims,
        method_results=method_results,
        consistency=consistency_result,
        aggregate_score=aggregate,
        risk_level=_compute_risk_level(aggregate),
        latency_ms=latency_ms,
        request_id=rid,
    )


def _compute_risk_level(score: float) -> Literal["low", "medium", "high"]:
    """Map aggregate score to a human-readable risk level using env thresholds."""
    if score < _RISK_LOW:
        return "low"
    if score >= _RISK_HIGH:
        return "high"
    return "medium"


def _weighted_aggregate(
    method_results: dict[str, MethodResult],
    active_methods: list[str],
) -> float:
    """
    Compute weighted average hallucination score across active methods.

    Normalises weights to sum to 1.0 when a subset of methods is used,
    so adding or omitting logprobs doesn't skew the aggregate.
    """
    valid = [m for m in active_methods if m in _METHOD_WEIGHTS and m in method_results]
    if not valid:
        return 0.0
    total_weight = sum(_METHOD_WEIGHTS[m] for m in valid)
    if total_weight == 0:
        return 0.0
    weighted_sum = sum(_METHOD_WEIGHTS[m] * method_results[m].raw_score for m in valid)
    return min(1.0, max(0.0, weighted_sum / total_weight))
