"""Orchestrates all hallucination detection methods and aggregates results."""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Literal

from chaincheck.decompose import decompose
from chaincheck.methods.consistency import check_consistency
from chaincheck.methods.judge import check_judge
from chaincheck.methods.logprobs import check_logprobs
from chaincheck.methods.nli import check_nli
from chaincheck.models import DetectionResult, MethodResult

_METHOD_WEIGHTS: dict[str, float] = {
    "nli": float(os.getenv("NLI_WEIGHT", "0.35")),
    "consistency": float(os.getenv("CONSISTENCY_WEIGHT", "0.25")),
    "judge": float(os.getenv("JUDGE_WEIGHT", "0.25")),
    "logprobs": float(os.getenv("LOGPROB_WEIGHT", "0.15")),
}

_RISK_LOW = float(os.getenv("RISK_LOW_THRESHOLD", "0.3"))
_RISK_HIGH = float(os.getenv("RISK_HIGH_THRESHOLD", "0.7"))

_DEFAULT_METHODS: list[str] = ["nli", "consistency", "judge"]


async def detect(
    response: str,
    context: str = "",
    prompt: str = "",
    methods: list[Literal["nli", "consistency", "judge", "logprobs"]] | None = None,
    request_id: str | None = None,
) -> DetectionResult:
    """
    Detect hallucinations in an LLM response.

    Args:
        response: The LLM output to analyse.
        context: Retrieved context (RAG) or reference document.
        prompt: Original user prompt that generated the response.
        methods: Detection methods to run. Defaults to ["nli", "consistency", "judge"].
                 Add "logprobs" to enable token-level uncertainty (requires OPENAI_API_KEY).
        request_id: Optional trace ID; auto-generated if not supplied.

    Returns:
        DetectionResult with per-claim labels, aggregate score, and risk level.
    """
    active = list(methods or _DEFAULT_METHODS)
    rid = request_id or str(uuid.uuid4())

    claims = await decompose(response)

    # Gate methods on available inputs to avoid silent no-ops
    tasks: dict[str, object] = {}
    if "nli" in active and context.strip():
        tasks["nli"] = check_nli(claims, context)
    if "judge" in active:
        tasks["judge"] = check_judge(claims, context)
    if "consistency" in active and prompt.strip():
        tasks["consistency"] = check_consistency(prompt, response)
    if "logprobs" in active and prompt.strip():
        tasks["logprobs"] = check_logprobs(prompt, claims)

    raw_results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    method_results: dict[str, MethodResult] = {}
    consistency_result = None
    latency_ms: dict[str, float] = {}

    for method_name, result in zip(tasks.keys(), raw_results):
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
