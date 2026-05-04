"""Orchestrates all hallucination detection methods and aggregates results."""

from __future__ import annotations

import asyncio
import json as _json
import os
import time
import uuid
from collections.abc import AsyncGenerator
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
from chaincheck.models import ClaimResult, DetectionResult, MethodResult

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


async def detect_stream(
    response: str,
    context: str = "",
    prompt: str = "",
    methods: list[str] | None = None,
    request_id: str | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Stream detection events via async generator as each stage completes.

    Yields dicts with a ``type`` key:
      - ``{"type": "claims",  "claims": [...], "request_id": "..."}``
      - ``{"type": "method",  "method": "nli", "score": 0.72, "latency_ms": 230, "error": null}``
      - ``{"type": "result",  "data": {...full DetectionResult as dict...}}``
    """
    t0 = time.time()
    active = list(methods or _DEFAULT_METHODS)
    rid = request_id or str(uuid.uuid4())

    claims = await decompose(response)
    yield {"type": "claims", "claims": claims, "request_id": rid}

    task_map: dict[asyncio.Task, str] = {}
    if "nli" in active and context.strip():
        task_map[asyncio.create_task(check_nli(claims, context))] = "nli"
    if "judge" in active:
        task_map[asyncio.create_task(check_judge(claims, context))] = "judge"
    if "qa" in active and context.strip():
        task_map[asyncio.create_task(check_qa(claims, context))] = "qa"
    if "consistency" in active and prompt.strip():
        task_map[asyncio.create_task(check_consistency(prompt, response))] = "consistency"
    if "logprobs" in active and prompt.strip():
        task_map[asyncio.create_task(check_logprobs(prompt, claims))] = "logprobs"

    method_results: dict[str, MethodResult] = {}
    consistency_result = None
    pending = set(task_map.keys())

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            name = task_map[task]
            try:
                raw = task.result()
                if name == "consistency":
                    consistency_result = raw
                    mr = MethodResult(
                        method=name,
                        raw_score=max(0.0, 1.0 - raw.consistency_score),
                        latency_ms=raw.latency_ms,
                    )
                else:
                    mr = raw
                method_results[name] = mr
            except Exception as exc:
                mr = MethodResult(method=name, raw_score=0.0, latency_ms=0.0, error=str(exc))
                method_results[name] = mr

            yield {
                "type": "method",
                "method": name,
                "score": round(mr.raw_score, 4),
                "latency_ms": round(mr.latency_ms, 1),
                "error": mr.error,
            }

    aggregate = _weighted_aggregate(method_results, list(method_results.keys()))
    final = DetectionResult(
        response=response,
        claims=claims,
        method_results=method_results,
        consistency=consistency_result,
        claim_details=_compute_claim_details(method_results),
        aggregate_score=aggregate,
        risk_level=_compute_risk_level(aggregate),
        latency_ms={m: mr.latency_ms for m, mr in method_results.items()},
        total_latency_ms=(time.time() - t0) * 1000,
        request_id=rid,
    )
    yield {"type": "result", "data": _json.loads(final.model_dump_json())}


async def _cascade_detect(
    claims: list[str],
    response: str,
    context: str,
    prompt: str,
    active: list[str],
    rid: str,
) -> DetectionResult:
    """Run NLI; escalate to judge only when score is in the ambiguous band (0.2–0.8)."""
    t0 = time.time()
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
        claim_details=_compute_claim_details(method_results),
        aggregate_score=aggregate,
        risk_level=_compute_risk_level(aggregate),
        latency_ms=latency_ms,
        total_latency_ms=(time.time() - t0) * 1000,
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
    t0 = time.time()
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
        claim_details=_compute_claim_details(method_results),
        aggregate_score=aggregate,
        risk_level=_compute_risk_level(aggregate),
        latency_ms=latency_ms,
        total_latency_ms=(time.time() - t0) * 1000,
        request_id=rid,
    )


def _compute_risk_level(score: float) -> Literal["low", "medium", "high"]:
    """Map aggregate score to a human-readable risk level using env thresholds."""
    if score < _RISK_LOW:
        return "low"
    if score >= _RISK_HIGH:
        return "high"
    return "medium"


def _compute_claim_details(
    method_results: dict[str, MethodResult],
) -> list[ClaimResult] | None:
    """
    Aggregate per-claim verdicts across all methods using confidence-weighted voting.

    When only one method has per-claim data its results are returned directly.
    When multiple methods have results, each method's weight × confidence votes
    for a label; the label with the most accumulated weight wins, and the best
    evidence (by weight × confidence) is selected.
    """
    active = [
        (m, mr)
        for m, mr in method_results.items()
        if mr.claims and not mr.error
    ]
    if not active:
        return None
    if len(active) == 1:
        return active[0][1].claims

    n = min(len(mr.claims) for _, mr in active)
    aggregated: list[ClaimResult] = []

    for i in range(n):
        label_weights: dict[str, float] = {}
        best_evidence = ""
        best_weight = -1.0

        for method_name, mr in active:
            w = _METHOD_WEIGHTS.get(method_name, 0.25)
            cr = mr.claims[i]
            vote = w * cr.confidence
            label_weights[cr.label] = label_weights.get(cr.label, 0.0) + vote
            if vote > best_weight and cr.evidence not in ("", "no relevant context found"):
                best_weight = vote
                best_evidence = cr.evidence

        best_label = max(label_weights, key=lambda k: label_weights[k])
        total = sum(label_weights.values())
        confidence = label_weights[best_label] / total if total > 0 else 0.0
        claim_text = active[0][1].claims[i].claim

        aggregated.append(
            ClaimResult(
                claim=claim_text,
                label=best_label,
                confidence=round(confidence, 4),
                evidence=best_evidence or "no evidence available",
                method="ensemble",
            )
        )

    return aggregated


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
