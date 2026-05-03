"""Detection orchestration for ChainCheck."""

from __future__ import annotations

import os
import time
from typing import Any

from pydantic import BaseModel, Field

from chaincheck.config import env_float, env_int
from chaincheck.decompose import decompose
from chaincheck.methods.consistency import ConsistencyResult, check_consistency
from chaincheck.methods.judge import JudgeClaimResult, default_model, judge_claims
from chaincheck.methods.nli import NLIClaimResult, check_claims_nli
from chaincheck.score import aggregate_score, configured_risk_level, judge_risk, mean, nli_risk


class ClaimResult(BaseModel):
    """Unified per-claim aggregate view across selected methods."""

    claim: str
    hallucination_score: float
    label: str
    evidence: str = ""


class DetectionResult(BaseModel):
    """Unified result returned by ChainCheck detection."""

    response: str
    claims: list[str] = Field(default_factory=list)
    claim_results: list[ClaimResult] = Field(default_factory=list)
    methods: dict[str, object] = Field(default_factory=dict)
    aggregate_score: float = 0.0
    risk_level: str = "low"
    latency_ms: dict[str, float] = Field(default_factory=dict)


async def detect(
    response: str,
    context: str = "",
    prompt: str = "",
    methods: list[str] | None = None,
) -> DetectionResult:
    """Run selected hallucination detection methods for a response."""
    selected = _normalize_methods(methods)
    claims = await _timed_decompose(response)
    method_results, latencies = await _run_methods(selected, claims, response, context, prompt)
    method_scores = _method_scores(method_results)
    score = aggregate_score(method_scores, selected)
    return DetectionResult(
        response=response,
        claims=claims,
        claim_results=build_claim_results(claims, method_results, selected),
        methods=method_results,
        aggregate_score=score,
        risk_level=configured_risk_level(score),
        latency_ms=latencies,
    )


def build_claim_results(
    claims: list[str], methods: dict[str, Any], selected: list[str]
) -> list[ClaimResult]:
    """Build claim-level aggregate rows from method outputs."""
    rows: list[ClaimResult] = []
    for index, claim in enumerate(claims):
        risks = _claim_risks(index, methods, selected)
        score = mean(risks)
        rows.append(
            ClaimResult(claim=claim, hallucination_score=score, label=configured_risk_level(score))
        )
    return rows


async def _timed_decompose(response: str) -> list[str]:
    return await decompose(response)


async def _run_methods(
    selected: list[str], claims: list[str], response: str, context: str, prompt: str
) -> tuple[dict[str, Any], dict[str, float]]:
    results: dict[str, Any] = {}
    latencies: dict[str, float] = {}
    for method in selected:
        started = time.perf_counter()
        results[method] = await _run_method(method, claims, response, context, prompt)
        latencies[method] = (time.perf_counter() - started) * 1000
    return results, latencies


async def _run_method(
    method: str, claims: list[str], response: str, context: str, prompt: str
) -> Any:
    if method == "nli":
        return await check_claims_nli(claims, context)
    if method == "consistency":
        return await check_consistency(
            response,
            prompt,
            sample_count=env_int("CONSISTENCY_SAMPLES", 5),
            threshold=env_float("CONSISTENCY_THRESHOLD", 0.82),
            model=os.getenv("CONSISTENCY_MODEL", "claude-haiku-4-5"),
        )
    if method == "judge":
        return await judge_claims(claims, context, model=default_model())
    raise ValueError(f"Unknown detection method: {method}")


def _method_scores(methods: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    if "nli" in methods:
        scores["nli"] = mean([nli_risk(item.label, item.confidence) for item in methods["nli"]])
    if "judge" in methods:
        scores["judge"] = mean(
            [judge_risk(item.label, item.confidence) for item in methods["judge"]]
        )
    if "consistency" in methods:
        result: ConsistencyResult = methods["consistency"]
        scores["consistency"] = 1.0 - result.consistency_score
    return scores


def _claim_risks(index: int, methods: dict[str, Any], selected: list[str]) -> list[float]:
    risks: list[float] = []
    if "nli" in selected and "nli" in methods:
        result: NLIClaimResult = methods["nli"][index]
        risks.append(nli_risk(result.label, result.confidence))
    if "judge" in selected and "judge" in methods:
        result: JudgeClaimResult = methods["judge"][index]
        risks.append(judge_risk(result.label, result.confidence))
    if "consistency" in selected and "consistency" in methods:
        result = methods["consistency"]
        risks.append(1.0 - result.consistency_score)
    return risks


def _normalize_methods(methods: list[str] | None) -> list[str]:
    selected = methods or ["nli", "consistency", "judge"]
    if selected == ["all"]:
        return ["nli", "consistency", "judge"]
    return selected
