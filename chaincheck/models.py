"""Shared Pydantic models for ChainCheck data structures."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClaimResult(BaseModel):
    """Verification result for a single atomic claim from one detection method."""

    claim: str = Field(description="The atomic factual assertion being verified")
    label: Literal["supported", "unsupported", "contradicted", "unknown"] = Field(
        description="Verification verdict"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the label (0–1)")
    evidence: str = Field(description="Supporting quote or 'no relevant context found'")
    method: str = Field(description="Detection method that produced this result")


class MethodResult(BaseModel):
    """Aggregated output from a single detection method across all claims."""

    method: str
    claims: list[ClaimResult] = Field(default_factory=list)
    raw_score: float = Field(
        ge=0.0, le=1.0, description="Method-specific hallucination risk score"
    )
    latency_ms: float = Field(ge=0.0)
    error: str | None = None


class ConsistencyResult(BaseModel):
    """Result specific to the self-consistency detection method."""

    consistency_score: float = Field(ge=0.0, le=1.0)
    similarity_matrix: list[list[float]]
    sample_count: int
    latency_ms: float


class DetectionResult(BaseModel):
    """Complete hallucination detection result for an LLM response."""

    response: str
    claims: list[str] = Field(description="Atomic claims extracted from the response")
    method_results: dict[str, MethodResult] = Field(default_factory=dict)
    consistency: ConsistencyResult | None = None
    aggregate_score: float = Field(
        ge=0.0, le=1.0, description="Weighted hallucination risk score"
    )
    risk_level: Literal["low", "medium", "high"]
    latency_ms: dict[str, float] = Field(default_factory=dict)
    request_id: str | None = None
