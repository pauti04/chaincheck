"""Aggregates per-method claim results into a single hallucination risk score."""

from __future__ import annotations

from chaincheck.models import ClaimResult, MethodResult

_HALLUCINATED_LABELS = {"unsupported", "contradicted"}


def aggregate_claim_scores(claims: list[ClaimResult]) -> float:
    """
    Compute a hallucination risk score for a list of claim results.

    Score is the confidence-weighted fraction of claims that are unsupported
    or contradicted. Falls back to unweighted fraction when all confidences
    are zero. Returns 0.0 for empty input.

    Args:
        claims: Per-claim verification results from a single method.

    Returns:
        Float in [0, 1] where 1 = fully hallucinated.
    """
    if not claims:
        return 0.0
    total_weight = sum(c.confidence for c in claims)
    if total_weight == 0:
        bad = sum(1 for c in claims if c.label in _HALLUCINATED_LABELS)
        return float(bad / len(claims))
    weighted_bad = sum(c.confidence for c in claims if c.label in _HALLUCINATED_LABELS)
    return min(1.0, max(0.0, weighted_bad / total_weight))


def method_score(result: MethodResult) -> float:
    """
    Extract or compute the scalar risk score for a single MethodResult.

    Recomputes from claims when claims are present so the score stays
    consistent with the claim-level labels; otherwise returns raw_score.
    """
    if result.claims:
        return aggregate_claim_scores(result.claims)
    return result.raw_score
