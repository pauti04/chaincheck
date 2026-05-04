"""
Claim-level discrimination evaluation.

Standard benchmarks (HaluEval, TruthfulQA) measure *response-level* F1:
does the pipeline correctly label the whole response as hallucinated or not?
ChainCheck's pitch is *claim-level*: which specific claim is wrong?

Without human annotation of individual claims, exact claim-level precision/recall
cannot be computed. This module uses HaluEval pairs (each question ships with
both a correct answer and a hallucinated answer against the same context) to
compute two proxy metrics that characterise claim-level behaviour:

  clean_flagging_rate  — fraction of claims in CORRECT responses that ChainCheck
                         marks as contradicted/unsupported.  Should be low (false
                         positive rate at claim level).

  halluc_flagging_rate — fraction of claims in HALLUCINATED responses that
                         ChainCheck marks as contradicted/unsupported.  Should be
                         high (coverage of bad claims).

  discrimination_ratio — halluc / clean.  Ratio > 1 means ChainCheck flags
                         proportionally more claims in hallucinated responses
                         than in correct ones.  A ratio of 3 means hallucinated
                         responses have 3× more flagged claims.

  claim_auc            — AUC of per-claim scores against response-level labels
                         (0 = clean, 1 = hallucinated), averaged across all
                         claims in the evaluation set.  Does not require claim-
                         level ground truth and is a scalar summary of how well
                         claim scores discriminate response quality.

These metrics are reported alongside the standard response-level metrics so
the gap between what ChainCheck claims to do and what is measured is explicit
and shrinking.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClaimLevelMetrics:
    """Per-claim discrimination statistics across a paired evaluation set."""

    clean_flagging_rate: float      # FPR at claim level
    halluc_flagging_rate: float     # coverage of hallucinated claims
    discrimination_ratio: float     # halluc / clean (higher = better separation)
    claim_auc: float                # AUC of claim scores vs response labels
    n_pairs: int                    # number of (correct, hallucinated) pairs
    n_clean_claims: int             # total claims evaluated from correct responses
    n_halluc_claims: int            # total claims evaluated from hallucinated responses
    avg_claims_per_response: float  # mean claims per response (decomposition quality proxy)
    latency_ms: float


@dataclass
class ClaimLevelRun:
    """Results from a claim-level evaluation run."""

    method: str
    pairs: int
    metrics: ClaimLevelMetrics
    raw_results: list[dict] = field(default_factory=list)


async def run_claimlevel(
    method: str = "nli",
    n_pairs: int = 100,
) -> ClaimLevelRun:
    """
    Evaluate claim-level discrimination on HaluEval paired responses.

    For each question in HaluEval we have both a correct answer and a
    hallucinated answer against the same knowledge context.  We run
    ChainCheck on both and compare the per-claim flagging rates.

    Args:
        method: Detection method — "nli" (fastest, needs context) or "judge".
        n_pairs: Number of (correct, hallucinated) pairs to evaluate.

    Returns:
        ClaimLevelRun with discrimination metrics and raw per-pair results.
    """
    pairs = _load_pairs(n_pairs)
    start = time.time()

    raw: list[dict] = []
    for i, pair in enumerate(pairs):
        row = await _eval_pair(pair, method)
        raw.append(row)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(pairs)} pairs done", flush=True)
            await asyncio.sleep(0.3)

    metrics = _compute_claimlevel_metrics(raw, time.time() - start)
    return ClaimLevelRun(method=method, pairs=len(pairs), metrics=metrics, raw_results=raw)


async def _eval_pair(pair: dict, method: str) -> dict:
    """Run ChainCheck on both the correct and hallucinated answer for one pair."""
    from chaincheck.detect import detect

    correct_result = await detect(
        pair["correct_answer"],
        context=pair["context"],
        prompt=pair["question"],
        methods=[method],
    )
    halluc_result = await detect(
        pair["hallucinated_answer"],
        context=pair["context"],
        prompt=pair["question"],
        methods=[method],
    )

    bad_labels = {"contradicted", "unsupported"}

    def _claim_scores(result) -> list[float]:
        for mr in result.method_results.values():
            return [c.confidence if c.label in bad_labels else 0.0 for c in mr.claims]
        return []

    def _flagged_count(result) -> tuple[int, int]:
        for mr in result.method_results.values():
            flagged = sum(1 for c in mr.claims if c.label in bad_labels)
            return flagged, len(mr.claims)
        return 0, 0

    c_flagged, c_total = _flagged_count(correct_result)
    h_flagged, h_total = _flagged_count(halluc_result)

    return {
        "question": pair["question"][:100],
        "correct_score": correct_result.aggregate_score,
        "halluc_score": halluc_result.aggregate_score,
        "correct_claims": c_total,
        "halluc_claims": h_total,
        "correct_flagged": c_flagged,
        "halluc_flagged": h_flagged,
        "correct_claim_scores": _claim_scores(correct_result),
        "halluc_claim_scores": _claim_scores(halluc_result),
    }


def _compute_claimlevel_metrics(raw: list[dict], elapsed_s: float) -> ClaimLevelMetrics:
    """Aggregate per-pair results into claim-level discrimination metrics."""
    total_clean_claims = sum(r["correct_claims"] for r in raw)
    total_halluc_claims = sum(r["halluc_claims"] for r in raw)
    total_clean_flagged = sum(r["correct_flagged"] for r in raw)
    total_halluc_flagged = sum(r["halluc_flagged"] for r in raw)

    clean_rate = total_clean_flagged / total_clean_claims if total_clean_claims else 0.0
    halluc_rate = total_halluc_flagged / total_halluc_claims if total_halluc_claims else 0.0
    ratio = halluc_rate / clean_rate if clean_rate > 0 else float("inf")

    # Claim-level AUC: scores from correct responses → label 0, hallucinated → label 1
    all_scores: list[float] = []
    all_labels: list[int] = []
    for r in raw:
        all_scores.extend(r["correct_claim_scores"])
        all_labels.extend([0] * len(r["correct_claim_scores"]))
        all_scores.extend(r["halluc_claim_scores"])
        all_labels.extend([1] * len(r["halluc_claim_scores"]))

    auc = _roc_auc(all_labels, all_scores) if all_scores else 0.5

    n_total_claims = total_clean_claims + total_halluc_claims
    avg_per_response = n_total_claims / (2 * len(raw)) if raw else 0.0

    return ClaimLevelMetrics(
        clean_flagging_rate=clean_rate,
        halluc_flagging_rate=halluc_rate,
        discrimination_ratio=ratio,
        claim_auc=auc,
        n_pairs=len(raw),
        n_clean_claims=total_clean_claims,
        n_halluc_claims=total_halluc_claims,
        avg_claims_per_response=avg_per_response,
        latency_ms=elapsed_s * 1000,
    )


def _roc_auc(labels: list[int], scores: list[float]) -> float:
    """Compute ROC-AUC via the trapezoidal rule without sklearn dependency."""
    paired = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = 0
    auc = 0.0
    prev_fp = 0
    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp  # area under step
    return auc / (n_pos * n_neg)


def _load_pairs(n: int) -> list[dict]:
    """Load n (correct_answer, hallucinated_answer) pairs from HaluEval QA split."""
    from datasets import load_dataset

    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    pairs: list[dict] = []
    for row in ds:
        if len(pairs) >= n:
            break
        context = str(row.get("knowledge", "") or "")
        question = str(row.get("question", "") or "")
        right = str(row.get("right_answer", "") or "")
        hallucinated = str(row.get("hallucinated_answer", "") or "")
        if right and hallucinated and context:
            pairs.append({
                "question": question,
                "context": context,
                "correct_answer": right,
                "hallucinated_answer": hallucinated,
            })
    return pairs[:n]
