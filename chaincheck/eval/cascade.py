"""
Cascade cost-accuracy tradeoff analysis.

Simulates the NLI-first cascade across a grid of escalation thresholds
using pre-computed eval results, without making any API calls.

The cascade logic:
  - If NLI score < low_threshold  → use NLI prediction (fast path, "clearly clean")
  - If NLI score > high_threshold → use NLI prediction (fast path, "clearly hallucinated")
  - Otherwise                     → combine NLI + judge (slow path, ambiguous band)

For each (low, high) threshold pair we compute:
  - F1 on the full 500-sample eval set
  - Average latency (fraction_escalated × judge_latency + rest × nli_latency)

The result is a Pareto frontier: the set of threshold pairs where no other pair
achieves both higher F1 AND lower latency.

IMPORTANT BENCHMARK NOTE:
HaluEval scores are near-binary (≈97% of samples score either 0.0 or 1.0) because
HaluEval responses are either entirely correct or entirely hallucinated — never mixed.
This means the cascade escalation rate is ~0% on HaluEval, making it appear as if
the cascade offers no benefit. In real-world RAG, responses contain partial
hallucinations (some claims correct, some wrong), producing continuous scores in
[0.2, 0.8] where the cascade saves the most latency. The Pareto frontier from this
analysis is most useful as a planning tool for real-world tuning, not as a benchmark result.

Usage:
    from chaincheck.eval.cascade import run_cascade_analysis, pareto_frontier
    results = run_cascade_analysis("nli_eval_results.json", "judge_eval_results.json")
    frontier = pareto_frontier(results)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CascadePoint:
    """Result for one (low, high) threshold pair."""

    low: float
    high: float
    f1: float
    precision: float
    recall: float
    avg_latency_ms: float
    escalation_rate: float   # fraction of samples that ran judge


@dataclass
class CascadeAnalysis:
    """Full grid of cascade simulation results."""

    points: list[CascadePoint]
    frontier: list[CascadePoint]     # Pareto-optimal points
    optimal: CascadePoint            # highest F1 on the frontier
    nli_only: CascadePoint           # baseline: NLI with no escalation
    both_methods: CascadePoint       # baseline: always run both


def run_cascade_analysis(
    nli_path: str | Path = "nli_eval_results.json",
    judge_path: str | Path = "judge_eval_results.json",
    grid_steps: int = 9,
) -> CascadeAnalysis:
    """
    Simulate cascade at every point on a threshold grid.

    Args:
        nli_path: Path to NLI eval results JSON.
        judge_path: Path to judge eval results JSON.
        grid_steps: Number of steps per axis; 9 gives 0.1, 0.2, …, 0.9.

    Returns:
        CascadeAnalysis with full grid, Pareto frontier, and baselines.
    """
    nli_results = json.loads(Path(nli_path).read_text())["raw_results"]
    judge_results = json.loads(Path(judge_path).read_text())["raw_results"]

    nli_scores = np.array([r["score"] for r in nli_results])
    judge_scores = np.array([r["score"] for r in judge_results])
    y_true = np.array([1 if r["ground_truth"] == "yes" else 0 for r in nli_results])
    nli_latencies = np.array([r["latency_ms"] for r in nli_results])
    judge_latencies = np.array([r["latency_ms"] for r in judge_results])

    # Ensemble weights (normalised NLI+judge subset)
    w_nli, w_judge = 0.10, 0.60
    total = w_nli + w_judge
    w_nli_n, w_judge_n = w_nli / total, w_judge / total

    thresholds = np.linspace(0.1, 0.9, grid_steps)
    points: list[CascadePoint] = []

    for low in thresholds:
        for high in thresholds:
            if low >= high:
                continue
            escalate = (nli_scores >= low) & (nli_scores <= high)
            # Prediction: escalated → weighted ensemble; fast path → NLI alone
            combined = np.where(
                escalate,
                w_nli_n * nli_scores + w_judge_n * judge_scores,
                nli_scores,
            )
            preds = (combined >= 0.5).astype(int)
            tp = int(((preds == 1) & (y_true == 1)).sum())
            fp = int(((preds == 1) & (y_true == 0)).sum())
            fn = int(((preds == 0) & (y_true == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            esc_rate = float(escalate.mean())
            avg_lat = float(
                np.where(escalate, nli_latencies + judge_latencies, nli_latencies).mean()
            )
            points.append(CascadePoint(
                low=round(float(low), 2),
                high=round(float(high), 2),
                f1=round(f1, 4),
                precision=round(prec, 4),
                recall=round(rec, 4),
                avg_latency_ms=round(avg_lat, 1),
                escalation_rate=round(esc_rate, 3),
            ))

    # Baselines
    nli_preds = (nli_scores >= 0.5).astype(int)
    nli_only = _point_from_preds(nli_preds, y_true, float(nli_latencies.mean()), 0.0, 0.0, 1.0)

    ensemble_scores = w_nli_n * nli_scores + w_judge_n * judge_scores
    both_preds = (ensemble_scores >= 0.5).astype(int)
    both_lat = float((nli_latencies + judge_latencies).mean())
    both = _point_from_preds(both_preds, y_true, both_lat, 0.0, 1.0, 1.0)

    frontier = pareto_frontier(points)
    optimal = max(frontier, key=lambda p: p.f1)

    return CascadeAnalysis(
        points=points,
        frontier=frontier,
        optimal=optimal,
        nli_only=nli_only,
        both_methods=both,
    )


def pareto_frontier(points: list[CascadePoint]) -> list[CascadePoint]:
    """
    Return Pareto-optimal points: no other point has both higher F1 and lower latency.

    Sorted by ascending latency.
    """
    sorted_pts = sorted(points, key=lambda p: p.avg_latency_ms)
    frontier: list[CascadePoint] = []
    best_f1 = -1.0
    for p in sorted_pts:
        if p.f1 > best_f1:
            best_f1 = p.f1
            frontier.append(p)
    return frontier


def _point_from_preds(
    preds: np.ndarray,
    y_true: np.ndarray,
    avg_lat: float,
    low: float,
    high: float,
    esc_rate: float,
) -> CascadePoint:
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return CascadePoint(
        low=low, high=high,
        f1=round(f1, 4), precision=round(prec, 4), recall=round(rec, 4),
        avg_latency_ms=round(avg_lat, 1), escalation_rate=round(esc_rate, 3),
    )
