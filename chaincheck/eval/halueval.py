"""
HaluEval benchmark runner.

Loads the HaluEval QA split from HuggingFace and evaluates a specified
detection method across a configurable number of samples. Each dataset row
yields two samples (right answer → label "no", hallucinated answer → label "yes")
so the evaluation set is balanced.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from chaincheck.eval.metrics import EvalMetrics, compute_metrics

_HALUEVAL_REPO = "pminervini/HaluEval"
_DEFAULT_SPLIT = "qa"


@dataclass
class EvalSample:
    """A single HaluEval sample with ground-truth label."""

    question: str
    context: str
    response: str
    ground_truth: Literal["yes", "no"]  # yes = hallucinated


@dataclass
class EvalRun:
    """Results from a full benchmark evaluation run."""

    method: str
    samples: int
    metrics: EvalMetrics
    raw_results: list[dict] = field(default_factory=list)


async def run_halueval(
    method: Literal["nli", "consistency", "judge"],
    n_samples: int = 500,
) -> EvalRun:
    """
    Evaluate a detection method against the HaluEval QA split.

    Args:
        method: Detection method identifier.
        n_samples: Number of samples to evaluate (500 for speed, 2000 for full).

    Returns:
        EvalRun with precision, recall, F1, accuracy, and latency stats.
    """
    from chaincheck.detect import detect

    samples = _load_samples(_DEFAULT_SPLIT, n_samples)

    y_true: list[str] = []
    y_pred: list[str] = []
    latencies: list[float] = []
    raw_results: list[dict] = []

    for sample in samples:
        start = time.time()
        result = await detect(
            sample.response,
            context=sample.context,
            prompt=sample.question,
            methods=[method],
        )
        latency = (time.time() - start) * 1000

        predicted = _predict_label(result.aggregate_score)
        y_true.append(sample.ground_truth)
        y_pred.append(predicted)
        latencies.append(latency)
        raw_results.append(
            {
                "question": sample.question,
                "response": sample.response[:200],
                "ground_truth": sample.ground_truth,
                "predicted": predicted,
                "score": result.aggregate_score,
                "latency_ms": latency,
            }
        )

    metrics = compute_metrics(y_true, y_pred, latencies)
    return EvalRun(method=method, samples=len(samples), metrics=metrics, raw_results=raw_results)


def _load_samples(split: str, n: int) -> list[EvalSample]:
    """
    Load and convert n samples from the HaluEval dataset on HuggingFace.

    Each row yields two EvalSamples: one with the right answer (label "no")
    and one with the hallucinated answer (label "yes"), capped at n total.
    """
    from datasets import load_dataset

    ds = load_dataset(_HALUEVAL_REPO, split, split="data", trust_remote_code=True)
    samples: list[EvalSample] = []

    for row in ds:
        if len(samples) >= n:
            break
        context = str(row.get("knowledge", "") or "")
        question = str(row.get("question", "") or "")
        right = str(row.get("right_answer", "") or "")
        hallucinated = str(row.get("hallucinated_answer", "") or "")

        if right and len(samples) < n:
            samples.append(
                EvalSample(question=question, context=context, response=right, ground_truth="no")
            )
        if hallucinated and len(samples) < n:
            samples.append(
                EvalSample(
                    question=question,
                    context=context,
                    response=hallucinated,
                    ground_truth="yes",
                )
            )

    return samples[:n]


def _predict_label(score: float, threshold: float = 0.5) -> Literal["yes", "no"]:
    """Convert a continuous risk score to a binary hallucination label."""
    return "yes" if score >= threshold else "no"
