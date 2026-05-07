"""
TruthfulQA benchmark runner.

Evaluates ChainCheck against the TruthfulQA generation split — 817 adversarial
questions designed to elicit confident-but-wrong answers from LLMs.

Unlike HaluEval, TruthfulQA has no retrieved context, so only context-free
methods run (judge). This directly measures claim-level hallucination detection
on the hardest class of errors: confident confabulations with no grounding doc.

Each question yields two EvalSamples:
  - best_answer → ground truth "no" (truthful)
  - incorrect_answers[0] → ground truth "yes" (hallucinated)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Literal

from chaincheck.eval.halueval import EvalRun
from chaincheck.eval.metrics import compute_metrics

_TRUTHFULQA_REPO = "truthful_qa"
_TRUTHFULQA_CONFIG = "generation"
_TRUTHFULQA_SPLIT = "validation"


@dataclass
class TruthfulQASample:
    question: str
    response: str
    ground_truth: Literal["yes", "no"]
    category: str = ""


async def run_truthfulqa(
    method: Literal["judge", "consistency"] = "judge",
    n_samples: int = 200,
) -> EvalRun:
    """
    Evaluate a detection method against TruthfulQA (generation split).

    Args:
        method: Detection method — only context-free methods make sense here.
                "nli" is silently excluded since there is no reference document.
        n_samples: Number of samples to evaluate.

    Returns:
        EvalRun with precision, recall, F1, accuracy, and latency stats.
    """
    from chaincheck.detect import detect

    if method == "nli":
        raise ValueError(
            "NLI requires a reference context document; TruthfulQA has none. "
            "Use method='judge' instead."
        )

    samples = _load_samples(n_samples)

    async def _eval_one(sample: TruthfulQASample) -> dict:
        start = time.time()
        result = await detect(
            sample.response,
            context="",
            prompt=sample.question,
            methods=[method],
        )
        latency = (time.time() - start) * 1000
        return {
            "question": sample.question,
            "response": sample.response[:200],
            "category": sample.category,
            "ground_truth": sample.ground_truth,
            "predicted": "yes" if result.aggregate_score >= 0.5 else "no",
            "score": result.aggregate_score,
            "latency_ms": latency,
        }

    raw_results: list[dict] = []
    for i, sample in enumerate(samples):
        raw_results.append(await _eval_one(sample))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(samples)} done", flush=True)
            await asyncio.sleep(0.5)

    y_true = [r["ground_truth"] for r in raw_results]
    y_pred = [r["predicted"] for r in raw_results]
    latencies = [r["latency_ms"] for r in raw_results]
    scores = [r["score"] for r in raw_results]

    metrics = compute_metrics(y_true, y_pred, latencies, scores=scores)
    return EvalRun(
        method=f"truthfulqa/{method}",
        samples=len(raw_results),
        metrics=metrics,
        raw_results=raw_results,
    )


def _load_samples(n: int) -> list[TruthfulQASample]:
    """
    Load n balanced samples from TruthfulQA validation split.

    Each question contributes one truthful sample (best_answer, label="no")
    and one hallucinated sample (incorrect_answers[0], label="yes").
    """
    from datasets import load_dataset

    ds = load_dataset(_TRUTHFULQA_REPO, _TRUTHFULQA_CONFIG, split=_TRUTHFULQA_SPLIT)
    samples: list[TruthfulQASample] = []

    for row in ds:
        if len(samples) >= n:
            break
        question = str(row.get("question", "") or "")
        best = str(row.get("best_answer", "") or "")
        wrongs: list = row.get("incorrect_answers", []) or []
        category = str(row.get("category", "") or "")

        if best and len(samples) < n:
            samples.append(TruthfulQASample(question=question, response=best,
                                             ground_truth="no", category=category))
        if wrongs and len(samples) < n:
            samples.append(TruthfulQASample(question=question, response=str(wrongs[0]),
                                             ground_truth="yes", category=category))

    return samples[:n]
