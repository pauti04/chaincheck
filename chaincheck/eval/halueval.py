"""HaluEval benchmark runner for ChainCheck."""

from __future__ import annotations

import math
import os
import time
from typing import Any

from chaincheck.detect import detect
from chaincheck.eval.metrics import classification_metrics, latency_metrics

HALUEVAL_QA_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"


async def run_halueval(method: str = "nli", samples: int = 500) -> dict[str, object]:
    """Run a HaluEval subset and return aggregate metrics."""
    rows = _load_rows(samples)
    predictions: list[bool] = []
    labels: list[bool] = []
    latencies: list[float] = []
    results: list[dict[str, Any]] = []
    for row in rows:
        started = time.perf_counter()
        result = await detect(
            row["answer"], context=row["knowledge"], prompt=row["question"], methods=[method]
        )
        latency_ms = (time.perf_counter() - started) * 1000
        prediction = result.aggregate_score >= 0.5
        predictions.append(prediction)
        labels.append(row["hallucinated"])
        latencies.append(latency_ms)
        results.append(_row_result(row, result.aggregate_score, prediction, latency_ms))
    metrics = classification_metrics(predictions, labels) | latency_metrics(latencies)
    return {"method": method, "samples": len(rows), "metrics": metrics, "results": results}


def _load_rows(samples: int) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset

        source_count = max(1, math.ceil(samples / 2))
        dataset = load_dataset("json", data_files=HALUEVAL_QA_URL, split="train")
        selected = dataset.select(range(min(source_count, len(dataset))))
        return _expand_rows([dict(row) for row in selected])[:samples]
    except (ImportError, OSError, ValueError, ConnectionError) as exc:
        if os.getenv("CHAINCHECK_EVAL_ALLOW_FALLBACK") == "1":
            return _fallback_rows()[:samples]
        raise RuntimeError("Could not load HaluEval QA data. Check network access.") from exc


def _expand_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        expanded.append(_qa_row(row, answer_key="right_answer", hallucinated=False))
        expanded.append(_qa_row(row, answer_key="hallucinated_answer", hallucinated=True))
    return expanded


def _qa_row(row: dict[str, Any], answer_key: str, hallucinated: bool) -> dict[str, Any]:
    return {
        "question": str(row.get("question", "")),
        "knowledge": str(row.get("knowledge", "")),
        "answer": str(row.get(answer_key, "")),
        "hallucinated": hallucinated,
    }


def _fallback_rows() -> list[dict[str, Any]]:
    return [
        {
            "question": "What is ChainCheck?",
            "knowledge": "ChainCheck detects unsupported claims in model responses.",
            "answer": "ChainCheck detects unsupported claims in model responses.",
            "hallucinated": False,
        },
        {
            "question": "What is ChainCheck?",
            "knowledge": "ChainCheck detects unsupported claims in model responses.",
            "answer": "ChainCheck is a database created in 1998 by NASA.",
            "hallucinated": True,
        },
    ]


def _row_result(
    row: dict[str, Any], score: float, prediction: bool, latency_ms: float
) -> dict[str, Any]:
    return {
        "question": row["question"],
        "score": score,
        "prediction": prediction,
        "latency_ms": latency_ms,
    }
