"""Tests for benchmark evaluation helpers."""

import pytest

from chaincheck.eval.halueval import run_halueval
from chaincheck.eval.metrics import classification_metrics, latency_metrics


def test_classification_metrics() -> None:
    """Classification metrics compute true positives and misses."""
    metrics = classification_metrics([True, False, True], [True, True, False])
    assert metrics["precision"] == 0.5


def test_latency_metrics() -> None:
    """Latency metrics include average and percentile values."""
    metrics = latency_metrics([1.0, 2.0, 3.0])
    assert metrics["avg_latency_ms"] == 2.0


@pytest.mark.asyncio
async def test_run_halueval_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """HaluEval runner works with fallback rows."""
    from chaincheck.eval import halueval

    monkeypatch.setattr(halueval, "_load_rows", lambda samples: halueval._fallback_rows()[:samples])
    monkeypatch.setattr(halueval, "detect", _fake_detect)
    report = await run_halueval(samples=1)
    assert report["samples"] == 1


def test_expand_rows_creates_supported_and_hallucinated_examples() -> None:
    """HaluEval QA rows expand into binary examples."""
    from chaincheck.eval.halueval import _expand_rows

    rows = [
        {
            "question": "Q",
            "knowledge": "K",
            "right_answer": "Right",
            "hallucinated_answer": "Wrong",
        }
    ]
    expanded = _expand_rows(rows)
    assert expanded[0]["hallucinated"] is False
    assert expanded[1]["hallucinated"] is True


async def _fake_detect(response: str, context: str = "", prompt: str = "", methods=None):
    del context, prompt, methods
    from chaincheck.detect import DetectionResult

    return DetectionResult(response=response, aggregate_score=0.2)
