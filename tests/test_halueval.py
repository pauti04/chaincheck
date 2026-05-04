"""Tests for chaincheck.eval.halueval — HaluEval benchmark runner."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chaincheck.eval.halueval import EvalRun, EvalSample, _load_samples, run_halueval


class TestRunHalueval:
    @pytest.mark.asyncio
    async def test_returns_eval_run(self):
        fake_samples = [
            EvalSample(question="q?", context="ctx", response="correct", ground_truth="no"),
            EvalSample(question="q?", context="ctx", response="hallucinated", ground_truth="yes"),
        ]
        fake_result = MagicMock()
        fake_result.aggregate_score = 0.8

        with (
            patch("chaincheck.eval.halueval._load_samples", return_value=fake_samples),
            patch("chaincheck.detect.detect", new=AsyncMock(return_value=fake_result)),
        ):
            run = await run_halueval(method="nli", n_samples=2)

        assert isinstance(run, EvalRun)
        assert run.samples == 2
        assert run.method == "nli"

    @pytest.mark.asyncio
    async def test_raw_results_have_required_keys(self):
        fake_samples = [
            EvalSample(question="What is X?", context="ctx", response="X is Y.", ground_truth="no"),
        ]
        fake_result = MagicMock()
        fake_result.aggregate_score = 0.3

        with (
            patch("chaincheck.eval.halueval._load_samples", return_value=fake_samples),
            patch("chaincheck.detect.detect", new=AsyncMock(return_value=fake_result)),
        ):
            run = await run_halueval(method="judge", n_samples=1)

        r = run.raw_results[0]
        assert r["question"] == "What is X?"
        assert r["ground_truth"] == "no"
        assert r["predicted"] in {"yes", "no"}
        assert "latency_ms" in r
        assert "score" in r

    @pytest.mark.asyncio
    async def test_prediction_threshold(self):
        fake_samples = [
            EvalSample(question="q?", context="ctx", response="r1", ground_truth="yes"),
            EvalSample(question="q?", context="ctx", response="r2", ground_truth="no"),
        ]
        high = MagicMock()
        high.aggregate_score = 0.9
        low = MagicMock()
        low.aggregate_score = 0.1

        with (
            patch("chaincheck.eval.halueval._load_samples", return_value=fake_samples),
            patch("chaincheck.detect.detect", new=AsyncMock(side_effect=[high, low])),
        ):
            run = await run_halueval(method="judge", n_samples=2)

        preds = [r["predicted"] for r in run.raw_results]
        assert preds[0] == "yes"
        assert preds[1] == "no"

    @pytest.mark.asyncio
    async def test_metrics_computed(self):
        # Perfect predictions: both correct
        fake_samples = [
            EvalSample(question="q?", context="ctx", response="r1", ground_truth="yes"),
            EvalSample(question="q?", context="ctx", response="r2", ground_truth="no"),
        ]
        high = MagicMock()
        high.aggregate_score = 0.9
        low = MagicMock()
        low.aggregate_score = 0.1

        with (
            patch("chaincheck.eval.halueval._load_samples", return_value=fake_samples),
            patch("chaincheck.detect.detect", new=AsyncMock(side_effect=[high, low])),
        ):
            run = await run_halueval(method="judge", n_samples=2)

        assert run.metrics.f1 == pytest.approx(1.0, abs=1e-6)
        assert run.metrics.precision == pytest.approx(1.0, abs=1e-6)


class TestLoadSamples:
    def test_returns_list_of_eval_samples(self):
        fake_row = {
            "question": "What is X?",
            "knowledge": "X is defined as Y.",
            "right_answer": "X is Y.",
            "hallucinated_answer": "X is Z.",
        }
        mock_ds = [fake_row] * 5

        with patch("datasets.load_dataset", return_value=mock_ds):
            samples = _load_samples("qa", 4)

        assert len(samples) <= 4
        assert all(isinstance(s, EvalSample) for s in samples)

    def test_balanced_labels(self):
        fake_row = {
            "question": "q?",
            "knowledge": "ctx",
            "right_answer": "right",
            "hallucinated_answer": "wrong",
        }
        mock_ds = [fake_row] * 10

        with patch("datasets.load_dataset", return_value=mock_ds):
            samples = _load_samples("qa", 4)

        labels = [s.ground_truth for s in samples]
        assert labels.count("yes") == labels.count("no")

    def test_caps_at_n(self):
        fake_row = {
            "question": "q?",
            "knowledge": "ctx",
            "right_answer": "right",
            "hallucinated_answer": "wrong",
        }
        mock_ds = [fake_row] * 100

        with patch("datasets.load_dataset", return_value=mock_ds):
            samples = _load_samples("qa", 6)

        assert len(samples) == 6

    def test_skips_empty_individual_answers(self):
        rows = [
            # only hallucinated answer → 1 sample (yes)
            {"question": "q?", "knowledge": "ctx", "right_answer": "", "hallucinated_answer": "w"},
            # only right answer → 1 sample (no)
            {"question": "q?", "knowledge": "ctx", "right_answer": "r", "hallucinated_answer": ""},
        ]

        with patch("datasets.load_dataset", return_value=rows):
            samples = _load_samples("qa", 10)

        # Each row contributes only 1 sample (not 2)
        assert len(samples) == 2
        labels = [s.ground_truth for s in samples]
        assert "yes" in labels
        assert "no" in labels
