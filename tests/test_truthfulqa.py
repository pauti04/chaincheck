"""Tests for chaincheck.eval.truthfulqa — TruthfulQA benchmark runner."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chaincheck.eval.halueval import EvalRun
from chaincheck.eval.truthfulqa import TruthfulQASample, _load_samples, run_truthfulqa


class TestRunTruthfulqa:
    def test_nli_raises_value_error(self):
        import asyncio
        with pytest.raises(ValueError, match="NLI requires a reference context"):
            asyncio.run(run_truthfulqa(method="nli"))

    @pytest.mark.asyncio
    async def test_returns_eval_run(self):
        fake_samples = [
            TruthfulQASample(question="q?", response="correct", ground_truth="no"),
            TruthfulQASample(question="q?", response="wrong", ground_truth="yes"),
        ]
        fake_result = MagicMock()
        fake_result.aggregate_score = 0.8

        with (
            patch("chaincheck.eval.truthfulqa._load_samples", return_value=fake_samples),
            patch("chaincheck.detect.detect", new=AsyncMock(return_value=fake_result)),
        ):
            run = await run_truthfulqa(method="judge", n_samples=2)

        assert isinstance(run, EvalRun)
        assert run.samples == 2
        assert run.method == "truthfulqa/judge"

    @pytest.mark.asyncio
    async def test_predictions_based_on_threshold(self):
        fake_samples = [
            TruthfulQASample(question="q?", response="r1", ground_truth="yes"),
            TruthfulQASample(question="q?", response="r2", ground_truth="no"),
        ]

        high_result = MagicMock()
        high_result.aggregate_score = 0.9  # → predicted "yes"
        low_result = MagicMock()
        low_result.aggregate_score = 0.1   # → predicted "no"

        with (
            patch("chaincheck.eval.truthfulqa._load_samples", return_value=fake_samples),
            patch("chaincheck.detect.detect", new=AsyncMock(
                side_effect=[high_result, low_result]
            )),
        ):
            run = await run_truthfulqa(method="judge", n_samples=2)

        preds = [r["predicted"] for r in run.raw_results]
        assert preds[0] == "yes"
        assert preds[1] == "no"

    @pytest.mark.asyncio
    async def test_raw_results_include_required_keys(self):
        fake_samples = [
            TruthfulQASample(question="What is X?", response="X is Y.", ground_truth="no",
                             category="Science"),
        ]
        fake_result = MagicMock()
        fake_result.aggregate_score = 0.3

        with (
            patch("chaincheck.eval.truthfulqa._load_samples", return_value=fake_samples),
            patch("chaincheck.detect.detect", new=AsyncMock(return_value=fake_result)),
        ):
            run = await run_truthfulqa(method="judge", n_samples=1)

        r = run.raw_results[0]
        assert r["question"] == "What is X?"
        assert r["ground_truth"] == "no"
        assert r["category"] == "Science"
        assert "latency_ms" in r


class TestLoadSamples:
    def test_returns_list_of_samples(self):
        fake_row = {
            "question": "What is X?",
            "best_answer": "X is Y.",
            "incorrect_answers": ["X is Z.", "X is W."],
            "category": "Science",
        }
        mock_ds = [fake_row] * 5

        with patch("datasets.load_dataset", return_value=mock_ds):
            samples = _load_samples(4)

        assert len(samples) <= 4
        assert all(isinstance(s, TruthfulQASample) for s in samples)

    def test_balanced_labels(self):
        fake_row = {
            "question": "q?",
            "best_answer": "correct answer",
            "incorrect_answers": ["wrong answer"],
            "category": "test",
        }
        mock_ds = [fake_row] * 10

        with patch("datasets.load_dataset", return_value=mock_ds):
            samples = _load_samples(4)

        labels = [s.ground_truth for s in samples]
        assert labels.count("yes") == labels.count("no")

    def test_caps_at_n(self):
        fake_row = {
            "question": "q?",
            "best_answer": "best",
            "incorrect_answers": ["wrong"],
            "category": "",
        }
        mock_ds = [fake_row] * 100

        with patch("datasets.load_dataset", return_value=mock_ds):
            samples = _load_samples(6)

        assert len(samples) == 6
