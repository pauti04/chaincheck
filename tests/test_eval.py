"""Tests for eval/metrics.py and eval/report.py."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from chaincheck.eval.claimlevel import ClaimLevelMetrics, ClaimLevelRun
from chaincheck.eval.metrics import EvalMetrics, _compute_ece, compute_metrics
from chaincheck.eval.report import (
    print_claimlevel_report,
    print_report,
    save_claimlevel_report,
    save_report,
)


def _metrics(**kwargs) -> EvalMetrics:
    defaults = {
        "precision": 0.8,
        "recall": 0.75,
        "f1": 0.77,
        "accuracy": 0.82,
        "avg_latency_ms": 200.0,
        "p50_latency_ms": 190.0,
        "p95_latency_ms": 450.0,
        "n_samples": 100,
    }
    defaults.update(kwargs)
    return EvalMetrics(**defaults)


@dataclass
class _FakeRun:
    method: str = "nli"
    samples: int = 100
    metrics: EvalMetrics = field(default_factory=_metrics)
    raw_results: list[dict] = field(default_factory=list)


class TestComputeMetrics:
    def test_all_correct(self):
        y_true = ["yes", "yes", "no", "no"]
        y_pred = ["yes", "yes", "no", "no"]
        latencies = [100.0, 110.0, 90.0, 95.0]
        m = compute_metrics(y_true, y_pred, latencies)
        assert m.precision == pytest.approx(1.0, abs=1e-6)
        assert m.recall == pytest.approx(1.0, abs=1e-6)
        assert m.f1 == pytest.approx(1.0, abs=1e-6)
        assert m.accuracy == pytest.approx(1.0, abs=1e-6)

    def test_all_wrong(self):
        y_true = ["yes", "yes", "no", "no"]
        y_pred = ["no", "no", "yes", "yes"]
        latencies = [100.0] * 4
        m = compute_metrics(y_true, y_pred, latencies)
        assert m.precision == pytest.approx(0.0, abs=1e-6)
        assert m.recall == pytest.approx(0.0, abs=1e-6)
        assert m.f1 == pytest.approx(0.0, abs=1e-6)
        assert m.accuracy == pytest.approx(0.0, abs=1e-6)

    def test_mixed(self):
        y_true = ["yes", "no", "yes", "no"]
        y_pred = ["yes", "no", "no", "yes"]
        latencies = [100.0] * 4
        m = compute_metrics(y_true, y_pred, latencies)
        # TP=1, FP=1, FN=1, TN=1
        assert m.precision == pytest.approx(0.5, abs=1e-6)
        assert m.recall == pytest.approx(0.5, abs=1e-6)
        assert m.accuracy == pytest.approx(0.5, abs=1e-6)

    def test_latency_stats(self):
        y_true = ["yes", "no"]
        y_pred = ["yes", "no"]
        latencies = [100.0, 200.0]
        m = compute_metrics(y_true, y_pred, latencies)
        assert m.avg_latency_ms == pytest.approx(150.0, abs=0.1)
        assert m.p50_latency_ms == pytest.approx(150.0, abs=1.0)

    def test_n_samples(self):
        y_true = ["yes"] * 10
        y_pred = ["yes"] * 10
        m = compute_metrics(y_true, y_pred, [100.0] * 10)
        assert m.n_samples == 10

    def test_empty_tp_fp(self):
        # All predictions are "no" but truth is "yes"
        y_true = ["yes", "yes"]
        y_pred = ["no", "no"]
        latencies = [50.0, 60.0]
        m = compute_metrics(y_true, y_pred, latencies)
        assert m.precision == pytest.approx(0.0, abs=1e-6)
        assert m.recall == pytest.approx(0.0, abs=1e-6)


class TestSaveReport:
    def test_creates_json_file(self):
        run = _FakeRun()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        save_report(run, path)
        data = json.loads(path.read_text())
        assert data["method"] == "nli"
        assert data["samples"] == 100
        assert "precision" in data["metrics"]
        path.unlink()

    def test_metrics_written_correctly(self):
        run = _FakeRun(metrics=_metrics(f1=0.81))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        save_report(run, path)
        data = json.loads(path.read_text())
        assert data["metrics"]["f1"] == pytest.approx(0.81, abs=1e-6)
        path.unlink()

    def test_raw_results_included(self):
        run = _FakeRun(raw_results=[{"question": "q", "score": 0.5}])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        save_report(run, path)
        data = json.loads(path.read_text())
        assert len(data["raw_results"]) == 1
        path.unlink()


class TestComputeEce:
    def test_perfect_calibration(self):
        # Score exactly 1.0 for all positives → confidence == accuracy == 1.0 → ECE = 0
        y_true = ["yes", "yes", "yes"]
        scores = [1.0, 1.0, 1.0]
        ece = _compute_ece(y_true, scores, "yes", n_bins=10)
        assert ece == pytest.approx(0.0, abs=1e-6)

    def test_worst_calibration(self):
        # High confidence, all wrong → ECE close to confidence level
        y_true = ["no", "no", "no"]
        scores = [0.95, 0.95, 0.95]
        ece = _compute_ece(y_true, scores, "yes", n_bins=10)
        assert ece > 0.5

    def test_empty_scores(self):
        assert _compute_ece([], [], "yes", n_bins=10) == pytest.approx(0.0, abs=1e-6)

    def test_returns_float(self):
        y_true = ["yes", "no", "yes", "no"]
        scores = [0.8, 0.2, 0.7, 0.3]
        result = _compute_ece(y_true, scores, "yes", n_bins=10)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_compute_metrics_with_scores_sets_ece(self):
        y_true = ["yes", "yes", "no", "no"]
        y_pred = ["yes", "yes", "no", "no"]
        scores = [0.9, 0.85, 0.1, 0.15]
        latencies = [100.0] * 4
        m = compute_metrics(y_true, y_pred, latencies, scores=scores)
        assert m.ece == pytest.approx(0.0, abs=0.15)

    def test_compute_metrics_without_scores_ece_is_zero(self):
        y_true = ["yes", "no"]
        y_pred = ["yes", "no"]
        m = compute_metrics(y_true, y_pred, [100.0, 100.0])
        assert m.ece == pytest.approx(0.0, abs=1e-6)


def _fake_claimlevel_run(pairs: int = 5) -> ClaimLevelRun:
    metrics = ClaimLevelMetrics(
        clean_flagging_rate=0.1,
        halluc_flagging_rate=0.6,
        discrimination_ratio=6.0,
        claim_auc=0.85,
        n_pairs=pairs,
        n_clean_claims=20,
        n_halluc_claims=22,
        avg_claims_per_response=4.2,
        latency_ms=1234.5,
    )
    return ClaimLevelRun(method="nli", pairs=pairs, metrics=metrics)


class TestSaveClaimlevelReport:
    def test_creates_json_file(self):
        run = _fake_claimlevel_run()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        save_claimlevel_report(run, path)
        data = json.loads(path.read_text())
        assert data["method"] == "nli"
        assert data["pairs"] == 5
        assert "clean_flagging_rate" in data["metrics"]
        path.unlink()

    def test_metrics_written_correctly(self):
        run = _fake_claimlevel_run()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        save_claimlevel_report(run, path)
        data = json.loads(path.read_text())
        assert data["metrics"]["discrimination_ratio"] == pytest.approx(6.0, abs=1e-6)
        assert data["metrics"]["claim_auc"] == pytest.approx(0.85, abs=1e-6)
        path.unlink()

    def test_raw_results_empty(self):
        run = _fake_claimlevel_run()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        save_claimlevel_report(run, path)
        data = json.loads(path.read_text())
        assert data["raw_results"] == []
        path.unlink()


class TestPrintReport:
    def test_runs_without_error(self, capsys):
        run = _FakeRun()
        print_report(run)

    def test_claimlevel_runs_without_error(self):
        run = _fake_claimlevel_run()
        print_claimlevel_report(run)


class TestRunHaluevalMethodRouting:
    @pytest.mark.parametrize(
        ("method", "expected_methods"),
        [
            ("nli", ["nli"]),
            ("judge", ["judge"]),
            ("ensemble", ["nli", "judge"]),
        ],
    )
    async def test_method_maps_to_detect_methods(self, method, expected_methods, monkeypatch):
        from unittest.mock import AsyncMock

        from chaincheck.eval import halueval
        from chaincheck.models import DetectionResult

        sample = halueval.EvalSample(
            question="q", context="ctx", response="resp", ground_truth="no"
        )
        monkeypatch.setattr(halueval, "_load_samples", lambda split, n: [sample])

        fake_result = DetectionResult(
            response="resp", claims=[], aggregate_score=0.1, risk_level="low"
        )
        # chaincheck/__init__.py rebinds the `detect` attribute from the module
        # to the function, so resolve the module via sys.modules instead.
        import sys

        import chaincheck.detect  # noqa: F401

        mock_detect = AsyncMock(return_value=fake_result)
        monkeypatch.setattr(sys.modules["chaincheck.detect"], "detect", mock_detect)

        run = await halueval.run_halueval(method=method, n_samples=1)

        assert mock_detect.await_args.kwargs["methods"] == expected_methods
        assert run.method == method
        assert run.samples == 1
