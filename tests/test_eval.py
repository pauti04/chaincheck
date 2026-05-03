"""Tests for eval/metrics.py and eval/report.py."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from chaincheck.eval.metrics import EvalMetrics, compute_metrics
from chaincheck.eval.report import print_report, save_report


def _metrics(**kwargs) -> EvalMetrics:
    defaults = dict(
        precision=0.8,
        recall=0.75,
        f1=0.77,
        accuracy=0.82,
        avg_latency_ms=200.0,
        p50_latency_ms=190.0,
        p95_latency_ms=450.0,
        n_samples=100,
    )
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


class TestPrintReport:
    def test_runs_without_error(self, capsys):
        run = _FakeRun()
        print_report(run)
        # Just check it doesn't raise — Rich output goes through its own system
