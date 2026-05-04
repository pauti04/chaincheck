"""Tests for chaincheck.eval.cascade — Pareto frontier analysis."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from chaincheck.eval.cascade import CascadePoint, pareto_frontier, run_cascade_analysis


def _make_results(n: int = 10) -> list[dict]:
    """Generate synthetic eval results alternating hallucinated / clean."""
    results = []
    for i in range(n):
        gt = "yes" if i % 2 == 0 else "no"
        score = 1.0 if gt == "yes" else 0.0
        results.append({
            "question": f"q{i}",
            "response": f"r{i}",
            "ground_truth": gt,
            "predicted": gt,
            "score": score,
            "latency_ms": 50.0 if i % 2 == 0 else 1000.0,
        })
    return results


def _write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data))


class TestParetoFrontier:
    def test_empty_input(self):
        assert pareto_frontier([]) == []

    def test_single_point_is_frontier(self):
        p = CascadePoint(0.2, 0.8, f1=0.7, precision=0.8, recall=0.6,
                         avg_latency_ms=100.0, escalation_rate=0.3)
        assert pareto_frontier([p]) == [p]

    def test_dominated_point_excluded(self):
        better = CascadePoint(0.2, 0.8, f1=0.8, precision=0.9, recall=0.7,
                              avg_latency_ms=100.0, escalation_rate=0.3)
        worse = CascadePoint(0.3, 0.7, f1=0.6, precision=0.7, recall=0.5,
                             avg_latency_ms=200.0, escalation_rate=0.5)
        frontier = pareto_frontier([better, worse])
        assert better in frontier
        assert worse not in frontier

    def test_tradeoff_both_on_frontier(self):
        fast_low = CascadePoint(0.1, 0.2, f1=0.5, precision=0.6, recall=0.4,
                                avg_latency_ms=60.0, escalation_rate=0.0)
        slow_high = CascadePoint(0.2, 0.8, f1=0.8, precision=0.9, recall=0.7,
                                 avg_latency_ms=800.0, escalation_rate=0.6)
        frontier = pareto_frontier([fast_low, slow_high])
        assert fast_low in frontier
        assert slow_high in frontier

    def test_sorted_by_latency(self):
        points = [
            CascadePoint(0.3, 0.7, f1=0.8, precision=0.9, recall=0.7,
                         avg_latency_ms=500.0, escalation_rate=0.4),
            CascadePoint(0.1, 0.2, f1=0.5, precision=0.6, recall=0.4,
                         avg_latency_ms=60.0, escalation_rate=0.0),
        ]
        frontier = pareto_frontier(points)
        latencies = [p.avg_latency_ms for p in frontier]
        assert latencies == sorted(latencies)


class TestRunCascadeAnalysis:
    def test_returns_analysis_object(self):
        results = _make_results(20)
        with tempfile.TemporaryDirectory() as d:
            nli_path = Path(d) / "nli.json"
            judge_path = Path(d) / "judge.json"
            _write_json({"raw_results": results}, nli_path)
            _write_json({"raw_results": results}, judge_path)
            analysis = run_cascade_analysis(nli_path, judge_path, grid_steps=3)

        assert analysis.frontier is not None
        assert analysis.optimal is not None
        assert analysis.nli_only is not None
        assert analysis.both_methods is not None

    def test_generates_grid_points(self):
        results = _make_results(20)
        with tempfile.TemporaryDirectory() as d:
            nli_path = Path(d) / "nli.json"
            judge_path = Path(d) / "judge.json"
            _write_json({"raw_results": results}, nli_path)
            _write_json({"raw_results": results}, judge_path)
            analysis = run_cascade_analysis(nli_path, judge_path, grid_steps=3)

        assert len(analysis.points) > 0

    def test_optimal_is_on_frontier(self):
        results = _make_results(20)
        with tempfile.TemporaryDirectory() as d:
            nli_path = Path(d) / "nli.json"
            judge_path = Path(d) / "judge.json"
            _write_json({"raw_results": results}, nli_path)
            _write_json({"raw_results": results}, judge_path)
            analysis = run_cascade_analysis(nli_path, judge_path, grid_steps=3)

        assert analysis.optimal in analysis.frontier

    def test_baselines_have_correct_escalation_rates(self):
        results = _make_results(20)
        with tempfile.TemporaryDirectory() as d:
            nli_path = Path(d) / "nli.json"
            judge_path = Path(d) / "judge.json"
            _write_json({"raw_results": results}, nli_path)
            _write_json({"raw_results": results}, judge_path)
            analysis = run_cascade_analysis(nli_path, judge_path, grid_steps=3)

        assert analysis.nli_only.escalation_rate == pytest.approx(0.0)   # never escalates
        assert analysis.both_methods.escalation_rate == pytest.approx(1.0)  # always escalates
