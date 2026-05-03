"""Benchmark report persistence and terminal rendering."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table


def save_report(report: dict[str, object], output: Path) -> None:
    """Save a benchmark report to disk."""
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def print_report(report: dict[str, object]) -> None:
    """Print a benchmark report as a Rich summary table."""
    metrics = report.get("metrics", {})
    columns = ["precision", "recall", "f1", "accuracy", "avg_latency_ms", "p95_latency_ms"]
    table = Table(title=f"HaluEval: {report.get('method', 'unknown')}")
    for column in columns:
        table.add_column(column)
    table.add_row(*[_fmt(metrics.get(column, 0.0)) for column in columns])
    Console().print(table)


def _fmt(value: object) -> str:
    return f"{float(value):.3f}" if isinstance(value, int | float) else str(value)
