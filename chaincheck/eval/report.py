"""Evaluation report generator — saves JSON results and prints a Rich summary table."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from chaincheck.eval.halueval import EvalRun

console = Console()


def save_report(run: EvalRun, output_path: Path) -> None:
    """
    Serialise an EvalRun to JSON and write to output_path.

    Args:
        run: Completed evaluation run.
        output_path: Destination file path (created if it doesn't exist).
    """
    m = run.metrics
    data = {
        "method": run.method,
        "samples": run.samples,
        "metrics": {
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "accuracy": m.accuracy,
            "ece": m.ece,
            "avg_latency_ms": m.avg_latency_ms,
            "p50_latency_ms": m.p50_latency_ms,
            "p95_latency_ms": m.p95_latency_ms,
            "n_samples": m.n_samples,
        },
        "raw_results": run.raw_results,
    }
    Path(output_path).write_text(json.dumps(data, indent=2))


def print_report(run: EvalRun) -> None:
    """
    Print a formatted Rich table summarising the evaluation results.

    Columns: Metric / Value — covers F1, precision, recall, accuracy, latency.

    Args:
        run: Completed evaluation run.
    """
    m = run.metrics
    table = Table(title=f"ChainCheck Eval — {run.method.upper()} on {run.samples} samples")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    rows = [
        ("Precision", f"{m.precision:.4f}"),
        ("Recall", f"{m.recall:.4f}"),
        ("F1", f"{m.f1:.4f}"),
        ("Accuracy", f"{m.accuracy:.4f}"),
        ("ECE ↓", f"{m.ece:.4f}"),
        ("Avg Latency", f"{m.avg_latency_ms:.1f} ms"),
        ("P50 Latency", f"{m.p50_latency_ms:.1f} ms"),
        ("P95 Latency", f"{m.p95_latency_ms:.1f} ms"),
        ("Samples", str(m.n_samples)),
    ]
    for key, val in rows:
        table.add_row(key, val)

    console.print(table)
