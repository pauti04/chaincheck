"""Evaluation report generator — saves JSON results and prints a Rich summary table."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from chaincheck.eval.halueval import EvalRun

console = Console()


def save_report(run: EvalRun, output_path: Path) -> None:
    """Serialise an EvalRun to JSON and write to output_path."""
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


def save_claimlevel_report(run, output_path: Path) -> None:
    """Serialise a ClaimLevelRun to JSON."""
    from chaincheck.eval.claimlevel import ClaimLevelRun

    assert isinstance(run, ClaimLevelRun)
    m = run.metrics
    data = {
        "method": run.method,
        "pairs": run.pairs,
        "metrics": {
            "clean_flagging_rate": m.clean_flagging_rate,
            "halluc_flagging_rate": m.halluc_flagging_rate,
            "discrimination_ratio": m.discrimination_ratio,
            "claim_auc": m.claim_auc,
            "avg_claims_per_response": m.avg_claims_per_response,
            "n_clean_claims": m.n_clean_claims,
            "n_halluc_claims": m.n_halluc_claims,
            "latency_ms": m.latency_ms,
        },
        "raw_results": [
            {k: v for k, v in r.items() if k not in ("correct_claim_scores", "halluc_claim_scores")}
            for r in run.raw_results
        ],
    }
    Path(output_path).write_text(json.dumps(data, indent=2))


def print_report(run: EvalRun) -> None:
    """Print a Rich table for a standard (response-level) eval run."""
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


def print_claimlevel_report(run) -> None:
    """Print a Rich table for a claim-level discrimination eval run."""
    m = run.metrics
    table = Table(title=f"ChainCheck Claim-Level Eval — {run.method.upper()} on {run.pairs} pairs")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Note", style="dim")

    rows = [
        ("Clean flagging rate ↓", f"{m.clean_flagging_rate:.4f}",
         "fraction of correct-response claims flagged (FPR)"),
        ("Halluc flagging rate ↑", f"{m.halluc_flagging_rate:.4f}",
         "fraction of hallucinated-response claims flagged"),
        ("Discrimination ratio ↑", f"{m.discrimination_ratio:.2f}x",
         "halluc / clean rate — higher = better separation"),
        ("Claim AUC ↑", f"{m.claim_auc:.4f}",
         "AUC of per-claim scores vs response labels"),
        ("Avg claims / response", f"{m.avg_claims_per_response:.1f}",
         "decomposition quality proxy"),
        ("Clean claims evaluated", str(m.n_clean_claims), ""),
        ("Halluc claims evaluated", str(m.n_halluc_claims), ""),
        ("Latency total", f"{m.latency_ms / 1000:.1f} s", ""),
    ]
    for key, val, note in rows:
        table.add_row(key, val, note)
    console.print(table)
