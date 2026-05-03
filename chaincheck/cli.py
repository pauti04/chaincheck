"""Typer command-line interface for ChainCheck."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

from chaincheck.detect import DetectionResult, detect
from chaincheck.eval.halueval import run_halueval
from chaincheck.eval.report import print_report, save_report

app = typer.Typer(help="Claim-level LLM hallucination detection.")
console = Console()
DEFAULT_EVAL_OUTPUT = Path("eval_results.json")


@app.command()
def check(
    response: Annotated[str, typer.Option(help="LLM response to audit.")],
    context: Annotated[str, typer.Option(help="Source context.")] = "",
    methods: Annotated[
        str, typer.Option(help="Comma-separated methods.")
    ] = "nli,consistency,judge",
) -> None:
    """Run hallucination detection for one response."""
    parsed_methods = _parse_methods(methods)
    result = asyncio.run(detect(response=response, context=context, methods=parsed_methods))
    _print_detection(result)


@app.command()
def batch(
    input: Annotated[Path, typer.Option(help="Input JSONL path.")],
    output: Annotated[Path, typer.Option(help="Output JSONL path.")],
    methods: Annotated[str, typer.Option(help="Comma-separated methods or all.")] = "all",
) -> None:
    """Run detection over a JSONL batch."""
    rows = _read_jsonl(input)
    results = asyncio.run(_run_batch(rows, _parse_methods(methods)))
    _write_jsonl(output, results)


@app.command()
def eval(
    method: Annotated[str, typer.Option(help="Method to evaluate.")] = "nli",
    samples: Annotated[int, typer.Option(help="Sample count.")] = 500,
    output: Annotated[Path, typer.Option(help="Output JSON path.")] = DEFAULT_EVAL_OUTPUT,
) -> None:
    """Run the HaluEval benchmark."""
    report = asyncio.run(run_halueval(method=method, samples=samples))
    save_report(report, output)
    print_report(report)


@app.command()
def serve(port: int = 8000, reload: bool = False) -> None:
    """Start the ChainCheck FastAPI server."""
    import uvicorn

    uvicorn.run("chaincheck.server:app", host="0.0.0.0", port=port, reload=reload)


@app.command()
def compare(
    response: Annotated[str, typer.Option(help="LLM response to audit.")],
    context: Annotated[str, typer.Option(help="Source context.")] = "",
    prompt: Annotated[str, typer.Option(help="Original prompt.")] = "",
) -> None:
    """Compare all three detection methods side by side."""
    result = asyncio.run(detect(response=response, context=context, prompt=prompt))
    _print_comparison(result)


def _print_detection(result: DetectionResult) -> None:
    title = f"ChainCheck: {result.risk_level.upper()} risk ({result.aggregate_score:.2f})"
    table = Table(title=title)
    table.add_column("Claim")
    table.add_column("Status")
    for row in result.claim_results:
        table.add_row(row.claim, row.label, style=_style(row.label))
    console.print(table)


def _print_comparison(result: DetectionResult) -> None:
    table = Table(title=f"Method comparison: {result.aggregate_score:.2f} {result.risk_level}")
    table.add_column("Method")
    table.add_column("Summary")
    for name, payload in result.methods.items():
        table.add_row(name, _summary(payload))
    console.print(table)


async def _run_batch(rows: list[dict[str, object]], methods: list[str]) -> list[DetectionResult]:
    tasks = [
        detect(
            response=str(row.get("response", "")),
            context=str(row.get("context", "")),
            prompt=str(row.get("prompt", "")),
            methods=methods,
        )
        for row in track(rows, description="Checking")
    ]
    return list(await asyncio.gather(*tasks))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _write_jsonl(path: Path, results: list[DetectionResult]) -> None:
    lines = [result.model_dump_json() for result in results]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_methods(methods: str) -> list[str]:
    parsed = [part.strip() for part in methods.split(",") if part.strip()]
    return ["nli", "consistency", "judge"] if parsed == ["all"] else parsed


def _style(level: str) -> str:
    return {"low": "green", "medium": "yellow", "high": "red"}.get(level, "white")


def _summary(payload: object) -> str:
    if hasattr(payload, "model_dump_json"):
        return payload.model_dump_json()
    return json.dumps(payload, default=str)[:400]
