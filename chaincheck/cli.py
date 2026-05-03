"""
Typer-based CLI for ChainCheck.

Commands:
  check    — detect hallucinations in a single response
  batch    — process multiple inputs from a JSONL file
  eval     — run HaluEval benchmark
  serve    — start FastAPI server
  compare  — run all methods and print a side-by-side comparison table
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

app = typer.Typer(
    name="chaincheck",
    help="LLM hallucination detection toolkit.",
    add_completion=False,
)

console = Console()

_RISK_COLOUR: dict[str, str] = {"low": "green", "medium": "yellow", "high": "red"}
_LABEL_COLOUR: dict[str, str] = {
    "supported": "green",
    "unsupported": "yellow",
    "contradicted": "red",
    "unknown": "white",
}


@app.command()
def check(
    response: str = typer.Option(..., "--response", "-r", help="LLM response to analyse"),
    context: str = typer.Option("", "--context", "-c", help="Retrieved context or reference"),
    methods: str = typer.Option(
        "nli,judge", "--methods", "-m", help="Comma-separated method list"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON instead of table"),
) -> None:
    """Detect hallucinations in a single response and print a colour-coded table."""
    from chaincheck.detect import detect as _detect

    method_list = [m.strip() for m in methods.split(",")]
    result = asyncio.run(_detect(response, context=context, methods=method_list))  # type: ignore[arg-type]

    if json_output:
        console.print_json(result.model_dump_json())
        return

    risk_col = _RISK_COLOUR.get(result.risk_level, "white")
    table = Table(
        title=(
            f"ChainCheck  |  Score: {result.aggregate_score:.2f}  "
            f"|  Risk: [{risk_col}]{result.risk_level.upper()}[/{risk_col}]"
        )
    )
    table.add_column("Claim", max_width=55)
    table.add_column("Label", justify="center")
    table.add_column("Conf", justify="right")
    table.add_column("Evidence", max_width=40)
    table.add_column("Method")

    for mr in result.method_results.values():
        for cr in mr.claims:
            style = _LABEL_COLOUR.get(cr.label, "white")
            table.add_row(
                cr.claim,
                f"[{style}]{cr.label}[/{style}]",
                f"{cr.confidence:.2f}",
                (cr.evidence or "—")[:60],
                cr.method,
            )

    console.print(table)


@app.command()
def batch(
    input_file: Path = typer.Option(..., "--input", "-i", help="JSONL file of inputs"),
    output_file: Path = typer.Option(..., "--output", "-o", help="JSONL file for results"),
    methods: str = typer.Option("all", "--methods", "-m", help="Methods to run"),
) -> None:
    """Process a JSONL file of responses in async batch mode with a progress bar."""
    from chaincheck.detect import detect as _detect

    method_list = None if methods == "all" else [m.strip() for m in methods.split(",")]

    with input_file.open() as f:
        inputs = [json.loads(line) for line in f if line.strip()]

    async def _run() -> list[dict]:
        results: list[dict] = []
        with Progress() as progress:
            task = progress.add_task("Processing...", total=len(inputs))
            for inp in inputs:
                r = await _detect(
                    inp.get("response", ""),
                    context=inp.get("context", ""),
                    methods=method_list,  # type: ignore[arg-type]
                )
                results.append(r.model_dump())
                progress.advance(task)
        return results

    results = asyncio.run(_run())
    with output_file.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    console.print(f"[green]Processed {len(results)} inputs → {output_file}[/green]")


@app.command()
def eval(
    method: str = typer.Option("nli", "--method", "-m", help="Detection method to benchmark"),
    samples: int = typer.Option(500, "--samples", "-n", help="Number of HaluEval samples"),
    output: Path = typer.Option(Path("eval_results.json"), "--output", "-o"),
) -> None:
    """Run a HaluEval benchmark for the specified detection method."""
    from chaincheck.eval.halueval import run_halueval
    from chaincheck.eval.report import print_report, save_report

    console.print(f"[bold]Running HaluEval — method: {method}, samples: {samples}[/bold]")
    run = asyncio.run(run_halueval(method=method, n_samples=samples))  # type: ignore[arg-type]
    save_report(run, output)
    print_report(run)
    console.print(f"\n[green]Results saved to {output}[/green]")


@app.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the FastAPI server."""
    import uvicorn

    uvicorn.run("chaincheck.server:app", host="0.0.0.0", port=port, reload=reload)


@app.command()
def compare(
    response: str = typer.Option(..., "--response", "-r"),
    context: str = typer.Option("", "--context", "-c"),
) -> None:
    """Run all three methods and print a side-by-side comparison table."""
    from chaincheck.detect import detect as _detect

    result = asyncio.run(
        _detect(response, context=context, methods=["nli", "consistency", "judge"])
    )

    # Build claim → {method: label} index
    claim_index: dict[str, dict[str, str]] = {}
    for mr in result.method_results.values():
        for cr in mr.claims:
            claim_index.setdefault(cr.claim, {})[cr.method] = cr.label

    table = Table(title="Method Comparison — All Three Methods")
    table.add_column("Claim", max_width=50)
    table.add_column("NLI")
    table.add_column("Judge")
    table.add_column("Agree?", justify="center")

    for claim, labels in claim_index.items():
        nli = labels.get("nli", "—")
        judge = labels.get("judge", "—")
        agree = "[green]✓[/green]" if nli == judge else "[red]✗[/red]"
        table.add_row(claim[:50], nli, judge, agree)

    console.print(table)

    risk_col = _RISK_COLOUR.get(result.risk_level, "white")
    console.print(
        f"\nAggregate score: {result.aggregate_score:.2f}  "
        f"Risk: [{risk_col}]{result.risk_level.upper()}[/{risk_col}]"
    )
