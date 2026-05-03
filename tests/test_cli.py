"""Tests for the Typer CLI helpers."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from chaincheck import cli
from chaincheck.detect import DetectionResult


def test_parse_methods_all() -> None:
    """The all alias expands to every method."""
    assert cli._parse_methods("all") == ["nli", "consistency", "judge"]


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    """CLI JSONL helpers read and write batch files."""
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text('{"response":"R"}\n', encoding="utf-8")
    rows = cli._read_jsonl(input_path)
    cli._write_jsonl(output_path, [DetectionResult(response="R")])
    assert rows == [{"response": "R"}]
    assert '"response":"R"' in output_path.read_text(encoding="utf-8")


def test_check_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """The check command renders a successful detection."""
    monkeypatch.setattr(cli, "detect", _fake_detect)
    result = CliRunner().invoke(cli.app, ["check", "--response", "R", "--methods", "nli"])
    assert result.exit_code == 0
    assert "ChainCheck" in result.output


def test_eval_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The eval command saves a report."""
    output = tmp_path / "eval.json"
    monkeypatch.setattr(cli, "run_halueval", _fake_eval)
    result = CliRunner().invoke(cli.app, ["eval", "--output", str(output)])
    assert result.exit_code == 0
    assert output.exists()


def test_compare_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """The compare command renders method summaries."""
    monkeypatch.setattr(cli, "detect", _fake_detect)
    result = CliRunner().invoke(cli.app, ["compare", "--response", "R"])
    assert result.exit_code == 0
    assert "Method comparison" in result.output


async def _fake_detect(*args, **kwargs) -> DetectionResult:
    del args, kwargs
    return DetectionResult(response="R", claims=["R"], aggregate_score=0.1, risk_level="low")


async def _fake_eval(method: str = "nli", samples: int = 500) -> dict[str, object]:
    return {
        "method": method,
        "samples": samples,
        "metrics": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0},
        "results": [],
    }
