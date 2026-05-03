"""Tests for chaincheck.cli — Typer CLI commands."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from chaincheck.cli import app
from chaincheck.models import DetectionResult, MethodResult

runner = CliRunner()


def _fake_result() -> DetectionResult:
    return DetectionResult(
        response="test",
        claims=["test claim"],
        method_results={
            "judge": MethodResult(method="judge", raw_score=0.1, latency_ms=50.0)
        },
        aggregate_score=0.1,
        risk_level="low",
        latency_ms={"judge": 50.0},
        request_id="test-id",
    )


class TestCheckCommand:
    def test_check_runs(self):
        with patch("chaincheck.cli.asyncio") as mock_asyncio:
            mock_asyncio.run.return_value = _fake_result()
            result = runner.invoke(
                app,
                ["check", "--response", "The sky is blue.", "--methods", "judge"],
            )
        assert result.exit_code == 0

    def test_check_json_output(self):
        with patch("chaincheck.cli.asyncio") as mock_asyncio:
            mock_asyncio.run.return_value = _fake_result()
            result = runner.invoke(
                app,
                ["check", "--response", "Test.", "--methods", "judge", "--json"],
            )
        assert result.exit_code == 0

    def test_check_missing_response_fails(self):
        result = runner.invoke(app, ["check"])
        assert result.exit_code != 0


class TestBatchCommand:
    def test_batch_processes_jsonl(self):
        inputs = [
            {"response": "The sky is blue.", "context": "The sky is blue."},
            {"response": "Water is wet.", "context": "Water is wet."},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as inf:
            for item in inputs:
                inf.write(json.dumps(item) + "\n")
            in_path = Path(inf.name)

        out_path = in_path.with_suffix(".out.jsonl")

        with patch("chaincheck.cli.asyncio") as mock_asyncio:
            fake = _fake_result()
            mock_asyncio.run.return_value = [fake.model_dump(), fake.model_dump()]

            # Patch the inner async logic by making asyncio.run return list of dicts
            async def _fake_run_fn(coro):
                return [fake.model_dump(), fake.model_dump()]

            mock_asyncio.run.side_effect = lambda coro: [fake.model_dump(), fake.model_dump()]

            result = runner.invoke(
                app,
                ["batch", "--input", str(in_path), "--output", str(out_path), "--methods", "judge"],
            )

        in_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
        assert result.exit_code == 0


class TestServeCommand:
    def test_serve_calls_uvicorn(self):
        with patch("uvicorn.run") as mock_uv:
            runner.invoke(app, ["serve", "--port", "9999"])
        mock_uv.assert_called_once()
        assert 9999 in mock_uv.call_args[1].values() or 9999 in mock_uv.call_args[0]


class TestCompareCommand:
    def test_compare_runs(self):
        with patch("chaincheck.cli.asyncio") as mock_asyncio:
            mock_asyncio.run.return_value = _fake_result()
            result = runner.invoke(
                app,
                ["compare", "--response", "Test response.", "--context", "context"],
            )
        assert result.exit_code == 0
