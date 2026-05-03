"""Tests for chaincheck.server — FastAPI endpoint integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from chaincheck.models import DetectionResult, MethodResult


def _fake_result() -> DetectionResult:
    return DetectionResult(
        response="test response",
        claims=["test claim"],
        method_results={
            "judge": MethodResult(method="judge", raw_score=0.2, latency_ms=50.0)
        },
        aggregate_score=0.2,
        risk_level="low",
        latency_ms={"judge": 50.0},
        request_id="test-id",
    )


@pytest.fixture
async def client():
    """Async test client — skips lifespan model loading."""
    from chaincheck.server import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient):
    """GET /health should return 200 with status 'ok'."""
    with patch("chaincheck.server._models_loaded", True):
        response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


@pytest.mark.asyncio
async def test_check_validation_empty_response(client: AsyncClient):
    """POST /check with empty response string should return 422."""
    payload = {"response": "", "context": "", "methods": ["nli"]}
    response = await client.post("/check", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_check_returns_request_id_header(client: AsyncClient):
    """POST /check response headers should include X-Request-ID."""
    with patch("chaincheck.server.detect", new=AsyncMock(return_value=_fake_result())):
        response = await client.post(
            "/check",
            json={"response": "Hello world.", "context": "ctx", "methods": ["judge"]},
        )
    assert "x-request-id" in response.headers


@pytest.mark.asyncio
async def test_check_returns_latency_header(client: AsyncClient):
    """POST /check should include X-Latency-Ms header."""
    with patch("chaincheck.server.detect", new=AsyncMock(return_value=_fake_result())):
        response = await client.post(
            "/check",
            json={"response": "Hello world.", "methods": ["judge"]},
        )
    assert "x-latency-ms" in response.headers


@pytest.mark.asyncio
async def test_batch_returns_list(client: AsyncClient):
    """POST /batch should return a list with one result per input."""
    with patch("chaincheck.server.detect", new=AsyncMock(return_value=_fake_result())):
        response = await client.post(
            "/batch",
            json={
                "inputs": [
                    {"response": "Response one.", "methods": ["judge"]},
                    {"response": "Response two.", "methods": ["judge"]},
                ]
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 2


@pytest.mark.asyncio
async def test_check_returns_200_with_valid_payload(client: AsyncClient):
    """POST /check with valid payload should return 200."""
    with patch("chaincheck.server.detect", new=AsyncMock(return_value=_fake_result())):
        response = await client.post(
            "/check",
            json={
                "response": "The sky is blue.",
                "context": "The sky appears blue due to Rayleigh scattering.",
                "methods": ["judge"],
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert "aggregate_score" in body
    assert "risk_level" in body
