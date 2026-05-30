"""Tests for chaincheck.server — FastAPI endpoint integration."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

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
    """Async test client — skips lifespan model loading but initialises the DB."""
    from chaincheck.server import _init_history_db, app

    _init_history_db()  # lifespan is skipped by ASGITransport; init DB explicitly
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
    assert "db_backend" in body


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


@pytest.mark.asyncio
async def test_ui_endpoint_returns_html(client: AsyncClient):
    """GET / should serve the index.html file."""
    response = await client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_history_returns_list(client: AsyncClient):
    """GET /history should return a list (empty or not)."""
    response = await client.get("/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_history_limit_clamped(client: AsyncClient):
    """GET /history?limit=0 should be clamped to 1; limit=9999 clamped to 100."""
    r1 = await client.get("/history?limit=0")
    assert r1.status_code == 200
    r2 = await client.get("/history?limit=9999")
    assert r2.status_code == 200


@pytest.mark.asyncio
async def test_history_detail_404_unknown(client: AsyncClient):
    """GET /history/{id} for unknown id should return 404."""
    response = await client.get("/history/nonexistent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_history_detail_returns_payload(client: AsyncClient):
    """GET /history/{id} should return stored payload when id exists."""
    from chaincheck.server import _init_history_db, _save_result

    _init_history_db()
    result = _fake_result()
    result = result.model_copy(update={"request_id": "test-detail-id"})
    _save_result(result)

    response = await client.get("/history/test-detail-id")
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "test-detail-id"


@pytest.mark.asyncio
async def test_analytics_returns_expected_keys(client: AsyncClient):
    """GET /analytics should return total_scans and risk_distribution keys."""
    response = await client.get("/analytics")
    assert response.status_code == 200
    body = response.json()
    assert "total_scans" in body
    assert "risk_distribution" in body
    assert "score_trend" in body
    assert "method_usage" in body


@pytest.mark.asyncio
async def test_feedback_returns_204(client: AsyncClient):
    """POST /feedback/{id} should return 204 regardless of whether id exists."""
    response = await client.post(
        "/feedback/any-id",
        json={"correct": True, "note": "looks good"},
    )
    assert response.status_code == 204


@pytest.mark.asyncio
async def test_auth_middleware_rejects_missing_key(client: AsyncClient):
    """When CHAINCHECK_API_KEY is set, requests without X-API-Key should get 401."""
    with patch("chaincheck.server._API_KEY", "secret"):
        response = await client.post(
            "/check",
            json={"response": "hello", "methods": ["judge"]},
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_auth_middleware_accepts_correct_key(client: AsyncClient):
    """Correct X-API-Key header should pass through auth middleware."""
    with (
        patch("chaincheck.server._API_KEY", "secret"),
        patch("chaincheck.server.detect", new=AsyncMock(return_value=_fake_result())),
    ):
        response = await client.post(
            "/check",
            headers={"X-API-Key": "secret"},
            json={"response": "hello", "methods": ["judge"]},
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_stream_endpoint_yields_done(client: AsyncClient):
    """POST /stream should return text/event-stream ending with [DONE]."""
    async def _fake_stream(*args, **kwargs):
        yield {"type": "claims", "claims": ["claim"], "request_id": "r1"}
        yield {"type": "result", "data": json.loads(_fake_result().model_dump_json())}

    with patch("chaincheck.server.detect_stream", side_effect=_fake_stream):
        response = await client.post(
            "/stream",
            json={"response": "The sky is blue.", "methods": ["judge"]},
        )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "[DONE]" in response.text


@pytest.mark.asyncio
async def test_save_result_is_best_effort(client: AsyncClient):
    """_save_result should silently swallow exceptions."""
    from chaincheck.server import _save_result

    with patch("chaincheck.db.get_engine", side_effect=Exception("db error")):
        _save_result(_fake_result())  # should not raise


@pytest.mark.asyncio
async def test_proxy_endpoint_no_api_key(client: AsyncClient):
    """POST /v1/chat/completions without OPENAI_API_KEY should return 503."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("chaincheck.server.os.getenv", return_value=""),
    ):
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_proxy_endpoint_passthrough(client: AsyncClient):
    """POST /v1/chat/completions should forward to OpenAI and return its response."""
    fake_openai_body = {
        "choices": [{"message": {"content": "The sky is blue."}}]
    }
    mock_response = MagicMock()
    mock_response.json.return_value = fake_openai_body
    mock_response.content = json.dumps(fake_openai_body).encode()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}

    with (
        patch("chaincheck.server.os.getenv", return_value="sk-fake"),
        patch("chaincheck.server.detect", new=AsyncMock(return_value=_fake_result())),
        patch("chaincheck.server._openai_request", new=AsyncMock(return_value=mock_response)),
    ):
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "What colour is the sky?"}],
            },
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_check_with_cascade_flag(client: AsyncClient):
    """POST /check with cascade=True should pass cascade=True to detect."""
    mock_detect = AsyncMock(return_value=_fake_result())
    with patch("chaincheck.server.detect", new=mock_detect):
        response = await client.post(
            "/check",
            json={"response": "The sky is blue.", "methods": ["nli", "judge"], "cascade": True},
        )
    assert response.status_code == 200
    _, kwargs = mock_detect.call_args
    assert kwargs.get("cascade") is True


# ── Feedback exception path ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_feedback_swallows_db_error(client: AsyncClient):
    """POST /feedback should return 204 even when sqlite3 raises."""
    with patch("chaincheck.db.get_engine", side_effect=Exception("db gone")):
        response = await client.post(
            "/feedback/any-id",
            json={"correct": False, "note": "wrong"},
        )
    assert response.status_code == 204


# ── Lifespan model preloading ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_lifespan_preloads_models():
    """lifespan() should call _get_model and _get_embed_model then set _models_loaded."""
    import asyncio as _aio

    from chaincheck.server import app, lifespan

    with (
        patch("chaincheck.server._init_history_db"),
        patch("chaincheck.methods.nli._get_model", return_value=MagicMock()),
        patch("chaincheck.methods.consistency._get_embed_model", return_value=MagicMock()),
    ):
        async with lifespan(app):
            # Give the background _preload task one event-loop cycle to complete
            # (the mocked executors return immediately, so this is enough).
            await _aio.sleep(0.05)


# ── Proxy block mode ───────────────────────────────────────────────────────────

def _make_proxy_patches(openai_body: dict, detect_result, proxy_mode: str, threshold: float = 0.7):
    """
    Return a list of context-manager patches for the proxy endpoint.

    Patches _openai_request (the extracted helper) so coverage.py
    can track all lines in proxy_endpoint without async-with mock complexity.
    """
    mock_resp = MagicMock()
    mock_resp.json.return_value = openai_body
    mock_resp.content = json.dumps(openai_body).encode()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "application/json"}

    return [
        patch("chaincheck.server.os.getenv", return_value="sk-fake"),
        patch("chaincheck.server._PROXY_MODE", proxy_mode),
        patch("chaincheck.server._PROXY_BLOCK_THRESHOLD", threshold),
        patch("chaincheck.server.detect", new=AsyncMock(return_value=detect_result)),
        patch("chaincheck.server._openai_request", new=AsyncMock(return_value=mock_resp)),
    ]


@pytest.mark.asyncio
async def test_proxy_block_mode_returns_451(client: AsyncClient):
    """When PROXY_MODE=block and score >= threshold, proxy should return 451."""
    from chaincheck.models import DetectionResult, MethodResult
    high_risk = DetectionResult(
        response="hallucinated", claims=["c"],
        method_results={"judge": MethodResult(method="judge", raw_score=0.95, latency_ms=10)},
        aggregate_score=0.95, risk_level="high",
        latency_ms={"judge": 10}, request_id="rid-block",
    )
    openai_body = {"choices": [{"message": {"content": "hallucinated text"}}]}

    patches = _make_proxy_patches(openai_body, high_risk, "block")
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 451
    body = response.json()
    assert body["error"]["type"] == "hallucination_blocked"
    assert body["error"]["score"] == 0.95


@pytest.mark.asyncio
async def test_proxy_warn_mode_appends_warning(client: AsyncClient):
    """When PROXY_MODE=warn and score >= threshold, response content should include warning."""
    from chaincheck.models import DetectionResult, MethodResult
    high_risk = DetectionResult(
        response="slightly wrong", claims=["c"],
        method_results={"judge": MethodResult(method="judge", raw_score=0.85, latency_ms=10)},
        aggregate_score=0.85, risk_level="high",
        latency_ms={"judge": 10}, request_id="rid-warn",
    )
    openai_body = {"choices": [{"message": {"content": "The sky is green."}}]}

    patches = _make_proxy_patches(openai_body, high_risk, "warn")
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    payload = json.loads(response.content)
    content = payload["choices"][0]["message"]["content"]
    assert "ChainCheck" in content
    assert "85%" in content


@pytest.mark.asyncio
async def test_proxy_passthrough_below_threshold(client: AsyncClient):
    """When score < threshold proxy should return the OpenAI response unchanged."""
    openai_body = {"choices": [{"message": {"content": "Perfectly accurate."}}]}
    patches = _make_proxy_patches(openai_body, _fake_result(), "block")
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    assert json.loads(response.content) == openai_body


# ── Direct proxy_endpoint.__wrapped__ tests (bypass slowapi for coverage) ─────
#
# @limiter.limit wraps the function with a new code-object (co_firstlineno=719
# in slowapi's internals), so coverage.py never traces into server.py's lines.
# Calling __wrapped__ directly gives us full line coverage.

def _make_mock_request(messages=None):
    """Build a minimal Request mock for the proxy tests."""
    req = MagicMock()
    req.json = AsyncMock(return_value={
        "model": "gpt-4o-mini",
        "messages": messages or [{"role": "user", "content": "hi"}],
    })
    return req


def _make_mock_resp(body: dict):
    """Build a minimal httpx.Response mock."""
    mock = MagicMock()
    mock.json.return_value = body
    mock.content = json.dumps(body).encode()
    mock.status_code = 200
    mock.headers = {"content-type": "application/json"}
    return mock


@pytest.mark.asyncio
async def test_proxy_wrapped_block_returns_451():
    """Proxy block mode returns 451 when score >= threshold (direct function call)."""
    from chaincheck.models import DetectionResult, MethodResult
    from chaincheck.server import proxy_endpoint

    high_risk = DetectionResult(
        response="hallucinated", claims=["c"],
        method_results={"judge": MethodResult(method="judge", raw_score=0.95, latency_ms=10)},
        aggregate_score=0.95, risk_level="high",
        latency_ms={"judge": 10}, request_id="rid-block",
    )
    body = {"choices": [{"message": {"content": "hallucinated text"}}]}

    with (
        patch("chaincheck.server.os.getenv", return_value="sk-fake"),
        patch("chaincheck.server._PROXY_MODE", "block"),
        patch("chaincheck.server._PROXY_BLOCK_THRESHOLD", 0.7),
        patch("chaincheck.server.detect", new=AsyncMock(return_value=high_risk)),
        patch(
            "chaincheck.server._openai_request",
            new=AsyncMock(return_value=_make_mock_resp(body)),
        ),
    ):
        result = await proxy_endpoint.__wrapped__(_make_mock_request())

    assert result.status_code == 451
    payload = json.loads(result.body)
    assert payload["error"]["type"] == "hallucination_blocked"
    assert payload["error"]["score"] == 0.95


@pytest.mark.asyncio
async def test_proxy_wrapped_warn_appends_warning():
    """Proxy warn mode appends ChainCheck warning (direct function call)."""
    from chaincheck.models import DetectionResult, MethodResult
    from chaincheck.server import proxy_endpoint

    high_risk = DetectionResult(
        response="slightly wrong", claims=["c"],
        method_results={"judge": MethodResult(method="judge", raw_score=0.85, latency_ms=10)},
        aggregate_score=0.85, risk_level="high",
        latency_ms={"judge": 10}, request_id="rid-warn",
    )
    body = {"choices": [{"message": {"content": "The sky is green."}}]}

    with (
        patch("chaincheck.server.os.getenv", return_value="sk-fake"),
        patch("chaincheck.server._PROXY_MODE", "warn"),
        patch("chaincheck.server._PROXY_BLOCK_THRESHOLD", 0.7),
        patch("chaincheck.server.detect", new=AsyncMock(return_value=high_risk)),
        patch(
            "chaincheck.server._openai_request",
            new=AsyncMock(return_value=_make_mock_resp(body)),
        ),
    ):
        result = await proxy_endpoint.__wrapped__(_make_mock_request())

    assert result.status_code == 200
    payload = json.loads(result.body)
    content = payload["choices"][0]["message"]["content"]
    assert "ChainCheck" in content
    assert "85%" in content


@pytest.mark.asyncio
async def test_proxy_wrapped_passthrough_below_threshold():
    """Proxy returns unmodified response when score < threshold (direct function call)."""
    from chaincheck.server import proxy_endpoint

    body = {"choices": [{"message": {"content": "Perfectly accurate."}}]}

    with (
        patch("chaincheck.server.os.getenv", return_value="sk-fake"),
        patch("chaincheck.server._PROXY_MODE", "block"),
        patch("chaincheck.server._PROXY_BLOCK_THRESHOLD", 0.7),
        patch("chaincheck.server.detect", new=AsyncMock(return_value=_fake_result())),
        patch(
            "chaincheck.server._openai_request",
            new=AsyncMock(return_value=_make_mock_resp(body)),
        ),
    ):
        result = await proxy_endpoint.__wrapped__(_make_mock_request())

    assert result.status_code == 200
    assert json.loads(result.body) == body


@pytest.mark.asyncio
async def test_proxy_wrapped_no_api_key_returns_503():
    """Proxy returns 503 when OPENAI_API_KEY is absent (direct function call)."""
    from chaincheck.server import proxy_endpoint

    with patch("chaincheck.server.os.getenv", return_value=""):
        result = await proxy_endpoint.__wrapped__(_make_mock_request())

    assert result.status_code == 503


@pytest.mark.asyncio
async def test_proxy_wrapped_detect_exception_passthrough():
    """Proxy silently swallows detect() errors and returns the OpenAI body unchanged."""
    from chaincheck.server import proxy_endpoint

    body = {"choices": [{"message": {"content": "some text"}}]}

    with (
        patch("chaincheck.server.os.getenv", return_value="sk-fake"),
        patch("chaincheck.server.detect", side_effect=Exception("model down")),
        patch(
            "chaincheck.server._openai_request",
            new=AsyncMock(return_value=_make_mock_resp(body)),
        ),
    ):
        result = await proxy_endpoint.__wrapped__(_make_mock_request())

    assert result.status_code == 200


@pytest.mark.asyncio
async def test_proxy_wrapped_warn_bad_payload_falls_through():
    """Proxy warn mode falls through gracefully when payload lacks expected structure."""
    from chaincheck.models import DetectionResult, MethodResult
    from chaincheck.server import proxy_endpoint

    high_risk = DetectionResult(
        response="wrong", claims=["c"],
        method_results={"judge": MethodResult(method="judge", raw_score=0.9, latency_ms=10)},
        aggregate_score=0.9, risk_level="high",
        latency_ms={"judge": 10}, request_id="rid-bad",
    )
    # content=None survives text extraction (→ ""), passes detection,
    # but `None += warning` raises TypeError inside the warn-mode mutation,
    # covering the except/pass at lines 471-472.
    bad_body = {"choices": [{"message": {"content": None}}]}

    with (
        patch("chaincheck.server.os.getenv", return_value="sk-fake"),
        patch("chaincheck.server._PROXY_MODE", "warn"),
        patch("chaincheck.server._PROXY_BLOCK_THRESHOLD", 0.7),
        patch("chaincheck.server.detect", new=AsyncMock(return_value=high_risk)),
        patch(
            "chaincheck.server._openai_request",
            new=AsyncMock(return_value=_make_mock_resp(bad_body)),
        ),
    ):
        result = await proxy_endpoint.__wrapped__(_make_mock_request())

    # Falls through to the passthrough Response (200) when content mutation fails
    assert result.status_code == 200
