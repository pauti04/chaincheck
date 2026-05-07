"""
FastAPI server exposing ChainCheck detection via REST API.

Endpoints:
  POST /check    — detect hallucinations in a single response
  POST /stream   — same as /check but streams events via SSE as each method completes
  POST /batch    — batch detection for multiple inputs
  GET  /history  — recent detection results (persisted in SQLite or PostgreSQL)
  GET  /health   — liveness check with version and model status
  GET  /metrics  — Prometheus metrics (text/plain)
  GET  /docs     — auto-generated OpenAPI documentation
"""

from __future__ import annotations

import asyncio
import json as _json
import os
import time
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import httpx

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from chaincheck import __version__
from chaincheck import db as _db
from chaincheck.detect import detect, detect_stream
from chaincheck.metrics_prom import (
    active_requests_gauge,
    models_loaded_gauge,
    record_detection,
    request_latency_ms,
    requests_total,
)
from chaincheck.models import DetectionResult, Document

limiter = Limiter(key_func=get_remote_address)

_models_loaded = False
_API_KEY = os.getenv("CHAINCHECK_API_KEY", "")
_PROXY_BLOCK_THRESHOLD = float(os.getenv("PROXY_BLOCK_THRESHOLD", "0.8"))
_PROXY_MODE = os.getenv("PROXY_MODE", "passthrough")  # passthrough | warn | block


# ── Persistence helpers (thin wrappers over chaincheck.db) ────────────────────

def _init_history_db() -> None:
    _db.init_db()


def _save_result(result: DetectionResult) -> None:
    """Persist a DetectionResult. Best-effort — never raises."""
    _db.save_result(
        request_id=result.request_id,
        created_at=time.time(),
        response_preview=result.response[:120],
        aggregate_score=result.aggregate_score,
        risk_level=result.risk_level,
        total_latency_ms=result.total_latency_ms,
        methods=list(result.method_results.keys()),
        payload_json=result.model_dump_json(),
    )
    # Fire-and-forget Prometheus metrics (no I/O, always succeeds)
    with suppress(Exception):
        record_detection(result)


# ── Pydantic models ────────────────────────────────────────────────────────────

class CheckRequest(BaseModel):
    """Request body for POST /check and POST /stream."""

    response: str = Field(min_length=1)
    context: str = ""
    prompt: str = ""
    methods: list[str] = Field(default_factory=lambda: ["nli", "judge"])
    cascade: bool = False
    documents: list[Document] = Field(default_factory=list)


class BatchRequest(BaseModel):
    """Request body for POST /batch."""

    inputs: list[CheckRequest]


class FeedbackRequest(BaseModel):
    """Request body for POST /feedback/{request_id}."""

    correct: bool
    note: str = ""


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str
    version: str
    models_loaded: bool
    db_backend: str  # "sqlite" or "postgresql"


# ── App setup ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bind port immediately; warm NLI and embedding models in the background."""
    global _models_loaded
    _init_history_db()

    loop = asyncio.get_event_loop()

    async def _preload():
        global _models_loaded
        from chaincheck.methods.consistency import _get_embed_model
        from chaincheck.methods.nli import _get_model
        await loop.run_in_executor(None, _get_model)
        await loop.run_in_executor(None, _get_embed_model)
        _models_loaded = True
        models_loaded_gauge.set(1)

    task = asyncio.create_task(_preload())
    yield
    task.cancel()


app = FastAPI(
    title="ChainCheck",
    description="LLM hallucination detection API",
    version=__version__,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware ─────────────────────────────────────────────────────────────────

_OPEN_PATHS = {"/", "/health", "/docs", "/openapi.json", "/favicon.ico", "/metrics"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next) -> Response:
    """Enforce API key auth when CHAINCHECK_API_KEY is set."""
    if (
        _API_KEY
        and request.url.path not in _OPEN_PATHS
        and request.headers.get("X-API-Key", "") != _API_KEY
    ):
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key. Set X-API-Key header."},
        )
    return await call_next(request)


@app.middleware("http")
async def observability_middleware(request: Request, call_next) -> Response:
    """Attach X-Request-ID / X-Latency-Ms headers and record Prometheus metrics."""
    request_id = str(uuid.uuid4())
    endpoint = request.url.path
    start = time.time()
    active_requests_gauge.labels(endpoint=endpoint).inc()
    try:
        response = await call_next(request)
    finally:
        elapsed = (time.time() - start) * 1000
        active_requests_gauge.labels(endpoint=endpoint).dec()
        with suppress(Exception):
            requests_total.labels(
                endpoint=endpoint,
                method=request.method,
                status_code=str(response.status_code),
            ).inc()
            request_latency_ms.labels(endpoint=endpoint).observe(elapsed)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-Ms"] = f"{elapsed:.2f}"
    return response


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/check", response_model=DetectionResult)
@limiter.limit("60/minute")
async def check_endpoint(request: Request, body: CheckRequest) -> DetectionResult:
    """Detect hallucinations in a single LLM response."""
    result = await detect(
        body.response,
        context=body.context,
        prompt=body.prompt,
        methods=body.methods or None,  # type: ignore[arg-type]
        cascade=body.cascade,
        documents=body.documents or None,
    )
    _save_result(result)
    return result


@app.post("/stream", include_in_schema=True)
@limiter.limit("60/minute")
async def stream_endpoint(request: Request, body: CheckRequest) -> StreamingResponse:
    """
    Stream detection events via Server-Sent Events as each method completes.

    Event types:
      ``{"type": "claims",  "claims": [...], "request_id": "..."}``
      ``{"type": "method",  "method": "nli", "score": 0.72, "latency_ms": 230}``
      ``{"type": "result",  "data": {...DetectionResult...}}``
      ``data: [DONE]``
    """
    async def _event_gen():
        async for event in detect_stream(
            body.response,
            context=body.context,
            prompt=body.prompt,
            methods=body.methods or None,  # type: ignore[arg-type]
            documents=body.documents or None,
        ):
            yield f"data: {_json.dumps(event)}\n\n"
            if event.get("type") == "result":
                with suppress(Exception):
                    _save_result(DetectionResult.model_validate(event["data"]))
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/batch", response_model=list[DetectionResult])
@limiter.limit("20/minute")
async def batch_endpoint(request: Request, body: BatchRequest) -> list[DetectionResult]:
    """Detect hallucinations across a batch of LLM responses."""
    results = list(
        await asyncio.gather(
            *[
                detect(
                    inp.response,
                    context=inp.context,
                    prompt=inp.prompt,
                    methods=inp.methods or None,  # type: ignore[arg-type]
                    cascade=inp.cascade,
                    documents=inp.documents or None,
                )
                for inp in body.inputs
            ]
        )
    )
    for r in results:
        _save_result(r)
    return results


@app.get("/history")
async def history_endpoint(limit: int = 20) -> list[dict]:
    """Return the most recent detection results (max 100)."""
    return _db.fetch_history(min(max(1, limit), 100))


@app.get("/history/{request_id}")
async def history_detail_endpoint(request_id: str) -> dict:
    """Return the full DetectionResult payload for a single past scan."""
    payload = _db.fetch_result_payload(request_id)
    if not payload:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Result not found")
    return _json.loads(payload)


@app.get("/analytics")
async def analytics_endpoint() -> dict:
    """Aggregate detection history into charts-ready stats."""
    return _db.fetch_analytics()


@app.post("/feedback/{request_id}", status_code=204)
async def feedback_endpoint(request_id: str, body: FeedbackRequest) -> None:
    """Record whether a detection result was correct."""
    _db.save_feedback(request_id, body.correct, body.note)


@app.get("/health", response_model=HealthResponse)
async def health_endpoint() -> HealthResponse:
    """Return service liveness status, loaded model info, and DB backend."""
    return HealthResponse(
        status="ok",
        version=__version__,
        models_loaded=_models_loaded,
        db_backend="postgresql" if _db._IS_POSTGRES else "sqlite",
    )


@app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def metrics_endpoint() -> PlainTextResponse:
    """Expose Prometheus metrics in text format (scrape target for Prometheus)."""
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return PlainTextResponse(
        content=generate_latest().decode(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ── OpenAI-compatible proxy ────────────────────────────────────────────────────

async def _openai_request(api_key: str, body: dict) -> httpx.Response:  # pragma: no cover
    """Forward a request to OpenAI. Extracted for testability."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        return await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
        )


@app.post("/v1/chat/completions")
@limiter.limit("30/minute")
async def proxy_endpoint(request: Request) -> Response:
    """
    Drop-in OpenAI-compatible proxy with automatic hallucination detection.

    Forwards the request to OpenAI, runs ChainCheck on the response, and
    returns the original OpenAI payload with extra headers:
      X-Hallucination-Score  — aggregate score 0.0–1.0
      X-Risk-Level           — low | medium | high
      X-Request-ID           — ChainCheck trace ID
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return JSONResponse(status_code=503, content={"error": "OPENAI_API_KEY not configured"})

    body = await request.json()
    resp = await _openai_request(api_key, body)

    payload = resp.json()
    headers = dict(resp.headers)

    detection_result = None
    try:
        text = payload["choices"][0]["message"]["content"] or ""
        prompt_text = ""
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") == "user":
                prompt_text = msg.get("content", "")
                break
        detection_result = await detect(text, prompt=prompt_text, methods=["nli", "judge"])
        headers["X-Hallucination-Score"] = str(round(detection_result.aggregate_score, 4))
        headers["X-Risk-Level"] = detection_result.risk_level
        headers["X-Request-ID"] = detection_result.request_id or ""
        _save_result(detection_result)
    except Exception:
        pass

    _skip = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    clean_headers = {k: v for k, v in headers.items() if k.lower() not in _skip}

    if detection_result and detection_result.aggregate_score >= _PROXY_BLOCK_THRESHOLD:
        if _PROXY_MODE == "block":
            return JSONResponse(
                status_code=451,
                headers=clean_headers,
                content={
                    "error": {
                        "type": "hallucination_blocked",
                        "message": (
                            f"Response blocked by ChainCheck — hallucination score "
                            f"{detection_result.aggregate_score:.2f} exceeds threshold "
                            f"{_PROXY_BLOCK_THRESHOLD}."
                        ),
                        "score": detection_result.aggregate_score,
                        "risk_level": detection_result.risk_level,
                        "request_id": detection_result.request_id,
                    }
                },
            )
        if _PROXY_MODE == "warn":
            try:
                warning = (
                    f"\n\n⚠️ ChainCheck: This response has a hallucination risk score of "
                    f"{detection_result.aggregate_score:.0%} ({detection_result.risk_level} risk). "
                    "Verify key facts before acting on this information."
                )
                payload["choices"][0]["message"]["content"] += warning
                import json as _json_mod
                return Response(
                    content=_json_mod.dumps(payload).encode(),
                    status_code=resp.status_code,
                    headers=clean_headers,
                )
            except Exception:
                pass

    return Response(content=resp.content, status_code=resp.status_code, headers=clean_headers)


_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
async def ui() -> FileResponse:
    """Serve the ChainCheck web UI."""
    return FileResponse(_STATIC_DIR / "index.html")
