"""
FastAPI server exposing ChainCheck detection via REST API.

Endpoints:
  POST /check    — detect hallucinations in a single response
  POST /stream   — same as /check but streams events via SSE as each method completes
  POST /batch    — batch detection for multiple inputs
  GET  /history  — recent detection results (persisted in SQLite)
  GET  /health   — liveness check with version and model status
  GET  /docs     — auto-generated OpenAPI documentation
"""

from __future__ import annotations

import asyncio
import json as _json
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from chaincheck import __version__
from chaincheck.detect import detect, detect_stream
from chaincheck.models import DetectionResult, Document

limiter = Limiter(key_func=get_remote_address)

_models_loaded = False
_API_KEY = os.getenv("CHAINCHECK_API_KEY", "")
_HISTORY_DB = Path(os.getenv("HISTORY_DB", "chaincheck_history.db"))


# ── History persistence ────────────────────────────────────────────────────────

def _init_history_db() -> None:
    with sqlite3.connect(_HISTORY_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                request_id      TEXT PRIMARY KEY,
                created_at      REAL NOT NULL,
                response_preview TEXT,
                aggregate_score REAL,
                risk_level      TEXT,
                total_latency_ms REAL,
                methods         TEXT,
                payload         TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON results (created_at DESC)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id  TEXT NOT NULL,
                correct     INTEGER NOT NULL,
                note        TEXT,
                created_at  REAL NOT NULL
            )
        """)


def _save_result(result: DetectionResult) -> None:
    """Persist a DetectionResult to SQLite. Best-effort — never raises."""
    try:
        with sqlite3.connect(_HISTORY_DB) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?,?)",
                (
                    result.request_id,
                    time.time(),
                    result.response[:120],
                    result.aggregate_score,
                    result.risk_level,
                    result.total_latency_ms,
                    _json.dumps(sorted(result.method_results.keys())),
                    result.model_dump_json(),
                ),
            )
    except Exception:
        pass


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

_OPEN_PATHS = {"/", "/health", "/docs", "/openapi.json", "/favicon.ico"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next) -> Response:
    """Enforce API key auth when CHAINCHECK_API_KEY is set."""
    if _API_KEY and request.url.path not in _OPEN_PATHS:
        if request.headers.get("X-API-Key", "") != _API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key. Set X-API-Key header."},
            )
    return await call_next(request)


@app.middleware("http")
async def add_request_id_and_latency(request: Request, call_next) -> Response:
    """Attach X-Request-ID and X-Latency-Ms headers to every response."""
    request_id = str(uuid.uuid4())
    start = time.time()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-Ms"] = f"{(time.time() - start) * 1000:.2f}"
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
                try:
                    _save_result(DetectionResult.model_validate(event["data"]))
                except Exception:
                    pass
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
    limit = min(max(1, limit), 100)
    with sqlite3.connect(_HISTORY_DB) as conn:
        rows = conn.execute(
            """SELECT request_id, created_at, response_preview,
                      aggregate_score, risk_level, total_latency_ms, methods
               FROM results ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [
        {
            "request_id": r[0],
            "created_at": r[1],
            "response_preview": r[2],
            "aggregate_score": r[3],
            "risk_level": r[4],
            "total_latency_ms": r[5],
            "methods": _json.loads(r[6]),
        }
        for r in rows
    ]


@app.post("/feedback/{request_id}", status_code=204)
async def feedback_endpoint(request_id: str, body: FeedbackRequest) -> None:
    """Record whether a detection result was correct."""
    try:
        with sqlite3.connect(_HISTORY_DB) as conn:
            conn.execute(
                "INSERT INTO feedback (request_id, correct, note, created_at) VALUES (?,?,?,?)",
                (request_id, int(body.correct), body.note, time.time()),
            )
    except Exception:
        pass


@app.get("/health", response_model=HealthResponse)
async def health_endpoint() -> HealthResponse:
    """Return service liveness status and loaded model info."""
    return HealthResponse(status="ok", version=__version__, models_loaded=_models_loaded)


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
    import httpx

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return JSONResponse(status_code=503, content={"error": "OPENAI_API_KEY not configured"})

    body = await request.json()
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
        )

    payload = resp.json()
    headers = dict(resp.headers)

    try:
        text = payload["choices"][0]["message"]["content"] or ""
        prompt_text = ""
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") == "user":
                prompt_text = msg.get("content", "")
                break
        result = await detect(text, prompt=prompt_text, methods=["nli", "judge"])
        headers["X-Hallucination-Score"] = str(round(result.aggregate_score, 4))
        headers["X-Risk-Level"] = result.risk_level
        headers["X-Request-ID"] = result.request_id or ""
        _save_result(result)
    except Exception:
        pass

    _skip = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    clean_headers = {k: v for k, v in headers.items() if k.lower() not in _skip}
    return Response(content=resp.content, status_code=resp.status_code, headers=clean_headers)


_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
async def ui() -> FileResponse:
    """Serve the ChainCheck web UI."""
    return FileResponse(_STATIC_DIR / "index.html")
