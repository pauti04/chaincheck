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
from chaincheck.models import DetectionResult

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


class BatchRequest(BaseModel):
    """Request body for POST /batch."""

    inputs: list[CheckRequest]


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str
    version: str
    models_loaded: bool


# ── App setup ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm NLI and embedding models; initialise history DB."""
    global _models_loaded
    from chaincheck.methods.consistency import _get_embed_model
    from chaincheck.methods.nli import _get_model

    _get_model()
    _get_embed_model()
    _models_loaded = True
    _init_history_db()
    yield


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


@app.get("/health", response_model=HealthResponse)
async def health_endpoint() -> HealthResponse:
    """Return service liveness status and loaded model info."""
    return HealthResponse(status="ok", version=__version__, models_loaded=_models_loaded)


_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
async def ui() -> FileResponse:
    """Serve the ChainCheck web UI."""
    return FileResponse(_STATIC_DIR / "index.html")
