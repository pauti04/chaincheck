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
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import httpx

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
_PROXY_BLOCK_THRESHOLD = float(os.getenv("PROXY_BLOCK_THRESHOLD", "0.8"))
_PROXY_MODE = os.getenv("PROXY_MODE", "passthrough")  # passthrough | warn | block


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


@app.get("/history/{request_id}")
async def history_detail_endpoint(request_id: str) -> dict:
    """Return the full DetectionResult payload for a single past scan."""
    with sqlite3.connect(_HISTORY_DB) as conn:
        row = conn.execute(
            "SELECT payload FROM results WHERE request_id = ?", (request_id,)
        ).fetchone()
    if not row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Result not found")
    return _json.loads(row[0])


@app.get("/analytics")
async def analytics_endpoint() -> dict:
    """Aggregate detection history into charts-ready stats."""
    with sqlite3.connect(_HISTORY_DB) as conn:
        risk_rows = conn.execute(
            "SELECT risk_level, COUNT(*) FROM results GROUP BY risk_level"
        ).fetchall()
        trend_rows = conn.execute(
            """SELECT created_at, aggregate_score, risk_level
               FROM results ORDER BY created_at DESC LIMIT 60"""
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        avg_latency = conn.execute(
            "SELECT AVG(total_latency_ms) FROM results"
        ).fetchone()[0]
        fb_rows = conn.execute(
            """SELECT r.risk_level, f.correct, COUNT(*)
               FROM feedback f
               JOIN results r ON f.request_id = r.request_id
               GROUP BY r.risk_level, f.correct"""
        ).fetchall()
        method_rows = conn.execute(
            "SELECT methods FROM results WHERE methods IS NOT NULL"
        ).fetchall()

    risk_dist = {r[0]: r[1] for r in risk_rows}

    score_trend = [
        {"ts": r[0], "score": round(r[1], 4), "risk": r[2]}
        for r in reversed(trend_rows)
    ]

    method_counts: dict[str, int] = {}
    for (methods_json,) in method_rows:
        for m in _json.loads(methods_json):
            method_counts[m] = method_counts.get(m, 0) + 1

    feedback_accuracy: dict[str, dict] = {}
    for risk_level, correct, count in fb_rows:
        if risk_level not in feedback_accuracy:
            feedback_accuracy[risk_level] = {"correct": 0, "incorrect": 0}
        key = "correct" if correct else "incorrect"
        feedback_accuracy[risk_level][key] += count

    return {
        "total_scans": total,
        "avg_latency_ms": round(avg_latency or 0, 1),
        "risk_distribution": risk_dist,
        "score_trend": score_trend,
        "method_usage": method_counts,
        "feedback_accuracy": feedback_accuracy,
    }


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
