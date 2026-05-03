"""
FastAPI server exposing ChainCheck detection via REST API.

Endpoints:
  POST /check   — detect hallucinations in a single response
  POST /batch   — batch detection for multiple inputs
  GET  /health  — liveness check with version and model status
  GET  /docs    — auto-generated OpenAPI documentation
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from chaincheck import __version__
from chaincheck.detect import detect
from chaincheck.models import DetectionResult

limiter = Limiter(key_func=get_remote_address)

_models_loaded = False


class CheckRequest(BaseModel):
    """Request body for POST /check."""

    response: str = Field(min_length=1)
    context: str = ""
    prompt: str = ""
    methods: list[str] = Field(default_factory=lambda: ["nli", "consistency", "judge"])


class BatchRequest(BaseModel):
    """Request body for POST /batch."""

    inputs: list[CheckRequest]


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str
    version: str
    models_loaded: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm NLI and embedding models on startup to eliminate cold-start latency."""
    global _models_loaded
    from chaincheck.methods.consistency import _get_embed_model
    from chaincheck.methods.nli import _get_model

    _get_model()
    _get_embed_model()
    _models_loaded = True
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


@app.middleware("http")
async def add_request_id_and_latency(request: Request, call_next) -> Response:
    """Attach X-Request-ID and X-Latency-Ms headers to every response."""
    request_id = str(uuid.uuid4())
    start = time.time()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-Ms"] = f"{(time.time() - start) * 1000:.2f}"
    return response


@app.post("/check", response_model=DetectionResult)
@limiter.limit("60/minute")
async def check_endpoint(request: Request, body: CheckRequest) -> DetectionResult:
    """Detect hallucinations in a single LLM response."""
    return await detect(
        body.response,
        context=body.context,
        prompt=body.prompt,
        methods=body.methods or None,  # type: ignore[arg-type]
    )


@app.post("/batch", response_model=list[DetectionResult])
@limiter.limit("20/minute")
async def batch_endpoint(request: Request, body: BatchRequest) -> list[DetectionResult]:
    """Detect hallucinations across a batch of LLM responses."""
    return list(
        await asyncio.gather(
            *[
                detect(
                    inp.response,
                    context=inp.context,
                    prompt=inp.prompt,
                    methods=inp.methods or None,  # type: ignore[arg-type]
                )
                for inp in body.inputs
            ]
        )
    )


@app.get("/health", response_model=HealthResponse)
async def health_endpoint() -> HealthResponse:
    """Return service liveness status and loaded model info."""
    return HealthResponse(status="ok", version=__version__, models_loaded=_models_loaded)
