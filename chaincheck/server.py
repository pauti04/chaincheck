"""FastAPI server for ChainCheck."""

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

from chaincheck import __version__
from chaincheck.detect import DetectionResult, detect
from chaincheck.methods.consistency import _embedding_model
from chaincheck.methods.nli import _model

app = FastAPI(title="ChainCheck", version="0.1.0")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CheckRequest(BaseModel):
    """Request schema for one ChainCheck detection."""

    response: str
    context: str = ""
    prompt: str = ""
    methods: list[str] = Field(default_factory=lambda: ["nli", "consistency", "judge"])


class BatchRequest(BaseModel):
    """Request schema for batch ChainCheck detection."""

    inputs: list[CheckRequest]


@app.middleware("http")
async def request_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Attach request IDs and latency metrics to every response."""
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    started = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - started) * 1000
    response.headers["x-request-id"] = request_id
    response.headers["x-chaincheck-latency-ms"] = f"{latency_ms:.2f}"
    return response


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Return JSON for rate limit errors."""
    del request, exc
    return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})


@app.get("/health")
async def health() -> dict[str, object]:
    """Return server health and model loading status."""
    return {"status": "ok", "version": __version__, "models_loaded": _models_loaded()}


@app.post("/check", response_model=DetectionResult)
@limiter.limit("60/minute")
async def check(payload: CheckRequest, request: Request) -> DetectionResult:
    """Run hallucination detection for one response."""
    del request
    return await detect(
        response=payload.response,
        context=payload.context,
        prompt=payload.prompt,
        methods=payload.methods,
    )


@app.post("/batch", response_model=list[DetectionResult])
@limiter.limit("60/minute")
async def batch(payload: BatchRequest, request: Request) -> list[DetectionResult]:
    """Run hallucination detection for a batch of responses."""
    del request
    tasks = [
        detect(item.response, context=item.context, prompt=item.prompt, methods=item.methods)
        for item in payload.inputs
    ]
    return list(await asyncio.gather(*tasks))


def _models_loaded() -> dict[str, bool]:
    return {
        "nli": _model.cache_info().currsize > 0,
        "embedding": _embedding_model.cache_info().currsize > 0,
    }
