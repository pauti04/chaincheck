"""
ChainCheck Model Server — standalone DeBERTa inference service.

Runs as a completely separate process/pod from the main API.
Loads the NLI model once at startup and serves predictions via REST,
so the main API pod stays CPU-light and scales independently.

Start with:
    uvicorn chaincheck.model_server:app --port 8001

Or via Docker Compose / Kubernetes (see k8s/model-server-deployment.yaml).

Endpoints:
  POST /predict      — run NLI inference on (premise, hypothesis) pairs
  GET  /health       — liveness check
  GET  /metrics      — Prometheus metrics
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

_MODEL_NAME = os.getenv(
    "CHAINCHECK_NLI_MODEL",
    "cross-encoder/nli-deberta-v3-base",
)
_BATCH_SIZE = int(os.getenv("NLI_BATCH_SIZE", "32"))  # larger batch = better GPU util
_USE_GPU = os.getenv("USE_GPU", "0") == "1"

_model = None
_is_ready = False


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup; release at shutdown."""
    global _model, _is_ready
    import asyncio

    loop = asyncio.get_event_loop()

    def _load():
        global _model
        if _MODEL_NAME.startswith(("./", "/")) or os.path.isdir(_MODEL_NAME):
            # Fine-tuned binary classifier (output of notebooks/deberta_finetune.ipynb)
            from transformers import pipeline as hf_pipeline

            device = 0 if _USE_GPU else -1
            _model = hf_pipeline(
                "text-classification",
                model=_MODEL_NAME,
                truncation=True,
                max_length=512,
                device=device,
                batch_size=_BATCH_SIZE,
            )
        else:
            # Default cross-encoder for 3-class NLI
            from sentence_transformers import CrossEncoder

            _model = CrossEncoder(_MODEL_NAME, num_labels=3)

    await loop.run_in_executor(None, _load)
    _is_ready = True
    print(f"[model-server] Model '{_MODEL_NAME}' loaded. GPU={_USE_GPU}", flush=True)
    yield
    _model = None
    _is_ready = False


app = FastAPI(
    title="ChainCheck Model Server",
    description="Dedicated NLI inference service for ChainCheck",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Batch of (premise, hypothesis) pairs for NLI inference."""

    pairs: list[tuple[str, str]] = Field(
        min_length=1,
        description="List of (premise, hypothesis) string pairs",
    )


class PredictResult(BaseModel):
    """NLI prediction for a single (premise, hypothesis) pair."""

    label: str   # "supported" | "contradicted" | "unknown" | "hallucinated" | "truthful"
    score: float  # confidence of the winning label (0.0–1.0)
    scores: list[float]  # raw scores for all labels


class PredictResponse(BaseModel):
    """Response containing one PredictResult per input pair."""

    results: list[PredictResult]
    latency_ms: float


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """Liveness + readiness probe."""
    return {
        "status": "ok" if _is_ready else "loading",
        "model": _MODEL_NAME,
        "ready": _is_ready,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest) -> PredictResponse:
    """
    Run NLI inference on a batch of (premise, hypothesis) pairs.

    For the default cross-encoder model:
      scores = [contradiction, entailment, neutral]

    For a fine-tuned binary classifier (output of deberta_finetune.ipynb):
      scores = [p_truthful, p_hallucinated]
    """
    import asyncio

    start = time.time()
    loop = asyncio.get_event_loop()

    def _run() -> list[PredictResult]:
        assert _model is not None
        results: list[PredictResult] = []

        if hasattr(_model, "predict"):
            # CrossEncoder path
            import numpy as np

            raw = _model.predict(body.pairs, apply_softmax=True)
            label_map = {0: "contradicted", 1: "supported", 2: "unknown"}
            # Try to infer label map from model config
            id2label = getattr(getattr(_model, "config", None), "id2label", {})
            if id2label:
                label_map = {}
                for idx, lbl in id2label.items():
                    lower = lbl.lower()
                    label_map[int(idx)] = (
                        "supported" if "entail" in lower
                        else "contradicted" if "contradict" in lower
                        else "unknown"
                    )
            for score_vec in raw:
                idx = int(np.argmax(score_vec))
                results.append(PredictResult(
                    label=label_map.get(idx, "unknown"),
                    score=float(score_vec[idx]),
                    scores=score_vec.tolist(),
                ))
        else:
            # HuggingFace pipeline (binary fine-tuned classifier)
            texts = [f"{p}\n{h}" for p, h in body.pairs]
            for out in _model(texts):
                lbl = (out["label"] or "").lower()
                sc = float(out["score"])
                if "hallucinated" in lbl or "contradict" in lbl:
                    mapped = "contradicted"
                    raw_scores = [1.0 - sc, sc]
                else:
                    mapped = "supported"
                    raw_scores = [sc, 1.0 - sc]
                results.append(PredictResult(
                    label=mapped, score=sc, scores=raw_scores
                ))
        return results

    results = await loop.run_in_executor(None, _run)
    return PredictResponse(
        results=results,
        latency_ms=(time.time() - start) * 1000,
    )


@app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def metrics() -> str:
    """Prometheus text-format metrics."""
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return PlainTextResponse(
        content=generate_latest().decode(),
        media_type=CONTENT_TYPE_LATEST,
    )
