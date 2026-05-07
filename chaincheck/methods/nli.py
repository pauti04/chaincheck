"""
NLI-based entailment method for claim verification.

Uses cross-encoder/nli-deberta-v3-base to classify whether each atomic claim
is entailed, neutral, or contradicted by the retrieved context. Claims are
batched in groups of 16 for throughput, targeting <80 ms per claim on CPU.
"""

from __future__ import annotations

import hashlib
import os
import re
import time

import numpy as np

from chaincheck.models import ClaimResult, Document, MethodResult

_NLI_MODEL_NAME = os.getenv(
    "CHAINCHECK_NLI_MODEL",
    "cross-encoder/nli-deberta-v3-base",
)
_BATCH_SIZE = int(os.getenv("NLI_BATCH_SIZE", "16"))
_NLI_THRESHOLD = float(os.getenv("NLI_THRESHOLD", "0.5"))

_model = None
_label_map: dict[int, str] | None = None

# When CHAINCHECK_NLI_MODEL points to a fine-tuned seq-classification model
# (e.g. the output of notebooks/deberta_finetune.ipynb), we use the HuggingFace
# pipeline directly instead of CrossEncoder.
_USE_HF_PIPELINE = bool(os.getenv("CHAINCHECK_NLI_MODEL", ""))


def _get_model():
    """Lazily load the NLI model (once per process).

    Supports two backends:
    * CrossEncoder (default) — cross-encoder/nli-deberta-v3-base
    * HuggingFace pipeline — any AutoModelForSequenceClassification pointed to
      by CHAINCHECK_NLI_MODEL (e.g. the fine-tuned DeBERTa from the Colab notebook)
    """
    global _model, _label_map
    if _model is None:
        if _USE_HF_PIPELINE:
            from transformers import pipeline as hf_pipeline

            _model = hf_pipeline(
                "text-classification",
                model=_NLI_MODEL_NAME,
                truncation=True,
                max_length=512,
                device=-1,  # CPU; set to 0 for GPU
            )
            _label_map = None  # pipeline returns label names directly
        else:
            from sentence_transformers import CrossEncoder

            _model = CrossEncoder(_NLI_MODEL_NAME, num_labels=3)
            _label_map = _build_label_map(_model)
    return _model


def _build_label_map(model) -> dict[int, str]:
    """
    Derive entailment/contradiction/neutral index mapping from model config.

    Falls back to a known-correct mapping for nli-deberta-v3-base if the
    config labels are missing.
    """
    id2label: dict = getattr(getattr(model, "config", None), "id2label", {})
    mapping: dict[int, str] = {}
    for idx, label in id2label.items():
        lower = label.lower()
        if "entail" in lower:
            mapping[int(idx)] = "supported"
        elif "contradict" in lower:
            mapping[int(idx)] = "contradicted"
        else:
            mapping[int(idx)] = "unknown"
    if not mapping:
        # cross-encoder/nli-deberta-v3-base: {0: contradiction, 1: entailment, 2: neutral}
        mapping = {0: "contradicted", 1: "supported", 2: "unknown"}
    return mapping


async def check_nli(
    claims: list[str],
    context: str,
) -> MethodResult:
    """
    Verify claims against context using NLI entailment.

    Args:
        claims: Atomic claim strings to verify.
        context: Reference text to verify claims against.

    Returns:
        MethodResult with a ClaimResult for each input claim.
    """
    if not claims:
        return MethodResult(method="nli", raw_score=0.0, latency_ms=0.0)

    if not context.strip():
        claim_results = [
            ClaimResult(
                claim=c,
                label="unknown",
                confidence=0.0,
                evidence="no context provided",
                method="nli",
            )
            for c in claims
        ]
        return MethodResult(method="nli", claims=claim_results, raw_score=0.0, latency_ms=0.0)

    start = time.time()
    _get_model()

    # NLI convention: premise = context, hypothesis = claim
    pairs = [(context, claim) for claim in claims]
    raw_preds = _batch_predict(pairs)

    claim_results = []
    for claim, pred in zip(claims, raw_preds, strict=True):
        label_idx = int(np.argmax(pred["scores"]))
        label = (_label_map or {})[label_idx]
        confidence = float(pred["scores"][label_idx])
        evidence = _find_best_evidence(claim, context)
        claim_results.append(
            ClaimResult(
                claim=claim,
                label=label,
                confidence=confidence,
                evidence=evidence,
                method="nli",
            )
        )

    raw_score = _score_from_claims(claim_results)
    latency = (time.time() - start) * 1000
    return MethodResult(
        method="nli", claims=claim_results, raw_score=raw_score, latency_ms=latency
    )


def _batch_predict(
    pairs: list[tuple[str, str]],
    batch_size: int = _BATCH_SIZE,
) -> list[dict]:
    """
    Run NLI inference on (premise, hypothesis) pairs in fixed-size batches.

    Supports two backends:

    * **CrossEncoder** (default): returns ``{"scores": [s0, s1, s2]}`` where
      indices are mapped via ``_label_map`` to supported/contradicted/unknown.
    * **HuggingFace pipeline** (CHAINCHECK_NLI_MODEL set): the fine-tuned binary
      classifier from ``notebooks/deberta_finetune.ipynb``. Returns the same
      ``{"scores": [p_supported, p_contradicted]}`` format so downstream code
      is unchanged.  ``_label_map`` is set to ``{0: "supported", 1: "contradicted"}``
      for these results.
    """
    global _label_map
    model = _get_model()
    results = []

    if _USE_HF_PIPELINE:
        # Flatten pairs to single strings: "context [SEP] claim"
        texts = [f"{prem}\n{hyp}" for prem, hyp in pairs]
        _label_map = {0: "supported", 1: "contradicted"}
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            outputs = model(batch_texts)
            for out in outputs:
                lbl = (out["label"] or "").lower()
                score = float(out["score"])
                if "hallucinated" in lbl or "contradict" in lbl:
                    results.append({"scores": [1.0 - score, score]})
                else:
                    results.append({"scores": [score, 1.0 - score]})
    else:
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            scores = model.predict(batch, apply_softmax=True)
            for score_vec in scores:
                results.append({"scores": score_vec.tolist()})
    return results


def _find_best_evidence(claim: str, context: str) -> str:
    """
    Return the most relevant sentence from context for a given claim.

    Uses word-overlap heuristic to rank context sentences; returns the
    top-scoring sentence or 'no relevant context found'.
    """
    if not context.strip():
        return "no relevant context found"
    sentences = re.split(r"(?<=[.!?])\s+", context.strip())
    claim_words = set(claim.lower().split())
    best = max(sentences, key=lambda s: len(claim_words & set(s.lower().split())))
    if not (claim_words & set(best.lower().split())):
        return "no relevant context found"
    return best


def _score_from_claims(claims: list[ClaimResult]) -> float:
    """Compute hallucination risk as mean confidence of bad claims across all claims.
    Falls back to unweighted count ratio when all confidences are zero."""
    if not claims:
        return 0.0
    bad = {"unsupported", "contradicted"}
    weighted = sum(c.confidence for c in claims if c.label in bad)
    if weighted == 0.0:
        return sum(1 for c in claims if c.label in bad) / len(claims)
    return min(1.0, weighted / len(claims))


def attribute_to_documents(
    claims: list[str],
    documents: list[Document],
) -> list[tuple[str | None, str | None]]:
    """
    For each claim, find the document that best supports or contradicts it.

    Runs NLI on every (document, claim) pair in a single batch and picks the
    document whose entailment score is highest (for supported claims) or whose
    contradiction score is highest (for contradicted claims).

    Returns a list of (source_id, source_url) tuples, one per claim.
    """
    if not documents or not claims:
        return [(None, None)] * len(claims)

    _get_model()
    pairs: list[tuple[str, str]] = []
    for claim in claims:
        for doc in documents:
            pairs.append((doc.content, claim))

    preds = _batch_predict(pairs)
    n_docs = len(documents)
    results: list[tuple[str | None, str | None]] = []

    label_map = _label_map or {0: "contradicted", 1: "supported", 2: "unknown"}
    ent_idx = next((k for k, v in label_map.items() if v == "supported"), 1)
    con_idx = next((k for k, v in label_map.items() if v == "contradicted"), 0)

    for ci in range(len(claims)):
        doc_preds = preds[ci * n_docs : (ci + 1) * n_docs]
        best_score = -1.0
        best_doc: Document | None = None
        for di, pred in enumerate(doc_preds):
            score = max(pred["scores"][ent_idx], pred["scores"][con_idx])
            if score > best_score:
                best_score = score
                best_doc = documents[di]
        doc_id  = best_doc.id if best_doc else None
        doc_url = best_doc.url or None if best_doc else None
        results.append((doc_id, doc_url))

    return results


def _claim_context_hash(claim: str, context: str) -> str:
    """Return a cache key for a (claim, context) pair."""
    return hashlib.sha256(f"{claim}|||{context}".encode()).hexdigest()
