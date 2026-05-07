"""
Prometheus metrics for ChainCheck.

Instruments:
  chaincheck_requests_total        — counter by endpoint, method, status_code
  chaincheck_request_latency_ms    — histogram of end-to-end request latency
  chaincheck_hallucination_score   — histogram of aggregate hallucination scores
  chaincheck_detection_method_used — counter per detection method
  chaincheck_models_loaded         — gauge (0 or 1)
  chaincheck_risk_level_total      — counter by risk_level (low/medium/high)

Usage:
  from chaincheck.metrics_prom import (
      record_detection, record_request, models_loaded_gauge
  )
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, multiprocess

# Use the default registry (works with both single-process and gunicorn multiprocess)
try:
    from prometheus_client import REGISTRY as _registry
except ImportError:
    _registry = CollectorRegistry()


# ── Counters ───────────────────────────────────────────────────────────────────

requests_total = Counter(
    "chaincheck_requests_total",
    "Total HTTP requests handled",
    ["endpoint", "method", "status_code"],
)

detection_method_used = Counter(
    "chaincheck_detection_method_used_total",
    "Number of times each detection method was invoked",
    ["method"],
)

risk_level_total = Counter(
    "chaincheck_risk_level_total",
    "Hallucination detections by risk level",
    ["risk_level"],
)


# ── Histograms ─────────────────────────────────────────────────────────────────

request_latency_ms = Histogram(
    "chaincheck_request_latency_ms",
    "End-to-end HTTP request latency in milliseconds",
    ["endpoint"],
    buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
)

hallucination_score = Histogram(
    "chaincheck_hallucination_score",
    "Distribution of aggregate hallucination scores (0.0 – 1.0)",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

detection_latency_ms = Histogram(
    "chaincheck_detection_latency_ms",
    "Time spent in detect() per method",
    ["method"],
    buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000],
)


# ── Gauges ─────────────────────────────────────────────────────────────────────

models_loaded_gauge = Gauge(
    "chaincheck_models_loaded",
    "1 when NLI and embedding models are fully loaded, 0 otherwise",
)

active_requests_gauge = Gauge(
    "chaincheck_active_requests",
    "Number of requests currently being processed",
    ["endpoint"],
)


# ── Convenience helpers ────────────────────────────────────────────────────────

def record_detection(result) -> None:
    """
    Record metrics from a completed DetectionResult.

    Call this after every successful detect() invocation.
    """
    hallucination_score.observe(result.aggregate_score)
    risk_level_total.labels(risk_level=result.risk_level).inc()

    for method_name, method_result in result.method_results.items():
        detection_method_used.labels(method=method_name).inc()
        if method_result.latency_ms:
            detection_latency_ms.labels(method=method_name).observe(
                method_result.latency_ms
            )
