"""
Centralised environment-backed configuration for ChainCheck.

All settings can be overridden via environment variables. Import individual
constants rather than the module to avoid re-reading the environment on every
access:

    from chaincheck.config import JUDGE_MODEL, WEIGHT_NLI
"""

from __future__ import annotations

import os

# ── Cache ──────────────────────────────────────────────────────────────────────
CACHE_PATH: str = os.getenv("CACHE_PATH", ".chaincheck_cache")

# ── Models ────────────────────────────────────────────────────────────────────
DECOMPOSE_MODEL: str = os.getenv("DECOMPOSE_MODEL", "claude-haiku-4-5-20251001")
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "claude-haiku-4-5-20251001")
CONSISTENCY_MODEL: str = os.getenv("CONSISTENCY_MODEL", "claude-haiku-4-5-20251001")
LOGPROB_MODEL: str = os.getenv("LOGPROB_MODEL", "gpt-4o-mini")

# ── NLI ───────────────────────────────────────────────────────────────────────
NLI_BATCH_SIZE: int = int(os.getenv("NLI_BATCH_SIZE", "16"))
NLI_THRESHOLD: float = float(os.getenv("NLI_THRESHOLD", "0.5"))

# ── Consistency ───────────────────────────────────────────────────────────────
CONSISTENCY_SAMPLES: int = int(os.getenv("CONSISTENCY_SAMPLES", "5"))
CONSISTENCY_THRESHOLD: float = float(os.getenv("CONSISTENCY_THRESHOLD", "0.82"))

# ── Logprobs ──────────────────────────────────────────────────────────────────
LOGPROB_THRESHOLD: float = float(os.getenv("LOGPROB_THRESHOLD", "-1.5"))

# ── Aggregation weights ───────────────────────────────────────────────────────
WEIGHT_NLI: float = float(os.getenv("NLI_WEIGHT", "0.35"))
WEIGHT_CONSISTENCY: float = float(os.getenv("CONSISTENCY_WEIGHT", "0.25"))
WEIGHT_JUDGE: float = float(os.getenv("JUDGE_WEIGHT", "0.25"))
WEIGHT_LOGPROBS: float = float(os.getenv("LOGPROB_WEIGHT", "0.15"))

# ── Risk thresholds ───────────────────────────────────────────────────────────
RISK_LOW_THRESHOLD: float = float(os.getenv("RISK_LOW_THRESHOLD", "0.3"))
RISK_HIGH_THRESHOLD: float = float(os.getenv("RISK_HIGH_THRESHOLD", "0.7"))
