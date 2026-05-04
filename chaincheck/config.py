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
DECOMPOSE_MODEL: str = os.getenv("DECOMPOSE_MODEL", "gpt-4o-mini")
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
CONSISTENCY_MODEL: str = os.getenv("CONSISTENCY_MODEL", "gpt-4o-mini")
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
# Tuned via Nelder-Mead on 80% of HaluEval QA (n=500); held-out F1=0.741 vs 0.500 with old weights.
# Consistency excluded (F1=0.168 on factual tasks — below random; weight=0 disables it in ensemble).
WEIGHT_NLI: float = float(os.getenv("NLI_WEIGHT", "0.10"))
WEIGHT_CONSISTENCY: float = float(os.getenv("CONSISTENCY_WEIGHT", "0.0"))
WEIGHT_JUDGE: float = float(os.getenv("JUDGE_WEIGHT", "0.60"))
WEIGHT_LOGPROBS: float = float(os.getenv("LOGPROB_WEIGHT", "0.30"))

# ── Risk thresholds ───────────────────────────────────────────────────────────
RISK_LOW_THRESHOLD: float = float(os.getenv("RISK_LOW_THRESHOLD", "0.3"))
RISK_HIGH_THRESHOLD: float = float(os.getenv("RISK_HIGH_THRESHOLD", "0.7"))
