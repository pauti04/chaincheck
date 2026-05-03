"""Environment-backed configuration for ChainCheck."""

from __future__ import annotations

import os
from pathlib import Path


def env_float(name: str, default: float) -> float:
    """Read a float from the environment."""
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    """Read an integer from the environment."""
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def cache_path() -> Path:
    """Return the configured cache path."""
    return Path(os.getenv("CACHE_PATH", ".chaincheck_cache"))


def method_weights(selected: list[str]) -> dict[str, float]:
    """Return normalized method weights for selected detectors."""
    weights = {
        "nli": env_float("WEIGHT_NLI", 0.4),
        "consistency": env_float("WEIGHT_CONSISTENCY", 0.3),
        "judge": env_float("WEIGHT_JUDGE", 0.3),
    }
    chosen = {name: weights[name] for name in selected if name in weights}
    total = sum(chosen.values())
    return {name: value / total for name, value in chosen.items()} if total else {}
