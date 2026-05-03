"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import warnings

# Suppress "coroutine never awaited" noise from tests that patch async functions
# but don't await the patched coroutine (e.g. when patching detect() in CLI tests).
warnings.filterwarnings("ignore", category=RuntimeWarning, message="coroutine.*never awaited")
