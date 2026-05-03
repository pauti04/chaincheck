"""Tests for chaincheck.config — environment-backed configuration defaults."""

from __future__ import annotations

from unittest.mock import patch


class TestConfigDefaults:
    def test_cache_path_default(self):
        from chaincheck.config import CACHE_PATH

        assert isinstance(CACHE_PATH, str)
        assert len(CACHE_PATH) > 0

    def test_model_defaults_are_strings(self):
        from chaincheck.config import CONSISTENCY_MODEL, DECOMPOSE_MODEL, JUDGE_MODEL, LOGPROB_MODEL

        for model in (DECOMPOSE_MODEL, JUDGE_MODEL, CONSISTENCY_MODEL, LOGPROB_MODEL):
            assert isinstance(model, str)
            assert len(model) > 0

    def test_numeric_defaults_are_in_range(self):
        from chaincheck.config import (
            CONSISTENCY_SAMPLES,
            CONSISTENCY_THRESHOLD,
            NLI_BATCH_SIZE,
            NLI_THRESHOLD,
        )

        assert NLI_BATCH_SIZE > 0
        assert 0.0 < NLI_THRESHOLD < 1.0
        assert CONSISTENCY_SAMPLES > 0
        assert 0.0 < CONSISTENCY_THRESHOLD < 1.0

    def test_weights_sum_to_one(self):
        from chaincheck.config import WEIGHT_CONSISTENCY, WEIGHT_JUDGE, WEIGHT_LOGPROBS, WEIGHT_NLI

        total = WEIGHT_NLI + WEIGHT_CONSISTENCY + WEIGHT_JUDGE + WEIGHT_LOGPROBS
        assert abs(total - 1.0) < 1e-6

    def test_risk_thresholds_ordered(self):
        from chaincheck.config import RISK_HIGH_THRESHOLD, RISK_LOW_THRESHOLD

        assert RISK_LOW_THRESHOLD < RISK_HIGH_THRESHOLD

    def test_env_override(self):
        with patch.dict("os.environ", {"JUDGE_MODEL": "gpt-4o"}):
            import importlib

            import chaincheck.config as cfg

            importlib.reload(cfg)
            assert cfg.JUDGE_MODEL == "gpt-4o"
            importlib.reload(cfg)
