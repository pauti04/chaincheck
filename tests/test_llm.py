"""Tests for provider-independent LLM helpers."""

import httpx
import pytest

from chaincheck import llm


def test_provider_for_model() -> None:
    """Provider inference follows model naming conventions."""
    assert llm.provider_for_model("gpt-4o-mini") == "openai"
    assert llm.provider_for_model("ollama:llama3") == "ollama"
    assert llm.provider_for_model("claude-haiku-4-5") == "anthropic"


def test_anthropic_model_alias() -> None:
    """Anthropic aliases map friendly model names."""
    assert llm._anthropic_model_alias("claude-haiku-4-5") == "claude-3-5-haiku-latest"
    assert llm._anthropic_model_alias("custom") == "custom"


@pytest.mark.asyncio
async def test_complete_missing_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI calls fail clearly when the key is missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(llm.LLMError, match="OPENAI_API_KEY"):
        await llm.complete("prompt", "gpt-4o-mini")


@pytest.mark.asyncio
async def test_complete_missing_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anthropic calls fail clearly when the key is missing."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(llm.LLMError, match="ANTHROPIC_API_KEY"):
        await llm.complete("prompt", "claude-haiku-4-5")


def test_raise_for_status_wraps_http_error() -> None:
    """HTTP errors are wrapped in LLMError."""
    request = httpx.Request("GET", "https://example.test")
    response = httpx.Response(500, request=request)
    with pytest.raises(llm.LLMError):
        llm._raise_for_status(response)
