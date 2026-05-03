"""Tests for chaincheck.llm — multi-provider LLM routing."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chaincheck.llm import LLMError, provider_for_model


class TestProviderForModel:
    def test_anthropic_default(self):
        assert provider_for_model("claude-haiku-4-5-20251001") == "anthropic"

    def test_anthropic_opus(self):
        assert provider_for_model("claude-opus-4-7") == "anthropic"

    def test_openai_gpt(self):
        assert provider_for_model("gpt-4o") == "openai"

    def test_openai_gpt_mini(self):
        assert provider_for_model("gpt-4o-mini") == "openai"

    def test_openai_o1(self):
        assert provider_for_model("o1-preview") == "openai"

    def test_openai_o3(self):
        assert provider_for_model("o3-mini") == "openai"

    def test_ollama_prefix(self):
        assert provider_for_model("ollama:llama3") == "ollama"

    def test_ollama_custom_model(self):
        assert provider_for_model("ollama:mistral-7b") == "ollama"


class TestLLMError:
    def test_is_exception(self):
        err = LLMError("something went wrong")
        assert isinstance(err, Exception)
        assert "something went wrong" in str(err)


@pytest.mark.asyncio
async def test_complete_anthropic_missing_key_raises():
    """complete() with an Anthropic model raises LLMError when API key is absent."""
    from chaincheck.llm import complete

    with patch.dict("os.environ", {}, clear=True):
        import os

        os.environ.pop("ANTHROPIC_API_KEY", None)
        with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
            await complete("hello", "claude-haiku-4-5-20251001")


@pytest.mark.asyncio
async def test_complete_openai_missing_key_raises():
    """complete() with a gpt-* model raises LLMError when OPENAI_API_KEY is absent."""
    from chaincheck.llm import complete

    with patch.dict("os.environ", {}, clear=True):
        import os

        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(LLMError, match="OPENAI_API_KEY"):
            await complete("hello", "gpt-4o-mini")


@pytest.mark.asyncio
async def test_complete_anthropic_returns_text():
    """complete() with Anthropic model returns the response text."""
    from chaincheck.llm import complete

    mock_content = MagicMock()
    mock_content.text = "Paris is the capital of France."
    mock_msg = MagicMock()
    mock_msg.content = [mock_content]

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_msg)

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}),
        patch("anthropic.AsyncAnthropic", return_value=mock_client),
    ):
        result = await complete("What is the capital of France?", "claude-haiku-4-5-20251001")

    assert result == "Paris is the capital of France."


@pytest.mark.asyncio
async def test_complete_openai_returns_text():
    """complete() with gpt-* model returns the response text via OpenAI SDK."""
    from chaincheck.llm import complete

    mock_choice = MagicMock()
    mock_choice.message.content = "The sky is blue."
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    mock_openai = MagicMock()
    mock_openai.AsyncOpenAI.return_value = mock_client

    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        patch.dict(sys.modules, {"openai": mock_openai}),
    ):
        result = await complete("Why is the sky blue?", "gpt-4o-mini")

    assert result == "The sky is blue."


@pytest.mark.asyncio
async def test_complete_ollama_calls_local_endpoint():
    """complete() with ollama: prefix posts to the Ollama HTTP endpoint."""
    from chaincheck.llm import complete

    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "42 is the answer."}
    mock_response.raise_for_status = MagicMock()

    mock_httpx_client = MagicMock()
    mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
    mock_httpx_client.__aexit__ = AsyncMock(return_value=False)
    mock_httpx_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        result = await complete("What is the answer?", "ollama:llama3")

    assert result == "42 is the answer."
    call_kwargs = mock_httpx_client.post.call_args
    assert "api/generate" in call_kwargs[0][0]
