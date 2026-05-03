"""
Multi-provider LLM routing — Anthropic, OpenAI, Ollama.

Route by model prefix:
  gpt-* / o1* / o3*  → OpenAI Chat Completions
  ollama:<name>       → local Ollama generate endpoint
  anything else       → Anthropic Messages

Usage:
    from chaincheck.llm import complete, provider_for_model
    text = await complete("Tell me about X.", model="claude-haiku-4-5-20251001")
"""

from __future__ import annotations

import os

_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


class LLMError(Exception):
    """Raised when a provider call fails or a required API key is missing."""


def provider_for_model(model: str) -> str:
    """Return the provider name for a given model identifier."""
    if model.startswith(("gpt-", "o1", "o3", "text-")):
        return "openai"
    if model.startswith("ollama:"):
        return "ollama"
    return "anthropic"


async def complete(
    prompt: str,
    model: str,
    max_tokens: int = 512,
) -> str:
    """
    Send a user prompt to the appropriate provider and return the text response.

    Args:
        prompt: User message to send.
        model: Model identifier. Prefix with 'ollama:' for local Ollama models.
        max_tokens: Maximum tokens to generate.

    Returns:
        Generated text string.

    Raises:
        LLMError: If the required API key is absent or the provider returns an error.
    """
    provider = provider_for_model(model)
    if provider == "openai":
        return await _openai_complete(prompt, model, max_tokens)
    if provider == "ollama":
        return await _ollama_complete(prompt, model.removeprefix("ollama:"), max_tokens)
    return await _anthropic_complete(prompt, model, max_tokens)


async def _anthropic_complete(prompt: str, model: str, max_tokens: int) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMError("ANTHROPIC_API_KEY not set")
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)
    msg = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


async def _openai_complete(prompt: str, model: str, max_tokens: int) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("OPENAI_API_KEY not set")
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    resp = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


async def _ollama_complete(prompt: str, model: str, max_tokens: int) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{_OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
