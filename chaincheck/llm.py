"""Small async LLM client helpers used by ChainCheck."""

from __future__ import annotations

import os
from typing import Any

import httpx


class LLMError(RuntimeError):
    """Raised when an LLM provider call fails."""


def provider_for_model(model: str) -> str:
    """Infer the provider for a configured model name."""
    if model.startswith("gpt-"):
        return "openai"
    if model.startswith("ollama:"):
        return "ollama"
    return "anthropic"


async def complete(prompt: str, model: str, max_tokens: int = 1000) -> str:
    """Call the configured LLM provider and return text."""
    provider = provider_for_model(model)
    if provider == "openai":
        return await _openai_complete(prompt, model, max_tokens)
    if provider == "ollama":
        return await _ollama_complete(prompt, model.removeprefix("ollama:"), max_tokens)
    return await _anthropic_complete(prompt, model, max_tokens)


async def _anthropic_complete(prompt: str, model: str, max_tokens: int) -> str:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise LLMError("ANTHROPIC_API_KEY is not configured")
    payload: dict[str, Any] = {
        "model": _anthropic_model_alias(model),
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages", json=payload, headers=headers
            )
    except httpx.RequestError as exc:
        raise LLMError(str(exc)) from exc
    _raise_for_status(response)
    content = response.json().get("content", [])
    return "\n".join(block.get("text", "") for block in content if block.get("type") == "text")


async def _openai_complete(prompt: str, model: str, max_tokens: int) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise LLMError("OPENAI_API_KEY is not configured")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {key}"}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions", json=payload, headers=headers
            )
    except httpx.RequestError as exc:
        raise LLMError(str(exc)) from exc
    _raise_for_status(response)
    return str(response.json()["choices"][0]["message"]["content"])


async def _ollama_complete(prompt: str, model: str, max_tokens: int) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(f"{base_url}/api/generate", json=payload)
    except httpx.RequestError as exc:
        raise LLMError(str(exc)) from exc
    _raise_for_status(response)
    return str(response.json().get("response", ""))


def _anthropic_model_alias(model: str) -> str:
    aliases = {
        "claude-haiku-4-5": "claude-3-5-haiku-latest",
        "claude-haiku": "claude-3-5-haiku-latest",
    }
    return aliases.get(model, model)


def _raise_for_status(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise LLMError(str(exc)) from exc
