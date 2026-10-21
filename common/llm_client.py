"""Thin, provider-agnostic LLM client plus an offline stub.

Why a stub? Silent failures are statistical in real models, which makes them awkward to demo
and impossible to unit-test. Most catalog entries therefore ship a small handler that *models*
the failure deterministically, so the detector and mitigation code can be exercised offline.
Pass --live to any supporting script to run the same prompt against a real model.

Environment variables (see .env.example):
    RAG_CATALOG_PROVIDER   openai | anthropic   (auto-detected from available keys if unset)
    OPENAI_API_KEY, OPENAI_MODEL
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Protocol

try:  # convenience only; the catalog works without python-dotenv
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class LLMClient(Protocol):
    name: str

    def complete(
        self, user: str, system: str = "", temperature: float = 0.0, max_tokens: int = 512
    ) -> str: ...


# handler(system, user, temperature) -> text
Handler = Callable[[str, str, float], str]


class OfflineStub:
    """Deterministic stand-in. The handler is written by each example to model one failure."""

    def __init__(self, handler: Handler) -> None:
        self._handler = handler
        self.name = "offline-stub"
        self.calls = 0

    def complete(
        self, user: str, system: str = "", temperature: float = 0.0, max_tokens: int = 512
    ) -> str:
        self.calls += 1
        return self._handler(system, user, temperature)


class OpenAIClient:
    def __init__(self, model: Optional[str] = None) -> None:
        from openai import OpenAI  # imported lazily so offline runs need no SDK

        self._client = OpenAI()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.name = f"openai:{self.model}"

    def complete(
        self, user: str, system: str = "", temperature: float = 0.0, max_tokens: int = 512
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        return resp.choices[0].message.content or ""


class AnthropicClient:
    def __init__(self, model: Optional[str] = None) -> None:
        import anthropic  # imported lazily so offline runs need no SDK

        self._client = anthropic.Anthropic()
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
        self.name = f"anthropic:{self.model}"

    def complete(
        self, user: str, system: str = "", temperature: float = 0.0, max_tokens: int = 512
    ) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if block.type == "text")


def get_client(
    offline_handler: Optional[Handler] = None, live: bool = False, provider: Optional[str] = None
) -> LLMClient:
    """Return the offline stub (default) or a live provider client (live=True)."""
    if not live:
        if offline_handler is None:
            raise ValueError("offline mode needs an offline_handler; pass --live to use a real model")
        return OfflineStub(offline_handler)

    provider = provider or os.getenv("RAG_CATALOG_PROVIDER")
    if not provider:
        if os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise RuntimeError(
                "--live needs OPENAI_API_KEY or ANTHROPIC_API_KEY (copy .env.example to .env)."
            )
    if provider == "openai":
        return OpenAIClient()
    if provider == "anthropic":
        return AnthropicClient()
    raise ValueError(f"unknown provider {provider!r}; expected 'openai' or 'anthropic'")
