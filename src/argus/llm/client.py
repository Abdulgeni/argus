"""Provider clients (Anthropic, OpenAI). SDKs are imported lazily; keys come only from env vars."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from argus.core.config import LLMConfig
from argus.llm.retry import LLMError
from argus.llm.scrub import scrub

SYSTEM_PROMPT = "Reply with a single JSON object and nothing else."
DEFAULT_MODELS: dict[str, str] = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini"}
ENV_KEYS: dict[str, str] = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
_RETRYABLE_STATUS = frozenset({408, 409, 429})


class LLMClient(Protocol):
    """Minimal synchronous interface the summarizer needs."""

    model: str

    def complete(self, prompt: str) -> str:
        """Return the model's raw text reply or raise LLMError."""
        ...


def to_llm_error(exc: Exception) -> LLMError:
    """Translate an SDK exception into LLMError using duck typing (no SDK import needed)."""
    raw_status = getattr(exc, "status_code", None)
    status = raw_status if isinstance(raw_status, int) else None
    retry_after: float | None = None
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is not None:
        try:
            retry_after = float(headers.get("retry-after"))
        except (TypeError, ValueError):
            retry_after = None
    name = type(exc).__name__
    transient = status is None and ("Timeout" in name or "Connection" in name)
    retryable = transient or status in _RETRYABLE_STATUS or (status is not None and status >= 500)
    detail = scrub(str(exc))[:200]
    return LLMError(
        f"{name}: {detail}", status_code=status, retry_after=retry_after, retryable=retryable
    )


class AnthropicClient:
    """Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int,
        temperature: float,
        timeout_s: float,
        sdk: Any = None,
    ) -> None:
        self.model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        if sdk is None:
            import anthropic

            sdk = anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=0)
        self._sdk = sdk

    def complete(self, prompt: str) -> str:
        try:
            response = self._sdk.messages.create(
                model=self.model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise to_llm_error(exc) from exc
        text = "".join(
            getattr(block, "text", "")
            for block in response.content
            if getattr(block, "type", "") == "text"
        )
        if not text.strip():
            raise LLMError("empty response from Anthropic", retryable=False)
        return text


class OpenAIClient:
    """OpenAI Chat Completions API (JSON mode)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int,
        temperature: float,
        timeout_s: float,
        sdk: Any = None,
    ) -> None:
        self.model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        if sdk is None:
            import openai

            sdk = openai.OpenAI(api_key=api_key, timeout=timeout_s, max_retries=0)
        self._sdk = sdk

    def complete(self, prompt: str) -> str:
        try:
            response = self._sdk.chat.completions.create(
                model=self.model,
                max_completion_tokens=self._max_tokens,
                temperature=self._temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:
            raise to_llm_error(exc) from exc
        text: str = response.choices[0].message.content or ""
        if not text.strip():
            raise LLMError("empty response from OpenAI", retryable=False)
        return text


def create_client(config: LLMConfig, env: Mapping[str, str]) -> LLMClient | None:
    """Build the configured client, or None when its API-key env var is unset/empty."""
    api_key = env.get(ENV_KEYS[config.provider], "").strip()
    if not api_key:
        return None
    model = config.model or DEFAULT_MODELS[config.provider]
    cls = AnthropicClient if config.provider == "anthropic" else OpenAIClient
    return cls(api_key, model, config.max_tokens, config.temperature, config.timeout_s)
