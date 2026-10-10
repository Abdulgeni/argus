from types import SimpleNamespace
from typing import Any

import pytest

from argus.core.config import LLMConfig
from argus.llm.client import (
    AnthropicClient,
    OpenAIClient,
    create_client,
    to_llm_error,
)
from argus.llm.retry import LLMError


class FakeHTTPError(Exception):
    def __init__(self, status_code: int, retry_after: str | None = None) -> None:
        super().__init__(f"HTTP {status_code} key sk-ABCDEFGHIJKLMNOPQRSTUV")
        self.status_code = status_code
        headers = {"retry-after": retry_after} if retry_after else {}
        self.response = SimpleNamespace(headers=headers)


class APITimeoutError(Exception):
    pass


def test_error_mapping() -> None:
    err = to_llm_error(FakeHTTPError(429, "7"))
    assert (err.status_code, err.retry_after, err.retryable) == (429, 7.0, True)
    assert to_llm_error(FakeHTTPError(503)).retryable
    assert not to_llm_error(FakeHTTPError(401)).retryable
    assert not to_llm_error(FakeHTTPError(400)).retryable
    assert to_llm_error(APITimeoutError("slow")).retryable
    assert not to_llm_error(ValueError("bug")).retryable
    assert "ABCDEFGHIJKLMNOPQRSTUV" not in str(to_llm_error(FakeHTTPError(500)))
    assert to_llm_error(FakeHTTPError(429, "Wed, 21 Oct 2026")).retry_after is None


def test_anthropic_client_request_and_reply() -> None:
    seen: dict[str, Any] = {}

    def create(**kwargs: Any) -> Any:
        seen.update(kwargs)
        block = SimpleNamespace(type="text", text='{"a": 1}')
        return SimpleNamespace(content=[SimpleNamespace(type="thinking"), block])

    sdk = SimpleNamespace(messages=SimpleNamespace(create=create))
    client = AnthropicClient("k", "m", 321, 0.0, 30.0, sdk=sdk)
    assert client.complete("hello") == '{"a": 1}'
    assert seen["model"] == "m" and seen["max_tokens"] == 321 and seen["temperature"] == 0.0
    assert seen["messages"] == [{"role": "user", "content": "hello"}]


def test_anthropic_client_wraps_errors_and_empty_replies() -> None:
    def boom(**_: Any) -> Any:
        raise FakeHTTPError(429, "3")

    client = AnthropicClient(
        "k", "m", 10, 0.0, 5.0, sdk=SimpleNamespace(messages=SimpleNamespace(create=boom))
    )
    with pytest.raises(LLMError) as info:
        client.complete("x")
    assert info.value.retryable and info.value.retry_after == 3.0

    empty = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **_: SimpleNamespace(content=[]))
    )
    with pytest.raises(LLMError, match="empty"):
        AnthropicClient("k", "m", 10, 0.0, 5.0, sdk=empty).complete("x")


def test_openai_client_request_and_reply() -> None:
    seen: dict[str, Any] = {}

    def create(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = OpenAIClient("k", "gpt-x", 99, 0.2, 30.0, sdk=sdk)
    assert client.complete("hi") == "{}"
    assert seen["model"] == "gpt-x" and seen["max_completion_tokens"] == 99
    assert seen["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in seen["messages"]] == ["system", "user"]


def test_create_client_requires_env_key() -> None:
    assert create_client(LLMConfig(), {}) is None
    assert create_client(LLMConfig(), {"ANTHROPIC_API_KEY": "  "}) is None
    assert create_client(LLMConfig(provider="openai"), {"ANTHROPIC_API_KEY": "k"}) is None


def test_create_client_builds_real_sdk_clients_when_installed() -> None:
    pytest.importorskip("anthropic")
    pytest.importorskip("openai")
    a = create_client(LLMConfig(model="claude-x"), {"ANTHROPIC_API_KEY": "k"})
    o = create_client(LLMConfig(provider="openai"), {"OPENAI_API_KEY": "k"})
    assert a is not None and a.model == "claude-x"
    assert o is not None and o.model
