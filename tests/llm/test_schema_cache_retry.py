import asyncio
import json
from pathlib import Path

import pytest

from argus.llm.cache import DiskCache, cache_key, default_cache_dir
from argus.llm.retry import (
    MAX_DELAY_S,
    MAX_RETRY_AFTER_S,
    LLMError,
    backoff_delay,
    call_with_retries,
)
from argus.llm.schema import InvalidResponse, parse_response

VALID = {
    "summary": "Burst of failed logins.",
    "suspected_category": "brute_force",
    "confidence": 0.8,
    "evidence": ["error_rate: 0.95"],
    "recommended_action": "Block the source.",
}


def test_parse_plain_and_fenced_json() -> None:
    assert parse_response(json.dumps(VALID)).suspected_category == "brute_force"
    fenced = "Here you go:\n```json\n" + json.dumps(VALID) + "\n```"
    assert parse_response(fenced).confidence == 0.8


def test_extra_fields_ignored() -> None:
    assert parse_response(json.dumps({**VALID, "bonus": 1})).summary.startswith("Burst")


@pytest.mark.parametrize(
    "text",
    [
        "not json at all",
        "[1, 2]",
        "{broken",
        json.dumps({**VALID, "suspected_category": "malware"}),
        json.dumps({**VALID, "confidence": 1.5}),
        json.dumps({**VALID, "summary": ""}),
        json.dumps({k: v for k, v in VALID.items() if k != "evidence"}),
    ],
)
def test_invalid_replies_rejected(text: str) -> None:
    with pytest.raises(InvalidResponse):
        parse_response(text)


def test_cache_roundtrip_and_ttl(tmp_path: Path) -> None:
    now = [1000.0]
    cache = DiskCache(tmp_path, ttl_hours=1.0, clock=lambda: now[0])
    key = cache_key("prompt")
    assert cache.get(key) is None
    cache.put(key, "payload")
    assert cache.get(key) == "payload"
    now[0] += 3599
    assert cache.get(key) == "payload"
    now[0] += 2
    assert cache.get(key) is None


def test_cache_corrupt_entry_is_a_miss(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    key = cache_key("p")
    cache.put(key, "ok")
    (tmp_path / key[:2] / f"{key}.json").write_text("{not json", encoding="utf-8")
    assert cache.get(key) is None


def test_cache_key_is_sha256_and_namespaced() -> None:
    assert len(cache_key("p")) == 64
    assert cache_key("p") == cache_key("p")
    assert cache_key("p", "anthropic:a") != cache_key("p", "openai:b")
    assert "secret" not in cache_key("secret")


def test_cache_unwritable_raises_oserror(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(OSError):
        DiskCache(blocker / "sub").put(cache_key("p"), "x")


def test_default_cache_dir_precedence(tmp_path: Path) -> None:
    home = tmp_path / "home"
    assert default_cache_dir(None, {}, "linux", home) == home / ".cache" / "argus"
    env = {"ARGUS_CACHE_DIR": "/env/dir", "LOCALAPPDATA": "C:/L"}
    assert default_cache_dir(None, env, "linux", home) == Path("/env/dir")
    assert default_cache_dir(Path("/cfg"), env, "linux", home) == Path("/cfg")
    assert default_cache_dir(None, {"LOCALAPPDATA": "C:/L"}, "win32", home) == Path("C:/L/argus")


def test_backoff_bounds_and_retry_after() -> None:
    assert backoff_delay(0, None, lambda: 0.0) == pytest.approx(0.5)
    assert backoff_delay(0, None, lambda: 1.0) == pytest.approx(1.0)
    assert backoff_delay(2, None, lambda: 1.0) == pytest.approx(4.0)
    assert backoff_delay(20, None, lambda: 1.0) == MAX_DELAY_S
    assert backoff_delay(0, 7.0, lambda: 0.0) == 7.0
    assert backoff_delay(0, 9999.0, lambda: 0.0) == MAX_RETRY_AFTER_S


def _flaky(failures: list[LLMError]):
    state = {"calls": 0}

    async def call() -> str:
        state["calls"] += 1
        if failures:
            raise failures.pop(0)
        return "ok"

    return call, state


def test_retries_then_succeeds_and_sleeps_with_backoff() -> None:
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    call, state = _flaky(
        [LLMError("429", retryable=True, retry_after=7.0), LLMError("503", retryable=True)]
    )
    result = asyncio.run(call_with_retries(call, sleep=fake_sleep, rand=lambda: 0.0))
    assert result == "ok" and state["calls"] == 3
    assert delays[0] == 7.0
    assert delays[1] == pytest.approx(1.0)


def test_non_retryable_raises_immediately() -> None:
    call, state = _flaky([LLMError("401", status_code=401, retryable=False)])

    async def no_sleep(_: float) -> None:
        raise AssertionError("must not sleep")

    with pytest.raises(LLMError):
        asyncio.run(call_with_retries(call, sleep=no_sleep))
    assert state["calls"] == 1


def test_gives_up_after_max_attempts() -> None:
    async def no_sleep(_: float) -> None:
        return None

    call, state = _flaky([LLMError("500", retryable=True) for _ in range(10)])
    with pytest.raises(LLMError):
        asyncio.run(call_with_retries(call, max_attempts=3, sleep=no_sleep))
    assert state["calls"] == 3
