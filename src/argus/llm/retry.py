"""Exponential backoff with jitter; honors Retry-After."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")

MAX_ATTEMPTS = 5
BASE_DELAY_S = 1.0
MAX_DELAY_S = 30.0
MAX_RETRY_AFTER_S = 120.0


class LLMError(Exception):
    """A failed provider call; ``retryable`` is True for 429, 5xx, timeouts, connection errors."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.retryable = retryable


def backoff_delay(
    attempt: int, retry_after: float | None, rand: Callable[[], float] = random.random
) -> float:
    """Delay before retry number ``attempt`` (0-based)."""
    ceiling = min(MAX_DELAY_S, BASE_DELAY_S * float(2**attempt))
    delay = ceiling * (0.5 + 0.5 * rand())
    if retry_after is not None:
        delay = float(max(delay, min(retry_after, MAX_RETRY_AFTER_S)))
    return delay


async def call_with_retries(
    call: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = MAX_ATTEMPTS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rand: Callable[[], float] = random.random,
) -> T:
    """Run ``call``; retry retryable LLMErrors, re-raising the last error when attempts run out."""
    attempt = 0
    while True:
        try:
            return await call()
        except LLMError as exc:
            attempt += 1
            if not exc.retryable or attempt >= max_attempts:
                raise
            await sleep(backoff_delay(attempt - 1, exc.retry_after, rand))
