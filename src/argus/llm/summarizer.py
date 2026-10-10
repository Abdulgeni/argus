"""LLMSummarizer: batching, cache, retry, validation, and a never-fail template fallback."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from argus.core.config import LLMConfig
from argus.core.models import Finding, IncidentReport
from argus.llm.cache import DiskCache, cache_key, default_cache_dir
from argus.llm.client import ENV_KEYS, LLMClient, create_client
from argus.llm.context import IncidentContext
from argus.llm.prompt import build_prompt
from argus.llm.retry import LLMError, call_with_retries
from argus.llm.schema import InvalidResponse, LLMResponse, parse_response
from argus.llm.template import template_response

MAX_IN_FLIGHT = 5
_FATAL_STATUS = frozenset({401, 403, 404})
_RETRY_NOTE = (
    "\n\nYour previous reply was rejected ({reason}). "
    "Reply again with ONLY the JSON object described above."
)


@dataclass(frozen=True)
class SummaryResult:
    """Reports in input order, plus run-level findings."""

    reports: list[IncidentReport]
    findings: list[Finding] = field(default_factory=list)


@dataclass(frozen=True)
class _Outcome:
    response: LLMResponse
    llm_used: bool
    cached: bool = False
    warnings: tuple[str, ...] = ()


class LLMSummarizer:
    """Summarizes incidents with an LLM when possible, otherwise with the template."""

    def __init__(
        self,
        client: LLMClient | None,
        *,
        batch_size: int = 10,
        cache: DiskCache | None = None,
        namespace: str = "",
        disabled: Finding | None = None,
        max_in_flight: int = MAX_IN_FLIGHT,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rand: Callable[[], float] = random.random,
    ) -> None:
        self._client = client
        self._batch_size = max(1, batch_size)
        self._cache = cache
        self._namespace = namespace
        self._disabled = disabled
        self._max_in_flight = max(1, min(max_in_flight, MAX_IN_FLIGHT))
        self._sleep = sleep
        self._rand = rand
        self._fatal: str | None = None
        self._findings: dict[str, Finding] = {}

    def summarize(self, contexts: Sequence[IncidentContext]) -> SummaryResult:
        """Return one IncidentReport per context, in order."""
        if self._disabled is not None:
            self._findings[self._disabled.code] = self._disabled
        if self._client is None:
            outcomes = [_Outcome(template_response(c), llm_used=False) for c in contexts]
        else:
            outcomes = asyncio.run(self._run(contexts))
        reports = [
            self._report(i, c, o)
            for i, (c, o) in enumerate(zip(contexts, outcomes, strict=True), 1)
        ]
        return SummaryResult(reports=reports, findings=list(self._findings.values()))

    async def _run(self, contexts: Sequence[IncidentContext]) -> list[_Outcome]:
        semaphore = asyncio.Semaphore(self._max_in_flight)
        outcomes: list[_Outcome] = []
        for start in range(0, len(contexts), self._batch_size):
            batch = contexts[start : start + self._batch_size]
            outcomes.extend(await asyncio.gather(*(self._one(c, semaphore) for c in batch)))
        return outcomes

    async def _one(self, context: IncidentContext, semaphore: asyncio.Semaphore) -> _Outcome:
        prompt = build_prompt(context)
        key = cache_key(prompt, self._namespace)
        cached = self._read_cache(key)
        if cached is not None:
            return _Outcome(cached, llm_used=True, cached=True)
        fallback = template_response(context)
        if self._fatal is not None:
            return _Outcome(fallback, False, warnings=(f"LLM skipped: {self._fatal}",))
        try:
            response = await self._ask_validated(prompt, semaphore)
        except LLMError as exc:
            self._note_failure(exc)
            return _Outcome(fallback, False, warnings=(f"LLM call failed ({exc}); template used",))
        except InvalidResponse as exc:
            reason = f"LLM reply invalid after one retry ({exc}); template used"
            return _Outcome(fallback, False, warnings=(reason,))
        self._write_cache(key, response)
        return _Outcome(response, llm_used=True)

    async def _ask_validated(self, prompt: str, semaphore: asyncio.Semaphore) -> LLMResponse:
        """Call the model; on invalid JSON retry exactly once, then raise InvalidResponse."""
        text = await self._call(prompt, semaphore)
        try:
            return parse_response(text)
        except InvalidResponse as first:
            nudged = prompt + _RETRY_NOTE.format(reason=first)
            return parse_response(await self._call(nudged, semaphore))

    async def _call(self, prompt: str, semaphore: asyncio.Semaphore) -> str:
        client = self._client
        assert client is not None

        async def attempt() -> str:
            async with semaphore:
                return await asyncio.to_thread(client.complete, prompt)

        return await call_with_retries(attempt, sleep=self._sleep, rand=self._rand)

    def _read_cache(self, key: str) -> LLMResponse | None:
        if self._cache is None:
            return None
        payload = self._cache.get(key)
        if payload is None:
            return None
        try:
            return parse_response(payload)
        except InvalidResponse:
            return None

    def _write_cache(self, key: str, response: LLMResponse) -> None:
        if self._cache is None:
            return
        try:
            self._cache.put(key, response.model_dump_json())
        except OSError as exc:
            self._findings.setdefault(
                "cache_write_failed",
                Finding(
                    level="warning",
                    code="cache_write_failed",
                    message=f"could not write the response cache: {exc.strerror or exc}",
                ),
            )

    def _note_failure(self, exc: LLMError) -> None:
        if exc.status_code in _FATAL_STATUS and self._fatal is None:
            self._fatal = f"provider rejected the request (HTTP {exc.status_code})"
            self._findings["llm_unavailable"] = Finding(
                level="warning",
                code="llm_unavailable",
                message=f"{self._fatal}: check the API key and model name; "
                "remaining incidents use the template",
            )

    def _report(self, number: int, context: IncidentContext, outcome: _Outcome) -> IncidentReport:
        group, response = context.group, outcome.response
        return IncidentReport(
            id=f"INC-{number:04d}",
            severity=group.severity,
            entity_type=group.entity_type,
            entity=group.entity,
            window_start=context.window_start,
            window_end=context.window_end,
            anomaly_score=group.peak.anomaly_score,
            anomaly_count=len(group.members),
            features=context.features,
            top_deviations=context.deviations,
            sample_log_lines=context.log_lines,
            summary=response.summary,
            suspected_category=response.suspected_category,
            confidence=response.confidence,
            evidence=response.evidence,
            recommended_action=response.recommended_action,
            llm_used=outcome.llm_used,
            llm_cached=outcome.cached,
            llm_model=self._client.model if outcome.llm_used and self._client else None,
            warnings=list(outcome.warnings),
        )


def build_summarizer(
    config: LLMConfig,
    cache_dir: Path | None,
    env: Mapping[str, str],
    *,
    no_llm: bool = False,
    client: LLMClient | None = None,
) -> LLMSummarizer:
    """Create the summarizer for a run. ``client`` lets tests inject a fake provider."""
    if no_llm:
        note = Finding(level="info", code="llm_disabled", message="LLM disabled (--no-llm)")
        return LLMSummarizer(None, disabled=note)
    client = client or create_client(config, env)
    if client is None:
        key = ENV_KEYS[config.provider]
        note = Finding(
            level="info",
            code="llm_no_api_key",
            message=f"{key} is not set; using template summaries (llm_used=false)",
        )
        return LLMSummarizer(None, disabled=note)
    cache = DiskCache(default_cache_dir(cache_dir, env), config.cache_ttl_hours)
    namespace = f"{config.provider}:{client.model}:{config.temperature}:{config.max_tokens}"
    return LLMSummarizer(client, batch_size=config.batch_size, cache=cache, namespace=namespace)
