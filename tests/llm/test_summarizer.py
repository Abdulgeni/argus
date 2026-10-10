import json
import threading
import time
from pathlib import Path

from argus.core.config import LLMConfig
from argus.core.models import Finding
from argus.llm import LLMSummarizer, build_summarizer
from argus.llm.cache import DiskCache
from argus.llm.retry import LLMError

from .helpers import attacker_context

GOOD = {
    "summary": "Burst of failed logins from one IP.",
    "suspected_category": "brute_force",
    "confidence": 0.9,
    "evidence": ["failed_auth_count: 60"],
    "recommended_action": "Block the source IP.",
}


class FakeClient:
    """Scripted provider: each item is a reply string or an exception to raise."""

    model = "fake-model"

    def __init__(self, script: list[str | Exception] | None = None, delay: float = 0.0) -> None:
        self.script = script
        self.delay = delay
        self.prompts: list[str] = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def complete(self, prompt: str) -> str:
        with self._lock:
            self.prompts.append(prompt)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.script is None:
                return json.dumps(GOOD)
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        finally:
            with self._lock:
                self.active -= 1


async def _instant(_: float) -> None:
    return None


def _summarizer(client: FakeClient | None, **kwargs: object) -> LLMSummarizer:
    return LLMSummarizer(client, sleep=_instant, rand=lambda: 0.0, **kwargs)


def test_llm_success_populates_report() -> None:
    client = FakeClient()
    result = _summarizer(client).summarize([attacker_context()])
    report = result.reports[0]
    assert report.id == "INC-0001"
    assert report.llm_used and not report.llm_cached and report.llm_model == "fake-model"
    assert report.summary == GOOD["summary"] and report.suspected_category == "brute_force"
    assert report.confidence == 0.9 and report.warnings == []
    assert report.entity == "9.9.9.9" and report.sample_log_lines
    assert result.findings == []
    assert "hunter2" not in client.prompts[0]


def test_no_client_uses_template() -> None:
    result = _summarizer(None).summarize([attacker_context()])
    report = result.reports[0]
    assert not report.llm_used and report.llm_model is None
    assert report.suspected_category == "unknown" and report.confidence == 0.0


def test_build_summarizer_no_llm_and_missing_key(tmp_path: Path) -> None:
    ctx = attacker_context()
    off = build_summarizer(LLMConfig(), tmp_path, {}, no_llm=True).summarize([ctx])
    assert [f.code for f in off.findings] == ["llm_disabled"] and not off.reports[0].llm_used
    nokey = build_summarizer(LLMConfig(), tmp_path, {}).summarize([ctx])
    assert [f.code for f in nokey.findings] == ["llm_no_api_key"]
    assert "ANTHROPIC_API_KEY" in nokey.findings[0].message
    assert not nokey.reports[0].llm_used


def test_build_summarizer_with_injected_client(tmp_path: Path) -> None:
    client = FakeClient()
    summarizer = build_summarizer(LLMConfig(), tmp_path, {}, client=client)
    assert summarizer.summarize([attacker_context()]).reports[0].llm_used


def test_invalid_json_retried_once_then_ok() -> None:
    client = FakeClient(["sorry, here is prose", json.dumps(GOOD)])
    report = _summarizer(client).summarize([attacker_context()]).reports[0]
    assert report.llm_used and len(client.prompts) == 2
    assert "previous reply was rejected" in client.prompts[1]


def test_invalid_twice_falls_back_with_warning() -> None:
    client = FakeClient(["nope", '{"summary": "x"}'])
    report = _summarizer(client).summarize([attacker_context()]).reports[0]
    assert not report.llm_used and len(client.prompts) == 2
    assert report.suspected_category == "unknown"
    assert "invalid after one retry" in report.warnings[0]


def test_transient_errors_are_retried_with_retry_after() -> None:
    delays: list[float] = []

    async def record(seconds: float) -> None:
        delays.append(seconds)

    client = FakeClient(
        [LLMError("429", status_code=429, retry_after=7.0, retryable=True), json.dumps(GOOD)]
    )
    summarizer = LLMSummarizer(client, sleep=record, rand=lambda: 0.0)
    assert summarizer.summarize([attacker_context()]).reports[0].llm_used
    assert delays == [7.0]


def test_exhausted_retries_fall_back() -> None:
    client = FakeClient([LLMError("503", status_code=503, retryable=True) for _ in range(10)])
    report = _summarizer(client).summarize([attacker_context()]).reports[0]
    assert not report.llm_used and "LLM call failed" in report.warnings[0]
    assert len(client.prompts) == 5


def test_auth_failure_disables_remaining_calls() -> None:
    client = FakeClient([LLMError("bad key", status_code=401, retryable=False)])
    ctx = attacker_context()
    result = _summarizer(client, batch_size=1).summarize([ctx, ctx, ctx])
    assert len(client.prompts) == 1
    assert [r.llm_used for r in result.reports] == [False, False, False]
    assert "LLM skipped" in result.reports[1].warnings[0]
    assert [f.code for f in result.findings] == ["llm_unavailable"]


def test_cache_hit_avoids_provider_call(tmp_path: Path) -> None:
    ctx = attacker_context()
    cache = DiskCache(tmp_path / "cache")
    first = _summarizer(FakeClient(), cache=cache, namespace="ns").summarize([ctx]).reports[0]
    assert first.llm_used and not first.llm_cached
    silent = FakeClient([LLMError("must not be called")])
    second = _summarizer(silent, cache=cache, namespace="ns").summarize([ctx]).reports[0]
    assert second.llm_used and second.llm_cached and silent.prompts == []
    assert second.summary == first.summary
    other_model = FakeClient()
    _summarizer(other_model, cache=cache, namespace="other").summarize([ctx])
    assert len(other_model.prompts) == 1


def test_cache_write_failure_is_reported_not_fatal(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    result = _summarizer(FakeClient(), cache=DiskCache(blocker / "sub")).summarize(
        [attacker_context()]
    )
    assert result.reports[0].llm_used
    assert [f.code for f in result.findings] == ["cache_write_failed"]
    assert all(isinstance(f, Finding) for f in result.findings)


def test_concurrency_is_bounded_and_order_preserved() -> None:
    client = FakeClient(delay=0.03)
    ctx = attacker_context()
    result = _summarizer(client, batch_size=12).summarize([ctx] * 12)
    assert len(client.prompts) == 12
    assert 1 < client.max_active <= 5
    assert [r.id for r in result.reports] == [f"INC-{i:04d}" for i in range(1, 13)]


def test_batches_are_processed_in_order() -> None:
    client = FakeClient()
    result = _summarizer(client, batch_size=2).summarize([attacker_context()] * 5)
    assert len(result.reports) == 5 and len(client.prompts) == 5
