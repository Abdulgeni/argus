import json

import pytest

from argus.core.errors import ConfigError
from argus.core.models import Finding, ParseStats, RunReport, Severity
from argus.llm import LLMSummarizer
from argus.reporters import (
    JsonReporter,
    MarkdownReporter,
    TableReporter,
    get_reporter,
)
from argus.reporters.common import one_line

from ..llm.helpers import attacker_context


def _run(findings: list[Finding] | None = None, empty: bool = False) -> RunReport:
    reports = [] if empty else LLMSummarizer(None).summarize([attacker_context()]).reports
    if reports:
        llm = reports[0].model_copy(
            update={
                "id": "INC-0002",
                "severity": Severity.MEDIUM,
                "llm_used": True,
                "llm_model": "fake-model",
                "summary": "Pipes | in text. Second sentence is dropped in tables.",
                "suspected_category": "brute_force",
                "confidence": 0.85,
                "sample_log_lines": ["line with ``` fence", "plain"],
            }
        )
        reports = [reports[0], llm]
    return RunReport(
        source="access.log",
        log_format="nginx",
        stats=ParseStats(parsed=1200, failed=3),
        feature_rows=400,
        anomalous_rows=20,
        incidents=reports,
        findings=findings or [],
    )


WARN = Finding(level="warning", code="naive_timestamps", message="5 timestamps assumed UTC")


def test_one_line() -> None:
    assert one_line("First sentence. Second one.") == "First sentence."
    assert one_line("x" * 200, 20) == "x" * 19 + "…"
    assert one_line("multi\nline  text") == "multi line text"


def test_table_content() -> None:
    out = TableReporter(width=160, color=False).render(_run([WARN]))
    for needle in ("ID", "Severity", "Entity", "Time window", "Category", "Summary"):
        assert needle in out
    assert "INC-0001" in out and "INC-0002" in out
    assert "source_ip: 9.9.9.9" in out and "brute_force" in out
    assert "Second sentence" not in out
    assert "[WARNING] 5 timestamps assumed UTC" in out
    assert "\x1b[" not in out
    assert "1200 events" in out and "2 incident(s)" in out


def test_table_color_coding() -> None:
    out = TableReporter(width=160, color=True).render(_run())
    assert "\x1b[" in out


def test_table_empty() -> None:
    out = TableReporter(color=False).render(_run(empty=True))
    assert "No anomalies were flagged." in out


def test_json_roundtrip_and_fields() -> None:
    run = _run([WARN])
    text = JsonReporter().render(run)
    assert text == JsonReporter().render(run)
    data = json.loads(text)
    assert RunReport.model_validate(data) == run
    incident = data["incidents"][1]
    for key in ("features", "llm_used", "evidence", "warnings", "suspected_category"):
        assert key in incident
    assert incident["llm_used"] is True and data["findings"][0]["code"] == "naive_timestamps"


def test_markdown_structure() -> None:
    md = MarkdownReporter().render(_run([WARN]))
    assert md.startswith("# Argus incident report")
    summary_at, first_incident = md.index("## Executive summary"), md.index("## INC-0001")
    assert summary_at < md.index("| ID | Severity") < first_incident
    for heading in ("### Evidence", "### Recommended action", "### Sample log lines"):
        assert heading in md
    assert "Template summary (LLM not used)" in md
    assert "LLM summary (fake-model)" in md
    assert "Pipes \\| in text" in md
    assert "## Warnings" in md and "`naive_timestamps`" in md
    assert "relative ranking" in md
    assert md.endswith("\n") and not md.endswith("\n\n")


def test_markdown_fence_longer_than_backticks_in_logs() -> None:
    md = MarkdownReporter().render(_run())
    assert "````\nline with ``` fence\nplain\n````" in md


def test_markdown_empty() -> None:
    md = MarkdownReporter().render(_run(empty=True))
    assert "No incidents." in md and "## INC-" not in md


def test_get_reporter() -> None:
    assert isinstance(get_reporter("json"), JsonReporter)
    assert isinstance(get_reporter("markdown"), MarkdownReporter)
    assert isinstance(get_reporter("table", color=False), TableReporter)
    with pytest.raises(ConfigError, match="unknown output format"):
        get_reporter("xml")
