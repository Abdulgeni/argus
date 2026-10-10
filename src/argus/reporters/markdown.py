"""Markdown reporter: a ticket-ready document, executive summary first."""

from __future__ import annotations

import re

from argus.core.models import FeatureDeviation, IncidentReport, RunReport
from argus.llm.prompt import format_number
from argus.reporters.common import period, severity_counts

MAX_LOG_LINES = 10


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _fence(lines: list[str]) -> str:
    """A code fence longer than any backtick run inside the lines."""
    longest = max((len(m) for line in lines for m in re.findall(r"`+", line)), default=0)
    return "`" * max(3, longest + 1)


def _executive_summary(run: RunReport) -> list[str]:
    out = ["# Argus incident report", "", "## Executive summary", ""]
    stats = run.stats
    out.append(
        f"Scanned `{run.source}` ({run.log_format} format): {stats.parsed} events parsed, "
        f"{stats.failed} malformed, {run.feature_rows} feature rows, "
        f"{run.anomalous_rows} flagged as unusual."
    )
    out.append("")
    counts = severity_counts(run.incidents)
    if not counts:
        out += ["**No incidents.** Nothing was flagged as unusual.", ""]
        return out
    summary = ", ".join(f"{n} {s.value}" for s, n in counts.items())
    out += [f"**{len(run.incidents)} incident(s):** {summary}.", ""]
    out += ["| ID | Severity | Entity | Category | Summary |", "|---|---|---|---|---|"]
    for i in run.incidents:
        out.append(
            f"| {i.id} | {i.severity.value} | {_cell(i.entity_type + ': ' + i.entity)} | "
            f"{i.suspected_category} | {_cell(i.summary)} |"
        )
    out += [
        "",
        "_Severity is a relative ranking within this run, not an absolute risk rating. "
        "Detection flags unusual behavior; it does not establish malicious intent._",
        "",
    ]
    return out


def _warnings(run: RunReport) -> list[str]:
    if not run.findings:
        return []
    out = ["## Warnings", ""]
    out += [f"- **{f.level.upper()}** `{f.code}`: {_cell(f.message)}" for f in run.findings]
    return [*out, ""]


def _deviation_table(devs: list[FeatureDeviation]) -> list[str]:
    out = ["| Feature | Window | Value | Baseline |", "|---|---|---|---|"]
    for d in devs:
        out.append(
            f"| {d.feature} | {d.window or '-'} | {format_number(d.value)} | "
            f"{format_number(d.baseline)} |"
        )
    return [*out, ""]


def _incident(i: IncidentReport) -> list[str]:
    provenance = (
        f"LLM summary ({i.llm_model}{', cached' if i.llm_cached else ''})"
        if i.llm_used
        else "Template summary (LLM not used)"
    )
    out = [
        f"## {i.id}: {i.severity.value} - {i.entity_type} `{i.entity}`",
        "",
        f"- **Period:** {period(i)}",
        f"- **Anomalous windows:** {i.anomaly_count} (detector score {i.anomaly_score:.4f})",
        f"- **Suspected category:** {i.suspected_category} (confidence {i.confidence:.2f})",
        f"- **Source of summary:** {provenance}",
        "",
        i.summary,
        "",
        "### Evidence",
        "",
    ]
    out += [f"- {e}" for e in i.evidence] or ["- (none)"]
    out += ["", "### Recommended action", "", i.recommended_action, ""]
    if i.top_deviations:
        out += ["### Largest feature deviations", "", *_deviation_table(i.top_deviations)]
    if i.warnings:
        out += ["### Warnings", "", *[f"- {w}" for w in i.warnings], ""]
    shown = i.sample_log_lines[:MAX_LOG_LINES]
    if shown:
        fence = _fence(shown)
        label = f"Sample log lines (newest first, {len(shown)} of {len(i.sample_log_lines)} shown)"
        out += [f"### {label}", "", fence, *shown, fence, ""]
    return out


class MarkdownReporter:
    """Single .md document for a ticket."""

    def render(self, run: RunReport) -> str:
        parts = _executive_summary(run) + _warnings(run)
        for incident in run.incidents:
            parts += _incident(incident)
        return "\n".join(parts).rstrip() + "\n"
