"""Helpers shared by all reporters."""

from __future__ import annotations

import re
from collections import Counter

from argus.core.models import IncidentReport, Severity

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")

SEVERITY_STYLES: dict[Severity, str] = {
    Severity.CRITICAL: "bold white on red",
    Severity.HIGH: "bold red",
    Severity.MEDIUM: "yellow",
    Severity.LOW: "cyan",
    Severity.INFO: "dim",
}


def one_line(text: str, limit: int = 90) -> str:
    """First sentence of ``text`` on one line, truncated with an ellipsis."""
    flat = " ".join(text.split())
    first = _SENTENCE_END.split(flat, maxsplit=1)[0]
    return first if len(first) <= limit else first[: limit - 1].rstrip() + "…"


def severity_counts(incidents: list[IncidentReport]) -> dict[Severity, int]:
    """Incident count per severity, highest first, omitting zero counts."""
    counts = Counter(i.severity for i in incidents)
    ordered = sorted(counts, key=lambda s: -s.rank)
    return {s: counts[s] for s in ordered}


def period(incident: IncidentReport) -> str:
    """``2026-10-07 10:05:00 -> 10:06:12 UTC`` (end date repeated only if it differs)."""
    start, end = incident.window_start, incident.window_end
    end_fmt = "%H:%M:%S" if start.date() == end.date() else "%Y-%m-%d %H:%M:%S"
    return f"{start:%Y-%m-%d %H:%M:%S} -> {end:{end_fmt}} UTC"
