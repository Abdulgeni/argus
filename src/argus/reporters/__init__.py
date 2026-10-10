"""Reporters: RunReport -> text (table, JSON, Markdown)."""

from __future__ import annotations

from typing import Protocol

from argus.core.errors import ConfigError
from argus.core.models import RunReport
from argus.reporters.json_reporter import JsonReporter
from argus.reporters.markdown import MarkdownReporter
from argus.reporters.table import TableReporter


class Reporter(Protocol):
    """Anything that can turn a RunReport into text."""

    def render(self, run: RunReport) -> str: ...


OUTPUT_FORMATS = ("table", "json", "markdown")


def get_reporter(name: str, *, color: bool | None = None) -> Reporter:
    """Reporter for ``--output``; ``color`` only affects the table."""
    if name == "table":
        return TableReporter(color=color)
    if name == "json":
        return JsonReporter()
    if name == "markdown":
        return MarkdownReporter()
    raise ConfigError(f"unknown output format {name!r}; choose one of {list(OUTPUT_FORMATS)}")


__all__ = [
    "OUTPUT_FORMATS",
    "JsonReporter",
    "MarkdownReporter",
    "Reporter",
    "TableReporter",
    "get_reporter",
]
