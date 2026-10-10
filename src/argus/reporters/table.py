"""Rich table reporter: one row per incident, color-coded by severity."""

from __future__ import annotations

import io

from rich.console import Console
from rich.table import Table
from rich.text import Text

from argus.core.models import RunReport
from argus.reporters.common import SEVERITY_STYLES, one_line, period, severity_counts

_LEVEL_STYLES = {"info": "dim", "warning": "yellow", "error": "bold red"}


class TableReporter:
    """Renders to a string; ``color`` None lets rich decide, False/True force it off/on."""

    def __init__(self, width: int | None = None, color: bool | None = None) -> None:
        self._width = width
        self._color = color

    def render(self, run: RunReport) -> str:
        buffer = io.StringIO()
        console = Console(
            file=buffer,
            width=self._width or 140,
            force_terminal=self._color,
            color_system="standard" if self._color else None,
            highlight=False,
        )
        console.print(self._summary_line(run))
        if run.incidents:
            console.print(self._table(run))
        else:
            console.print("No anomalies were flagged.", style="green")
        for finding in run.findings:
            style = _LEVEL_STYLES[finding.level]
            console.print(Text(f"[{finding.level.upper()}] {finding.message}", style=style))
        return buffer.getvalue()

    @staticmethod
    def _summary_line(run: RunReport) -> Text:
        counts = ", ".join(f"{n} {s.value}" for s, n in severity_counts(run.incidents).items())
        text = (
            f"Argus: {run.stats.parsed} events from {run.source} -> "
            f"{run.feature_rows} feature rows, {run.anomalous_rows} anomalous, "
            f"{len(run.incidents)} incident(s)"
        )
        return Text(text + (f" ({counts})" if counts else ""), style="bold")

    @staticmethod
    def _table(run: RunReport) -> Table:
        table = Table(show_lines=False, header_style="bold")
        for name in ("ID", "Severity", "Entity", "Time window", "Category", "Summary", "LLM"):
            table.add_column(name, no_wrap=name in ("ID", "Severity", "LLM"), overflow="fold")
        for incident in run.incidents:
            style = SEVERITY_STYLES[incident.severity]
            table.add_row(
                incident.id,
                Text(incident.severity.value, style=style),
                f"{incident.entity_type}: {incident.entity}",
                period(incident),
                incident.suspected_category,
                one_line(incident.summary),
                "yes" if incident.llm_used else "no",
            )
        return table
