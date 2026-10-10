"""`argus explain`: every feature for one entity at one window (no LLM involved)."""

from __future__ import annotations

import io

from rich.console import Console
from rich.table import Table

from argus.core.errors import InputError
from argus.core.matrix import FeatureMatrix
from argus.llm.prompt import format_number

MAX_SUGGESTIONS = 5


def _console(buffer: io.StringIO, color: bool | None) -> Console:
    return Console(
        file=buffer,
        width=110,
        force_terminal=color,
        color_system="standard" if color else None,
        highlight=False,
    )


def render_explain(matrix: FeatureMatrix, entity: str, window: str, color: bool | None) -> str:
    """Per entity type that contains ``entity``: last / max / mean of each feature over its rows."""
    frame = matrix.frame
    rows = frame[frame["entity"] == entity]
    if rows.empty:
        known = sorted({str(e) for e in frame["entity"]})[:MAX_SUGGESTIONS]
        hint = f"; examples: {', '.join(known)}" if known else ""
        raise InputError(f"entity {entity!r} not found in the feature matrix{hint}")
    try:
        columns = matrix.window_columns(window)
    except ValueError as exc:
        raise InputError(str(exc)) from None
    buffer = io.StringIO()
    console = _console(buffer, color)
    for entity_type in dict.fromkeys(rows["entity_type"].tolist()):
        part = rows[rows["entity_type"] == entity_type]
        title = (
            f"{entity_type}: {entity} - window {window} - {len(part)} row(s), "
            f"{part['window_end'].iloc[0]:%Y-%m-%d %H:%M:%S} .. "
            f"{part['window_end'].iloc[-1]:%Y-%m-%d %H:%M:%S} UTC"
        )
        table = Table(title=title, title_justify="left", header_style="bold")
        for name in ("Feature", "Last", "Max", "Mean"):
            table.add_column(name, justify="left" if name == "Feature" else "right")
        for column in columns:
            values = part[column].astype("float64")
            table.add_row(
                column,
                format_number(float(values.iloc[-1])),
                format_number(float(values.max())),
                format_number(float(values.mean())),
            )
        console.print(table)
    return buffer.getvalue()
