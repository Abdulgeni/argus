"""Everything the LLM (or the template) needs to know about one incident."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from argus.core.matrix import FeatureMatrix
from argus.core.models import AnomalyScore, FeatureDeviation, LogRecord
from argus.features.behavioral import BEHAVIORAL_FEATURES
from argus.features.builder import parse_window
from argus.llm.grouping import GROUP_SPAN, AnomalyGroup
from argus.llm.scrub import scrub

MAX_LINES = 20
MAX_LINE_CHARS = 400
TOP_DEVIATIONS = 10
MIN_BASELINE_ROWS = 3
_MAD_TO_STD = 1.4826


@dataclass(frozen=True)
class IncidentContext:
    """One incident group plus its feature evidence and scrubbed log lines."""

    group: AnomalyGroup
    window_start: datetime
    window_end: datetime
    features: dict[str, float]
    deviations: list[FeatureDeviation]
    log_lines: list[str]
    total_lines: int


def split_feature(column: str, windows: Sequence[str]) -> tuple[str, str | None]:
    """'error_rate_5m' -> ('error_rate', '5m'); temporal columns -> (column, None)."""
    for window in windows:
        if column.endswith(f"_{window}"):
            return column[: -len(window) - 1], window
    return column, None


def _is_ranked(column: str, windows: Sequence[str]) -> bool:
    """Behavioral and gap features are ranked; calendar features and *_is_first flags are not."""
    base, window = split_feature(column, windows)
    if window is not None:
        return base in BEHAVIORAL_FEATURES
    return column.startswith("seconds_since_last") and not column.endswith("_is_first")


def feature_baselines(
    frame: pd.DataFrame, flags: np.ndarray, columns: list[str], entity_type: str
) -> tuple[pd.Series, pd.Series]:
    """Robust (median, scale) per feature for one entity type."""
    of_type = (frame["entity_type"] == entity_type).to_numpy()
    rows = of_type & ~flags
    if rows.sum() < MIN_BASELINE_ROWS:
        rows = of_type
    data = frame.loc[rows, columns].astype("float64")
    median = data.median()
    mad = (data - median).abs().median() * _MAD_TO_STD
    scale = mad.where(mad > 0, data.std(ddof=0)).fillna(0.0)
    return median, scale.where(scale > 0, 1.0)


def _deviations(
    peak_row: pd.Series,
    median: pd.Series,
    scale: pd.Series,
    windows: Sequence[str],
    columns: list[str],
) -> list[FeatureDeviation]:
    found = []
    for column in columns:
        if not _is_ranked(column, windows):
            continue
        value, base = float(peak_row[column]), float(median[column])
        dev = (value - base) / float(scale[column])
        if abs(dev) > 1e-9:
            name, window = split_feature(column, windows)
            found.append(
                FeatureDeviation(
                    feature=name, window=window, value=value, baseline=base, deviation=dev
                )
            )
    found.sort(key=lambda d: (-abs(d.deviation), d.feature, d.window or ""))
    return found[:TOP_DEVIATIONS]


def _record_key(record: LogRecord, entity_type: str) -> str | None:
    if entity_type == "source_ip":
        return record.source_ip
    if entity_type == "user":
        return record.user
    if record.source_ip is None or record.user is None:
        return None
    return f"{record.source_ip}|{record.user}"


def _collect_lines(
    records: Sequence[LogRecord], groups: Sequence[AnomalyGroup]
) -> dict[tuple[str, str], list[tuple[datetime, int, str]]]:
    wanted: dict[tuple[str, str], list[tuple[datetime, int, str]]] = {
        (g.entity_type, g.entity): [] for g in groups
    }
    types = sorted({entity_type for entity_type, _ in wanted})
    for index, record in enumerate(records):
        for entity_type in types:
            key = _record_key(record, entity_type)
            if key is not None and (entity_type, key) in wanted:
                wanted[(entity_type, key)].append((record.timestamp, index, record.raw))
    return wanted


def _line_window_start(
    group: AnomalyGroup, top: FeatureDeviation | None, windows: Sequence[str]
) -> datetime:
    label = top.window if top and top.window else None
    span = parse_window(label) if label else (parse_window(windows[0]) if windows else GROUP_SPAN)
    return group.start - span


def build_contexts(
    groups: Sequence[AnomalyGroup],
    matrix: FeatureMatrix,
    scores: Sequence[AnomalyScore],
    records: Sequence[LogRecord],
    reference: FeatureMatrix | None = None,
) -> list[IncidentContext]:
    """Assemble contexts in the order of ``groups``."""
    frame = matrix.frame
    flags = np.array([s.is_anomaly for s in scores], dtype=bool)
    columns = matrix.feature_columns
    row_of = {
        (et, ent, rid): i
        for i, (et, ent, rid) in enumerate(
            zip(frame["entity_type"], frame["entity"], frame["event_row_id"], strict=True)
        )
    }
    lines = _collect_lines(records, groups)
    cache: dict[str, tuple[pd.Series, pd.Series]] = {}
    contexts = []
    for group in groups:
        if group.entity_type not in cache:
            cache[group.entity_type] = _baselines_for(matrix, reference, flags, group.entity_type)
        median, scale = cache[group.entity_type]
        peak = group.peak
        row = frame.iloc[row_of[(peak.entity_type, peak.entity, peak.event_row_id)]]
        deviations = _deviations(row, median, scale, matrix.windows, columns)
        start = _line_window_start(group, deviations[0] if deviations else None, matrix.windows)
        selected = sorted(
            (
                item
                for item in lines[(group.entity_type, group.entity)]
                if start < item[0] <= group.end
            ),
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        contexts.append(
            IncidentContext(
                group=group,
                window_start=start,
                window_end=group.end,
                features={c: _clean(row[c]) for c in columns},
                deviations=deviations,
                log_lines=[_prepare(raw) for _, _, raw in selected[:MAX_LINES]],
                total_lines=len(selected),
            )
        )
    return contexts


def _baselines_for(
    matrix: FeatureMatrix, reference: FeatureMatrix | None, flags: np.ndarray, entity_type: str
) -> tuple[pd.Series, pd.Series]:
    columns = matrix.feature_columns
    if (
        reference is not None
        and (reference.frame["entity_type"] == entity_type).sum() >= MIN_BASELINE_ROWS
    ):
        none = np.zeros(len(reference.frame), dtype=bool)
        return feature_baselines(reference.frame, none, columns, entity_type)
    return feature_baselines(matrix.frame, flags, columns, entity_type)


def _clean(value: object) -> float:
    number = float(value)  # type: ignore[arg-type]
    return 0.0 if math.isnan(number) else number


def _prepare(raw: str) -> str:
    line = scrub(raw)
    return line if len(line) <= MAX_LINE_CHARS else line[:MAX_LINE_CHARS] + "...[truncated]"
