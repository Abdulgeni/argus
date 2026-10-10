"""PandasFeatureBuilder: LogRecord[] -> FeatureMatrix (deterministic)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from argus.core.config import FeaturesConfig
from argus.core.matrix import ID_COLUMNS, FeatureMatrix
from argus.core.models import Finding, LogRecord
from argus.features.behavioral import (
    BEHAVIORAL_FEATURES,
    add_event_flags,
    add_new_path_flag,
    add_zscores,
    window_features,
)
from argus.features.frame import records_to_frame
from argus.features.temporal import (
    TEMPORAL_FEATURES,
    add_calendar_and_global,
    add_same_entity_gap,
)

_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}
FLOAT_DECIMALS = 6


def parse_window(label: str) -> pd.Timedelta:
    """'5m' -> Timedelta(minutes=5). Labels are validated by FeaturesConfig."""
    return pd.Timedelta(**{_UNITS[label[-1]]: int(label[:-1])})


def feature_column_names(windows: Sequence[str]) -> list[str]:
    """Temporal columns followed by ``<feature>_<window>`` for each window (window-major)."""
    behavioral = [f"{f}_{w}" for w in windows for f in BEHAVIORAL_FEATURES]
    return [*TEMPORAL_FEATURES, *behavioral]


def entity_key(frame: pd.DataFrame, entity_type: str) -> pd.Series:
    """Entity key per event; NaN where a required component is null."""
    if entity_type in ("source_ip", "user"):
        return frame[entity_type]
    both = frame["source_ip"].notna() & frame["user"].notna()
    joined = frame["source_ip"].astype(str) + "|" + frame["user"].astype(str)
    return joined.where(both)


class PandasFeatureBuilder:
    """Builds the per-entity, per-window matrix. Same input -> byte-identical output."""

    def __init__(self, config: FeaturesConfig | None = None) -> None:
        self._cfg = config or FeaturesConfig()

    def build(self, records: Sequence[LogRecord]) -> FeatureMatrix:
        """Compute features for every configured entity type and window."""
        cfg = self._cfg
        columns = feature_column_names(cfg.windows)
        findings: list[Finding] = []
        if not records:
            findings.append(
                Finding(level="warning", code="no_records", message="no records to featurize")
            )
            return self._matrix(self._empty(columns), columns, findings)
        frame = records_to_frame(records)
        add_calendar_and_global(frame, cfg.business_hours_start, cfg.business_hours_end)
        add_event_flags(frame, cfg.failed_auth_paths)
        parts = []
        for entity_type in cfg.entities:
            part, skipped = self._entity_matrix(frame, entity_type)
            parts.append(part)
            if skipped:
                findings.append(self._skipped_finding(entity_type, skipped))
        result = pd.concat(parts, ignore_index=True) if parts else self._empty(columns)
        return self._matrix(self._finalize(result, columns), columns, findings)

    def _entity_matrix(self, frame: pd.DataFrame, entity_type: str) -> tuple[pd.DataFrame, int]:
        key = entity_key(frame, entity_type)
        keep = key.notna()
        view = frame.assign(entity=key)[keep]
        view = view.sort_values(["entity", "timestamp", "row_id"], kind="stable")
        view = view.reset_index(drop=True)
        skipped = int((~keep).sum())
        if view.empty:
            return self._empty(feature_column_names(self._cfg.windows)), skipped
        add_same_entity_gap(view)
        add_new_path_flag(view)
        blocks = [window_features(view, parse_window(w), w) for w in self._cfg.windows]
        full = pd.concat([view, *blocks], axis=1)
        full["entity_type"] = entity_type
        full["window_end"] = full["timestamp"]
        full["event_row_id"] = full["row_id"]
        rows = add_zscores(self._one_row_per_minute(full), self._cfg.windows)
        return rows, skipped

    @staticmethod
    def _one_row_per_minute(full: pd.DataFrame) -> pd.DataFrame:
        """Keep the last event of each (entity, UTC minute); its windows cover the burst."""
        minute = full["timestamp"].dt.floor("min")
        return full.groupby(["entity", minute], sort=False).tail(1)

    @staticmethod
    def _finalize(result: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        out = result[[*ID_COLUMNS, *columns]].copy()
        floats = out.select_dtypes("float64").columns
        out[floats] = out[floats].round(FLOAT_DECIMALS)
        return out.reset_index(drop=True)

    @staticmethod
    def _empty(columns: list[str]) -> pd.DataFrame:
        return pd.DataFrame({name: [] for name in [*ID_COLUMNS, *columns]})

    def _matrix(
        self, frame: pd.DataFrame, columns: list[str], findings: list[Finding]
    ) -> FeatureMatrix:
        return FeatureMatrix(
            frame=frame,
            feature_columns=columns,
            temporal_columns=list(TEMPORAL_FEATURES),
            windows=list(self._cfg.windows),
            findings=findings,
        )

    @staticmethod
    def _skipped_finding(entity_type: str, skipped: int) -> Finding:
        return Finding(
            level="info",
            code="entity_null_rows",
            message=(
                f"{skipped} events have no value for entity '{entity_type}' and are "
                "excluded from that entity's features"
            ),
        )
