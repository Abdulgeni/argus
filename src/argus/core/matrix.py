"""FeatureMatrix and DetectionResult: the contracts between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from argus.core.models import AnomalyScore, Finding

ID_COLUMNS: tuple[str, ...] = ("entity_type", "entity", "window_end", "event_row_id")


@dataclass(frozen=True)
class FeatureMatrix:
    """One row per (entity_type, entity, minute); trailing windows end at ``window_end``.

    ``frame`` holds ``ID_COLUMNS`` followed by ``feature_columns``. Feature columns are
    ``temporal_columns`` plus ``<feature>_<window>`` for every configured window.
    """

    frame: pd.DataFrame
    feature_columns: list[str]
    temporal_columns: list[str]
    windows: list[str]
    findings: list[Finding] = field(default_factory=list)

    def window_columns(self, window: str) -> list[str]:
        """Feature columns for one window, preceded by the temporal columns."""
        if window not in self.windows:
            raise ValueError(f"unknown window {window!r}; configured: {self.windows}")
        suffix = f"_{window}"
        return self.temporal_columns + [c for c in self.feature_columns if c.endswith(suffix)]

    def to_csv(self) -> str:
        """Deterministic CSV rendering (no index, UTC ISO timestamps)."""
        return str(self.frame.to_csv(index=False))


@dataclass(frozen=True)
class DetectionResult:
    """Detector output: one AnomalyScore per matrix row, in matrix row order."""

    scores: list[AnomalyScore]
    findings: list[Finding] = field(default_factory=list)
