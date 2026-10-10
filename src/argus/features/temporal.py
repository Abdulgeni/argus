"""Per-event temporal features.

NaN policy: the first event has no predecessor, so ``seconds_since_last_*`` is NaN there.
It is imputed as ``-1`` and an ``*_is_first`` flag (1/0) is added, so a model cannot
mistake "no previous event" for "very recent".
"""

from __future__ import annotations

import pandas as pd

TEMPORAL_FEATURES: tuple[str, ...] = (
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_business_hours",
    "seconds_since_last_event_same_entity",
    "seconds_since_last_event_same_entity_is_first",
    "seconds_since_last_event_globally",
    "seconds_since_last_event_globally_is_first",
)


def _impute_first(gaps: pd.Series) -> tuple[pd.Series, pd.Series]:
    first = gaps.isna()
    return gaps.fillna(-1.0).astype("float64"), first.astype("int64")


def add_calendar_and_global(frame: pd.DataFrame, start: int, end: int) -> None:
    """Add calendar features and the global inter-event gap to a time-sorted frame, in place."""
    ts = frame["timestamp"]
    hour = ts.dt.hour.astype("int64")
    dow = ts.dt.dayofweek.astype("int64")
    weekend = dow >= 5
    frame["hour_of_day"] = hour
    frame["day_of_week"] = dow
    frame["is_weekend"] = weekend.astype("int64")
    frame["is_business_hours"] = ((hour >= start) & (hour < end) & ~weekend).astype("int64")
    gaps, first = _impute_first(ts.diff().dt.total_seconds())
    frame["seconds_since_last_event_globally"] = gaps
    frame["seconds_since_last_event_globally_is_first"] = first


def add_same_entity_gap(view: pd.DataFrame) -> None:
    """Add the per-entity inter-event gap to a view sorted by (entity, timestamp), in place."""
    gaps = view.groupby("entity", sort=False)["timestamp"].diff().dt.total_seconds()
    values, first = _impute_first(gaps)
    view["seconds_since_last_event_same_entity"] = values
    view["seconds_since_last_event_same_entity_is_first"] = first
