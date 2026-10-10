"""Per-entity, per-window behavioral features via pandas groupby + time-based rolling.

Windows are trailing and event-time based: for an event at ``t`` the window is
``(t - w, t]``, so it always contains the current event. Truly empty windows therefore
cannot occur. Null handling:

* null ``status``: excluded from error counts and from the ``error_rate`` denominator
* null ``bytes_sent``: excluded from mean/std; all-null or <2 values -> 0.0
* null ``path`` / ``user_agent``: excluded from unique counts, entropy and new_path_ratio
* any ratio with a zero denominator -> 0.0
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from argus.features.entropy import count_unique, shannon_entropy

BEHAVIORAL_FEATURES: tuple[str, ...] = (
    "request_count",
    "unique_paths",
    "unique_user_agents",
    "error_count",
    "error_rate",
    "client_error_count",
    "server_error_count",
    "bytes_sent_mean",
    "bytes_sent_std",
    "path_entropy",
    "new_path_ratio",
    "failed_auth_count",
    "distinct_status_codes",
    "request_rate_zscore",
)
ZSCORE_STD_FLOOR = 1.0


def add_event_flags(frame: pd.DataFrame, failed_auth_paths: Sequence[str]) -> None:
    """Add the helper columns (prefixed ``_``) that rolling aggregations consume, in place."""
    status = frame["status"]
    known = status.notna()
    code = status.fillna(-1).astype("int64")
    path = frame["path"]
    pattern = "|".join(f"(?:{p})" for p in failed_auth_paths) or r"(?!)"
    frame["_one"] = 1.0
    frame["_has_status"] = known.astype("float64")
    frame["_is_error"] = (known & (code >= 400)).astype("float64")
    frame["_is_client_error"] = ((code >= 400) & (code < 500)).astype("float64")
    frame["_is_server_error"] = (code >= 500).astype("float64")
    frame["_status_code"] = code.astype("float64")
    frame["_has_path"] = path.notna().astype("float64")
    frame["_path_code"] = pd.factorize(path, sort=True)[0].astype("float64")
    frame["_ua_code"] = pd.factorize(frame["user_agent"], sort=True)[0].astype("float64")
    failed = code.isin([401, 403]) & path.str.contains(re.compile(pattern), na=False)
    frame["_is_failed_auth"] = failed.astype("float64")


def add_new_path_flag(view: pd.DataFrame) -> None:
    """Flag events whose path is the entity's first sighting (view sorted by entity, time)."""
    first = ~view.duplicated(["entity", "path"], keep="first")
    view["_is_new_path"] = (view["path"].notna() & first).astype("float64")


def _rolling(view: pd.DataFrame, column: str, window: pd.Timedelta) -> Any:
    return view.groupby("entity", sort=False).rolling(window, on="timestamp")[column]


def _align(series: pd.Series, view: pd.DataFrame) -> pd.Series:
    """Align a groupby-rolling/expanding result to ``view`` by position."""
    if len(series) != len(view):
        raise ValueError("rolling result length does not match the view")
    return pd.Series(series.to_numpy(), index=view.index)


def _rsum(view: pd.DataFrame, column: str, window: pd.Timedelta) -> pd.Series:
    return _align(_rolling(view, column, window).sum(), view)


def _rapply(view: pd.DataFrame, column: str, window: pd.Timedelta, func: Any) -> pd.Series:
    return _align(_rolling(view, column, window).apply(func, raw=True), view)


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.where(den > 0)).fillna(0.0)


def _counts(view: pd.DataFrame, w: pd.Timedelta) -> dict[str, pd.Series]:
    count = _rsum(view, "_one", w)
    errors = _rsum(view, "_is_error", w)
    return {
        "request_count": count,
        "error_count": errors,
        "error_rate": _ratio(errors, _rsum(view, "_has_status", w)),
        "client_error_count": _rsum(view, "_is_client_error", w),
        "server_error_count": _rsum(view, "_is_server_error", w),
        "failed_auth_count": _rsum(view, "_is_failed_auth", w),
        "new_path_ratio": _ratio(_rsum(view, "_is_new_path", w), _rsum(view, "_has_path", w)),
    }


def _diversity(view: pd.DataFrame, w: pd.Timedelta) -> dict[str, pd.Series]:
    return {
        "unique_paths": _rapply(view, "_path_code", w, count_unique),
        "unique_user_agents": _rapply(view, "_ua_code", w, count_unique),
        "distinct_status_codes": _rapply(view, "_status_code", w, count_unique),
        "path_entropy": _rapply(view, "_path_code", w, shannon_entropy),
    }


def _bytes(view: pd.DataFrame, w: pd.Timedelta) -> dict[str, pd.Series]:
    roll = _rolling(view, "bytes_sent", w)
    return {
        "bytes_sent_mean": _align(roll.mean(), view).fillna(0.0),
        "bytes_sent_std": _align(roll.std(), view).fillna(0.0),
    }


def request_zscore(view: pd.DataFrame, request_count: pd.Series) -> pd.Series:
    """Z-score of ``request_count`` against the entity's own earlier rows."""
    tmp = pd.DataFrame({"entity": view["entity"], "x": request_count})
    tmp["prev"] = tmp.groupby("entity", sort=False)["x"].shift(1)
    expanding = tmp.groupby("entity", sort=False)["prev"].expanding(min_periods=2)
    mean = _align(expanding.mean(), tmp)
    std = _align(expanding.std(), tmp)
    z = (tmp["x"] - mean) / np.maximum(std, ZSCORE_STD_FLOOR)
    return z.fillna(0.0)


def window_features(view: pd.DataFrame, window: pd.Timedelta, label: str) -> pd.DataFrame:
    """Behavioral features for one window except the z-score; columns are ``<feature>_<label>``."""
    feats: dict[str, pd.Series] = {}
    feats.update(_counts(view, window))
    feats.update(_diversity(view, window))
    feats.update(_bytes(view, window))
    return pd.DataFrame({f"{name}_{label}": feats[name] for name in feats})


def add_zscores(rows: pd.DataFrame, windows: Sequence[str]) -> pd.DataFrame:
    """Add ``request_rate_zscore_<w>`` to the per-minute rows (sorted by entity, time)."""
    rows = rows.reset_index(drop=True)
    for label in windows:
        rows[f"request_rate_zscore_{label}"] = request_zscore(rows, rows[f"request_count_{label}"])
    return rows
