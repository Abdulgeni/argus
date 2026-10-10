"""LogRecord[] -> canonical, stably sorted DataFrame (``raw`` is deliberately left out)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from argus.core.models import LogRecord

FRAME_COLUMNS: tuple[str, ...] = (
    "row_id",
    "timestamp",
    "source_ip",
    "user",
    "method",
    "path",
    "status",
    "bytes_sent",
    "user_agent",
)


def records_to_frame(records: Sequence[LogRecord]) -> pd.DataFrame:
    """Build the event frame, sorted by ``(timestamp, row_id)``.

    ``row_id`` is the index of the record in the input sequence, which makes every
    sort total and therefore deterministic. ``status`` is nullable ``Int64``;
    ``bytes_sent`` is ``float64`` with NaN for null.
    """
    frame = pd.DataFrame(
        {
            "row_id": np.arange(len(records), dtype=np.int64),
            "timestamp": pd.DatetimeIndex([r.timestamp for r in records], tz="UTC").as_unit("ns"),
            "source_ip": pd.Series([r.source_ip for r in records], dtype="object"),
            "user": pd.Series([r.user for r in records], dtype="object"),
            "method": pd.Series([r.method for r in records], dtype="object"),
            "path": pd.Series([r.path for r in records], dtype="object"),
            "status": pd.array([r.status for r in records], dtype="Int64"),
            "bytes_sent": pd.Series([r.bytes_sent for r in records], dtype="float64"),
            "user_agent": pd.Series([r.user_agent for r in records], dtype="object"),
        }
    )
    return frame.sort_values(["timestamp", "row_id"], kind="stable").reset_index(drop=True)
