"""Test helpers for building LogRecords."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from argus.core.models import LogRecord

T0 = datetime(2026, 10, 7, 10, 0, 0, tzinfo=UTC)


def rec(
    offset_s: float,
    ip: str | None = "1.1.1.1",
    user: str | None = None,
    path: str | None = "/",
    status: int | None = 200,
    nbytes: int | None = 100,
    ua: str | None = "ua",
) -> LogRecord:
    """A record ``offset_s`` seconds after T0."""
    return LogRecord(
        timestamp=T0 + timedelta(seconds=offset_s),
        source_ip=ip,
        user=user,
        method="GET",
        path=path,
        status=status,
        bytes_sent=nbytes,
        user_agent=ua,
        raw=f"line@{offset_s}",
    )
