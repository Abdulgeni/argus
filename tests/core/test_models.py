from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from argus.core.models import LogRecord, ParseStats, Severity


def test_timestamp_normalized_to_utc() -> None:
    ts = datetime(2026, 10, 9, 9, 14, tzinfo=timezone(timedelta(hours=3)))
    record = LogRecord(timestamp=ts, raw="x")
    assert record.timestamp == datetime(2026, 10, 9, 6, 14, tzinfo=UTC)
    assert record.timestamp.utcoffset() == timedelta(0)


def test_naive_timestamp_rejected() -> None:
    with pytest.raises(ValidationError):
        LogRecord(timestamp=datetime(2026, 10, 9), raw="x")


def test_severity_ordering_and_parsing() -> None:
    assert Severity.CRITICAL.at_least(Severity.HIGH)
    assert not Severity.LOW.at_least(Severity.MEDIUM)
    assert Severity.from_str("high") is Severity.HIGH
    with pytest.raises(ValueError, match="unknown severity"):
        Severity.from_str("nope")


def test_failure_rate_excludes_blank_lines() -> None:
    assert ParseStats().failure_rate == 0.0
    assert ParseStats(parsed=9, failed=1, blank_lines=50).failure_rate == pytest.approx(0.1)
