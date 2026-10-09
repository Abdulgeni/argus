import json
from datetime import UTC, datetime

import pytest

from argus.parsers import ParseError
from argus.parsers.jsonl import JsonLinesParser


def test_default_mapping() -> None:
    line = json.dumps(
        {
            "timestamp": "2026-10-09T09:14:02Z",
            "source_ip": "1.2.3.4",
            "path": "/a",
            "status": 200,
            "level": "info",
        }
    )
    r = JsonLinesParser().parse_line(line)
    assert r.timestamp == datetime(2026, 10, 9, 9, 14, 2, tzinfo=UTC)
    assert (r.source_ip, r.path, r.status) == ("1.2.3.4", "/a", 200)
    assert r.extra == {"level": "info"}


def test_dotted_paths_and_custom_mapping() -> None:
    parser = JsonLinesParser(
        {
            "timestamp": "@timestamp",
            "status": "http.response.status_code",
            "source_ip": "client.ip",
        }
    )
    line = (
        '{"@timestamp":"2026-10-09T09:14:02Z","client":{"ip":"1.2.3.4"},'
        '"http":{"response":{"status_code":503}}}'
    )
    r = parser.parse_line(line)
    assert (r.source_ip, r.status) == ("1.2.3.4", 503)
    assert r.extra == {}


def test_flat_dotted_key_supported() -> None:
    parser = JsonLinesParser({"status": "http.response.status_code"})
    r = parser.parse_line('{"timestamp":"2026-10-09T09:14:02Z","http.response.status_code":404}')
    assert r.status == 404 and r.extra == {}


def test_epoch_seconds_and_millis() -> None:
    parser = JsonLinesParser()
    a = parser.parse_line('{"timestamp": 1760000000}')
    b = parser.parse_line('{"timestamp": 1760000000000}')
    assert a.timestamp == b.timestamp
    assert a.timestamp.timestamp() == 1760000000
    assert parser.naive_timestamps == 0


def test_naive_timestamp_counted() -> None:
    parser = JsonLinesParser()
    r = parser.parse_line('{"timestamp":"2026-10-09T09:14:02"}')
    assert r.timestamp.tzinfo is not None
    assert parser.naive_timestamps == 1


@pytest.mark.parametrize(
    "line",
    [
        "{not json",
        "[1, 2]",
        '{"path": "/a"}',
        '{"timestamp": "yesterday"}',
        '{"timestamp": true}',
        '{"timestamp": 1e30}',
        '{"timestamp": "2026-10-09T09:14:02Z", "status": "abc"}',
    ],
)
def test_malformed_raises_value_error(line: str) -> None:
    with pytest.raises(ValueError):
        JsonLinesParser().parse_line(line)


def test_parse_error_is_value_error() -> None:
    assert issubclass(ParseError, ValueError)
