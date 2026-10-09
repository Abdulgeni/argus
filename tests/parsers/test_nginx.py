from datetime import UTC, datetime

import pytest

from argus.core.config import LogConfig
from argus.parsers import ParseError, create_parser
from argus.parsers.nginx import NginxParser


@pytest.fixture
def parser() -> NginxParser:
    p = create_parser(LogConfig(format="nginx"))
    assert isinstance(p, NginxParser)
    return p


def test_full_line(parser: NginxParser) -> None:
    line = (
        '203.0.113.9 - bob [09/Oct/2026:09:14:02 +0300] '
        '"POST /login?next=/home HTTP/1.1" 401 512 "-" "curl/8.0"'
    )
    r = parser.parse_line(line)
    assert r.timestamp == datetime(2026, 10, 9, 6, 14, 2, tzinfo=UTC)
    assert (r.source_ip, r.user, r.method, r.path) == ("203.0.113.9", "bob", "POST", "/login")
    assert (r.status, r.bytes_sent, r.user_agent) == (401, 512, "curl/8.0")
    assert r.extra == {"query": "next=/home"}
    assert r.raw == line
    assert parser.naive_timestamps == 0


def test_dashes_become_none(parser: NginxParser) -> None:
    r = parser.parse_line(
        '198.51.100.1 - - [09/Oct/2026:09:14:02 +0000] "GET / HTTP/1.1" 200 - "-" "-"'
    )
    assert r.user is None and r.bytes_sent is None and r.user_agent is None
    assert r.path == "/" and r.extra == {}


def test_garbage_request_keeps_record(parser: NginxParser) -> None:
    r = parser.parse_line(
        r'10.0.0.1 - - [09/Oct/2026:09:14:02 +0000] "\x16\x03\x01" 400 0 "-" "-"'
    )
    assert r.method is None and r.path is None and r.status == 400


def test_referer_and_ua_optional(parser: NginxParser) -> None:
    r = parser.parse_line('1.2.3.4 - - [09/Oct/2026:09:14:02 +0000] "GET /a HTTP/1.1" 200 10')
    assert r.user_agent is None and r.bytes_sent == 10


@pytest.mark.parametrize(
    "line",
    [
        "not a log line",
        '1.2.3.4 - - [99/Foo/2026:09:14:02 +0000] "GET / HTTP/1.1" 200 5',
    ],
)
def test_malformed_raises(parser: NginxParser, line: str) -> None:
    with pytest.raises(ParseError):
        parser.parse_line(line)
