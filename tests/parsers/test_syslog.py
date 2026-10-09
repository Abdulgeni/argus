from datetime import datetime

import pytest

from argus.core.config import LogConfig
from argus.core.errors import ConfigError
from argus.parsers import ParseError, create_parser
from argus.parsers.syslog import SyslogParser

PATTERN = (
    r"^(?P<timestamp>\w{3}\s+\d+ \d{2}:\d{2}:\d{2}) (?P<host>\S+) sshd\[\d+\]: "
    r"Failed password for (?P<user>\S+) from (?P<source_ip>\S+) port (?P<port>\d+)"
)
LINE = "Oct  9 09:14:02 web1 sshd[123]: Failed password for root from 203.0.113.9 port 4242"


def _parser() -> SyslogParser:
    return SyslogParser(PATTERN, "%b %d %H:%M:%S", 2026)


def test_named_groups_map_to_fields_and_extra() -> None:
    parser = _parser()
    r = parser.parse_line(LINE)
    assert r.timestamp == datetime(2026, 10, 9, 9, 14, 2, tzinfo=r.timestamp.tzinfo)
    assert (r.user, r.source_ip) == ("root", "203.0.113.9")
    assert r.extra == {"host": "web1", "port": "4242"}
    assert parser.naive_timestamps == 1


def test_non_matching_line_raises() -> None:
    with pytest.raises(ParseError):
        _parser().parse_line("totally different")


def test_bad_timestamp_raises() -> None:
    with pytest.raises(ParseError):
        _parser().parse_line(LINE.replace("Oct", "Foo"))


def test_iso_timestamp_without_format() -> None:
    parser = SyslogParser(r"^(?P<timestamp>\S+) (?P<user>\S+)$")
    r = parser.parse_line("2026-10-09T09:14:02+00:00 alice")
    assert r.user == "alice" and parser.naive_timestamps == 0


def test_config_errors() -> None:
    with pytest.raises(ConfigError, match="timestamp"):
        SyslogParser(r"(?P<user>\S+)")
    with pytest.raises(ConfigError, match="valid regex"):
        SyslogParser("(")
    with pytest.raises(ConfigError, match="default_year"):
        SyslogParser(PATTERN, "%b %d %H:%M:%S", None)
    with pytest.raises(ConfigError, match="syslog_pattern"):
        create_parser(LogConfig(format="syslog"))
