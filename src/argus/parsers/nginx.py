"""nginx / Apache combined log format."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from argus.core.config import LogConfig
from argus.core.models import LogRecord
from argus.parsers.base import LogParser, ParseError, register_parser, to_optional_int

_QUOTED = r'(?:[^"\\]|\\.)*'
_LINE = re.compile(
    r"^(?P<ip>\S+) \S+ (?P<user>\S+) \[(?P<ts>[^\]]+)\] "
    rf'"(?P<request>{_QUOTED})" (?P<status>\d{{3}}|-) (?P<bytes>\d+|-)'
    rf'(?: "(?P<referer>{_QUOTED})" "(?P<ua>{_QUOTED})")?'
)
_TS_FORMAT = "%d/%b/%Y:%H:%M:%S %z"
_METHOD = re.compile(r"^[A-Z]+$")


def _dash_to_none(value: str | None) -> str | None:
    return None if value is None or value in ("-", "") else value


def _split_request(request: str) -> tuple[str | None, str | None, str | None]:
    """'GET /a?b=1 HTTP/1.1' -> ('GET', '/a', 'b=1'). Garbage -> (None, None, None)."""
    parts = request.split(" ")
    if len(parts) not in (2, 3) or not _METHOD.match(parts[0]):
        return None, None, None
    target = parts[1]
    path, _, query = target.partition("?")
    return parts[0], path or None, query or None


@register_parser("nginx")
class NginxParser(LogParser):
    """Parses the combined format; referer and user-agent are optional."""

    @classmethod
    def from_config(cls, config: LogConfig) -> NginxParser:
        return cls()

    def parse_line(self, line: str) -> LogRecord:
        match = _LINE.match(line)
        if match is None:
            raise ParseError("line does not match combined log format")
        try:
            timestamp = self.to_utc(datetime.strptime(match["ts"], _TS_FORMAT))
        except ValueError:
            raise ParseError(f"bad timestamp {match['ts']!r}") from None
        method, path, query = _split_request(match["request"])
        extra: dict[str, Any] = {}
        if query:
            extra["query"] = query
        referer = _dash_to_none(match["referer"])
        if referer:
            extra["referer"] = referer
        return LogRecord(
            timestamp=timestamp,
            source_ip=_dash_to_none(match["ip"]),
            user=_dash_to_none(match["user"]),
            method=method,
            path=path,
            status=to_optional_int(match["status"], "status"),
            bytes_sent=to_optional_int(match["bytes"], "bytes_sent"),
            user_agent=_dash_to_none(match["ua"]),
            raw=line,
            extra=extra,
        )
