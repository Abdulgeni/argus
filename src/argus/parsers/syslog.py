"""Syslog-style parser driven by a user-supplied regex with named groups."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from argus.core.config import LogConfig
from argus.core.errors import ConfigError
from argus.core.models import LOG_RECORD_FIELDS, LogRecord
from argus.parsers.base import (
    LogParser,
    ParseError,
    register_parser,
    to_optional_int,
    to_optional_str,
)

_YEAR_DIRECTIVES = ("%Y", "%y")


@register_parser("syslog")
class SyslogParser(LogParser):
    """Named groups matching LogRecord field names populate them; other groups go to ``extra``."""

    def __init__(
        self,
        pattern: str,
        timestamp_format: str | None = None,
        default_year: int | None = None,
    ) -> None:
        super().__init__()
        try:
            self._regex = re.compile(pattern)
        except re.error as exc:
            raise ConfigError(f"log.syslog_pattern is not a valid regex: {exc}") from None
        if "timestamp" not in self._regex.groupindex:
            raise ConfigError("log.syslog_pattern must define a (?P<timestamp>...) group")
        self._format = timestamp_format
        self._default_year = default_year
        if (
            timestamp_format is not None
            and not any(d in timestamp_format for d in _YEAR_DIRECTIVES)
            and default_year is None
        ):
            raise ConfigError(
                "log.syslog_timestamp_format has no year; set log.syslog_default_year "
                "(the current year is never guessed, to keep runs deterministic)"
            )

    @classmethod
    def from_config(cls, config: LogConfig) -> SyslogParser:
        if not config.syslog_pattern:
            raise ConfigError("log.format is 'syslog' but log.syslog_pattern is not set")
        return cls(
            config.syslog_pattern,
            config.syslog_timestamp_format,
            config.syslog_default_year,
        )

    def parse_line(self, line: str) -> LogRecord:
        match = self._regex.match(line)
        if match is None:
            raise ParseError("line does not match syslog pattern")
        groups = {k: v for k, v in match.groupdict().items() if v is not None}
        extra: dict[str, Any] = {k: v for k, v in groups.items() if k not in LOG_RECORD_FIELDS}
        return LogRecord(
            timestamp=self._timestamp(groups["timestamp"]),
            source_ip=to_optional_str(groups.get("source_ip")),
            user=to_optional_str(groups.get("user")),
            method=to_optional_str(groups.get("method")),
            path=to_optional_str(groups.get("path")),
            status=to_optional_int(groups.get("status"), "status"),
            bytes_sent=to_optional_int(groups.get("bytes_sent"), "bytes_sent"),
            user_agent=to_optional_str(groups.get("user_agent")),
            raw=line,
            extra=extra,
        )

    def _timestamp(self, text: str) -> datetime:
        try:
            if self._format is None:
                return self.to_utc(datetime.fromisoformat(text.strip()))
            if any(d in self._format for d in _YEAR_DIRECTIVES):
                return self.to_utc(datetime.strptime(text.strip(), self._format))
            parsed = datetime.strptime(f"{self._default_year} {text.strip()}", f"%Y {self._format}")
            return self.to_utc(parsed)
        except ValueError:
            raise ParseError(f"bad timestamp {text!r}") from None