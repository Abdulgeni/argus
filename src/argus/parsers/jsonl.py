"""JSON-lines parser with configurable key mapping and dotted-path lookup."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from argus.core.config import LogConfig
from argus.core.models import LOG_RECORD_FIELDS, LogRecord
from argus.parsers.base import (
    LogParser,
    ParseError,
    register_parser,
    to_optional_int,
    to_optional_str,
)

DEFAULT_JSON_FIELDS: dict[str, str] = {name: name for name in LOG_RECORD_FIELDS}
_EPOCH_MS_THRESHOLD = 1e11


def lookup(obj: Mapping[str, Any], path: str) -> Any:
    """Resolve ``a.b.c`` in nested dicts. A flat key containing dots (ECS style) wins first."""
    if path in obj:
        return obj[path]
    current: Any = obj
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


@register_parser("json")
class JsonLinesParser(LogParser):
    """One JSON object per line; ``fields`` maps LogRecord field -> dotted JSON path."""

    def __init__(self, fields: Mapping[str, str] | None = None) -> None:
        super().__init__()
        self._fields: dict[str, str] = {**DEFAULT_JSON_FIELDS, **(fields or {})}
        self._paths = frozenset(self._fields.values())
        self._roots = frozenset(p.split(".")[0] for p in self._paths)

    @classmethod
    def from_config(cls, config: LogConfig) -> JsonLinesParser:
        return cls(config.json_fields)

    def parse_line(self, line: str) -> LogRecord:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ParseError(f"invalid JSON: {exc.msg}") from None
        if not isinstance(obj, dict):
            raise ParseError("JSON line is not an object")
        values = {name: lookup(obj, path) for name, path in self._fields.items()}
        return LogRecord(
            timestamp=self._timestamp(values["timestamp"]),
            source_ip=to_optional_str(values["source_ip"]),
            user=to_optional_str(values["user"]),
            method=to_optional_str(values["method"]),
            path=to_optional_str(values["path"]),
            status=to_optional_int(values["status"], "status"),
            bytes_sent=to_optional_int(values["bytes_sent"], "bytes_sent"),
            user_agent=to_optional_str(values["user_agent"]),
            raw=line,
            extra={k: v for k, v in obj.items() if not self._consumed(k)},
        )

    def _consumed(self, key: str) -> bool:
        return (
            key in self._roots
            or key in self._paths
            or any(p.startswith(key + ".") for p in self._paths)
        )

    def _timestamp(self, value: Any) -> datetime:
        if value is None or isinstance(value, bool):
            raise ParseError("missing or invalid timestamp")
        if isinstance(value, int | float):
            seconds = value / 1000 if abs(value) > _EPOCH_MS_THRESHOLD else value
            try:
                return datetime.fromtimestamp(seconds, tz=UTC)
            except (OverflowError, OSError, ValueError):
                raise ParseError(f"epoch timestamp out of range: {value!r}") from None
        if isinstance(value, str):
            try:
                return self.to_utc(datetime.fromisoformat(value.strip()))
            except ValueError:
                raise ParseError(f"bad ISO-8601 timestamp {value!r}") from None
        raise ParseError(f"unsupported timestamp type {type(value).__name__}")
