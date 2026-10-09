"""Parser ABC, registry, shared coercion helpers and the streaming parse loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from argus.core.config import LogConfig
from argus.core.errors import ConfigError
from argus.core.io import iter_lines
from argus.core.models import Finding, LogRecord, ParseStats

MAX_FAILURE_RATE = 0.10
SAMPLE_LIMIT = 5


class ParseError(ValueError):
    """A single line could not be parsed into a LogRecord."""


class LogParser(ABC):
    """Turns one text line into a LogRecord, or raises ParseError."""

    name: ClassVar[str]

    def __init__(self) -> None:
        self.naive_timestamps = 0

    @classmethod
    @abstractmethod
    def from_config(cls, config: LogConfig) -> LogParser:
        """Build a parser from the ``log`` config section."""

    @abstractmethod
    def parse_line(self, line: str) -> LogRecord:
        """Parse one line; raise ParseError (or ValueError) if it is malformed."""

    def to_utc(self, value: datetime) -> datetime:
        """Normalize to aware UTC. Naive datetimes are assumed UTC and counted."""
        if value.tzinfo is None or value.utcoffset() is None:
            self.naive_timestamps += 1
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


P = TypeVar("P", bound=LogParser)
_REGISTRY: dict[str, type[LogParser]] = {}


def register_parser(name: str) -> Callable[[type[P]], type[P]]:
    """Class decorator registering a parser under ``name``."""

    def decorator(cls: type[P]) -> type[P]:
        if name in _REGISTRY:
            raise ValueError(f"parser {name!r} already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def available_parsers() -> list[str]:
    """Sorted names of registered parsers."""
    return sorted(_REGISTRY)


def create_parser(config: LogConfig) -> LogParser:
    """Instantiate the parser selected by ``config.format``."""
    try:
        cls = _REGISTRY[config.format]
    except KeyError:
        raise ConfigError(
            f"no parser for format {config.format!r}; available: {available_parsers()}"
        ) from None
    return cls.from_config(config)


def to_optional_str(value: Any) -> str | None:
    """None/blank -> None; other values are stringified."""
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    return text if text.strip() else None


def to_optional_int(value: Any, field: str) -> int | None:
    """Coerce to int. None, '' and '-' -> None. Anything non-numeric raises ParseError."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ParseError(f"{field}: boolean is not an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text in ("", "-"):
            return None
        try:
            return int(text)
        except ValueError:
            raise ParseError(f"{field}: {value!r} is not an integer") from None
    raise ParseError(f"{field}: unsupported value {value!r}")


@dataclass(frozen=True)
class ParseOutcome:
    """Result of parsing one file."""

    records: list[LogRecord]
    stats: ParseStats
    findings: list[Finding]


def build_findings(stats: ParseStats) -> list[Finding]:
    """Turn line accounting into user-visible findings."""
    findings: list[Finding] = []
    if stats.failed:
        samples = ", ".join(str(n) for n in stats.failed_line_samples)
        rate = stats.failure_rate
        too_many = rate > MAX_FAILURE_RATE
        findings.append(
            Finding(
                level="warning" if too_many else "info",
                code="high_parse_failure_rate" if too_many else "malformed_lines",
                message=(
                    f"{stats.failed} of {stats.parsed + stats.failed} lines could not be "
                    f"parsed ({rate:.1%}); sample line numbers: {samples}"
                ),
            )
        )
    if stats.naive_timestamps:
        findings.append(
            Finding(
                level="warning",
                code="naive_timestamps",
                message=(
                    f"{stats.naive_timestamps} timestamps had no timezone and were "
                    "assumed to be UTC"
                ),
            )
        )
    if stats.parsed == 0:
        findings.append(
            Finding(level="warning", code="no_records", message="no log records were parsed")
        )
    return findings


def parse_lines(
    lines: Iterable[tuple[int, str]], parser: LogParser, *, sample_limit: int = SAMPLE_LIMIT
) -> ParseOutcome:
    """Parse ``(line_number, line)`` pairs; malformed lines are counted, never raised."""
    stats = ParseStats()
    records: list[LogRecord] = []
    naive_before = parser.naive_timestamps
    for number, line in lines:
        stats.total_lines += 1
        if not line.strip():
            stats.blank_lines += 1
            continue
        try:
            records.append(parser.parse_line(line))
        except ValueError:
            stats.failed += 1
            if len(stats.failed_line_samples) < sample_limit:
                stats.failed_line_samples.append(number)
        else:
            stats.parsed += 1
    stats.naive_timestamps = parser.naive_timestamps - naive_before
    return ParseOutcome(records=records, stats=stats, findings=build_findings(stats))


def parse_file(path: Path, parser: LogParser) -> ParseOutcome:
    """Stream ``path`` line by line (plain/.gz/.bz2) and parse it."""
    return parse_lines(iter_lines(path), parser)
