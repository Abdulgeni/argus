"""Pluggable log parsers. Importing this package registers the built-in parsers."""

from argus.parsers import jsonl, nginx, syslog  # noqa: F401
from argus.parsers.base import (
    LogParser,
    ParseError,
    ParseOutcome,
    available_parsers,
    create_parser,
    parse_file,
    parse_lines,
    register_parser,
)

__all__ = [
    "LogParser",
    "ParseError",
    "ParseOutcome",
    "available_parsers",
    "create_parser",
    "parse_file",
    "parse_lines",
    "register_parser",
]
