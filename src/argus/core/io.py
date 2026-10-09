"""Transparent, streaming text access to plain, .gz and .bz2 files."""

from __future__ import annotations

import bz2
import gzip
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

from argus.core.errors import InputError


def open_text(path: Path) -> TextIO:
    """Open ``path`` for text reading; compression is chosen by suffix."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".gz":
            return gzip.open(path, "rt", encoding="utf-8", errors="replace")
        if suffix == ".bz2":
            return bz2.open(path, "rt", encoding="utf-8", errors="replace")
        return path.open("rt", encoding="utf-8", errors="replace")
    except OSError as exc:
        raise InputError(f"cannot open {path}: {exc.strerror or exc}") from exc


def iter_lines(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(line_number, line)`` (1-based, trailing CR/LF stripped), one line at a time."""
    with open_text(path) as handle:
        try:
            for number, line in enumerate(handle, start=1):
                yield number, line.rstrip("\r\n")
        except (OSError, EOFError, zlib.error) as exc:
            raise InputError(f"error reading {path}: {exc}") from exc
