import bz2
import gzip
from pathlib import Path

import pytest

from argus.core.errors import InputError
from argus.core.io import iter_lines

EXPECTED = [(1, "a"), (2, "b"), (3, "c")]


def test_plain_strips_crlf(tmp_path: Path) -> None:
    path = tmp_path / "x.log"
    path.write_bytes(b"a\r\nb\nc")
    assert list(iter_lines(path)) == EXPECTED


def test_gzip(tmp_path: Path) -> None:
    path = tmp_path / "x.log.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("a\nb\nc\n")
    assert list(iter_lines(path)) == EXPECTED


def test_bz2(tmp_path: Path) -> None:
    path = tmp_path / "x.log.bz2"
    with bz2.open(path, "wt", encoding="utf-8") as fh:
        fh.write("a\nb\nc\n")
    assert list(iter_lines(path)) == EXPECTED


def test_invalid_utf8_is_replaced_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "x.log"
    path.write_bytes(b"ok\n\xff\xfe bad\n")
    lines = list(iter_lines(path))
    assert len(lines) == 2 and lines[0] == (1, "ok")


def test_missing_file_and_corrupt_gzip(tmp_path: Path) -> None:
    with pytest.raises(InputError):
        list(iter_lines(tmp_path / "missing.log"))
    bad = tmp_path / "bad.gz"
    bad.write_bytes(b"this is not gzip")
    with pytest.raises(InputError):
        list(iter_lines(bad))
