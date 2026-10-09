from pathlib import Path

from argus.core.config import LogConfig
from argus.parsers import available_parsers, create_parser, parse_file, parse_lines
from argus.parsers.nginx import NginxParser

NGINX_GOOD = '203.0.113.9 - - [09/Oct/2026:09:14:02 +0000] "GET / HTTP/1.1" 200 5 "-" "ua"'


def _write(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "a.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_registry() -> None:
    assert available_parsers() == ["json", "nginx", "syslog"]
    assert isinstance(create_parser(LogConfig(format="nginx")), NginxParser)


def test_at_threshold_is_info(tmp_path: Path) -> None:
    out = parse_file(
        _write(tmp_path, [NGINX_GOOD] * 9 + ["garbage"]), create_parser(LogConfig())
    )
    assert out.stats.parsed == 9 and out.stats.failed == 1
    assert out.stats.failed_line_samples == [10]
    assert [(f.code, f.level) for f in out.findings] == [("malformed_lines", "info")]


def test_above_threshold_warns(tmp_path: Path) -> None:
    out = parse_file(
        _write(tmp_path, [NGINX_GOOD] * 8 + ["bad", "bad"]), create_parser(LogConfig())
    )
    assert [(f.code, f.level) for f in out.findings] == [("high_parse_failure_rate", "warning")]
    assert out.stats.failed_line_samples == [9, 10]
    assert "20.0%" in out.findings[0].message


def test_sample_list_is_capped(tmp_path: Path) -> None:
    out = parse_file(_write(tmp_path, [NGINX_GOOD] + ["bad"] * 20), create_parser(LogConfig()))
    assert out.stats.failed == 20 and len(out.stats.failed_line_samples) == 5


def test_blank_lines_skipped_not_failed() -> None:
    lines = [(1, NGINX_GOOD), (2, ""), (3, "   "), (4, NGINX_GOOD)]
    out = parse_lines(lines, create_parser(LogConfig()))
    assert (out.stats.parsed, out.stats.blank_lines, out.stats.failed) == (2, 2, 0)
    assert out.findings == []


def test_naive_timestamp_finding(tmp_path: Path) -> None:
    path = _write(tmp_path, ['{"timestamp":"2026-10-09T09:14:02"}'])
    out = parse_file(path, create_parser(LogConfig(format="json")))
    assert [f.code for f in out.findings] == ["naive_timestamps"]
    assert out.stats.naive_timestamps == 1


def test_empty_input_warns(tmp_path: Path) -> None:
    path = tmp_path / "empty.log"
    path.write_text("", encoding="utf-8")
    out = parse_file(path, create_parser(LogConfig()))
    assert [f.code for f in out.findings] == ["no_records"]
