"""End-to-end CLI tests through typer's CliRunner (skipped if typer is not installed)."""

from pathlib import Path

import pytest

pytest.importorskip("typer")
from typer.testing import CliRunner

from argus.cli import app

from .logs import attack_lines, normal_lines, write_log

runner = CliRunner()


@pytest.fixture
def log(tmp_path: Path) -> Path:
    return write_log(tmp_path / "a.log", normal_lines() + attack_lines())


def test_scan_exit_codes(log: Path) -> None:
    ok = runner.invoke(app, ["scan", str(log), "--no-llm", "--seed", "1"])
    assert ok.exit_code == 0 and "9.9.9.9" in ok.output
    failing = runner.invoke(app, ["scan", str(log), "--no-llm", "--seed", "1", "--fail-on", "high"])
    assert failing.exit_code == 1


def test_usage_errors_exit_2(log: Path, tmp_path: Path) -> None:
    assert runner.invoke(app, ["scan", str(tmp_path / "missing.log")]).exit_code == 2
    assert runner.invoke(app, ["scan", str(log), "--fail-on", "severe"]).exit_code == 2
    assert runner.invoke(app, ["scan", str(log), "--format", "xml"]).exit_code == 2
    assert runner.invoke(app, ["scan", str(log), "--contamination", "9"]).exit_code == 2


def test_json_output_and_file(log: Path, tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    result = runner.invoke(app, ["scan", str(log), "--no-llm", "-o", str(out), "--output", "json"])
    assert result.exit_code == 0 and out.read_text(encoding="utf-8").lstrip().startswith("{")


def test_features_and_explain(log: Path) -> None:
    csv = runner.invoke(app, ["features", str(log)])
    assert csv.exit_code == 0 and "request_count_1m" in csv.output
    explain = runner.invoke(app, ["explain", str(log), "--entity", "9.9.9.9", "--window", "15m"])
    assert explain.exit_code == 0 and "request_count_15m" in explain.output
    missing = runner.invoke(app, ["explain", str(log), "--entity", "8.8.8.8"])
    assert missing.exit_code == 2


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and result.output.startswith("argus ")
