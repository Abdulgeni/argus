"""Command logic (framework-free, so it is unit-testable). ``cli.py`` is the typer shell."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from argus.core.config import ArgusConfig, apply_overrides, load_config
from argus.core.errors import ConfigError, InputError
from argus.core.models import Finding, Severity
from argus.pipeline import Pipeline, build_pipeline
from argus.reporters import get_reporter
from argus.reporters.explain import render_explain

PipelineFactory = Callable[..., Pipeline]


@dataclass(frozen=True)
class ScanOptions:
    """Parsed ``argus scan`` flags; ``None`` means the flag was not given."""

    logfile: Path
    log_format: str | None = None
    output: str = "table"
    contamination: str | None = None
    baseline: Path | None = None
    config: Path | None = None
    no_llm: bool = False
    fail_on: str | None = None
    seed: int | None = None
    out: Path | None = None


def parse_contamination(value: str | None) -> float | str | None:
    """'auto' or a float; anything else is a usage error."""
    if value is None:
        return None
    if value.strip().lower() == "auto":
        return "auto"
    try:
        return float(value)
    except ValueError:
        raise ConfigError(f"--contamination must be 'auto' or a number, got {value!r}") from None


def parse_fail_on(value: str | None) -> Severity | None:
    """Case-insensitive severity name, or None."""
    if value is None:
        return None
    try:
        return Severity.from_str(value)
    except ValueError as exc:
        raise ConfigError(f"--fail-on: {exc}") from None


def resolve_config(path: Path | None, overrides: Mapping[str, object]) -> ArgusConfig:
    """defaults < config file < CLI overrides."""
    return apply_overrides(load_config(path), overrides)


def use_color(stream: TextIO, env: Mapping[str, str]) -> bool:
    """Color only for a real terminal, honoring the NO_COLOR convention."""
    return stream.isatty() and "NO_COLOR" not in env


def print_findings(findings: list[Finding], stream: TextIO) -> None:
    """Findings go to stderr so stdout stays machine-readable."""
    for finding in findings:
        print(f"{finding.level}: {finding.message}", file=stream)


def _emit(text: str, out: Path | None, stdout: TextIO, stderr: TextIO) -> None:
    if out is None:
        stdout.write(text)
        stdout.flush()
        return
    try:
        out.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise InputError(f"cannot write {out}: {exc.strerror or exc}") from exc
    print(f"report written to {out}", file=stderr)


def run_scan(
    options: ScanOptions,
    *,
    env: Mapping[str, str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    pipeline_factory: PipelineFactory = build_pipeline,
) -> int:
    """Execute ``argus scan``. Returns 0 (clean) or 1 (findings at/above --fail-on)."""
    environ = os.environ if env is None else env
    out_stream, err_stream = stdout or sys.stdout, stderr or sys.stderr
    fail_on = parse_fail_on(options.fail_on)
    config = resolve_config(
        options.config,
        {
            "log.format": options.log_format,
            "detection.contamination": parse_contamination(options.contamination),
            "detection.random_state": options.seed,
            "detection.baseline": options.baseline,
        },
    )
    pipeline = pipeline_factory(config, no_llm=options.no_llm, env=environ)
    run = pipeline.run(options.logfile, config.detection.baseline)
    color = options.out is None and use_color(out_stream, environ)
    _emit(
        get_reporter(options.output, color=color).render(run), options.out, out_stream, err_stream
    )
    if fail_on is not None and any(i.severity.at_least(fail_on) for i in run.incidents):
        return 1
    return 0


def run_features(
    logfile: Path,
    *,
    log_format: str | None = None,
    config_path: Path | None = None,
    out: Path | None = None,
    env: Mapping[str, str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    pipeline_factory: PipelineFactory = build_pipeline,
) -> int:
    """Execute ``argus features``: the feature matrix as CSV."""
    out_stream, err_stream = stdout or sys.stdout, stderr or sys.stderr
    config = resolve_config(config_path, {"log.format": log_format})
    pipeline = pipeline_factory(config, no_llm=True, env=os.environ if env is None else env)
    parsed = pipeline.parse_and_build(logfile)
    print_findings(parsed.findings, err_stream)
    _emit(parsed.matrix.to_csv(), out, out_stream, err_stream)
    return 0


def run_explain(
    logfile: Path,
    entity: str,
    window: str,
    *,
    log_format: str | None = None,
    config_path: Path | None = None,
    env: Mapping[str, str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    pipeline_factory: PipelineFactory = build_pipeline,
) -> int:
    """Execute ``argus explain``: all features of one entity at one window (never uses an LLM)."""
    out_stream, err_stream = stdout or sys.stdout, stderr or sys.stderr
    base = load_config(config_path)
    windows = base.features.windows
    if window not in windows:
        windows = [*windows, window]
    config = apply_overrides(base, {"log.format": log_format, "features.windows": windows})
    environ = os.environ if env is None else env
    pipeline = pipeline_factory(config, no_llm=True, env=environ)
    parsed = pipeline.parse_and_build(logfile)
    print_findings(parsed.findings, err_stream)
    text = render_explain(parsed.matrix, entity, window, use_color(out_stream, environ))
    out_stream.write(text)
    out_stream.flush()
    return 0
