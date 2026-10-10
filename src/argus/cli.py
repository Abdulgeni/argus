"""Typer command-line interface. Thin shell: all logic lives in ``argus.commands``.

Exit codes: 0 clean, 1 findings at/above --fail-on, 2 usage or tool error.
"""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from argus import __version__, commands
from argus.core.errors import ArgusError

app = typer.Typer(
    name="argus",
    help="Log anomaly detection (IsolationForest) with optional LLM incident summaries.",
    no_args_is_help=True,
    add_completion=False,
)


class LogFormat(StrEnum):
    nginx = "nginx"
    json = "json"
    syslog = "syslog"


class OutputFormat(StrEnum):
    table = "table"
    json = "json"
    markdown = "markdown"


LogFile = Annotated[
    Path,
    typer.Argument(exists=True, dir_okay=False, readable=True, help="Log file (.log, .gz, .bz2)."),
]
FormatOpt = Annotated[
    LogFormat | None,
    typer.Option("--format", help="Log format (default: from config, else nginx)."),
]
ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", exists=True, dir_okay=False, help="YAML configuration file."),
]


def _fail(exc: ArgusError) -> "typer.Exit":
    typer.echo(f"error: {exc}", err=True)
    return typer.Exit(exc.exit_code)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"argus {__version__}")
        raise typer.Exit(0)


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    """Argus."""


@app.command()
def scan(
    logfile: LogFile,
    log_format: FormatOpt = None,
    output: Annotated[
        OutputFormat, typer.Option("--output", help="Report format.")
    ] = OutputFormat.table,
    contamination: Annotated[
        str | None,
        typer.Option("--contamination", help="Expected anomaly fraction: a number or 'auto'."),
    ] = None,
    baseline: Annotated[
        Path | None,
        typer.Option("--baseline", exists=True, dir_okay=False, help="Fit on this file instead."),
    ] = None,
    config: ConfigOpt = None,
    no_llm: Annotated[bool, typer.Option("--no-llm", help="Template summaries only.")] = False,
    fail_on: Annotated[
        str | None,
        typer.Option("--fail-on", help="Exit 1 if an incident is at/above this severity."),
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed.")] = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Write the report to this file.")
    ] = None,
) -> None:
    """Scan a log file and report anomalous entities as incidents."""
    options = commands.ScanOptions(
        logfile=logfile,
        log_format=log_format.value if log_format else None,
        output=output.value,
        contamination=contamination,
        baseline=baseline,
        config=config,
        no_llm=no_llm,
        fail_on=fail_on,
        seed=seed,
        out=out,
    )
    try:
        code = commands.run_scan(options)
    except ArgusError as exc:
        raise _fail(exc) from exc
    raise typer.Exit(code)


@app.command()
def features(
    logfile: LogFile,
    log_format: FormatOpt = None,
    config: ConfigOpt = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Write the CSV to this file.")
    ] = None,
) -> None:
    """Print the per-entity, per-window feature matrix as CSV."""
    try:
        code = commands.run_features(
            logfile,
            log_format=log_format.value if log_format else None,
            config_path=config,
            out=out,
        )
    except ArgusError as exc:
        raise _fail(exc) from exc
    raise typer.Exit(code)


@app.command()
def explain(
    logfile: LogFile,
    entity: Annotated[str, typer.Option("--entity", help="Entity value, e.g. 1.2.3.4 or bob.")],
    window: Annotated[str, typer.Option("--window", help="Window, e.g. 1m, 5m, 15m, 1h.")] = "15m",
    log_format: FormatOpt = None,
    config: ConfigOpt = None,
) -> None:
    """Show every feature of one entity for one window (no LLM)."""
    try:
        code = commands.run_explain(
            logfile,
            entity,
            window,
            log_format=log_format.value if log_format else None,
            config_path=config,
        )
    except ArgusError as exc:
        raise _fail(exc) from exc
    raise typer.Exit(code)
