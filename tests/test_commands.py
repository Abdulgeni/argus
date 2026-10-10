import io
import json
from pathlib import Path

import pytest

from argus.commands import ScanOptions, run_explain, run_features, run_scan
from argus.core.config import ArgusConfig
from argus.core.errors import ConfigError, InputError
from argus.core.matrix import DetectionResult, FeatureMatrix
from argus.core.models import AnomalyScore, RunReport, Severity
from argus.llm import IncidentContext, LLMSummarizer, SummaryResult
from argus.parsers import ParseOutcome
from argus.pipeline import Pipeline, build_pipeline

from .logs import attack_lines, normal_lines, write_log


@pytest.fixture
def attack_log(tmp_path: Path) -> Path:
    return write_log(tmp_path / "attack.log", normal_lines() + attack_lines())


def _scan(options: ScanOptions) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = run_scan(options, env={}, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_scan_table_finds_attacker_without_llm(attack_log: Path) -> None:
    code, out, _ = _scan(ScanOptions(attack_log, no_llm=True, seed=1))
    assert code == 0
    assert "9.9.9.9" in out and "INC-0001" in out
    assert "LLM disabled (--no-llm)" in out


def test_scan_json_is_valid_and_deterministic(attack_log: Path) -> None:
    opts = ScanOptions(attack_log, output="json", no_llm=True, seed=1)
    _, first, _ = _scan(opts)
    _, second, _ = _scan(opts)
    assert first == second
    run = RunReport.model_validate(json.loads(first))
    assert run.incidents[0].entity == "9.9.9.9" and not run.incidents[0].llm_used
    assert run.incidents[0].severity is Severity.CRITICAL
    assert "hunter2" not in first


def test_scan_markdown_to_file(attack_log: Path, tmp_path: Path) -> None:
    target = tmp_path / "report.md"
    code, out, err = _scan(ScanOptions(attack_log, output="markdown", no_llm=True, out=target))
    assert code == 0 and out == "" and "report written" in err
    assert target.read_text(encoding="utf-8").startswith("# Argus incident report")


def test_no_api_key_falls_back_with_finding(attack_log: Path) -> None:
    _, out, _ = _scan(ScanOptions(attack_log, output="json", seed=1))
    run = RunReport.model_validate(json.loads(out))
    assert any(f.code == "llm_no_api_key" for f in run.findings)
    assert not run.incidents[0].llm_used


@pytest.mark.parametrize(
    ("fail_on", "expected"), [(None, 0), ("critical", 1), ("HIGH", 1), ("low", 1)]
)
def test_exit_code_follows_fail_on(attack_log: Path, fail_on: str | None, expected: int) -> None:
    code, _, _ = _scan(ScanOptions(attack_log, no_llm=True, seed=1, fail_on=fail_on))
    assert code == expected


def test_clean_log_exits_zero_even_with_fail_on(tmp_path: Path) -> None:
    log = write_log(tmp_path / "tiny.log", normal_lines(n_ips=2, minutes=2))
    code, out, _ = _scan(ScanOptions(log, no_llm=True, fail_on="low"))
    assert code == 0 and "No anomalies were flagged." in out
    assert "too_few_rows" in out or "need 8" in out


@pytest.mark.parametrize(
    "options",
    [
        ScanOptions(Path("x"), fail_on="severe"),
        ScanOptions(Path("x"), contamination="lots"),
        ScanOptions(Path("x"), contamination="0.9"),
        ScanOptions(Path("x"), log_format="syslog"),
    ],
)
def test_usage_errors_raise_config_error(options: ScanOptions, tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        _scan(options)


def test_config_file_typo_and_cli_override(attack_log: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("detection:\n  contamnation: 0.1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="contamnation"):
        _scan(ScanOptions(attack_log, config=bad))
    ok = tmp_path / "ok.yaml"
    ok.write_text("detection:\n  contamination: 0.2\n  random_state: 3\n", encoding="utf-8")
    code, _, _ = _scan(ScanOptions(attack_log, config=ok, no_llm=True, contamination="auto"))
    assert code == 0


def test_baseline_mode(tmp_path: Path) -> None:
    baseline = write_log(tmp_path / "base.log", normal_lines(seed=5))
    target = write_log(tmp_path / "target.log", normal_lines(n_ips=5, seed=6) + attack_lines())
    _, out, _ = _scan(ScanOptions(target, output="json", no_llm=True, baseline=baseline, seed=1))
    run = RunReport.model_validate(json.loads(out))
    assert run.incidents[0].entity == "9.9.9.9"


def test_malformed_lines_are_reported_not_fatal(tmp_path: Path) -> None:
    log = write_log(tmp_path / "m.log", normal_lines() + attack_lines() + ["garbage"] * 2)
    _, out, _ = _scan(ScanOptions(log, output="json", no_llm=True, seed=1))
    run = RunReport.model_validate(json.loads(out))
    assert run.stats.failed == 2
    assert any(f.code == "malformed_lines" for f in run.findings)


def test_features_csv(attack_log: Path) -> None:
    out, err = io.StringIO(), io.StringIO()
    assert run_features(attack_log, env={}, stdout=out, stderr=err) == 0
    header = out.getvalue().splitlines()[0].split(",")
    assert header[:4] == ["entity_type", "entity", "window_end", "event_row_id"]
    assert "request_count_1m" in header and "path_entropy_1h" in header
    assert all(line.startswith("info:") for line in err.getvalue().splitlines())


def test_features_warnings_go_to_stderr(tmp_path: Path) -> None:
    log = write_log(tmp_path / "w.log", [*normal_lines(n_ips=2, minutes=2), "junk"])
    out, err = io.StringIO(), io.StringIO()
    run_features(log, env={}, stdout=out, stderr=err)
    assert "malformed" in err.getvalue() or "could not be parsed" in err.getvalue()
    assert "junk" not in out.getvalue()


def test_explain(attack_log: Path) -> None:
    out = io.StringIO()
    run_explain(attack_log, "9.9.9.9", "15m", env={}, stdout=out, stderr=io.StringIO())
    text = out.getvalue()
    assert "source_ip: 9.9.9.9" in text and "request_count_15m" in text
    assert "failed_auth_count_15m" in text and "request_count_1m" not in text


def test_explain_adds_unconfigured_valid_window(attack_log: Path) -> None:
    out = io.StringIO()
    run_explain(attack_log, "9.9.9.9", "30m", env={}, stdout=out, stderr=io.StringIO())
    assert "request_count_30m" in out.getvalue()


def test_explain_errors(attack_log: Path) -> None:
    with pytest.raises(InputError, match="not found"):
        run_explain(
            attack_log, "1.2.3.4", "15m", env={}, stdout=io.StringIO(), stderr=io.StringIO()
        )
    with pytest.raises(ConfigError):
        run_explain(attack_log, "9.9.9.9", "7x", env={}, stdout=io.StringIO(), stderr=io.StringIO())


class FakeParser:
    def __init__(self, real: Pipeline) -> None:
        self._real = real

    def parse(self, path: Path) -> ParseOutcome:
        return self._real.parser.parse(path)


class FlagEverything:
    def detect(
        self, target: FeatureMatrix, baseline: FeatureMatrix | None = None
    ) -> DetectionResult:
        frame = target.frame
        scores = [
            AnomalyScore(
                entity_type=str(r.entity_type),
                entity=str(r.entity),
                window_end=r.window_end.to_pydatetime(),
                event_row_id=int(r.event_row_id),
                anomaly_score=-0.5,
                is_anomaly=True,
                severity=Severity.HIGH,
            )
            for r in frame.itertuples()
        ]
        return DetectionResult(scores=scores)


class CountingSummarizer:
    def __init__(self) -> None:
        self.seen = 0

    def summarize(self, contexts: "list[IncidentContext]") -> SummaryResult:
        self.seen = len(contexts)
        return LLMSummarizer(None).summarize(contexts)


def test_pipeline_accepts_fake_stages(attack_log: Path) -> None:
    real = build_pipeline(ArgusConfig(), no_llm=True, env={})
    summarizer = CountingSummarizer()
    pipeline = Pipeline(FakeParser(real), real.features, FlagEverything(), summarizer)
    run = pipeline.run(attack_log)
    assert run.anomalous_rows == run.feature_rows > 0
    assert summarizer.seen == len(run.incidents) > 0
    assert all(i.severity is Severity.HIGH for i in run.incidents)
