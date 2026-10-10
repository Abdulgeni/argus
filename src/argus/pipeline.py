"""Pipeline: wires the stages together. Every stage is injected, so tests can pass fakes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from argus.core.config import ArgusConfig
from argus.core.matrix import DetectionResult, FeatureMatrix
from argus.core.models import Finding, LogRecord, RunReport
from argus.detection import IsolationForestDetector
from argus.features import PandasFeatureBuilder
from argus.llm import IncidentContext, SummaryResult, build_contexts, group_anomalies
from argus.llm.summarizer import build_summarizer
from argus.parsers import FileParser, ParseOutcome, create_parser


class ParserStage(Protocol):
    """Path -> parsed records with line accounting."""

    def parse(self, path: Path) -> ParseOutcome: ...


class FeatureBuilder(Protocol):
    """Records -> feature matrix."""

    def build(self, records: Sequence[LogRecord]) -> FeatureMatrix: ...


class Detector(Protocol):
    """Feature matrix (+ optional baseline matrix) -> one AnomalyScore per row."""

    def detect(
        self, target: FeatureMatrix, baseline: FeatureMatrix | None = None
    ) -> DetectionResult: ...


class Summarizer(Protocol):
    """Incident contexts -> incident reports."""

    def summarize(self, contexts: Sequence[IncidentContext]) -> SummaryResult: ...


@dataclass(frozen=True)
class ParsedMatrix:
    """Parse + feature stages for one file."""

    outcome: ParseOutcome
    matrix: FeatureMatrix

    @property
    def findings(self) -> list[Finding]:
        return [*self.outcome.findings, *self.matrix.findings]


def _prefixed(findings: Sequence[Finding], prefix: str) -> list[Finding]:
    return [
        Finding(level=f.level, code=f"{prefix}_{f.code}", message=f"{prefix}: {f.message}")
        for f in findings
    ]


def _unique(findings: Sequence[Finding]) -> list[Finding]:
    seen: set[tuple[str, str]] = set()
    result = []
    for finding in findings:
        key = (finding.code, finding.message)
        if key not in seen:
            seen.add(key)
            result.append(finding)
    return result


@dataclass(frozen=True)
class Pipeline:
    """Ingestion -> features -> detection -> reporting, with injected stages."""

    parser: ParserStage
    features: FeatureBuilder
    detector: Detector
    summarizer: Summarizer
    log_format: str = ""

    def parse_and_build(self, path: Path) -> ParsedMatrix:
        """Run only the parsing and feature stages (used by ``features`` and ``explain``)."""
        outcome = self.parser.parse(path)
        return ParsedMatrix(outcome, self.features.build(outcome.records))

    def run(self, path: Path, baseline: Path | None = None) -> RunReport:
        """Full scan of ``path``; when ``baseline`` is given the detector trains on it."""
        target = self.parse_and_build(path)
        reference = self.parse_and_build(baseline) if baseline is not None else None
        base_matrix = reference.matrix if reference else None
        detection = self.detector.detect(target.matrix, base_matrix)
        groups = group_anomalies(detection.scores)
        contexts = build_contexts(
            groups, target.matrix, detection.scores, target.outcome.records, base_matrix
        )
        summary = self.summarizer.summarize(contexts)
        findings = [
            *target.findings,
            *(_prefixed(reference.findings, "baseline") if reference else []),
            *detection.findings,
            *summary.findings,
        ]
        return RunReport(
            source=str(path),
            log_format=self.log_format,
            stats=target.outcome.stats,
            feature_rows=len(target.matrix.frame),
            anomalous_rows=sum(s.is_anomaly for s in detection.scores),
            incidents=summary.reports,
            findings=_unique(findings),
        )


def build_pipeline(
    config: ArgusConfig,
    *,
    no_llm: bool,
    env: Mapping[str, str],
    summarizer: Summarizer | None = None,
) -> Pipeline:
    """Production wiring from a validated config; ``summarizer`` lets tests inject a fake."""
    return Pipeline(
        parser=FileParser(create_parser(config.log)),
        features=PandasFeatureBuilder(config.features),
        detector=IsolationForestDetector(
            config.detection.contamination, config.detection.random_state
        ),
        summarizer=summarizer
        or build_summarizer(config.llm, config.cache.directory, env, no_llm=no_llm),
        log_format=config.log.format,
    )
