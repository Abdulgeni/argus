"""Normalized data models shared across pipeline stages."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# LogRecord fields that parsers may populate from a source line (excludes raw/extra).
LOG_RECORD_FIELDS: tuple[str, ...] = (
    "timestamp",
    "source_ip",
    "user",
    "method",
    "path",
    "status",
    "bytes_sent",
    "user_agent",
)


class Severity(StrEnum):
    """Five-level severity. INFO means 'not flagged as anomalous'."""

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        """Numeric rank: INFO=0 ... CRITICAL=4."""
        return _SEVERITY_ORDER.index(self)

    def at_least(self, other: Severity) -> bool:
        """True if this severity is >= ``other``."""
        return self.rank >= other.rank

    @classmethod
    def from_str(cls, value: str) -> Severity:
        """Parse a case-insensitive severity name; raises ValueError if unknown."""
        try:
            return cls(value.strip().upper())
        except ValueError:
            names = ", ".join(s.value for s in cls)
            raise ValueError(f"unknown severity {value!r}; expected one of: {names}") from None


_SEVERITY_ORDER: tuple[Severity, ...] = (
    Severity.INFO,
    Severity.LOW,
    Severity.MEDIUM,
    Severity.HIGH,
    Severity.CRITICAL,
)


class LogRecord(BaseModel):
    """One normalized log event. ``timestamp`` is always timezone-aware UTC."""

    timestamp: datetime
    source_ip: str | None = None
    user: str | None = None
    method: str | None = None
    path: str | None = None
    status: int | None = None
    bytes_sent: int | None = None
    user_agent: str | None = None
    raw: str
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def _require_aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(UTC)


class ParseStats(BaseModel):
    """Line accounting for one parsed file."""

    total_lines: int = 0
    blank_lines: int = 0
    parsed: int = 0
    failed: int = 0
    naive_timestamps: int = 0
    failed_line_samples: list[int] = Field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        """failed / (parsed + failed); blank lines are excluded. 0.0 if nothing was attempted."""
        attempted = self.parsed + self.failed
        return self.failed / attempted if attempted else 0.0


class Finding(BaseModel):
    """A warning or note surfaced to the user (never silently dropped)."""

    level: Literal["info", "warning", "error"]
    code: str
    message: str


class AnomalyScore(BaseModel):
    """Detector output for one feature-matrix row. It only says "unusual", never "malicious"."""

    entity_type: str
    entity: str
    window_end: datetime
    event_row_id: int
    anomaly_score: float
    is_anomaly: bool
    severity: Severity


Category = Literal[
    "brute_force",
    "scanner",
    "data_exfiltration",
    "dos",
    "misconfiguration",
    "normal_variation",
    "unknown",
]


class FeatureDeviation(BaseModel):
    """One feature of an incident's peak row compared with the run's baseline."""

    feature: str
    window: str | None
    value: float
    baseline: float
    deviation: float


class IncidentReport(BaseModel):
    """One incident: a group of related anomalous rows plus its (LLM or template) summary."""

    id: str
    severity: Severity
    entity_type: str
    entity: str
    window_start: datetime
    window_end: datetime
    anomaly_score: float
    anomaly_count: int
    features: dict[str, float]
    top_deviations: list[FeatureDeviation]
    sample_log_lines: list[str]
    summary: str
    suspected_category: Category
    confidence: float
    evidence: list[str]
    recommended_action: str
    llm_used: bool
    llm_cached: bool = False
    llm_model: str | None = None
    warnings: list[str] = Field(default_factory=list)


class RunReport(BaseModel):
    """Everything one scan produced; the single input of every reporter."""

    source: str
    log_format: str
    stats: ParseStats
    feature_rows: int
    anomalous_rows: int
    incidents: list[IncidentReport]
    findings: list[Finding]
