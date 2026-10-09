"""YAML-backed configuration. Precedence: defaults < config file < CLI overrides."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from argus.core.errors import ConfigError
from argus.core.models import LOG_RECORD_FIELDS

_WINDOW_RE = re.compile(r"^[1-9]\d*[smhd]$")


class _Strict(BaseModel):
    """Base for config sections: unknown keys are errors, so typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class LogConfig(_Strict):
    """Log format selection and parser settings."""

    format: Literal["nginx", "json", "syslog"] = "nginx"
    json_fields: dict[str, str] = {}
    syslog_pattern: str | None = None
    syslog_timestamp_format: str | None = None
    syslog_default_year: int | None = None

    @field_validator("json_fields")
    @classmethod
    def _known_fields(cls, value: dict[str, str]) -> dict[str, str]:
        unknown = sorted(set(value) - set(LOG_RECORD_FIELDS))
        if unknown:
            raise ValueError(
                f"unknown LogRecord fields: {unknown}; allowed: {list(LOG_RECORD_FIELDS)}"
            )
        return value


class FeaturesConfig(_Strict):
    """Feature-engineering settings."""

    entities: list[Literal["source_ip", "user", "source_ip+user"]] = [
        "source_ip",
        "user",
        "source_ip+user",
    ]
    windows: list[str] = ["1m", "5m", "15m", "1h"]
    business_hours_start: int = 9
    business_hours_end: int = 17
    failed_auth_paths: list[str] = [r"/login", r"/auth", r"/admin"]

    @field_validator("entities")
    @classmethod
    def _entities_ok(cls, value: list[str]) -> list[str]:
        if not value or len(set(value)) != len(value):
            raise ValueError("entities must be non-empty and unique")
        return value

    @field_validator("windows")
    @classmethod
    def _windows_ok(cls, value: list[str]) -> list[str]:
        if not value or len(set(value)) != len(value):
            raise ValueError("windows must be non-empty and unique")
        bad = [w for w in value if not _WINDOW_RE.match(w)]
        if bad:
            raise ValueError(f"invalid window(s) {bad}; use forms like 30s, 1m, 5m, 1h, 1d")
        return value

    @field_validator("failed_auth_paths")
    @classmethod
    def _regexes_compile(cls, value: list[str]) -> list[str]:
        for pattern in value:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid regex {pattern!r}: {exc}") from None
        return value

    @model_validator(mode="after")
    def _hours_ok(self) -> FeaturesConfig:
        if not (0 <= self.business_hours_start < self.business_hours_end <= 24):
            raise ValueError("require 0 <= business_hours_start < business_hours_end <= 24")
        return self


class DetectionConfig(_Strict):
    """Anomaly detection settings."""

    contamination: float | Literal["auto"] = 0.05
    random_state: int = 42
    baseline: Path | None = None

    @field_validator("contamination")
    @classmethod
    def _contamination_ok(cls, value: float | str) -> float | str:
        if isinstance(value, float) and not (0.0 < value <= 0.5):
            raise ValueError("contamination must be 'auto' or a float in (0, 0.5]")
        return value


class LLMConfig(_Strict):
    """LLM settings. API keys come only from environment variables."""

    provider: Literal["anthropic", "openai"] = "anthropic"
    model: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.0
    batch_size: int = 10
    cache_ttl_hours: float = 168.0
    timeout_s: float = 30.0

    @model_validator(mode="after")
    def _positive(self) -> LLMConfig:
        if self.max_tokens < 1 or self.batch_size < 1:
            raise ValueError("max_tokens and batch_size must be >= 1")
        if self.timeout_s <= 0 or self.cache_ttl_hours < 0:
            raise ValueError("timeout_s must be > 0 and cache_ttl_hours >= 0")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be within [0, 2]")
        return self


class CacheConfig(_Strict):
    """Disk cache location. None = ARGUS_CACHE_DIR, else the platform default."""

    directory: Path | None = None


class ArgusConfig(_Strict):
    """Root configuration object."""

    log: LogConfig = LogConfig()
    features: FeaturesConfig = FeaturesConfig()
    detection: DetectionConfig = DetectionConfig()
    llm: LLMConfig = LLMConfig()
    cache: CacheConfig = CacheConfig()


def _format_errors(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        location = ".".join(str(p) for p in err["loc"]) or "<root>"
        parts.append(f"{location}: {err['msg']}")
    return "; ".join(parts)


def load_config(path: Path | None) -> ArgusConfig:
    """Load and validate a YAML config; ``None`` returns pure defaults."""
    if path is None:
        return ArgusConfig()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc.strerror or exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    try:
        return ArgusConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{path}: {_format_errors(exc)}") from exc


def apply_overrides(config: ArgusConfig, overrides: Mapping[str, Any]) -> ArgusConfig:
    """Return a new config with CLI overrides applied.

    Keys are ``"section.field"`` (e.g. ``"detection.contamination"``).
    ``None`` values mean "flag not given" and are skipped.
    """
    data = config.model_dump(mode="python")
    for key, value in overrides.items():
        if value is None:
            continue
        section, _, field = key.partition(".")
        if not field or section not in data or field not in data[section]:
            raise ConfigError(f"unknown config override {key!r}")
        data[section][field] = value
    try:
        return ArgusConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_errors(exc)) from exc
