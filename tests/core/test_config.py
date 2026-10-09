from pathlib import Path

import pytest

from argus.core.config import ArgusConfig, apply_overrides, load_config
from argus.core.errors import ConfigError


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "argus.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults() -> None:
    cfg = load_config(None)
    assert cfg.features.windows == ["1m", "5m", "15m", "1h"]
    assert cfg.detection.contamination == 0.05
    assert cfg.llm.provider == "anthropic"


def test_empty_file_gives_defaults(tmp_path: Path) -> None:
    assert load_config(_write(tmp_path, "")) == ArgusConfig()


def test_typo_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="contamnation"):
        load_config(_write(tmp_path, "detection:\n  contamnation: 0.1\n"))


@pytest.mark.parametrize(
    "text",
    [
        "features:\n  windows: [5x]\n",
        "features:\n  windows: []\n",
        "features:\n  business_hours_start: 18\n  business_hours_end: 9\n",
        "features:\n  failed_auth_paths: ['(']\n",
        "detection:\n  contamination: 0.9\n",
        "log:\n  json_fields: {bogus: x}\n",
        "llm:\n  provider: gemini\n",
    ],
)
def test_invalid_values_rejected(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, text))


def test_contamination_auto_ok(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, "detection:\n  contamination: auto\n"))
    assert cfg.detection.contamination == "auto"


def test_cli_overrides_beat_config_and_none_is_skipped(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, "detection:\n  contamination: 0.1\n  random_state: 7\n"))
    merged = apply_overrides(
        cfg, {"detection.contamination": 0.2, "detection.random_state": None}
    )
    assert merged.detection.contamination == 0.2
    assert merged.detection.random_state == 7


def test_unknown_override_and_bad_value_rejected() -> None:
    cfg = ArgusConfig()
    with pytest.raises(ConfigError):
        apply_overrides(cfg, {"detection.nope": 1})
    with pytest.raises(ConfigError):
        apply_overrides(cfg, {"detection.contamination": 5.0})


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.yaml")
