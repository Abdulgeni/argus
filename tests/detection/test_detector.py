import random

import pytest

from argus.core.config import FeaturesConfig
from argus.core.errors import ConfigError
from argus.core.matrix import FeatureMatrix
from argus.core.models import LogRecord, Severity
from argus.detection import IsolationForestDetector
from argus.features import PandasFeatureBuilder

from ..features.helpers import rec

ATTACKER = "9.9.9.9"
PATHS = ["/", "/about", "/api/items", "/static/app.js"]


def _normal(n_ips: int = 30, minutes: int = 10, seed: int = 0) -> list[LogRecord]:
    rng = random.Random(seed)
    out = []
    for ip in range(n_ips):
        for minute in range(minutes):
            status = 404 if rng.random() < 0.05 else 200
            out.append(
                rec(
                    minute * 60 + ip,
                    ip=f"10.0.0.{ip}",
                    path=rng.choice(PATHS),
                    status=status,
                    nbytes=rng.randint(200, 2000),
                    ua=f"browser-{ip % 3}",
                )
            )
    return out


def _attack() -> list[LogRecord]:
    return [
        rec(300 + i * 0.5, ip=ATTACKER, path="/login", status=401, nbytes=512, ua="curl")
        for i in range(80)
    ]


def _matrix(records: list[LogRecord]) -> FeatureMatrix:
    config = FeaturesConfig(entities=["source_ip"], windows=["1m", "5m"])
    return PandasFeatureBuilder(config).build(records)


def test_attacker_row_is_flagged_and_most_unusual() -> None:
    matrix = _matrix(_normal() + _attack())
    result = IsolationForestDetector(random_state=1).detect(matrix)
    assert len(result.scores) == len(matrix.frame)
    flagged = [s for s in result.scores if s.is_anomaly]
    assert any(s.entity == ATTACKER for s in flagged)
    worst = min(result.scores, key=lambda s: s.anomaly_score)
    assert worst.entity == ATTACKER and worst.severity is Severity.CRITICAL


def test_non_anomalies_are_info_and_flag_fraction_near_contamination() -> None:
    result = IsolationForestDetector(contamination=0.05, random_state=1).detect(
        _matrix(_normal() + _attack())
    )
    assert all(s.severity is Severity.INFO for s in result.scores if not s.is_anomaly)
    fraction = sum(s.is_anomaly for s in result.scores) / len(result.scores)
    assert 0.02 < fraction < 0.10


def test_same_seed_same_output_and_row_alignment() -> None:
    matrix = _matrix(_normal() + _attack())
    a = IsolationForestDetector(random_state=7).detect(matrix).scores
    b = IsolationForestDetector(random_state=7).detect(matrix).scores
    assert [s.model_dump() for s in a] == [s.model_dump() for s in b]
    assert [s.entity for s in a] == matrix.frame["entity"].tolist()
    assert [s.event_row_id for s in a] == matrix.frame["event_row_id"].tolist()


def test_contamination_auto_runs() -> None:
    result = IsolationForestDetector(contamination="auto", random_state=1).detect(
        _matrix(_normal() + _attack())
    )
    assert any(s.is_anomaly for s in result.scores)


def test_too_few_rows_warns_and_flags_nothing() -> None:
    result = IsolationForestDetector().detect(_matrix(_normal(n_ips=2, minutes=2)))
    assert [f.code for f in result.findings] == ["too_few_rows"]
    assert not any(s.is_anomaly for s in result.scores)


def test_empty_matrix() -> None:
    result = IsolationForestDetector().detect(_matrix([]))
    assert result.scores == [] and [f.code for f in result.findings] == ["no_feature_rows"]


def test_baseline_mode_flags_attacker_and_new_entities() -> None:
    baseline = _matrix(_normal(seed=3))
    target = _matrix(_normal(n_ips=5, seed=4) + _attack())
    result = IsolationForestDetector(random_state=1).detect(target, baseline)
    assert result.findings == []
    assert len(result.scores) == len(target.frame)
    flagged = {s.entity for s in result.scores if s.is_anomaly}
    assert ATTACKER in flagged
    worst = min(result.scores, key=lambda s: s.anomaly_score)
    assert worst.entity == ATTACKER


def test_baseline_too_small_falls_back_with_warning() -> None:
    baseline = _matrix(_normal(n_ips=1, minutes=2))
    target = _matrix(_normal() + _attack())
    result = IsolationForestDetector(random_state=1).detect(target, baseline)
    assert [f.code for f in result.findings] == ["baseline_too_small"]
    assert any(s.is_anomaly for s in result.scores)


def test_incompatible_baseline_rejected() -> None:
    other = PandasFeatureBuilder(FeaturesConfig(entities=["source_ip"], windows=["1m"])).build(
        _normal()
    )
    with pytest.raises(ConfigError, match="different feature settings"):
        IsolationForestDetector().detect(_matrix(_normal()), other)
