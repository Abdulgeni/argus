import numpy as np
import pytest

from argus.core.models import Severity
from argus.detection.severity import assign_severity

S = Severity


def _sev(scores: list[float], flags: list[bool]) -> list[Severity]:
    return assign_severity(np.array(scores), np.array(flags))


def test_no_anomalies_all_info() -> None:
    assert _sev([0.1, 0.2], [False, False]) == [S.INFO, S.INFO]
    assert _sev([], []) == []


def test_single_anomaly_is_critical() -> None:
    assert _sev([0.2, -0.3], [False, True]) == [S.INFO, S.CRITICAL]


def test_two_anomalies_medium_and_critical() -> None:
    assert _sev([-0.1, -0.5], [True, True]) == [S.MEDIUM, S.CRITICAL]


def test_four_anomalies_one_per_level() -> None:
    result = _sev([-0.4, -0.1, -0.3, -0.2, 0.5], [True, True, True, True, False])
    assert result == [S.CRITICAL, S.LOW, S.HIGH, S.MEDIUM, S.INFO]


def test_eight_anomalies_two_per_quartile() -> None:
    scores = [-(i + 1) / 10 for i in range(8)]
    result = _sev(scores, [True] * 8)
    assert result == [S.LOW, S.LOW, S.MEDIUM, S.MEDIUM, S.HIGH, S.HIGH, S.CRITICAL, S.CRITICAL]


def test_ties_share_a_level() -> None:
    result = _sev([-0.2, -0.2, -0.9, -0.1], [True] * 4)
    assert result[0] == result[1] and result[2] is S.CRITICAL and result[3] is S.LOW


def test_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        assign_severity(np.zeros(2), np.zeros(3, dtype=bool))
