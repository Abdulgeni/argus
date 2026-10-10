"""Severity assignment. A pure function so the rule is easy to test and to read."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from argus.core.models import Severity

_QUARTILES: tuple[Severity, ...] = (
    Severity.LOW,
    Severity.MEDIUM,
    Severity.HIGH,
    Severity.CRITICAL,
)


def assign_severity(
    anomaly_scores: NDArray[np.float64], is_anomaly: NDArray[np.bool_]
) -> list[Severity]:
    """Map raw decision_function scores to one of five severities."""
    if anomaly_scores.shape != is_anomaly.shape:
        raise ValueError("anomaly_scores and is_anomaly must have the same shape")
    result = [Severity.INFO] * len(anomaly_scores)
    flagged = np.flatnonzero(is_anomaly)
    m = flagged.size
    if m == 0:
        return result
    keys = -anomaly_scores[flagged]
    counts = np.searchsorted(np.sort(keys), keys, side="right")
    quartile = np.ceil(4 * counts / m).astype(int) - 1
    for position, q in zip(flagged.tolist(), quartile.tolist(), strict=True):
        result[position] = _QUARTILES[q]
    return result
