"""Small numeric helpers used inside pandas ``rolling.apply`` (inputs are integer codes).

Categorical values (paths, user agents, status codes) are factorized to non-negative
float codes; a null value is encoded as ``-1`` and ignored by every helper here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _valid(codes: NDArray[np.float64]) -> NDArray[np.float64]:
    return codes[codes >= 0]


def shannon_entropy(codes: NDArray[np.float64]) -> float:
    """Shannon entropy in bits of the non-null codes; 0.0 if there are none or only one value."""
    valid = _valid(codes)
    if valid.size == 0:
        return 0.0
    _, counts = np.unique(valid, return_counts=True)
    p = counts / valid.size
    return abs(float(-(p * np.log2(p)).sum()))


def count_unique(codes: NDArray[np.float64]) -> float:
    """Number of distinct non-null codes."""
    return float(np.unique(_valid(codes)).size)
