"""Baseline-mode helpers: entity novelty and baseline/target compatibility."""

from __future__ import annotations

from argus.core.errors import ConfigError
from argus.core.matrix import FeatureMatrix

NEW_ENTITY_COLUMN = "is_new_entity"


def known_entities(baseline: FeatureMatrix) -> dict[str, frozenset[str]]:
    """Entities seen in the baseline, per entity type."""
    frame = baseline.frame
    result: dict[str, frozenset[str]] = {}
    for entity_type, group in frame.groupby("entity_type", sort=True):
        result[str(entity_type)] = frozenset(str(e) for e in group["entity"])
    return result


def ensure_compatible(target: FeatureMatrix, baseline: FeatureMatrix) -> None:
    """Raise ConfigError unless both matrices were built with the same feature schema."""
    if target.feature_columns != baseline.feature_columns:
        raise ConfigError(
            "baseline and target were built with different feature settings "
            "(windows/entities must match)"
        )
