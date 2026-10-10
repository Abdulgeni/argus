"""Anomaly detection: FeatureMatrix -> AnomalyScore[]."""

from argus.detection.detector import IsolationForestDetector
from argus.detection.severity import assign_severity

__all__ = ["IsolationForestDetector", "assign_severity"]
