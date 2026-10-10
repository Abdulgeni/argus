"""IsolationForest detector: StandardScaler (fit on train) + one forest per entity type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from argus.core.matrix import DetectionResult, FeatureMatrix
from argus.core.models import AnomalyScore, Finding
from argus.detection.baseline import NEW_ENTITY_COLUMN, ensure_compatible, known_entities
from argus.detection.severity import assign_severity

MIN_TRAIN_ROWS = 8
Floats = NDArray[np.float64]


@dataclass(frozen=True)
class _TypeResult:
    scores: Floats
    flags: NDArray[np.bool_]


class IsolationForestDetector:
    """Scores every matrix row; ``contamination`` is a float in (0, 0.5] or ``"auto"``."""

    def __init__(
        self,
        contamination: float | Literal["auto"] = 0.05,
        random_state: int = 42,
        n_estimators: int = 200,
    ) -> None:
        self._contamination = contamination
        self._random_state = random_state
        self._n_estimators = n_estimators

    def detect(
        self, target: FeatureMatrix, baseline: FeatureMatrix | None = None
    ) -> DetectionResult:
        """Score ``target``; if ``baseline`` is given, train on it instead of on ``target``."""
        frame = target.frame
        if frame.empty:
            warning = Finding(
                level="warning", code="no_feature_rows", message="no feature rows to score"
            )
            return DetectionResult(scores=[], findings=[warning])
        if baseline is not None:
            ensure_compatible(target, baseline)
        scores = np.zeros(len(frame), dtype=np.float64)
        flags = np.zeros(len(frame), dtype=np.bool_)
        findings: list[Finding] = []
        types = frame["entity_type"].to_numpy()
        for entity_type in dict.fromkeys(types.tolist()):
            rows = np.flatnonzero(types == entity_type)
            result = self._score_type(str(entity_type), target, baseline, rows, findings)
            scores[rows], flags[rows] = result.scores, result.flags
        return DetectionResult(scores=self._to_scores(target, scores, flags), findings=findings)

    def _score_type(
        self,
        entity_type: str,
        target: FeatureMatrix,
        baseline: FeatureMatrix | None,
        rows: NDArray[np.intp],
        findings: list[Finding],
    ) -> _TypeResult:
        cols = target.feature_columns
        known = known_entities(baseline).get(entity_type) if baseline is not None else None
        target_x = _design(target, rows, cols, known)
        train_x = target_x
        if baseline is not None:
            base_rows = np.flatnonzero((baseline.frame["entity_type"] == entity_type).to_numpy())
            if base_rows.size >= MIN_TRAIN_ROWS:
                train_x = _design(baseline, base_rows, cols, known)
            else:
                findings.append(_baseline_fallback(entity_type, int(base_rows.size)))
                target_x = _design(target, rows, cols, None)
                train_x = target_x
        if len(train_x) < MIN_TRAIN_ROWS:
            findings.append(_too_few_rows(entity_type, len(train_x)))
            return _TypeResult(np.zeros(len(rows)), np.zeros(len(rows), dtype=np.bool_))
        return self._fit_score(train_x, target_x)

    def _fit_score(self, train_x: Floats, target_x: Floats) -> _TypeResult:
        scaler = StandardScaler().fit(train_x)
        model = IsolationForest(
            n_estimators=self._n_estimators,
            contamination=self._contamination,
            random_state=self._random_state,
            n_jobs=1,
        ).fit(scaler.transform(train_x))
        scaled = scaler.transform(target_x)
        scores = np.asarray(model.decision_function(scaled), dtype=np.float64)
        flags = np.asarray(model.predict(scaled) == -1, dtype=np.bool_)
        return _TypeResult(scores, flags)

    @staticmethod
    def _to_scores(
        target: FeatureMatrix, scores: Floats, flags: NDArray[np.bool_]
    ) -> list[AnomalyScore]:
        severities = assign_severity(scores, flags)
        frame = target.frame
        return [
            AnomalyScore(
                entity_type=str(et),
                entity=str(ent),
                window_end=we.to_pydatetime(),
                event_row_id=int(rid),
                anomaly_score=float(sc),
                is_anomaly=bool(fl),
                severity=sev,
            )
            for et, ent, we, rid, sc, fl, sev in zip(
                frame["entity_type"],
                frame["entity"],
                frame["window_end"],
                frame["event_row_id"],
                scores,
                flags,
                severities,
                strict=True,
            )
        ]


def _design(
    matrix: FeatureMatrix,
    rows: NDArray[np.intp],
    cols: list[str],
    known: frozenset[str] | None,
) -> Floats:
    """Feature columns plus ``is_new_entity`` for the selected rows."""
    part = matrix.frame.iloc[rows]
    values = part[cols].to_numpy(dtype=np.float64)
    if known is None:
        new = np.zeros(len(part), dtype=np.float64)
    else:
        new = np.array([0.0 if str(e) in known else 1.0 for e in part["entity"]])
    return np.column_stack([values, new])


def _baseline_fallback(entity_type: str, n_rows: int) -> Finding:
    return Finding(
        level="warning",
        code="baseline_too_small",
        message=(
            f"baseline has only {n_rows} '{entity_type}' rows (need {MIN_TRAIN_ROWS}); "
            "fit on the target instead for this entity type"
        ),
    )


def _too_few_rows(entity_type: str, n_rows: int) -> Finding:
    return Finding(
        level="warning",
        code="too_few_rows",
        message=(
            f"only {n_rows} '{entity_type}' rows (need {MIN_TRAIN_ROWS}); "
            "no anomaly detection was run for this entity type"
        ),
    )


__all__ = ["NEW_ENTITY_COLUMN", "IsolationForestDetector"]
