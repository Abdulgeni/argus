"""Group related anomalies: same entity, windows that overlap or touch."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from argus.core.models import AnomalyScore, Severity

GROUP_SPAN = timedelta(minutes=5)


@dataclass(frozen=True)
class AnomalyGroup:
    """Anomalous rows of one entity that form a single incident."""

    entity_type: str
    entity: str
    members: tuple[AnomalyScore, ...]

    @property
    def start(self) -> datetime:
        """Earliest ``window_end`` in the group."""
        return self.members[0].window_end

    @property
    def end(self) -> datetime:
        """Latest ``window_end`` in the group."""
        return self.members[-1].window_end

    @property
    def severity(self) -> Severity:
        """Highest member severity."""
        return max((m.severity for m in self.members), key=lambda s: s.rank)

    @property
    def peak(self) -> AnomalyScore:
        """Most unusual member (lowest score; ties go to the latest row)."""
        return min(self.members, key=lambda m: (m.anomaly_score, -m.event_row_id))


def group_anomalies(scores: list[AnomalyScore], span: timedelta = GROUP_SPAN) -> list[AnomalyGroup]:
    """Merge anomalous rows into groups."""
    flagged = sorted(
        (s for s in scores if s.is_anomaly),
        key=lambda s: (s.entity_type, s.entity, s.window_end, s.event_row_id),
    )
    groups: list[AnomalyGroup] = []
    current: list[AnomalyScore] = []
    for score in flagged:
        if current and _continues(current[-1], score, span):
            current.append(score)
            continue
        if current:
            groups.append(_make(current))
        current = [score]
    if current:
        groups.append(_make(current))
    groups.sort(
        key=lambda g: (-g.severity.rank, g.peak.anomaly_score, g.entity_type, g.entity, g.start)
    )
    return groups


def _continues(prev: AnomalyScore, nxt: AnomalyScore, span: timedelta) -> bool:
    same = (prev.entity_type, prev.entity) == (nxt.entity_type, nxt.entity)
    return same and nxt.window_end - prev.window_end <= span


def _make(rows: list[AnomalyScore]) -> AnomalyGroup:
    return AnomalyGroup(rows[0].entity_type, rows[0].entity, tuple(rows))
