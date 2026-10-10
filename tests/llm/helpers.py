"""Shared builders for LLM-layer tests: records -> matrix -> scores -> incident contexts."""

from __future__ import annotations

from datetime import timedelta

from argus.core.models import LogRecord
from argus.detection import IsolationForestDetector
from argus.llm import IncidentContext, build_contexts, group_anomalies

from ..detection.test_detector import ATTACKER, _matrix, _normal
from ..features.helpers import T0


def attack_records(ip: str = ATTACKER, start: float = 300.0, n: int = 80) -> list[LogRecord]:
    """Brute-force-like burst whose raw lines contain secrets that must be scrubbed."""
    return [
        LogRecord(
            timestamp=T0 + timedelta(seconds=start + i * 0.5),
            source_ip=ip,
            method="POST",
            path="/login",
            status=401,
            bytes_sent=512,
            user_agent="curl",
            raw=f'{ip} "POST /login?password=hunter2&n={i}" 401 Authorization: Bearer tok{i}abcdefgh',
        )
        for i in range(n)
    ]


def make_contexts() -> list[IncidentContext]:
    """Contexts for a run with one clear attacker among normal traffic."""
    records = _normal() + attack_records()
    matrix = _matrix(records)
    result = IsolationForestDetector(random_state=1).detect(matrix)
    groups = group_anomalies(result.scores)
    return build_contexts(groups, matrix, result.scores, records)


def attacker_context() -> IncidentContext:
    return next(c for c in make_contexts() if c.group.entity == ATTACKER)
