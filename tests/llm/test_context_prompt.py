from datetime import timedelta

from argus.core.models import AnomalyScore, Severity
from argus.llm import group_anomalies
from argus.llm.context import MAX_LINES, split_feature
from argus.llm.prompt import PERSONA, build_prompt, format_number
from argus.llm.template import template_response

from ..features.helpers import T0
from .helpers import attacker_context


def _score(
    entity: str, minutes: float, score: float, sev: Severity, flag: bool = True, rid: int = 0
):
    return AnomalyScore(
        entity_type="source_ip",
        entity=entity,
        window_end=T0 + timedelta(minutes=minutes),
        event_row_id=rid,
        anomaly_score=score,
        is_anomaly=flag,
        severity=sev,
    )


def test_grouping_chains_close_windows_and_splits_far_ones() -> None:
    scores = [
        _score("a", 0, -0.1, Severity.LOW, rid=1),
        _score("a", 4, -0.3, Severity.HIGH, rid=2),
        _score("a", 9, -0.2, Severity.MEDIUM, rid=3),
        _score("a", 30, -0.2, Severity.MEDIUM, rid=4),
        _score("b", 1, -0.5, Severity.CRITICAL, rid=5),
        _score("b", 2, 0.4, Severity.INFO, flag=False, rid=6),
    ]
    groups = group_anomalies(scores)
    assert [(g.entity, len(g.members)) for g in groups] == [("b", 1), ("a", 3), ("a", 1)]
    first_a = groups[1]
    assert first_a.severity is Severity.HIGH
    assert first_a.peak.event_row_id == 2
    assert first_a.start == T0 and first_a.end == T0 + timedelta(minutes=9)


def test_grouping_ignores_non_anomalies_and_empty() -> None:
    assert group_anomalies([]) == []
    assert group_anomalies([_score("a", 0, 0.3, Severity.INFO, flag=False)]) == []


def test_context_lines_are_scrubbed_newest_first_and_capped() -> None:
    ctx = attacker_context()
    assert len(ctx.log_lines) == MAX_LINES
    assert ctx.total_lines >= MAX_LINES
    joined = "\n".join(ctx.log_lines)
    assert "hunter2" not in joined and "abcdefgh" not in joined
    assert "[REDACTED]" in joined
    numbers = [int(line.split("&n=")[1].split('"')[0]) for line in ctx.log_lines]
    assert numbers == sorted(numbers, reverse=True)


def test_context_deviations_and_features() -> None:
    ctx = attacker_context()
    assert ctx.deviations and len(ctx.deviations) <= 10
    devs = [abs(d.deviation) for d in ctx.deviations]
    assert devs == sorted(devs, reverse=True)
    by_name = {(d.feature, d.window): d for d in ctx.deviations}
    assert ("error_rate", "1m") in by_name or ("failed_auth_count", "1m") in by_name
    assert ctx.features["request_count_1m"] > 20
    assert ctx.window_start < ctx.window_end


def test_split_feature() -> None:
    assert split_feature("error_rate_5m", ["1m", "5m"]) == ("error_rate", "5m")
    assert split_feature("hour_of_day", ["1m", "5m"]) == ("hour_of_day", None)


def test_prompt_contains_required_parts() -> None:
    ctx = attacker_context()
    prompt = build_prompt(ctx)
    assert PERSONA in prompt
    assert "9.9.9.9" in prompt
    assert "(window: 1m, baseline:" in prompt or "(window: 5m, baseline:" in prompt
    assert "z-score:" in prompt
    for field in ("summary", "suspected_category", "confidence", "evidence", "recommended_action"):
        assert f'"{field}"' in prompt
    assert "brute_force | scanner | data_exfiltration | dos" in prompt
    assert "untrusted" in prompt
    assert "hunter2" not in prompt and "abcdefgh" not in prompt
    assert prompt.count("POST /login") == MAX_LINES


def test_prompt_is_deterministic() -> None:
    assert build_prompt(attacker_context()) == build_prompt(attacker_context())


def test_format_number() -> None:
    assert format_number(847.0) == "847"
    assert format_number(0.95) == "0.95"
    assert format_number(0.123456) == "0.1235"


def test_template_is_deterministic_factual_and_uncategorized() -> None:
    ctx = attacker_context()
    one, two = template_response(ctx), template_response(ctx)
    assert one == two
    assert one.suspected_category == "unknown" and one.confidence == 0.0
    assert "9.9.9.9" in one.summary and "vs baseline" in one.summary
    assert one.evidence and one.recommended_action
