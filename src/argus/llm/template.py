"""Deterministic fallback summary used when the LLM is off, unavailable, or fails."""

from __future__ import annotations

from argus.llm.context import IncidentContext
from argus.llm.prompt import format_deviation, format_number
from argus.llm.schema import LLMResponse

MAX_SUMMARY_FEATURES = 3
MAX_EVIDENCE = 6
RECOMMENDED_ACTION = (
    "Review the sample log lines for this entity and confirm whether the activity is expected."
)


def template_response(context: IncidentContext) -> LLMResponse:
    """Factual summary from feature deviations. Category is always 'unknown', confidence 0.0."""
    group = context.group
    when = f"{context.window_start:%Y-%m-%d %H:%M:%S} to {context.window_end:%H:%M:%S} UTC"
    lead = (
        f"{group.severity.value}: unusual activity for {group.entity_type} {group.entity} "
        f"({when}), {len(group.members)} anomalous window(s)."
    )
    top = context.deviations[:MAX_SUMMARY_FEATURES]
    if top:
        detail = "; ".join(
            f"{d.feature}{f' ({d.window})' if d.window else ''} {format_number(d.value)} "
            f"vs baseline {format_number(d.baseline)}"
            for d in top
        )
        summary = f"{lead} Largest deviations: {detail}."
    else:
        summary = f"{lead} No single feature deviates strongly from the baseline."
    evidence = [format_deviation(d, context) for d in context.deviations[:MAX_EVIDENCE]]
    evidence.append(f"{context.total_lines} log line(s) from this entity in the period")
    return LLMResponse(
        summary=summary,
        suspected_category="unknown",
        confidence=0.0,
        evidence=evidence,
        recommended_action=RECOMMENDED_ACTION,
    )
