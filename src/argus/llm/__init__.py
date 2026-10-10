"""AI reporting: AnomalyScore[] -> IncidentReport[] (LLM optional, template fallback)."""

from argus.llm.context import IncidentContext, build_contexts
from argus.llm.grouping import AnomalyGroup, group_anomalies
from argus.llm.summarizer import LLMSummarizer, SummaryResult, build_summarizer

__all__ = [
    "AnomalyGroup",
    "IncidentContext",
    "LLMSummarizer",
    "SummaryResult",
    "build_contexts",
    "build_summarizer",
    "group_anomalies",
]
