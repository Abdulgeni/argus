"""Pydantic schema for the LLM's JSON answer, and tolerant parsing of the raw reply."""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from argus.core.models import Category

OUTPUT_FIELDS = "summary, suspected_category, confidence, evidence, recommended_action"


class InvalidResponse(ValueError):
    """The model's reply was not valid JSON matching the schema."""


class LLMResponse(BaseModel):
    """Required structured output of one incident summary."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(min_length=1, max_length=1000)
    suspected_category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(max_length=20)
    recommended_action: str = Field(min_length=1, max_length=500)


def _extract_json_object(text: str) -> str:
    """Strip markdown fences / chatter around the first {...} block."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise InvalidResponse("no JSON object found in reply")
    return text[start : end + 1]


def parse_response(text: str) -> LLMResponse:
    """Parse and validate a raw model reply; raises InvalidResponse on any problem."""
    try:
        data = json.loads(_extract_json_object(text))
    except json.JSONDecodeError as exc:
        raise InvalidResponse(f"invalid JSON: {exc.msg}") from None
    if not isinstance(data, dict):
        raise InvalidResponse("JSON reply is not an object")
    try:
        return LLMResponse.model_validate(data)
    except ValidationError as exc:
        fields = ", ".join(sorted({str(e["loc"][0]) for e in exc.errors() if e["loc"]}))
        raise InvalidResponse(
            f"schema validation failed for: {fields or 'unknown field'}"
        ) from None
