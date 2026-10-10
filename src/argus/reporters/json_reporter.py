"""JSON reporter: the full RunReport, features and warnings included."""

from __future__ import annotations

import json

from argus.core.models import RunReport


class JsonReporter:
    """Stable, indented JSON (field order follows the model, so output is deterministic)."""

    def render(self, run: RunReport) -> str:
        return json.dumps(run.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
