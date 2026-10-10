"""Local secret scrubbing. Everything sent to an LLM (and everything cached) goes through here."""

from __future__ import annotations

import re

REDACTED = "[REDACTED]"

_SECRET_KEYS = (
    r"password|passwd|pwd|passphrase|secret|client[_-]?secret|api[_-]?key|apikey|"
    r"access[_-]?token|refresh[_-]?token|id[_-]?token|auth[_-]?token|token|"
    r"session[_-]?id|sessionid|private[_-]?key|credentials?"
)

_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(?i)(\bauthorization\b[\"']?\s*[:=]\s*[\"']?)[^\r\n\"']+"),
        rf"\1{REDACTED}",
    ),
    (re.compile(r"(?i)(\b(?:set-)?cookie\b[\"']?\s*[:=]\s*[\"']?)[^\r\n\"']+"), rf"\1{REDACTED}"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"), f"Bearer {REDACTED}"),
    (re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@"), rf"\1{REDACTED}@"),
    (
        re.compile(
            rf"(?i)([\"']?\b(?:{_SECRET_KEYS})\b[\"']?\s*[:=]\s*)"
            r"(\"[^\"]*\"|'[^']*'|[^\s&,;\"']+)"
        ),
        rf"\1{REDACTED}",
    ),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]*"), REDACTED),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), REDACTED),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"), REDACTED),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), REDACTED),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"), REDACTED),
)


def scrub(text: str) -> str:
    """Return ``text`` with credentials replaced by ``[REDACTED]``."""
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text
