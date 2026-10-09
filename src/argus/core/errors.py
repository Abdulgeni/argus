"""Exception hierarchy. Every ArgusError maps to CLI exit code 2."""

from __future__ import annotations


class ArgusError(Exception):
    """Base class for tool errors (usage, config, input)."""

    exit_code = 2


class ConfigError(ArgusError):
    """Invalid or inconsistent configuration."""


class InputError(ArgusError):
    """Input file missing, unreadable, or corrupt."""
