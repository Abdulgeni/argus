"""On-disk response cache keyed by sha256(prompt) with a TTL."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path


def default_cache_dir(
    configured: Path | None = None,
    env: Mapping[str, str] | None = None,
    platform: str | None = None,
    home: Path | None = None,
) -> Path:
    """Resolve the cache directory."""
    environ = os.environ if env is None else env
    if configured is not None:
        return configured
    if environ.get("ARGUS_CACHE_DIR"):
        return Path(environ["ARGUS_CACHE_DIR"])
    if (platform or sys.platform).startswith("win") and environ.get("LOCALAPPDATA"):
        return Path(environ["LOCALAPPDATA"]) / "argus"
    return (home or Path.home()) / ".cache" / "argus"


def cache_key(prompt: str, namespace: str = "") -> str:
    """sha256 over the (already scrubbed) prompt."""
    return hashlib.sha256(f"{namespace}\0{prompt}".encode()).hexdigest()


class DiskCache:
    """Stores validated response JSON as ``<dir>/<aa>/<key>.json`` with a creation time."""

    def __init__(
        self,
        directory: Path,
        ttl_hours: float = 168.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._dir = directory
        self._ttl_s = ttl_hours * 3600.0
        self._clock = clock

    def _path(self, key: str) -> Path:
        return self._dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | None:
        """Return the cached payload, or None if missing, expired or unreadable."""
        path = self._path(key)
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            created = float(entry["created"])
            payload = entry["payload"]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if not isinstance(payload, str) or self._clock() - created > self._ttl_s:
            return None
        return payload

    def put(self, key: str, payload: str) -> None:
        """Atomically write an entry; raises OSError if the cache directory is not writable."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = json.dumps({"created": self._clock(), "payload": payload})
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(entry)
            os.replace(tmp, path)
        except OSError:
            Path(tmp).unlink(missing_ok=True)
            raise
