"""File-based JSON cache shared by metric and perturbation subsystems.

Each subclass stores one JSON file per entry under ``{cache_dir}/{subdir}/``.
The filename is a truncated SHA-256 of the key fields so that changing any
input automatically produces a new entry and leaves stale ones untouched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_persona(persona_kwargs: dict | None) -> str:
    if not persona_kwargs:
        return ""
    canonical = json.dumps(persona_kwargs, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def config_hash(config: dict) -> str:
    """Deterministic hash of a metric's cache-affecting configuration."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON to ``path`` via a temp file + rename to avoid partial writes."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    tmp.replace(path)


class JsonFileCache:
    """Base class: file-per-entry JSON cache with hit/miss bookkeeping.

    Subclasses provide a ``_subdir`` class attribute and build cache keys
    via :meth:`_hash_key`. When ``enabled`` is False, ``_get`` returns
    ``None`` and ``_put`` is a no-op so call sites stay clean.
    """

    _subdir: str = "cache"

    def __init__(self, cache_dir: str | Path, enabled: bool = True):
        self.enabled = enabled
        self.cache_dir = Path(cache_dir) / self._subdir
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _hash_key(*parts: str) -> str:
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def _path(self, key_hash: str) -> Path:
        return self.cache_dir / f"{key_hash}.json"

    def _get(self, key_hash: str, payload_field: str) -> dict | str | None:
        if not self.enabled:
            return None
        path = self._path(key_hash)
        if not path.exists():
            self.misses += 1
            return None
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            self.misses += 1
            return None
        self.hits += 1
        return data.get(payload_field)

    def _put(self, key_hash: str, payload: dict) -> None:
        if not self.enabled:
            return
        atomic_write_json(self._path(key_hash), payload)

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0

    def clear(self) -> int:
        """Delete all cache entries. Returns the number of files removed."""
        if not self.cache_dir.exists():
            return 0
        n = 0
        for f in self.cache_dir.glob("*.json"):
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
        return n
