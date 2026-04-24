"""Disk-based cache for metric computation results.

Caches the output of ``metric.score()`` and ``metric.score_pair()`` keyed
by ``(metric_name, metric_config, summary, source, persona)``. Hits are
reused across runs and across modes (annotations vs. dataset-based
robustness), so the same ``(summary, source)`` pair evaluated in either
path reuses the stored result.

Follows the same file-per-entry pattern as ``PerturbationCache`` and
``OpenAlexClient``: one JSON file per entry under
``{cache_dir}/metrics/{sha256[:32]}.json``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_persona(persona_kwargs: dict | None) -> str:
    if not persona_kwargs:
        return ""
    canonical = json.dumps(persona_kwargs, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def config_hash(config: dict) -> str:
    """Deterministic hash of a metric's cache-affecting configuration."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class MetricCache:
    """File-based cache for metric results.

    Each entry is a small JSON file stored under ``{cache_dir}/metrics/``.
    The filename is a truncated SHA-256 of all cache-key fields so that
    changing any of them (config, inputs, persona) automatically produces
    a new entry and leaves stale ones untouched.

    When ``enabled`` is False, ``get`` always returns ``None`` and
    ``put`` is a no-op. This keeps call sites clean.
    """

    def __init__(self, cache_dir: str | Path, enabled: bool = True):
        self.enabled = enabled
        self.cache_dir = Path(cache_dir) / "metrics"
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    # -- key construction -------------------------------------------------

    @staticmethod
    def _key_single(
        metric_name: str,
        config_hash: str,
        summary: str,
        source: str,
        persona_kwargs: dict | None,
    ) -> str:
        raw = "|".join([
            "single",
            metric_name,
            config_hash,
            _hash_text(summary),
            _hash_text(source),
            _hash_persona(persona_kwargs),
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _key_pair(
        metric_name: str,
        config_hash: str,
        summary_a: str,
        summary_b: str,
        source: str,
        persona_kwargs: dict | None,
    ) -> str:
        raw = "|".join([
            "pair",
            metric_name,
            config_hash,
            _hash_text(summary_a),
            _hash_text(summary_b),
            _hash_text(source),
            _hash_persona(persona_kwargs),
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def _path(self, key_hash: str) -> Path:
        return self.cache_dir / f"{key_hash}.json"

    # -- single-score API -------------------------------------------------

    def get(
        self,
        metric_name: str,
        config_hash: str,
        summary: str,
        source: str,
        persona_kwargs: dict | None = None,
    ) -> dict | None:
        if not self.enabled:
            return None
        key = self._key_single(metric_name, config_hash, summary, source, persona_kwargs)
        path = self._path(key)
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
        return data.get("scores")

    def put(
        self,
        metric_name: str,
        config_hash: str,
        summary: str,
        source: str,
        persona_kwargs: dict | None,
        scores: dict,
    ) -> None:
        if not self.enabled:
            return
        key = self._key_single(metric_name, config_hash, summary, source, persona_kwargs)
        payload = {
            "kind": "single",
            "metric_name": metric_name,
            "config_hash": config_hash,
            "summary_sha256": _hash_text(summary),
            "source_sha256": _hash_text(source),
            "persona_sha256": _hash_persona(persona_kwargs),
            "scores": scores,
        }
        _atomic_write_json(self._path(key), payload)

    # -- pairwise API -----------------------------------------------------

    def get_pair(
        self,
        metric_name: str,
        config_hash: str,
        summary_a: str,
        summary_b: str,
        source: str,
        persona_kwargs: dict | None = None,
    ) -> dict | None:
        if not self.enabled:
            return None
        key = self._key_pair(
            metric_name, config_hash, summary_a, summary_b, source, persona_kwargs,
        )
        path = self._path(key)
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
        return data.get("prefs")

    def put_pair(
        self,
        metric_name: str,
        config_hash: str,
        summary_a: str,
        summary_b: str,
        source: str,
        persona_kwargs: dict | None,
        prefs: dict,
    ) -> None:
        if not self.enabled:
            return
        key = self._key_pair(
            metric_name, config_hash, summary_a, summary_b, source, persona_kwargs,
        )
        payload = {
            "kind": "pair",
            "metric_name": metric_name,
            "config_hash": config_hash,
            "summary_a_sha256": _hash_text(summary_a),
            "summary_b_sha256": _hash_text(summary_b),
            "source_sha256": _hash_text(source),
            "persona_sha256": _hash_persona(persona_kwargs),
            "prefs": prefs,
        }
        _atomic_write_json(self._path(key), payload)

    # -- bookkeeping ------------------------------------------------------

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


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON to ``path`` via a temp file + rename to avoid partial writes."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    tmp.replace(path)
