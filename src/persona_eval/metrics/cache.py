"""Disk-based cache for metric computation results.

Caches the output of ``metric.score()`` and ``metric.score_pair()`` keyed
by ``(metric_name, metric_config, summary, source, persona)``. Hits are
reused across runs and across modes (annotations vs. dataset-based
robustness), so the same ``(summary, source)`` pair evaluated in either
path reuses the stored result.
"""

from __future__ import annotations

from persona_eval.core.cache import (
    JsonFileCache,
    config_hash,  # re-exported for backward compatibility
    hash_persona,
    hash_text,
)

__all__ = ["MetricCache", "config_hash"]


class MetricCache(JsonFileCache):
    """File-based cache for metric results (both single and pairwise)."""

    _subdir = "metrics"

    def _key_single(
        self, metric_name: str, config_hash: str, summary: str, source: str,
        persona_kwargs: dict | None,
    ) -> str:
        return self._hash_key(
            "single", metric_name, config_hash,
            hash_text(summary), hash_text(source), hash_persona(persona_kwargs),
        )

    def _key_pair(
        self, metric_name: str, config_hash: str, summary_a: str, summary_b: str,
        source: str, persona_kwargs: dict | None,
    ) -> str:
        return self._hash_key(
            "pair", metric_name, config_hash,
            hash_text(summary_a), hash_text(summary_b),
            hash_text(source), hash_persona(persona_kwargs),
        )

    # -- single-score API -------------------------------------------------

    def get(
        self, metric_name: str, config_hash: str, summary: str, source: str,
        persona_kwargs: dict | None = None,
    ) -> dict | None:
        key = self._key_single(metric_name, config_hash, summary, source, persona_kwargs)
        return self._get(key, "scores")

    def put(
        self, metric_name: str, config_hash: str, summary: str, source: str,
        persona_kwargs: dict | None, scores: dict,
    ) -> None:
        key = self._key_single(metric_name, config_hash, summary, source, persona_kwargs)
        self._put(key, {
            "kind": "single",
            "metric_name": metric_name,
            "config_hash": config_hash,
            "summary_sha256": hash_text(summary),
            "source_sha256": hash_text(source),
            "persona_sha256": hash_persona(persona_kwargs),
            "scores": scores,
        })

    # -- pairwise API -----------------------------------------------------

    def get_pair(
        self, metric_name: str, config_hash: str, summary_a: str, summary_b: str,
        source: str, persona_kwargs: dict | None = None,
    ) -> dict | None:
        key = self._key_pair(
            metric_name, config_hash, summary_a, summary_b, source, persona_kwargs,
        )
        return self._get(key, "prefs")

    def put_pair(
        self, metric_name: str, config_hash: str, summary_a: str, summary_b: str,
        source: str, persona_kwargs: dict | None, prefs: dict,
    ) -> None:
        key = self._key_pair(
            metric_name, config_hash, summary_a, summary_b, source, persona_kwargs,
        )
        self._put(key, {
            "kind": "pair",
            "metric_name": metric_name,
            "config_hash": config_hash,
            "summary_a_sha256": hash_text(summary_a),
            "summary_b_sha256": hash_text(summary_b),
            "source_sha256": hash_text(source),
            "persona_sha256": hash_persona(persona_kwargs),
            "prefs": prefs,
        })
