"""Disk-based cache for LLM perturbation outputs."""

from __future__ import annotations

from persona_eval.core.cache import JsonFileCache


class PerturbationCache(JsonFileCache):
    """Keyed by ``(test_name, sample_id, level, model, prompt_hash)``."""

    _subdir = "robustness"

    def __init__(self, cache_dir):
        super().__init__(cache_dir, enabled=True)

    def get(
        self, test_name: str, sample_id: str, level: int | str,
        model: str, prompt_hash: str,
    ) -> str | None:
        key = self._hash_key(test_name, sample_id, str(level), model, prompt_hash)
        return self._get(key, "output")

    def put(
        self, test_name: str, sample_id: str, level: int | str,
        model: str, prompt_hash: str, text: str,
    ) -> None:
        key = self._hash_key(test_name, sample_id, str(level), model, prompt_hash)
        self._put(key, {
            "test_name": test_name,
            "sample_id": sample_id,
            "level": level,
            "model": model,
            "output": text,
        })
