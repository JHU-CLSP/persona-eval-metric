"""Disk-based cache for LLM perturbation outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class PerturbationCache:
    """File-based cache keyed by (test, sample, level, model, prompt_hash).

    Each entry is a small JSON file stored under ``{cache_dir}/robustness/``.
    The filename is a truncated SHA-256 of the key fields so that changing
    the model or the prompt template automatically invalidates the cache.
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir) / "robustness"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash_key(
        test_name: str,
        sample_id: str,
        level: int | str,
        model: str,
        prompt_hash: str,
    ) -> str:
        raw = f"{test_name}|{sample_id}|{level}|{model}|{prompt_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def _path(self, key_hash: str) -> Path:
        return self.cache_dir / f"{key_hash}.json"

    def get(
        self,
        test_name: str,
        sample_id: str,
        level: int | str,
        model: str,
        prompt_hash: str,
    ) -> str | None:
        """Return cached LLM output text, or ``None`` on miss."""
        h = self._hash_key(test_name, sample_id, level, model, prompt_hash)
        path = self._path(h)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return data.get("output")

    def put(
        self,
        test_name: str,
        sample_id: str,
        level: int | str,
        model: str,
        prompt_hash: str,
        text: str,
    ) -> None:
        """Write an LLM output to the cache."""
        h = self._hash_key(test_name, sample_id, level, model, prompt_hash)
        payload = {
            "test_name": test_name,
            "sample_id": sample_id,
            "level": level,
            "model": model,
            "output": text,
        }
        with open(self._path(h), "w") as f:
            json.dump(payload, f)
