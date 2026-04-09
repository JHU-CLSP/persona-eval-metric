"""Log full LLM prompts and responses for reproducibility and analysis."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ResponseLogger:
    """Append-only JSONL logger for LLM interactions.

    Each entry records the metric, prompt, full response text, and
    the parsed result, along with metadata like query_index and labels.
    """

    def __init__(self, log_dir: str | Path):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = self._log_dir / f"llm_responses_{timestamp}.jsonl"
        self._file = open(self._path, "a", encoding="utf-8")
        logger.info("Logging LLM responses to %s", self._path)

    def log(
        self,
        metric: str,
        prompt: str,
        response: str,
        parsed_result: str | float | dict | None = None,
        **metadata,
    ):
        """Write one log entry."""
        entry = {
            "metric": metric,
            "prompt": prompt,
            "response": response,
            "parsed_result": parsed_result,
            **metadata,
        }
        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._file.flush()

    @property
    def path(self) -> Path:
        return self._path

    def close(self):
        self._file.close()
