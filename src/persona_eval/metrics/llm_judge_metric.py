"""LLM-as-judge metric with configurable prompts.

Uses the Prometheus prompt format by default: one rubric-based evaluation
per LLM call with ``[RESULT]`` tags. Makes a separate call for each
dimension (relevance, coherence, consistency, fluency).

Supports local vLLM and TogetherAI backends via the shared LLM client.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from persona_eval.metrics.base import BaseMetric, register_metric

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_DEFAULT_PROMPT_PATH = _PROMPTS_DIR / "llm_judge_default.txt"
_RUBRICS_DIR = _PROMPTS_DIR / "rubrics"

_SCORE_DIMENSIONS = ("relevance", "coherence", "consistency", "fluency")

# Matches "[RESULT] 4" or "[RESULT]4" with optional whitespace
_RESULT_RE = re.compile(r"\[RESULT\]\s*(\d)")


def _load_prompt_template(prompt_file: str | None = None) -> str:
    """Load a prompt template from a file path, falling back to the default."""
    path = Path(prompt_file) if prompt_file else _DEFAULT_PROMPT_PATH
    return path.read_text()


def _load_rubric(dimension: str) -> str:
    """Load the rubric text for a given dimension."""
    return (_RUBRICS_DIR / f"{dimension}.txt").read_text()


def _parse_prometheus_score(text: str) -> int | None:
    """Extract the first [RESULT] score from a Prometheus-style response."""
    match = _RESULT_RE.search(text)
    return int(match.group(1)) if match else None


@register_metric("llm_judge")
class LLMJudgeMetric(BaseMetric):
    """LLM-as-judge: uses an LLM to rate summaries on multiple dimensions.

    Makes one LLM call per dimension using the Prometheus prompt format.
    Each call evaluates a single rubric and returns a ``[RESULT] <score>``
    tag on a 1-5 scale.

    The prompt template is configurable via ``prompt_file``. It should contain
    ``{summary}``, ``{source}``, ``{dimension}``, and ``{rubric}`` placeholders.
    Rubrics are loaded from ``prompts/rubrics/<dimension>.txt``.
    """

    def __init__(
        self,
        provider: str = "vllm",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        prompt_file: str | None = None,
        **kwargs,
    ):
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._prompt_template = _load_prompt_template(prompt_file)
        self._rubrics = {dim: _load_rubric(dim) for dim in _SCORE_DIMENSIONS}
        self._client = None

    @property
    def name(self) -> str:
        return "LLM Judge"

    @property
    def is_reference_free(self) -> bool:
        return True

    def _load(self):
        if self._client is None:
            from persona_eval.llm_client import LLMClient

            self._client = LLMClient(
                provider=self._provider,
                model=self._model,
                api_key=self._api_key,
                base_url=self._base_url,
            )

    def _score_dimension(self, summary: str, source: str, dimension: str) -> float:
        """Score a single dimension via one LLM call."""
        prompt = self._prompt_template.format(
            summary=summary,
            source=source,
            dimension=dimension,
            rubric=self._rubrics[dimension],
        )
        response_text = self._client.generate(prompt)
        result = _parse_prometheus_score(response_text)
        if result is None:
            logger.warning("LLM judge: no [RESULT] tag found for %s", dimension)
            return float("nan")
        return float(result)

    def score(self, summary: str, source: str) -> dict[str, float]:
        self._load()

        scores = {}
        for dim in _SCORE_DIMENSIONS:
            try:
                scores[f"llm_judge_{dim}"] = self._score_dimension(summary, source, dim)
            except Exception:
                logger.warning("LLM judge call failed for %s, returning NaN", dim)
                scores[f"llm_judge_{dim}"] = float("nan")

        valid = [v for v in scores.values() if v == v]  # filter NaN
        scores["llm_judge_overall"] = sum(valid) / len(valid) if valid else float("nan")

        return scores
