"""LLM-as-judge metric with configurable prompts.

Uses the Prometheus prompt format by default: rubric-based evaluation
with ``[RESULT]`` tags. Supports local vLLM and TogetherAI backends
via the shared LLM client.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from persona_eval.metrics.base import BaseMetric, register_metric

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "llm_judge_default.txt"

_SCORE_DIMENSIONS = ("relevance", "coherence", "consistency", "fluency")

# Matches "[RESULT] 4" or "[RESULT]4" with optional whitespace
_RESULT_RE = re.compile(r"\[RESULT\]\s*(\d)")


def _load_prompt_template(prompt_file: str | None = None) -> str:
    """Load a prompt template from a file path, falling back to the default."""
    path = Path(prompt_file) if prompt_file else _DEFAULT_PROMPT_PATH
    return path.read_text()


def _parse_prometheus_scores(text: str) -> list[int]:
    """Extract all [RESULT] scores from a Prometheus-style response."""
    return [int(m.group(1)) for m in _RESULT_RE.finditer(text)]


@register_metric("llm_judge")
class LLMJudgeMetric(BaseMetric):
    """LLM-as-judge: uses an LLM to rate summaries on multiple dimensions.

    The prompt template is configurable via ``prompt_file``. It should contain
    ``{summary}`` and ``{source}`` placeholders.

    By default, the metric uses the Prometheus prompt format with rubric-based
    evaluation. The LLM is expected to return feedback with ``[RESULT] <score>``
    tags for each dimension (relevance, coherence, consistency, fluency) on a
    1-5 scale.
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

    def score(self, summary: str, source: str) -> dict[str, float]:
        self._load()
        prompt = self._prompt_template.format(summary=summary, source=source)

        try:
            response_text = self._client.generate(prompt)
            result_scores = _parse_prometheus_scores(response_text)
        except Exception:
            logger.warning("LLM judge failed to parse response, returning NaN scores")
            return {f"llm_judge_{d}": float("nan") for d in _SCORE_DIMENSIONS} | {
                "llm_judge_overall": float("nan")
            }

        scores = {}
        for i, dim in enumerate(_SCORE_DIMENSIONS):
            if i < len(result_scores):
                scores[f"llm_judge_{dim}"] = float(result_scores[i])
            else:
                logger.warning("LLM judge missing [RESULT] for %s", dim)
                scores[f"llm_judge_{dim}"] = float("nan")

        valid = [v for v in scores.values() if v == v]  # filter NaN
        scores["llm_judge_overall"] = sum(valid) / len(valid) if valid else float("nan")

        return scores
