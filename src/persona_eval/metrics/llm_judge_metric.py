"""LLM-as-judge metric with configurable prompts.

Uses the Prometheus prompt format by default: one rubric-based evaluation
per LLM call with ``[RESULT]`` tags. Makes a separate call for each
dimension (relevance, coherence, consistency, fluency, informativeness).

Supports local vLLM and TogetherAI backends via the shared LLM client.
When ``--persona`` is enabled, uses persona-aware prompts and rubrics that
incorporate annotator profile information.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from persona_eval.metrics.base import BaseMetric, register_metric

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_DEFAULT_PROMPT_PATH = _PROMPTS_DIR / "llm_judge_default.txt"
_PERSONA_PROMPT_PATH = _PROMPTS_DIR / "llm_judge_persona.txt"
_RUBRICS_DIR = _PROMPTS_DIR / "rubrics"

_BASE_DIMENSIONS = ("relevance", "coherence", "consistency", "fluency")
_ALL_DIMENSIONS = _BASE_DIMENSIONS + ("informativeness",)

# Matches "[RESULT] 4" or "[RESULT]4" with optional whitespace
_RESULT_RE = re.compile(r"\[RESULT\]\s*(\d)")


def _load_prompt_template(prompt_file: str | None, persona: bool = False) -> str:
    """Load a prompt template from a file path, falling back to the default."""
    if prompt_file:
        return Path(prompt_file).read_text()
    path = _PERSONA_PROMPT_PATH if persona else _DEFAULT_PROMPT_PATH
    return path.read_text()


def _load_rubric(dimension: str, persona: bool = False) -> str:
    """Load the rubric text for a given dimension."""
    if persona and dimension == "informativeness":
        return (_RUBRICS_DIR / "informativeness_persona.txt").read_text()
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

    When ``persona=True``, uses persona-aware prompts and the persona
    informativeness rubric. The ``annotator_profile`` dict must be passed
    to ``score()`` via the pipeline.
    """

    def __init__(
        self,
        provider: str = "vllm",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        prompt_file: str | None = None,
        persona: bool = False,
        **kwargs,
    ):
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._persona = persona
        self._prompt_template = _load_prompt_template(prompt_file, persona=persona)
        self._rubrics = {
            dim: _load_rubric(dim, persona=persona) for dim in _ALL_DIMENSIONS
        }
        self._client = None

    @property
    def name(self) -> str:
        return "LLM Judge"

    @property
    def is_reference_free(self) -> bool:
        return True

    @property
    def needs_persona(self) -> bool:
        return self._persona

    def _load(self):
        if self._client is None:
            from persona_eval.llm_client import LLMClient

            self._client = LLMClient(
                provider=self._provider,
                model=self._model,
                api_key=self._api_key,
                base_url=self._base_url,
            )

    def _score_dimension(
        self,
        summary: str,
        source: str,
        dimension: str,
        persona_kwargs: dict | None = None,
    ) -> float:
        """Score a single dimension via one LLM call."""
        rubric = self._rubrics[dimension]
        fmt = dict(summary=summary, source=source, dimension=dimension)

        # Fill persona placeholders in rubric and prompt if available
        pk = persona_kwargs or {}
        if pk:
            rubric = rubric.format(**pk)
            fmt.update(pk)

        fmt["rubric"] = rubric

        prompt = self._prompt_template.format(**fmt)
        response_text = self._client.generate(prompt)
        result = _parse_prometheus_score(response_text)
        if result is None:
            logger.warning("LLM judge: no [RESULT] tag found for %s", dimension)
            return float("nan")
        return float(result)

    def score(
        self,
        summary: str,
        source: str,
        persona_kwargs: dict | None = None,
    ) -> dict[str, float]:
        self._load()

        scores = {}
        for dim in _ALL_DIMENSIONS:
            try:
                scores[f"llm_judge_{dim}"] = self._score_dimension(
                    summary, source, dim, persona_kwargs=persona_kwargs,
                )
            except Exception:
                logger.warning("LLM judge call failed for %s, returning NaN", dim)
                scores[f"llm_judge_{dim}"] = float("nan")

        valid = [v for v in scores.values() if v == v]  # filter NaN
        scores["llm_judge_overall"] = sum(valid) / len(valid) if valid else float("nan")

        return scores
