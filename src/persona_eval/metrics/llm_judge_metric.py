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

from persona_eval.metrics.base import register_metric
from persona_eval.metrics.base_llm import (
    BaseLLMMetric,
    PROMPTS_DIR,
    SCORE_RE,
    load_prompt_template,
    load_rubric,
)

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_PATH = PROMPTS_DIR / "llm_judge_default.txt"
_PERSONA_PROMPT_PATH = PROMPTS_DIR / "llm_judge_persona.txt"

_ALL_DIMENSIONS = ("relevance", "coherence", "consistency", "fluency", "informativeness")


@register_metric("llm_judge")
class LLMJudgeMetric(BaseLLMMetric):
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
        prompt_file: str | None = None,
        persona: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._persona = persona
        self._prompt_file = prompt_file
        self._prompt_template = load_prompt_template(
            prompt_file, _DEFAULT_PROMPT_PATH, _PERSONA_PROMPT_PATH, persona=persona,
        )
        self._rubrics = {
            dim: load_rubric(dim, persona=persona) for dim in _ALL_DIMENSIONS
        }

    def cache_config(self) -> dict:
        return {
            **super().cache_config(),
            "persona": self._persona,
            "prompt_file": self._prompt_file,
        }

    @property
    def name(self) -> str:
        return "LLM Judge"

    @property
    def is_reference_free(self) -> bool:
        return True

    @property
    def needs_persona(self) -> bool:
        return self._persona

    def _score_dimension(
        self,
        summary: str,
        source: str,
        dimension: str,
        persona_kwargs: dict | None = None,
    ) -> float:
        """Score a single dimension via one LLM call."""
        rubric = self._rubrics[dimension]
        pk = persona_kwargs or {}

        query = pk.get("query", "")
        query_section = f"\nQuery: {query}\n" if query else "\n"

        fmt = dict(summary=summary, source=source, dimension=dimension,
                   query_section=query_section)

        if pk:
            rubric = rubric.format(**pk)
            fmt.update(pk)

        fmt["rubric"] = rubric

        prompt = self._prompt_template.format(**fmt)
        response_text = self._client.generate(prompt)
        match = SCORE_RE.search(response_text)
        result = int(match.group(1)) if match else None

        self._log_response(
            metric="llm_judge",
            prompt=prompt,
            response=response_text,
            parsed_result=result,
            dimension=dimension,
        )

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
