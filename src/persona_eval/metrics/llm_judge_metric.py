"""LLM-as-judge absolute (single-summary) grading across rubric dimensions."""

from __future__ import annotations

import logging

from persona_eval.metrics.base import register_metric
from persona_eval.metrics.base_llm import (
    BaseDimensionalLLMMetric,
    PROMPTS_DIR,
    SCORE_RE,
)

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_PATH = PROMPTS_DIR / "llm_judge_default.txt"
_PERSONA_PROMPT_PATH = PROMPTS_DIR / "llm_judge_persona.txt"


def _parse_int_score(match) -> int | None:
    return int(match.group(1)) if match else None


@register_metric("llm_judge")
class LLMJudgeMetric(BaseDimensionalLLMMetric):
    """Rate a summary on each of five dimensions via one LLM call per dimension.

    Uses the Prometheus absolute-grading prompt format: each call returns
    ``[RESULT] <score>`` on a 1-5 scale. When ``persona=True``, uses
    persona-aware prompts and the persona informativeness rubric.
    """

    _metric_log_name = "llm_judge"

    def __init__(self, prompt_file: str | None = None, persona: bool = False, **kwargs):
        super().__init__(
            default_prompt_path=_DEFAULT_PROMPT_PATH,
            persona_prompt_path=_PERSONA_PROMPT_PATH,
            prompt_file=prompt_file, persona=persona, **kwargs,
        )

    @property
    def name(self) -> str:
        return "LLM Judge"

    def score(
        self, summary: str, source: str, persona_kwargs: dict | None = None,
    ) -> dict[str, float]:
        self._load()
        scores = {}
        for dim in self._dimensions:
            try:
                result = self._score_dim(
                    dim,
                    prompt_vars={"summary": summary, "source": source},
                    pattern=SCORE_RE,
                    parser=_parse_int_score,
                    persona_kwargs=persona_kwargs,
                )
                if result is None:
                    logger.warning("LLM judge: no [RESULT] tag found for %s", dim)
                    scores[f"llm_judge_{dim}"] = float("nan")
                else:
                    scores[f"llm_judge_{dim}"] = float(result)
            except Exception:
                logger.warning("LLM judge call failed for %s, returning NaN", dim)
                scores[f"llm_judge_{dim}"] = float("nan")

        valid = [v for v in scores.values() if v == v]
        scores["llm_judge_overall"] = sum(valid) / len(valid) if valid else float("nan")
        return scores
