"""LLM-as-judge pairwise (relative) grading across rubric dimensions."""

from __future__ import annotations

import logging

from persona_eval.metrics.base import register_metric
from persona_eval.metrics.base_llm import (
    BaseDimensionalLLMMetric,
    PAIRWISE_RE,
    PROMPTS_DIR,
    parse_pairwise_result,
)

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_PATH = PROMPTS_DIR / "llm_judge_relative.txt"
_PERSONA_PROMPT_PATH = PROMPTS_DIR / "llm_judge_relative_persona.txt"


@register_metric("llm_judge_relative")
class LLMJudgeRelativeMetric(BaseDimensionalLLMMetric):
    """Pairwise LLM-as-judge using Prometheus relative-grading format.

    Compares two summaries per LLM call for each dimension and returns
    ``"A"``/``"B"``/``"tie"``. Reuses the same rubric files as the
    absolute ``llm_judge`` metric.
    """

    _metric_log_name = "llm_judge_relative"

    def __init__(self, prompt_file: str | None = None, persona: bool = False, **kwargs):
        super().__init__(
            default_prompt_path=_DEFAULT_PROMPT_PATH,
            persona_prompt_path=_PERSONA_PROMPT_PATH,
            prompt_file=prompt_file, persona=persona, **kwargs,
        )

    @property
    def name(self) -> str:
        return "LLM Judge (Relative)"

    @property
    def is_pairwise(self) -> bool:
        return True

    def score(self, summary: str, source: str, **kwargs) -> dict[str, float]:
        raise NotImplementedError(
            "llm_judge_relative is a pairwise metric. "
            "Use score_pair() or let the pipeline compute tournament scores."
        )

    def score_pair(
        self, summary_a: str, summary_b: str, source: str,
        persona_kwargs: dict | None = None,
    ) -> dict[str, str]:
        self._load()
        results = {}
        for dim in self._dimensions:
            try:
                result = self._score_dim(
                    dim,
                    prompt_vars={
                        "summary_a": summary_a,
                        "summary_b": summary_b,
                        "source": source,
                    },
                    pattern=PAIRWISE_RE,
                    parser=parse_pairwise_result,
                    persona_kwargs=persona_kwargs,
                )
                if result is None:
                    logger.warning("LLM judge relative: no [RESULT] tag for %s", dim)
                    results[f"llm_judge_rel_{dim}"] = "tie"
                else:
                    results[f"llm_judge_rel_{dim}"] = result
            except Exception:
                logger.warning("LLM judge relative call failed for %s, tie", dim)
                results[f"llm_judge_rel_{dim}"] = "tie"

        votes = list(results.values())
        a_wins = votes.count("A")
        b_wins = votes.count("B")
        if a_wins > b_wins:
            results["llm_judge_rel_overall"] = "A"
        elif b_wins > a_wins:
            results["llm_judge_rel_overall"] = "B"
        else:
            results["llm_judge_rel_overall"] = "tie"
        return results
