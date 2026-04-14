"""LLM-as-judge metric using Prometheus relative (pairwise) grading.

Compares two summaries side-by-side for each dimension and returns a
preference ("A" or "B"). The pipeline converts these pairwise preferences
into tournament scores matching the human annotation structure.

Supports local vLLM and TogetherAI backends via the shared LLM client.
When ``--persona`` is enabled, uses persona-aware prompts and rubrics.
"""

from __future__ import annotations

import logging

from persona_eval.metrics.base import register_metric
from persona_eval.metrics.base_llm import (
    BaseLLMMetric,
    PAIRWISE_RE,
    parse_pairwise_result,
    PROMPTS_DIR,
    load_prompt_template,
    load_rubric,
)

logger = logging.getLogger(__name__)

_DEFAULT_RELATIVE_PROMPT_PATH = PROMPTS_DIR / "llm_judge_relative.txt"
_PERSONA_RELATIVE_PROMPT_PATH = PROMPTS_DIR / "llm_judge_relative_persona.txt"

_ALL_DIMENSIONS = ("relevance", "coherence", "consistency", "fluency", "informativeness")


@register_metric("llm_judge_relative")
class LLMJudgeRelativeMetric(BaseLLMMetric):
    """LLM-as-judge with Prometheus relative (pairwise) grading.

    Compares two summaries per LLM call for each evaluation dimension.
    Returns ``"A"`` or ``"B"`` per dimension. The pipeline computes
    tournament scores matching the human annotation structure.

    Reuses the same rubric files as the absolute ``llm_judge`` metric.
    When ``persona=True``, uses persona-aware prompts and rubrics.
    """

    def __init__(
        self,
        prompt_file: str | None = None,
        persona: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._persona = persona
        self._prompt_template = load_prompt_template(
            prompt_file, _DEFAULT_RELATIVE_PROMPT_PATH,
            _PERSONA_RELATIVE_PROMPT_PATH, persona=persona,
        )
        self._rubrics = {
            dim: load_rubric(dim, persona=persona) for dim in _ALL_DIMENSIONS
        }

    @property
    def name(self) -> str:
        return "LLM Judge (Relative)"

    @property
    def is_reference_free(self) -> bool:
        return True

    @property
    def is_pairwise(self) -> bool:
        return True

    @property
    def needs_persona(self) -> bool:
        return self._persona

    def score(self, summary: str, source: str, **kwargs) -> dict[str, float]:
        """Not used for pairwise metrics — raises an error."""
        raise NotImplementedError(
            "llm_judge_relative is a pairwise metric. "
            "Use score_pair() or let the pipeline compute tournament scores."
        )

    def _compare_dimension(
        self,
        summary_a: str,
        summary_b: str,
        source: str,
        dimension: str,
        persona_kwargs: dict | None = None,
    ) -> str:
        """Compare two summaries on one dimension. Returns 'A', 'B', or 'tie'."""
        rubric = self._rubrics[dimension]
        pk = persona_kwargs or {}

        query = pk.get("query", "")
        query_section = f"\nQuery: {query}\n" if query else "\n"

        fmt = dict(
            summary_a=summary_a, summary_b=summary_b,
            source=source, dimension=dimension,
            query_section=query_section,
        )

        if pk:
            rubric = rubric.format(**pk)
            fmt.update(pk)

        fmt["rubric"] = rubric

        prompt = self._prompt_template.format(**fmt)
        response_text = self._client.generate(prompt)
        match = PAIRWISE_RE.search(response_text)
        result = parse_pairwise_result(match)

        self._log_response(
            metric="llm_judge_relative",
            prompt=prompt,
            response=response_text,
            parsed_result=result,
            dimension=dimension,
        )

        if result is None:
            logger.warning(
                "LLM judge relative: no [RESULT] tag found for %s", dimension
            )
            return "tie"
        return result

    def score_pair(
        self,
        summary_a: str,
        summary_b: str,
        source: str,
        persona_kwargs: dict | None = None,
    ) -> dict[str, str]:
        """Compare two summaries on all dimensions.

        Returns:
            Dict mapping sub-metric names to ``"A"``, ``"B"``, or ``"tie"``.
        """
        self._load()

        results = {}
        for dim in _ALL_DIMENSIONS:
            try:
                results[f"llm_judge_rel_{dim}"] = self._compare_dimension(
                    summary_a, summary_b, source, dim,
                    persona_kwargs=persona_kwargs,
                )
            except Exception:
                logger.warning(
                    "LLM judge relative call failed for %s, marking as tie", dim
                )
                results[f"llm_judge_rel_{dim}"] = "tie"

        # Overall: majority vote across dimensions
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
