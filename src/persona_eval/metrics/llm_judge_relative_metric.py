"""LLM-as-judge metric using Prometheus relative (pairwise) grading.

Compares two summaries side-by-side for each dimension and returns a
preference ("A" or "B"). The pipeline converts these pairwise preferences
into per-summary win-rate scores.

Supports local vLLM and TogetherAI backends via the shared LLM client.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from persona_eval.metrics.base import BaseMetric, register_metric

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_DEFAULT_RELATIVE_PROMPT_PATH = _PROMPTS_DIR / "llm_judge_relative.txt"
_RUBRICS_DIR = _PROMPTS_DIR / "rubrics"

_SCORE_DIMENSIONS = ("relevance", "coherence", "consistency", "fluency")

# Matches "[RESULT] A" or "[RESULT] B" with optional whitespace
_RESULT_RE = re.compile(r"\[RESULT\]\s*([ABab])")


def _load_prompt_template(prompt_file: str | None = None) -> str:
    path = Path(prompt_file) if prompt_file else _DEFAULT_RELATIVE_PROMPT_PATH
    return path.read_text()


def _load_rubric(dimension: str) -> str:
    return (_RUBRICS_DIR / f"{dimension}.txt").read_text()


def _parse_pairwise_result(text: str) -> str | None:
    """Extract 'A' or 'B' from a Prometheus relative grading response."""
    match = _RESULT_RE.search(text)
    return match.group(1).upper() if match else None


@register_metric("llm_judge_relative")
class LLMJudgeRelativeMetric(BaseMetric):
    """LLM-as-judge with Prometheus relative (pairwise) grading.

    Compares two summaries per LLM call for each evaluation dimension.
    Returns ``"A"`` or ``"B"`` per dimension. The pipeline computes
    win-rates across all pairs within a query to produce per-summary scores.

    Reuses the same rubric files as the absolute ``llm_judge`` metric.
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
        return "LLM Judge (Relative)"

    @property
    def is_reference_free(self) -> bool:
        return True

    @property
    def is_pairwise(self) -> bool:
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
        """Not used for pairwise metrics — raises an error.

        Use ``score_pair()`` instead, or let the pipeline compute win-rates.
        """
        raise NotImplementedError(
            "llm_judge_relative is a pairwise metric. "
            "Use score_pair() or let the pipeline compute win-rates."
        )

    def _compare_dimension(
        self, summary_a: str, summary_b: str, source: str, dimension: str,
    ) -> str:
        """Compare two summaries on one dimension. Returns 'A', 'B', or 'tie'."""
        prompt = self._prompt_template.format(
            summary_a=summary_a,
            summary_b=summary_b,
            source=source,
            dimension=dimension,
            rubric=self._rubrics[dimension],
        )
        response_text = self._client.generate(prompt)
        result = _parse_pairwise_result(response_text)
        if result is None:
            logger.warning(
                "LLM judge relative: no [RESULT] tag found for %s", dimension
            )
            return "tie"
        return result

    def score_pair(
        self, summary_a: str, summary_b: str, source: str,
    ) -> dict[str, str]:
        """Compare two summaries on all dimensions.

        Returns:
            Dict mapping sub-metric names to ``"A"``, ``"B"``, or ``"tie"``.
        """
        self._load()

        results = {}
        for dim in _SCORE_DIMENSIONS:
            try:
                results[f"llm_judge_rel_{dim}"] = self._compare_dimension(
                    summary_a, summary_b, source, dim,
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
