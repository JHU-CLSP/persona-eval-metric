"""LLM-as-judge metric using Prometheus relative (pairwise) grading.

Compares two summaries side-by-side for each dimension and returns a
preference ("A" or "B"). The pipeline converts these pairwise preferences
into tournament scores matching the human annotation structure.

Supports local vLLM and TogetherAI backends via the shared LLM client.
When ``--persona`` is enabled, uses persona-aware prompts and rubrics.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from persona_eval.metrics.base import BaseMetric, register_metric

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_DEFAULT_RELATIVE_PROMPT_PATH = _PROMPTS_DIR / "llm_judge_relative.txt"
_PERSONA_RELATIVE_PROMPT_PATH = _PROMPTS_DIR / "llm_judge_relative_persona.txt"
_RUBRICS_DIR = _PROMPTS_DIR / "rubrics"

_BASE_DIMENSIONS = ("relevance", "coherence", "consistency", "fluency")
_ALL_DIMENSIONS = _BASE_DIMENSIONS + ("informativeness",)

# Matches "[RESULT] A" or "[RESULT] B" with optional whitespace
_RESULT_RE = re.compile(r"\[RESULT\]\s*([ABab])")


def _load_prompt_template(prompt_file: str | None, persona: bool = False) -> str:
    if prompt_file:
        return Path(prompt_file).read_text()
    path = _PERSONA_RELATIVE_PROMPT_PATH if persona else _DEFAULT_RELATIVE_PROMPT_PATH
    return path.read_text()


def _load_rubric(dimension: str, persona: bool = False) -> str:
    if persona and dimension == "informativeness":
        return (_RUBRICS_DIR / "informativeness_persona.txt").read_text()
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
    tournament scores matching the human annotation structure.

    Reuses the same rubric files as the absolute ``llm_judge`` metric.
    When ``persona=True``, uses persona-aware prompts and rubrics.
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

    def _load(self):
        if self._client is None:
            from persona_eval.llm_client import LLMClient

            self._client = LLMClient(
                provider=self._provider,
                model=self._model,
                api_key=self._api_key,
                base_url=self._base_url,
            )

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
        fmt = dict(
            summary_a=summary_a, summary_b=summary_b,
            source=source, dimension=dimension,
        )

        pk = persona_kwargs or {}
        if pk:
            rubric = rubric.format(**pk)
            fmt.update(pk)

        fmt["rubric"] = rubric

        prompt = self._prompt_template.format(**fmt)
        response_text = self._client.generate(prompt)
        result = _parse_pairwise_result(response_text)
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
