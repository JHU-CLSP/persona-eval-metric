"""LLM-as-judge annotator metric using absolute 1-5 Likert grading.

Uses the same annotator instructions as ``llm_judge_annotator`` (role,
domain, information needs, query) but grades each summary independently
on a 1-5 Likert scale rather than comparing two summaries directly.
A pair verdict (A/B/tie) is derived from the two absolute scores:
ties occur when the judge gives both summaries the same score.

Supports local vLLM and TogetherAI backends via the shared LLM client.
"""

from __future__ import annotations

import logging
from pathlib import Path

from persona_eval.metrics.base import register_metric
from persona_eval.metrics.base_llm import (
    BaseLLMMetric,
    PROMPTS_DIR,
    RUBRICS_DIR,
    SCORE_RE,
)

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_PATH = PROMPTS_DIR / "llm_judge_annotator_absolute.md"
_DEFAULT_RUBRIC_PATH = RUBRICS_DIR / "annotator_query.txt"


@register_metric("llm_judge_annotator_absolute")
class LLMJudgeAnnotatorAbsoluteMetric(BaseLLMMetric):
    """Annotator-perspective absolute grading on a 1-5 Likert scale.

    Scores each summary independently with one LLM call, returning an
    integer 1-5. For pairwise tournaments, the two scores are compared:
    higher score wins, equal scores produce ``"tie"``.
    """

    def __init__(
        self,
        prompt_file: str | None = None,
        rubric_file: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._prompt_file = prompt_file
        self._rubric_file = rubric_file
        self._prompt_template = (
            Path(prompt_file).read_text() if prompt_file else _DEFAULT_PROMPT_PATH.read_text()
        )
        self._rubric = (
            Path(rubric_file).read_text() if rubric_file else _DEFAULT_RUBRIC_PATH.read_text()
        )

    def cache_config(self) -> dict:
        return {
            **super().cache_config(),
            "prompt_file": self._prompt_file,
            "rubric_file": self._rubric_file,
        }

    @property
    def name(self) -> str:
        return "LLM Judge (Annotator Absolute)"

    @property
    def is_reference_free(self) -> bool:
        return True

    @property
    def is_pairwise(self) -> bool:
        return True

    @property
    def needs_persona(self) -> bool:
        return True

    @property
    def needs_query(self) -> bool:
        return True

    def score(self, summary: str, source: str, **kwargs) -> dict[str, float]:
        raise NotImplementedError(
            "llm_judge_annotator_absolute is run pairwise. "
            "Use score_pair() or let the pipeline drive the tournament."
        )

    def _score_one(self, summary: str, persona_kwargs: dict) -> int | None:
        """Run one LLM call to grade ``summary`` on the 1-5 rubric.

        Returns the parsed integer score, or ``None`` if parsing fails.
        """
        pk = persona_kwargs or {}
        rubric = self._rubric.format(
            role=pk.get("role", "unspecified"),
            domain=pk.get("domain", "unspecified"),
            info_needs=pk.get("info_needs", "unspecified"),
            query=pk.get("query", ""),
        )
        prompt = self._prompt_template.format(
            summary=summary,
            role=pk.get("role", "unspecified"),
            domain=pk.get("domain", "unspecified"),
            info_needs=pk.get("info_needs", "unspecified"),
            query=pk.get("query", ""),
            rubric=rubric,
        )

        response_text = self._client.generate(prompt)
        match = SCORE_RE.search(response_text)
        score = int(match.group(1)) if match else None

        self._log_response(
            metric="llm_judge_annotator_absolute",
            prompt=prompt,
            response=response_text,
            parsed_result=score,
            query=pk.get("query", ""),
        )
        return score

    def score_pair(
        self,
        summary_a: str,
        summary_b: str,
        source: str,
        persona_kwargs: dict | None = None,
    ) -> dict[str, str]:
        """Grade each summary on 1-5 and return a pair verdict.

        Returns ``"A"`` if summary_a's score is higher, ``"B"`` if
        summary_b's score is higher, ``"tie"`` when the scores are equal
        or either call fails to parse a score.
        """
        self._load()
        pk = persona_kwargs or {}

        try:
            score_a = self._score_one(summary_a, pk)
        except Exception:
            logger.warning("LLM judge annotator absolute call failed for A, marking tie")
            score_a = None

        try:
            score_b = self._score_one(summary_b, pk)
        except Exception:
            logger.warning("LLM judge annotator absolute call failed for B, marking tie")
            score_b = None

        if score_a is None or score_b is None:
            return {"llm_judge_annotator_absolute": "tie"}

        if score_a > score_b:
            verdict = "A"
        elif score_b > score_a:
            verdict = "B"
        else:
            verdict = "tie"
        return {"llm_judge_annotator_absolute": verdict}
