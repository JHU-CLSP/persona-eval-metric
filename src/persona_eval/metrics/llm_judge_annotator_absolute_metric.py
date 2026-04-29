"""LLM-as-judge annotator metric using absolute 1-5 Likert grading.

Uses the same annotator instructions as ``llm_judge_annotator`` (role,
domain, information needs, query) but grades each summary independently
on a 1-5 Likert scale (one LLM call per summary).

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

    One LLM call per summary returns an integer 1-5 score reflecting how
    well the summary addresses the annotator's query, given their role,
    domain, and information needs.
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
        return False

    @property
    def needs_persona(self) -> bool:
        return True

    @property
    def needs_query(self) -> bool:
        return True

    def score(
        self,
        summary: str,
        source: str,
        persona_kwargs: dict | None = None,
    ) -> dict[str, float]:
        """Grade ``summary`` on a 1-5 Likert scale from the annotator's perspective.

        Returns ``{"llm_judge_annotator_absolute": <float>}``. The value
        is ``NaN`` if the LLM call fails or the response cannot be parsed.
        """
        self._load()
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

        try:
            response_text = self._client.generate(prompt)
        except Exception:
            logger.warning("LLM judge annotator absolute call failed, returning NaN")
            return {"llm_judge_annotator_absolute": float("nan")}

        match = SCORE_RE.search(response_text)
        parsed = int(match.group(1)) if match else None

        self._log_response(
            metric="llm_judge_annotator_absolute",
            prompt=prompt,
            response=response_text,
            parsed_result=parsed,
            query=pk.get("query", ""),
        )

        if parsed is None:
            logger.warning("LLM judge annotator absolute: no [RESULT] tag found")
            return {"llm_judge_annotator_absolute": float("nan")}

        return {"llm_judge_annotator_absolute": float(parsed)}
