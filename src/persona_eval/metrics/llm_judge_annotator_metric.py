"""LLM-as-judge metric using annotator-specific query-focused evaluation.

Compares two summaries from the perspective of a specific annotator,
focusing on whether the summary addresses their query rather than
general quality dimensions like coherence or fluency.

Uses a single LLM call per pair (no per-dimension splitting).
Requires annotator profile and query information.

Supports local vLLM and TogetherAI backends via the shared LLM client.
"""

from __future__ import annotations

import logging

from persona_eval.metrics.base import register_metric
from persona_eval.metrics.base_llm import (
    BaseLLMMetric,
    PAIRWISE_RE,
    parse_pairwise_result,
    PROMPTS_DIR,
)

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_PATH = PROMPTS_DIR / "llm_judge_annotator.md"


@register_metric("llm_judge_annotator")
class LLMJudgeAnnotatorMetric(BaseLLMMetric):
    """LLM-as-judge with annotator query-focused pairwise evaluation.

    Compares two summaries in a single LLM call from the annotator's
    perspective, focused on whether each summary addresses their specific
    query. Returns ``"A"``, ``"B"``, or ``"tie"``.

    Requires annotator profile (role, domain, info_needs) and the
    annotator's query, passed via ``persona_kwargs``.
    """

    def __init__(
        self,
        prompt_file: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        from pathlib import Path
        self._prompt_template = Path(prompt_file).read_text() if prompt_file else _DEFAULT_PROMPT_PATH.read_text()

    @property
    def name(self) -> str:
        return "LLM Judge (Annotator)"

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
            "llm_judge_annotator is a pairwise metric. "
            "Use score_pair() or let the pipeline compute tournament scores."
        )

    def score_pair(
        self,
        summary_a: str,
        summary_b: str,
        source: str,
        persona_kwargs: dict | None = None,
    ) -> dict[str, str]:
        """Compare two summaries from the annotator's perspective.

        Args:
            summary_a: First summary text.
            summary_b: Second summary text.
            source: Source document text.
            persona_kwargs: Must include role, domain, info_needs, and query.

        Returns:
            Dict with single key ``"llm_judge_annotator"`` mapping to
            ``"A"``, ``"B"``, or ``"tie"``.
        """
        self._load()
        pk = persona_kwargs or {}

        prompt = self._prompt_template.format(
            summary_a=summary_a,
            summary_b=summary_b,
            source=source,
            role=pk.get("role", "unspecified"),
            domain=pk.get("domain", "unspecified"),
            info_needs=pk.get("info_needs", "unspecified"),
            query=pk.get("query", ""),
        )

        try:
            response_text = self._client.generate(prompt)
            match = PAIRWISE_RE.search(response_text)
            result = parse_pairwise_result(match)
        except Exception:
            logger.warning("LLM judge annotator call failed, marking as tie")
            return {"llm_judge_annotator": "tie"}

        self._log_response(
            metric="llm_judge_annotator",
            prompt=prompt,
            response=response_text,
            parsed_result=result,
            query=pk.get("query", ""),
        )

        if result is None:
            logger.warning("LLM judge annotator: no [RESULT] tag found")
            return {"llm_judge_annotator": "tie"}

        return {"llm_judge_annotator": result}
