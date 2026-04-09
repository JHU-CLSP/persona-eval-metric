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
import re
from pathlib import Path

from persona_eval.metrics.base import BaseMetric, register_metric

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_DEFAULT_PROMPT_PATH = _PROMPTS_DIR / "llm_judge_annotator.md"

# Matches "[RESULT] A" or "[RESULT] B" with optional whitespace
_RESULT_RE = re.compile(r"\[RESULT\]\s*([ABab])")


def _load_prompt_template(prompt_file: str | None = None) -> str:
    path = Path(prompt_file) if prompt_file else _DEFAULT_PROMPT_PATH
    return path.read_text()


def _parse_pairwise_result(text: str) -> str | None:
    match = _RESULT_RE.search(text)
    return match.group(1).upper() if match else None


@register_metric("llm_judge_annotator")
class LLMJudgeAnnotatorMetric(BaseMetric):
    """LLM-as-judge with annotator query-focused pairwise evaluation.

    Compares two summaries in a single LLM call from the annotator's
    perspective, focused on whether each summary addresses their specific
    query. Returns ``"A"``, ``"B"``, or ``"tie"``.

    Requires annotator profile (role, domain, info_needs) and the
    annotator's query, passed via ``persona_kwargs``.
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
        self._client = None

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
            result = _parse_pairwise_result(response_text)
        except Exception:
            logger.warning("LLM judge annotator call failed, marking as tie")
            return {"llm_judge_annotator": "tie"}

        if result is None:
            logger.warning("LLM judge annotator: no [RESULT] tag found")
            return {"llm_judge_annotator": "tie"}

        return {"llm_judge_annotator": result}
