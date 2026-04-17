"""Persona-Precision metric: relevance of summary nuggets to the persona.

Measures what fraction of informational nuggets in the summary are relevant
to the persona's role, domain, needs, and query.  Steps:

1. Extract informational nuggets from the summary
2. Assess each nugget's relevance to the persona and query (binary: relevant / not relevant)
3. Return precision = relevant_nuggets / total_nuggets

Supports local vLLM and TogetherAI backends via the shared LLM client.
"""

from __future__ import annotations

import logging

from persona_eval.metrics.base import register_metric
from persona_eval.metrics.base_llm import BaseLLMMetric, PROMPTS_DIR, parse_bullet_list

logger = logging.getLogger(__name__)


@register_metric("persona_precision")
class PersonaPrecisionMetric(BaseLLMMetric):
    """Persona-Precision: fraction of summary nuggets relevant to the persona."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._extract_template = (PROMPTS_DIR / "persona_precision_extract.txt").read_text()
        self._relevance_template = (PROMPTS_DIR / "persona_precision_relevance.txt").read_text()

    @property
    def name(self) -> str:
        return "Persona Precision"

    @property
    def is_reference_free(self) -> bool:
        return True

    @property
    def needs_persona(self) -> bool:
        return True

    @property
    def needs_query(self) -> bool:
        return True

    def _extract_nuggets(self, summary: str) -> list[str]:
        """Decompose the summary into atomic informational nuggets."""
        prompt = self._extract_template.format(summary=summary)
        response = self._client.generate(prompt)
        nuggets = parse_bullet_list(response)

        self._log_response(
            metric="persona_precision",
            prompt=prompt,
            response=response,
            parsed_result=nuggets,
            step="extract_nuggets",
        )

        return nuggets

    def _assess_relevance(
        self, nugget: str, role: str, domain: str, info_needs: str, query: str
    ) -> bool:
        """Check whether a single nugget is relevant to the persona."""
        prompt = self._relevance_template.format(
            nugget=nugget,
            role=role,
            domain=domain,
            info_needs=info_needs,
            query=query,
        )
        response = self._client.generate(prompt, max_tokens=16)
        relevant = "relevant" in response.lower() and "not relevant" not in response.lower()

        self._log_response(
            metric="persona_precision",
            prompt=prompt,
            response=response,
            parsed_result=relevant,
            step="assess_relevance",
            nugget=nugget,
        )

        return relevant

    def score(self, summary: str, source: str, persona_kwargs=None) -> dict[str, float]:
        self._load()
        pk = persona_kwargs or {}

        nan_result = {
            "persona_precision": float("nan"),
            "persona_precision_num_nuggets": 0.0,
            "persona_precision_num_relevant": 0.0,
        }

        # Step 1: Extract nuggets from the summary
        try:
            nuggets = self._extract_nuggets(summary)
        except Exception:
            logger.warning("Persona Precision: failed to extract nuggets, returning NaN")
            return nan_result

        if not nuggets:
            logger.warning("Persona Precision: no nuggets extracted from summary")
            return nan_result

        # Step 2: Assess relevance of each nugget
        num_relevant = 0
        for nugget in nuggets:
            try:
                if self._assess_relevance(
                    nugget,
                    pk.get("role", "unspecified"),
                    pk.get("domain", "unspecified"),
                    pk.get("info_needs", "unspecified"),
                    pk.get("query", ""),
                ):
                    num_relevant += 1
            except Exception:
                logger.warning("Persona Precision: failed relevance check: %s", nugget[:80])

        return {
            "persona_precision": num_relevant / len(nuggets),
            "persona_precision_num_nuggets": float(len(nuggets)),
            "persona_precision_num_relevant": float(num_relevant),
        }
