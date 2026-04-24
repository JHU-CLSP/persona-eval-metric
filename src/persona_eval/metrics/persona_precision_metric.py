"""Persona-Precision metric.

Fraction of informational nuggets in the summary that are relevant to
the persona (role + domain + info needs + query). Pipeline:

  1. Extract nuggets from the summary.
  2. For each nugget, ask the LLM whether it's relevant to the persona.
  3. Precision = relevant_nuggets / total_nuggets.
"""

from __future__ import annotations

import logging

from persona_eval.metrics.base import register_metric
from persona_eval.metrics.base_llm import (
    MultiStepLLMMetric,
    PROMPTS_DIR,
    parse_bullet_list,
)

logger = logging.getLogger(__name__)


@register_metric("persona_precision")
class PersonaPrecisionMetric(MultiStepLLMMetric):
    """Fraction of summary nuggets relevant to the persona."""

    _metric_log_name = "persona_precision"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._extract_template = (PROMPTS_DIR / "persona_precision_extract.txt").read_text()
        self._relevance_template = (PROMPTS_DIR / "persona_precision_relevance.txt").read_text()

    @property
    def name(self) -> str:
        return "Persona Precision"

    def score(self, summary: str, source: str, persona_kwargs=None) -> dict[str, float]:
        self._load()
        pk = persona_kwargs or {}

        nan_result = {
            "persona_precision": float("nan"),
            "persona_precision_num_nuggets": 0.0,
            "persona_precision_num_relevant": 0.0,
        }

        try:
            nuggets = self._run_step(
                "extract_nuggets", self._extract_template, parse_bullet_list,
                summary=summary,
            )
        except Exception:
            logger.warning("Persona Precision: failed to extract nuggets, returning NaN")
            return nan_result

        if not nuggets:
            logger.warning("Persona Precision: no nuggets extracted from summary")
            return nan_result

        num_relevant = 0
        for nugget in nuggets:
            try:
                if self._run_step(
                    "assess_relevance", self._relevance_template,
                    lambda r: self._classify_positive(r, "relevant"),
                    max_tokens=16,
                    log_extra={"nugget": nugget},
                    nugget=nugget,
                    role=pk.get("role", "unspecified"),
                    domain=pk.get("domain", "unspecified"),
                    info_needs=pk.get("info_needs", "unspecified"),
                    query=pk.get("query", ""),
                ):
                    num_relevant += 1
            except Exception:
                logger.warning("Persona Precision: failed relevance check: %s", nugget[:80])

        return {
            "persona_precision": num_relevant / len(nuggets),
            "persona_precision_num_nuggets": float(len(nuggets)),
            "persona_precision_num_relevant": float(num_relevant),
        }
