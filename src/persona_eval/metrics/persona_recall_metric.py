"""Persona-Recall metric.

Fraction of information requirements (derived from persona + query +
source) that are covered by the summary. Pipeline:

  1. Generate requirements from persona + query + source.
  2. Extract nuggets from the summary.
  3. For each requirement, ask the LLM whether any nugget covers it.
  4. Recall = covered_requirements / total_requirements.
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


@register_metric("persona_recall")
class PersonaRecallMetric(MultiStepLLMMetric):
    """Fraction of persona-specific requirements covered by the summary."""

    _metric_log_name = "persona_recall"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._requirements_template = (PROMPTS_DIR / "persona_recall_requirements.txt").read_text()
        self._extract_template = (PROMPTS_DIR / "persona_recall_extract.txt").read_text()
        self._coverage_template = (PROMPTS_DIR / "persona_recall_coverage.txt").read_text()

    @property
    def name(self) -> str:
        return "Persona Recall"

    def score(self, summary: str, source: str, persona_kwargs=None) -> dict[str, float]:
        self._load()
        pk = persona_kwargs or {}

        nan_result = {
            "persona_recall": float("nan"),
            "persona_recall_num_requirements": 0.0,
            "persona_recall_num_covered": 0.0,
            "persona_recall_num_nuggets": 0.0,
        }

        try:
            requirements = self._run_step(
                "generate_requirements", self._requirements_template, parse_bullet_list,
                source=source,
                role=pk.get("role", "unspecified"),
                domain=pk.get("domain", "unspecified"),
                info_needs=pk.get("info_needs", "unspecified"),
                query=pk.get("query", ""),
            )
        except Exception:
            logger.warning("Persona Recall: failed to generate requirements, returning NaN")
            return nan_result

        if not requirements:
            logger.warning("Persona Recall: no requirements generated")
            return nan_result

        try:
            nuggets = self._run_step(
                "extract_nuggets", self._extract_template, parse_bullet_list,
                summary=summary,
            )
        except Exception:
            logger.warning("Persona Recall: failed to extract nuggets, returning NaN")
            return nan_result

        if not nuggets:
            logger.warning("Persona Recall: no nuggets extracted from summary")
            return {
                "persona_recall": 0.0,
                "persona_recall_num_requirements": float(len(requirements)),
                "persona_recall_num_covered": 0.0,
                "persona_recall_num_nuggets": 0.0,
            }

        nuggets_text = "\n".join(f"- {n}" for n in nuggets)
        num_covered = 0
        for req in requirements:
            try:
                if self._run_step(
                    "check_coverage", self._coverage_template,
                    lambda r: self._classify_positive(r, "covered"),
                    max_tokens=16,
                    log_extra={"requirement": req},
                    requirement=req,
                    nuggets=nuggets_text,
                ):
                    num_covered += 1
            except Exception:
                logger.warning("Persona Recall: failed coverage check: %s", req[:80])

        return {
            "persona_recall": num_covered / len(requirements),
            "persona_recall_num_requirements": float(len(requirements)),
            "persona_recall_num_covered": float(num_covered),
            "persona_recall_num_nuggets": float(len(nuggets)),
        }
