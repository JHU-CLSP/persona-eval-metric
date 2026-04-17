"""Persona-Recall metric: coverage of persona-specific information requirements.

Measures what fraction of information requirements (derived from the persona,
query, and source document) are covered by the summary.  Steps:

1. Generate informational nugget requirements from persona + query + source
2. Extract informational nuggets from the summary
3. Check each requirement against the summary nuggets (binary: covered / not covered)
4. Return recall = covered_requirements / total_requirements

Supports local vLLM and TogetherAI backends via the shared LLM client.
"""

from __future__ import annotations

import logging

from persona_eval.metrics.base import register_metric
from persona_eval.metrics.base_llm import BaseLLMMetric, PROMPTS_DIR, parse_bullet_list

logger = logging.getLogger(__name__)


@register_metric("persona_recall")
class PersonaRecallMetric(BaseLLMMetric):
    """Persona-Recall: fraction of persona-specific requirements covered by the summary."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._requirements_template = (PROMPTS_DIR / "persona_recall_requirements.txt").read_text()
        self._extract_template = (PROMPTS_DIR / "persona_recall_extract.txt").read_text()
        self._coverage_template = (PROMPTS_DIR / "persona_recall_coverage.txt").read_text()

    @property
    def name(self) -> str:
        return "Persona Recall"

    @property
    def is_reference_free(self) -> bool:
        return True

    @property
    def needs_persona(self) -> bool:
        return True

    @property
    def needs_query(self) -> bool:
        return True

    def _generate_requirements(
        self, source: str, role: str, domain: str, info_needs: str, query: str
    ) -> list[str]:
        """Use the LLM to generate informational requirements for this persona."""
        prompt = self._requirements_template.format(
            source=source,
            role=role,
            domain=domain,
            info_needs=info_needs,
            query=query,
        )
        response = self._client.generate(prompt)
        requirements = parse_bullet_list(response)

        self._log_response(
            metric="persona_recall",
            prompt=prompt,
            response=response,
            parsed_result=requirements,
            step="generate_requirements",
        )

        return requirements

    def _extract_nuggets(self, summary: str) -> list[str]:
        """Decompose the summary into atomic informational nuggets."""
        prompt = self._extract_template.format(summary=summary)
        response = self._client.generate(prompt)
        nuggets = parse_bullet_list(response)

        self._log_response(
            metric="persona_recall",
            prompt=prompt,
            response=response,
            parsed_result=nuggets,
            step="extract_nuggets",
        )

        return nuggets

    def _check_coverage(self, requirement: str, nuggets: list[str]) -> bool:
        """Check whether a single requirement is covered by any summary nugget."""
        nuggets_text = "\n".join(f"- {n}" for n in nuggets)
        prompt = self._coverage_template.format(
            requirement=requirement,
            nuggets=nuggets_text,
        )
        response = self._client.generate(prompt, max_tokens=16)
        covered = "covered" in response.lower() and "not covered" not in response.lower()

        self._log_response(
            metric="persona_recall",
            prompt=prompt,
            response=response,
            parsed_result=covered,
            step="check_coverage",
            requirement=requirement,
        )

        return covered

    def score(self, summary: str, source: str, persona_kwargs=None) -> dict[str, float]:
        self._load()
        pk = persona_kwargs or {}

        nan_result = {
            "persona_recall": float("nan"),
            "persona_recall_num_requirements": 0.0,
            "persona_recall_num_covered": 0.0,
            "persona_recall_num_nuggets": 0.0,
        }

        # Step 1: Generate requirements from persona + query + source
        try:
            requirements = self._generate_requirements(
                source,
                pk.get("role", "unspecified"),
                pk.get("domain", "unspecified"),
                pk.get("info_needs", "unspecified"),
                pk.get("query", ""),
            )
        except Exception:
            logger.warning("Persona Recall: failed to generate requirements, returning NaN")
            return nan_result

        if not requirements:
            logger.warning("Persona Recall: no requirements generated")
            return nan_result

        # Step 2: Extract nuggets from the summary
        try:
            nuggets = self._extract_nuggets(summary)
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

        # Step 3: Check coverage of each requirement
        num_covered = 0
        for req in requirements:
            try:
                if self._check_coverage(req, nuggets):
                    num_covered += 1
            except Exception:
                logger.warning("Persona Recall: failed coverage check: %s", req[:80])

        return {
            "persona_recall": num_covered / len(requirements),
            "persona_recall_num_requirements": float(len(requirements)),
            "persona_recall_num_covered": float(num_covered),
            "persona_recall_num_nuggets": float(len(nuggets)),
        }
