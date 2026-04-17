"""FACTScore metric: atomic fact verification against source documents.

Implements the FACTScore algorithm (Min et al., 2023) using an LLM to:
1. Decompose the summary into atomic facts
2. Verify each fact against the source document
3. Return the fraction of supported facts

Supports local vLLM and TogetherAI backends via the shared LLM client.
"""

from __future__ import annotations

import logging

from persona_eval.metrics.base import register_metric
from persona_eval.metrics.base_llm import BaseLLMMetric, PROMPTS_DIR, parse_bullet_list

logger = logging.getLogger(__name__)


@register_metric("factscore")
class FACTScoreMetric(BaseLLMMetric):
    """FACTScore: fraction of atomic facts in the summary supported by the source.

    Uses an LLM to decompose the summary into atomic facts, then checks each
    fact against the source document.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._extract_template = (PROMPTS_DIR / "factscore_extract.txt").read_text()
        self._verify_template = (PROMPTS_DIR / "factscore_verify.txt").read_text()

    @property
    def name(self) -> str:
        return "FACTScore"

    @property
    def is_reference_free(self) -> bool:
        return True

    def _extract_facts(self, summary: str) -> list[str]:
        prompt = self._extract_template.format(summary=summary)
        response = self._client.generate(prompt)
        facts = parse_bullet_list(response)

        self._log_response(
            metric="factscore",
            prompt=prompt,
            response=response,
            parsed_result=facts,
            step="extract_facts",
        )

        return facts

    def _verify_fact(self, source: str, claim: str) -> bool:
        prompt = self._verify_template.format(source=source, claim=claim)
        response = self._client.generate(prompt, max_tokens=16)
        supported = "supported" in response.lower() and "not supported" not in response.lower()

        self._log_response(
            metric="factscore",
            prompt=prompt,
            response=response,
            parsed_result=supported,
            step="verify_fact",
            claim=claim,
        )

        return supported

    def score(self, summary: str, source: str, persona_kwargs=None) -> dict[str, float]:
        self._load()

        try:
            facts = self._extract_facts(summary)
        except Exception:
            logger.warning("FACTScore: failed to extract facts, returning NaN")
            return {
                "factscore": float("nan"),
                "factscore_num_facts": 0.0,
                "factscore_num_supported": 0.0,
            }

        if not facts:
            logger.warning("FACTScore: no atomic facts extracted from summary")
            return {
                "factscore": float("nan"),
                "factscore_num_facts": 0.0,
                "factscore_num_supported": 0.0,
            }

        num_supported = 0
        for fact in facts:
            try:
                if self._verify_fact(source, fact):
                    num_supported += 1
            except Exception:
                logger.warning("FACTScore: failed to verify fact: %s", fact[:80])

        return {
            "factscore": num_supported / len(facts),
            "factscore_num_facts": float(len(facts)),
            "factscore_num_supported": float(num_supported),
        }
