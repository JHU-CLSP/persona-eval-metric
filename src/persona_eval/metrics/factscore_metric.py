"""FACTScore metric: atomic fact verification against source documents.

Implements the FACTScore algorithm (Min et al., 2023) using an LLM to:
1. Decompose the summary into atomic facts
2. Verify each fact against the source document
3. Return the fraction of supported facts

Supports local vLLM and TogetherAI backends via the shared LLM client.
"""

from __future__ import annotations

import logging
from pathlib import Path

from persona_eval.metrics.base import BaseMetric, register_metric

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text()


def _parse_facts(response: str) -> list[str]:
    """Parse bullet-pointed facts from the LLM response."""
    facts = []
    for line in response.strip().splitlines():
        line = line.strip()
        if line.startswith("- "):
            line = line[2:].strip()
        elif line.startswith("* "):
            line = line[2:].strip()
        else:
            # Skip non-bullet lines (e.g. blank lines, headers)
            continue
        if line:
            facts.append(line)
    return facts


@register_metric("factscore")
class FACTScoreMetric(BaseMetric):
    """FACTScore: fraction of atomic facts in the summary supported by the source.

    Uses an LLM to decompose the summary into atomic facts, then checks each
    fact against the source document.
    """

    def __init__(
        self,
        provider: str = "vllm",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        response_logger=None,
        **kwargs,
    ):
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._response_logger = response_logger
        self._extract_template = _load_prompt("factscore_extract.txt")
        self._verify_template = _load_prompt("factscore_verify.txt")
        self._client = None

    @property
    def name(self) -> str:
        return "FACTScore"

    @property
    def is_reference_free(self) -> bool:
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

    def _extract_facts(self, summary: str) -> list[str]:
        prompt = self._extract_template.format(summary=summary)
        response = self._client.generate(prompt)
        facts = _parse_facts(response)

        if self._response_logger:
            self._response_logger.log(
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

        if self._response_logger:
            self._response_logger.log(
                metric="factscore",
                prompt=prompt,
                response=response,
                parsed_result=supported,
                step="verify_fact",
                claim=claim,
            )

        return supported

    def score(self, summary: str, source: str) -> dict[str, float]:
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
