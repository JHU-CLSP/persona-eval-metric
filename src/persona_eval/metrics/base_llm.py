"""Base class for LLM-backed metrics with shared client and logging logic."""

from __future__ import annotations

import re
from pathlib import Path

from persona_eval.metrics.base import BaseMetric

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
RUBRICS_DIR = PROMPTS_DIR / "rubrics"

# Shared regex patterns
SCORE_RE = re.compile(r"\[RESULT\]\s*(\d)")  # Matches "[RESULT] 4"
PAIRWISE_RE = re.compile(r"\[RESULT\]\s*(Neither|[ABNabn])", re.IGNORECASE)  # Matches "[RESULT] A", "B", "N", or "Neither"


def parse_pairwise_result(match: re.Match | None) -> str | None:
    """Parse a pairwise regex match into 'A', 'B', or 'tie'.

    Returns None if no match was found.
    """
    if match is None:
        return None
    token = match.group(1).upper()
    if token in ("N", "NEITHER"):
        return "tie"
    return token


def load_rubric(dimension: str, persona: bool = False) -> str:
    """Load the rubric text for a given dimension."""
    if persona and dimension == "informativeness":
        return (RUBRICS_DIR / "informativeness_persona.txt").read_text()
    return (RUBRICS_DIR / f"{dimension}.txt").read_text()


def load_prompt_template(
    prompt_file: str | None,
    default_path: Path,
    persona_path: Path | None = None,
    persona: bool = False,
) -> str:
    """Load a prompt template from a file path, falling back to defaults."""
    if prompt_file:
        return Path(prompt_file).read_text()
    if persona and persona_path:
        return persona_path.read_text()
    return default_path.read_text()


class BaseLLMMetric(BaseMetric):
    """Base class for metrics that use an LLM backend.

    Handles shared concerns: client initialization, lazy loading,
    and response logging.
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
        self._client = None

    def _load(self):
        """Lazily initialize the LLM client on first use."""
        if self._client is None:
            from persona_eval.llm_client import LLMClient

            self._client = LLMClient(
                provider=self._provider,
                model=self._model,
                api_key=self._api_key,
                base_url=self._base_url,
            )

    def _log_response(
        self,
        metric: str,
        prompt: str,
        response: str,
        parsed_result,
        **extra,
    ):
        """Log an LLM response if a logger is configured."""
        if self._response_logger:
            self._response_logger.log(
                metric=metric,
                prompt=prompt,
                response=response,
                parsed_result=parsed_result,
                **extra,
            )
