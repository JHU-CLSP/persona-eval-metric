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


def parse_bullet_list(response: str) -> list[str]:
    """Parse bullet-pointed items from an LLM response.

    Recognizes lines starting with ``- `` or ``* ``.
    """
    items = []
    for line in response.strip().splitlines():
        line = line.strip()
        if line.startswith("- "):
            line = line[2:].strip()
        elif line.startswith("* "):
            line = line[2:].strip()
        else:
            continue
        if line:
            items.append(line)
    return items


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

    def cache_config(self) -> dict:
        return {
            **super().cache_config(),
            "provider": self._provider,
            "model": self._model,
        }


# Default dimensions used by both absolute and pairwise LLM judges.
LLM_JUDGE_DIMENSIONS = ("relevance", "coherence", "consistency", "fluency", "informativeness")


class BaseDimensionalLLMMetric(BaseLLMMetric):
    """LLM judge that iterates over a fixed set of rubric dimensions.

    Subclasses pick one of two calling conventions:
      * absolute: call :meth:`_score_dim` per dimension with ``SCORE_RE``
      * pairwise: call :meth:`_score_dim` per dimension with ``PAIRWISE_RE``

    Shared responsibilities handled here:
      * loading the prompt template and per-dimension rubrics,
      * running one LLM call per dimension with consistent formatting,
      * logging the response via :meth:`_log_response`,
      * reporting ``cache_config`` that reflects persona/prompt-file state.
    """

    _metric_log_name: str = "llm_judge"
    _dimensions: tuple[str, ...] = LLM_JUDGE_DIMENSIONS

    def __init__(
        self,
        default_prompt_path: Path,
        persona_prompt_path: Path,
        prompt_file: str | None = None,
        persona: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._persona = persona
        self._prompt_file = prompt_file
        self._prompt_template = load_prompt_template(
            prompt_file, default_prompt_path, persona_prompt_path, persona=persona,
        )
        self._rubrics = {
            dim: load_rubric(dim, persona=persona) for dim in self._dimensions
        }

    @property
    def needs_persona(self) -> bool:
        return self._persona

    @property
    def is_reference_free(self) -> bool:
        return True

    def cache_config(self) -> dict:
        return {
            **super().cache_config(),
            "persona": self._persona,
            "prompt_file": self._prompt_file,
        }

    def _score_dim(
        self,
        dimension: str,
        prompt_vars: dict,
        pattern: re.Pattern,
        parser,
        persona_kwargs: dict | None,
    ):
        """Run one LLM call for a dimension and return the parsed result.

        ``parser`` is invoked on the regex ``Match`` (or ``None`` when
        nothing matched); it returns the final value to store.
        """
        rubric = self._rubrics[dimension]
        pk = persona_kwargs or {}

        query = pk.get("query", "")
        fmt = {
            **prompt_vars,
            "dimension": dimension,
            "query_section": f"\nQuery: {query}\n" if query else "\n",
        }

        if pk:
            rubric = rubric.format(**pk)
            fmt.update(pk)
        fmt["rubric"] = rubric

        prompt = self._prompt_template.format(**fmt)
        response_text = self._client.generate(prompt)
        match = pattern.search(response_text)
        result = parser(match)

        self._log_response(
            metric=self._metric_log_name,
            prompt=prompt,
            response=response_text,
            parsed_result=result,
            dimension=dimension,
        )
        return result


class MultiStepLLMMetric(BaseLLMMetric):
    """LLM metric that orchestrates multiple prompt templates in sequence.

    Provides :meth:`_run_step` (format template → call LLM → parse →
    log) and :meth:`_classify_positive` (yes/no substring match used by
    the persona precision/recall binary checks). Subclasses own the
    step ordering and control flow.
    """

    _metric_log_name: str = "multi_step_llm"

    @property
    def is_reference_free(self) -> bool:
        return True

    @property
    def needs_persona(self) -> bool:
        return True

    @property
    def needs_query(self) -> bool:
        return True

    def _run_step(
        self,
        step: str,
        template: str,
        parser,
        *,
        max_tokens: int | None = None,
        log_extra: dict | None = None,
        **template_kwargs,
    ):
        """Format ``template`` with kwargs, call the LLM, parse, log. Returns parsed value."""
        prompt = template.format(**template_kwargs)
        response = (
            self._client.generate(prompt, max_tokens=max_tokens)
            if max_tokens is not None
            else self._client.generate(prompt)
        )
        parsed = parser(response)
        self._log_response(
            metric=self._metric_log_name,
            prompt=prompt,
            response=response,
            parsed_result=parsed,
            step=step,
            **(log_extra or {}),
        )
        return parsed

    @staticmethod
    def _classify_positive(response: str, positive_term: str) -> bool:
        """True iff ``positive_term`` appears without its negation.

        Handles both single- and two-word "not <term>" negations.
        """
        lower = response.lower()
        return positive_term in lower and f"not {positive_term}" not in lower
