"""Shared LLM client supporting vLLM (local) and TogetherAI backends.

Both backends expose OpenAI-compatible APIs, so we use the openai SDK
as a unified client. This module also contains ``ResponseLogger``, the
JSONL writer that records every prompt/response pair emitted by LLM
metrics.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_PROVIDER_DEFAULTS = {
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "api_key": "EMPTY",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
    },
}


class LLMClient:
    """Unified LLM client for vLLM and TogetherAI."""

    def __init__(
        self,
        provider: str = "vllm",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        if provider not in _PROVIDER_DEFAULTS:
            raise ValueError(
                f"Unknown provider '{provider}'. Choose from: {list(_PROVIDER_DEFAULTS)}"
            )
        if model is None:
            raise ValueError("model is required for LLM-based metrics")

        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        defaults = _PROVIDER_DEFAULTS[provider]
        self._base_url = base_url or defaults["base_url"]

        if api_key:
            self._api_key = api_key
        elif "api_key" in defaults:
            self._api_key = defaults["api_key"]
        else:
            self._api_key = os.environ.get(defaults["api_key_env"], "")
            if not self._api_key:
                raise ValueError(
                    f"API key required for provider '{provider}'. "
                    f"Set --llm-api-key or the {defaults['api_key_env']} env var."
                )

        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
            )
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a single completion.

        Args:
            prompt: User message content.
            system_prompt: Optional system message.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.

        Returns:
            The assistant's response text.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        return response.choices[0].message.content

    def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Generate a completion and parse the response as JSON.

        Extracts JSON from the response, handling cases where the model
        wraps the JSON in markdown code fences.
        """
        text = self.generate(prompt, system_prompt, temperature, max_tokens)
        return _parse_json_response(text)


def _parse_json_response(text: str) -> dict:
    """Extract and parse JSON from an LLM response.

    Uses regex to find the first JSON object in the response, which is
    robust to preamble text, markdown code fences, and trailing commentary.
    """
    # Match the first { ... } block (greedy inner match to handle nested braces)
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}")


class ResponseLogger:
    """Append-only JSONL logger for LLM interactions.

    Each entry records the metric, prompt, full response text, and the
    parsed result, along with arbitrary metadata (e.g. query_index,
    dimension). One logger instance is shared across all LLM metrics
    within a single CLI invocation.
    """

    def __init__(self, log_dir: str | Path):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = self._log_dir / f"llm_responses_{timestamp}.jsonl"
        self._file = open(self._path, "a", encoding="utf-8")
        logger.info("Logging LLM responses to %s", self._path)

    def log(
        self,
        metric: str,
        prompt: str,
        response: str,
        parsed_result: str | float | dict | None = None,
        **metadata,
    ):
        entry = {
            "metric": metric,
            "prompt": prompt,
            "response": response,
            "parsed_result": parsed_result,
            **metadata,
        }
        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._file.flush()

    @property
    def path(self) -> Path:
        return self._path

    def close(self):
        self._file.close()
