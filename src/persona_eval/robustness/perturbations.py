"""Perturbation tests for summarization metric robustness evaluation.

Each test generates a sequence of increasingly-perturbed summaries and
declares whether scores should increase, decrease, or stay stable.
"""

from __future__ import annotations

import hashlib
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from persona_eval.robustness.dataset import SummarizationSample

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------


@dataclass
class PerturbedSummary:
    """One level of perturbation applied to a summary."""

    level: int
    label: str
    text: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sentence splitting utility
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\"\u201c])"  # split after sentence-ending punct + space before capital
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using a lightweight regex heuristic."""
    text = text.strip()
    if not text:
        return []
    parts = _SENT_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BasePerturbationTest(ABC):
    """Abstract base for robustness perturbation tests."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this test."""

    @property
    @abstractmethod
    def expected_direction(self) -> str:
        """``"decrease"``, ``"increase"``, or ``"stable"``."""

    @property
    def requires_llm(self) -> bool:
        return False

    @abstractmethod
    def generate_levels(
        self,
        sample: SummarizationSample,
    ) -> list[PerturbedSummary]:
        """Produce an ordered list of perturbed summaries for *sample*."""


# ---------------------------------------------------------------------------
# Test 1: Distractor sentences
# ---------------------------------------------------------------------------


class DistractorSentenceTest(BasePerturbationTest):
    """Add distractor sentences randomly sampled from a different document.

    Scores are expected to **decrease** as more noise is injected.
    """

    def __init__(
        self,
        distractor_pool: list[str],
        max_distractors: int = 5,
        seed: int = 42,
        **kwargs,
    ):
        self._pool = distractor_pool
        self._max = max_distractors
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "distractor_sentences"

    @property
    def expected_direction(self) -> str:
        return "decrease"

    def generate_levels(
        self,
        sample: SummarizationSample,
    ) -> list[PerturbedSummary]:
        # Collect distractor sentences from other documents
        all_distractor_sents: list[str] = []
        for doc in self._pool:
            if doc == sample.source:
                continue
            all_distractor_sents.extend(split_sentences(doc))
        if not all_distractor_sents:
            all_distractor_sents = ["This is an unrelated distractor sentence."]

        levels = [PerturbedSummary(level=0, label="original", text=sample.summary)]

        # Use a copy of the RNG so that each sample is reproducible but
        # independent (seeded by sample_id).
        rng = random.Random(self._rng.random() + hash(sample.sample_id))
        chosen = rng.sample(
            all_distractor_sents,
            min(self._max, len(all_distractor_sents)),
        )

        for i, distractor in enumerate(chosen, start=1):
            text = levels[-1].text + " " + distractor
            levels.append(
                PerturbedSummary(
                    level=i,
                    label=f"{i}_distractor{'s' if i > 1 else ''}",
                    text=text,
                    metadata={"distractor": distractor},
                )
            )
        return levels


# ---------------------------------------------------------------------------
# Test 2: Incremental sentence addition
# ---------------------------------------------------------------------------


class IncrementalAdditionTest(BasePerturbationTest):
    """Add back one sentence at a time from the original summary.

    Scores are expected to **increase** with each added sentence.
    """

    @property
    def name(self) -> str:
        return "incremental_addition"

    @property
    def expected_direction(self) -> str:
        return "increase"

    def generate_levels(
        self,
        sample: SummarizationSample,
    ) -> list[PerturbedSummary]:
        sentences = split_sentences(sample.summary)
        levels = [PerturbedSummary(level=0, label="0_sentences", text="")]
        for i in range(1, len(sentences) + 1):
            text = " ".join(sentences[:i])
            levels.append(
                PerturbedSummary(
                    level=i,
                    label=f"{i}_sentence{'s' if i > 1 else ''}",
                    text=text,
                )
            )
        return levels


# ---------------------------------------------------------------------------
# LLM-based test helpers
# ---------------------------------------------------------------------------


def _prompt_hash(template: str) -> str:
    """Short hash of a prompt template for cache keying."""
    return hashlib.sha256(template.encode()).hexdigest()[:16]


class _LLMPerturbationTest(BasePerturbationTest):
    """Mixin for tests that require an LLM client and cache."""

    def __init__(self, llm_client, cache, **kwargs):
        self._client = llm_client
        self._cache = cache

    @property
    def requires_llm(self) -> bool:
        return True

    def _generate_with_cache(
        self,
        prompt_template: str,
        sample: SummarizationSample,
        level: int | str,
        format_kwargs: dict,
    ) -> str:
        """Call the LLM or return a cached result."""
        ph = _prompt_hash(prompt_template)
        cached = self._cache.get(
            self.name, sample.sample_id, level, self._client.model, ph,
        )
        if cached is not None:
            return cached

        prompt = prompt_template.format(**format_kwargs)
        result = self._client.generate(prompt)

        self._cache.put(
            self.name, sample.sample_id, level, self._client.model, ph,
            result,
        )
        return result


# ---------------------------------------------------------------------------
# Test 3: Lengthen prose
# ---------------------------------------------------------------------------


class LengthenProseTest(_LLMPerturbationTest):
    """Ask an LLM to expand the summary without adding new information.

    Scores are expected to stay **stable**.
    """

    def __init__(self, llm_client, cache, prompt_file: str | None = None, **kwargs):
        super().__init__(llm_client, cache, **kwargs)
        if prompt_file:
            self._template = Path(prompt_file).read_text()
        else:
            self._template = (PROMPTS_DIR / "robustness_lengthen.txt").read_text()

    @property
    def name(self) -> str:
        return "lengthen_prose"

    @property
    def expected_direction(self) -> str:
        return "stable"

    def generate_levels(
        self,
        sample: SummarizationSample,
    ) -> list[PerturbedSummary]:
        levels = [PerturbedSummary(level=0, label="original", text=sample.summary)]

        lengthened = self._generate_with_cache(
            self._template,
            sample,
            level=1,
            format_kwargs={"source": sample.source, "summary": sample.summary},
        )
        levels.append(
            PerturbedSummary(level=1, label="lengthened", text=lengthened.strip())
        )
        return levels


# ---------------------------------------------------------------------------
# Test 4: Shorten prose
# ---------------------------------------------------------------------------


class ShortenProseTest(_LLMPerturbationTest):
    """Ask an LLM to condense the summary without removing information.

    Scores are expected to stay **stable**.
    """

    def __init__(self, llm_client, cache, prompt_file: str | None = None, **kwargs):
        super().__init__(llm_client, cache, **kwargs)
        if prompt_file:
            self._template = Path(prompt_file).read_text()
        else:
            self._template = (PROMPTS_DIR / "robustness_shorten.txt").read_text()

    @property
    def name(self) -> str:
        return "shorten_prose"

    @property
    def expected_direction(self) -> str:
        return "stable"

    def generate_levels(
        self,
        sample: SummarizationSample,
    ) -> list[PerturbedSummary]:
        levels = [PerturbedSummary(level=0, label="original", text=sample.summary)]

        shortened = self._generate_with_cache(
            self._template,
            sample,
            level=1,
            format_kwargs={"source": sample.source, "summary": sample.summary},
        )
        levels.append(
            PerturbedSummary(level=1, label="shortened", text=shortened.strip())
        )
        return levels


# ---------------------------------------------------------------------------
# Test 5: Different audience
# ---------------------------------------------------------------------------

DEFAULT_TARGET_AUDIENCES = [
    "undergraduate student",
    "journalist",
    "domain expert in the field",
]


class DifferentAudienceTest(_LLMPerturbationTest):
    """Rewrite the summary for a different target audience.

    Scores are expected to **decrease** because the information emphasis
    changes to serve a different reader.
    """

    def __init__(
        self,
        llm_client,
        cache,
        target_audiences: list[str] | None = None,
        prompt_file: str | None = None,
        **kwargs,
    ):
        super().__init__(llm_client, cache, **kwargs)
        self._audiences = target_audiences or DEFAULT_TARGET_AUDIENCES
        if prompt_file:
            self._template = Path(prompt_file).read_text()
        else:
            self._template = (PROMPTS_DIR / "robustness_audience.txt").read_text()

    @property
    def name(self) -> str:
        return "different_audience"

    @property
    def expected_direction(self) -> str:
        return "decrease"

    def generate_levels(
        self,
        sample: SummarizationSample,
    ) -> list[PerturbedSummary]:
        levels = [PerturbedSummary(level=0, label="original", text=sample.summary)]

        for i, audience in enumerate(self._audiences, start=1):
            rewritten = self._generate_with_cache(
                self._template,
                sample,
                level=audience,  # use audience string as level key for caching
                format_kwargs={
                    "source": sample.source,
                    "summary": sample.summary,
                    "original_audience": sample.audience,
                    "target_audience": audience,
                },
            )
            levels.append(
                PerturbedSummary(
                    level=i,
                    label=f"audience_{audience.replace(' ', '_')}",
                    text=rewritten.strip(),
                    metadata={"target_audience": audience},
                )
            )
        return levels


# ---------------------------------------------------------------------------
# Registry / convenience
# ---------------------------------------------------------------------------

ALL_TESTS = {
    "distractor_sentences": DistractorSentenceTest,
    "incremental_addition": IncrementalAdditionTest,
    "lengthen_prose": LengthenProseTest,
    "shorten_prose": ShortenProseTest,
    "different_audience": DifferentAudienceTest,
}
