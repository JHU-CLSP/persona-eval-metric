"""Base dataset abstraction for summarization robustness testing.

Provides a unified ``SummarizationSample`` dataclass and adapter functions
to convert from various dataset formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SummarizationSample:
    """A single summarization example for robustness testing.

    This is the common currency across all robustness tests and dataset
    adapters.  Each sample pairs a source document with one summary and
    describes the intended audience.
    """

    sample_id: str
    source: str
    summary: str
    audience: str
    reference: str = ""
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Adapter: persona-eval annotation format
# ---------------------------------------------------------------------------


def load_from_persona_eval(
    entries,
    source_texts: dict[int, str],
    reference_texts: dict[int, str],
    profiles_by_id: dict | None = None,
) -> list[SummarizationSample]:
    """Convert persona-eval annotations into ``SummarizationSample`` objects.

    Each ``(query_index, label)`` pair becomes one sample.  The audience
    description is derived from the annotator's profile when available.

    Args:
        entries: List of ``AnnotationEntry`` objects.
        source_texts: ``{query_index: concatenated_abstracts}``.
        reference_texts: ``{query_index: concatenated_titles}``.
        profiles_by_id: Optional ``{annotator_id: AnnotatorProfile}`` mapping.

    Returns:
        Deduplicated list of ``SummarizationSample`` objects.
    """
    seen: set[tuple[int, str]] = set()
    samples: list[SummarizationSample] = []

    for entry in entries:
        for label in ("A", "B", "C", "D"):
            if label not in entry.summaries:
                continue

            key = (entry.query_index, label)
            if key in seen:
                continue
            seen.add(key)

            audience = _build_audience(entry.annotator_id, profiles_by_id)

            samples.append(
                SummarizationSample(
                    sample_id=f"q{entry.query_index}_{label}",
                    source=source_texts.get(entry.query_index, ""),
                    summary=entry.summaries[label],
                    audience=audience,
                    reference=reference_texts.get(entry.query_index, ""),
                    metadata={
                        "query_index": entry.query_index,
                        "label": label,
                        "annotator_id": entry.annotator_id,
                    },
                )
            )

    return samples


def _build_audience(
    annotator_id: str,
    profiles_by_id: dict | None,
) -> str:
    """Build an audience description from an annotator profile."""
    if not profiles_by_id or annotator_id not in profiles_by_id:
        return "general reader"
    profile = profiles_by_id[annotator_id]
    parts = []
    if getattr(profile, "role", ""):
        parts.append(profile.role)
    if getattr(profile, "domain", ""):
        parts.append(f"in {profile.domain}")
    if getattr(profile, "info_needs", ""):
        parts.append(f"({profile.info_needs})")
    return " ".join(parts) if parts else "general reader"
