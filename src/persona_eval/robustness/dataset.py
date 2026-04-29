"""Base dataset abstraction for summarization robustness testing.

Provides a unified ``SummarizationSample`` dataclass and adapter functions
to convert from various dataset formats including HuggingFace datasets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Dataset registry for HuggingFace datasets
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, dict] = {
    "arxiv": {
        "hf_id": "armanc/scientific_papers",
        "hf_config": "arxiv",
        "default_split": "test",
        "source_col": "article",
        "summary_col": "abstract",
        "reference_col": None,
        "description": "Arxiv scientific paper summarization",
        "intended_audience": "researcher in the field",
        "available": True,
    },
    "pubmed": {
        "hf_id": "armanc/scientific_papers",
        "hf_config": "pubmed",
        "default_split": "test",
        "source_col": "article",
        "summary_col": "abstract",
        "reference_col": None,
        "description": "PubMed biomedical paper summarization",
        "intended_audience": "biomedical researcher",
        "available": True,
    },
    "scitldr": {
        "hf_id": "allenai/scitldr",
        "hf_config": "AIC",
        "default_split": "test",
        "source_col": "source",
        "summary_col": "target",
        "reference_col": None,
        "description": "SciTLDR scientific paper TLDRs",
        "intended_audience": "researcher in the field",
        "available": True,
    },
    "elife": {
        "hf_id": "tomasg25/scientific_lay_summarisation",
        "hf_config": "elife",
        "default_split": "test",
        "source_col": "article",
        "summary_col": "summary",
        "reference_col": "title",
        "description": "eLife journal lay summaries",
        "intended_audience": "non-expert layperson",
        "available": True,
    },
    "plos": {
        "hf_id": "tomasg25/scientific_lay_summarisation",
        "hf_config": "plos",
        "default_split": "test",
        "source_col": "article",
        "summary_col": "summary",
        "reference_col": "title",
        "description": "PLOS journal lay summaries",
        "intended_audience": "non-expert layperson",
        "available": True,
    },
    "mup": {
        "hf_id": "allenai/mup",
        "hf_config": None,
        "default_split": "validation",
        "source_col": "text",
        "summary_col": "summary",
        "reference_col": None,
        "description": "Multi-perspective scientific paper summarization (no test split)",
        "intended_audience": "researcher in the field",
        "available": True,
    },
    "cdsr": {
        "hf_id": None,
        "hf_config": None,
        "default_split": "test",
        "source_col": None,
        "summary_col": None,
        "reference_col": None,
        "description": "Cochrane Database of Systematic Reviews",
        "intended_audience": "clinician",
        "available": False,
    },
    "eureka": {
        "hf_id": None,
        "hf_config": None,
        "default_split": "test",
        "source_col": None,
        "summary_col": None,
        "reference_col": None,
        "description": "EurekAlert scientific press release summarization",
        "intended_audience": "journalist",
        "available": False,
    },
    "cells": {
        "hf_id": None,
        "hf_config": None,
        "default_split": "test",
        "source_col": None,
        "summary_col": None,
        "reference_col": None,
        "description": "CELLS scientific summarization",
        "intended_audience": "researcher in the field",
        "available": False,
    },
    "scinews": {
        "hf_id": "dongqi-me/SciNews",
        "hf_config": None,
        "default_split": "test",
        "source_col": "Paper_Body",
        "summary_col": "News_Body",
        "reference_col": "News_Title",
        "description": "SciNews scientific news report generation (dongqi-me/SciNews)",
        "intended_audience": "general news reader",
        "available": True,
    },
    "longsumm": {
        "hf_id": None,
        "hf_config": None,
        "default_split": "test",
        "source_col": None,
        "summary_col": None,
        "reference_col": None,
        "description": "Long scientific document summarization",
        "intended_audience": "researcher in the field",
        "available": False,
    },
}


def list_available_datasets() -> list[str]:
    """Return names of datasets that can be loaded from HuggingFace."""
    return [name for name, cfg in DATASET_REGISTRY.items() if cfg["available"]]


def load_from_huggingface(
    dataset_name: str,
    split: str | None = None,
    num_samples: int | None = None,
    seed: int = 42,
) -> list[SummarizationSample]:
    """Load a summarization dataset from HuggingFace and convert to samples.

    Args:
        dataset_name: Key in ``DATASET_REGISTRY`` (e.g. ``"arxiv"``).
        split: Dataset split to load. Defaults to the registry default
            (typically ``"test"``).
        num_samples: If set, randomly subsample to this many examples.
        seed: Random seed for subsampling.

    Returns:
        List of ``SummarizationSample`` objects.

    Raises:
        ValueError: If the dataset is unknown or not available on HuggingFace.
    """
    from datasets import load_dataset

    name_lower = dataset_name.lower()
    if name_lower not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available datasets: {available}"
        )

    cfg = DATASET_REGISTRY[name_lower]
    if not cfg["available"]:
        raise ValueError(
            f"Dataset '{dataset_name}' is not available on HuggingFace. "
            f"Description: {cfg['description']}. "
            f"To add support, implement a loader and update DATASET_REGISTRY."
        )

    split = split or cfg["default_split"]
    logger.info(
        "Loading dataset %s (HF: %s, config: %s, split: %s)",
        dataset_name, cfg["hf_id"], cfg["hf_config"], split,
    )

    try:
        ds = load_dataset(
            cfg["hf_id"], cfg["hf_config"], split=split,
            trust_remote_code=True,  # needed for some legacy HF dataset scripts
        )
    except (TypeError, RuntimeError):
        # Newer datasets versions removed trust_remote_code support;
        # fall back to loading without it.
        ds = load_dataset(cfg["hf_id"], cfg["hf_config"], split=split)

    if num_samples and num_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(num_samples))

    samples = []
    for idx, row in enumerate(ds):
        source = _extract_text(row, cfg["source_col"], name_lower)
        summary = _extract_text(row, cfg["summary_col"], name_lower)

        if not source or not summary:
            continue

        reference = ""
        if cfg["reference_col"] and cfg["reference_col"] in row:
            reference = row[cfg["reference_col"]] or ""

        samples.append(
            SummarizationSample(
                sample_id=f"{name_lower}_{split}_{idx}",
                source=source,
                summary=summary,
                audience=cfg.get("intended_audience") or "general reader",
                reference=reference,
                metadata={
                    "dataset": name_lower,
                    "split": split,
                    "original_index": idx,
                },
            )
        )

    logger.info("Loaded %d samples from %s/%s", len(samples), dataset_name, split)
    return samples


def _extract_text(row: dict, col: str, dataset_name: str) -> str:
    """Extract text from a dataset row, handling list-valued columns."""
    value = row.get(col)
    if value is None:
        return ""

    # SciTLDR: source is a list of sentences, target is a list of TLDRs
    if dataset_name == "scitldr":
        if col == "source" and isinstance(value, list):
            return " ".join(value)
        if col == "target" and isinstance(value, list):
            return value[0] if value else ""

    if isinstance(value, list):
        return " ".join(str(v) for v in value)

    return str(value)
