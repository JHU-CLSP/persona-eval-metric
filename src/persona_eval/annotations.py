"""Load and normalize annotation data from zip or directory."""

from __future__ import annotations

import json
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class AnnotationEntry:
    """One annotator's response to one query."""

    annotator_id: str
    query_index: int
    query: str
    summaries: dict[str, str]  # {"A": "...", "B": "...", "C": "...", "D": "..."}
    metadata: dict[str, dict]  # {"A": {"model": ..., "has_pre_question": ...}, ...}
    paper_ids: list[str]
    round1_ab: str | None  # "A", "B", "N", or None
    round1_cd: str | None  # "C", "D", "N", or None
    final: str | None  # "A", "B", "C", "D", or None
    feedback: dict | None = None


@dataclass
class AnnotatorProfile:
    """Annotator metadata from users.json + pre_task_answers."""

    annotator_id: str
    name: str
    role: str = ""
    domain: str = ""
    info_needs: str = ""


def load_annotations(
    path: str | Path,
) -> tuple[list[AnnotatorProfile], list[AnnotationEntry]]:
    """Load annotations from a zip file or extracted directory.

    Args:
        path: Path to a .zip file or a directory containing users.json and results/.

    Returns:
        Tuple of (annotator profiles, annotation entries).
    """
    path = Path(path)

    if path.suffix == ".zip":
        tmp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp_dir)
        # The zip may contain a top-level directory or files directly
        extracted = Path(tmp_dir)
        # Check if there's a single subdirectory
        children = list(extracted.iterdir())
        if len(children) == 1 and children[0].is_dir():
            root = children[0]
        else:
            root = extracted
    elif path.is_dir():
        root = path
    else:
        raise ValueError(f"Path must be a .zip file or directory: {path}")

    profiles = _load_profiles(root)
    entries, pre_task_by_id = _load_entries(root)
    for profile in profiles:
        answers = pre_task_by_id.get(profile.annotator_id, {})
        profile.role = answers.get("role", "")
        profile.domain = answers.get("domain", "")
        profile.info_needs = answers.get("info_needs", "")
    return profiles, entries


def _load_profiles(root: Path) -> list[AnnotatorProfile]:
    users_path = root / "users.json"
    if not users_path.exists():
        return []

    with open(users_path) as f:
        users = json.load(f)

    profiles = []
    # users.json can be a dict keyed by annotator_id or a list of objects
    if isinstance(users, dict):
        for token, user_data in users.items():
            profiles.append(
                AnnotatorProfile(
                    annotator_id=token,
                    name=user_data.get("name", "") if isinstance(user_data, dict) else "",
                )
            )
    else:
        for user in users:
            profiles.append(
                AnnotatorProfile(
                    annotator_id=user.get("token", ""),
                    name=user.get("name", ""),
                )
            )
    return profiles


def _load_entries(
    root: Path,
) -> tuple[list[AnnotationEntry], dict[str, dict]]:
    results_dir = root / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"No results/ directory found in {root}")

    entries = []
    pre_task_by_id: dict[str, dict] = {}
    for json_file in sorted(results_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        annotator_id = data.get("token", json_file.stem)
        pre_task_by_id[annotator_id] = data.get("pre_task_answers", {}) or {}

        for ann in data.get("annotations", []):
            if not ann.get("summaries"):
                continue

            entries.append(
                AnnotationEntry(
                    annotator_id=annotator_id,
                    query_index=ann["query_index"],
                    query=ann["query"],
                    summaries=ann["summaries"],
                    metadata=ann.get("metadata", {}),
                    paper_ids=ann.get("paper_ids", []),
                    round1_ab=ann.get("round1_ab"),
                    round1_cd=ann.get("round1_cd"),
                    final=ann.get("final"),
                    feedback=ann.get("feedback"),
                )
            )

    return entries, pre_task_by_id


def get_pairwise_preferences(
    entries: list[AnnotationEntry],
    include_neither: bool = False,
) -> pd.DataFrame:
    """Extract pairwise preference pairs from annotation entries.

    Args:
        entries: List of AnnotationEntry objects.
        include_neither: If True, include "N" (neither) annotations as rows
            with preferred="N". These can be used with metric score thresholds
            to test whether a metric agrees that neither summary is better.

    Returns a DataFrame with columns:
        annotator_id, query_index, preferred, other, comparison_type
    Where preferred is "A"/"B"/"C"/"D" for a preference, or "N" for neither.
    """
    rows = []

    for entry in entries:
        base = {
            "annotator_id": entry.annotator_id,
            "query_index": entry.query_index,
        }

        # Round 1: A vs B
        if entry.round1_ab:
            if entry.round1_ab != "N":
                preferred = entry.round1_ab
                other = "B" if preferred == "A" else "A"
                rows.append(
                    {
                        **base,
                        "preferred": preferred,
                        "other": other,
                        "comparison_type": "round1",
                    }
                )
            elif include_neither:
                rows.append(
                    {
                        **base,
                        "preferred": "N",
                        "other": "A",  # pair_a
                        "pair_b": "B",
                        "comparison_type": "round1",
                    }
                )

        # Round 1: C vs D
        if entry.round1_cd:
            if entry.round1_cd != "N":
                preferred = entry.round1_cd
                other = "D" if preferred == "C" else "C"
                rows.append(
                    {
                        **base,
                        "preferred": preferred,
                        "other": other,
                        "comparison_type": "round1",
                    }
                )
            elif include_neither:
                rows.append(
                    {
                        **base,
                        "preferred": "N",
                        "other": "C",  # pair_a
                        "pair_b": "D",
                        "comparison_type": "round1",
                    }
                )

        # Final preference
        if entry.final:
            if entry.final in ("A", "B", "C", "D"):
                all_labels = {"A", "B", "C", "D"}
                for loser in all_labels - {entry.final}:
                    rows.append(
                        {
                            **base,
                            "preferred": entry.final,
                            "other": loser,
                            "comparison_type": "final",
                        }
                    )
            elif entry.final == "N" and include_neither:
                # For final "neither", we don't know which pair was compared
                # in the final round, so we skip it
                pass

    return pd.DataFrame(rows)
