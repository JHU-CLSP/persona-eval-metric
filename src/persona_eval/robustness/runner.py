"""Orchestrator for robustness testing: perturb, score, collect."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from persona_eval.metrics import get_metric
from persona_eval.robustness.dataset import SummarizationSample
from persona_eval.robustness.perturbations import (
    BasePerturbationTest,
    PerturbedSummary,
)

logger = logging.getLogger(__name__)


@dataclass
class PerturbationResult:
    """All perturbation levels for one (sample, test) pair."""

    sample_id: str
    test_name: str
    levels: list[PerturbedSummary]


# ---------------------------------------------------------------------------
# Phase 1: Generate perturbations
# ---------------------------------------------------------------------------


def generate_perturbations(
    samples: list[SummarizationSample],
    tests: list[BasePerturbationTest],
) -> list[PerturbationResult]:
    """Generate perturbation levels for every (sample, test) pair.

    Returns:
        List of ``PerturbationResult`` objects.
    """
    all_results: list[PerturbationResult] = []
    for test in tests:
        print(f"Generating perturbations: {test.name}")
        for sample in tqdm(samples, desc=test.name):
            levels = test.generate_levels(sample)
            all_results.append(
                PerturbationResult(
                    sample_id=sample.sample_id,
                    test_name=test.name,
                    levels=levels,
                )
            )
    return all_results


# ---------------------------------------------------------------------------
# Phase 2: Score perturbations
# ---------------------------------------------------------------------------


def score_perturbations(
    all_results: list[PerturbationResult],
    samples: list[SummarizationSample],
    metric_names: list[str],
    metric_kwargs: dict | None = None,
    output_dir: str | Path = "robustness_results",
    response_logger=None,
) -> pd.DataFrame:
    """Score every perturbed summary with every requested metric.

    Args:
        all_results: Perturbation results from ``generate_perturbations()``
            or ``load_perturbations()``.
        samples: The original samples (needed for source/reference text).
        metric_names: Names of metrics to evaluate (from the registry).
        metric_kwargs: Extra kwargs passed to ``get_metric()``.
        output_dir: Directory for output CSV files.
        response_logger: Optional ``ResponseLogger`` for LLM metrics.

    Returns:
        DataFrame with columns: sample_id, test_name, level, level_label,
        and one column per metric sub-score.
    """
    metric_kwargs = metric_kwargs or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    sample_lookup = {s.sample_id: s for s in samples}

    for metric_name in metric_names:
        print(f"Scoring with {metric_name}...")
        kwargs = dict(metric_kwargs)
        if response_logger is not None:
            kwargs["response_logger"] = response_logger
        metric = get_metric(metric_name, **kwargs)
        text_key = "source" if metric.is_reference_free else "reference"

        for result in tqdm(all_results, desc=metric_name):
            sample = sample_lookup[result.sample_id]
            compare_text = sample.source if text_key == "source" else sample.reference

            for ps in result.levels:
                row = {
                    "sample_id": result.sample_id,
                    "test_name": result.test_name,
                    "level": ps.level,
                    "level_label": ps.label,
                }
                try:
                    scores = metric.score(ps.text, compare_text)
                except Exception:
                    logger.warning(
                        "Metric %s failed on %s level %s",
                        metric_name, result.sample_id, ps.level,
                    )
                    scores = {}
                row.update(scores)
                rows.append(row)

    if not rows:
        scores_df = pd.DataFrame()
    else:
        scores_df = pd.DataFrame(rows)

    # Save raw scores
    out_path = output_dir / "robustness_scores.csv"
    scores_df.to_csv(out_path, index=False)
    print(f"Saved robustness scores to {out_path} ({len(scores_df)} rows)")

    return scores_df


# ---------------------------------------------------------------------------
# Serialization: save / load perturbations to disk
# ---------------------------------------------------------------------------


def save_perturbations(
    all_results: list[PerturbationResult],
    samples: list[SummarizationSample],
    output_dir: str | Path,
) -> Path:
    """Save perturbation results and samples to disk as JSONL.

    Creates ``{output_dir}/perturbations/`` with two files:
    ``samples.jsonl`` and ``perturbations.jsonl``.

    Returns:
        Path to the perturbations directory.
    """
    out = Path(output_dir) / "perturbations"
    out.mkdir(parents=True, exist_ok=True)

    # Save samples
    with open(out / "samples.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")

    # Save perturbation results
    with open(out / "perturbations.jsonl", "w") as f:
        for r in all_results:
            obj = {
                "sample_id": r.sample_id,
                "test_name": r.test_name,
                "levels": [asdict(lv) for lv in r.levels],
            }
            f.write(json.dumps(obj) + "\n")

    print(
        f"Saved {len(all_results)} perturbation results and "
        f"{len(samples)} samples to {out}"
    )
    return out


def load_perturbations(
    perturbations_dir: str | Path,
) -> tuple[list[PerturbationResult], list[SummarizationSample]]:
    """Load perturbation results and samples from disk.

    Args:
        perturbations_dir: Directory containing ``samples.jsonl`` and
            ``perturbations.jsonl`` (as created by ``save_perturbations``).

    Returns:
        Tuple of (perturbation_results, samples).
    """
    d = Path(perturbations_dir)

    samples: list[SummarizationSample] = []
    with open(d / "samples.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            samples.append(SummarizationSample(**obj))

    all_results: list[PerturbationResult] = []
    with open(d / "perturbations.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            levels = [PerturbedSummary(**lv) for lv in obj["levels"]]
            all_results.append(
                PerturbationResult(
                    sample_id=obj["sample_id"],
                    test_name=obj["test_name"],
                    levels=levels,
                )
            )

    print(
        f"Loaded {len(all_results)} perturbation results and "
        f"{len(samples)} samples from {d}"
    )
    return all_results, samples


# ---------------------------------------------------------------------------
# Combined pipeline (backward compatible)
# ---------------------------------------------------------------------------


def run_robustness(
    samples: list[SummarizationSample],
    tests: list[BasePerturbationTest],
    metric_names: list[str],
    metric_kwargs: dict | None = None,
    output_dir: str | Path = "robustness_results",
    response_logger=None,
) -> pd.DataFrame:
    """Run the full robustness pipeline.

    1. Generate perturbation levels for each (sample, test) pair.
    2. Score every perturbed summary with every requested metric.
    3. Return and save the raw scores DataFrame.

    Args:
        samples: Dataset converted to ``SummarizationSample`` objects.
        tests: Instantiated perturbation test objects.
        metric_names: Names of metrics to evaluate (from the registry).
        metric_kwargs: Extra kwargs passed to ``get_metric()`` (e.g. device,
            provider, model).
        output_dir: Directory for output CSV files.
        response_logger: Optional ``ResponseLogger`` for LLM metrics.

    Returns:
        DataFrame with columns: sample_id, test_name, level, level_label,
        and one column per metric sub-score.
    """
    all_results = generate_perturbations(samples, tests)

    return score_perturbations(
        all_results=all_results,
        samples=samples,
        metric_names=metric_names,
        metric_kwargs=metric_kwargs,
        output_dir=output_dir,
        response_logger=response_logger,
    )
