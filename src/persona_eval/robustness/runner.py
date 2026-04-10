"""Orchestrator for robustness testing: perturb, score, collect."""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
    metric_kwargs = metric_kwargs or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Generate perturbations ---
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

    # --- Phase 2: Score ---
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
