"""Orchestrator for robustness testing: perturb, score, collect."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from persona_eval.metrics import get_metric
from persona_eval.metrics.cache import MetricCache, config_hash
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
    cache: MetricCache | None = None,
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
        cache: Optional ``MetricCache``. When provided, per-metric results
            are read from / written to the cache. The on-disk CSV is also
            used as a checkpoint: already-computed ``(sample_id, test_name,
            level)`` rows with non-null metric columns are not recomputed.

    Returns:
        DataFrame with columns: sample_id, test_name, level, level_label,
        and one column per metric sub-score.
    """
    metric_kwargs = metric_kwargs or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "robustness_scores.csv"

    # Load prior CSV to enable skip-on-re-run.
    existing_df = pd.read_csv(out_path) if out_path.exists() else None
    if existing_df is not None:
        print(f"Found existing {out_path} ({len(existing_df)} rows); will skip already-computed metrics")

    sample_lookup = {s.sample_id: s for s in samples}

    # Pre-index existing_df for fast lookups by (sample_id, test_name, level).
    existing_index = None
    if existing_df is not None and not existing_df.empty:
        existing_index = existing_df.set_index(["sample_id", "test_name", "level"])

    per_metric_dfs: list[pd.DataFrame] = []
    if existing_df is not None and not existing_df.empty:
        per_metric_dfs.append(existing_df.copy())

    for metric_name in metric_names:
        kwargs = dict(metric_kwargs)
        if response_logger is not None:
            kwargs["response_logger"] = response_logger
        metric = get_metric(metric_name, **kwargs)
        text_key = "source" if metric.is_reference_free else "reference"
        cfg_hash = config_hash(metric.cache_config())

        if cache is not None:
            cache.reset_counters()

        # Probe the first (result, level) pair to learn sub-metric column names.
        first_key = None
        first_scores: dict | None = None
        sub_metric_cols: list[str] = []
        for result in all_results:
            sample = sample_lookup[result.sample_id]
            compare_text = sample.source if text_key == "source" else sample.reference
            for ps in result.levels:
                try:
                    first_scores = _score_cached_robustness(
                        metric, metric_name, cfg_hash, ps.text, compare_text, cache,
                    )
                except Exception:
                    first_scores = {}
                sub_metric_cols = list(first_scores.keys())
                first_key = (result.sample_id, result.test_name, ps.level, ps.label, compare_text)
                break
            if first_key is not None:
                break

        if first_key is None:
            continue

        # Skip-on-re-run: if every (sample_id, test_name, level) already has
        # non-null values for sub_metric_cols in existing_df, skip this metric.
        if _robustness_metric_already_complete(
            existing_index, all_results, sub_metric_cols,
        ):
            print(f"Skipping {metric_name} (already in output CSV)")
            continue

        print(f"Scoring with {metric_name}...")

        rows: list[dict] = []
        rows.append({
            "sample_id": first_key[0],
            "test_name": first_key[1],
            "level": first_key[2],
            "level_label": first_key[3],
            **first_scores,
        })

        first_result_id = first_key[0]
        first_test_name = first_key[1]
        first_level = first_key[2]
        probed = False

        for result in tqdm(all_results, desc=metric_name):
            sample = sample_lookup[result.sample_id]
            compare_text = sample.source if text_key == "source" else sample.reference
            for ps in result.levels:
                if not probed and result.sample_id == first_result_id \
                        and result.test_name == first_test_name \
                        and ps.level == first_level:
                    probed = True
                    continue
                try:
                    scores = _score_cached_robustness(
                        metric, metric_name, cfg_hash, ps.text, compare_text, cache,
                    )
                except Exception:
                    logger.warning(
                        "Metric %s failed on %s level %s",
                        metric_name, result.sample_id, ps.level,
                    )
                    scores = {}
                rows.append({
                    "sample_id": result.sample_id,
                    "test_name": result.test_name,
                    "level": ps.level,
                    "level_label": ps.label,
                    **scores,
                })

        if cache is not None and (cache.hits or cache.misses):
            print(f"  cache: {cache.hits} hits / {cache.misses} misses")

        if rows:
            per_metric_dfs.append(pd.DataFrame(rows))

        # Incremental save after each metric completes.
        scores_df = _merge_robustness_dfs(per_metric_dfs)
        if not scores_df.empty:
            scores_df.to_csv(out_path, index=False)

    scores_df = _merge_robustness_dfs(per_metric_dfs)
    scores_df.to_csv(out_path, index=False)
    print(f"Saved robustness scores to {out_path} ({len(scores_df)} rows)")

    return scores_df


def _score_cached_robustness(
    metric,
    metric_name: str,
    cfg_hash: str,
    summary: str,
    source: str,
    cache: MetricCache | None,
) -> dict:
    if cache is not None:
        hit = cache.get(metric_name, cfg_hash, summary, source, None)
        if hit is not None:
            return hit
    scores = metric.score(summary, source)
    if cache is not None:
        cache.put(metric_name, cfg_hash, summary, source, None, scores)
    return scores


def _robustness_metric_already_complete(
    existing_index,
    all_results: list[PerturbationResult],
    sub_metric_cols: list[str],
) -> bool:
    if existing_index is None or not sub_metric_cols:
        return False
    if any(c not in existing_index.columns for c in sub_metric_cols):
        return False
    for result in all_results:
        for ps in result.levels:
            key = (result.sample_id, result.test_name, ps.level)
            if key not in existing_index.index:
                return False
            row = existing_index.loc[key]
            if hasattr(row, "ndim") and row.ndim > 1:
                row = row.iloc[0]
            for col in sub_metric_cols:
                if pd.isna(row[col]):
                    return False
    return True


def _merge_robustness_dfs(per_metric_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge per-metric DataFrames on (sample_id, test_name, level, level_label)."""
    if not per_metric_dfs:
        return pd.DataFrame()
    merged = per_metric_dfs[0]
    merge_cols = ["sample_id", "test_name", "level", "level_label"]
    for df in per_metric_dfs[1:]:
        cols = [c for c in merge_cols if c in merged.columns and c in df.columns]
        overlap = [c for c in df.columns if c in merged.columns and c not in cols]
        if overlap:
            merged = merged.drop(columns=overlap)
        merged = merged.merge(df, on=cols, how="outer")
    return merged


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
    cache: MetricCache | None = None,
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
        cache=cache,
    )
