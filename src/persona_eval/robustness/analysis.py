"""Analyze robustness test results: effect sizes, direction checks, reports."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from persona_eval.robustness.perturbations import BasePerturbationTest


def analyze_robustness(
    scores_df: pd.DataFrame,
    tests: list[BasePerturbationTest],
) -> pd.DataFrame:
    """Compute per-(test, metric) robustness statistics.

    For each combination of test and metric sub-score:

    * **Direction**: Spearman correlation of level vs. score (averaged
      across samples).
    * **Direction match**: whether the observed direction agrees with
      the test's ``expected_direction``.
    * **Effect size**: Cohen's d between baseline (level 0) and
      final level scores.
    * **Monotonicity**: fraction of consecutive-level pairs where the
      score change matches the expected direction.

    Args:
        scores_df: Raw scores from :func:`run_robustness`.  Must contain
            columns ``sample_id``, ``test_name``, ``level``, ``level_label``.
            All other columns are treated as metric sub-scores.
        tests: The perturbation test objects (used for ``expected_direction``).

    Returns:
        DataFrame with one row per (test_name, metric) pair.
    """
    meta_cols = {"sample_id", "test_name", "level", "level_label"}
    metric_cols = [c for c in scores_df.columns if c not in meta_cols]

    test_directions = {t.name: t.expected_direction for t in tests}
    rows: list[dict] = []

    for test_name, test_group in scores_df.groupby("test_name"):
        expected = test_directions.get(test_name, "unknown")

        for metric_col in metric_cols:
            sub = test_group[["sample_id", "level", metric_col]].dropna()
            if sub.empty:
                continue

            # Aggregate across samples: mean score at each level
            level_means = sub.groupby("level")[metric_col].mean().sort_index()
            levels_arr = np.array(level_means.index, dtype=float)
            scores_arr = np.array(level_means.values, dtype=float)

            # Spearman correlation of level vs. mean score
            if len(levels_arr) >= 3:
                rho, p_val = stats.spearmanr(levels_arr, scores_arr)
            elif len(levels_arr) == 2:
                # With only 2 points, use sign of difference
                diff = scores_arr[1] - scores_arr[0]
                rho = 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)
                p_val = float("nan")
            else:
                rho, p_val = float("nan"), float("nan")

            # Determine actual direction
            actual = _classify_direction(rho, p_val)

            # Effect size: baseline vs. final
            baseline = sub[sub["level"] == sub["level"].min()][metric_col]
            final = sub[sub["level"] == sub["level"].max()][metric_col]
            effect = _effect_size(baseline.values, final.values)

            # Monotonicity
            mono = _monotonicity(level_means, expected)

            rows.append({
                "test_name": test_name,
                "metric": metric_col,
                "expected_direction": expected,
                "actual_direction": actual,
                "direction_match": actual == expected,
                "spearman_rho": round(rho, 4) if not np.isnan(rho) else None,
                "p_value": round(p_val, 4) if not np.isnan(p_val) else None,
                "mean_score_baseline": round(float(baseline.mean()), 4),
                "mean_score_final": round(float(final.mean()), 4),
                "effect_size": round(effect, 4) if not np.isnan(effect) else None,
                "monotonicity": round(mono, 4),
            })

    return pd.DataFrame(rows)


def _classify_direction(rho: float, p_val: float) -> str:
    """Map a Spearman rho + p-value to increase / decrease / stable."""
    if np.isnan(rho):
        return "stable"
    # For 2-point comparisons p_val is NaN; use a threshold on rho instead
    sig = (not np.isnan(p_val) and p_val < 0.05) or np.isnan(p_val)
    if sig and rho > 0.1:
        return "increase"
    if sig and rho < -0.1:
        return "decrease"
    return "stable"


def _effect_size(baseline: np.ndarray, final: np.ndarray) -> float:
    """Cohen's d between baseline and final score distributions."""
    if len(baseline) == 0 or len(final) == 0:
        return float("nan")
    pooled_std = np.sqrt(
        (np.var(baseline, ddof=1) + np.var(final, ddof=1)) / 2
    )
    if pooled_std == 0:
        return 0.0
    return (np.mean(final) - np.mean(baseline)) / pooled_std


def _monotonicity(
    level_means: pd.Series,
    expected_direction: str,
) -> float:
    """Fraction of consecutive-level transitions matching expected direction.

    For ``"stable"``, a transition is correct if the absolute change is
    less than 5% of the baseline score (or 0.01 for near-zero baselines).
    """
    values = level_means.values
    if len(values) < 2:
        return 1.0

    baseline_mag = max(abs(values[0]), 0.01)
    tolerance = 0.05 * baseline_mag
    correct = 0
    total = len(values) - 1

    for i in range(total):
        diff = values[i + 1] - values[i]
        if expected_direction == "increase":
            correct += diff > 0
        elif expected_direction == "decrease":
            correct += diff < 0
        else:  # stable
            correct += abs(diff) < tolerance

    return correct / total if total > 0 else 1.0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_robustness_report(analysis_df: pd.DataFrame) -> None:
    """Print a formatted robustness analysis to stdout."""
    if analysis_df.empty:
        print("No robustness results to report.")
        return

    for test_name, group in analysis_df.groupby("test_name"):
        expected = group["expected_direction"].iloc[0]
        print(f"\n{'=' * 60}")
        print(f"Test: {test_name}  (expected: {expected})")
        print("=" * 60)

        display_cols = [
            "metric", "actual_direction", "direction_match",
            "spearman_rho", "effect_size", "monotonicity",
            "mean_score_baseline", "mean_score_final",
        ]
        cols = [c for c in display_cols if c in group.columns]
        print(group[cols].to_string(index=False))

    # Summary
    total = len(analysis_df)
    matches = analysis_df["direction_match"].sum()
    print(f"\n{'=' * 60}")
    print(f"Overall: {matches}/{total} metric-test pairs match expected direction")
    print(f"Match rate: {matches / total:.1%}" if total > 0 else "")
    print("=" * 60)
