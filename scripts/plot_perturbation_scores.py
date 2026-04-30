#!/usr/bin/env python3
"""Plot how metric scores change across perturbation levels.

Reads ``robustness_scores.csv`` (located next to the perturbations
directory by default, or pointed to with ``--scores``) and renders one
subplot per perturbation test. Within each subplot, mean score is
plotted against perturbation level, with one line per metric and a
shaded band for ±1 standard error across samples.

Because metrics live on different scales, scores are normalized to the
mean score at the lowest level (delta from baseline) by default. Pass
``--no-normalize`` to plot raw means instead, in which case each metric
gets its own subplot to keep the y-axis meaningful.

Usage:
    python scripts/plot_perturbation_scores.py \\
        --perturbations-dir robustness_results/run_<ts>/perturbations \\
        --output perturbation_scores.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


META_COLS = {"sample_id", "test_name", "level", "level_label"}


def _resolve_scores_path(perturbations_dir: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    candidates = [
        perturbations_dir.parent / "robustness_scores.csv",
        perturbations_dir / "robustness_scores.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"robustness_scores.csv not found in {candidates[0]} or {candidates[1]}; "
        "pass --scores explicitly."
    )


def _per_level_stats(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean, std-error, and level_label for one metric, indexed by level."""
    sub = df[["level", "level_label", metric]].dropna(subset=[metric])
    if sub.empty:
        return sub
    grp = sub.groupby("level")[metric]
    out = pd.DataFrame({
        "mean": grp.mean(),
        "sem": grp.std(ddof=1) / np.sqrt(grp.count().clip(lower=1)),
        "n": grp.count(),
    }).sort_index()
    label_map = sub.drop_duplicates("level").set_index("level")["level_label"]
    out["level_label"] = label_map
    return out


def _grid_shape(n: int) -> tuple[int, int]:
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    return rows, cols


def plot_normalized(scores_df: pd.DataFrame, output: Path) -> None:
    """One subplot per test; one line per metric, normalized to baseline."""
    tests = sorted(scores_df["test_name"].unique())
    metrics = [c for c in scores_df.columns if c not in META_COLS]

    rows, cols = _grid_shape(len(tests))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows), squeeze=False)
    cmap = plt.get_cmap("tab20" if len(metrics) > 10 else "tab10")

    for i, test in enumerate(tests):
        ax = axes[i // cols][i % cols]
        test_df = scores_df[scores_df["test_name"] == test]
        for j, metric in enumerate(metrics):
            stats = _per_level_stats(test_df, metric)
            if stats.empty:
                continue
            baseline = stats["mean"].iloc[0]
            mean = stats["mean"] - baseline
            sem = stats["sem"].fillna(0.0)
            ax.plot(stats.index, mean, marker="o", label=metric, color=cmap(j % cmap.N))
            ax.fill_between(stats.index, mean - sem, mean + sem, alpha=0.15, color=cmap(j % cmap.N))
        ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.set_title(test)
        ax.set_xlabel("perturbation level")
        ax.set_ylabel("score - baseline")
        ax.grid(True, alpha=0.3)

    for k in range(len(tests), rows * cols):
        axes[k // cols][k % cols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
                   bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_raw(scores_df: pd.DataFrame, output: Path) -> None:
    """One subplot per (test, metric); raw mean score with SEM band."""
    tests = sorted(scores_df["test_name"].unique())
    metrics = [c for c in scores_df.columns if c not in META_COLS]

    rows = len(tests)
    cols = max(len(metrics), 1)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

    for i, test in enumerate(tests):
        test_df = scores_df[scores_df["test_name"] == test]
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            stats = _per_level_stats(test_df, metric)
            if stats.empty:
                ax.set_visible(False)
                continue
            sem = stats["sem"].fillna(0.0)
            ax.plot(stats.index, stats["mean"], marker="o", color="C0")
            ax.fill_between(stats.index, stats["mean"] - sem, stats["mean"] + sem, alpha=0.2, color="C0")
            if i == 0:
                ax.set_title(metric, fontsize=9)
            if j == 0:
                ax.set_ylabel(test, fontsize=9)
            ax.set_xlabel("level")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--perturbations-dir", type=Path, required=True,
                        help="Perturbations directory (used to locate robustness_scores.csv).")
    parser.add_argument("--scores", type=Path, default=None,
                        help="Path to robustness_scores.csv. Defaults to <perturbations-dir>/../robustness_scores.csv.")
    parser.add_argument("--output", type=Path, default=Path("perturbation_scores.png"),
                        help="Output image path (default: perturbation_scores.png).")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Plot raw mean scores instead of delta from baseline.")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Optional subset of metric columns to include.")
    parser.add_argument("--tests", nargs="+", default=None,
                        help="Optional subset of perturbation tests to include.")
    args = parser.parse_args()

    scores_path = _resolve_scores_path(args.perturbations_dir, args.scores)
    scores_df = pd.read_csv(scores_path)
    print(f"Loaded {len(scores_df)} rows from {scores_path}")

    if args.tests:
        scores_df = scores_df[scores_df["test_name"].isin(args.tests)]
    if args.metrics:
        keep = [c for c in scores_df.columns if c in META_COLS or c in args.metrics]
        scores_df = scores_df[keep]

    if scores_df.empty:
        raise SystemExit("No rows to plot after filtering.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.no_normalize:
        plot_raw(scores_df, args.output)
    else:
        plot_normalized(scores_df, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
