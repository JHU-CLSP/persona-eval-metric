#!/usr/bin/env python3
"""Plot the distribution of scores for absolute (non-pairwise) metrics.

Reads metric_scores.csv and creates distribution plots specifically for
absolute metrics, filtering out pairwise metrics. Generates histograms
with KDE overlays to visualize score distributions.

Usage:
    python scripts/plot_absolute_distributions.py metric_scores.csv -o analysis/absolute_dist.png
    python scripts/plot_absolute_distributions.py metric_scores.csv -o analysis/ --metric-names rouge1_f bertscore_f supert
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# Allow running from repo root without installing.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from persona_eval.metrics.base import get_metric, list_metrics  # noqa: E402


def _score_columns(df: pd.DataFrame) -> list[str]:
    """Return metric score column names (everything except index columns)."""
    skip = {"query_index", "label", "annotator_id"}
    return [c for c in df.columns if c not in skip]


def _get_absolute_metrics() -> set[str]:
    """Return set of non-pairwise metric names available in the system."""
    absolute = set()
    for metric_name in list_metrics():
        try:
            m = get_metric(metric_name)
            if not m.is_pairwise:
                absolute.add(metric_name)
        except Exception:
            # Skip metrics that can't be instantiated
            pass
    return absolute


def _filter_absolute_columns(cols: list[str]) -> list[str]:
    """Filter column names to only those corresponding to absolute metrics."""
    # Try to get registered absolute metrics for more precise filtering
    absolute_metrics = _get_absolute_metrics()
    
    # Heuristic: match column prefixes to metric names
    absolute_cols = []
    for col in cols:
        # Check if column starts with an absolute metric name
        for metric_name in absolute_metrics:
            if col.startswith(metric_name):
                absolute_cols.append(col)
                break
        # Fallback heuristic: exclude known pairwise metrics
        if col not in absolute_cols:
            pairwise_keywords = {"judge", "annotator"}
            if not any(kw in col.lower() for kw in pairwise_keywords):
                absolute_cols.append(col)
    
    return absolute_cols


def plot_absolute_distributions(
    df: pd.DataFrame,
    cols: list[str],
    output_path: Path,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Create histogram + KDE plots for absolute metric scores.
    
    Args:
        df: DataFrame with metric scores
        cols: List of metric column names
        output_path: Path to save PNG file
        figsize: Optional figure size (default: auto-calculated)
    """
    if not cols:
        print("No absolute metric columns to plot")
        return

    # Auto-calculate figure size if not provided
    if figsize is None:
        ncols = min(4, len(cols))
        nrows = (len(cols) + ncols - 1) // ncols
        figsize = (4 * ncols, 3 * nrows)

    fig, axes = plt.subplots(
        (len(cols) + 3) // 4,
        min(4, len(cols)),
        figsize=figsize
    )
    axes = np.array(axes).flatten() if len(cols) > 1 else np.array([axes])

    for i, col in enumerate(cols):
        ax = axes[i]
        data = df[col].dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(col, fontsize=9)
            continue

        # Histogram
        ax.hist(
            data,
            bins=30,
            alpha=0.7,
            density=True,
            color="steelblue",
            edgecolor="white",
        )

        # KDE overlay
        if len(data) > 2 and data.std() > 0:
            try:
                kde_x = np.linspace(data.min(), data.max(), 200)
                kde = scipy_stats.gaussian_kde(data)
                ax.plot(kde_x, kde(kde_x), color="darkred", lw=1.5, label="KDE")
            except Exception:
                pass

        # Statistics annotation
        mean = data.mean()
        median = data.median()
        ax.axvline(mean, color="orange", linestyle="--", lw=1.5, label=f"Mean: {mean:.3f}")
        ax.axvline(median, color="green", linestyle=":", lw=1.5, label=f"Median: {median:.3f}")

        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.set_xlabel("Score", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)

    # Hide unused subplots
    for i in range(len(cols), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Absolute Metric Score Distributions", fontsize=13, y=1.00)
    fig.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "scores",
        help="Path to metric_scores.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="analysis/absolute_distributions.png",
        help="Output PNG file path (default: analysis/absolute_distributions.png)",
    )
    parser.add_argument(
        "--metric-names",
        nargs="+",
        default=None,
        help="Specific metric columns to plot (default: auto-detect absolute metrics)",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=int,
        default=None,
        help="Figure size (width height in inches)",
    )
    args = parser.parse_args()

    # Load data
    scores_path = Path(args.scores)
    if not scores_path.exists():
        parser.error(f"scores file not found: {scores_path}")

    df = pd.read_csv(scores_path)
    print(f"Loaded {len(df)} rows from {scores_path}")
    print(f"  Queries: {df['query_index'].nunique()}")
    print(f"  Labels: {sorted(df['label'].unique())}")

    # Select columns
    all_cols = _score_columns(df)
    if args.metric_names:
        cols = [c for c in args.metric_names if c in all_cols]
        missing = [c for c in args.metric_names if c not in all_cols]
        if missing:
            print(f"  Warning: columns not found: {missing}")
    else:
        cols = _filter_absolute_columns(all_cols)

    if not cols:
        print("  No absolute metric columns found.")
        sys.exit(1)

    print(f"  Plotting {len(cols)} absolute metric columns: {cols[:5]}{'...' if len(cols) > 5 else ''}")

    # Generate plot
    output_path = Path(args.output)
    figsize = tuple(args.figsize) if args.figsize else None
    plot_absolute_distributions(df, cols, output_path, figsize=figsize)

    return 0


if __name__ == "__main__":
    sys.exit(main())
