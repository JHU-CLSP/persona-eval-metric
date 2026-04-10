#!/usr/bin/env python3
"""Analyze metric scores from the persona-eval pipeline.

Reads a metric_scores.csv produced by `persona-eval compute-metrics` and
generates summary statistics and visualisation plots. Optionally loads
annotations to show examples of metric agreement/disagreement with humans.

Usage:
    python scripts/analyze_scores.py metric_scores.csv -o analysis/
    python scripts/analyze_scores.py metric_scores.csv --metrics rouge1_f bertscore_f supert
    python scripts/analyze_scores.py metric_scores.csv --annotations annotations.zip
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
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_columns(df: pd.DataFrame) -> list[str]:
    """Return metric score column names (everything except index columns)."""
    skip = {"query_index", "label", "annotator_id"}
    return [c for c in df.columns if c not in skip]


def _group_metrics(cols: list[str]) -> dict[str, list[str]]:
    """Group sub-metric columns by their parent metric prefix."""
    groups: dict[str, list[str]] = {}
    for c in cols:
        # Heuristic: split on first '_' boundary that matches a known prefix
        # e.g. "rouge1_f" -> "rouge", "llm_judge_relevance" -> "llm_judge"
        for prefix in (
            "rouge1", "rouge2", "rougeL",
            "bertscore",
            "llm_judge_rel", "llm_judge",
            "factscore",
            "summaqa",
            "syn",
            "percentage",
        ):
            if c.startswith(prefix):
                groups.setdefault(prefix, []).append(c)
                break
        else:
            groups.setdefault(c, []).append(c)
    return groups


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def print_summary_stats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Print and return descriptive statistics for each metric column."""
    stats_df = df[cols].describe().T
    stats_df["missing"] = df[cols].isna().sum()
    stats_df["missing_pct"] = (stats_df["missing"] / len(df) * 100).round(1)

    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)
    print(stats_df[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "missing_pct"]].to_string())
    return stats_df


def print_per_label_means(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Print mean score per summary label (A/B/C/D)."""
    means = df.groupby("label")[cols].mean().T
    print("\n" + "=" * 70)
    print("MEAN SCORES PER LABEL")
    print("=" * 70)
    print(means.round(4).to_string())
    return means


def print_per_query_variance(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Print how much scores vary within each query (discriminative power)."""
    per_query_std = df.groupby("query_index")[cols].std().mean()
    disc = per_query_std.sort_values(ascending=False)
    print("\n" + "=" * 70)
    print("DISCRIMINATIVE POWER (mean within-query std)")
    print("=" * 70)
    print(disc.round(4).to_string())
    return disc


def print_correlation_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Print pairwise Spearman correlations between metrics."""
    corr = df[cols].corr(method="spearman")
    print("\n" + "=" * 70)
    print("INTER-METRIC SPEARMAN CORRELATION")
    print("=" * 70)
    # Print only lower triangle for readability
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    display = corr.where(~mask).round(3)
    print(display.to_string())
    return corr


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_distributions(df: pd.DataFrame, cols: list[str], output_dir: Path):
    """Histogram + KDE for each metric score."""
    n = len(cols)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(cols):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=30, alpha=0.7, density=True, color="steelblue", edgecolor="white")
        if len(data) > 2 and data.std() > 0:
            try:
                kde_x = np.linspace(data.min(), data.max(), 200)
                kde = stats.gaussian_kde(data)
                ax.plot(kde_x, kde(kde_x), color="darkred", lw=1.5)
            except Exception:
                pass
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)

    for i in range(len(cols), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Score Distributions", fontsize=13, y=1.01)
    fig.tight_layout()
    path = output_dir / "distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_boxplots_by_label(df: pd.DataFrame, cols: list[str], output_dir: Path):
    """Boxplots comparing score distributions across labels A/B/C/D."""
    n = len(cols)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    labels_order = ["A", "B", "C", "D"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    for i, col in enumerate(cols):
        ax = axes[i]
        data = [df[df["label"] == l][col].dropna() for l in labels_order]
        bp = ax.boxplot(data, labels=labels_order, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)

    for i in range(len(cols), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Scores by Summary Label", fontsize=13, y=1.01)
    fig.tight_layout()
    path = output_dir / "boxplots_by_label.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_correlation_heatmap(corr: pd.DataFrame, output_dir: Path):
    """Heatmap of inter-metric Spearman correlations."""
    fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.6), max(6, len(corr) * 0.5)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")
    ax.set_title("Inter-Metric Correlation", fontsize=12)
    fig.tight_layout()
    path = output_dir / "correlation_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_discriminative_power(disc: pd.Series, output_dir: Path):
    """Bar chart of mean within-query std (discriminative power)."""
    fig, ax = plt.subplots(figsize=(max(8, len(disc) * 0.4), 5))
    disc_sorted = disc.sort_values(ascending=True)
    ax.barh(range(len(disc_sorted)), disc_sorted.values, color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(disc_sorted)))
    ax.set_yticklabels(disc_sorted.index, fontsize=8)
    ax.set_xlabel("Mean within-query std", fontsize=10)
    ax.set_title("Discriminative Power by Metric", fontsize=12)
    fig.tight_layout()
    path = output_dir / "discriminative_power.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_per_query_heatmap(df: pd.DataFrame, cols: list[str], output_dir: Path):
    """Heatmap of mean metric scores per query (queries x metrics)."""
    if len(cols) > 20:
        cols = cols[:20]  # limit for readability

    per_query = df.groupby("query_index")[cols].mean()
    if len(per_query) > 50:
        per_query = per_query.iloc[:50]  # limit for readability

    fig, ax = plt.subplots(figsize=(max(10, len(cols) * 0.5), max(6, len(per_query) * 0.25)))
    # Normalize each column to [0, 1] for comparable heatmap
    normed = per_query.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10))
    im = ax.imshow(normed.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(per_query)))
    ax.set_yticklabels(per_query.index, fontsize=7)
    ax.set_ylabel("Query Index")
    ax.set_title("Per-Query Metric Scores (normalized)", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Normalized score")
    fig.tight_layout()
    path = output_dir / "per_query_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Agreement / disagreement examples
# ---------------------------------------------------------------------------

def _get_score(df: pd.DataFrame, qi: int, label: str, metric_col: str) -> float | None:
    """Look up a single metric score."""
    row = df[(df["query_index"] == qi) & (df["label"] == label)]
    if row.empty:
        return None
    val = row[metric_col].iloc[0]
    return None if pd.isna(val) else float(val)


def print_agreement_examples(
    scores_df: pd.DataFrame,
    preferences: "pd.DataFrame",
    entries: list,
    cols: list[str],
    n_examples: int = 3,
    output_dir: Path | None = None,
):
    """Print concrete examples of agreement and disagreement with humans.

    For each metric, shows n_examples of agreements and n_examples of
    disagreements, including the summaries and scores involved.
    """
    # Build a lookup: (query_index, label) -> summary text
    summary_lookup: dict[tuple[int, str], str] = {}
    for entry in entries:
        for label, text in entry.summaries.items():
            summary_lookup[(entry.query_index, label)] = text

    # Only look at standard preferences (not "neither")
    standard = preferences[preferences["preferred"] != "N"]
    # Only round1 comparisons for cleaner examples
    round1 = standard[standard["comparison_type"] == "round1"]

    if round1.empty:
        print("\n  No round1 preferences available for examples.")
        return

    all_examples = []

    for metric_col in cols:
        agreements = []
        disagreements = []

        for _, row in round1.iterrows():
            qi = row["query_index"]
            pref = row["preferred"]
            other = row["other"]

            pref_score = _get_score(scores_df, qi, pref, metric_col)
            other_score = _get_score(scores_df, qi, other, metric_col)

            if pref_score is None or other_score is None:
                continue

            example = {
                "query_index": qi,
                "human_preferred": pref,
                "human_other": other,
                "metric": metric_col,
                "preferred_score": pref_score,
                "other_score": other_score,
                "score_diff": pref_score - other_score,
                "preferred_summary": summary_lookup.get((qi, pref), ""),
                "other_summary": summary_lookup.get((qi, other), ""),
            }

            if pref_score > other_score:
                agreements.append(example)
            elif pref_score < other_score:
                disagreements.append(example)

        # Sort by score difference magnitude for interesting examples
        agreements.sort(key=lambda x: x["score_diff"], reverse=True)
        disagreements.sort(key=lambda x: x["score_diff"])

        all_examples.append({
            "metric": metric_col,
            "agreements": agreements[:n_examples],
            "disagreements": disagreements[:n_examples],
        })

    # Print
    print("\n" + "=" * 70)
    print("AGREEMENT / DISAGREEMENT EXAMPLES (round1 comparisons)")
    print("=" * 70)

    for group in all_examples:
        metric_col = group["metric"]

        if group["agreements"]:
            print(f"\n--- {metric_col}: AGREEMENTS (metric agrees with human) ---")
            for i, ex in enumerate(group["agreements"], 1):
                print(f"\n  Example {i}: query {ex['query_index']}, "
                      f"human preferred {ex['human_preferred']} over {ex['human_other']}")
                print(f"    {metric_col}({ex['human_preferred']}) = {ex['preferred_score']:.4f}  >  "
                      f"{metric_col}({ex['human_other']}) = {ex['other_score']:.4f}  "
                      f"(diff: +{ex['score_diff']:.4f})")
                print(f"    Preferred: {ex['preferred_summary']}...")
                print(f"    Other:     {ex['other_summary']}...")

        if group["disagreements"]:
            print(f"\n--- {metric_col}: DISAGREEMENTS (metric disagrees with human) ---")
            for i, ex in enumerate(group["disagreements"], 1):
                print(f"\n  Example {i}: query {ex['query_index']}, "
                      f"human preferred {ex['human_preferred']} over {ex['human_other']}")
                print(f"    {metric_col}({ex['human_preferred']}) = {ex['preferred_score']:.4f}  <  "
                      f"{metric_col}({ex['human_other']}) = {ex['other_score']:.4f}  "
                      f"(diff: {ex['score_diff']:.4f})")
                print(f"    Preferred: {ex['preferred_summary']}...")
                print(f"    Other:     {ex['other_summary']}...")

        if not group["agreements"] and not group["disagreements"]:
            print(f"\n--- {metric_col}: no examples available ---")

    # Save to file
    if output_dir is not None:
        rows = []
        for group in all_examples:
            for kind, examples in [("agreement", group["agreements"]),
                                   ("disagreement", group["disagreements"])]:
                for ex in examples:
                    rows.append({**ex, "type": kind})
        if rows:
            examples_df = pd.DataFrame(rows)
            path = output_dir / "agreement_examples.csv"
            examples_df.to_csv(path, index=False)
            print(f"\n  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze metric scores from persona-eval pipeline",
    )
    parser.add_argument(
        "scores", help="Path to metric_scores.csv",
    )
    parser.add_argument(
        "-o", "--output-dir", default="analysis",
        help="Directory to save plots (default: analysis/)",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=None,
        help="Subset of metric columns to analyze (default: all)",
    )
    parser.add_argument(
        "--annotations", default=None,
        help="Path to annotations zip/dir (enables agreement/disagreement examples)",
    )
    parser.add_argument(
        "--n-examples", type=int, default=3,
        help="Number of agreement/disagreement examples per metric (default: 3)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation, print stats only",
    )
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.scores)
    print(f"Loaded {len(df)} rows from {args.scores}")
    print(f"  Queries: {df['query_index'].nunique()}")
    print(f"  Labels:  {sorted(df['label'].unique())}")
    if "annotator_id" in df.columns:
        print(f"  Annotators: {df['annotator_id'].nunique()}")

    # Select columns
    all_cols = _score_columns(df)
    if args.metrics:
        cols = [c for c in args.metrics if c in all_cols]
        missing = [c for c in args.metrics if c not in all_cols]
        if missing:
            print(f"  Warning: columns not found: {missing}")
    else:
        cols = all_cols

    if not cols:
        print("No metric columns found. Exiting.")
        sys.exit(1)

    print(f"  Analyzing {len(cols)} metric columns")

    # --- Summary statistics ---
    stats_df = print_summary_stats(df, cols)
    label_means = print_per_label_means(df, cols)
    disc = print_per_query_variance(df, cols)

    if len(cols) > 1:
        corr = print_correlation_matrix(df, cols)

    # --- Agreement / disagreement examples ---
    if args.annotations:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from persona_eval.annotations import load_annotations, get_pairwise_preferences

        _, entries = load_annotations(args.annotations)
        preferences = get_pairwise_preferences(entries)

        output_dir_for_examples = Path(args.output_dir) if not args.no_plots else None
        if output_dir_for_examples:
            output_dir_for_examples.mkdir(parents=True, exist_ok=True)

        print_agreement_examples(
            df, preferences, entries, cols,
            n_examples=args.n_examples,
            output_dir=output_dir_for_examples,
        )

    # --- Plots ---
    if not args.no_plots:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating plots in {output_dir}/...")
        plot_distributions(df, cols, output_dir)
        plot_boxplots_by_label(df, cols, output_dir)
        plot_discriminative_power(disc, output_dir)

        if len(cols) > 1:
            plot_correlation_heatmap(corr, output_dir)

        plot_per_query_heatmap(df, cols, output_dir)

        # Save stats to CSV
        stats_path = output_dir / "summary_stats.csv"
        stats_df.to_csv(stats_path)
        print(f"  Saved {stats_path}")

        label_path = output_dir / "per_label_means.csv"
        label_means.to_csv(label_path)
        print(f"  Saved {label_path}")

        disc_path = output_dir / "discriminative_power.csv"
        disc.to_frame("mean_within_query_std").to_csv(disc_path)
        print(f"  Saved {disc_path}")

        if len(cols) > 1:
            corr_path = output_dir / "inter_metric_correlation.csv"
            corr.to_csv(corr_path)
            print(f"  Saved {corr_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
