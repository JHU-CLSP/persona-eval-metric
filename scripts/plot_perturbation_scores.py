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

# ---------------------------------------------------------------------------
# Display registries — edit these to control labels in the rendered figure.
# ``expected`` is one of {"increase", "decrease", "stable"}.
# Tests / metrics not listed here fall back to their raw column name.
# ---------------------------------------------------------------------------

TEST_DISPLAY: dict[str, dict[str, str]] = {
    "distractor_sentences":  {"label": "Distractor sentences",  "expected": "decrease"},
    "incremental_addition":  {"label": "Incremental addition",  "expected": "increase"},
    "lengthen_prose":        {"label": "Lengthen prose",        "expected": "stable"},
    "shorten_prose":         {"label": "Shorten prose",         "expected": "stable"},
    "different_audience":    {"label": "Different audience",    "expected": "decrease"},
}

METRIC_DISPLAY: dict[str, str] = {
    "rouge1_f":       "ROUGE-1",
    "rouge2_f":       "ROUGE-2",
    "rougeL_f":       "ROUGE-L",
    "bleu":           "BLEU",
    "chrf":           "chrF",
    "meteor":         "METEOR",
    "bertscore_p":    "BERTScore (P)",
    "bertscore_r":    "BERTScore (R)",
    "bertscore_f":    "BERTScore (F)",
    "coverage":       "Coverage",
    "density":        "Density",
    "compression":    "Compression",
    "length":         "Length",
    "syn_words":      "Syntactic (words)",
    "syn_sentences":  "Syntactic (sentences)",
}


def _test_title(test_name: str) -> str:
    info = TEST_DISPLAY.get(test_name)
    if info is None:
        return test_name
    return f"{info['label']} (expected: {info['expected']})"


def _metric_label(col: str) -> str:
    return METRIC_DISPLAY.get(col, col)


# Okabe-Ito colorblind-safe palette (8 colors, distinguishable for the
# three most common forms of color vision deficiency).
CB_PALETTE = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":"]


def _line_style(i: int) -> dict:
    """Return a (color, marker, linestyle) combo for the i-th line.

    Cycles colors fastest, then markers, then linestyles, so the first
    len(palette) lines have unique colors; beyond that, marker and dash
    pattern keep them distinguishable even in grayscale.
    """
    nc = len(CB_PALETTE)
    nm = len(MARKERS)
    return {
        "color":     CB_PALETTE[i % nc],
        "marker":    MARKERS[(i // nc) % nm],
        "linestyle": LINESTYLES[(i // (nc * nm)) % len(LINESTYLES)],
    }


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
            style = _line_style(j)
            ax.plot(stats.index, mean, label=_metric_label(metric),
                    linewidth=1.6, markersize=5, **style)
            ax.fill_between(stats.index, mean - sem, mean + sem,
                            alpha=0.15, color=style["color"], linewidth=0)
        ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.set_title(_test_title(test))
        ax.set_xlabel("perturbation level")
        ax.set_ylabel("score - baseline")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    empty_slots = [k for k in range(len(tests), rows * cols)]

    if handles and empty_slots:
        # Use the first empty subplot as a legend panel (bottom-right area).
        legend_ax = axes[empty_slots[0] // cols][empty_slots[0] % cols]
        legend_ax.axis("off")
        ncol = 2 if len(labels) > 6 else 1
        legend_ax.legend(handles, labels, loc="center", ncol=ncol, frameon=False,
                         fontsize="medium")
        for k in empty_slots[1:]:
            axes[k // cols][k % cols].set_visible(False)
        fig.tight_layout()
    else:
        for k in empty_slots:
            axes[k // cols][k % cols].set_visible(False)
        if handles:
            fig.legend(handles, labels, loc="lower center",
                       ncol=min(len(labels), 4),
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
            ax.plot(stats.index, stats["mean"], marker="o", color=CB_PALETTE[5])
            ax.fill_between(stats.index, stats["mean"] - sem, stats["mean"] + sem, alpha=0.2, color=CB_PALETTE[5])
            if i == 0:
                ax.set_title(_metric_label(metric), fontsize=9)
            if j == 0:
                ax.set_ylabel(_test_title(test), fontsize=9)
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
