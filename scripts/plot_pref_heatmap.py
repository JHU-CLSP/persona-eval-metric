#!/usr/bin/env python3
"""Heatmap of human vs LLM-judge pairwise preferences.

Cross-tabulates pairwise preferences from human annotations against an
LLM-judge metric's pairwise_prefs.csv and renders a 3x3 confusion
heatmap with axes {Option A, Option B, Neither}.

Round 1 AB (A/B/N) and Round 1 CD (C/D/N) are normalized to a shared
{first option, second option, neither} axis and combined by default.
The final round is skipped because reconstructing which two summaries
the metric compared in the final requires the metric's intermediate
round-1 winners, which are not stored in pairwise_prefs.csv.

Usage:
    python scripts/plot_pref_heatmap.py \
        --annotations annotations.zip \
        --pairwise pairwise_prefs.csv \
        --output heatmap.png

    # only one round:
    python scripts/plot_pref_heatmap.py ... --comparison round1_ab

    # different metric column:
    python scripts/plot_pref_heatmap.py ... --metric llm_judge_relative_overall
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running from repo root without installing.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from persona_eval.annotations import load_annotations  # noqa: E402


LABELS = ["Option A", "Option B", "Neither"]

NORMALIZE = {
    "round1_ab": {"A": "Option A", "B": "Option B", "N": "Neither", "tie": "Neither"},
    "round1_cd": {"C": "Option A", "D": "Option B", "N": "Neither", "tie": "Neither"},
}


def build_human_prefs(entries) -> pd.DataFrame:
    rows = []
    for e in entries:
        if e.round1_ab:
            rows.append({
                "annotator_id": e.annotator_id,
                "query_index": e.query_index,
                "comparison": "round1_ab",
                "human_raw": e.round1_ab,
            })
        if e.round1_cd:
            rows.append({
                "annotator_id": e.annotator_id,
                "query_index": e.query_index,
                "comparison": "round1_cd",
                "human_raw": e.round1_cd,
            })
    return pd.DataFrame(rows)


def normalize_pref(comparison: str, value) -> str | None:
    if pd.isna(value):
        return None
    return NORMALIZE.get(comparison, {}).get(str(value))


def build_confusion(df: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(df["human"], df["metric"])
    return counts.reindex(index=LABELS, columns=LABELS, fill_value=0)


def plot_heatmap(matrix: pd.DataFrame, title: str, out_path: Path,
                 normalize_rows: bool = False) -> None:
    if normalize_rows:
        row_sums = matrix.sum(axis=1).replace(0, np.nan)
        display = matrix.div(row_sums, axis=0).fillna(0.0)
        fmt = "{:.2f}"
        cbar_label = "Row-normalized share"
    else:
        display = matrix.astype(float)
        fmt = "{:.0f}"
        cbar_label = "Count"

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(display.values, cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Metric preference")
    ax.set_ylabel("Human preference")
    ax.set_title(title)

    vmax = display.values.max() if display.values.size else 1.0
    threshold = vmax / 2.0
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            v = display.values[i, j]
            color = "white" if v > threshold else "black"
            count = matrix.values[i, j]
            label = fmt.format(v)
            if normalize_rows:
                label = f"{label}\n(n={int(count)})"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=10)

    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def agreement_stats(matrix: pd.DataFrame) -> dict:
    total = int(matrix.values.sum())
    if total == 0:
        return {"total": 0, "exact_agreement": 0.0, "off_diagonal": 0}
    diag = int(np.trace(matrix.values))
    return {
        "total": total,
        "exact_agreement": diag / total,
        "off_diagonal": total - diag,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True, help="Path to annotations zip or directory.")
    ap.add_argument("--pairwise", required=True, help="Path to pairwise_prefs.csv.")
    ap.add_argument("--metric", default="llm_judge_annotator",
                    help="Metric column name in pairwise_prefs.csv (default: llm_judge_annotator).")
    ap.add_argument("--comparison", choices=["round1_ab", "round1_cd", "both"], default="both",
                    help="Which round(s) to include (default: both).")
    ap.add_argument("--output", required=True, help="Output PNG path.")
    ap.add_argument("--normalize", action="store_true",
                    help="Display row-normalized shares instead of raw counts.")
    ap.add_argument("--split", action="store_true",
                    help="Also write per-comparison heatmaps next to the combined one.")
    args = ap.parse_args()

    pairwise_path = Path(args.pairwise)
    if not pairwise_path.exists():
        ap.error(f"pairwise file not found: {pairwise_path}")
    pairwise = pd.read_csv(pairwise_path)

    if args.metric not in pairwise.columns:
        ap.error(
            f"metric column {args.metric!r} not in {pairwise_path}. "
            f"Available: {[c for c in pairwise.columns if c not in ('query_index', 'comparison')]}"
        )

    _, entries = load_annotations(args.annotations)
    human_df = build_human_prefs(entries)
    if human_df.empty:
        ap.error("No round-1 human preferences found in annotations.")

    if args.comparison != "both":
        human_df = human_df[human_df["comparison"] == args.comparison]
        pairwise = pairwise[pairwise["comparison"] == args.comparison]

    metric_df = pairwise[["query_index", "comparison", args.metric]].rename(
        columns={args.metric: "metric_raw"}
    )

    merged = human_df.merge(metric_df, on=["query_index", "comparison"], how="inner")
    if merged.empty:
        ap.error(
            "No overlap between human annotations and metric pairwise prefs. "
            "Check that --pairwise and --annotations refer to the same queries."
        )

    merged["human"] = merged.apply(lambda r: normalize_pref(r["comparison"], r["human_raw"]), axis=1)
    merged["metric"] = merged.apply(lambda r: normalize_pref(r["comparison"], r["metric_raw"]), axis=1)
    dropped = merged[merged["human"].isna() | merged["metric"].isna()]
    if not dropped.empty:
        print(f"Dropping {len(dropped)} rows with unparseable labels "
              f"(human_raw={dropped['human_raw'].unique().tolist()[:5]}..., "
              f"metric_raw={dropped['metric_raw'].unique().tolist()[:5]}...)")
    merged = merged.dropna(subset=["human", "metric"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = build_confusion(merged)
    stats = agreement_stats(matrix)
    label = "Round 1 (AB + CD)" if args.comparison == "both" else args.comparison
    title = (
        f"Human vs {args.metric} - {label}\n"
        f"n={stats['total']}, exact agreement={stats['exact_agreement']:.1%}"
    )
    plot_heatmap(matrix, title, out_path, normalize_rows=args.normalize)

    if args.split and args.comparison == "both":
        for sub in ("round1_ab", "round1_cd"):
            sub_df = merged[merged["comparison"] == sub]
            if sub_df.empty:
                continue
            sub_matrix = build_confusion(sub_df)
            sub_stats = agreement_stats(sub_matrix)
            sub_path = out_path.with_name(f"{out_path.stem}.{sub}{out_path.suffix}")
            sub_title = (
                f"Human vs {args.metric} - {sub}\n"
                f"n={sub_stats['total']}, exact agreement={sub_stats['exact_agreement']:.1%}"
            )
            plot_heatmap(sub_matrix, sub_title, sub_path, normalize_rows=args.normalize)

    print("\nConfusion matrix (rows=human, cols=metric):")
    print(matrix)
    print(f"\nExact agreement: {stats['exact_agreement']:.3f}  (n={stats['total']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
