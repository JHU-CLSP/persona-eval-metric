"""Correlate automatic metric scores with human pairwise preferences."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_pairwise_agreement(
    preferences: pd.DataFrame,
    metric_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Compute how often each metric agrees with human pairwise preferences.

    Args:
        preferences: DataFrame from get_pairwise_preferences() with columns:
            annotator_id, query_index, preferred, other, comparison_type
        metric_scores: DataFrame with columns:
            query_index, label (A/B/C/D), and one column per metric score

    Returns:
        DataFrame with columns: metric, n_comparisons, n_agreements, agreement_rate
    """
    # Get metric column names (everything except query_index and label)
    score_cols = [
        c for c in metric_scores.columns if c not in ("query_index", "label")
    ]

    results = []
    for metric_col in score_cols:
        n_agree = 0
        n_total = 0

        for _, row in preferences.iterrows():
            qi = row["query_index"]
            pref_label = row["preferred"]
            other_label = row["other"]

            # Look up metric scores
            pref_scores = metric_scores[
                (metric_scores["query_index"] == qi)
                & (metric_scores["label"] == pref_label)
            ]
            other_scores = metric_scores[
                (metric_scores["query_index"] == qi)
                & (metric_scores["label"] == other_label)
            ]

            if pref_scores.empty or other_scores.empty:
                continue

            pref_val = pref_scores[metric_col].iloc[0]
            other_val = other_scores[metric_col].iloc[0]

            if pd.isna(pref_val) or pd.isna(other_val):
                continue

            n_total += 1
            if pref_val > other_val:
                n_agree += 1
            elif pref_val == other_val:
                n_agree += 0.5  # Tie counts as half agreement

        results.append(
            {
                "metric": metric_col,
                "n_comparisons": n_total,
                "n_agreements": n_agree,
                "agreement_rate": n_agree / n_total if n_total > 0 else float("nan"),
            }
        )

    return pd.DataFrame(results)


def _human_ranking(entry) -> dict[str, float]:
    """Derive a ranking of summaries A-D from human preference data.

    Returns {label: rank} where lower rank = better (1 = best).
    """
    labels = ["A", "B", "C", "D"]
    scores = {l: 0.0 for l in labels}

    # Round 1 winners get a point
    if entry.round1_ab and entry.round1_ab in labels:
        scores[entry.round1_ab] += 1
    if entry.round1_cd and entry.round1_cd in labels:
        scores[entry.round1_cd] += 1

    # Final winner gets 2 additional points
    if entry.final and entry.final in labels:
        scores[entry.final] += 2

    # Convert scores to ranks (higher score = lower/better rank)
    score_list = [(l, scores[l]) for l in labels]
    score_list.sort(key=lambda x: -x[1])

    ranks = {}
    i = 0
    while i < len(score_list):
        # Find tied group
        j = i + 1
        while j < len(score_list) and score_list[j][1] == score_list[i][1]:
            j += 1
        # Assign average rank to tied group
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[score_list[k][0]] = avg_rank
        i = j

    return ranks


def compute_rank_correlation(
    entries: list,
    metric_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-query rank correlation between human and metric rankings.

    Args:
        entries: List of AnnotationEntry objects.
        metric_scores: DataFrame with query_index, label, and metric score columns.

    Returns:
        DataFrame with columns: query_index, annotator_id, metric,
            kendall_tau, kendall_p, spearman_rho, spearman_p
    """
    score_cols = [
        c for c in metric_scores.columns if c not in ("query_index", "label")
    ]

    rows = []
    for entry in entries:
        if not entry.final:
            continue

        human_ranks = _human_ranking(entry)
        labels = ["A", "B", "C", "D"]
        human_rank_vec = [human_ranks[l] for l in labels]

        qi = entry.query_index
        qi_scores = metric_scores[metric_scores["query_index"] == qi]

        if len(qi_scores) < 4:
            continue

        for metric_col in score_cols:
            metric_vals = []
            for l in labels:
                row = qi_scores[qi_scores["label"] == l]
                if row.empty or pd.isna(row[metric_col].iloc[0]):
                    break
                metric_vals.append(row[metric_col].iloc[0])

            if len(metric_vals) != 4:
                continue

            # Metric rank: higher score = better = rank 1
            metric_rank_vec = stats.rankdata([-v for v in metric_vals])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", stats.ConstantInputWarning)

                # Kendall's tau-b (handles ties)
                tau, tau_p = stats.kendalltau(human_rank_vec, metric_rank_vec)

                # Spearman's rho
                rho, rho_p = stats.spearmanr(human_rank_vec, metric_rank_vec)

            rows.append(
                {
                    "query_index": qi,
                    "annotator_id": entry.annotator_id,
                    "metric": metric_col,
                    "kendall_tau": tau,
                    "kendall_p": tau_p,
                    "spearman_rho": rho,
                    "spearman_p": rho_p,
                }
            )

    return pd.DataFrame(rows)


def aggregate_correlations(per_query: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-query correlations into summary statistics per metric.

    Returns DataFrame with columns: metric, mean_tau, std_tau, median_tau,
        mean_rho, std_rho, median_rho, n_queries
    """
    if per_query.empty:
        return pd.DataFrame()

    def _agg(group):
        return pd.Series(
            {
                "mean_tau": group["kendall_tau"].mean(),
                "std_tau": group["kendall_tau"].std(),
                "median_tau": group["kendall_tau"].median(),
                "mean_rho": group["spearman_rho"].mean(),
                "std_rho": group["spearman_rho"].std(),
                "median_rho": group["spearman_rho"].median(),
                "n_queries": len(group),
            }
        )

    return per_query.groupby("metric").apply(_agg, include_groups=False).reset_index()
