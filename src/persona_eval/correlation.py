"""Correlate automatic metric scores with human pairwise preferences."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def _check_preference_agreement(
    row: pd.Series,
    metric_col: str,
    metric_scores: pd.DataFrame,
    threshold: float | None,
) -> float | None:
    """Check if a metric agrees with a single human preference.

    Returns:
        1.0 for agreement, 0.5 for tie, 0.0 for disagreement,
        or None if the comparison should be skipped (missing data).
    """
    qi = row["query_index"]
    pref_label = row["preferred"]

    if pref_label == "N":
        if threshold is None:
            return None

        label_a = row["other"]
        label_b = row.get("pair_b")
        if pd.isna(label_b) if label_b is None else not label_b:
            return None

        scores_a = metric_scores[
            (metric_scores["query_index"] == qi)
            & (metric_scores["label"] == label_a)
        ]
        scores_b = metric_scores[
            (metric_scores["query_index"] == qi)
            & (metric_scores["label"] == label_b)
        ]

        if scores_a.empty or scores_b.empty:
            return None

        val_a = scores_a[metric_col].iloc[0]
        val_b = scores_b[metric_col].iloc[0]

        if pd.isna(val_a) or pd.isna(val_b):
            return None

        return 1.0 if abs(val_a - val_b) < threshold else 0.0
    else:
        other_label = row["other"]

        pref_scores = metric_scores[
            (metric_scores["query_index"] == qi)
            & (metric_scores["label"] == pref_label)
        ]
        other_scores = metric_scores[
            (metric_scores["query_index"] == qi)
            & (metric_scores["label"] == other_label)
        ]

        if pref_scores.empty or other_scores.empty:
            return None

        pref_val = pref_scores[metric_col].iloc[0]
        other_val = other_scores[metric_col].iloc[0]

        if pd.isna(pref_val) or pd.isna(other_val):
            return None

        if pref_val > other_val:
            return 1.0
        elif pref_val == other_val:
            return 0.5
        else:
            return 0.0


def _build_pairwise_lookup(
    pairwise_prefs: pd.DataFrame,
) -> dict[tuple, dict]:
    """Index raw pairwise preferences for fast lookup.

    Returns:
        Dict mapping (query_index, comparison) to a dict of
        {sub_metric: winner_label_or_tie}.
    """
    lookup: dict[tuple, dict] = {}
    for _, row in pairwise_prefs.iterrows():
        key = (row["query_index"], row["comparison"])
        vals = {
            col: row[col]
            for col in pairwise_prefs.columns
            if col not in ("query_index", "comparison")
        }
        lookup[key] = vals
    return lookup


def _check_pairwise_preference(
    human_row: pd.Series,
    metric_col: str,
    pairwise_lookup: dict[tuple, dict],
) -> float | None:
    """Check if a raw LLM pairwise preference agrees with a human preference.

    Resolves the comparison by matching the human preference against
    the LLM's round1 and final comparisons:
    - round1 A-vs-B or C-vs-D: direct lookup
    - final winner vs final loser: direct lookup from final
    - final winner vs round loser (own bracket): from round1
    - final winner vs round loser (other bracket): transitive
      (round1 + final must both agree)

    Returns 1.0 (agree), 0.5 (tie), 0.0 (disagree), or None (skip).
    """
    qi = human_row["query_index"]
    pref = human_row["preferred"]
    other = human_row["other"]
    comp_type = human_row.get("comparison_type", "")

    if pref == "N":
        # "Neither" — pairwise metrics return A/B/tie, check for tie
        if comp_type == "round1":
            if {other, human_row.get("pair_b", "")} & {"A", "B"} == {"A", "B"}:
                data = pairwise_lookup.get((qi, "round1_ab"), {})
            else:
                data = pairwise_lookup.get((qi, "round1_cd"), {})
            llm_winner = data.get(metric_col)
            if llm_winner is None:
                return None
            return 1.0 if llm_winner == "tie" else 0.0
        return None  # can't resolve "neither" for final

    # Standard preference
    pair = frozenset({pref, other})

    if comp_type == "round1":
        # Direct round1 comparison
        if pair == frozenset({"A", "B"}):
            data = pairwise_lookup.get((qi, "round1_ab"), {})
        elif pair == frozenset({"C", "D"}):
            data = pairwise_lookup.get((qi, "round1_cd"), {})
        else:
            return None

        llm_winner = data.get(metric_col)
        if llm_winner is None:
            return None
        if llm_winner == pref:
            return 1.0
        elif llm_winner == "tie":
            return 0.5
        return 0.0

    elif comp_type == "final":
        # Determine what LLM compared in the final
        final_data = pairwise_lookup.get((qi, "final"), {})
        ab_data = pairwise_lookup.get((qi, "round1_ab"), {})
        cd_data = pairwise_lookup.get((qi, "round1_cd"), {})

        llm_final = final_data.get(metric_col)
        llm_ab = ab_data.get(metric_col)
        llm_cd = cd_data.get(metric_col)

        if llm_final is None or llm_ab is None or llm_cd is None:
            return None

        # Who were the finalists in the LLM's tournament?
        llm_ab_winner = llm_ab if llm_ab != "tie" else "A"  # tiebreak
        llm_cd_winner = llm_cd if llm_cd != "tie" else "C"
        llm_finalists = {llm_ab_winner, llm_cd_winner}

        if pref in llm_finalists and other in llm_finalists:
            # Direct final comparison
            if llm_final == pref:
                return 1.0
            elif llm_final == "tie":
                return 0.5
            return 0.0

        elif pref in llm_finalists and other not in llm_finalists:
            # Human prefers a finalist over a round loser
            # Check: LLM's round1 agrees the loser lost their round
            if other in ("A", "B"):
                round_winner = llm_ab
            else:
                round_winner = llm_cd

            if round_winner == other:
                # LLM thinks the "loser" actually won their round — disagree
                return 0.0
            elif round_winner == "tie":
                return 0.5

            # LLM agrees the round loser lost. Now check final.
            if llm_final == pref:
                return 1.0
            elif llm_final == "tie":
                return 0.5
            # Final winner is not pref, but pref's bracket winner still
            # beat the loser in round1, so transitively pref > loser
            # only if pref IS the bracket winner who beat the loser
            if other in ("A", "B") and pref == llm_cd_winner:
                # pref is from CD bracket, other lost AB round
                # pref beat their bracket (CD), final winner beat AB winner
                # but pref lost to final winner — can we say pref > other?
                # Only if final_winner == pref's opponent's bracket winner
                # This is ambiguous — fall back to disagree
                return 0.0
            elif other in ("C", "D") and pref == llm_ab_winner:
                return 0.0

            return 1.0  # pref is bracket winner who beat the round loser

        else:
            # Human prefers a non-finalist — shouldn't happen in normal data
            return None

    return None


def compute_pairwise_agreement(
    preferences: pd.DataFrame,
    metric_scores: pd.DataFrame,
    neither_thresholds: dict[str, float] | None = None,
    strict: bool = False,
    pairwise_prefs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute how often each metric agrees with human pairwise preferences.

    Args:
        preferences: DataFrame from get_pairwise_preferences() with columns:
            annotator_id, query_index, preferred, other, comparison_type.
        metric_scores: DataFrame with columns:
            query_index, label (A/B/C/D), and one column per metric score.
        neither_thresholds: Optional per-metric thresholds for "neither".
        strict: If True, final-round comparisons auto-disagree when the
            metric got any round1 comparison wrong for that annotator/query.
        pairwise_prefs: Optional DataFrame of raw LLM pairwise preferences
            from pairwise metrics (e.g. llm_judge_relative). Has columns:
            query_index, comparison, and one column per sub-metric with
            values "A"/"B"/"C"/"D" or "tie". When provided, these sub-metrics
            use direct preference matching instead of score comparison.

    Returns:
        DataFrame with columns: metric, n_comparisons, n_agreements, agreement_rate
    """
    if neither_thresholds is None:
        neither_thresholds = {}

    # Build lookup for raw pairwise preferences
    pairwise_lookup = None
    pairwise_cols = set()
    if pairwise_prefs is not None and not pairwise_prefs.empty:
        pairwise_lookup = _build_pairwise_lookup(pairwise_prefs)
        pairwise_cols = {
            c for c in pairwise_prefs.columns
            if c not in ("query_index", "comparison")
        }

    # Get metric column names
    score_cols = [
        c for c in metric_scores.columns if c not in ("query_index", "label", "annotator_id")
    ]
    # Also include pairwise-only columns not in score_cols
    all_metric_cols = list(dict.fromkeys(score_cols + sorted(pairwise_cols - set(score_cols))))

    results = []
    for metric_col in all_metric_cols:
        n_agree = 0
        n_total = 0
        threshold = neither_thresholds.get(metric_col)
        use_pairwise = metric_col in pairwise_cols and pairwise_lookup is not None

        # In strict mode, first pass: check round1 agreement
        round1_agreed: dict[tuple, bool] = {}
        if strict:
            round1_rows = preferences[preferences["comparison_type"] == "round1"]
            for _, row in round1_rows.iterrows():
                key = (row["annotator_id"], row["query_index"])
                if use_pairwise:
                    result = _check_pairwise_preference(
                        row, metric_col, pairwise_lookup,
                    )
                else:
                    result = _check_preference_agreement(
                        row, metric_col, metric_scores, threshold
                    )
                if result is not None and result < 1.0:
                    round1_agreed[key] = False
                elif key not in round1_agreed:
                    round1_agreed[key] = result is None or result >= 1.0

        for _, row in preferences.iterrows():
            if use_pairwise:
                result = _check_pairwise_preference(
                    row, metric_col, pairwise_lookup,
                )
            else:
                result = _check_preference_agreement(
                    row, metric_col, metric_scores, threshold
                )
            if result is None:
                continue

            n_total += 1

            if strict and row.get("comparison_type") == "final":
                key = (row["annotator_id"], row["query_index"])
                if not round1_agreed.get(key, True):
                    continue

            n_agree += result

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
