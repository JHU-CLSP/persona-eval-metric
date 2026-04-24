"""Correlate automatic metric scores with human pairwise preferences."""

from __future__ import annotations

import warnings

import pandas as pd
from scipy import stats


def _build_score_index(metric_scores: pd.DataFrame) -> dict[tuple, pd.Series]:
    """Index metric scores by ``(query_index, label)`` for O(1) lookup.

    The previous implementation filtered the whole DataFrame once per
    comparison, which was quadratic in the number of rows.
    """
    if metric_scores.empty:
        return {}
    return {
        (qi, lbl): row for (qi, lbl), row in
        metric_scores.set_index(["query_index", "label"]).iterrows()
    }


def _score_agreement(
    pref_label: str,
    other_label: str,
    qi: int,
    metric_col: str,
    score_index: dict[tuple, pd.Series],
    threshold: float | None,
    pair_b: str | None = None,
) -> float | None:
    """Check if a score-based metric agrees with one human preference.

    Returns 1.0 (agree), 0.5 (tie), 0.0 (disagree), or None (skip).
    """
    if pref_label == "N":
        if threshold is None or not pair_b:
            return None
        row_a = score_index.get((qi, other_label))
        row_b = score_index.get((qi, pair_b))
        if row_a is None or row_b is None:
            return None
        val_a, val_b = row_a.get(metric_col), row_b.get(metric_col)
        if pd.isna(val_a) or pd.isna(val_b):
            return None
        return 1.0 if abs(val_a - val_b) < threshold else 0.0

    pref_row = score_index.get((qi, pref_label))
    other_row = score_index.get((qi, other_label))
    if pref_row is None or other_row is None:
        return None
    pref_val, other_val = pref_row.get(metric_col), other_row.get(metric_col)
    if pd.isna(pref_val) or pd.isna(other_val):
        return None
    if pref_val > other_val:
        return 1.0
    if pref_val == other_val:
        return 0.5
    return 0.0


def _build_pairwise_lookup(pairwise_prefs: pd.DataFrame) -> dict[tuple, dict]:
    """Index raw pairwise preferences by ``(query_index, comparison)``."""
    lookup: dict[tuple, dict] = {}
    meta_cols = {"query_index", "comparison"}
    for _, row in pairwise_prefs.iterrows():
        key = (row["query_index"], row["comparison"])
        lookup[key] = {col: row[col] for col in pairwise_prefs.columns if col not in meta_cols}
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
    """
    qi = human_row["query_index"]
    pref = human_row["preferred"]
    other = human_row["other"]
    comp_type = human_row.get("comparison_type", "")

    if pref == "N":
        if comp_type == "round1":
            if {other, human_row.get("pair_b", "")} & {"A", "B"} == {"A", "B"}:
                data = pairwise_lookup.get((qi, "round1_ab"), {})
            else:
                data = pairwise_lookup.get((qi, "round1_cd"), {})
            llm_winner = data.get(metric_col)
            if llm_winner is None:
                return None
            return 1.0 if llm_winner == "tie" else 0.0
        return None

    pair = frozenset({pref, other})

    if comp_type == "round1":
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
        if llm_winner == "tie":
            return 0.5
        return 0.0

    if comp_type == "final":
        final_data = pairwise_lookup.get((qi, "final"), {})
        ab_data = pairwise_lookup.get((qi, "round1_ab"), {})
        cd_data = pairwise_lookup.get((qi, "round1_cd"), {})
        llm_final = final_data.get(metric_col)
        llm_ab = ab_data.get(metric_col)
        llm_cd = cd_data.get(metric_col)
        if llm_final is None or llm_ab is None or llm_cd is None:
            return None

        llm_ab_winner = llm_ab if llm_ab != "tie" else "A"
        llm_cd_winner = llm_cd if llm_cd != "tie" else "C"
        llm_finalists = {llm_ab_winner, llm_cd_winner}

        if pref in llm_finalists and other in llm_finalists:
            if llm_final == pref:
                return 1.0
            if llm_final == "tie":
                return 0.5
            return 0.0

        if pref in llm_finalists and other not in llm_finalists:
            round_winner = llm_ab if other in ("A", "B") else llm_cd
            if round_winner == other:
                return 0.0
            if round_winner == "tie":
                return 0.5
            if llm_final == pref:
                return 1.0
            if llm_final == "tie":
                return 0.5
            # Transitive disagreement: pref lost the final to a non-"other"
            # finalist. The only scenario where we can still say pref > other
            # is if pref is the bracket winner who beat the loser directly.
            if other in ("A", "B") and pref == llm_cd_winner:
                return 0.0
            if other in ("C", "D") and pref == llm_ab_winner:
                return 0.0
            return 1.0

        return None

    return None


def _evaluate_metric_agreement(
    preferences: pd.DataFrame,
    metric_col: str,
    score_index: dict[tuple, pd.Series],
    pairwise_lookup: dict[tuple, dict] | None,
    use_pairwise: bool,
    threshold: float | None,
    strict: bool,
) -> tuple[float, int]:
    """Return (n_agree, n_total) for one metric over all preference rows."""

    def check(row):
        if use_pairwise:
            return _check_pairwise_preference(row, metric_col, pairwise_lookup)
        return _score_agreement(
            row["preferred"], row["other"], row["query_index"],
            metric_col, score_index, threshold, pair_b=row.get("pair_b"),
        )

    # Strict mode: a round1 disagreement invalidates the final for that
    # (annotator, query). Precompute which keys are still eligible.
    round1_ok: dict[tuple, bool] = {}
    if strict:
        round1 = preferences[preferences["comparison_type"] == "round1"]
        for _, row in round1.iterrows():
            key = (row["annotator_id"], row["query_index"])
            result = check(row)
            if result is not None and result < 1.0:
                round1_ok[key] = False
            elif key not in round1_ok:
                round1_ok[key] = True

    n_agree = 0.0
    n_total = 0
    for _, row in preferences.iterrows():
        result = check(row)
        if result is None:
            continue
        n_total += 1
        if strict and row.get("comparison_type") == "final":
            key = (row["annotator_id"], row["query_index"])
            if not round1_ok.get(key, True):
                continue
        n_agree += result

    return n_agree, n_total


def compute_pairwise_agreement(
    preferences: pd.DataFrame,
    metric_scores: pd.DataFrame,
    neither_thresholds: dict[str, float] | None = None,
    strict: bool = False,
    pairwise_prefs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """How often each metric agrees with human pairwise preferences.

    Args:
        preferences: From :func:`get_pairwise_preferences`. Columns:
            annotator_id, query_index, preferred, other, comparison_type.
        metric_scores: Columns: query_index, label (A/B/C/D), one per metric.
        neither_thresholds: Optional per-metric thresholds for "neither".
        strict: If True, final-round comparisons auto-disagree when the
            metric got any round1 comparison wrong for that annotator/query.
        pairwise_prefs: Optional raw LLM pairwise preferences. Columns:
            query_index, comparison, one per sub-metric (values A/B/C/D/tie).
            Sub-metrics listed here use direct preference matching.
    """
    neither_thresholds = neither_thresholds or {}

    pairwise_lookup = None
    pairwise_cols: set[str] = set()
    if pairwise_prefs is not None and not pairwise_prefs.empty:
        pairwise_lookup = _build_pairwise_lookup(pairwise_prefs)
        pairwise_cols = {
            c for c in pairwise_prefs.columns if c not in ("query_index", "comparison")
        }

    score_index = _build_score_index(metric_scores)
    score_cols = [
        c for c in metric_scores.columns
        if c not in ("query_index", "label", "annotator_id")
    ]
    all_metric_cols = list(dict.fromkeys(
        score_cols + sorted(pairwise_cols - set(score_cols))
    ))

    results = []
    for metric_col in all_metric_cols:
        use_pairwise = metric_col in pairwise_cols and pairwise_lookup is not None
        n_agree, n_total = _evaluate_metric_agreement(
            preferences, metric_col, score_index, pairwise_lookup,
            use_pairwise, neither_thresholds.get(metric_col), strict,
        )
        results.append({
            "metric": metric_col,
            "n_comparisons": n_total,
            "n_agreements": n_agree,
            "agreement_rate": n_agree / n_total if n_total > 0 else float("nan"),
        })

    return pd.DataFrame(results)


def _human_ranking(entry) -> dict[str, float]:
    """Derive a {label: rank} mapping from human tournament preferences.

    Lower rank = better (1 = best). Ties get the average rank.
    """
    labels = ["A", "B", "C", "D"]
    scores = {l: 0.0 for l in labels}
    if entry.round1_ab and entry.round1_ab in labels:
        scores[entry.round1_ab] += 1
    if entry.round1_cd and entry.round1_cd in labels:
        scores[entry.round1_cd] += 1
    if entry.final and entry.final in labels:
        scores[entry.final] += 2

    score_list = sorted(((l, scores[l]) for l in labels), key=lambda x: -x[1])
    ranks = {}
    i = 0
    while i < len(score_list):
        j = i + 1
        while j < len(score_list) and score_list[j][1] == score_list[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[score_list[k][0]] = avg_rank
        i = j
    return ranks


def compute_rank_correlation(
    entries: list,
    metric_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Per-query rank correlation between human and metric rankings.

    Returns columns: query_index, annotator_id, metric, kendall_tau,
    kendall_p, spearman_rho, spearman_p.
    """
    labels = ["A", "B", "C", "D"]
    score_cols = [
        c for c in metric_scores.columns
        if c not in ("annotator_id", "query_index", "label")
    ]
    score_index = _build_score_index(metric_scores)

    rows = []
    for entry in entries:
        if not entry.final:
            continue
        qi = entry.query_index
        human_rank_vec = [_human_ranking(entry)[l] for l in labels]

        # All four labels must exist for this query.
        if not all((qi, l) in score_index for l in labels):
            continue

        for metric_col in score_cols:
            metric_vals = [score_index[(qi, l)].get(metric_col) for l in labels]
            if any(pd.isna(v) for v in metric_vals):
                continue
            metric_rank_vec = stats.rankdata([-v for v in metric_vals])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", stats.ConstantInputWarning)
                tau, tau_p = stats.kendalltau(human_rank_vec, metric_rank_vec)
                rho, rho_p = stats.spearmanr(human_rank_vec, metric_rank_vec)

            rows.append({
                "query_index": qi,
                "annotator_id": entry.annotator_id,
                "metric": metric_col,
                "kendall_tau": tau,
                "kendall_p": tau_p,
                "spearman_rho": rho,
                "spearman_p": rho_p,
            })

    return pd.DataFrame(rows)


def aggregate_correlations(per_query: pd.DataFrame) -> pd.DataFrame:
    """Summary stats per metric: mean/std/median of tau and rho, plus n_queries."""
    if per_query.empty:
        return pd.DataFrame()

    def _agg(group):
        return pd.Series({
            "mean_tau": group["kendall_tau"].mean(),
            "std_tau": group["kendall_tau"].std(),
            "median_tau": group["kendall_tau"].median(),
            "mean_rho": group["spearman_rho"].mean(),
            "std_rho": group["spearman_rho"].std(),
            "median_rho": group["spearman_rho"].median(),
            "n_queries": len(group),
        })

    return per_query.groupby("metric").apply(_agg, include_groups=False).reset_index()
