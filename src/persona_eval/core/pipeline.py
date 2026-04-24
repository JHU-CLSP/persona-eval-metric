"""Metric orchestration: task building, pairwise tournaments, caching."""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from persona_eval.annotations import AnnotationEntry, AnnotatorProfile
from persona_eval.core.cache import config_hash
from persona_eval.metrics import get_metric
from persona_eval.metrics.cache import MetricCache


def make_persona_kwargs(profile: AnnotatorProfile | None) -> dict | None:
    """Build persona template kwargs from an annotator profile."""
    if profile is None:
        return None
    return {
        "role": profile.role or "unspecified",
        "domain": profile.domain or "unspecified",
        "info_needs": profile.info_needs or "unspecified",
    }


def score_cached(
    metric, metric_name: str, cfg_hash: str,
    task: dict, text_key: str, cache: MetricCache | None,
) -> dict:
    """Call ``metric.score`` through the cache."""
    summary = task["summary"]
    source = task[text_key]
    pk = task.get("persona_kwargs")
    if cache is not None:
        hit = cache.get(metric_name, cfg_hash, summary, source, pk)
        if hit is not None:
            return hit
    scores = metric.score(summary, source, persona_kwargs=pk)
    if cache is not None:
        cache.put(metric_name, cfg_hash, summary, source, pk, scores)
    return scores


def score_pair_cached(
    metric, metric_name: str, cfg_hash: str,
    summary_a: str, summary_b: str, source: str,
    persona_kwargs: dict | None, cache: MetricCache | None,
) -> dict[str, str]:
    """Call ``metric.score_pair`` through the cache."""
    if cache is not None:
        hit = cache.get_pair(metric_name, cfg_hash, summary_a, summary_b, source, persona_kwargs)
        if hit is not None:
            return hit
    prefs = metric.score_pair(summary_a, summary_b, source, persona_kwargs=persona_kwargs)
    if cache is not None:
        cache.put_pair(metric_name, cfg_hash, summary_a, summary_b, source, persona_kwargs, prefs)
    return prefs


def run_pairwise_comparisons(
    tasks: list[dict], metric, text_key: str,
    metric_name: str | None = None, cache: MetricCache | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run pairwise comparisons mirroring the human annotation tournament.

    Round 1: A vs B, C vs D. Final: round1 winners face off.

    Returns ``(raw_preferences, tournament_scores)``.
    """
    mn = metric_name or getattr(metric, "name", type(metric).__name__)
    cfg_hash = config_hash(metric.cache_config())

    by_query: dict[int, dict[str, dict]] = {}
    for task in tasks:
        by_query.setdefault(task["query_index"], {})[task["label"]] = task

    raw_prefs: list[dict] = []
    score_rows: list[dict] = []

    for qi, label_tasks in tqdm(by_query.items(), desc=metric.name):
        if not all(l in label_tasks for l in ("A", "B", "C", "D")):
            continue

        source = label_tasks["A"][text_key]
        pk = label_tasks["A"].get("persona_kwargs")
        sub_metrics: list[str] | None = None
        scores: dict[str, dict[str, float]] = {l: {} for l in ("A", "B", "C", "D")}

        # Round 1: A vs B
        result_ab = score_pair_cached(
            metric, mn, cfg_hash,
            label_tasks["A"]["summary"], label_tasks["B"]["summary"],
            source, pk, cache,
        )
        if sub_metrics is None:
            sub_metrics = list(result_ab.keys())
            for l in ("A", "B", "C", "D"):
                scores[l] = {sm: 0.0 for sm in sub_metrics}

        # Round 1: C vs D
        result_cd = score_pair_cached(
            metric, mn, cfg_hash,
            label_tasks["C"]["summary"], label_tasks["D"]["summary"],
            source, pk, cache,
        )

        ab_winners: dict[str, str] = {}
        cd_winners: dict[str, str] = {}
        pref_ab_row = {"query_index": qi, "comparison": "round1_ab"}
        pref_cd_row = {"query_index": qi, "comparison": "round1_cd"}

        for sm in sub_metrics:
            pref_ab = result_ab[sm]
            if pref_ab == "A":
                scores["A"][sm] += 1.0
                ab_winners[sm] = "A"
                pref_ab_row[sm] = "A"
            elif pref_ab == "B":
                scores["B"][sm] += 1.0
                ab_winners[sm] = "B"
                pref_ab_row[sm] = "B"
            else:
                scores["A"][sm] += 0.5
                scores["B"][sm] += 0.5
                ab_winners[sm] = "A"  # tiebreak
                pref_ab_row[sm] = "tie"

            pref_cd = result_cd[sm]
            if pref_cd == "A":
                scores["C"][sm] += 1.0
                cd_winners[sm] = "C"
                pref_cd_row[sm] = "C"
            elif pref_cd == "B":
                scores["D"][sm] += 1.0
                cd_winners[sm] = "D"
                pref_cd_row[sm] = "D"
            else:
                scores["C"][sm] += 0.5
                scores["D"][sm] += 0.5
                cd_winners[sm] = "C"
                pref_cd_row[sm] = "tie"

        raw_prefs.append(pref_ab_row)
        raw_prefs.append(pref_cd_row)

        # Final: winner of AB vs winner of CD
        final_pairs: dict[tuple[str, str], list[str]] = {}
        for sm in sub_metrics:
            pair = (ab_winners[sm], cd_winners[sm])
            final_pairs.setdefault(pair, []).append(sm)

        pref_final_row = {"query_index": qi, "comparison": "final"}
        for (w_ab, w_cd), sms in final_pairs.items():
            result_final = score_pair_cached(
                metric, mn, cfg_hash,
                label_tasks[w_ab]["summary"], label_tasks[w_cd]["summary"],
                source, pk, cache,
            )
            for sm in sms:
                pref_final = result_final[sm]
                if pref_final == "A":
                    scores[w_ab][sm] += 2.0
                    pref_final_row[sm] = w_ab
                elif pref_final == "B":
                    scores[w_cd][sm] += 2.0
                    pref_final_row[sm] = w_cd
                else:
                    scores[w_ab][sm] += 1.0
                    scores[w_cd][sm] += 1.0
                    pref_final_row[sm] = "tie"

        raw_prefs.append(pref_final_row)
        for label in ("A", "B", "C", "D"):
            score_rows.append({"query_index": qi, "label": label, **scores[label]})

    return raw_prefs, score_rows


def build_tasks(
    entries: list[AnnotationEntry],
    source_texts: dict[int, str],
    reference_texts: dict[int, str],
    profiles_by_id: dict[str, AnnotatorProfile] | None = None,
    per_annotator: bool = False,
    include_query: bool = False,
) -> list[dict]:
    """Build a deduplicated task list from annotation entries.

    When ``per_annotator`` is True, dedupe by
    ``(annotator_id, query_index, label)`` and include persona info;
    otherwise dedupe by ``(query_index, label)``. When ``include_query``
    is True, the annotator's query is attached to ``persona_kwargs``.
    """
    seen = set()
    tasks: list[dict] = []
    for entry in entries:
        for label in ("A", "B", "C", "D"):
            if label not in entry.summaries:
                continue
            key = (
                (entry.annotator_id, entry.query_index, label)
                if per_annotator else (entry.query_index, label)
            )
            if key in seen:
                continue
            seen.add(key)

            pk: dict = {"query": entry.query or ""} if include_query else {}
            task = {
                "query_index": entry.query_index,
                "label": label,
                "summary": entry.summaries[label],
                "source": source_texts.get(entry.query_index, ""),
                "reference": reference_texts.get(entry.query_index, ""),
                "persona_kwargs": pk,
            }
            if per_annotator:
                task["annotator_id"] = entry.annotator_id
                profile = (profiles_by_id or {}).get(entry.annotator_id)
                persona_pk = make_persona_kwargs(profile)
                if persona_pk is not None:
                    persona_pk.update(pk)
                    task["persona_kwargs"] = persona_pk
            tasks.append(task)
    return tasks


def merge_score_dfs(all_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge per-metric DataFrames into one row per (annotator_id?, query_index, label)."""
    if not all_dfs:
        return pd.DataFrame()
    merged = all_dfs[0]
    for df in all_dfs[1:]:
        merge_cols = ["query_index", "label"]
        if "annotator_id" in merged.columns and "annotator_id" in df.columns:
            merge_cols.insert(0, "annotator_id")
        overlap = [c for c in df.columns if c in merged.columns and c not in merge_cols]
        if overlap:
            merged = merged.drop(columns=overlap)
        merged = merged.merge(df, on=merge_cols, how="outer")
    return merged


def metric_already_complete(
    existing_df: pd.DataFrame | None, tasks: list[dict], sub_metric_cols: list[str],
) -> bool:
    """True iff ``existing_df`` has non-null values for ``sub_metric_cols`` for every task."""
    if existing_df is None or existing_df.empty or not sub_metric_cols:
        return False
    if any(c not in existing_df.columns for c in sub_metric_cols):
        return False

    group_cols = ["query_index", "label"]
    if "annotator_id" in existing_df.columns and any("annotator_id" in t for t in tasks):
        group_cols = ["annotator_id", "query_index", "label"]

    indexed = existing_df.set_index(group_cols)
    for task in tasks:
        key = tuple(task[c] for c in group_cols)
        if key not in indexed.index:
            return False
        row = indexed.loc[key]
        if hasattr(row, "iloc") and hasattr(row, "ndim") and row.ndim > 1:
            row = row.iloc[0]
        for col in sub_metric_cols:
            if pd.isna(row[col]):
                return False
    return True


def compute_metric_scores(
    entries: list[AnnotationEntry],
    source_texts: dict[int, str],
    reference_texts: dict[int, str],
    metric_names: list[str],
    device: str = "cpu",
    llm_kwargs: dict | None = None,
    profiles_by_id: dict[str, AnnotatorProfile] | None = None,
    response_logger=None,
    cache: MetricCache | None = None,
    existing_df: pd.DataFrame | None = None,
    on_metric_done=None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Compute metric scores for all (query, summary) pairs.

    ``on_metric_done(scores_df, pairwise_prefs_df)`` fires after each
    metric finishes so callers can persist output incrementally.
    """
    llm_kwargs = dict(llm_kwargs or {})
    include_query = llm_kwargs.pop("include_query", False)

    if not include_query:
        for mn in metric_names:
            m = get_metric(mn, device=device, **llm_kwargs)
            if m.needs_query:
                include_query = True
                break

    _tasks_cache: dict[bool, list[dict]] = {}

    def get_tasks(per_annotator: bool) -> list[dict]:
        if per_annotator not in _tasks_cache:
            _tasks_cache[per_annotator] = build_tasks(
                entries, source_texts, reference_texts,
                profiles_by_id=profiles_by_id,
                per_annotator=per_annotator,
                include_query=include_query,
            )
        return _tasks_cache[per_annotator]

    all_dfs: list[pd.DataFrame] = []
    all_raw_prefs: list[dict] = []

    if existing_df is not None and not existing_df.empty:
        all_dfs.append(existing_df.copy())

    for metric_name in metric_names:
        kwargs = {"device": device, **llm_kwargs}
        if response_logger is not None:
            kwargs["response_logger"] = response_logger
        metric = get_metric(metric_name, **kwargs)
        text_key = "source" if metric.is_reference_free else "reference"
        tasks = get_tasks(per_annotator=metric.needs_persona)
        cfg_hash = config_hash(metric.cache_config())

        if cache is not None:
            cache.reset_counters()

        rows: list[dict] = []
        if metric.is_pairwise:
            print(f"Computing {metric_name}...")
            raw_prefs, score_rows = run_pairwise_comparisons(
                tasks, metric, text_key, metric_name=metric_name, cache=cache,
            )
            all_raw_prefs.extend(raw_prefs)
            rows.extend(score_rows)
        else:
            if not tasks:
                continue

            # Compute first task to discover sub-metric columns, then
            # decide whether to skip the rest.
            task0 = tasks[0]
            first_scores = score_cached(metric, metric_name, cfg_hash, task0, text_key, cache)
            sub_metric_cols = list(first_scores.keys())

            if metric_already_complete(existing_df, tasks, sub_metric_cols):
                print(f"Skipping {metric_name} (already in output CSV)")
                continue

            print(f"Computing {metric_name}...")
            row0 = {"query_index": task0["query_index"], "label": task0["label"], **first_scores}
            if "annotator_id" in task0:
                row0["annotator_id"] = task0["annotator_id"]
            rows.append(row0)

            for task in tqdm(tasks[1:], desc=metric_name):
                scores = score_cached(metric, metric_name, cfg_hash, task, text_key, cache)
                row = {"query_index": task["query_index"], "label": task["label"], **scores}
                if "annotator_id" in task:
                    row["annotator_id"] = task["annotator_id"]
                rows.append(row)

        if cache is not None and (cache.hits or cache.misses):
            print(f"  cache: {cache.hits} hits / {cache.misses} misses")

        if rows:
            df = pd.DataFrame(rows)
            group_cols = ["query_index", "label"]
            if "annotator_id" in df.columns:
                group_cols.insert(0, "annotator_id")
            df = df.groupby(group_cols, as_index=False).first()
            all_dfs.append(df)

        if on_metric_done is not None:
            partial_scores = merge_score_dfs(all_dfs)
            partial_prefs = pd.DataFrame(all_raw_prefs) if all_raw_prefs else None
            on_metric_done(partial_scores, partial_prefs)

    scores_df = merge_score_dfs(all_dfs)
    pairwise_prefs_df = pd.DataFrame(all_raw_prefs) if all_raw_prefs else None
    return scores_df, pairwise_prefs_df
