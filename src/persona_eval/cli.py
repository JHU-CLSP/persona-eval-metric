"""CLI entry point for the persona evaluation pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from persona_eval.annotations import (
    AnnotationEntry,
    AnnotatorProfile,
    load_annotations,
    get_pairwise_preferences,
)
from persona_eval.correlation import (
    aggregate_correlations,
    compute_pairwise_agreement,
    compute_rank_correlation,
)
from persona_eval.metrics import get_metric, list_metrics, list_llm_metrics, list_non_llm_metrics
from persona_eval.metrics.cache import MetricCache, config_hash
from persona_eval.openalex import OpenAlexClient


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_metrics(metrics: list[str] | None, default: list[str] | None = None) -> list[str]:
    """Resolve the ``--metrics`` argument, expanding shorthand groups.

    Recognised group names: ``"all"``, ``"non-llm"``, ``"all-llm"``.
    """
    if not metrics:
        return default if default is not None else list_metrics()

    GROUP_MAP = {
        "all": list_metrics,
        "non-llm": list_non_llm_metrics,
        "all-llm": list_llm_metrics,
    }

    resolved: list[str] = []
    for m in metrics:
        if m in GROUP_MAP:
            resolved.extend(GROUP_MAP[m]())
        else:
            resolved.append(m)

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for m in resolved:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped


def _make_persona_kwargs(profile: AnnotatorProfile | None) -> dict | None:
    """Build persona template kwargs from an annotator profile."""
    if profile is None:
        return None
    return {
        "role": profile.role or "unspecified",
        "domain": profile.domain or "unspecified",
        "info_needs": profile.info_needs or "unspecified",
    }


def _load_data(args):
    """Load annotations and fetch source/reference texts.

    Returns:
        Tuple of (profiles_by_id, entries, source_texts, reference_texts).
    """
    profiles, entries = load_annotations(args.annotations)
    profiles_by_id = {p.annotator_id: p for p in profiles}
    client = OpenAlexClient(cache_dir=args.cache_dir, email=args.email)
    source_texts, reference_texts = client.get_texts_batch(entries)
    return profiles_by_id, entries, source_texts, reference_texts


def _load_neither_thresholds(path: str | None) -> dict[str, float] | None:
    """Load per-metric neither thresholds from a YAML config file."""
    if path is None:
        return None
    with open(path) as f:
        data = yaml.safe_load(f)
    return {str(k): float(v) for k, v in data.items()}


def _create_response_logger(output_path: str | Path):
    """Create a ResponseLogger for LLM response tracking."""
    from persona_eval.response_logger import ResponseLogger
    log_dir = Path(output_path).parent / "llm_responses"
    return ResponseLogger(log_dir)


def _save_run_params(path: Path, args) -> None:
    """Write CLI arguments to a JSON file for reproducibility."""
    params = {k: v for k, v in vars(args).items() if k != "func"}
    path.write_text(json.dumps(params, indent=2, default=str))


def _collect_llm_kwargs(args) -> dict:
    """Collect LLM-related kwargs from CLI args."""
    kwargs = {}
    for key in ("provider", "model", "api_key", "base_url", "prompt_file"):
        attr = f"llm_{key}"
        val = getattr(args, attr, None)
        if val is not None:
            kwargs[key] = val
    if getattr(args, "persona", False):
        kwargs["persona"] = True
    if getattr(args, "include_query", False):
        kwargs["include_query"] = True
    return kwargs


def _compute_and_save(
    entries, source_texts, reference_texts, profiles_by_id,
    metric_names, args, output_path,
):
    """Compute metrics and save results. Returns (scores_df, pairwise_prefs_df, response_logger)."""
    llm_kwargs = _collect_llm_kwargs(args)
    response_logger = _create_response_logger(output_path)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    prefs_path = output.parent / (output.stem + "_pairwise_prefs.csv")

    # Load existing output CSV to enable skip-on-re-run.
    existing_df = pd.read_csv(output) if output.exists() else None
    if existing_df is not None:
        print(f"Found existing {output} ({len(existing_df)} rows); will skip already-computed metrics")

    cache = _build_metric_cache(args)

    def save_partial(scores_df: pd.DataFrame, prefs_df: pd.DataFrame | None):
        if scores_df is not None and not scores_df.empty:
            scores_df.to_csv(output, index=False)
        if prefs_df is not None:
            prefs_df.to_csv(prefs_path, index=False)

    scores_df, pairwise_prefs_df = _compute_metric_scores(
        entries, source_texts, reference_texts,
        metric_names, device=args.device,
        llm_kwargs=llm_kwargs, profiles_by_id=profiles_by_id,
        response_logger=response_logger,
        cache=cache,
        existing_df=existing_df,
        on_metric_done=save_partial,
    )

    # Final save (covers the case where on_metric_done wasn't called — e.g.,
    # empty metric list — and ensures the on-disk file matches the returned df).
    if not scores_df.empty:
        scores_df.to_csv(output, index=False)
    print(f"Saved metric scores to {output} ({len(scores_df)} rows)")

    if pairwise_prefs_df is not None:
        pairwise_prefs_df.to_csv(prefs_path, index=False)
        print(f"Saved raw pairwise preferences to {prefs_path}")

    if response_logger is not None:
        print(f"Saved LLM responses to {response_logger.path}")
        response_logger.close()

    return scores_df, pairwise_prefs_df


def _build_metric_cache(args) -> "MetricCache | None":
    """Build a MetricCache from CLI args. Returns ``None`` when disabled."""
    cache_dir = getattr(args, "cache_dir", None)
    if cache_dir is None:
        return None
    enabled = not getattr(args, "no_cache", False)
    cache = MetricCache(cache_dir, enabled=enabled)
    if enabled and getattr(args, "clear_metric_cache", False):
        n = cache.clear()
        if n:
            print(f"Cleared {n} metric cache entries from {cache.cache_dir}")
    return cache


def _run_correlations(entries, scores_df, pairwise_prefs_df, args, output_dir):
    """Compute and save correlation results. Returns (agreement, agg)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    include_neither = getattr(args, "include_neither", False)
    neither_config = getattr(args, "neither_config", None)
    neither_thresholds = _load_neither_thresholds(neither_config) if include_neither else None
    strict = getattr(args, "strict_pairwise", False)

    preferences = get_pairwise_preferences(entries, include_neither=include_neither)

    agreement = compute_pairwise_agreement(
        preferences, scores_df, neither_thresholds=neither_thresholds,
        strict=strict, pairwise_prefs=pairwise_prefs_df,
    )
    agreement.to_csv(output_dir / "pairwise_agreement.csv", index=False)

    per_query = compute_rank_correlation(entries, scores_df)
    per_query.to_csv(output_dir / "rank_correlation_per_query.csv", index=False)

    agg = aggregate_correlations(per_query)
    agg.to_csv(output_dir / "rank_correlation_aggregate.csv", index=False)

    return agreement, agg, include_neither, neither_config, strict


# ---------------------------------------------------------------------------
# Pipeline internals
# ---------------------------------------------------------------------------


def _score_cached(
    metric,
    metric_name: str,
    cfg_hash: str,
    task: dict,
    text_key: str,
    cache: "MetricCache | None",
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


def _score_pair_cached(
    metric,
    metric_name: str,
    cfg_hash: str,
    summary_a: str,
    summary_b: str,
    source: str,
    persona_kwargs: dict | None,
    cache: "MetricCache | None",
) -> dict[str, str]:
    """Call ``metric.score_pair`` through the cache."""
    if cache is not None:
        hit = cache.get_pair(
            metric_name, cfg_hash, summary_a, summary_b, source, persona_kwargs,
        )
        if hit is not None:
            return hit
    prefs = metric.score_pair(summary_a, summary_b, source, persona_kwargs=persona_kwargs)
    if cache is not None:
        cache.put_pair(
            metric_name, cfg_hash, summary_a, summary_b, source, persona_kwargs, prefs,
        )
    return prefs


def _run_pairwise_comparisons(
    tasks: list[dict],
    metric,
    text_key: str,
    metric_name: str | None = None,
    cache: "MetricCache | None" = None,
) -> tuple[list[dict], list[dict]]:
    """Run pairwise comparisons mirroring the human annotation tournament.

    Mirrors the human annotation tournament structure:
      Round 1: A vs B, C vs D
      Final:   winner of AB vs winner of CD

    Returns:
        Tuple of (raw_preferences, tournament_scores).

        raw_preferences: list of dicts with columns query_index,
            comparison ("round1_ab", "round1_cd", "final"), and one
            column per sub-metric containing the winning label
            ("A"/"B"/"C"/"D") or "tie".

        tournament_scores: list of per-(query, label) dicts with
            tournament point columns for rank correlation.
    """
    mn = metric_name or getattr(metric, "name", type(metric).__name__)
    cfg_hash = config_hash(metric.cache_config())

    # Group tasks by query_index, keyed by label
    by_query: dict[int, dict[str, dict]] = {}
    for task in tasks:
        by_query.setdefault(task["query_index"], {})[task["label"]] = task

    raw_prefs = []
    score_rows = []

    for qi, label_tasks in tqdm(by_query.items(), desc=metric.name):
        if not all(l in label_tasks for l in ("A", "B", "C", "D")):
            continue

        source = label_tasks["A"][text_key]
        sub_metrics = None
        scores: dict[str, dict[str, float]] = {
            l: {} for l in ("A", "B", "C", "D")
        }

        pk = label_tasks["A"].get("persona_kwargs")

        # --- Round 1: A vs B ---
        result_ab = _score_pair_cached(
            metric, mn, cfg_hash,
            label_tasks["A"]["summary"],
            label_tasks["B"]["summary"],
            source, pk, cache,
        )
        if sub_metrics is None:
            sub_metrics = list(result_ab.keys())
            for l in ("A", "B", "C", "D"):
                scores[l] = {sm: 0.0 for sm in sub_metrics}

        # --- Round 1: C vs D ---
        result_cd = _score_pair_cached(
            metric, mn, cfg_hash,
            label_tasks["C"]["summary"],
            label_tasks["D"]["summary"],
            source, pk, cache,
        )

        # Translate score_pair results ("A"/"B") to actual labels and store
        ab_winners: dict[str, str] = {}
        cd_winners: dict[str, str] = {}
        pref_ab_row = {"query_index": qi, "comparison": "round1_ab"}
        pref_cd_row = {"query_index": qi, "comparison": "round1_cd"}

        for sm in sub_metrics:
            # A vs B
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

            # C vs D
            pref_cd = result_cd[sm]
            if pref_cd == "A":  # first arg = C
                scores["C"][sm] += 1.0
                cd_winners[sm] = "C"
                pref_cd_row[sm] = "C"
            elif pref_cd == "B":  # second arg = D
                scores["D"][sm] += 1.0
                cd_winners[sm] = "D"
                pref_cd_row[sm] = "D"
            else:
                scores["C"][sm] += 0.5
                scores["D"][sm] += 0.5
                cd_winners[sm] = "C"  # tiebreak
                pref_cd_row[sm] = "tie"

        raw_prefs.append(pref_ab_row)
        raw_prefs.append(pref_cd_row)

        # --- Final: winner of AB vs winner of CD ---
        final_pairs: dict[tuple[str, str], list[str]] = {}
        for sm in sub_metrics:
            pair = (ab_winners[sm], cd_winners[sm])
            final_pairs.setdefault(pair, []).append(sm)

        pref_final_row = {"query_index": qi, "comparison": "final"}

        for (w_ab, w_cd), sms in final_pairs.items():
            result_final = _score_pair_cached(
                metric, mn, cfg_hash,
                label_tasks[w_ab]["summary"],
                label_tasks[w_cd]["summary"],
                source, pk, cache,
            )
            for sm in sms:
                pref_final = result_final[sm]
                if pref_final == "A":  # first arg = AB winner
                    scores[w_ab][sm] += 2.0
                    pref_final_row[sm] = w_ab
                elif pref_final == "B":  # second arg = CD winner
                    scores[w_cd][sm] += 2.0
                    pref_final_row[sm] = w_cd
                else:
                    scores[w_ab][sm] += 1.0
                    scores[w_cd][sm] += 1.0
                    pref_final_row[sm] = "tie"

        raw_prefs.append(pref_final_row)

        for label in ("A", "B", "C", "D"):
            score_rows.append({
                "query_index": qi,
                "label": label,
                **scores[label],
            })

    return raw_prefs, score_rows


def _build_tasks(
    entries: list[AnnotationEntry],
    source_texts: dict[int, str],
    reference_texts: dict[int, str],
    profiles_by_id: dict[str, AnnotatorProfile] | None = None,
    per_annotator: bool = False,
    include_query: bool = False,
) -> list[dict]:
    """Build deduplicated task list from annotation entries.

    When ``per_annotator`` is True, tasks are deduplicated by
    (annotator_id, query_index, label) and include persona info.
    Otherwise, deduplicated by (query_index, label).

    When ``include_query`` is True, the annotator's query is included
    in persona_kwargs so prompts can render it.
    """
    seen = set()
    tasks = []
    for entry in entries:
        for label in ("A", "B", "C", "D"):
            if label not in entry.summaries:
                continue

            if per_annotator:
                key = (entry.annotator_id, entry.query_index, label)
            else:
                key = (entry.query_index, label)

            if key in seen:
                continue
            seen.add(key)

            pk: dict = {}
            if include_query:
                pk["query"] = entry.query or ""

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
                persona_pk = _make_persona_kwargs(profile)
                if persona_pk is not None:
                    persona_pk.update(pk)
                    task["persona_kwargs"] = persona_pk

            tasks.append(task)

    return tasks


def _merge_score_dfs(all_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge per-metric DataFrames into one row per (annotator_id?, query_index, label)."""
    if not all_dfs:
        return pd.DataFrame()
    merged = all_dfs[0]
    for df in all_dfs[1:]:
        merge_cols = ["query_index", "label"]
        if "annotator_id" in merged.columns and "annotator_id" in df.columns:
            merge_cols.insert(0, "annotator_id")
        # Columns overlapping between merged and df (other than merge_cols) should
        # be unified — prefer the latest computed value from df.
        overlap = [
            c for c in df.columns
            if c in merged.columns and c not in merge_cols
        ]
        if overlap:
            merged = merged.drop(columns=overlap)
        merged = merged.merge(df, on=merge_cols, how="outer")
    return merged


def _metric_already_complete(
    existing_df: pd.DataFrame | None,
    tasks: list[dict],
    sub_metric_cols: list[str],
) -> bool:
    """Whether ``existing_df`` has non-null values for ``sub_metric_cols``
    for every task in ``tasks``.

    If yes, the caller can skip re-running the metric.
    """
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
            val = row[col]
            if pd.isna(val):
                return False
    return True


def _compute_metric_scores(
    entries: list[AnnotationEntry],
    source_texts: dict[int, str],
    reference_texts: dict[int, str],
    metric_names: list[str],
    device: str = "cpu",
    llm_kwargs: dict | None = None,
    profiles_by_id: dict[str, AnnotatorProfile] | None = None,
    response_logger=None,
    cache: "MetricCache | None" = None,
    existing_df: pd.DataFrame | None = None,
    on_metric_done=None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Compute metric scores for all (query, summary) pairs.

    Args:
        cache: Optional ``MetricCache``. When provided, per-metric results
            are read from / written to the cache.
        existing_df: Optional ``scores_df`` from a prior run. Metrics whose
            sub-metric columns are already fully populated will be skipped.
        on_metric_done: Optional ``(scores_df, pairwise_prefs_df) -> None``
            callback invoked after each metric finishes. Used by the caller
            to write the output CSV incrementally.

    Returns:
        Tuple of (scores_df, pairwise_prefs_df).
        scores_df: per-(query, label) metric scores (for all metrics).
        pairwise_prefs_df: raw LLM pairwise preferences per comparison,
            or None if no pairwise metrics were run.
    """
    llm_kwargs = llm_kwargs or {}
    include_query = llm_kwargs.pop("include_query", False)

    # Force include_query if any requested metric needs it
    if not include_query:
        for mn in metric_names:
            m = get_metric(mn, device=device, **llm_kwargs)
            if m.needs_query:
                include_query = True
                break

    _tasks_cache: dict[bool, list[dict]] = {}

    def get_tasks(per_annotator: bool) -> list[dict]:
        if per_annotator not in _tasks_cache:
            _tasks_cache[per_annotator] = _build_tasks(
                entries, source_texts, reference_texts,
                profiles_by_id=profiles_by_id,
                per_annotator=per_annotator,
                include_query=include_query,
            )
        return _tasks_cache[per_annotator]

    all_dfs = []
    all_raw_prefs = []

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

        rows = []
        if metric.is_pairwise:
            print(f"Computing {metric_name}...")
            raw_prefs, score_rows = _run_pairwise_comparisons(
                tasks, metric, text_key,
                metric_name=metric_name, cache=cache,
            )
            all_raw_prefs.extend(raw_prefs)
            rows.extend(score_rows)
        else:
            if not tasks:
                continue

            # Compute the first task to learn the sub-metric column names,
            # then decide whether to skip the rest via existing_df.
            task0 = tasks[0]
            first_scores = _score_cached(
                metric, metric_name, cfg_hash, task0, text_key, cache,
            )
            sub_metric_cols = list(first_scores.keys())

            if _metric_already_complete(existing_df, tasks, sub_metric_cols):
                print(f"Skipping {metric_name} (already in output CSV)")
                continue

            print(f"Computing {metric_name}...")
            row0 = {
                "query_index": task0["query_index"],
                "label": task0["label"],
                **first_scores,
            }
            if "annotator_id" in task0:
                row0["annotator_id"] = task0["annotator_id"]
            rows.append(row0)

            for task in tqdm(tasks[1:], desc=metric_name):
                scores = _score_cached(
                    metric, metric_name, cfg_hash, task, text_key, cache,
                )
                row = {
                    "query_index": task["query_index"],
                    "label": task["label"],
                    **scores,
                }
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
            partial_scores = _merge_score_dfs(all_dfs)
            partial_prefs = pd.DataFrame(all_raw_prefs) if all_raw_prefs else None
            on_metric_done(partial_scores, partial_prefs)

    scores_df = _merge_score_dfs(all_dfs)
    pairwise_prefs_df = pd.DataFrame(all_raw_prefs) if all_raw_prefs else None

    return scores_df, pairwise_prefs_df


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def cmd_list_metrics(args):
    """List all registered metrics."""
    for name in list_metrics():
        metric = get_metric(name)
        ref = "reference-free" if metric.is_reference_free else "reference-based"
        print(f"  {name:20s}  {metric.name} ({ref})")


def cmd_fetch_sources(args):
    """Fetch and cache OpenAlex source documents."""
    _, entries = load_annotations(args.annotations)
    client = OpenAlexClient(cache_dir=args.cache_dir, email=args.email)
    source_texts, reference_texts = client.get_texts_batch(entries)
    print(f"Fetched texts for {len(source_texts)} unique queries")
    n_empty_src = sum(1 for t in source_texts.values() if not t)
    n_empty_ref = sum(1 for t in reference_texts.values() if not t)
    if n_empty_src:
        print(f"  ({n_empty_src} queries had no abstracts available)")
    if n_empty_ref:
        print(f"  ({n_empty_ref} queries had no titles available)")


def cmd_compute_metrics(args):
    """Compute automatic metrics for all summaries."""
    profiles_by_id, entries, source_texts, reference_texts = _load_data(args)
    metric_names = _resolve_metrics(args.metrics)
    output = Path(args.output)
    output = output.parent / f"{output.stem}_run_{args.run_id}{output.suffix}"
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_run_params(output.parent / f"{output.stem}_params.json", args)
    _compute_and_save(
        entries, source_texts, reference_texts, profiles_by_id,
        metric_names, args, output,
    )


def cmd_correlate(args):
    """Compute correlations between metrics and human preferences."""
    _, entries = load_annotations(args.annotations)
    scores_df = pd.read_csv(args.scores)

    pairwise_prefs_df = None
    pairwise_prefs_path = getattr(args, "pairwise_prefs", None)
    if pairwise_prefs_path:
        pairwise_prefs_df = pd.read_csv(pairwise_prefs_path)

    output_dir = Path(args.output_dir) / f"run_{args.run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_run_params(output_dir / "run_params.json", args)

    agreement, agg, include_neither, neither_config, strict = _run_correlations(
        entries, scores_df, pairwise_prefs_df, args, output_dir,
    )

    print("\n=== Pairwise Agreement ===")
    if include_neither:
        print(f"(including 'neither' annotations with thresholds from {neither_config})")
    if strict:
        print("(strict mode: final disagreement when round1 is wrong)")
    print(agreement.to_string(index=False))

    print("\n=== Aggregate Rank Correlation ===")
    print(agg.to_string(index=False))


def cmd_robustness(args):
    """Run robustness tests on summarization metrics."""
    from persona_eval.robustness import (
        ALL_TESTS,
        PerturbationCache,
        analyze_robustness,
        load_from_persona_eval,
        print_robustness_report,
        run_robustness,
    )
    from persona_eval.robustness.dataset import load_from_huggingface

    output_dir = Path(args.output_dir) / f"run_{args.run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_run_params(output_dir / "run_params.json", args)

    # Load data: either from --dataset or from annotations
    dataset_name = getattr(args, "dataset", None)
    if dataset_name:
        samples = load_from_huggingface(
            dataset_name,
            split=getattr(args, "split", None),
            num_samples=args.num_samples,
            seed=args.seed,
        )
        print(f"Loaded {len(samples)} samples from {dataset_name}")
    elif args.annotations:
        profiles_by_id, entries, source_texts, reference_texts = _load_data(args)
        samples = load_from_persona_eval(
            entries, source_texts, reference_texts, profiles_by_id,
        )
        print(f"Loaded {len(samples)} samples")
    else:
        print("Error: provide either --dataset or annotations path")
        return

    # Optionally subsample
    if args.num_samples and args.num_samples < len(samples):
        import random
        rng = random.Random(args.seed)
        samples = rng.sample(samples, args.num_samples)
        print(f"Subsampled to {len(samples)} samples (seed={args.seed})")

    # Build perturbation tests
    test_names = args.tests or list(ALL_TESTS.keys())
    llm_client = None
    cache = None

    # Check if any selected test needs an LLM
    LLM_TESTS = {"lengthen_prose", "shorten_prose", "different_audience"}
    needs_llm = bool(LLM_TESTS & set(test_names))

    if needs_llm:
        from persona_eval.llm_client import LLMClient
        llm_kwargs = _collect_llm_kwargs(args)
        llm_kwargs.pop("include_query", None)
        llm_kwargs.pop("persona", None)
        llm_kwargs.pop("prompt_file", None)
        # Perturbation-specific overrides (fall back to --llm-* values)
        perturb_provider = getattr(args, "perturb_provider", None) or llm_kwargs.get("provider", "vllm")
        perturb_model = getattr(args, "perturb_model", None) or llm_kwargs.get("model")
        perturb_api_key = getattr(args, "perturb_api_key", None) or llm_kwargs.get("api_key")
        perturb_base_url = getattr(args, "perturb_base_url", None) or llm_kwargs.get("base_url")
        llm_client = LLMClient(
            provider=perturb_provider,
            model=perturb_model,
            api_key=perturb_api_key,
            base_url=perturb_base_url,
        )
        cache = PerturbationCache(args.cache_dir)

    distractor_pool = [s.source for s in samples]

    tests = []
    for tn in test_names:
        if tn not in ALL_TESTS:
            print(f"Warning: unknown test '{tn}', skipping")
            continue
        if tn == "distractor_sentences":
            tests.append(ALL_TESTS[tn](
                distractor_pool=distractor_pool, seed=args.seed,
            ))
        elif tn == "incremental_addition":
            tests.append(ALL_TESTS[tn]())
        elif tn == "different_audience":
            tests.append(ALL_TESTS[tn](
                llm_client=llm_client, cache=cache,
                target_audiences=args.target_audiences,
            ))
        else:
            # lengthen_prose, shorten_prose
            tests.append(ALL_TESTS[tn](llm_client=llm_client, cache=cache))

    # Run metrics
    metric_names = _resolve_metrics(args.metrics, default=["rouge"])
    metric_kwargs = {}
    for key in ("provider", "model", "api_key", "base_url"):
        attr = f"llm_{key}"
        val = getattr(args, attr, None)
        if val is not None:
            metric_kwargs[key] = val
    metric_kwargs["device"] = args.device

    response_logger = _create_response_logger(output_dir / "scores.csv")
    metric_cache = _build_metric_cache(args)

    scores_df = run_robustness(
        samples=samples,
        tests=tests,
        metric_names=metric_names,
        metric_kwargs=metric_kwargs,
        output_dir=output_dir,
        response_logger=response_logger,
        cache=metric_cache,
    )

    if response_logger is not None:
        response_logger.close()

    # Analyze
    analysis_df = analyze_robustness(scores_df, tests)
    analysis_path = output_dir / "robustness_analysis.csv"
    analysis_df.to_csv(analysis_path, index=False)
    print(f"\nSaved analysis to {analysis_path}")

    print_robustness_report(analysis_df)


def cmd_robustness_generate(args):
    """Generate perturbations only (no metric evaluation)."""
    from persona_eval.robustness import (
        ALL_TESTS,
        PerturbationCache,
        generate_perturbations,
        load_from_persona_eval,
        save_perturbations,
    )
    from persona_eval.robustness.dataset import load_from_huggingface

    output_dir = Path(args.output_dir) / f"run_{args.run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_run_params(output_dir / "run_params.json", args)

    # Load data
    dataset_name = getattr(args, "dataset", None)
    if dataset_name:
        samples = load_from_huggingface(
            dataset_name,
            split=getattr(args, "split", None),
            num_samples=args.num_samples,
            seed=args.seed,
        )
        print(f"Loaded {len(samples)} samples from {dataset_name}")
    elif args.annotations:
        profiles_by_id, entries, source_texts, reference_texts = _load_data(args)
        samples = load_from_persona_eval(
            entries, source_texts, reference_texts, profiles_by_id,
        )
        print(f"Loaded {len(samples)} samples")
    else:
        print("Error: provide either --dataset or annotations path")
        return

    # Optionally subsample
    if args.num_samples and args.num_samples < len(samples):
        import random
        rng = random.Random(args.seed)
        samples = rng.sample(samples, args.num_samples)
        print(f"Subsampled to {len(samples)} samples (seed={args.seed})")

    # Build perturbation tests
    test_names = args.tests or list(ALL_TESTS.keys())
    llm_client = None
    cache = None

    LLM_TESTS = {"lengthen_prose", "shorten_prose", "different_audience"}
    needs_llm = bool(LLM_TESTS & set(test_names))

    if needs_llm:
        from persona_eval.llm_client import LLMClient
        llm_kwargs = _collect_llm_kwargs(args)
        llm_kwargs.pop("include_query", None)
        llm_kwargs.pop("persona", None)
        llm_kwargs.pop("prompt_file", None)
        perturb_provider = getattr(args, "perturb_provider", None) or llm_kwargs.get("provider", "vllm")
        perturb_model = getattr(args, "perturb_model", None) or llm_kwargs.get("model")
        perturb_api_key = getattr(args, "perturb_api_key", None) or llm_kwargs.get("api_key")
        perturb_base_url = getattr(args, "perturb_base_url", None) or llm_kwargs.get("base_url")
        llm_client = LLMClient(
            provider=perturb_provider,
            model=perturb_model,
            api_key=perturb_api_key,
            base_url=perturb_base_url,
        )
        cache = PerturbationCache(args.cache_dir)

    distractor_pool = [s.source for s in samples]

    tests = []
    for tn in test_names:
        if tn not in ALL_TESTS:
            print(f"Warning: unknown test '{tn}', skipping")
            continue
        if tn == "distractor_sentences":
            tests.append(ALL_TESTS[tn](
                distractor_pool=distractor_pool, seed=args.seed,
            ))
        elif tn == "incremental_addition":
            tests.append(ALL_TESTS[tn]())
        elif tn == "different_audience":
            tests.append(ALL_TESTS[tn](
                llm_client=llm_client, cache=cache,
                target_audiences=args.target_audiences,
            ))
        else:
            tests.append(ALL_TESTS[tn](llm_client=llm_client, cache=cache))

    # Generate and save
    all_results = generate_perturbations(samples, tests)
    perturb_dir = save_perturbations(all_results, samples, output_dir)
    print(f"\nPerturbations saved to {perturb_dir}")
    print("Run 'persona-eval robustness-eval' to evaluate metrics on these perturbations.")


def cmd_robustness_eval(args):
    """Evaluate metrics on previously generated perturbations."""
    from persona_eval.robustness import (
        ALL_TESTS,
        analyze_robustness,
        load_perturbations,
        print_robustness_report,
        score_perturbations,
    )

    output_dir = Path(args.output_dir) / f"run_{args.run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_run_params(output_dir / "run_params.json", args)

    # Load saved perturbations
    all_results, samples = load_perturbations(args.perturbations_dir)

    # Reconstruct test objects (needed for expected_direction in analysis)
    test_names_in_results = sorted(set(r.test_name for r in all_results))
    tests = []
    distractor_pool = [s.source for s in samples]
    for tn in test_names_in_results:
        if tn not in ALL_TESTS:
            print(f"Warning: unknown test '{tn}' in saved perturbations, skipping analysis for it")
            continue
        if tn == "distractor_sentences":
            tests.append(ALL_TESTS[tn](distractor_pool=distractor_pool))
        elif tn == "incremental_addition":
            tests.append(ALL_TESTS[tn]())
        elif tn in ("lengthen_prose", "shorten_prose"):
            # These need llm_client/cache for generation, but we only need
            # the object for expected_direction — pass None.
            tests.append(ALL_TESTS[tn](llm_client=None, cache=None))
        elif tn == "different_audience":
            tests.append(ALL_TESTS[tn](llm_client=None, cache=None))

    # Score
    metric_names = _resolve_metrics(args.metrics, default=["rouge"])
    metric_kwargs = {}
    for key in ("provider", "model", "api_key", "base_url"):
        attr = f"llm_{key}"
        val = getattr(args, attr, None)
        if val is not None:
            metric_kwargs[key] = val
    metric_kwargs["device"] = args.device

    response_logger = _create_response_logger(output_dir / "scores.csv")
    metric_cache = _build_metric_cache(args)

    scores_df = score_perturbations(
        all_results=all_results,
        samples=samples,
        metric_names=metric_names,
        metric_kwargs=metric_kwargs,
        output_dir=output_dir,
        response_logger=response_logger,
        cache=metric_cache,
    )

    if response_logger is not None:
        response_logger.close()

    # Analyze
    analysis_df = analyze_robustness(scores_df, tests)
    analysis_path = output_dir / "robustness_analysis.csv"
    analysis_df.to_csv(analysis_path, index=False)
    print(f"\nSaved analysis to {analysis_path}")

    print_robustness_report(analysis_df)


def cmd_run_all(args):
    """Run the full pipeline: fetch sources, compute metrics, correlate."""
    output_dir = Path(args.output_dir) / f"run_{args.run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_run_params(output_dir / "run_params.json", args)

    # Load and fetch
    profiles_by_id, entries, source_texts, reference_texts = _load_data(args)
    print(f"Loaded {len(entries)} annotations from {len(profiles_by_id)} annotators")
    print(f"Fetched texts for {len(source_texts)} unique queries")

    # Compute metrics
    metric_names = _resolve_metrics(args.metrics)
    scores_df, pairwise_prefs_df = _compute_and_save(
        entries, source_texts, reference_texts, profiles_by_id,
        metric_names, args, output_dir / "metric_scores.csv",
    )

    # Correlations
    agreement, agg, _, _, _ = _run_correlations(
        entries, scores_df, pairwise_prefs_df, args, output_dir,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n--- Pairwise Agreement ---")
    print(agreement.to_string(index=False))

    print("\n--- Aggregate Rank Correlation ---")
    print(agg.to_string(index=False))

    print(f"\nAll outputs saved to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI argument helpers
# ---------------------------------------------------------------------------


def _add_data_args(parser):
    """Add common data-loading arguments."""
    parser.add_argument("annotations", help="Path to annotations zip or directory")
    parser.add_argument("--cache-dir", default="cache", help="Cache directory")
    parser.add_argument("--email", help="Email for OpenAlex polite pool")


def _add_metric_args(parser):
    """Add metric selection arguments."""
    parser.add_argument("--metrics", nargs="+", help="Metrics to compute (default: all). Use 'all' for all metrics, 'non-llm' for non-LLM metrics, 'all-llm' for LLM-based metrics.")
    parser.add_argument("--device", default="cpu", help="Device for model inference (cpu, cuda, cuda:0, etc.)")
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable the metric result cache (cache is on by default).",
    )
    parser.add_argument(
        "--clear-metric-cache", action="store_true",
        help="Delete all entries from the metric cache before running.",
    )


def _add_llm_args(parser):
    """Add LLM-related arguments to a subcommand parser."""
    group = parser.add_argument_group("LLM options (for llm_judge and factscore metrics)")
    group.add_argument(
        "--llm-provider", choices=["vllm", "together"], default="vllm",
        help="LLM backend provider (default: vllm)",
    )
    group.add_argument(
        "--llm-model",
        help="Model name or path (required for llm_judge/factscore metrics)",
    )
    group.add_argument(
        "--llm-api-key",
        help="API key (or set TOGETHER_API_KEY env var for together provider)",
    )
    group.add_argument(
        "--llm-base-url",
        help="Override base URL (default: http://localhost:8000/v1 for vllm)",
    )
    group.add_argument(
        "--llm-prompt-file",
        help="Path to custom prompt template for llm_judge metric",
    )
    group.add_argument(
        "--persona", action="store_true",
        help="Enable persona-aware evaluation using annotator profiles",
    )
    group.add_argument(
        "--include-query", action="store_true",
        help="Include the annotator's query in LLM judge prompts",
    )


def _add_perturb_llm_args(parser):
    """Add perturbation-specific LLM arguments (override --llm-* for generation)."""
    group = parser.add_argument_group(
        "Perturbation LLM options (override --llm-* for perturbation generation)"
    )
    group.add_argument(
        "--perturb-provider", choices=["vllm", "together"],
        help="LLM provider for perturbation generation (default: same as --llm-provider)",
    )
    group.add_argument(
        "--perturb-model",
        help="Model for perturbation generation (default: same as --llm-model)",
    )
    group.add_argument(
        "--perturb-api-key",
        help="API key for perturbation LLM (default: same as --llm-api-key)",
    )
    group.add_argument(
        "--perturb-base-url",
        help="Base URL for perturbation LLM (default: same as --llm-base-url)",
    )


def _add_correlation_args(parser):
    """Add correlation-related arguments."""
    parser.add_argument("--include-neither", action="store_true",
                        help="Include 'neither' annotations in pairwise agreement")
    parser.add_argument("--neither-config", default=None,
                        help="Path to YAML config with per-metric thresholds for 'neither' agreement")
    parser.add_argument("--strict-pairwise", action="store_true",
                        help="Strict mode: auto-disagree on final when metric got round1 wrong")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        prog="persona-eval",
        description="Evaluate summarization metrics against human preferences",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list-metrics
    sp = subparsers.add_parser("list-metrics", help="List available metrics")
    sp.set_defaults(func=cmd_list_metrics)

    # fetch-sources
    sp = subparsers.add_parser("fetch-sources", help="Fetch OpenAlex source documents")
    _add_data_args(sp)
    sp.set_defaults(func=cmd_fetch_sources)

    # compute-metrics
    sp = subparsers.add_parser("compute-metrics", help="Compute automatic metrics")
    _add_data_args(sp)
    _add_metric_args(sp)
    sp.add_argument("--output", default="metric_scores.csv", help="Output CSV path")
    _add_llm_args(sp)
    sp.set_defaults(func=cmd_compute_metrics)

    # correlate
    sp = subparsers.add_parser("correlate", help="Compute metric-human correlations")
    sp.add_argument("annotations", help="Path to annotations zip or directory")
    sp.add_argument("--scores", required=True, help="Path to metric scores CSV")
    sp.add_argument("--pairwise-prefs", default=None,
                    help="Path to raw pairwise preferences CSV (from pairwise metrics)")
    _add_correlation_args(sp)
    sp.add_argument("--output-dir", default="results", help="Output directory")
    sp.set_defaults(func=cmd_correlate)

    # robustness
    sp = subparsers.add_parser("robustness", help="Run robustness tests on metrics")
    sp.add_argument("annotations", nargs="?", default=None,
                    help="Path to annotations zip or directory (not needed with --dataset)")
    sp.add_argument("--cache-dir", default="cache", help="Cache directory")
    sp.add_argument("--email", help="Email for OpenAlex polite pool")
    _add_metric_args(sp)
    _add_llm_args(sp)
    _add_perturb_llm_args(sp)
    sp.add_argument("--output-dir", default="robustness_results", help="Output directory")
    sp.add_argument(
        "--dataset",
        help="Load a standard dataset instead of annotations. "
             "Available: arxiv, scitldr, pubmed, elife, plos, mup. "
             "Stubs (not yet loadable): cdsr, eureka, cells, scinews, longsumm.",
    )
    sp.add_argument("--split", default=None,
                    help="Dataset split to use (default: test, or dataset-specific default)")
    sp.add_argument(
        "--tests", nargs="+",
        help="Robustness tests to run (default: all). "
             "Choices: distractor_sentences, incremental_addition, lengthen_prose, shorten_prose, different_audience",
    )
    sp.add_argument("--num-samples", type=int, default=None,
                    help="Subsample N samples to limit compute")
    sp.add_argument("--seed", type=int, default=42, help="Random seed")
    sp.add_argument("--target-audiences", nargs="+", default=None,
                    help="Target audiences for the audience rewrite test")
    sp.set_defaults(func=cmd_robustness)

    # robustness-generate (perturbation generation only)
    sp = subparsers.add_parser(
        "robustness-generate",
        help="Generate perturbations only (no metric evaluation)",
    )
    sp.add_argument("annotations", nargs="?", default=None,
                    help="Path to annotations zip or directory (not needed with --dataset)")
    sp.add_argument("--cache-dir", default="cache", help="Cache directory")
    sp.add_argument("--email", help="Email for OpenAlex polite pool")
    _add_llm_args(sp)
    _add_perturb_llm_args(sp)
    sp.add_argument("--output-dir", default="robustness_results", help="Output directory")
    sp.add_argument(
        "--dataset",
        help="Load a standard dataset instead of annotations. "
             "Available: arxiv, scitldr, pubmed, elife, plos, mup, scinews.",
    )
    sp.add_argument("--split", default=None,
                    help="Dataset split to use (default: test, or dataset-specific default)")
    sp.add_argument(
        "--tests", nargs="+",
        help="Robustness tests to run (default: all). "
             "Choices: distractor_sentences, incremental_addition, lengthen_prose, shorten_prose, different_audience",
    )
    sp.add_argument("--num-samples", type=int, default=None,
                    help="Subsample N samples to limit compute")
    sp.add_argument("--seed", type=int, default=42, help="Random seed")
    sp.add_argument("--target-audiences", nargs="+", default=None,
                    help="Target audiences for the audience rewrite test")
    sp.set_defaults(func=cmd_robustness_generate)

    # robustness-eval (evaluate metrics on saved perturbations)
    sp = subparsers.add_parser(
        "robustness-eval",
        help="Evaluate metrics on previously generated perturbations",
    )
    sp.add_argument("--perturbations-dir", required=True,
                    help="Path to saved perturbations directory (from robustness-generate)")
    sp.add_argument("--cache-dir", default="cache", help="Cache directory")
    _add_metric_args(sp)
    _add_llm_args(sp)
    sp.add_argument("--output-dir", default="robustness_results", help="Output directory")
    sp.set_defaults(func=cmd_robustness_eval)

    # run-all
    sp = subparsers.add_parser("run-all", help="Run full pipeline")
    _add_data_args(sp)
    _add_metric_args(sp)
    _add_correlation_args(sp)
    sp.add_argument("--output-dir", default="results", help="Output directory")
    _add_llm_args(sp)
    sp.set_defaults(func=cmd_run_all)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.func(args)


if __name__ == "__main__":
    main()
