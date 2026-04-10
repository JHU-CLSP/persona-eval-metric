"""CLI entry point for the persona evaluation pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
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
from persona_eval.metrics import get_metric, list_metrics
from persona_eval.openalex import OpenAlexClient


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


_LLM_METRICS = {"llm_judge", "llm_judge_relative", "llm_judge_annotator", "factscore"}


def _make_persona_kwargs(profile: AnnotatorProfile | None) -> dict | None:
    """Build persona template kwargs from an annotator profile."""
    if profile is None:
        return None
    return {
        "role": profile.role or "unspecified",
        "domain": profile.domain or "unspecified",
        "info_needs": profile.info_needs or "unspecified",
    }


def _run_pairwise_comparisons(
    tasks: list[dict],
    metric,
    text_key: str,
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
        result_ab = metric.score_pair(
            label_tasks["A"]["summary"],
            label_tasks["B"]["summary"],
            source,
            persona_kwargs=pk,
        )
        if sub_metrics is None:
            sub_metrics = list(result_ab.keys())
            for l in ("A", "B", "C", "D"):
                scores[l] = {sm: 0.0 for sm in sub_metrics}

        # --- Round 1: C vs D ---
        result_cd = metric.score_pair(
            label_tasks["C"]["summary"],
            label_tasks["D"]["summary"],
            source,
            persona_kwargs=pk,
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
            result_final = metric.score_pair(
                label_tasks[w_ab]["summary"],
                label_tasks[w_cd]["summary"],
                source,
                persona_kwargs=pk,
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


def _compute_metric_scores(
    entries: list[AnnotationEntry],
    source_texts: dict[int, str],
    reference_texts: dict[int, str],
    metric_names: list[str],
    device: str = "cpu",
    llm_kwargs: dict | None = None,
    profiles_by_id: dict[str, AnnotatorProfile] | None = None,
    response_logger=None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Compute metric scores for all (query, summary) pairs.

    Reference-free metrics receive concatenated abstracts as their source.
    Reference-based metrics receive concatenated titles as their source.
    Pairwise metrics run a tournament and store both raw preferences
    and tournament point scores.

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
            if mn in _LLM_METRICS:
                m = get_metric(mn, **{**{"device": device}, **llm_kwargs})
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

    rows = []
    all_raw_prefs = []

    for metric_name in metric_names:
        print(f"Computing {metric_name}...")
        kwargs = {"device": device}
        if metric_name in _LLM_METRICS:
            kwargs.update(llm_kwargs)
            if response_logger is not None:
                kwargs["response_logger"] = response_logger
        metric = get_metric(metric_name, **kwargs)
        text_key = "source" if metric.is_reference_free else "reference"
        tasks = get_tasks(per_annotator=metric.needs_persona)

        if metric.is_pairwise:
            raw_prefs, score_rows = _run_pairwise_comparisons(
                tasks, metric, text_key,
            )
            all_raw_prefs.extend(raw_prefs)
            rows.extend(score_rows)
        else:
            for task in tqdm(tasks, desc=metric_name):
                score_kwargs = {}
                if metric_name in _LLM_METRICS:
                    score_kwargs["persona_kwargs"] = task.get("persona_kwargs")
                scores = metric.score(
                    task["summary"], task[text_key],
                    **score_kwargs,
                )
                row = {
                    "query_index": task["query_index"],
                    "label": task["label"],
                    **scores,
                }
                if "annotator_id" in task:
                    row["annotator_id"] = task["annotator_id"]
                rows.append(row)

    # Merge all metric scores into one row per key
    if not rows:
        scores_df = pd.DataFrame()
    else:
        scores_df = pd.DataFrame(rows)
        group_cols = ["query_index", "label"]
        if "annotator_id" in scores_df.columns:
            group_cols.insert(0, "annotator_id")
        scores_df = scores_df.groupby(group_cols, as_index=False).first()

    pairwise_prefs_df = pd.DataFrame(all_raw_prefs) if all_raw_prefs else None

    return scores_df, pairwise_prefs_df


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


def _create_response_logger(output_path: str | Path, metric_names: list[str]):
    """Create a ResponseLogger if any LLM metrics are being computed."""
    if not any(m in _LLM_METRICS for m in metric_names):
        return None
    from persona_eval.response_logger import ResponseLogger
    log_dir = Path(output_path).parent / "llm_responses"
    return ResponseLogger(log_dir)


def cmd_compute_metrics(args):
    """Compute automatic metrics for all summaries."""
    profiles, entries = load_annotations(args.annotations)
    profiles_by_id = {p.annotator_id: p for p in profiles}
    client = OpenAlexClient(cache_dir=args.cache_dir, email=args.email)
    source_texts, reference_texts = client.get_texts_batch(entries)

    metric_names = args.metrics if args.metrics else list_metrics()
    llm_kwargs = _collect_llm_kwargs(args)
    response_logger = _create_response_logger(args.output, metric_names)
    scores_df, pairwise_prefs_df = _compute_metric_scores(
        entries, source_texts, reference_texts,
        metric_names, device=args.device,
        llm_kwargs=llm_kwargs, profiles_by_id=profiles_by_id,
        response_logger=response_logger,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(output, index=False)
    print(f"Saved metric scores to {output} ({len(scores_df)} rows)")

    if pairwise_prefs_df is not None:
        prefs_path = output.parent / (output.stem + "_pairwise_prefs.csv")
        pairwise_prefs_df.to_csv(prefs_path, index=False)
        print(f"Saved raw pairwise preferences to {prefs_path}")

    if response_logger is not None:
        print(f"Saved LLM responses to {response_logger.path}")
        response_logger.close()


def _load_neither_thresholds(path: str | None) -> dict[str, float] | None:
    """Load per-metric neither thresholds from a YAML config file."""
    if path is None:
        return None
    with open(path) as f:
        data = yaml.safe_load(f)
    return {str(k): float(v) for k, v in data.items()}


def cmd_correlate(args):
    """Compute correlations between metrics and human preferences."""
    _, entries = load_annotations(args.annotations)
    scores_df = pd.read_csv(args.scores)

    pairwise_prefs_df = None
    pairwise_prefs_path = getattr(args, "pairwise_prefs", None)
    if pairwise_prefs_path:
        pairwise_prefs_df = pd.read_csv(pairwise_prefs_path)

    include_neither = getattr(args, "include_neither", False)
    neither_config = getattr(args, "neither_config", None)
    neither_thresholds = _load_neither_thresholds(neither_config) if include_neither else None

    preferences = get_pairwise_preferences(entries, include_neither=include_neither)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strict = getattr(args, "strict_pairwise", False)

    # Pairwise agreement
    agreement = compute_pairwise_agreement(
        preferences, scores_df, neither_thresholds=neither_thresholds,
        strict=strict, pairwise_prefs=pairwise_prefs_df,
    )
    agreement.to_csv(output_dir / "pairwise_agreement.csv", index=False)
    print("\n=== Pairwise Agreement ===")
    if include_neither:
        print(f"(including 'neither' annotations with thresholds from {neither_config})")
    if strict:
        print("(strict mode: final disagreement when round1 is wrong)")
    print(agreement.to_string(index=False))

    # Rank correlation
    per_query = compute_rank_correlation(entries, scores_df)
    per_query.to_csv(output_dir / "rank_correlation_per_query.csv", index=False)

    agg = aggregate_correlations(per_query)
    agg.to_csv(output_dir / "rank_correlation_aggregate.csv", index=False)
    print("\n=== Aggregate Rank Correlation ===")
    print(agg.to_string(index=False))


def cmd_run_all(args):
    """Run the full pipeline: fetch sources, compute metrics, correlate."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    profiles, entries = load_annotations(args.annotations)
    profiles_by_id = {p.annotator_id: p for p in profiles}
    print(f"Loaded {len(entries)} annotations from {len(profiles)} annotators")

    # Fetch sources
    client = OpenAlexClient(cache_dir=args.cache_dir, email=args.email)
    source_texts, reference_texts = client.get_texts_batch(entries)
    print(f"Fetched texts for {len(source_texts)} unique queries")

    # Compute metrics
    metric_names = args.metrics if args.metrics else list_metrics()
    llm_kwargs = _collect_llm_kwargs(args)
    response_logger = _create_response_logger(output_dir / "metric_scores.csv", metric_names)
    scores_df, pairwise_prefs_df = _compute_metric_scores(
        entries, source_texts, reference_texts,
        metric_names, device=args.device,
        llm_kwargs=llm_kwargs, profiles_by_id=profiles_by_id,
        response_logger=response_logger,
    )
    scores_path = output_dir / "metric_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"Saved metric scores to {scores_path}")

    if response_logger is not None:
        print(f"Saved LLM responses to {response_logger.path}")
        response_logger.close()

    if pairwise_prefs_df is not None:
        prefs_path = output_dir / "pairwise_prefs.csv"
        pairwise_prefs_df.to_csv(prefs_path, index=False)
        print(f"Saved raw pairwise preferences to {prefs_path}")

    # Compute correlations
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

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n--- Pairwise Agreement ---")
    print(agreement.to_string(index=False))

    print("\n--- Aggregate Rank Correlation ---")
    print(agg.to_string(index=False))

    print(f"\nAll outputs saved to {output_dir}/")


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
    sp.add_argument("annotations", help="Path to annotations zip or directory")
    sp.add_argument("--cache-dir", default="cache", help="Cache directory")
    sp.add_argument("--email", help="Email for OpenAlex polite pool")
    sp.set_defaults(func=cmd_fetch_sources)

    # compute-metrics
    sp = subparsers.add_parser("compute-metrics", help="Compute automatic metrics")
    sp.add_argument("annotations", help="Path to annotations zip or directory")
    sp.add_argument("--metrics", nargs="+", help="Metrics to compute (default: all)")
    sp.add_argument("--cache-dir", default="cache", help="Cache directory")
    sp.add_argument("--email", help="Email for OpenAlex polite pool")
    sp.add_argument("--device", default="cpu", help="Device for model inference (cpu, cuda, cuda:0, etc.)")
    sp.add_argument("--output", default="metric_scores.csv", help="Output CSV path")
    _add_llm_args(sp)
    sp.set_defaults(func=cmd_compute_metrics)

    # correlate
    sp = subparsers.add_parser("correlate", help="Compute metric-human correlations")
    sp.add_argument("annotations", help="Path to annotations zip or directory")
    sp.add_argument("--scores", required=True, help="Path to metric scores CSV")
    sp.add_argument("--pairwise-prefs", default=None,
                    help="Path to raw pairwise preferences CSV (from pairwise metrics)")
    sp.add_argument("--include-neither", action="store_true",
                    help="Include 'neither' annotations in pairwise agreement")
    sp.add_argument("--neither-config", default=None,
                    help="Path to YAML config with per-metric thresholds for 'neither' agreement")
    sp.add_argument("--strict-pairwise", action="store_true",
                    help="Strict mode: auto-disagree on final when metric got round1 wrong")
    sp.add_argument("--output-dir", default="results", help="Output directory")
    sp.set_defaults(func=cmd_correlate)

    # run-all
    sp = subparsers.add_parser("run-all", help="Run full pipeline")
    sp.add_argument("annotations", help="Path to annotations zip or directory")
    sp.add_argument("--metrics", nargs="+", help="Metrics to compute (default: all)")
    sp.add_argument("--cache-dir", default="cache", help="Cache directory")
    sp.add_argument("--email", help="Email for OpenAlex polite pool")
    sp.add_argument("--device", default="cpu", help="Device for model inference (cpu, cuda, cuda:0, etc.)")
    sp.add_argument("--include-neither", action="store_true",
                    help="Include 'neither' annotations in pairwise agreement")
    sp.add_argument("--neither-config", default=None,
                    help="Path to YAML config with per-metric thresholds for 'neither' agreement")
    sp.add_argument("--strict-pairwise", action="store_true",
                    help="Strict mode: auto-disagree on final when metric got round1 wrong")
    sp.add_argument("--output-dir", default="results", help="Output directory")
    _add_llm_args(sp)
    sp.set_defaults(func=cmd_run_all)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    args.func(args)


if __name__ == "__main__":
    main()
