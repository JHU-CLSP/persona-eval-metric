"""CLI entry point for the persona evaluation pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from persona_eval.annotations import AnnotationEntry, load_annotations, get_pairwise_preferences
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


_LLM_METRICS = {"llm_judge", "llm_judge_relative", "factscore"}


def _compute_pairwise_tournament(
    tasks: list[dict],
    metric,
    text_key: str,
) -> list[dict]:
    """Compute tournament scores for a pairwise metric.

    Mirrors the human annotation tournament structure:
      Round 1: A vs B, C vs D
      Final:   winner of AB vs winner of CD

    Each summary receives a score based on tournament progression:
      - Final winner: 3 points (won round + won final)
      - Final loser:  1 point  (won round, lost final)
      - Round losers: 0 points
      - Ties award 0.5 to each side in that round

    These scores integrate naturally with the existing correlation
    pipeline (pairwise agreement and rank correlation).
    """
    # Group tasks by query_index, keyed by label
    by_query: dict[int, dict[str, dict]] = {}
    for task in tasks:
        by_query.setdefault(task["query_index"], {})[task["label"]] = task

    rows = []
    for qi, label_tasks in tqdm(by_query.items(), desc=metric.name):
        # Need all four summaries for the tournament
        if not all(l in label_tasks for l in ("A", "B", "C", "D")):
            continue

        source = label_tasks["A"][text_key]
        sub_metrics = None
        # Per-label, per-sub-metric scores
        scores: dict[str, dict[str, float]] = {
            l: {} for l in ("A", "B", "C", "D")
        }

        # --- Round 1: A vs B ---
        result_ab = metric.score_pair(
            label_tasks["A"]["summary"],
            label_tasks["B"]["summary"],
            source,
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
        )

        # Determine round winners per sub-metric and award points
        ab_winners: dict[str, str] = {}  # sub-metric -> "A", "B", or "tie"
        cd_winners: dict[str, str] = {}

        for sm in sub_metrics:
            # A vs B
            pref_ab = result_ab[sm]
            if pref_ab == "A":
                scores["A"][sm] += 1.0
                ab_winners[sm] = "A"
            elif pref_ab == "B":
                scores["B"][sm] += 1.0
                ab_winners[sm] = "B"
            else:
                scores["A"][sm] += 0.5
                scores["B"][sm] += 0.5
                ab_winners[sm] = "A"  # tiebreak: first label advances

            # C vs D
            pref_cd = result_cd[sm]
            if pref_cd == "A":  # "A" means first arg = C
                scores["C"][sm] += 1.0
                cd_winners[sm] = "C"
            elif pref_cd == "B":  # "B" means second arg = D
                scores["D"][sm] += 1.0
                cd_winners[sm] = "D"
            else:
                scores["C"][sm] += 0.5
                scores["D"][sm] += 0.5
                cd_winners[sm] = "C"  # tiebreak: first label advances

        # --- Final: winner of AB vs winner of CD ---
        # We need to run the final for each sub-metric's winners.
        # Group sub-metrics by the same (ab_winner, cd_winner) pair to
        # minimise LLM calls.
        final_pairs: dict[tuple[str, str], list[str]] = {}
        for sm in sub_metrics:
            pair = (ab_winners[sm], cd_winners[sm])
            final_pairs.setdefault(pair, []).append(sm)

        for (w_ab, w_cd), sms in final_pairs.items():
            result_final = metric.score_pair(
                label_tasks[w_ab]["summary"],
                label_tasks[w_cd]["summary"],
                source,
            )
            for sm in sms:
                pref_final = result_final[sm]
                if pref_final == "A":  # first arg = AB winner
                    scores[w_ab][sm] += 2.0
                elif pref_final == "B":  # second arg = CD winner
                    scores[w_cd][sm] += 2.0
                else:
                    scores[w_ab][sm] += 1.0
                    scores[w_cd][sm] += 1.0

        for label in ("A", "B", "C", "D"):
            rows.append({
                "query_index": qi,
                "label": label,
                **scores[label],
            })

    return rows


def _compute_metric_scores(
    entries: list[AnnotationEntry],
    source_texts: dict[int, str],
    reference_texts: dict[int, str],
    metric_names: list[str],
    device: str = "cpu",
    llm_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Compute metric scores for all (query, summary) pairs.

    Reference-free metrics receive concatenated abstracts as their source.
    Reference-based metrics receive concatenated titles as their source.
    Pairwise metrics compute win-rates across all pairs within each query.
    """
    # Deduplicate: compute once per unique (query_index, label)
    seen = set()
    tasks = []
    for entry in entries:
        for label in ("A", "B", "C", "D"):
            key = (entry.query_index, label)
            if key not in seen and label in entry.summaries:
                seen.add(key)
                tasks.append(
                    {
                        "query_index": entry.query_index,
                        "label": label,
                        "summary": entry.summaries[label],
                        "source": source_texts.get(entry.query_index, ""),
                        "reference": reference_texts.get(entry.query_index, ""),
                    }
                )

    llm_kwargs = llm_kwargs or {}

    rows = []
    for metric_name in metric_names:
        print(f"Computing {metric_name}...")
        kwargs = {"device": device}
        if metric_name in _LLM_METRICS:
            kwargs.update(llm_kwargs)
        metric = get_metric(metric_name, **kwargs)
        text_key = "source" if metric.is_reference_free else "reference"

        if metric.is_pairwise:
            rows.extend(_compute_pairwise_tournament(tasks, metric, text_key))
        else:
            for task in tqdm(tasks, desc=metric_name):
                scores = metric.score(task["summary"], task[text_key])
                rows.append(
                    {
                        "query_index": task["query_index"],
                        "label": task["label"],
                        **scores,
                    }
                )

    # Merge all metric scores into one row per (query_index, label)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Group by (query_index, label) and combine columns from different metrics
    df = df.groupby(["query_index", "label"], as_index=False).first()
    return df


def _collect_llm_kwargs(args) -> dict:
    """Collect LLM-related kwargs from CLI args."""
    kwargs = {}
    for key in ("provider", "model", "api_key", "base_url", "prompt_file"):
        attr = f"llm_{key}"
        val = getattr(args, attr, None)
        if val is not None:
            kwargs[key] = val
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


def cmd_compute_metrics(args):
    """Compute automatic metrics for all summaries."""
    _, entries = load_annotations(args.annotations)
    client = OpenAlexClient(cache_dir=args.cache_dir, email=args.email)
    source_texts, reference_texts = client.get_texts_batch(entries)

    metric_names = args.metrics if args.metrics else list_metrics()
    llm_kwargs = _collect_llm_kwargs(args)
    scores_df = _compute_metric_scores(entries, source_texts, reference_texts,
                                        metric_names, device=args.device,
                                        llm_kwargs=llm_kwargs)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(output, index=False)
    print(f"Saved metric scores to {output} ({len(scores_df)} rows)")


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
        strict=strict,
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
    print(f"Loaded {len(entries)} annotations from {len(profiles)} annotators")

    # Fetch sources
    client = OpenAlexClient(cache_dir=args.cache_dir, email=args.email)
    source_texts, reference_texts = client.get_texts_batch(entries)
    print(f"Fetched texts for {len(source_texts)} unique queries")

    # Compute metrics
    metric_names = args.metrics if args.metrics else list_metrics()
    llm_kwargs = _collect_llm_kwargs(args)
    scores_df = _compute_metric_scores(entries, source_texts, reference_texts,
                                        metric_names, device=args.device,
                                        llm_kwargs=llm_kwargs)
    scores_path = output_dir / "metric_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"Saved metric scores to {scores_path}")

    # Compute correlations
    include_neither = getattr(args, "include_neither", False)
    neither_config = getattr(args, "neither_config", None)
    neither_thresholds = _load_neither_thresholds(neither_config) if include_neither else None

    strict = getattr(args, "strict_pairwise", False)
    preferences = get_pairwise_preferences(entries, include_neither=include_neither)

    agreement = compute_pairwise_agreement(
        preferences, scores_df, neither_thresholds=neither_thresholds,
        strict=strict,
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
