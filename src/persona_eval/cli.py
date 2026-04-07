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


def _compute_metric_scores(
    entries: list[AnnotationEntry],
    source_texts: dict[int, str],
    reference_texts: dict[int, str],
    metric_names: list[str],
    device: str = "cpu",
) -> pd.DataFrame:
    """Compute metric scores for all (query, summary) pairs.

    Reference-free metrics receive concatenated abstracts as their source.
    Reference-based metrics receive concatenated titles as their source.
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

    rows = []
    for metric_name in metric_names:
        print(f"Computing {metric_name}...")
        metric = get_metric(metric_name, device=device)
        text_key = "source" if metric.is_reference_free else "reference"

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


def cmd_compute_metrics(args):
    """Compute automatic metrics for all summaries."""
    _, entries = load_annotations(args.annotations)
    client = OpenAlexClient(cache_dir=args.cache_dir, email=args.email)
    source_texts, reference_texts = client.get_texts_batch(entries)

    metric_names = args.metrics if args.metrics else list_metrics()
    scores_df = _compute_metric_scores(entries, source_texts, reference_texts,
                                        metric_names, device=args.device)

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

    # Pairwise agreement
    agreement = compute_pairwise_agreement(
        preferences, scores_df, neither_thresholds=neither_thresholds,
    )
    agreement.to_csv(output_dir / "pairwise_agreement.csv", index=False)
    print("\n=== Pairwise Agreement ===")
    if include_neither:
        print(f"(including 'neither' annotations with thresholds from {neither_config})")
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
    scores_df = _compute_metric_scores(entries, source_texts, reference_texts,
                                        metric_names, device=args.device)
    scores_path = output_dir / "metric_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"Saved metric scores to {scores_path}")

    # Compute correlations
    include_neither = getattr(args, "include_neither", False)
    neither_config = getattr(args, "neither_config", None)
    neither_thresholds = _load_neither_thresholds(neither_config) if include_neither else None

    preferences = get_pairwise_preferences(entries, include_neither=include_neither)

    agreement = compute_pairwise_agreement(
        preferences, scores_df, neither_thresholds=neither_thresholds,
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
    sp.set_defaults(func=cmd_compute_metrics)

    # correlate
    sp = subparsers.add_parser("correlate", help="Compute metric-human correlations")
    sp.add_argument("annotations", help="Path to annotations zip or directory")
    sp.add_argument("--scores", required=True, help="Path to metric scores CSV")
    sp.add_argument("--include-neither", action="store_true",
                    help="Include 'neither' annotations in pairwise agreement")
    sp.add_argument("--neither-config", default=None,
                    help="Path to YAML config with per-metric thresholds for 'neither' agreement")
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
    sp.add_argument("--output-dir", default="results", help="Output directory")
    sp.set_defaults(func=cmd_run_all)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    args.func(args)


if __name__ == "__main__":
    main()
