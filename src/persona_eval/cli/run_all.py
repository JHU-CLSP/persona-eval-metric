"""`persona-eval run-all` — full pipeline: fetch, compute, correlate."""

from __future__ import annotations

from pathlib import Path

from persona_eval.cli._args import (
    add_correlation_args,
    add_data_args,
    add_llm_args,
    add_metric_args,
)
from persona_eval.cli._common import (
    compute_and_save,
    load_data,
    resolve_metrics,
    run_correlations,
    save_run_params,
)


def register(subparsers):
    sp = subparsers.add_parser("run-all", help="Run full pipeline")
    add_data_args(sp)
    add_metric_args(sp)
    add_correlation_args(sp)
    sp.add_argument("--output-dir", default="results", help="Output directory")
    add_llm_args(sp)
    sp.set_defaults(func=run)


def run(args):
    output_dir = Path(args.output_dir) / f"run_{args.run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_params(output_dir / "run_params.json", args)

    profiles_by_id, entries, source_texts, reference_texts = load_data(args)
    print(f"Loaded {len(entries)} annotations from {len(profiles_by_id)} annotators")
    print(f"Fetched texts for {len(source_texts)} unique queries")

    metric_names = resolve_metrics(args.metrics)
    scores_df, pairwise_prefs_df = compute_and_save(
        entries, source_texts, reference_texts, profiles_by_id,
        metric_names, args, output_dir / "metric_scores.csv",
    )

    agreement, agg, _, _, _ = run_correlations(
        entries, scores_df, pairwise_prefs_df, args, output_dir,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("\n--- Pairwise Agreement ---")
    print(agreement.to_string(index=False))
    print("\n--- Aggregate Rank Correlation ---")
    print(agg.to_string(index=False))
    print(f"\nAll outputs saved to {output_dir}/")
