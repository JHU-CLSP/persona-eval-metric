"""`persona-eval compute-metrics` — compute automatic metrics for all summaries."""

from __future__ import annotations

from pathlib import Path

from persona_eval.cli._args import add_data_args, add_llm_args, add_metric_args
from persona_eval.cli._common import (
    compute_and_save,
    load_data,
    resolve_metrics,
    save_run_params,
)


def register(subparsers):
    sp = subparsers.add_parser("compute-metrics", help="Compute automatic metrics")
    add_data_args(sp)
    add_metric_args(sp)
    sp.add_argument("--output", default="metric_scores.csv", help="Output CSV path")
    add_llm_args(sp)
    sp.set_defaults(func=run)


def run(args):
    profiles_by_id, entries, source_texts, reference_texts = load_data(args)
    metric_names = resolve_metrics(args.metrics)
    output = Path(args.output)
    output = output.parent / f"{output.stem}_run_{args.run_id}{output.suffix}"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_run_params(output.parent / f"{output.stem}_params.json", args)
    compute_and_save(
        entries, source_texts, reference_texts, profiles_by_id,
        metric_names, args, output,
    )
