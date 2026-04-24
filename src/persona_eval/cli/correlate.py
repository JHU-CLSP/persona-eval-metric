"""`persona-eval correlate` — compute metric-human correlations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from persona_eval.annotations import load_annotations
from persona_eval.cli._args import add_correlation_args
from persona_eval.cli._common import run_correlations, save_run_params


def register(subparsers):
    sp = subparsers.add_parser("correlate", help="Compute metric-human correlations")
    sp.add_argument("annotations", help="Path to annotations zip or directory")
    sp.add_argument("--scores", required=True, help="Path to metric scores CSV")
    sp.add_argument(
        "--pairwise-prefs", default=None,
        help="Path to raw pairwise preferences CSV (from pairwise metrics)",
    )
    add_correlation_args(sp)
    sp.add_argument("--output-dir", default="results", help="Output directory")
    sp.set_defaults(func=run)


def run(args):
    _, entries = load_annotations(args.annotations)
    scores_df = pd.read_csv(args.scores)

    pairwise_prefs_df = None
    if getattr(args, "pairwise_prefs", None):
        pairwise_prefs_df = pd.read_csv(args.pairwise_prefs)

    output_dir = Path(args.output_dir) / f"run_{args.run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_params(output_dir / "run_params.json", args)

    agreement, agg, include_neither, neither_config, strict = run_correlations(
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
