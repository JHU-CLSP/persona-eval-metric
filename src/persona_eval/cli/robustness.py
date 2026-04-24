"""`persona-eval robustness` — run robustness tests on metrics."""

from __future__ import annotations

from pathlib import Path

from persona_eval.cli._args import (
    add_llm_args,
    add_metric_args,
    add_perturb_llm_args,
)
from persona_eval.cli._common import (
    build_metric_cache,
    build_perturbation_tests,
    create_response_logger,
    load_samples,
    metric_kwargs_from_args,
    resolve_metrics,
    save_run_params,
)


def register(subparsers):
    sp = subparsers.add_parser("robustness", help="Run robustness tests on metrics")
    sp.add_argument(
        "annotations", nargs="?", default=None,
        help="Path to annotations zip or directory (not needed with --dataset)",
    )
    sp.add_argument("--cache-dir", default="cache", help="Cache directory")
    sp.add_argument("--email", help="Email for OpenAlex polite pool")
    add_metric_args(sp)
    add_llm_args(sp)
    add_perturb_llm_args(sp)
    sp.add_argument("--output-dir", default="robustness_results", help="Output directory")
    sp.add_argument(
        "--dataset",
        help="Load a standard dataset instead of annotations. "
             "Available: arxiv, scitldr, pubmed, elife, plos, mup. "
             "Stubs (not yet loadable): cdsr, eureka, cells, scinews, longsumm.",
    )
    sp.add_argument(
        "--split", default=None,
        help="Dataset split to use (default: test, or dataset-specific default)",
    )
    sp.add_argument(
        "--tests", nargs="+",
        help="Robustness tests to run (default: all). "
             "Choices: distractor_sentences, incremental_addition, lengthen_prose, "
             "shorten_prose, different_audience",
    )
    sp.add_argument("--num-samples", type=int, default=None,
                    help="Subsample N samples to limit compute")
    sp.add_argument("--seed", type=int, default=42, help="Random seed")
    sp.add_argument("--target-audiences", nargs="+", default=None,
                    help="Target audiences for the audience rewrite test")
    sp.set_defaults(func=run)


def run(args):
    from persona_eval.robustness import (
        ALL_TESTS,
        analyze_robustness,
        print_robustness_report,
        run_robustness,
    )

    output_dir = Path(args.output_dir) / f"run_{args.run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_params(output_dir / "run_params.json", args)

    samples = load_samples(args)
    if samples is None:
        return

    test_names = args.tests or list(ALL_TESTS.keys())
    tests = build_perturbation_tests(args, samples, test_names)

    metric_names = resolve_metrics(args.metrics, default=["rouge"])
    metric_kwargs = metric_kwargs_from_args(args)

    response_logger = create_response_logger(output_dir / "scores.csv")
    metric_cache = build_metric_cache(args)

    scores_df = run_robustness(
        samples=samples, tests=tests,
        metric_names=metric_names, metric_kwargs=metric_kwargs,
        output_dir=output_dir,
        response_logger=response_logger,
        cache=metric_cache,
    )

    if response_logger is not None:
        response_logger.close()

    analysis_df = analyze_robustness(scores_df, tests)
    analysis_path = output_dir / "robustness_analysis.csv"
    analysis_df.to_csv(analysis_path, index=False)
    print(f"\nSaved analysis to {analysis_path}")
    print_robustness_report(analysis_df)
