"""`persona-eval robustness-eval` — score metrics on saved perturbations."""

from __future__ import annotations

from pathlib import Path

from persona_eval.cli._args import add_llm_args, add_metric_args
from persona_eval.cli._common import (
    build_metric_cache,
    create_response_logger,
    metric_kwargs_from_args,
    resolve_metrics,
    save_run_params,
)


def register(subparsers):
    sp = subparsers.add_parser(
        "robustness-eval",
        help="Evaluate metrics on previously generated perturbations",
    )
    sp.add_argument(
        "--perturbations-dir", required=True,
        help="Path to saved perturbations directory (from robustness-generate)",
    )
    sp.add_argument("--cache-dir", default="cache", help="Cache directory")
    add_metric_args(sp)
    add_llm_args(sp)
    sp.add_argument("--output-dir", default="robustness_results", help="Output directory")
    sp.set_defaults(func=run)


def run(args):
    from persona_eval.robustness import (
        ALL_TESTS,
        analyze_robustness,
        load_perturbations,
        print_robustness_report,
        score_perturbations,
    )

    output_dir = Path(args.output_dir) / f"run_{args.run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_params(output_dir / "run_params.json", args)

    all_results, samples = load_perturbations(args.perturbations_dir)

    # Reconstruct test objects — only ``expected_direction`` is needed from
    # each here, so LLM-backed tests can be built with ``None`` dependencies.
    test_names_in_results = sorted(set(r.test_name for r in all_results))
    distractor_pool = [s.source for s in samples]
    tests = []
    for tn in test_names_in_results:
        if tn not in ALL_TESTS:
            print(f"Warning: unknown test '{tn}' in saved perturbations, skipping analysis for it")
            continue
        if tn == "distractor_sentences":
            tests.append(ALL_TESTS[tn](distractor_pool=distractor_pool))
        elif tn == "incremental_addition":
            tests.append(ALL_TESTS[tn]())
        elif tn in ("lengthen_prose", "shorten_prose", "different_audience"):
            tests.append(ALL_TESTS[tn](llm_client=None, cache=None))

    metric_names = resolve_metrics(args.metrics, default=["rouge"])
    metric_kwargs = metric_kwargs_from_args(args)

    response_logger = create_response_logger(output_dir / "scores.csv")
    metric_cache = build_metric_cache(args)

    scores_df = score_perturbations(
        all_results=all_results, samples=samples,
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
