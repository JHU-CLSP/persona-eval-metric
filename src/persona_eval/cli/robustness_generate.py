"""`persona-eval robustness-generate` — emit perturbations without metric evaluation."""

from __future__ import annotations

from pathlib import Path

from persona_eval.cli._args import add_llm_args, add_perturb_llm_args
from persona_eval.cli._common import (
    build_perturbation_tests,
    load_samples,
    save_run_params,
)


def register(subparsers):
    sp = subparsers.add_parser(
        "robustness-generate",
        help="Generate perturbations only (no metric evaluation)",
    )
    sp.add_argument(
        "annotations", nargs="?", default=None,
        help="Path to annotations zip or directory (not needed with --dataset)",
    )
    sp.add_argument("--cache-dir", default="cache", help="Cache directory")
    sp.add_argument("--email", help="Email for OpenAlex polite pool")
    add_llm_args(sp)
    add_perturb_llm_args(sp)
    sp.add_argument("--output-dir", default="robustness_results", help="Output directory")
    sp.add_argument(
        "--dataset",
        help="Load a standard dataset instead of annotations. "
             "Available: arxiv, scitldr, pubmed, elife, plos, mup, scinews.",
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
        generate_perturbations,
        save_perturbations,
    )

    output_dir = Path(args.output_dir) / f"run_{args.run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_params(output_dir / "run_params.json", args)

    samples = load_samples(args)
    if samples is None:
        return

    test_names = args.tests or list(ALL_TESTS.keys())
    tests = build_perturbation_tests(args, samples, test_names)

    all_results = generate_perturbations(samples, tests)
    perturb_dir = save_perturbations(all_results, samples, output_dir)
    print(f"\nPerturbations saved to {perturb_dir}")
    print("Run 'persona-eval robustness-eval' to evaluate metrics on these perturbations.")
