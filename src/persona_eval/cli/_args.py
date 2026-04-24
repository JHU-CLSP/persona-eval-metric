"""Reusable argparse helpers shared across CLI commands."""

from __future__ import annotations


def add_data_args(parser):
    parser.add_argument("annotations", help="Path to annotations zip or directory")
    parser.add_argument("--cache-dir", default="cache", help="Cache directory")
    parser.add_argument("--email", help="Email for OpenAlex polite pool")


def add_metric_args(parser):
    parser.add_argument(
        "--metrics", nargs="+",
        help="Metrics to compute (default: all). Groups: 'all', 'non-llm', 'all-llm'.",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for model inference (cpu, cuda, cuda:0, etc.)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable the metric result cache (cache is on by default).",
    )
    parser.add_argument(
        "--clear-metric-cache", action="store_true",
        help="Delete all entries from the metric cache before running.",
    )


def add_llm_args(parser):
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


def add_perturb_llm_args(parser):
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


def add_correlation_args(parser):
    parser.add_argument(
        "--include-neither", action="store_true",
        help="Include 'neither' annotations in pairwise agreement",
    )
    parser.add_argument(
        "--neither-config", default=None,
        help="Path to YAML config with per-metric thresholds for 'neither' agreement",
    )
    parser.add_argument(
        "--strict-pairwise", action="store_true",
        help="Strict mode: auto-disagree on final when metric got round1 wrong",
    )
