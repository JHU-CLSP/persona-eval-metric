"""Shared helpers used by multiple CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from persona_eval.annotations import load_annotations
from persona_eval.correlation import (
    aggregate_correlations,
    compute_pairwise_agreement,
    compute_rank_correlation,
)
from persona_eval.annotations import get_pairwise_preferences
from persona_eval.core.pipeline import compute_metric_scores
from persona_eval.metrics import list_llm_metrics, list_metrics, list_non_llm_metrics
from persona_eval.metrics.cache import MetricCache
from persona_eval.openalex import OpenAlexClient


def resolve_metrics(metrics: list[str] | None, default: list[str] | None = None) -> list[str]:
    """Expand metric group aliases (``all``, ``non-llm``, ``all-llm``) and dedupe."""
    if not metrics:
        return default if default is not None else list_metrics()

    group_map = {
        "all": list_metrics,
        "non-llm": list_non_llm_metrics,
        "all-llm": list_llm_metrics,
    }

    resolved: list[str] = []
    for m in metrics:
        if m in group_map:
            resolved.extend(group_map[m]())
        else:
            resolved.append(m)

    seen: set[str] = set()
    deduped: list[str] = []
    for m in resolved:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped


def load_data(args):
    """Load annotations and fetch source/reference texts.

    Returns ``(profiles_by_id, entries, source_texts, reference_texts)``.
    """
    profiles, entries = load_annotations(args.annotations)
    profiles_by_id = {p.annotator_id: p for p in profiles}
    client = OpenAlexClient(cache_dir=args.cache_dir, email=args.email)
    source_texts, reference_texts = client.get_texts_batch(entries)
    return profiles_by_id, entries, source_texts, reference_texts


def load_neither_thresholds(path: str | None) -> dict[str, float] | None:
    if path is None:
        return None
    with open(path) as f:
        data = yaml.safe_load(f)
    return {str(k): float(v) for k, v in data.items()}


def create_response_logger(output_path: str | Path):
    """Create a ResponseLogger writing under ``<parent>/llm_responses/``."""
    from persona_eval.llm_client import ResponseLogger
    return ResponseLogger(Path(output_path).parent / "llm_responses")


def save_run_params(path: Path, args) -> None:
    """Write CLI arguments to a JSON file for reproducibility."""
    params = {k: v for k, v in vars(args).items() if k != "func"}
    path.write_text(json.dumps(params, indent=2, default=str))


def collect_llm_kwargs(args) -> dict:
    """Collect LLM-related kwargs from CLI args."""
    kwargs = {}
    for key in ("provider", "model", "api_key", "base_url", "prompt_file"):
        val = getattr(args, f"llm_{key}", None)
        if val is not None:
            kwargs[key] = val
    if getattr(args, "persona", False):
        kwargs["persona"] = True
    if getattr(args, "include_query", False):
        kwargs["include_query"] = True
    return kwargs


def build_metric_cache(args) -> MetricCache | None:
    cache_dir = getattr(args, "cache_dir", None)
    if cache_dir is None:
        return None
    enabled = not getattr(args, "no_cache", False)
    cache = MetricCache(cache_dir, enabled=enabled)
    if enabled and getattr(args, "clear_metric_cache", False):
        n = cache.clear()
        if n:
            print(f"Cleared {n} metric cache entries from {cache.cache_dir}")
    return cache


def compute_and_save(
    entries, source_texts, reference_texts, profiles_by_id,
    metric_names, args, output_path,
):
    """Compute metrics and save results incrementally.

    Returns ``(scores_df, pairwise_prefs_df)``.
    """
    llm_kwargs = collect_llm_kwargs(args)
    response_logger = create_response_logger(output_path)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    prefs_path = output.parent / (output.stem + "_pairwise_prefs.csv")

    existing_df = pd.read_csv(output) if output.exists() else None
    if existing_df is not None:
        print(f"Found existing {output} ({len(existing_df)} rows); will skip already-computed metrics")

    cache = build_metric_cache(args)

    def save_partial(scores_df, prefs_df):
        if scores_df is not None and not scores_df.empty:
            scores_df.to_csv(output, index=False)
        if prefs_df is not None:
            prefs_df.to_csv(prefs_path, index=False)

    scores_df, pairwise_prefs_df = compute_metric_scores(
        entries, source_texts, reference_texts,
        metric_names, device=args.device,
        llm_kwargs=llm_kwargs, profiles_by_id=profiles_by_id,
        response_logger=response_logger,
        cache=cache,
        existing_df=existing_df,
        on_metric_done=save_partial,
    )

    if not scores_df.empty:
        scores_df.to_csv(output, index=False)
    print(f"Saved metric scores to {output} ({len(scores_df)} rows)")

    if pairwise_prefs_df is not None:
        pairwise_prefs_df.to_csv(prefs_path, index=False)
        print(f"Saved raw pairwise preferences to {prefs_path}")

    if response_logger is not None:
        print(f"Saved LLM responses to {response_logger.path}")
        response_logger.close()

    return scores_df, pairwise_prefs_df


def run_correlations(entries, scores_df, pairwise_prefs_df, args, output_dir):
    """Compute and save correlation results.

    Returns ``(agreement, aggregate, include_neither, neither_config, strict)``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    include_neither = getattr(args, "include_neither", False)
    neither_config = getattr(args, "neither_config", None)
    neither_thresholds = load_neither_thresholds(neither_config) if include_neither else None
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

    return agreement, agg, include_neither, neither_config, strict


# ---------------------------------------------------------------------------
# Robustness-specific helpers (shared by robustness and robustness-generate)
# ---------------------------------------------------------------------------

_LLM_PERTURB_TESTS = {"lengthen_prose", "shorten_prose", "different_audience"}


def load_samples(args):
    """Load samples for robustness from either ``--dataset`` or annotations."""
    from persona_eval.robustness import load_from_persona_eval
    from persona_eval.robustness.dataset import load_from_huggingface

    dataset_name = getattr(args, "dataset", None)
    if dataset_name:
        samples = load_from_huggingface(
            dataset_name,
            split=getattr(args, "split", None),
            num_samples=args.num_samples,
            seed=args.seed,
        )
        print(f"Loaded {len(samples)} samples from {dataset_name}")
    elif getattr(args, "annotations", None):
        profiles_by_id, entries, source_texts, reference_texts = load_data(args)
        samples = load_from_persona_eval(
            entries, source_texts, reference_texts, profiles_by_id,
        )
        print(f"Loaded {len(samples)} samples")
    else:
        print("Error: provide either --dataset or annotations path")
        return None

    if args.num_samples and args.num_samples < len(samples):
        import random
        rng = random.Random(args.seed)
        samples = rng.sample(samples, args.num_samples)
        print(f"Subsampled to {len(samples)} samples (seed={args.seed})")
    return samples


def build_perturbation_tests(args, samples, test_names):
    """Construct the list of test objects, wiring up LLM client + cache if needed."""
    from persona_eval.robustness import ALL_TESTS, PerturbationCache

    llm_client = None
    cache = None
    if _LLM_PERTURB_TESTS & set(test_names):
        from persona_eval.llm_client import LLMClient

        llm_kwargs = collect_llm_kwargs(args)
        for k in ("include_query", "persona", "prompt_file"):
            llm_kwargs.pop(k, None)
        provider = getattr(args, "perturb_provider", None) or llm_kwargs.get("provider", "vllm")
        model = getattr(args, "perturb_model", None) or llm_kwargs.get("model")
        api_key = getattr(args, "perturb_api_key", None) or llm_kwargs.get("api_key")
        base_url = getattr(args, "perturb_base_url", None) or llm_kwargs.get("base_url")
        llm_client = LLMClient(provider=provider, model=model, api_key=api_key, base_url=base_url)
        cache = PerturbationCache(args.cache_dir)

    distractor_pool = [s.source for s in samples]
    tests = []
    for tn in test_names:
        if tn not in ALL_TESTS:
            print(f"Warning: unknown test '{tn}', skipping")
            continue
        if tn == "distractor_sentences":
            tests.append(ALL_TESTS[tn](distractor_pool=distractor_pool, seed=args.seed))
        elif tn == "incremental_addition":
            tests.append(ALL_TESTS[tn]())
        elif tn == "different_audience":
            tests.append(ALL_TESTS[tn](
                llm_client=llm_client, cache=cache,
                target_audiences=args.target_audiences,
            ))
        else:
            tests.append(ALL_TESTS[tn](llm_client=llm_client, cache=cache))
    return tests


def metric_kwargs_from_args(args) -> dict:
    """Build per-metric constructor kwargs from CLI args (used by robustness flows)."""
    kwargs = {}
    for key in ("provider", "model", "api_key", "base_url"):
        val = getattr(args, f"llm_{key}", None)
        if val is not None:
            kwargs[key] = val
    kwargs["device"] = args.device
    return kwargs
