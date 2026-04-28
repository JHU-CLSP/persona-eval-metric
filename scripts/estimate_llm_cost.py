#!/usr/bin/env python3
"""Dry-run the LLM-based metrics pipeline to estimate Together AI cost.

Walks the same code paths as ``persona-eval compute-metrics`` but
intercepts every ``LLMClient.generate`` call: instead of hitting the
API, it tokenizes the would-be prompt, returns a synthetic response
that satisfies every parser in the metric layer, and accumulates
per-metric token counts. Multiplies by per-million-token pricing for a
final cost estimate.

Notes & caveats:
  * Token counts are approximate. We use the ``cl100k_base`` tokenizer
    via ``tiktoken`` if available (close enough for Llama / Qwen /
    DeepSeek for budgeting), else fall back to a 1.3 * word-count
    heuristic.
  * Output tokens are estimated. Calls that pin ``max_tokens`` (e.g.
    factscore verify, persona-recall coverage) are counted at the cap.
    Other calls use ``--default-output-tokens``.
  * The on-disk metric cache is bypassed (otherwise prior real runs
    would short-circuit prompts and undercount).
  * Source/reference texts come from the OpenAlex cache; if abstracts
    aren't cached locally yet, the script will fetch them once.

Usage:
    python scripts/estimate_llm_cost.py annotations.zip \\
        --llm-model meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo \\
        --metrics all-llm
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Make src/ importable when running from the repo root.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from persona_eval import llm_client as llm_client_mod
from persona_eval.cli._common import (
    collect_llm_kwargs,
    load_data,
    resolve_metrics,
)
from persona_eval.core.pipeline import compute_metric_scores
from persona_eval.metrics import list_llm_metrics


# Together AI pricing (USD per 1M tokens) — current as of 2026-04.
# Format: model_name -> (input_price, output_price). Verify before
# trusting; pass --input-price/--output-price to override.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": (0.18, 0.18),
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": (0.88, 0.88),
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": (3.50, 3.50),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (0.88, 0.88),
    "Qwen/Qwen2.5-72B-Instruct-Turbo": (1.20, 1.20),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": (0.30, 0.30),
    "deepseek-ai/DeepSeek-V3": (1.25, 1.25),
    "deepseek-ai/DeepSeek-V4-Pro": (2.10, 4.40),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": (0.60, 0.60),
    "moonshotai/Kimi-K2.6": (1.20, 4.50),
}


# A response string that satisfies every parser used by the LLM metrics:
#   - parse_bullet_list -> three "- ..." lines
#   - SCORE_RE          -> "[RESULT] 3"
#   - PAIRWISE_RE       -> "[RESULT] A"
#   - _classify_positive("supported"/"relevant"/"covered") -> True
SYNTHETIC_RESPONSE = (
    "- nugget alpha\n"
    "- nugget beta\n"
    "- nugget gamma\n"
    "\n"
    "[RESULT] A\n"
    "[RESULT] 3\n"
    "\n"
    "The claim is supported by the source. "
    "The nugget is relevant to the persona. "
    "The requirement is covered by the nuggets.\n"
)


def _make_token_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s))
    except Exception:
        print("warning: tiktoken not available; using word-count heuristic", file=sys.stderr)
        return lambda s: int(len(s.split()) * 1.3)


def main():
    p = argparse.ArgumentParser(
        description="Dry-run cost estimation for LLM metrics on Together AI",
    )
    p.add_argument("annotations", help="Path to annotations zip or directory")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--email", help="Email for OpenAlex polite pool")
    p.add_argument(
        "--metrics", nargs="+", default=["all-llm"],
        help="Metrics or groups (default: all-llm). Non-LLM metrics are filtered out.",
    )
    p.add_argument("--device", default="cpu")

    # LLM config (mirrors add_llm_args, minus --no-cache controls).
    p.add_argument("--llm-provider", default="together", choices=["vllm", "together"])
    p.add_argument("--llm-model", required=True)
    p.add_argument("--llm-api-key", default="dry-run-fake-key")
    p.add_argument("--llm-base-url")
    p.add_argument("--llm-prompt-file")
    p.add_argument("--persona", action="store_true")
    p.add_argument("--include-query", action="store_true")

    # Pricing.
    p.add_argument("--input-price", type=float,
                   help="USD per 1M input tokens (default: from DEFAULT_PRICING)")
    p.add_argument("--output-price", type=float,
                   help="USD per 1M output tokens (default: from DEFAULT_PRICING)")
    p.add_argument(
        "--default-output-tokens", type=int, default=200,
        help="Estimated output tokens for calls without an explicit max_tokens cap.",
    )

    # Sampling / output.
    p.add_argument("--num-samples", type=int,
                   help="Limit to first N annotation entries (faster dry run)")
    p.add_argument("--output", default="cost_estimate.json")

    args = p.parse_args()

    # Resolve pricing.
    if args.input_price is None or args.output_price is None:
        if args.llm_model not in DEFAULT_PRICING:
            sys.exit(
                f"No default pricing for '{args.llm_model}'. "
                "Pass --input-price and --output-price (USD per 1M tokens)."
            )
        d_in, d_out = DEFAULT_PRICING[args.llm_model]
        if args.input_price is None:
            args.input_price = d_in
        if args.output_price is None:
            args.output_price = d_out

    count_tokens = _make_token_counter()

    state = {"current_metric": "(unknown)"}
    counters: dict[str, dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0},
    )
    synthetic_out_tokens = count_tokens(SYNTHETIC_RESPONSE)

    # Patch LLMClient: bypass the api-key check and short-circuit generate().
    real_init = llm_client_mod.LLMClient.__init__

    def patched_init(self, *a, **kw):
        if not kw.get("api_key"):
            kw["api_key"] = "dry-run-fake-key"
        real_init(self, *a, **kw)
        # Never actually instantiate a network client.
        self._client = object()

    def patched_generate(self, prompt, system_prompt=None, temperature=None, max_tokens=None):
        full = (system_prompt or "") + prompt
        in_tok = count_tokens(full)
        cap = max_tokens if max_tokens is not None else self.max_tokens
        if cap is not None and cap <= 32:
            est_out = cap
        else:
            est_out = min(args.default_output_tokens, cap or args.default_output_tokens)
        # Use the synthetic response, but truncate token count if cap is tiny
        # so factscore verify / coverage checks are charged correctly.
        m = state["current_metric"]
        c = counters[m]
        c["calls"] += 1
        c["input_tokens"] += in_tok
        c["output_tokens"] += min(est_out, synthetic_out_tokens) if cap and cap <= 32 else est_out
        return SYNTHETIC_RESPONSE

    llm_client_mod.LLMClient.__init__ = patched_init
    llm_client_mod.LLMClient.generate = patched_generate

    # Load data.
    profiles_by_id, entries, source_texts, reference_texts = load_data(args)
    if args.num_samples and args.num_samples < len(entries):
        entries = entries[: args.num_samples]
    print(f"Loaded {len(entries)} annotation entries")

    # Resolve & filter metrics.
    metric_names = resolve_metrics(args.metrics)
    llm_only = set(list_llm_metrics())
    selected = [m for m in metric_names if m in llm_only]
    skipped = [m for m in metric_names if m not in llm_only]
    if skipped:
        print(f"Skipping non-LLM metrics: {', '.join(skipped)}")
    if not selected:
        sys.exit("No LLM metrics selected.")
    print(f"Estimating cost for: {', '.join(selected)}")

    llm_kwargs = collect_llm_kwargs(args)

    # Run each metric separately so we can attribute calls. cache=None
    # forces every call through the patched generate().
    for mn in selected:
        state["current_metric"] = mn
        try:
            compute_metric_scores(
                entries, source_texts, reference_texts,
                [mn], device=args.device,
                llm_kwargs=llm_kwargs, profiles_by_id=profiles_by_id,
                cache=None,
            )
        except Exception as e:
            print(f"  warning: metric '{mn}' raised {type(e).__name__}: {e}")

    # Report.
    rows = []
    for mn, c in counters.items():
        cost = (
            c["input_tokens"] * args.input_price / 1e6
            + c["output_tokens"] * args.output_price / 1e6
        )
        rows.append((mn, c["calls"], c["input_tokens"], c["output_tokens"], cost))
    rows.sort(key=lambda r: -r[4])

    print()
    print(f"Model: {args.llm_model}")
    print(
        f"Pricing (USD per 1M tokens): "
        f"input ${args.input_price:.3f}, output ${args.output_price:.3f}"
    )
    print(f"Default output tokens (uncapped calls): {args.default_output_tokens}")
    print()
    print(f"{'Metric':<32} {'Calls':>8} {'In tokens':>14} {'Out tokens':>12} {'Cost USD':>10}")
    print("-" * 80)
    tc = ti = to = 0
    for mn, calls, _ti, _to, cost in rows:
        print(f"{mn:<32} {calls:>8,} {_ti:>14,} {_to:>12,} ${cost:>9.4f}")
        tc += calls; ti += _ti; to += _to
    total_cost = ti * args.input_price / 1e6 + to * args.output_price / 1e6
    print("-" * 80)
    print(f"{'TOTAL':<32} {tc:>8,} {ti:>14,} {to:>12,} ${total_cost:>9.4f}")

    out_path = Path(args.output)
    out_path.write_text(json.dumps({
        "model": args.llm_model,
        "provider": args.llm_provider,
        "input_price_per_1m": args.input_price,
        "output_price_per_1m": args.output_price,
        "default_output_tokens": args.default_output_tokens,
        "num_entries": len(entries),
        "metrics": [
            {
                "metric": mn,
                "calls": calls,
                "input_tokens": _ti,
                "output_tokens": _to,
                "cost_usd": round(cost, 6),
            }
            for mn, calls, _ti, _to, cost in rows
        ],
        "total": {
            "calls": tc,
            "input_tokens": ti,
            "output_tokens": to,
            "cost_usd": round(total_cost, 6),
        },
    }, indent=2))
    print(f"\nSaved breakdown to {out_path}")


if __name__ == "__main__":
    main()
