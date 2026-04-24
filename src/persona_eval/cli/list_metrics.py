"""`persona-eval list-metrics` — print registered metrics."""

from __future__ import annotations

from persona_eval.metrics import get_metric, list_metrics


def register(subparsers):
    sp = subparsers.add_parser("list-metrics", help="List available metrics")
    sp.set_defaults(func=run)


def run(args):
    for name in list_metrics():
        metric = get_metric(name)
        ref = "reference-free" if metric.is_reference_free else "reference-based"
        print(f"  {name:20s}  {metric.name} ({ref})")
