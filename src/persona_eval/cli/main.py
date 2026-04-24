"""Entry point: parse args, dispatch to the selected command."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

from persona_eval.cli import (
    compute_metrics,
    correlate,
    fetch_sources,
    list_metrics,
    robustness,
    robustness_eval,
    robustness_generate,
    run_all,
)

_COMMANDS = (
    list_metrics,
    fetch_sources,
    compute_metrics,
    correlate,
    robustness,
    robustness_generate,
    robustness_eval,
    run_all,
)


def main():
    parser = argparse.ArgumentParser(
        prog="persona-eval",
        description="Evaluate summarization metrics against human preferences",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for cmd in _COMMANDS:
        cmd.register(subparsers)

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.func(args)


if __name__ == "__main__":
    main()
