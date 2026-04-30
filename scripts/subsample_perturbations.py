#!/usr/bin/env python3
"""Subsample a perturbations directory.

Reads ``samples.jsonl`` and ``perturbations.jsonl`` from a perturbations
directory, randomly selects ``--n`` original samples, and writes those
samples plus all of their perturbation rows to a new directory in the
same format.

Usage:
    python scripts/subsample_perturbations.py \\
        --perturbations-dir robustness_results/run_<ts>/perturbations \\
        --output-dir robustness_results/run_<ts>/perturbations_subset \\
        --n 50 [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--perturbations-dir",
        type=Path,
        required=True,
        help="Directory containing samples.jsonl and perturbations.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the subsampled samples.jsonl and perturbations.jsonl.",
    )
    parser.add_argument("--n", type=int, required=True, help="Number of original samples to keep.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    args = parser.parse_args()

    samples_path = args.perturbations_dir / "samples.jsonl"
    perturbations_path = args.perturbations_dir / "perturbations.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(samples_path)
    if not perturbations_path.exists():
        raise FileNotFoundError(perturbations_path)

    samples = [json.loads(line) for line in samples_path.read_text().splitlines() if line.strip()]
    if args.n > len(samples):
        raise ValueError(f"--n={args.n} exceeds available samples ({len(samples)})")

    rng = random.Random(args.seed)
    chosen = rng.sample(samples, args.n)
    chosen_ids = {s["sample_id"] for s in chosen}

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output_dir / "samples.jsonl", "w") as f:
        for s in chosen:
            f.write(json.dumps(s) + "\n")

    kept = 0
    with open(args.output_dir / "perturbations.jsonl", "w") as f_out, open(perturbations_path) as f_in:
        for line in f_in:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj["sample_id"] in chosen_ids:
                f_out.write(json.dumps(obj) + "\n")
                kept += 1

    print(
        f"Wrote {len(chosen)} samples and {kept} perturbation results to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
