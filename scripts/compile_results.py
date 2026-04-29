#!/usr/bin/env python3
"""Compile results from multiple persona-eval runs into combined long-format CSVs.

Walks ``--eval-dir`` (default ``results/``) and/or ``--robustness-dir``
(default ``robustness_results/``), discovers ``run_<timestamp>/`` subdirs,
reads each run's output CSVs along with its ``run_params.json``, tags every
row with the run config (dataset, persona flag, LLM model, prompt file, ...)
and concatenates the rows. When the same logical result appears in multiple
runs, the row from the most recent run wins.

"Same logical result" means identical dedup keys, which always include the
run's config plus per-CSV identity columns:

* metric_scores.csv .................... query_index, label, annotator_id, metric
* pairwise_agreement.csv ............... metric
* rank_correlation_aggregate.csv ....... metric
* rank_correlation_per_query.csv ....... query_index, metric
* metric_scores_pairwise_prefs.csv ..... query_index, summary_a, summary_b,
                                         annotator_id, metric (when present)
* robustness_scores.csv ................ sample_id, test_name, level, metric
* robustness_analysis.csv .............. test_name, metric

Wide tables with multiple metric columns are melted to long format so each
row has a single ``metric`` + ``score`` pair. Outputs are written under
``--output-dir`` (default ``compiled_results/``) using the source filename.

Usage:
    python scripts/compile_results.py
    python scripts/compile_results.py --mode eval --eval-dir results
    python scripts/compile_results.py --mode robustness --output-dir merged/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

# Config fields lifted from run_params.json into every output row. These also
# participate in dedup so runs with different configs are kept side-by-side.
CONFIG_FIELDS = (
    "dataset",            # robustness HF dataset name (None for eval)
    "split",              # robustness split
    "annotations",        # eval annotations path (None for HF runs)
    "persona",
    "include_query",
    "llm_model",
    "llm_provider",
    "llm_prompt_file",
    "neither_config",
    "include_neither",
    "strict_pairwise",
    "num_samples",
    "seed",
)


@dataclass(frozen=True)
class Run:
    run_id: str           # e.g. "20260429_154212"
    kind: str             # "eval" or "robustness"
    path: Path            # the run_<id> directory
    params: dict          # parsed run_params.json (may be {} if missing)

    def config(self) -> dict:
        """Subset of params persisted alongside every row in the compiled CSVs."""
        return {f: self.params.get(f) for f in CONFIG_FIELDS}

    def dataset_key(self) -> str:
        """Stable identifier for the data source (HF dataset name or annotations path)."""
        return self.params.get("dataset") or self.params.get("annotations") or ""


def discover_runs(root: Path, kind: str) -> list[Run]:
    if not root.exists():
        return []
    runs: list[Run] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("run_"):
            continue
        run_id = sub.name[len("run_"):]
        params_path = sub / "run_params.json"
        params: dict = {}
        if params_path.exists():
            try:
                params = json.loads(params_path.read_text())
            except json.JSONDecodeError as e:
                print(f"  Warning: could not parse {params_path}: {e}")
        runs.append(Run(run_id=run_id, kind=kind, path=sub, params=params))
    return runs


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"  Warning: could not read {path}: {e}")
        return None


def _melt_metric_columns(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    """Melt a wide metric-scored DataFrame to long form (metric, score)."""
    id_cols = [c for c in id_cols if c in df.columns]
    metric_cols = [c for c in df.columns if c not in id_cols]
    if not metric_cols:
        return df.iloc[0:0].assign(metric=pd.Series(dtype=str), score=pd.Series(dtype=float))
    long = df.melt(id_vars=id_cols, value_vars=metric_cols,
                   var_name="metric", value_name="score")
    return long.dropna(subset=["score"])


def _attach_run(df: pd.DataFrame, run: Run) -> pd.DataFrame:
    """Prepend run_id + config columns to ``df``."""
    out = df.copy()
    out.insert(0, "run_id", run.run_id)
    cfg = run.config()
    for i, key in enumerate(CONFIG_FIELDS, start=1):
        out.insert(i, key, cfg.get(key))
    return out


# Per-CSV spec: filename -> (id_cols_for_melt_or_None, dedup_keys)
# When the first element is None, the file is already long form (one row per
# metric) and we only attach run/config columns. Otherwise we melt the metric
# columns into rows.
EVAL_SPECS = {
    "metric_scores.csv": (
        ["query_index", "label", "annotator_id"],
        ["query_index", "label", "annotator_id", "metric"],
    ),
    "pairwise_agreement.csv": (
        None,
        ["metric"],
    ),
    "rank_correlation_aggregate.csv": (
        None,
        ["metric"],
    ),
    "rank_correlation_per_query.csv": (
        None,
        ["query_index", "metric"],
    ),
    "metric_scores_pairwise_prefs.csv": (
        None,
        # The full preference identity; columns may vary across versions, so we
        # intersect with what's actually present at dedup time.
        ["query_index", "summary_a", "summary_b", "annotator_id", "metric"],
    ),
}

ROBUSTNESS_SPECS = {
    "robustness_scores.csv": (
        ["sample_id", "test_name", "level", "level_label"],
        ["sample_id", "test_name", "level", "metric"],
    ),
    "robustness_analysis.csv": (
        None,
        ["test_name", "metric"],
    ),
}


# ---------------------------------------------------------------------------
# Compile per-spec
# ---------------------------------------------------------------------------

def _compile_spec(
    runs: list[Run],
    filename: str,
    id_cols: list[str] | None,
    dedup_keys: list[str],
) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    for run in runs:
        df = _read_csv(run.path / filename)
        if df is None or df.empty:
            continue
        if id_cols is not None:
            df = _melt_metric_columns(df, id_cols)
            if df.empty:
                continue
        frames.append(_attach_run(df, run))

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True, sort=False)

    # Dedup: most-recent run wins. ``run_id`` is a sortable timestamp string.
    full_keys = list(CONFIG_FIELDS) + [k for k in dedup_keys if k in combined.columns]
    # Some keys may legitimately be NaN (e.g. annotations vs dataset); fill so
    # rows with the same NaN values group together rather than being treated
    # as distinct by drop_duplicates.
    sentinel = "\x00__NA__\x00"
    keyed = combined.assign(**{k: combined[k].fillna(sentinel) for k in full_keys
                               if combined[k].dtype == object or combined[k].isna().any()})
    keyed = keyed.sort_values("run_id", kind="mergesort")  # stable ascending
    deduped_idx = keyed.drop_duplicates(subset=full_keys, keep="last").index
    return combined.loc[sorted(deduped_idx)].reset_index(drop=True)


def _write(df: pd.DataFrame | None, out_path: Path, label: str) -> None:
    if df is None or df.empty:
        print(f"  {label}: no rows found, skipping")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  {label}: wrote {len(df):>6} rows -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compile persona-eval run outputs (eval + robustness) "
                    "into combined long-format CSVs, keeping the most recent "
                    "run for any duplicate (config, dataset, metric, ...) row.",
    )
    parser.add_argument("--eval-dir", default="results",
                        help="Directory containing run_<id>/ from run-all/correlate "
                             "(default: results/)")
    parser.add_argument("--robustness-dir", default="robustness_results",
                        help="Directory containing run_<id>/ from robustness/"
                             "robustness-eval (default: robustness_results/)")
    parser.add_argument("--output-dir", default="compiled_results",
                        help="Where to write compiled CSVs (default: compiled_results/)")
    parser.add_argument("--mode", choices=("eval", "robustness", "both"),
                        default="both",
                        help="Which pipeline's results to compile (default: both)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    eval_runs: list[Run] = []
    rob_runs: list[Run] = []
    if args.mode in ("eval", "both"):
        eval_runs = discover_runs(Path(args.eval_dir), "eval")
        print(f"Discovered {len(eval_runs)} eval run(s) under {args.eval_dir}/")
    if args.mode in ("robustness", "both"):
        rob_runs = discover_runs(Path(args.robustness_dir), "robustness")
        print(f"Discovered {len(rob_runs)} robustness run(s) under {args.robustness_dir}/")

    if not eval_runs and not rob_runs:
        print("No runs found. Nothing to compile.")
        return

    if eval_runs:
        print("\nCompiling eval results:")
        for filename, (id_cols, dedup_keys) in EVAL_SPECS.items():
            df = _compile_spec(eval_runs, filename, id_cols, dedup_keys)
            _write(df, output_dir / filename, filename)

    if rob_runs:
        print("\nCompiling robustness results:")
        for filename, (id_cols, dedup_keys) in ROBUSTNESS_SPECS.items():
            df = _compile_spec(rob_runs, filename, id_cols, dedup_keys)
            _write(df, output_dir / filename, filename)

    print(f"\nDone. Compiled outputs in {output_dir}/")


if __name__ == "__main__":
    main()
