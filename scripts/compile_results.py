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
* perturbation_samples.csv ............. sample_id   (from <run>/perturbations/samples.jsonl)
* perturbations.csv .................... sample_id, test_name, level   (from
                                         <run>/perturbations/perturbations.jsonl,
                                         flattened so each level is one row;
                                         joined with the sample's untouched
                                         ``summary`` and ``source`` so each row
                                         shows original vs. perturbed text)

Wide tables with multiple metric columns are melted to long format so each
row has a single ``metric`` + ``score`` pair. Dict-valued metadata fields
in the perturbation JSONL files are JSON-encoded into a string column so
they round-trip through CSV. Outputs are written under ``--output-dir``
(default ``compiled_results/``) using the source filename.

In addition, ``samples.jsonl`` and ``perturbations.jsonl`` are written
alongside the CSVs in the format that ``persona-eval robustness-eval
--perturbations-dir <output_dir>`` reads, so the compiled directory can
be fed straight back into the pipeline.

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


def _load_params(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"  Warning: could not parse {path}: {e}")
        return {}


def _reconcile_with_samples(run_path: Path, params: dict) -> dict:
    """Prefer embedded sample metadata over run_params.json for dataset/split.

    HF-loaded samples carry ``metadata.dataset`` and ``metadata.split`` set
    at load time by ``load_from_huggingface``. When these disagree with the
    run's argparse-recorded params, the embedded values are closer to
    ground truth — they reflect what was actually loaded, while
    ``run_params.json`` only records what the CLI was invoked with.

    The most common cause of disagreement is a re-used run directory (an
    earlier generate run with a different ``--dataset`` wrote samples,
    then a later run with a new ``--dataset`` overwrote ``run_params.json``
    but never re-wrote the perturbations). When we detect this, we emit
    a one-line warning per run and use the embedded values for the
    compiled output.
    """
    samples_path = run_path / "perturbations" / "samples.jsonl"
    if not samples_path.exists():
        return params

    try:
        with open(samples_path) as f:
            first_line = f.readline().strip()
    except OSError:
        return params
    if not first_line:
        return params
    try:
        first = json.loads(first_line)
    except json.JSONDecodeError:
        return params

    embedded = first.get("metadata") or {}
    merged = dict(params)
    mismatches: list[str] = []
    for field in ("dataset", "split"):
        ev = embedded.get(field)
        pv = merged.get(field)
        if ev and pv and ev != pv:
            mismatches.append(f"{field}: params={pv!r} -> samples={ev!r}")
            merged[field] = ev
        elif ev and not pv:
            merged[field] = ev

    if mismatches:
        print(f"  Warning: {run_path.name} run_params.json disagrees with "
              f"samples.jsonl ({'; '.join(mismatches)}); using embedded values")
    return merged


def _inherit_from_perturbations(params: dict) -> dict:
    """For robustness-eval runs, fill missing CONFIG_FIELDS from the upstream
    robustness-generate run.

    ``robustness-generate`` writes ``run_params.json`` at ``<run>/`` and
    perturbations at ``<run>/perturbations/``. ``robustness-eval`` then
    receives ``--perturbations-dir <run>/perturbations`` but its own
    ``run_params.json`` lacks ``dataset`` / ``annotations`` / etc.
    Walk back up to ``<run>/run_params.json`` and inherit any missing
    fields without overwriting eval-specific ones (e.g. ``llm_model``).
    """
    pdir = params.get("perturbations_dir")
    if not pdir:
        return params
    upstream = Path(pdir).parent / "run_params.json"
    if not upstream.exists():
        return params
    upstream_params = _load_params(upstream)
    if not upstream_params:
        return params
    merged = dict(params)
    for field in CONFIG_FIELDS:
        if merged.get(field) in (None, "") and upstream_params.get(field) not in (None, ""):
            merged[field] = upstream_params[field]
    return merged


def discover_runs(root: Path, kind: str) -> list[Run]:
    if not root.exists():
        return []
    runs: list[Run] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("run_"):
            continue
        run_id = sub.name[len("run_"):]
        params = _load_params(sub / "run_params.json")
        params = _inherit_from_perturbations(params)
        params = _reconcile_with_samples(sub, params)
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
# Perturbation JSONL compilation
# ---------------------------------------------------------------------------
#
# robustness-generate writes <run>/perturbations/{samples,perturbations}.jsonl.
# We compile both into long-format CSVs alongside the metric outputs, with the
# same run-tagging + dedup model: most recent run wins for any
# (config, sample_id) or (config, sample_id, test_name, level) collision.

def _read_jsonl(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    rows: list[dict] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as e:
        print(f"  Warning: could not read {path}: {e}")
        return None
    return rows


def _flatten_metadata(row: dict, key: str = "metadata") -> dict:
    """Replace a dict-valued field with a JSON string so it round-trips through CSV."""
    out = dict(row)
    val = out.get(key)
    if isinstance(val, (dict, list)):
        out[key] = json.dumps(val, ensure_ascii=False)
    return out


def _compile_perturbations_jsonl(runs: list[Run]) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Compile samples.jsonl and perturbations.jsonl across runs.

    Returns ``(samples_df, perturbations_df)`` — each may be ``None`` if no
    run carried that file. Dedup keys: samples = (config, sample_id);
    perturbations = (config, sample_id, test_name, level).
    """
    sample_frames: list[pd.DataFrame] = []
    perturb_frames: list[pd.DataFrame] = []

    for run in runs:
        pdir = run.path / "perturbations"
        if not pdir.is_dir():
            continue

        samples = _read_jsonl(pdir / "samples.jsonl")
        if samples:
            df = pd.DataFrame([_flatten_metadata(s) for s in samples])
            sample_frames.append(_attach_run(df, run))

        perturbs = _read_jsonl(pdir / "perturbations.jsonl")
        if perturbs:
            flat: list[dict] = []
            for p in perturbs:
                sid = p.get("sample_id")
                tn = p.get("test_name")
                for lv in p.get("levels", []):
                    flat.append(_flatten_metadata({
                        "sample_id": sid,
                        "test_name": tn,
                        "level": lv.get("level"),
                        "level_label": lv.get("label"),
                        "text": lv.get("text"),
                        "metadata": lv.get("metadata"),
                    }))
            if flat:
                df = pd.DataFrame(flat)
                perturb_frames.append(_attach_run(df, run))

    samples_df = _dedup_by_keys(sample_frames, ["sample_id"]) if sample_frames else None
    perturbations_df = _dedup_by_keys(
        perturb_frames, ["sample_id", "test_name", "level"]
    ) if perturb_frames else None
    return samples_df, perturbations_df


def _augment_perturbations_with_originals(
    perturbations_df: pd.DataFrame | None,
    samples_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Add ``original_summary`` and ``source`` columns to the perturbations CSV.

    Each perturbation row already carries the perturbed text in ``text``;
    this merges in the corresponding sample's untouched summary and source
    document so the same row shows the before-and-after view. Joins on
    ``CONFIG_FIELDS + sample_id`` so samples from different configs don't
    cross-contaminate.
    """
    if perturbations_df is None or perturbations_df.empty:
        return perturbations_df
    if samples_df is None or samples_df.empty:
        return perturbations_df

    join_keys = list(CONFIG_FIELDS) + ["sample_id"]
    join_keys = [k for k in join_keys if k in perturbations_df.columns and k in samples_df.columns]

    take = ["summary", "source"]
    take = [c for c in take if c in samples_df.columns]
    if not take:
        return perturbations_df

    right = samples_df[join_keys + take].rename(columns={"summary": "original_summary"})
    merged = perturbations_df.merge(right, on=join_keys, how="left")
    # Reorder so original_summary sits next to the perturbed text.
    if "text" in merged.columns and "original_summary" in merged.columns:
        cols = list(merged.columns)
        cols.remove("original_summary")
        if "source" in cols:
            cols.remove("source")
        idx = cols.index("text")
        new_cols = cols[:idx] + ["original_summary", "source"] + cols[idx:]
        # Some columns may not actually be present; filter
        new_cols = [c for c in new_cols if c in merged.columns]
        merged = merged[new_cols]
    return merged


def _write_perturbations_jsonl(
    samples_df: pd.DataFrame | None,
    perturbations_df: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    """Write samples.jsonl + perturbations.jsonl in the format that
    ``robustness.runner.load_perturbations`` consumes, so the compiled
    output directory itself can be passed to
    ``persona-eval robustness-eval --perturbations-dir <output_dir>``.

    Each ``(sample_id, test_name)`` group across runs is collapsed back
    into a single record with its sorted-by-level list of
    ``PerturbedSummary`` dicts. Within a group, level rows are already
    deduplicated by ``_compile_perturbations_jsonl`` (most recent run
    per ``(config, sample_id, test_name, level)``); the JSONL keeps the
    latest run's text for each level.

    Note: ``robustness-eval`` doesn't reason about config columns, so
    samples are deduplicated to one record per ``sample_id`` here too.
    Different-config rows for the same ``sample_id`` would collide; the
    most recent wins (sorted by ``run_id``).
    """
    if samples_df is None or samples_df.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Samples: one record per sample_id, most recent run wins.
    samples_sorted = samples_df.sort_values("run_id", kind="mergesort")
    samples_one = samples_sorted.drop_duplicates(subset=["sample_id"], keep="last")

    samples_path = output_dir / "samples.jsonl"
    sample_fields = ("sample_id", "source", "summary", "audience", "reference", "metadata")
    with open(samples_path, "w") as f:
        for _, row in samples_one.iterrows():
            obj: dict = {}
            for fld in sample_fields:
                if fld not in row.index:
                    continue
                val = row[fld]
                if pd.isna(val):
                    val = "" if fld != "metadata" else {}
                if fld == "metadata" and isinstance(val, str) and val:
                    try:
                        val = json.loads(val)
                    except json.JSONDecodeError:
                        val = {}
                obj[fld] = val
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"  samples.jsonl: wrote {len(samples_one):>6} records -> {samples_path}")

    if perturbations_df is None or perturbations_df.empty:
        return

    # Perturbations: regroup level rows back into one record per
    # (sample_id, test_name). Across configs, the most recent run wins
    # for each (sample_id, test_name, level), then we drop config keys.
    perturbs_sorted = perturbations_df.sort_values("run_id", kind="mergesort")
    perturbs_one = perturbs_sorted.drop_duplicates(
        subset=["sample_id", "test_name", "level"], keep="last",
    )

    perturbations_path = output_dir / "perturbations.jsonl"
    with open(perturbations_path, "w") as f:
        for (sid, tn), grp in perturbs_one.groupby(["sample_id", "test_name"], sort=False):
            levels = []
            for _, row in grp.sort_values("level").iterrows():
                meta = row.get("metadata", {})
                if isinstance(meta, str) and meta:
                    try:
                        meta = json.loads(meta)
                    except json.JSONDecodeError:
                        meta = {}
                elif pd.isna(meta) if not isinstance(meta, (dict, list)) else False:
                    meta = {}
                levels.append({
                    "level": int(row["level"]),
                    "label": row["level_label"] if not pd.isna(row.get("level_label")) else "",
                    "text": row["text"] if not pd.isna(row.get("text")) else "",
                    "metadata": meta if isinstance(meta, dict) else {},
                })
            f.write(json.dumps({
                "sample_id": sid, "test_name": tn, "levels": levels,
            }, ensure_ascii=False) + "\n")
    print(f"  perturbations.jsonl: wrote {len(perturbs_one.groupby(['sample_id', 'test_name'])):>6} records -> {perturbations_path}")


def _dedup_by_keys(frames: list[pd.DataFrame], extra_keys: list[str]) -> pd.DataFrame:
    """Concat + dedup helper — same logic as ``_compile_spec`` but for already-attached frames."""
    combined = pd.concat(frames, ignore_index=True, sort=False)
    full_keys = list(CONFIG_FIELDS) + [k for k in extra_keys if k in combined.columns]
    sentinel = "\x00__NA__\x00"
    keyed = combined.assign(**{k: combined[k].fillna(sentinel) for k in full_keys
                               if combined[k].dtype == object or combined[k].isna().any()})
    keyed = keyed.sort_values("run_id", kind="mergesort")
    deduped_idx = keyed.drop_duplicates(subset=full_keys, keep="last").index
    return combined.loc[sorted(deduped_idx)].reset_index(drop=True)


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

        samples_df, perturbations_df = _compile_perturbations_jsonl(rob_runs)
        _write(samples_df, output_dir / "perturbation_samples.csv", "perturbation_samples.csv")
        # Per-row CSV with original + perturbed side-by-side.
        perturbations_csv_df = _augment_perturbations_with_originals(perturbations_df, samples_df)
        _write(perturbations_csv_df, output_dir / "perturbations.csv", "perturbations.csv")
        # JSONL pair in the layout that robustness-eval --perturbations-dir consumes.
        _write_perturbations_jsonl(samples_df, perturbations_df, output_dir)

    print(f"\nDone. Compiled outputs in {output_dir}/")


if __name__ == "__main__":
    main()
