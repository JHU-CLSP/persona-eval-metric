# persona-eval — architecture notes

Evaluates automatic summarization metrics against human pairwise
preferences and measures their robustness under controlled
perturbations. Source in `src/persona_eval/`.

## Package layout

```
persona_eval/
├── cli/                    One module per subcommand; main.py is the dispatcher.
│   ├── main.py             argparse wiring; registers each command module.
│   ├── _args.py            Reusable argparse fragments (data/metric/LLM/correlation).
│   ├── _common.py          Shared helpers: metric resolution, data loading,
│   │                       response-logger, metric-cache construction,
│   │                       compute_and_save, run_correlations, robustness helpers.
│   └── <command>.py        One file per subcommand, each exposes register() + run().
├── core/
│   ├── cache.py            JsonFileCache base + hashing helpers + config_hash.
│   └── pipeline.py         Metric orchestration: build_tasks,
│                           compute_metric_scores, run_pairwise_comparisons,
│                           score_cached, score_pair_cached.
├── metrics/
│   ├── base.py             BaseMetric ABC, @register_metric, get_metric/list_*.
│   ├── base_llm.py         BaseLLMMetric (client + logger),
│   │                       BaseDimensionalLLMMetric (per-dimension judges),
│   │                       MultiStepLLMMetric (multi-call pipelines),
│   │                       load_prompt_template, load_rubric, parse_bullet_list,
│   │                       SCORE_RE / PAIRWISE_RE / parse_pairwise_result.
│   ├── cache.py            MetricCache (thin subclass of core.cache.JsonFileCache).
│   ├── summeval_metrics.py 8 wrappers defined declaratively in SUMMEVAL_METRICS;
│   │                       plus METEOR (nltk-based, not summ-eval).
│   ├── rouge_metrics.py    rouge-score wrapper.
│   ├── bertscore_metric.py BERTScore wrapper (batched).
│   ├── syntactic_metric.py spaCy syntactic-complexity metric.
│   ├── factscore_metric.py FACTScore atomic-fact verification.
│   ├── llm_judge_metric.py           Absolute grading via BaseDimensionalLLMMetric.
│   ├── llm_judge_relative_metric.py  Pairwise grading via BaseDimensionalLLMMetric.
│   ├── llm_judge_annotator_metric.py Query-focused pairwise (single call per pair).
│   ├── persona_precision_metric.py   Extracts nuggets, checks relevance.
│   └── persona_recall_metric.py      Generates requirements, checks coverage.
├── robustness/
│   ├── dataset.py          SummarizationSample + HuggingFace adapters.
│   ├── perturbations.py    5 perturbation tests (distractor/incremental/
│   │                       lengthen/shorten/audience).
│   ├── runner.py           run_robustness / generate_perturbations /
│   │                       score_perturbations workflows.
│   ├── analysis.py         Effect analysis and reporting.
│   └── cache.py            PerturbationCache (thin subclass of JsonFileCache).
├── annotations.py          AnnotationEntry/AnnotatorProfile, zip/dir loader,
│                           get_pairwise_preferences.
├── correlation.py          Pairwise-agreement + rank-correlation computation.
├── llm_client.py           LLMClient (vLLM / TogetherAI) + ResponseLogger (JSONL).
├── openalex.py             OpenAlex client with file-based abstract cache.
└── prompts/                Prompt templates + rubrics (plain text, not code).
```

## How to add a new metric

1. Pick the right base class:
   - **Plain metric** (no LLM): subclass `BaseMetric` (`metrics/base.py`). Implement
     `name`, `score(summary, source, persona_kwargs=None)`, set
     `is_reference_free` / `is_pairwise` as needed.
   - **summ-eval wrapper**: add an entry to `SUMMEVAL_METRICS` in
     `summeval_metrics.py`. No new class needed.
   - **LLM judge (per-dimension)**: subclass `BaseDimensionalLLMMetric` and
     call `_score_dim(...)` per dimension, passing a regex + parser.
   - **Multi-step LLM metric**: subclass `MultiStepLLMMetric` and orchestrate
     calls via `_run_step(...)`; use `_classify_positive` for yes/no checks.
   - **LLM metric, other shape**: subclass `BaseLLMMetric` directly.
2. Decorate the class with `@register_metric("your_metric_name")`.
3. Import the module in `metrics/__init__.py` so the decorator fires.
4. Override `cache_config()` to include anything that changes output
   (model, prompt file, persona flag). The default covers plain metrics.

## How to add a new CLI command

1. Create `cli/<command>.py` with two functions:
   ```python
   def register(subparsers):
       sp = subparsers.add_parser("my-cmd", help="...")
       # add arguments (use helpers from cli._args)
       sp.set_defaults(func=run)

   def run(args):
       ...
   ```
2. Import the module in `cli/main.py` and add it to `_COMMANDS`.
3. Pull shared plumbing from `cli._common` (metric resolution, data
   loading, cache construction, response logger, correlation runner).

## Caches

Two on-disk caches, both subclasses of `core.cache.JsonFileCache`:

- **`MetricCache`** at `{cache_dir}/metrics/{sha256[:32]}.json` — one file per
  `(metric_name, config_hash, summary, source, persona)` key. Used by
  `score_cached` / `score_pair_cached` in `core.pipeline`.
- **`PerturbationCache`** at `{cache_dir}/robustness/{sha256[:32]}.json` — one
  file per `(test, sample_id, level, model, prompt_hash)` key. Used by the
  LLM-based perturbation tests.

Cache invalidation is automatic: any change to the key fields (including
the metric's `cache_config()`) produces a new filename, leaving stale
entries untouched. `--clear-metric-cache` empties the metrics dir.

`OpenAlexClient` also uses its own file-per-paper cache under
`{cache_dir}/openalex/`.

## Where to look for more

- User-facing feature list & quick start → `README.md`
- Included and optional metrics + setup instructions → `METRICS.md`
- Annotation/correlation CLI reference → `docs/annotation-analysis.md`
- Robustness framework details → `docs/robustness-testing.md`
- Per-metric "neither" thresholds → `neither_thresholds.yaml`
