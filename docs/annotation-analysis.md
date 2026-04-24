# Annotation Analysis

Compute automatic summarization metrics and measure how well they correlate with human pairwise preferences.

## Annotation data format

The tool accepts a zip file or directory containing:

```
annotations/
├── users.json              # Annotator metadata
└── results/
    ├── annotator-xxx.json  # Per-annotator annotations
    └── ...
```

Each annotator JSON file contains queries with four summaries (A/B/C/D), OpenAlex paper IDs, and pairwise preferences:

- `round1_ab`: preference between A and B (`"A"`, `"B"`, or `"N"` for neither)
- `round1_cd`: preference between C and D
- `final`: overall winner (`"A"`/`"B"`/`"C"`/`"D"` or `"N"`)

## CLI commands

### `run-all` -- Full pipeline

```bash
persona-eval run-all annotations.zip \
    --metrics rouge bleu supert \
    --cache-dir cache/ \
    --device cpu \
    --output-dir results/
```

Outputs:
- `results/metric_scores.csv` -- per-(query, summary) metric scores (including tournament points for pairwise metrics)
- `results/pairwise_prefs.csv` -- raw LLM pairwise preferences per comparison (only when pairwise metrics are run)
- `results/llm_responses/llm_responses_<timestamp>.jsonl` -- full LLM prompts, responses, and parsed results (when LLM metrics are run)
- `results/pairwise_agreement.csv` -- agreement rate per metric
- `results/rank_correlation_per_query.csv` -- per-query Kendall tau and Spearman rho
- `results/rank_correlation_aggregate.csv` -- mean/median/std across queries

### `compute-metrics` -- Compute metrics only

```bash
persona-eval compute-metrics annotations.zip \
    --metrics rouge bertscore supert \
    --output metric_scores.csv
```

### `correlate` -- Compute correlations from precomputed scores

```bash
persona-eval correlate annotations.zip \
    --scores metric_scores.csv \
    --output-dir results/

# With raw pairwise preferences (for direct preference matching)
persona-eval correlate annotations.zip \
    --scores metric_scores.csv \
    --pairwise-prefs pairwise_prefs.csv \
    --output-dir results/
```

### `list-metrics` -- Show available metrics

```bash
persona-eval list-metrics
```

### `fetch-sources` -- Pre-fetch OpenAlex data

```bash
persona-eval fetch-sources annotations.zip --cache-dir cache/ --email you@example.com
```

Passing `--email` uses the OpenAlex polite pool (100 req/s vs 10 req/s).

## LLM-based metrics

The `llm_judge`, `llm_judge_relative`, and `factscore` metrics require an LLM backend. Both local vLLM and TogetherAI are supported via their OpenAI-compatible APIs.

### Using vLLM (local)

```bash
# In a separate terminal
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8000

# Run metrics
persona-eval compute-metrics annotations.zip \
    --metrics llm_judge factscore \
    --llm-provider vllm \
    --llm-model meta-llama/Meta-Llama-3-8B-Instruct
```

The default base URL is `http://localhost:8000/v1`. Override with `--llm-base-url`.

### Using TogetherAI

```bash
export TOGETHER_API_KEY=your-key-here

persona-eval compute-metrics annotations.zip \
    --metrics llm_judge \
    --llm-provider together \
    --llm-model meta-llama/Meta-Llama-3-8B-Instruct
```

### LLM CLI options

| Flag | Description | Default |
|---|---|---|
| `--llm-provider` | Backend: `vllm` or `together` | `vllm` |
| `--llm-model` | Model name/path (required for LLM metrics) | -- |
| `--llm-api-key` | API key (or use `TOGETHER_API_KEY` env var) | `EMPTY` for vLLM |
| `--llm-base-url` | Override API base URL | `localhost:8000/v1` (vLLM) |
| `--llm-prompt-file` | Custom prompt template for `llm_judge` | built-in default |
| `--persona` | Enable persona-aware evaluation using annotator profiles | off |
| `--include-query` | Include the annotator's query in LLM judge prompts | off |

### Caching options

| Flag | Description | Default |
|---|---|---|
| `--cache-dir` | Root directory for all caches (OpenAlex, metrics, LLM perturbations) | `cache` |
| `--no-cache` | Disable the metric result cache for this run | cache enabled |
| `--clear-metric-cache` | Delete all metric cache entries before running | off |

See [Caching](#caching) below for how the cache works.

### Evaluation dimensions

Both `llm_judge` and `llm_judge_relative` evaluate summaries on five dimensions:

- **Relevance**: Does the summary capture key information from the source?
- **Coherence**: Is the summary well-organized and easy to read?
- **Consistency**: Is the summary factually consistent with the source?
- **Fluency**: Is the summary grammatically correct and well-written?
- **Informativeness**: How useful is the summary? (persona variant available)

### Absolute grading

The `llm_judge` metric uses the Prometheus absolute grading format. It makes one LLM call per dimension, each with its own rubric. Each call returns a 1-5 score via a `[RESULT]` tag. The overall score is the average across dimensions.

```bash
persona-eval compute-metrics annotations.zip \
    --metrics llm_judge \
    --llm-provider vllm \
    --llm-model prometheus-eval/prometheus-7b-v2.0
```

### Relative grading (pairwise)

The `llm_judge_relative` metric compares summaries head-to-head, mirroring the human annotation tournament:

1. **Round 1**: A vs B, C vs D (2 LLM calls per dimension x 5 dimensions)
2. **Final**: winner of AB vs winner of CD (1 LLM call per dimension x 5 dimensions)

This produces two outputs:

- **`pairwise_prefs.csv`** -- raw LLM preferences per comparison, used for pairwise agreement
- **`metric_scores.csv`** -- tournament point scores per summary (0-3 scale), used for rank correlation

Pairwise agreement resolves preferences as follows:
- **Round 1 (A vs B, C vs D)**: direct match against the LLM's round 1 result
- **Final winner vs final loser**: direct match against the LLM's final result
- **Final winner vs round loser (same bracket)**: resolved from the LLM's round 1 result
- **Final winner vs round loser (other bracket)**: resolved transitively (round 1 + final must both agree)

### Custom judge prompts

1. Copy the default prompt from `src/persona_eval/prompts/llm_judge_default.txt` (absolute) or `llm_judge_relative.txt` (relative)
2. Edit it using `{summary}`, `{source}`, `{dimension}`, and `{rubric}` placeholders (or `{summary_a}`, `{summary_b}` for relative)
3. The LLM must return a `[RESULT]` tag: an integer 1-5 (absolute) or `A`/`B` (relative)
4. Pass your template with `--llm-prompt-file my_prompt.txt`

Rubrics for each dimension live in `src/persona_eval/prompts/rubrics/`.

### Including the annotator's query

Use `--include-query` to add the annotator's query to LLM judge prompts:

```bash
persona-eval compute-metrics annotations.zip \
    --metrics llm_judge \
    --llm-provider vllm --llm-model my-model \
    --include-query
```

Note: `llm_judge_annotator` always includes the query automatically.

### Persona-aware evaluation

The `--persona` flag enables evaluation from the perspective of each individual annotator:

```bash
persona-eval compute-metrics annotations.zip \
    --metrics llm_judge \
    --llm-provider vllm \
    --llm-model prometheus-eval/prometheus-7b-v2.0 \
    --persona
```

When enabled:

- The **informativeness** rubric changes to persona-specific, incorporating the annotator's role, domain, and information needs
- The prompt template includes the annotator's profile
- Scoring is **per-annotator**: the same summary may receive different scores for different annotators

Annotator profiles are loaded from `users.json` with fields: `role`, `domain`, `info_needs`.

### Annotator query-focused evaluation

The `llm_judge_annotator` metric evaluates summaries based on whether they address the annotator's specific query. It uses a single LLM call per pair, with the annotator's profile and query in the prompt.

```bash
persona-eval compute-metrics annotations.zip \
    --metrics llm_judge_annotator \
    --llm-provider vllm --llm-model my-model \
    --persona
```

### FACTScore

Implements the FACTScore algorithm (Min et al., 2023):

1. Decomposes the summary into atomic facts using the LLM
2. Verifies each fact against the source document
3. Returns the fraction of supported facts, along with `factscore_num_facts` and `factscore_num_supported`

## "Neither" annotations

By default, "N" (neither) annotations are excluded. To include them:

```bash
persona-eval run-all annotations.zip \
    --include-neither \
    --neither-config neither_thresholds.yaml \
    --output-dir results/
```

The metric "agrees" with a "neither" judgment if the absolute score difference is below a per-metric threshold:

```yaml
# neither_thresholds.yaml
rouge1_f: 0.05
bertscore_f: 0.02
supert: 0.05
```

A default config is provided in `neither_thresholds.yaml`.

## Strict pairwise agreement

With `--strict-pairwise`, the final-round comparison automatically counts as disagreement if the metric got any round 1 comparison wrong for that annotator and query:

```bash
persona-eval run-all annotations.zip --strict-pairwise --output-dir results/

# Combined with neither
persona-eval run-all annotations.zip \
    --strict-pairwise --include-neither --neither-config neither_thresholds.yaml \
    --output-dir results/
```

## How correlation is computed

**Pairwise agreement** (primary): For each human preference (e.g., "A is better than B"), check whether the metric agrees. Standard metrics compare numeric scores. Pairwise metrics match raw LLM preferences directly.

**Rank correlation** (complementary): For each query, derive a human ranking from preferences (final winner = rank 1, round winners = rank 2, rest = rank 3.5). Compute Kendall's tau-b and Spearman's rho against the metric's ranking. Results are aggregated (mean, median, std) across queries.

## Analyzing metric scores

```bash
# Full analysis with plots
python scripts/analyze_scores.py metric_scores.csv -o analysis/

# Analyze specific metrics
python scripts/analyze_scores.py metric_scores.csv --metrics rouge1_f bertscore_f supert

# With agreement/disagreement examples
python scripts/analyze_scores.py metric_scores.csv --annotations annotations.zip -o analysis/

# Stats only, no plots
python scripts/analyze_scores.py metric_scores.csv --no-plots
```

**Terminal output:** descriptive statistics, mean scores per label, discriminative power, inter-metric correlations, agreement/disagreement examples.

**Saved plots:** distributions, boxplots by label, discriminative power, correlation heatmap, per-query heatmap.

## Caching

Two layers of caching make re-runs cheap and crash-resilient. Both are on by default.

### 1. Metric result cache

Per-summary metric outputs are stored on disk under `{cache-dir}/metrics/` as one JSON file per entry. The cache key is a SHA-256 hash of:

- Metric name and its `cache_config()` (model, prompt file, persona flag, etc.)
- Summary text
- Source/reference text
- Persona kwargs (for per-annotator runs)

On a hit, `metric.score()` (or `score_pair()` for pairwise metrics) is not called. This works across invocations *and* across run modes -- a summary scored in annotation analysis is reused if the same `(summary, source)` pair shows up in robustness testing, and vice versa.

```bash
# Normal run (cache enabled, default)
persona-eval compute-metrics annotations.zip --metrics llm_judge \
    --llm-model my-model

# Re-run: LLM calls are skipped for cached (summary, source) pairs
persona-eval compute-metrics annotations.zip --metrics llm_judge \
    --llm-model my-model
# Output includes a per-metric line like: "cache: 240 hits / 0 misses"

# Disable the cache for a single run
persona-eval compute-metrics annotations.zip --metrics llm_judge \
    --llm-model my-model --no-cache

# Wipe the cache before running (e.g. after editing a prompt template)
persona-eval compute-metrics annotations.zip --metrics llm_judge \
    --llm-model my-model --clear-metric-cache
```

**What invalidates a cache entry?** Anything in `cache_config()`: changing the LLM model, flipping `--persona`, or pointing to a different `--llm-prompt-file` all produce new cache keys, leaving old entries untouched.

**Caveat:** `cache_config()` records the prompt-file *path*, not its contents. If you edit a prompt template in place, the cache key doesn't change and you'll get stale results. Run with `--clear-metric-cache` (or delete the relevant files under `{cache-dir}/metrics/`) after editing a prompt.

### 2. Output CSV as a checkpoint

`compute-metrics` (and `run-all`) save the output CSV *progressively* -- after each metric finishes, not just at the end. On re-run, the tool reads the existing CSV and skips any metric whose sub-metric columns are already populated for every task.

```bash
# Initial run: ROUGE + BERTScore
persona-eval compute-metrics annotations.zip --metrics rouge bertscore \
    --output results/scores.csv

# Add LLM Judge later; ROUGE and BERTScore are skipped
persona-eval compute-metrics annotations.zip --metrics rouge bertscore llm_judge \
    --llm-model my-model --output results/scores.csv
# Output: "Skipping rouge (already in output CSV)"
#         "Skipping bertscore (already in output CSV)"
#         "Computing llm_judge..."
```

If a run crashes partway through, completed metrics are already saved -- re-running picks up where it left off. Pairwise preferences (`*_pairwise_prefs.csv`) are saved on the same cadence.

**Note on `run-all` / `compute-metrics` output paths:** these commands append a `_run_{timestamp}` suffix, so each invocation writes a fresh file by default. To resume a prior run, reuse its output path explicitly.

## Adding a custom metric

```python
from persona_eval.metrics.base import BaseMetric, register_metric

@register_metric("my_metric")
class MyMetric(BaseMetric):
    def __init__(self, model_name: str = "default", **kwargs):
        self._model = None
        self._model_name = model_name

    @property
    def name(self) -> str:
        return "My Metric"

    @property
    def is_reference_free(self) -> bool:
        return True  # True -> abstracts, False -> titles

    def score(self, summary: str, source: str, persona_kwargs=None) -> dict[str, float]:
        return {"my_metric_score": 0.5}

    def cache_config(self) -> dict:
        # Override when your metric has extra constructor state that affects
        # output. Defaults to {"class": type(self).__name__}, which is enough
        # for metrics with no tunable knobs.
        return {**super().cache_config(), "model_name": self._model_name}
```

For LLM-backed metrics, inherit from `BaseLLMMetric` instead:

```python
from persona_eval.metrics.base import register_metric
from persona_eval.metrics.base_llm import BaseLLMMetric

@register_metric("my_llm_metric")
class MyLLMMetric(BaseLLMMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "My LLM Metric"

    def score(self, summary: str, source: str, persona_kwargs=None) -> dict[str, float]:
        self._load()  # lazily creates the LLM client
        response = self._client.generate("your prompt here")
        self._log_response("my_llm_metric", "prompt", response, parsed_result=None)
        return {"my_llm_score": 1.0}
```

Then register it in `src/persona_eval/metrics/__init__.py`:

```python
from persona_eval.metrics import my_module  # noqa: F401
```
