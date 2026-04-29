# Robustness Testing

Systematically perturb summaries and verify that metrics respond as expected. This validates that a metric is actually measuring what it claims to measure -- for example, that adding noise to a summary causes scores to drop, or that rephrasing without changing content keeps scores stable.

## Overview

The robustness testing system:

1. Converts annotation data into a generic `SummarizationSample` format
2. Applies perturbation tests that produce multiple "levels" of a modified summary
3. Scores each level with the requested metrics
4. Analyzes whether the score trajectory matches the expected direction

## Quick start

```bash
# Run on a standard HuggingFace dataset
persona-eval robustness --dataset scitldr \
    --tests distractor incremental \
    --metrics rouge bertscore \
    --num-samples 50 \
    --output-dir robustness_results/

# Run on persona-eval annotations
persona-eval robustness path/to/annotations.zip \
    --tests distractor incremental \
    --metrics rouge bertscore \
    --output-dir robustness_results/

# LLM-based perturbations with a local vLLM server (default provider)
persona-eval robustness --dataset elife \
    --tests distractor incremental lengthen shorten audience \
    --metrics rouge bertscore llm_judge \
    --llm-model meta-llama/Meta-Llama-3-8B-Instruct \
    --num-samples 20 \
    --output-dir robustness_results/

# LLM-based perturbations with TogetherAI
persona-eval robustness --dataset elife \
    --tests distractor incremental lengthen shorten audience \
    --metrics rouge bertscore llm_judge \
    --llm-provider together --llm-model meta-llama/Meta-Llama-3-8B-Instruct \
    --num-samples 20 \
    --output-dir robustness_results/

# Use different models for perturbation generation vs evaluation
persona-eval robustness --dataset elife \
    --tests lengthen shorten audience \
    --metrics rouge llm_judge \
    --llm-provider together --llm-model meta-llama/Meta-Llama-3-70B-Instruct \
    --perturb-provider vllm --perturb-model meta-llama/Meta-Llama-3-8B-Instruct \
    --output-dir robustness_results/
```

## CLI options

| Flag | Description | Default |
|---|---|---|
| `annotations` | Path to annotations zip or directory (positional, optional if `--dataset` is used) | -- |
| `--dataset` | Load a standard dataset from HuggingFace (see [Supported datasets](#supported-datasets)) | -- |
| `--split` | Dataset split to use | `test` (dataset-specific) |
| `--tests` | Which tests to run (see below) | all |
| `--metrics` | Metrics to evaluate | `rouge` |
| `--num-samples` | Subsample N samples to limit compute | all |
| `--seed` | Random seed for reproducibility | 42 |
| `--target-audiences` | Target audiences for the audience test | 3 defaults |
| `--output-dir` | Output directory | `robustness_results/` |
| `--cache-dir` | Cache directory for LLM outputs and OpenAlex | `cache` |
| `--llm-provider` | LLM backend for perturbation generation (`vllm`, `together`, `openai`, or `anthropic`) | `vllm` |
| `--llm-model` | Model name or path | -- |
| `--llm-base-url` | Override API base URL | `http://localhost:8000/v1` (vllm) |
| `--llm-api-key` | API key (or set `TOGETHER_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` env var) | -- |
| `--llm-temperature` | Sampling temperature for the metric LLM. Must be `1.0` when `--llm-thinking-budget` is set | `1.0` |
| `--llm-thinking-budget` | Enable Anthropic extended thinking with this token budget (≥ 1024 and strictly less than `max_tokens`). Anthropic provider only | off |
| `--perturb-provider` | LLM provider for perturbation generation (overrides `--llm-provider`) | same as `--llm-provider` |
| `--perturb-model` | Model for perturbation generation (overrides `--llm-model`) | same as `--llm-model` |
| `--perturb-base-url` | Base URL for perturbation LLM (overrides `--llm-base-url`) | same as `--llm-base-url` |
| `--perturb-api-key` | API key for perturbation LLM (overrides `--llm-api-key`) | same as `--llm-api-key` |
| `--no-cache` | Disable the metric result cache for this run (see [Caching](#caching)) | cache enabled |
| `--clear-metric-cache` | Delete all metric cache entries before running | off |

You must provide either `annotations` or `--dataset`. LLM options are required when running tests that use LLM perturbations (lengthen, shorten, audience). The `--perturb-*` flags let you use a different model/provider for perturbation generation while `--llm-*` controls the evaluation metrics.

> **vLLM setup:** Start the server before running LLM-based tests:
> ```bash
> vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8000
> ```

## Supported datasets

The `--dataset` flag supports loading scientific summarization datasets directly from HuggingFace. All HF datasets default to the `test` split unless noted otherwise.

### Available (HuggingFace)

| Name | HF ID | Description | Default split |
|---|---|---|---|
| `arxiv` | `armanc/scientific_papers` (arxiv) | Arxiv scientific paper summarization | `test` |
| `pubmed` | `armanc/scientific_papers` (pubmed) | PubMed biomedical paper summarization | `test` |
| `scitldr` | `allenai/scitldr` (AIC) | Scientific paper TLDRs | `test` |
| `elife` | `tomasg25/scientific_lay_summarisation` (elife) | eLife journal lay summaries | `test` |
| `plos` | `tomasg25/scientific_lay_summarisation` (plos) | PLOS journal lay summaries | `test` |
| `mup` | `allenai/mup` | Multi-perspective scientific paper summarization | `validation` |

### Stubs (not yet loadable)

These datasets have registry entries but are not currently available on HuggingFace. Attempting to load them raises an informative error.

| Name | Description |
|---|---|
| `cdsr` | Cochrane Database of Systematic Reviews |
| `eureka` | EurekAlert scientific press release summarization |
| `cells` | CELLS scientific summarization |
| `scinews` | Science news summarization (gated) |
| `longsumm` | Long scientific document summarization |

### Examples

```bash
# Run on SciTLDR with a small sample
persona-eval robustness --dataset scitldr --num-samples 50 --metrics rouge

# Run on PubMed test split
persona-eval robustness --dataset pubmed --metrics rouge bertscore

# Override the default split
persona-eval robustness --dataset mup --split validation --num-samples 30

# Use eLife with LLM-based tests
persona-eval robustness --dataset elife \
    --tests lengthen shorten audience \
    --llm-provider together --llm-model meta-llama/Meta-Llama-3-8B-Instruct \
    --num-samples 20
```

## The five tests

### 1. `distractor` -- Distractor sentences (expected: scores decrease)

Adds randomly sampled sentences from a different document's source text to the summary. Each level adds one more distractor sentence.

- **Level 0**: original summary
- **Level 1**: original + 1 distractor sentence
- **Level 2**: original + 2 distractor sentences
- ...up to 5 distractors by default

**Rationale**: Injecting irrelevant content should degrade summary quality scores. A robust metric should produce monotonically decreasing scores as noise increases.

### 2. `incremental` -- Incremental sentence addition (expected: scores increase)

Builds the summary back up one sentence at a time from an empty string.

- **Level 0**: empty string
- **Level 1**: first sentence only
- **Level 2**: first two sentences
- ...up to the full original summary

**Rationale**: As more of the original content is included, scores should rise. This tests whether a metric can distinguish partial from complete summaries.

### 3. `lengthen` -- Lengthen prose (expected: scores stable)

Uses an LLM to expand the summary into longer prose without adding new information. The source document is provided as context.

- **Level 0**: original summary
- **Level 1**: lengthened version (~50% longer)

**Rationale**: If the information content is the same, a robust metric should produce similar scores regardless of verbosity. Metrics that are overly sensitive to length will fail this test.

### 4. `shorten` -- Shorten prose (expected: scores stable)

Uses an LLM to condense the summary without removing information.

- **Level 0**: original summary
- **Level 1**: shortened version (~50% shorter)

**Rationale**: Same information in fewer words should not change quality scores. This is the inverse of the lengthen test.

### 5. `audience` -- Different audience (expected: scores decrease)

Uses an LLM to rewrite the summary for a different target audience. The rewrite may change which information is emphasized.

- **Level 0**: original summary (written for the original audience)
- **Level 1**: rewritten for audience 1 (default: "undergraduate student")
- **Level 2**: rewritten for audience 2 (default: "journalist")
- **Level 3**: rewritten for audience 3 (default: "domain expert in the field")

**Rationale**: When a summary is rewritten for a different audience, information emphasis changes. Persona-aware metrics should detect this as a quality decrease relative to the original audience. The default audiences can be overridden:

```bash
persona-eval robustness annotations.zip \
    --tests audience \
    --target-audiences "high school student" "policy maker" \
    --llm-provider together --llm-model my-model
```

## Output files

The robustness subcommand produces:

| File | Description |
|---|---|
| `robustness_scores.csv` | Raw scores: one row per (sample, test, level, metric) |
| `robustness_analysis.csv` | Per-(test, metric) statistics |
| `llm_responses/` | Full LLM prompts and responses (when LLM metrics are used) |

## Analysis methodology

For each (test, metric) pair, the analysis computes:

| Statistic | Description |
|---|---|
| **Spearman rho** | Correlation between perturbation level and mean score across samples |
| **Actual direction** | Classified as `increase`, `decrease`, or `stable` based on rho and p-value |
| **Direction match** | Whether actual direction matches the test's expected direction |
| **Effect size** | Cohen's d between baseline (level 0) and final level scores |
| **Monotonicity** | Fraction of consecutive-level pairs where the score change matches expectation |

**Direction classification**:
- `increase`: Spearman rho > 0.1 and statistically significant (p < 0.05)
- `decrease`: Spearman rho < -0.1 and statistically significant
- `stable`: neither of the above

**Monotonicity for stable tests**: a level transition is "correct" if the absolute score change is less than 5% of the baseline score.

## Caching

Three layers of caching make re-runs cheap. All are on by default under `{cache-dir}/`.

| Cache | Location | What it stores |
|---|---|---|
| LLM perturbations | `{cache-dir}/robustness/` | Output text of LLM-generated perturbations (tests 3-5). Keyed by prompt hash so editing a prompt auto-invalidates. |
| Metric results | `{cache-dir}/metrics/` | Per-`(summary, source)` metric output (e.g. ROUGE, BERTScore, LLM Judge sub-scores). Shared across modes: a result computed during annotation analysis is reused in robustness testing, and vice versa. |
| `robustness_scores.csv` | `{output-dir}/run_*/` | Acts as a checkpoint. On re-run, metrics whose columns are already populated for every `(sample_id, test_name, level)` are skipped. The file is rewritten after each metric completes. |

Disable the metric cache for a single run with `--no-cache`. Wipe it with `--clear-metric-cache` (useful after editing a prompt template, since `cache_config()` tracks prompt-file path, not contents).

```bash
# Evaluate with a new metric on already-generated perturbations -- only the new
# metric runs; prior ones are skipped via the CSV checkpoint
persona-eval robustness-eval --perturbations-dir robustness_results/run_123/perturbations \
    --metrics rouge bertscore llm_judge --llm-model my-model \
    --output-dir robustness_results/run_123/

# Force a clean recompute
persona-eval robustness --dataset scitldr --metrics rouge \
    --clear-metric-cache --output-dir robustness_results/
```

## Dataset abstraction

The robustness system uses a generic `SummarizationSample` dataclass:

```python
@dataclass
class SummarizationSample:
    sample_id: str      # Unique identifier
    source: str         # Source document text
    summary: str        # Summary text
    audience: str       # Intended audience description
    reference: str      # Optional reference text
    metadata: dict      # Extra info
```

Two built-in adapters convert external data into this format:

- **`load_from_persona_eval()`** -- converts the persona-eval annotation format (used with the `annotations` positional argument).
- **`load_from_huggingface(dataset_name)`** -- loads any registered dataset from HuggingFace (used with `--dataset`). The `DATASET_REGISTRY` maps each dataset name to its HF ID, config, column names, and default split.

To add a new HuggingFace dataset, add an entry to `DATASET_REGISTRY` in `src/persona_eval/robustness/dataset.py`:

```python
DATASET_REGISTRY["my_dataset"] = {
    "hf_id": "org/dataset-name",
    "hf_config": None,            # or a specific config name
    "default_split": "test",
    "source_col": "article",
    "summary_col": "summary",
    "reference_col": "title",     # or None
    "description": "My dataset description",
    "available": True,
}
```

For datasets not on HuggingFace, you can write a custom adapter:

```python
def load_from_my_dataset(path) -> list[SummarizationSample]:
    samples = []
    for item in read_data(path):
        samples.append(SummarizationSample(
            sample_id=item["id"],
            source=item["article"],
            summary=item["summary"],
            audience="general news reader",
        ))
    return samples
```

## Adding a custom perturbation test

Subclass `BasePerturbationTest`:

```python
from persona_eval.robustness.perturbations import BasePerturbationTest, PerturbedSummary
from persona_eval.robustness.dataset import SummarizationSample

class MyTest(BasePerturbationTest):
    @property
    def name(self) -> str:
        return "my_test"

    @property
    def expected_direction(self) -> str:
        return "decrease"  # or "increase" or "stable"

    def generate_levels(self, sample: SummarizationSample) -> list[PerturbedSummary]:
        levels = [PerturbedSummary(level=0, label="original", text=sample.summary)]
        # Apply your perturbation
        perturbed = my_perturbation(sample.summary)
        levels.append(PerturbedSummary(level=1, label="perturbed", text=perturbed))
        return levels
```

For LLM-based tests, inherit from `_LLMPerturbationTest` instead to get caching and LLM client access via `self._generate_with_cache()`.

## Programmatic usage

### Loading from HuggingFace

```python
from persona_eval.robustness import (
    load_from_huggingface,
    list_available_datasets,
    DistractorSentenceTest,
    IncrementalAdditionTest,
    run_robustness,
    analyze_robustness,
    print_robustness_report,
)

# See which datasets are available
print(list_available_datasets())
# ['arxiv', 'pubmed', 'scitldr', 'elife', 'plos', 'mup']

# Load samples from a HuggingFace dataset
samples = load_from_huggingface("scitldr", num_samples=50, seed=42)

tests = [
    DistractorSentenceTest(distractor_pool=[s.source for s in samples], seed=42),
    IncrementalAdditionTest(),
]

scores_df = run_robustness(
    samples=samples,
    tests=tests,
    metric_names=["rouge"],
    output_dir="my_results/",
)

analysis_df = analyze_robustness(scores_df, tests)
print_robustness_report(analysis_df)
```

### Manual sample construction

```python
from persona_eval.robustness import (
    SummarizationSample,
    DistractorSentenceTest,
    IncrementalAdditionTest,
    run_robustness,
    analyze_robustness,
    print_robustness_report,
)

samples = [
    SummarizationSample(
        sample_id="s1",
        source="Full source document text...",
        summary="A concise summary of the document.",
        audience="general reader",
    ),
]

tests = [
    DistractorSentenceTest(distractor_pool=[s.source for s in samples], seed=42),
    IncrementalAdditionTest(),
]

scores_df = run_robustness(
    samples=samples,
    tests=tests,
    metric_names=["rouge"],
    output_dir="my_results/",
)

analysis_df = analyze_robustness(scores_df, tests)
print_robustness_report(analysis_df)
```
