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
# Deterministic tests only (no LLM needed)
persona-eval robustness path/to/annotations.zip \
    --tests distractor incremental \
    --metrics rouge bertscore \
    --output-dir robustness_results/

# All tests including LLM-based perturbations
persona-eval robustness path/to/annotations.zip \
    --tests distractor incremental lengthen shorten audience \
    --metrics rouge bertscore llm_judge \
    --llm-provider together --llm-model meta-llama/Meta-Llama-3-8B-Instruct \
    --output-dir robustness_results/
```

## CLI options

| Flag | Description | Default |
|---|---|---|
| `--tests` | Which tests to run (see below) | all |
| `--metrics` | Metrics to evaluate | `rouge` |
| `--num-samples` | Subsample N samples to limit compute | all |
| `--seed` | Random seed for reproducibility | 42 |
| `--target-audiences` | Target audiences for the audience test | 3 defaults |
| `--output-dir` | Output directory | `robustness_results/` |
| `--cache-dir` | Cache directory for LLM outputs and OpenAlex | `cache` |

LLM options (`--llm-provider`, `--llm-model`, `--llm-api-key`, `--llm-base-url`) are the same as for annotation analysis and are required when running tests that use LLM perturbations (lengthen, shorten, audience).

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

LLM-generated perturbations (tests 3-5) are cached to disk under `{cache-dir}/robustness/`. Cache keys include a hash of the prompt template, so modifying a prompt automatically invalidates stale entries. Cached results are reused on subsequent runs, making it cheap to re-run with different metrics.

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

The `load_from_persona_eval()` adapter converts the existing annotation format into this representation. New dataset adapters (e.g., for CNN/DailyMail) can follow the same pattern:

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
