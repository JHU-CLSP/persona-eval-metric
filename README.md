# persona-eval-metric

Evaluate how well automatic summarization metrics correlate with human pairwise preferences. Built for scientific paper summarization with source documents from [OpenAlex](https://openalex.org/).

## Overview

This pipeline:

1. **Loads human annotation data** — pairwise preferences over AI-generated summaries (A vs B, C vs D, final winner)
2. **Fetches source documents** from OpenAlex — paper abstracts (for reference-free metrics) and titles (for reference-based metrics)
3. **Computes automatic metrics** — 15 metrics including LLM-as-judge (absolute, relative, and annotator-focused) and FACTScore, plus traditional metrics from summ-eval, rouge-score, bert-score, nltk, and spacy
4. **Measures correlation** — pairwise agreement rate and rank correlation (Kendall tau, Spearman rho) between metrics and human judgments

## Requirements

- **Python 3.9** (required by summ-eval dependency constraints)
- ~3 GB disk for model downloads on first run

## Setup

```bash
# Create a Python 3.9 virtual environment
python3.9 -m venv .venv
source .venv/bin/activate

# Install with pinned dependencies for summ-eval compatibility
pip install "spacy>=2.2,<3.8"
pip install -e .
pip install "transformers>=4.20,<4.37"
pip install "setuptools<71"

# Download spacy English model (required by DataStats and Syntactic)
python -m spacy download en_core_web_sm
```

## Quick start

```bash
# Run the full pipeline with ROUGE only (fast, no model downloads)
persona-eval run-all path/to/annotations.zip --metrics rouge --output-dir results/

# Run with all metrics
persona-eval run-all path/to/annotations.zip --output-dir results/

# Run on GPU
persona-eval run-all path/to/annotations.zip --device cuda --output-dir results/
```

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

### `run-all` — Full pipeline

```bash
persona-eval run-all annotations.zip \
    --metrics rouge bleu supert \
    --cache-dir cache/ \
    --device cpu \
    --output-dir results/
```

Outputs:
- `results/metric_scores.csv` — per-(query, summary) metric scores (including tournament points for pairwise metrics)
- `results/pairwise_prefs.csv` — raw LLM pairwise preferences per comparison (only when pairwise metrics are run)
- `results/llm_responses/llm_responses_<timestamp>.jsonl` — full LLM prompts, responses, and parsed results (when LLM metrics are run)
- `results/pairwise_agreement.csv` — agreement rate per metric
- `results/rank_correlation_per_query.csv` — per-query Kendall tau and Spearman rho
- `results/rank_correlation_aggregate.csv` — mean/median/std across queries

### `compute-metrics` — Compute metrics only

```bash
persona-eval compute-metrics annotations.zip \
    --metrics rouge bertscore supert \
    --output metric_scores.csv
```

### `correlate` — Compute correlations from precomputed scores

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

### `list-metrics` — Show available metrics

```bash
persona-eval list-metrics
```

### `fetch-sources` — Pre-fetch OpenAlex data

```bash
persona-eval fetch-sources annotations.zip --cache-dir cache/ --email you@example.com
```

Passing `--email` uses the OpenAlex polite pool (100 req/s vs 10 req/s).

## Including "neither" annotations

By default, annotations where the human selected "N" (neither summary is better) are excluded. To include them:

```bash
persona-eval run-all annotations.zip \
    --include-neither \
    --neither-config neither_thresholds.yaml \
    --output-dir results/
```

When enabled, the metric "agrees" with a "neither" judgment if the absolute score difference between the two summaries is below a per-metric threshold defined in the YAML config:

```yaml
# neither_thresholds.yaml
rouge1_f: 0.05
bertscore_f: 0.02
supert: 0.05
bleu: 2.0
```

A default config covering all metrics is provided in `neither_thresholds.yaml`.

## Strict pairwise agreement

By default, pairwise agreement evaluates each comparison independently. With `--strict-pairwise`, the final-round comparison automatically counts as disagreement if the metric got any round 1 comparison wrong for that annotator and query.

This reflects the logic that if a metric picks the wrong winner in A vs B or C vs D, the final comparison (between round winners) is meaningless — even if the metric happens to agree with the human's final choice, it arrived there via the wrong path.

```bash
# With correlate
persona-eval correlate annotations.zip --scores metric_scores.csv --strict-pairwise

# With run-all
persona-eval run-all annotations.zip --strict-pairwise --output-dir results/
```

Can be combined with `--include-neither`:

```bash
persona-eval run-all annotations.zip \
    --strict-pairwise \
    --include-neither \
    --neither-config neither_thresholds.yaml \
    --output-dir results/
```

## Available metrics

| Metric | Name | Type | Compared against |
|---|---|---|---|
| `rouge` | ROUGE-1/2/L | reference-based | titles |
| `bertscore` | BERTScore | reference-based | titles |
| `bleu` | BLEU | reference-based | titles |
| `chrf` | ChrF++ | reference-based | titles |
| `cider` | CIDEr | reference-based | titles |
| `meteor` | METEOR | reference-based | titles |
| `supert` | SUPERT | reference-free | abstracts |
| `summaqa` | SummaQA | reference-free | abstracts |
| `blanc` | BLANC | reference-free | abstracts |
| `data_stats` | DataStats | reference-free | abstracts |
| `syntactic` | Syntactic | reference-free | _(summary only)_ |
| `llm_judge` | LLM Judge (Absolute) | reference-free | abstracts |
| `llm_judge_relative` | LLM Judge (Relative) | reference-free, pairwise | abstracts |
| `llm_judge_annotator` | LLM Judge (Annotator) | reference-free, pairwise, persona | abstracts |
| `factscore` | FACTScore | reference-free | abstracts |

**Reference-based** metrics compare the summary against concatenated paper titles.
**Reference-free** metrics compare the summary against concatenated paper abstracts.
**Syntactic** only analyzes the summary text itself (L2 Syntactic Complexity indices).
**LLM-based** metrics (`llm_judge`, `llm_judge_relative`, `factscore`) use an LLM to evaluate summaries — see [LLM-based metrics](#llm-based-metrics) below.
**Pairwise** metrics (`llm_judge_relative`) compare summaries head-to-head rather than scoring individually — see [Relative grading](#relative-grading-pairwise) below.

Additional summ-eval metrics (MoverScore, SentenceMovers, ROUGE-WE, S3) require external dependencies — see [METRICS.md](METRICS.md) for setup instructions.

## LLM-based metrics

The `llm_judge`, `llm_judge_relative`, and `factscore` metrics require an LLM backend. Both local [vLLM](https://docs.vllm.ai/) and [TogetherAI](https://www.together.ai/) are supported via their OpenAI-compatible APIs.

### Using vLLM (local)

Start a vLLM server, then point the CLI at it:

```bash
# In a separate terminal
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8000

# Run metrics
persona-eval compute-metrics annotations.zip \
    --metrics llm_judge factscore \
    --llm-provider vllm \
    --llm-model meta-llama/Meta-Llama-3-8B-Instruct
```

The default base URL is `http://localhost:8000/v1`. Override with `--llm-base-url` if your server runs elsewhere.

### Using TogetherAI

Set your API key and specify the provider:

```bash
export TOGETHER_API_KEY=your-key-here

persona-eval compute-metrics annotations.zip \
    --metrics llm_judge \
    --llm-provider together \
    --llm-model meta-llama/Meta-Llama-3-8B-Instruct
```

Alternatively, pass the key directly with `--llm-api-key`.

### LLM CLI options

| Flag | Description | Default |
|---|---|---|
| `--llm-provider` | Backend: `vllm` or `together` | `vllm` |
| `--llm-model` | Model name/path (required for LLM metrics) | — |
| `--llm-api-key` | API key (or use `TOGETHER_API_KEY` env var) | `EMPTY` for vLLM |
| `--llm-base-url` | Override API base URL | `localhost:8000/v1` (vLLM) |
| `--llm-prompt-file` | Custom prompt template for `llm_judge` | built-in default |
| `--persona` | Enable persona-aware evaluation using annotator profiles | off |
| `--include-query` | Include the annotator's query in LLM judge prompts | off |

### Evaluation dimensions

Both `llm_judge` and `llm_judge_relative` evaluate summaries on five dimensions:

- **Relevance**: Does the summary capture key information from the source?
- **Coherence**: Is the summary well-organized and easy to read?
- **Consistency**: Is the summary factually consistent with the source?
- **Fluency**: Is the summary grammatically correct and well-written?
- **Informativeness**: How useful is the summary? (see [Persona-aware evaluation](#persona-aware-evaluation) for the persona variant)

### Absolute grading (default)

The `llm_judge` metric uses the Prometheus absolute grading format. It makes one LLM call per dimension (relevance, coherence, consistency, fluency, informativeness), each with its own rubric. Each call returns a 1-5 score via a `[RESULT]` tag. The overall score is the average across dimensions.

```bash
persona-eval compute-metrics annotations.zip \
    --metrics llm_judge \
    --llm-provider vllm \
    --llm-model prometheus-eval/prometheus-7b-v2.0
```

### Relative grading (pairwise)

The `llm_judge_relative` metric uses Prometheus relative grading to compare summaries head-to-head. It mirrors the human annotation tournament structure:

1. **Round 1**: A vs B, C vs D (2 LLM calls per dimension × 5 dimensions)
2. **Final**: winner of AB vs winner of CD (1 LLM call per dimension × 5 dimensions)

```bash
persona-eval compute-metrics annotations.zip \
    --metrics llm_judge_relative \
    --llm-provider together \
    --llm-model prometheus-eval/prometheus-7b-v2.0
```

This produces two outputs:

- **`pairwise_prefs.csv`** — Raw LLM preferences per comparison (`round1_ab`, `round1_cd`, `final`) with the winning label or `"tie"` for each sub-metric. These are used directly for **pairwise agreement** by matching each human preference against the LLM's corresponding comparison.
- **`metric_scores.csv`** — Tournament point scores per summary (0–3 scale) used for **rank correlation**, since Kendall tau and Spearman rho need per-summary numeric scores.

For pairwise agreement, the raw preferences are resolved as follows:
- **Round 1 (A vs B, C vs D)**: direct match against the LLM's round 1 result
- **Final winner vs final loser**: direct match against the LLM's final result
- **Final winner vs round loser (same bracket)**: resolved from the LLM's round 1 result
- **Final winner vs round loser (other bracket)**: resolved transitively (round 1 + final must both agree)

### Custom judge prompts

Both `llm_judge` and `llm_judge_relative` use configurable prompt templates. To customize:

1. Copy the default prompt from `src/persona_eval/prompts/llm_judge_default.txt` (absolute) or `llm_judge_relative.txt` (relative)
2. Edit it — use `{summary}`, `{source}`, `{dimension}`, and `{rubric}` placeholders (for absolute) or `{summary_a}`, `{summary_b}`, `{source}`, `{dimension}`, and `{rubric}` (for relative)
3. The LLM must return a `[RESULT]` tag: an integer 1-5 (absolute) or `A`/`B` (relative)
4. Pass your template with `--llm-prompt-file my_prompt.txt`

Rubrics for each dimension live in `src/persona_eval/prompts/rubrics/` and can be edited independently.

### Including the annotator's query

By default, the LLM judge prompts do not include the annotator's query. Use `--include-query` to add it:

```bash
# Absolute grading with query context
persona-eval compute-metrics annotations.zip \
    --metrics llm_judge \
    --llm-provider vllm --llm-model my-model \
    --include-query

# Relative grading with query + persona
persona-eval compute-metrics annotations.zip \
    --metrics llm_judge_relative \
    --llm-provider vllm --llm-model my-model \
    --persona --include-query
```

When enabled, the prompt includes a `Query: <text>` section so the LLM can consider what the annotator was looking for when evaluating the summary. This works with all prompt variants (absolute, relative, persona, non-persona).

Note: `llm_judge_annotator` always includes the query automatically since query relevance is its core evaluation criterion — `--include-query` is not needed for it.

### Persona-aware evaluation

The `--persona` flag enables persona-aware evaluation, where the LLM judges summaries from the perspective of each individual annotator using their profile (role, domain, information needs).

```bash
persona-eval compute-metrics annotations.zip \
    --metrics llm_judge \
    --llm-provider vllm \
    --llm-model prometheus-eval/prometheus-7b-v2.0 \
    --persona
```

When `--persona` is enabled:

- The **informativeness** rubric changes from general ("How useful is this summary?") to persona-specific ("How useful is this summary to this specific person?"), incorporating the annotator's role, domain, and information needs
- The prompt template includes the annotator's profile so the LLM evaluates from their perspective
- Scoring is **per-annotator**: the same summary may receive different scores for different annotators, since each has different information needs
- All other dimensions (relevance, coherence, consistency, fluency) are also evaluated from the annotator's perspective

Without `--persona`, the informativeness rubric evaluates general usefulness and scoring is deduplicated per (query, summary) as usual.

Annotator profiles are loaded from `users.json` in the annotations directory, with fields:
- `role`: the annotator's professional role
- `domain`: their area of expertise
- `info_needs`: what information they are looking for

### Annotator query-focused evaluation

The `llm_judge_annotator` metric is a pairwise metric that evaluates summaries based on whether they address the annotator's specific query, rather than general quality dimensions. It uses a single LLM call per pair (no per-dimension splitting), with the annotator's profile and query embedded in the prompt.

```bash
persona-eval compute-metrics annotations.zip \
    --metrics llm_judge_annotator \
    --llm-provider vllm \
    --llm-model prometheus-eval/prometheus-7b-v2.0 \
    --persona
```

This metric requires `--persona` since it needs the annotator's role, domain, information needs, and query. Unlike `llm_judge_relative` which evaluates five quality dimensions separately, `llm_judge_annotator` makes a single holistic judgment: which summary better addresses this person's query?

The prompt can be customized by editing `src/persona_eval/prompts/llm_judge_annotator.md` or by passing `--llm-prompt-file`. The template uses `{summary_a}`, `{summary_b}`, `{source}`, `{role}`, `{domain}`, `{info_needs}`, and `{query}` placeholders.

### FACTScore

The `factscore` metric implements the FACTScore algorithm (Min et al., 2023):

1. Decomposes the summary into atomic facts using the LLM
2. Verifies each fact against the source document
3. Returns the fraction of supported facts (`factscore`), along with `factscore_num_facts` and `factscore_num_supported`

## Adding a custom metric

Create a new file in `src/persona_eval/metrics/`:

```python
from persona_eval.metrics.base import BaseMetric, register_metric

@register_metric("my_metric")
class MyMetric(BaseMetric):
    def __init__(self, **kwargs):
        self._model = None

    @property
    def name(self) -> str:
        return "My Metric"

    @property
    def is_reference_free(self) -> bool:
        # True  -> receives concatenated abstracts
        # False -> receives concatenated titles
        return True

    def score(self, summary: str, source: str, persona_kwargs=None) -> dict[str, float]:
        # Your scoring logic here
        return {"my_metric_score": 0.5}
```

For LLM-backed metrics, inherit from `BaseLLMMetric` instead — it provides LLM client initialization, lazy loading, and response logging out of the box:

```python
from persona_eval.metrics.base import register_metric
from persona_eval.metrics.base_llm import BaseLLMMetric

@register_metric("my_llm_metric")
class MyLLMMetric(BaseLLMMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load your prompt templates here

    @property
    def name(self) -> str:
        return "My LLM Metric"

    def score(self, summary: str, source: str, persona_kwargs=None) -> dict[str, float]:
        self._load()  # lazily creates the LLM client
        response = self._client.generate("your prompt here")
        self._log_response("my_llm_metric", "prompt", response, parsed_result=None)
        return {"my_llm_score": 1.0}
```

Then add to `src/persona_eval/metrics/__init__.py`:

```python
from persona_eval.metrics import my_module  # noqa: F401
```

The metric is now available via `--metrics my_metric` on the CLI.

## How correlation is computed

**Pairwise agreement** (primary measure): For each human preference (e.g., "A is better than B"), check whether the metric agrees. For standard metrics, this compares numeric scores (`score(A) > score(B)`). For pairwise metrics (`llm_judge_relative`), the raw LLM preference is matched directly against the human preference — no score comparison involved. With `--strict-pairwise`, final-round comparisons automatically count as disagreement if the metric got any round 1 comparison wrong for that annotator/query.

**Rank correlation** (complementary measure): For each query, derive a human ranking from the preference data (final winner = rank 1, round winners = rank 2, rest = rank 3.5). Compute Kendall's tau-b and Spearman's rho against the metric's ranking. For pairwise metrics, tournament points are used for this step. Results are aggregated (mean, median, std) across all queries.

## Analyzing metric scores

The `scripts/analyze_scores.py` script provides a comprehensive analysis of the metric scores CSV produced by `compute-metrics` or `run-all`.

```bash
# Full analysis with plots
python scripts/analyze_scores.py metric_scores.csv -o analysis/

# Analyze specific metrics only
python scripts/analyze_scores.py metric_scores.csv --metrics rouge1_f bertscore_f supert llm_judge_overall

# With agreement/disagreement examples (requires annotations)
python scripts/analyze_scores.py metric_scores.csv --annotations annotations.zip -o analysis/

# More examples per metric
python scripts/analyze_scores.py metric_scores.csv --annotations annotations.zip --n-examples 5

# Stats only, no plots
python scripts/analyze_scores.py metric_scores.csv --no-plots
```

**Terminal output:**
- Descriptive statistics (mean, std, quartiles, missing %) per metric
- Mean scores per summary label (A/B/C/D)
- Discriminative power (mean within-query std — how well each metric differentiates between summaries for the same query)
- Inter-metric Spearman correlation matrix
- Agreement/disagreement examples (when `--annotations` is provided): concrete cases showing where each metric agrees or disagrees with human preferences, including the summaries and scores involved

**Saved plots** (to output directory):
- `distributions.png` — histograms with KDE for each metric
- `boxplots_by_label.png` — score distributions per summary label
- `discriminative_power.png` — bar chart of within-query variance
- `correlation_heatmap.png` — inter-metric Spearman correlation heatmap
- `per_query_heatmap.png` — normalized scores across queries and metrics

**Saved CSVs**: `summary_stats.csv`, `per_label_means.csv`, `discriminative_power.csv`, `inter_metric_correlation.csv`, `agreement_examples.csv` (when `--annotations` is provided)

## Project structure

```
src/persona_eval/
├── __init__.py
├── cli.py                    # CLI entry point
├── annotations.py            # Load annotation data, extract preferences
├── openalex.py               # Fetch paper abstracts/titles from OpenAlex
├── correlation.py            # Pairwise agreement + rank correlation
├── llm_client.py             # Shared LLM client (vLLM / TogetherAI)
├── response_logger.py        # JSONL logger for full LLM interactions
├── prompts/
│   ├── llm_judge_default.txt          # Prometheus absolute grading prompt
│   ├── llm_judge_persona.txt          # Persona-aware absolute grading prompt
│   ├── llm_judge_relative.txt         # Prometheus relative grading prompt
│   ├── llm_judge_relative_persona.txt # Persona-aware relative grading prompt
│   ├── factscore_extract.txt          # Atomic fact extraction prompt
│   ├── factscore_verify.txt           # Fact verification prompt
│   ├── llm_judge_annotator.md         # Annotator query-focused prompt
│   └── rubrics/                       # Per-dimension scoring rubrics
│       ├── relevance.txt
│       ├── coherence.txt
│       ├── consistency.txt
│       ├── fluency.txt
│       ├── informativeness.txt         # General informativeness
│       └── informativeness_persona.txt # Persona-aware informativeness
└── metrics/
    ├── __init__.py            # Registry imports
    ├── base.py                # BaseMetric ABC + @register_metric
    ├── base_llm.py            # BaseLLMMetric — shared base for LLM-backed metrics
    ├── summeval_metrics.py    # SUPERT, SummaQA, BLANC, BLEU, ChrF++, CIDEr, METEOR, DataStats
    ├── rouge_metrics.py       # ROUGE (via rouge-score)
    ├── bertscore_metric.py    # BERTScore (via bert-score)
    ├── syntactic_metric.py    # Syntactic complexity (via spacy)
    ├── llm_judge_metric.py    # LLM-as-judge absolute grading
    ├── llm_judge_relative_metric.py  # LLM-as-judge relative grading
    ├── llm_judge_annotator_metric.py # LLM-as-judge annotator query-focused
    └── factscore_metric.py    # FACTScore (via vLLM / TogetherAI)
scripts/
└── analyze_scores.py          # Metric scores analysis and visualisation
neither_thresholds.yaml        # Default thresholds for "neither" agreement
METRICS.md                     # Setup for metrics with external dependencies
```
