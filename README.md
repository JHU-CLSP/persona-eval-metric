# persona-eval-metric

Evaluate how well automatic summarization metrics correlate with human pairwise preferences. Built for scientific paper summarization with source documents from [OpenAlex](https://openalex.org/).

## Overview

This pipeline:

1. **Loads human annotation data** — pairwise preferences over AI-generated summaries (A vs B, C vs D, final winner)
2. **Fetches source documents** from OpenAlex — paper abstracts (for reference-free metrics) and titles (for reference-based metrics)
3. **Computes automatic metrics** — 13 metrics including LLM-as-judge and FACTScore, plus traditional metrics from summ-eval, rouge-score, bert-score, nltk, and spacy
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
- `results/metric_scores.csv` — per-(query, summary) metric scores
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
| `llm_judge` | LLM Judge | reference-free | abstracts |
| `factscore` | FACTScore | reference-free | abstracts |

**Reference-based** metrics compare the summary against concatenated paper titles.
**Reference-free** metrics compare the summary against concatenated paper abstracts.
**Syntactic** only analyzes the summary text itself (L2 Syntactic Complexity indices).
**LLM-based** metrics (`llm_judge`, `factscore`) use an LLM to evaluate summaries — see [LLM-based metrics](#llm-based-metrics) below.

Additional summ-eval metrics (MoverScore, SentenceMovers, ROUGE-WE, S3) require external dependencies — see [METRICS.md](METRICS.md) for setup instructions.

## LLM-based metrics

The `llm_judge` and `factscore` metrics require an LLM backend. Both local [vLLM](https://docs.vllm.ai/) and [TogetherAI](https://www.together.ai/) are supported via their OpenAI-compatible APIs.

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

### Custom judge prompts

The `llm_judge` metric uses a configurable prompt template. The default prompt rates summaries on relevance, coherence, consistency, and fluency (1-5 scale). To customize:

1. Copy the default prompt from `src/persona_eval/prompts/llm_judge_default.txt`
2. Edit it — use `{summary}` and `{source}` placeholders
3. The LLM must return a JSON object with numeric scores
4. Pass your template with `--llm-prompt-file my_prompt.txt`

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

    def score(self, summary: str, source: str) -> dict[str, float]:
        # Your scoring logic here
        return {"my_metric_score": 0.5}
```

Then add to `src/persona_eval/metrics/__init__.py`:

```python
from persona_eval.metrics import my_module  # noqa: F401
```

The metric is now available via `--metrics my_metric` on the CLI.

## How correlation is computed

**Pairwise agreement** (primary measure): For each human preference (e.g., "A is better than B"), check whether the metric assigns a higher score to A than B. Reports the fraction of agreements across all comparisons.

**Rank correlation** (complementary measure): For each query, derive a human ranking from the preference data (final winner = rank 1, round winners = rank 2, rest = rank 3.5). Compute Kendall's tau-b and Spearman's rho against the metric's ranking. Results are aggregated (mean, median, std) across all queries.

## Project structure

```
src/persona_eval/
├── __init__.py
├── cli.py                    # CLI entry point
├── annotations.py            # Load annotation data, extract preferences
├── openalex.py               # Fetch paper abstracts/titles from OpenAlex
├── correlation.py            # Pairwise agreement + rank correlation
├── llm_client.py             # Shared LLM client (vLLM / TogetherAI)
├── prompts/
│   ├── llm_judge_default.txt  # Default LLM judge prompt template
│   ├── factscore_extract.txt  # Atomic fact extraction prompt
│   └── factscore_verify.txt   # Fact verification prompt
└── metrics/
    ├── __init__.py            # Registry imports
    ├── base.py                # BaseMetric ABC + @register_metric
    ├── summeval_metrics.py    # SUPERT, SummaQA, BLANC, BLEU, ChrF++, CIDEr, METEOR, DataStats
    ├── rouge_metrics.py       # ROUGE (via rouge-score)
    ├── bertscore_metric.py    # BERTScore (via bert-score)
    ├── syntactic_metric.py    # Syntactic complexity (via spacy)
    ├── llm_judge_metric.py    # LLM-as-judge (via vLLM / TogetherAI)
    └── factscore_metric.py    # FACTScore (via vLLM / TogetherAI)
neither_thresholds.yaml        # Default thresholds for "neither" agreement
METRICS.md                     # Setup for metrics with external dependencies
```
