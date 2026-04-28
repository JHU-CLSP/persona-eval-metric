# persona-eval-metric

Evaluate how well automatic summarization metrics correlate with human pairwise preferences, and test metric robustness under controlled perturbations. Built for scientific paper summarization with source documents from [OpenAlex](https://openalex.org/).

## Features

- **Annotation analysis** -- compute 15+ automatic metrics on summaries and measure correlation with human judgments (pairwise agreement, Kendall tau, Spearman rho)
- **Robustness testing** -- systematically perturb summaries and verify that metrics respond as expected (scores drop with noise, stay stable under paraphrasing, etc.)
- **LLM-as-judge** -- absolute grading, relative (pairwise) grading, annotator-focused evaluation, and FACTScore, via vLLM or TogetherAI
- **Persona-aware evaluation** -- score summaries from the perspective of individual annotators using their role, domain, and information needs
- **Result caching** -- metric outputs and the scores CSV are cached on disk by default, so re-runs (including after a crash or with a new metric added) skip already-computed work

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

### LLM backend setup

LLM-based metrics and robustness tests that use LLM perturbations require one of: a local [vLLM](https://docs.vllm.ai/) server, a [TogetherAI](https://www.together.ai/) API key, or an [Anthropic](https://www.anthropic.com/) API key.

**vLLM (local):**

```bash
# Start a vLLM server in a separate terminal
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8000
```

**TogetherAI:**

```bash
export TOGETHER_API_KEY=your-key-here
```

**Anthropic:**

```bash
export ANTHROPIC_API_KEY=your-key-here
# then pass --llm-provider anthropic --llm-model claude-sonnet-4-6
```

## Quick start

### Annotation analysis

```bash
# Run the full pipeline with ROUGE only (fast, no model downloads)
persona-eval run-all path/to/annotations.zip --metrics rouge --output-dir results/

# Run with all metrics
persona-eval run-all path/to/annotations.zip --output-dir results/

# Include LLM-based metrics
persona-eval run-all path/to/annotations.zip \
    --metrics rouge bertscore llm_judge factscore \
    --llm-provider vllm --llm-model meta-llama/Meta-Llama-3-8B-Instruct \
    --output-dir results/
```

See [docs/annotation-analysis.md](docs/annotation-analysis.md) for full CLI reference, metric descriptions, persona-aware evaluation, custom prompts, and correlation methodology.

### Robustness testing

```bash
# Run all robustness tests with ROUGE
persona-eval robustness path/to/annotations.zip \
    --metrics rouge --output-dir robustness_results/

# Run specific tests with LLM-based perturbations (vLLM, default)
persona-eval robustness path/to/annotations.zip \
    --tests distractor incremental lengthen shorten audience \
    --metrics rouge bertscore \
    --llm-model meta-llama/Meta-Llama-3-8B-Instruct \
    --output-dir robustness_results/

# Or use TogetherAI as the LLM provider
persona-eval robustness path/to/annotations.zip \
    --tests lengthen shorten audience \
    --metrics rouge bertscore \
    --llm-provider together --llm-model meta-llama/Meta-Llama-3-8B-Instruct \
    --output-dir robustness_results/

# Use different models for perturbation generation vs evaluation
persona-eval robustness path/to/annotations.zip \
    --tests lengthen shorten audience --metrics rouge llm_judge \
    --llm-provider together --llm-model meta-llama/Meta-Llama-3-70B-Instruct \
    --perturb-provider vllm --perturb-model meta-llama/Meta-Llama-3-8B-Instruct \
    --output-dir robustness_results/

# Subsample for faster iteration
persona-eval robustness path/to/annotations.zip \
    --tests distractor incremental --metrics rouge \
    --num-samples 20 --seed 42 \
    --output-dir robustness_results/

# Two-step workflow: generate perturbations first, evaluate later
# Step 1: Generate perturbations only (no metrics computed)
persona-eval robustness-generate --dataset arxiv \
    --tests distractor incremental lengthen shorten \
    --llm-model meta-llama/Meta-Llama-3-8B-Instruct \
    --num-samples 50 --output-dir robustness_results/

# Step 2: Evaluate metrics on saved perturbations (can re-run with different metrics)
persona-eval robustness-eval \
    --perturbations-dir robustness_results/perturbations \
    --metrics rouge bertscore --output-dir robustness_results/
```

See [docs/robustness-testing.md](docs/robustness-testing.md) for test descriptions, expected behaviors, analysis methodology, and extending with custom tests.

## Available metrics

| Metric | Type | Notes |
|---|---|---|
| `rouge` | reference-based | ROUGE-1/2/L |
| `bertscore` | reference-based | Contextual embeddings |
| `bleu`, `chrf`, `cider`, `meteor` | reference-based | Via summ-eval |
| `supert`, `summaqa`, `blanc` | reference-free | Via summ-eval |
| `data_stats` | reference-free | Via summ-eval |
| `syntactic` | reference-free | L2 Syntactic Complexity (spacy) |
| `llm_judge` | reference-free | Absolute 1-5 grading on 5 dimensions |
| `llm_judge_relative` | reference-free, pairwise | Head-to-head comparison |
| `llm_judge_annotator` | reference-free, pairwise, persona | Query-focused comparison |
| `factscore` | reference-free | Atomic fact verification |

Additional metrics (MoverScore, SentenceMovers, ROUGE-WE, S3) require external setup -- see [METRICS.md](METRICS.md).

## Project structure

```
src/persona_eval/
├── cli/                      # CLI package: one module per subcommand
│   ├── main.py               # argparse dispatcher (entry point)
│   ├── _args.py              # Shared argparse fragments
│   ├── _common.py            # Shared helpers (data loading, caches, etc.)
│   └── <command>.py          # One file per subcommand
├── core/
│   ├── cache.py              # JsonFileCache base + hashing helpers
│   └── pipeline.py           # Metric orchestration (build_tasks, compute_metric_scores, ...)
├── annotations.py            # Load annotation data, extract preferences
├── openalex.py               # Fetch paper abstracts/titles from OpenAlex
├── correlation.py            # Pairwise agreement + rank correlation
├── llm_client.py             # LLM client (vLLM / TogetherAI) + ResponseLogger
├── prompts/                  # LLM prompt templates and rubrics
├── metrics/                  # Metric implementations + registry
│   ├── base.py               # BaseMetric ABC + @register_metric
│   ├── base_llm.py           # BaseLLMMetric, BaseDimensionalLLMMetric, MultiStepLLMMetric
│   ├── cache.py              # MetricCache (subclass of JsonFileCache)
│   └── ...                   # Individual metric modules
└── robustness/               # Robustness testing framework
    ├── dataset.py            # SummarizationSample + dataset adapters
    ├── perturbations.py      # 5 perturbation tests
    ├── cache.py              # PerturbationCache (subclass of JsonFileCache)
    ├── runner.py             # Test orchestrator
    └── analysis.py           # Effect analysis + reporting
scripts/
└── analyze_scores.py         # Metric score analysis and visualisation
```

See [CLAUDE.md](CLAUDE.md) for architecture notes, extension points, and cache layout details.
