# Metrics

## Included metrics

These metrics work out of the box after `pip install -e .`:

| Metric | Type | Source text | Package |
|---|---|---|---|
| **ROUGE** | reference-based | titles | `rouge-score` |
| **BERTScore** | reference-based | titles | `bert-score` |
| **BLEU** | reference-based | titles | `summ-eval` (sacrebleu) |
| **ChrF++** | reference-based | titles | `summ-eval` (sacrebleu) |
| **CIDEr** | reference-based | titles | `summ-eval` |
| **METEOR** | reference-based | titles | `nltk` |
| **SUPERT** | reference-free | abstracts | `summ-eval` |
| **SummaQA** | reference-free | abstracts | `summ-eval` |
| **BLANC** | reference-free | abstracts | `summ-eval` |
| **DataStats** | reference-free | abstracts | `summ-eval` |
| **Syntactic** | reference-free | _(summary only)_ | `spacy` |

## Metrics requiring additional setup

The following `summ-eval` metrics are not included by default because they
require external dependencies. Each section describes the dependency and how to
install it. Once the dependency is satisfied, adding the metric wrapper is
straightforward (see [Adding a metric](#adding-a-metric) below).

---

### MoverScore

**Issue:** The `moverscore_v2` module imports a transformers code path that
triggers `from pkg_resources import packaging`, which is removed in newer
setuptools. Additionally, MoverScore v2 hard-codes `device='cuda:0'` and
requires a CUDA-capable GPU.

**Fix:**

```bash
# Ensure setuptools ships pkg_resources with packaging
pip install "setuptools<70"

# A CUDA-capable GPU is required for MoverScore v2
# For CPU-only, use version=1 (uses pytorch-pretrained-bert instead):
#   MoverScoreMetric(version=1)
```

---

### SentenceMovers

**Dependency:** GloVe word vectors via spacy, plus the `wmd` (Word Mover's
Distance) package.

**Setup:**

```bash
pip install wmd
python -m spacy download en_core_web_md   # medium model includes GloVe vectors
```

The `en_core_web_sm` model does *not* include word vectors. You need
`en_core_web_md` or `en_core_web_lg`.

---

### ROUGE-WE (ROUGE with Word Embeddings)

**Dependency:** Dependency-based word embeddings file (`deps.words`).

**Setup:** The file is auto-downloaded on first use to
`<summ_eval>/embeddings/deps.words`. If auto-download fails:

```bash
# Manual download
SUMM_EVAL_DIR=$(python -c "import summ_eval, os; print(os.path.dirname(summ_eval.__file__))")
mkdir -p "$SUMM_EVAL_DIR/embeddings"
curl -L http://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2 \
  | bunzip2 > "$SUMM_EVAL_DIR/embeddings/deps.words"
```

Source: [Dependency-Based Word Embeddings](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/)

---

### S3 (Semantic Structure Similarity)

**Dependencies:** The same `deps.words` embeddings as ROUGE-WE (see above),
plus trained pyramid and responsiveness models.

**Setup:**

```bash
# 1. Install deps.words (same as ROUGE-WE above)
SUMM_EVAL_DIR=$(python -c "import summ_eval, os; print(os.path.dirname(summ_eval.__file__))")
mkdir -p "$SUMM_EVAL_DIR/embeddings"
curl -L http://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2 \
  | bunzip2 > "$SUMM_EVAL_DIR/embeddings/deps.words"

# 2. Download S3 models
mkdir -p "$SUMM_EVAL_DIR/models/en"
# Clone the S3 repo and copy the model pickle files:
git clone https://github.com/UKPLab/emnlp-ws-2017-s3.git /tmp/s3-models
cp /tmp/s3-models/models/en/*pyr* "$SUMM_EVAL_DIR/models/en/"
cp /tmp/s3-models/models/en/*resp* "$SUMM_EVAL_DIR/models/en/"
rm -rf /tmp/s3-models
```

The model folder must contain two pickle files: one with `pyr` in the name
(pyramid score model) and one with `resp` (responsiveness model).

---

## Adding a metric

To add any of the above metrics once its dependencies are installed, create a
wrapper class in `src/persona_eval/metrics/summeval_metrics.py`:

```python
@register_metric("my_metric")
class MyMetric(BaseMetric):
    def __init__(self, **kwargs):
        self._metric = None

    @property
    def name(self) -> str:
        return "My Metric"

    @property
    def is_reference_free(self) -> bool:
        # True  -> receives concatenated abstracts
        # False -> receives concatenated titles
        return False

    def _load(self):
        if self._metric is None:
            from summ_eval.my_metric import MyMetric as _M
            self._metric = _M()

    def score(self, summary: str, source: str) -> dict[str, float]:
        self._load()
        result = self._metric.evaluate_example(summary, source)
        return {k: float(v) for k, v in result.items()}
```

Then import the module in `src/persona_eval/metrics/__init__.py` to register it.
