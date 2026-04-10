"""Metric wrappers using the summ-eval package."""

from __future__ import annotations

import logging
import os
import sys

from persona_eval.metrics.base import BaseMetric, register_metric

logger = logging.getLogger(__name__)


def _ensure_summeval_path():
    """Add summ_eval to PYTHONPATH if needed (required by SUPERT internals)."""
    try:
        import summ_eval

        se_dir = os.path.dirname(summ_eval.__file__)
        if se_dir not in sys.path:
            sys.path.insert(0, se_dir)
        # SUPERT also checks the PYTHONPATH env var directly
        pythonpath = os.environ.get("PYTHONPATH", "")
        if se_dir not in pythonpath:
            os.environ["PYTHONPATH"] = se_dir + os.pathsep + pythonpath if pythonpath else se_dir
    except ImportError:
        pass


@register_metric("supert")
class SupertMetric(BaseMetric):
    """SUPERT: reference-free multi-document summarization metric (via summ-eval)."""

    def __init__(self, device: str = "cpu", **kwargs):
        self._metric = None
        self._device = device

    @property
    def name(self) -> str:
        return "SUPERT"

    def _load(self):
        if self._metric is None:
            _ensure_summeval_path()
            import torch

            # summ-eval SUPERT uses the default torch device internally
            if self._device != "cpu" and torch.cuda.is_available():
                torch.cuda.set_device(self._device if self._device != "cuda" else 0)

            from summ_eval.supert_metric import SupertMetric as _Supert

            self._metric = _Supert()

    def score(self, summary: str, source: str, persona_kwargs=None) -> dict[str, float]:
        self._load()
        result = self._metric.evaluate_example(summary, source)
        return {"supert": float(result.get("supert", 0.0))}


@register_metric("summaqa")
class SummaQAMetric(BaseMetric):
    """SummaQA: QA-based reference-free summarization metric (via summ-eval)."""

    def __init__(self, device: str = "cpu", **kwargs):
        self._metric = None
        self._device = device

    @property
    def name(self) -> str:
        return "SummaQA"

    def _load(self):
        if self._metric is None:
            from summ_eval.summa_qa_metric import SummaQAMetric as _SummaQA
            import transformers
            transformers.logging.set_verbosity_error()

            self._metric = _SummaQA(use_gpu=self._device != "cpu")

    def score(self, summary: str, source: str, persona_kwargs=None) -> dict[str, float]:
        self._load()
        result = self._metric.evaluate_example(summary, source)
        return {
            "summaqa_avg_prob": float(result.get("summaqa_avg_prob", 0.0)),
            "summaqa_avg_fscore": float(result.get("summaqa_avg_fscore", 0.0)),
        }


@register_metric("blanc")
class BlancMetric(BaseMetric):
    """BLANC: reference-free metric based on language model filling (via summ-eval)."""

    def __init__(self, device: str = "cpu", **kwargs):
        self._metric = None
        self._device = device

    @property
    def name(self) -> str:
        return "BLANC"

    def _load(self):
        if self._metric is None:
            from summ_eval.blanc_metric import BlancMetric as _Blanc

            self._metric = _Blanc(device=self._device)

    def score(self, summary: str, source: str, persona_kwargs=None) -> dict[str, float]:
        self._load()
        result = self._metric.evaluate_example(summary, source)
        return {"blanc": float(result.get("blanc", 0.0))}


# ---------------------------------------------------------------------------
# Reference-based metrics
# ---------------------------------------------------------------------------


@register_metric("chrf")
class ChrfMetric(BaseMetric):
    """ChrF++: character n-gram F-score metric (via summ-eval / sacrebleu)."""

    def __init__(self, **kwargs):
        self._metric = None

    @property
    def name(self) -> str:
        return "ChrF++"

    @property
    def is_reference_free(self) -> bool:
        return False

    def _load(self):
        if self._metric is None:
            from summ_eval.chrfpp_metric import ChrfppMetric as _Chrf

            self._metric = _Chrf()

    def score(self, summary: str, source: str, persona_kwargs=None) -> dict[str, float]:
        self._load()
        result = self._metric.evaluate_example(summary, source)
        return {"chrf": float(result.get("chrf", 0.0))}


@register_metric("meteor")
class MeteorMetric(BaseMetric):
    """METEOR: alignment-based metric using synonyms and stemming (via nltk).

    Uses nltk's pure-Python METEOR implementation instead of the Java-based
    Meteor 1.5 JAR required by summ-eval, so no Java installation is needed.
    """

    def __init__(self, **kwargs):
        self._loaded = False

    @property
    def name(self) -> str:
        return "METEOR"

    @property
    def is_reference_free(self) -> bool:
        return False

    def _load(self):
        if not self._loaded:
            import nltk

            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            self._loaded = True

    def score(self, summary: str, source: str, persona_kwargs=None) -> dict[str, float]:
        self._load()
        from nltk.translate.meteor_score import meteor_score

        reference_tokens = source.split()
        hypothesis_tokens = summary.split()
        score = meteor_score([reference_tokens], hypothesis_tokens)
        return {"meteor": float(score)}


@register_metric("bleu")
class BleuMetric(BaseMetric):
    """BLEU: n-gram precision metric (via summ-eval / sacrebleu)."""

    def __init__(self, **kwargs):
        self._metric = None

    @property
    def name(self) -> str:
        return "BLEU"

    @property
    def is_reference_free(self) -> bool:
        return False

    def _load(self):
        if self._metric is None:
            from summ_eval.bleu_metric import BleuMetric as _Bleu

            self._metric = _Bleu()

    def score(self, summary: str, source: str, persona_kwargs=None) -> dict[str, float]:
        self._load()
        result = self._metric.evaluate_example(summary, source)
        return {"bleu": float(result.get("bleu", 0.0))}


@register_metric("cider")
class CiderMetric(BaseMetric):
    """CIDEr: consensus-based evaluation metric (via summ-eval)."""

    def __init__(self, **kwargs):
        self._metric = None

    @property
    def name(self) -> str:
        return "CIDEr"

    @property
    def is_reference_free(self) -> bool:
        return False

    def _load(self):
        if self._metric is None:
            from summ_eval.cider_metric import CiderMetric as _Cider

            self._metric = _Cider()

    def score(self, summary: str, source: str, persona_kwargs=None) -> dict[str, float]:
        self._load()
        result = self._metric.evaluate_example(summary, source)
        return {"cider": float(result.get("cider", 0.0))}


# ---------------------------------------------------------------------------
# Reference-free metrics
# ---------------------------------------------------------------------------


@register_metric("data_stats")
class DataStatsMetric(BaseMetric):
    """DataStats: extractive statistics — coverage, density, compression, novelty (via summ-eval)."""

    def __init__(self, **kwargs):
        self._metric = None

    @property
    def name(self) -> str:
        return "DataStats"

    def _load(self):
        if self._metric is None:
            from summ_eval.data_stats_metric import DataStatsMetric as _DS

            self._metric = _DS()

    def score(self, summary: str, source: str, persona_kwargs=None) -> dict[str, float]:
        self._load()
        result = self._metric.evaluate_example(summary, source)
        return {k: float(v) for k, v in result.items()}
