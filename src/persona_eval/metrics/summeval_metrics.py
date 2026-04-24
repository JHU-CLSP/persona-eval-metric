"""Metric wrappers for the summ-eval package.

Eight wrappers follow the same pattern (lazy import → call
``evaluate_example`` → pluck one or more fields), so they're defined
declaratively in ``SUMMEVAL_METRICS`` and instantiated through a single
``SummEvalWrapper`` class. METEOR is the exception (uses nltk, not the
Java-based summ-eval implementation) and keeps its own class.
"""

from __future__ import annotations

import importlib
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
        pythonpath = os.environ.get("PYTHONPATH", "")
        if se_dir not in pythonpath:
            os.environ["PYTHONPATH"] = se_dir + os.pathsep + pythonpath if pythonpath else se_dir
    except ImportError:
        pass


def _load_class(import_path: str):
    """Load ``module.path:ClassName`` lazily."""
    module_name, class_name = import_path.split(":")
    return getattr(importlib.import_module(module_name), class_name)


class SummEvalWrapper(BaseMetric):
    """Generic wrapper around ``summ_eval`` metrics driven by a config dict."""

    def __init__(self, config: dict, device: str = "cpu", **kwargs):
        self._config = config
        self._device = device
        self._metric = None

    @property
    def name(self) -> str:
        return self._config["name"]

    @property
    def is_reference_free(self) -> bool:
        return self._config.get("reference_free", True)

    def _load(self):
        if self._metric is not None:
            return
        setup = self._config.get("setup")
        if setup is not None:
            setup()
        cls = _load_class(self._config["import_path"])
        init_kwargs_fn = self._config.get("init_kwargs")
        init_kwargs = init_kwargs_fn(self._device) if init_kwargs_fn else {}
        # Handle SUPERT's implicit device-via-torch idiom.
        on_load = self._config.get("on_load")
        if on_load is not None:
            on_load(self._device)
        self._metric = cls(**init_kwargs)

    def score(self, summary: str, source: str, persona_kwargs=None) -> dict[str, float]:
        self._load()
        result = self._metric.evaluate_example(summary, source)
        output_fields = self._config.get("output_fields")
        if output_fields is None:
            # DataStats returns a dict of floats; pass through as-is.
            return {k: float(v) for k, v in result.items()}
        return {out_key: float(result.get(src_key, 0.0))
                for out_key, src_key in output_fields.items()}


def _supert_on_load(device: str):
    import torch
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.set_device(device if device != "cuda" else 0)


def _summaqa_init_kwargs(device: str) -> dict:
    import transformers
    transformers.logging.set_verbosity_error()
    return {"use_gpu": device != "cpu"}


def _blanc_init_kwargs(device: str) -> dict:
    return {"device": device}


SUMMEVAL_METRICS = [
    {
        "key": "supert",
        "name": "SUPERT",
        "import_path": "summ_eval.supert_metric:SupertMetric",
        "output_fields": {"supert": "supert"},
        "setup": _ensure_summeval_path,
        "on_load": _supert_on_load,
    },
    {
        "key": "summaqa",
        "name": "SummaQA",
        "import_path": "summ_eval.summa_qa_metric:SummaQAMetric",
        "output_fields": {
            "summaqa_avg_prob": "summaqa_avg_prob",
            "summaqa_avg_fscore": "summaqa_avg_fscore",
        },
        "init_kwargs": _summaqa_init_kwargs,
    },
    {
        "key": "blanc",
        "name": "BLANC",
        "import_path": "summ_eval.blanc_metric:BlancMetric",
        "output_fields": {"blanc": "blanc"},
        "init_kwargs": _blanc_init_kwargs,
    },
    {
        "key": "chrf",
        "name": "ChrF++",
        "import_path": "summ_eval.chrfpp_metric:ChrfppMetric",
        "output_fields": {"chrf": "chrf"},
        "reference_free": False,
    },
    {
        "key": "bleu",
        "name": "BLEU",
        "import_path": "summ_eval.bleu_metric:BleuMetric",
        "output_fields": {"bleu": "bleu"},
        "reference_free": False,
    },
    {
        "key": "cider",
        "name": "CIDEr",
        "import_path": "summ_eval.cider_metric:CiderMetric",
        "output_fields": {"cider": "cider"},
        "reference_free": False,
    },
    {
        "key": "data_stats",
        "name": "DataStats",
        "import_path": "summ_eval.data_stats_metric:DataStatsMetric",
        # No output_fields: pass the full dict through.
    },
]


def _make_wrapper(config: dict) -> type[BaseMetric]:
    """Build a tiny BaseMetric subclass bound to one config entry.

    A concrete subclass is used (rather than passing the config at
    construction time) so that ``type(self).__name__`` used in
    ``cache_config`` remains distinct per metric.
    """
    name = f"{config['key'].title().replace('_', '')}Metric"

    class _Wrapper(SummEvalWrapper):
        def __init__(self, device: str = "cpu", **kwargs):
            super().__init__(config, device=device, **kwargs)

    _Wrapper.__name__ = name
    _Wrapper.__qualname__ = name
    return _Wrapper


for _cfg in SUMMEVAL_METRICS:
    register_metric(_cfg["key"])(_make_wrapper(_cfg))


# METEOR uses nltk's pure-Python implementation (no Java dep), so it's
# its own class rather than a summ-eval wrapper entry.
@register_metric("meteor")
class MeteorMetric(BaseMetric):
    """METEOR: alignment-based metric using synonyms and stemming (via nltk)."""

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
        return {"meteor": float(meteor_score([reference_tokens], hypothesis_tokens))}
