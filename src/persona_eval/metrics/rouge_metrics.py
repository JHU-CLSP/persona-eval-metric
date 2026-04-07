"""ROUGE metric using Google's rouge-score package.

Uses rouge-score directly rather than summ-eval's RougeMetric which requires
perl and pyrouge setup.
"""

from __future__ import annotations

import logging

from persona_eval.metrics.base import BaseMetric, register_metric

logger = logging.getLogger(__name__)


@register_metric("rouge")
class RougeMetric(BaseMetric):
    """ROUGE-1/2/L using source text as reference."""

    def __init__(self, **kwargs):
        self._scorer = None

    @property
    def name(self) -> str:
        return "ROUGE"

    @property
    def is_reference_free(self) -> bool:
        return False

    def _load(self):
        if self._scorer is None:
            from rouge_score import rouge_scorer

            self._scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )

    def score(self, summary: str, source: str) -> dict[str, float]:
        self._load()
        result = self._scorer.score(source, summary)
        return {
            "rouge1_f": result["rouge1"].fmeasure,
            "rouge1_p": result["rouge1"].precision,
            "rouge1_r": result["rouge1"].recall,
            "rouge2_f": result["rouge2"].fmeasure,
            "rouge2_p": result["rouge2"].precision,
            "rouge2_r": result["rouge2"].recall,
            "rougeL_f": result["rougeL"].fmeasure,
            "rougeL_p": result["rougeL"].precision,
            "rougeL_r": result["rougeL"].recall,
        }
