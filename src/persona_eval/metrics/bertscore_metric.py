"""BERTScore metric using the bert-score package directly.

Uses bert-score directly rather than summ-eval's BertScoreMetric which has
unpacking bugs with newer bert-score versions.
"""

from __future__ import annotations

import logging

from persona_eval.metrics.base import BaseMetric, register_metric

logger = logging.getLogger(__name__)


@register_metric("bertscore")
class BertScoreMetric(BaseMetric):
    """BERTScore: semantic similarity using contextual embeddings."""

    def __init__(self, model_type: str = "bert-base-uncased", **kwargs):
        self._scorer = None
        self._model_type = model_type

    @property
    def name(self) -> str:
        return "BERTScore"

    @property
    def is_reference_free(self) -> bool:
        return False

    def _load(self):
        if self._scorer is None:
            import bert_score

            self._scorer = bert_score

    def score(self, summary: str, source: str) -> dict[str, float]:
        self._load()
        P, R, F = self._scorer.score(
            [summary],
            [source],
            model_type=self._model_type,
            verbose=False,
        )
        return {
            "bertscore_p": float(P[0]),
            "bertscore_r": float(R[0]),
            "bertscore_f": float(F[0]),
        }

    def score_batch(
        self, summaries: list[str], sources: list[str]
    ) -> list[dict[str, float]]:
        self._load()
        P, R, F = self._scorer.score(
            summaries,
            sources,
            model_type=self._model_type,
            verbose=False,
        )
        return [
            {
                "bertscore_p": float(P[i]),
                "bertscore_r": float(R[i]),
                "bertscore_f": float(F[i]),
            }
            for i in range(len(summaries))
        ]
