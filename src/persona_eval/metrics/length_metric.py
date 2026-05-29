"""Length-based pairwise metric for comparing summary lengths."""

from __future__ import annotations

from persona_eval.metrics.base import BaseMetric, register_metric


@register_metric("length")
class LengthMetric(BaseMetric):
    """Pairwise metric that prefers the shorter summary.

    Compares the word counts of two summaries. The preference is for the
    shorter summary. If the difference is small (within threshold), returns
    "tie" to indicate similar length.
    """

    @property
    def name(self) -> str:
        return "Length"

    @property
    def is_pairwise(self) -> bool:
        return True

    @property
    def is_reference_free(self) -> bool:
        return True

    def score(self, summary: str, source: str, persona_kwargs=None) -> dict[str, float]:
        """Not used for pairwise metrics, but required by base class."""
        length = len(summary.split())
        return {"length": float(length)}

    def score_pair(
        self,
        summary_a: str,
        summary_b: str,
        source: str,
        persona_kwargs=None,
    ) -> dict[str, str]:
        """Compare lengths of two summaries.

        Args:
            summary_a: First summary text.
            summary_b: Second summary text.
            source: The source document text (unused).
            persona_kwargs: Optional annotator profile (unused).

        Returns:
            Dict with "length" key mapping to "A" (summary_a shorter),
            "B" (summary_b shorter), or "tie" (similar length).
        """
        len_a = len(summary_a.split())
        len_b = len(summary_b.split())

        # Threshold for considering lengths as "similar" (in words)
        # Using 5 words to match summary_length threshold from neither_thresholds.yaml
        threshold = 5

        length_diff = abs(len_a - len_b)

        if length_diff < threshold:
            return {"length": "tie"}
        elif len_a < len_b:
            return {"length": "A"}
        else:
            return {"length": "B"}
