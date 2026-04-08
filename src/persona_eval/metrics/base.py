"""Base metric class and registry for extensible metric framework."""

from __future__ import annotations

from abc import ABC, abstractmethod

_METRIC_REGISTRY: dict[str, type[BaseMetric]] = {}


def register_metric(name: str):
    """Decorator to register a metric class in the global registry."""

    def decorator(cls: type[BaseMetric]) -> type[BaseMetric]:
        _METRIC_REGISTRY[name] = cls
        return cls

    return decorator


def get_metric(name: str, **kwargs) -> BaseMetric:
    """Instantiate a registered metric by name."""
    if name not in _METRIC_REGISTRY:
        available = ", ".join(sorted(_METRIC_REGISTRY))
        raise ValueError(f"Unknown metric '{name}'. Available: {available}")
    return _METRIC_REGISTRY[name](**kwargs)


def list_metrics() -> list[str]:
    """Return sorted list of registered metric names."""
    return sorted(_METRIC_REGISTRY)


class BaseMetric(ABC):
    """Abstract base class for summarization evaluation metrics.

    To add a custom metric:
        1. Subclass BaseMetric
        2. Implement `name` property and `score()` method
        3. Decorate with @register_metric("your_metric_name")
        4. Import the module in metrics/__init__.py
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this metric."""

    @property
    def is_reference_free(self) -> bool:
        """Whether this metric can operate without a reference summary."""
        return True

    @property
    def is_pairwise(self) -> bool:
        """Whether this metric compares pairs of summaries (relative grading).

        Pairwise metrics implement ``score_pair()`` instead of ``score()``.
        The pipeline computes win-rates across all pairs within a query to
        produce a per-summary score.
        """
        return False

    @abstractmethod
    def score(self, summary: str, source: str) -> dict[str, float]:
        """Score a single summary.

        Args:
            summary: The generated summary text.
            source: The source document text (concatenated abstracts).

        Returns:
            Dict mapping sub-metric names to float scores.
            E.g., {"rouge1_f": 0.45, "rouge1_p": 0.5, "rouge1_r": 0.41}
        """

    def score_pair(
        self, summary_a: str, summary_b: str, source: str
    ) -> dict[str, str]:
        """Compare two summaries and return a preference per sub-metric.

        Only used when ``is_pairwise`` is True. Override this method for
        pairwise (relative grading) metrics.

        Args:
            summary_a: First summary text.
            summary_b: Second summary text.
            source: The source document text.

        Returns:
            Dict mapping sub-metric names to preference strings:
            ``"A"`` if summary_a is better, ``"B"`` if summary_b is better,
            or ``"tie"`` if they are equal.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support pairwise scoring"
        )

    def score_batch(
        self, summaries: list[str], sources: list[str]
    ) -> list[dict[str, float]]:
        """Score a batch of summaries. Override for optimized batch processing."""
        return [self.score(s, src) for s, src in zip(summaries, sources)]
