"""Shared, lightweight result/report containers.

Each pipeline step returns a :class:`StepReport` describing *what it did* (for
audit trails) and any data-health *warnings* it raised. These are mutable by
design — they are scratch accumulators, not part of the functional data flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class StepReport:
    """Structured outcome of a single transformation step.

    Attributes
    ----------
    step:
        Stable identifier, e.g. ``"impute"`` or ``"downcast"``.
    actions:
        Ordered, human-readable descriptions of changes applied.
    metrics:
        Machine-readable key/values (counts, bytes, ratios) for the report.
    warnings:
        Data-health concerns surfaced to the user (e.g. high missingness).
    """

    step: str
    actions: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def act(self, message: str) -> None:
        """Record an applied action."""
        self.actions.append(message)

    def warn(self, message: str) -> None:
        """Record a data-health warning."""
        self.warnings.append(message)

    def measure(self, key: str, value: Any) -> None:
        """Record a metric."""
        self.metrics[key] = value
