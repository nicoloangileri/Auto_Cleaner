"""Advanced, research-grade analysis layer.

Bundles normality testing, automatic distribution-normalising transforms,
multicollinearity diagnostics (VIF + PCA) and target-aware feature relevance
into a single :class:`AdvancedAnalysis` result produced by :func:`run_advanced`.
Every component degrades gracefully if its optional backend (scipy / sklearn)
is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from auto_cleaner.analyze.feature_relevance import RelevanceResult, feature_relevance
from auto_cleaner.analyze.multicollinearity import (
    PCAResult,
    VIFResult,
    pca_summary,
    vif_scores,
)
from auto_cleaner.analyze.tests import NormalityResult, normality_tests
from auto_cleaner.analyze.transforms import (
    TransformSuggestion,
    apply_transforms,
    suggest_transforms,
)
from auto_cleaner.config import CleanConfig

__all__ = [
    "AdvancedAnalysis",
    "run_advanced",
    "NormalityResult",
    "normality_tests",
    "TransformSuggestion",
    "suggest_transforms",
    "apply_transforms",
    "VIFResult",
    "PCAResult",
    "vif_scores",
    "pca_summary",
    "RelevanceResult",
    "feature_relevance",
]


@dataclass(slots=True)
class AdvancedAnalysis:
    """Aggregate container for every advanced diagnostic."""

    normality: list[NormalityResult] = field(default_factory=list)
    transforms: list[TransformSuggestion] = field(default_factory=list)
    vif: list[VIFResult] = field(default_factory=list)
    pca: PCAResult | None = None
    relevance: list[RelevanceResult] = field(default_factory=list)
    target: str | None = None
    task_type: str | None = None
    # Populated by the pipeline from the top-level advanced modules:
    specialization: Any = None   # auto_cleaner.specialize.Specialization
    inference: Any = None        # auto_cleaner.inference.InferenceReport
    modeling: Any = None         # auto_cleaner.modeling.ModelReport
    fda: Any = None              # auto_cleaner.functional.FDAReport
    extended: Any = None         # auto_cleaner.stats.ExtendedAnalysis
    tuning: Any = None           # auto_cleaner.tuning.TuneResult
    causal: Any = None           # auto_cleaner.causal.CausalReport


def run_advanced(
    df: pl.DataFrame, config: CleanConfig | None = None, *, target: str | None = None
) -> AdvancedAnalysis:
    """Run all advanced diagnostics and return an :class:`AdvancedAnalysis`."""
    config = config or CleanConfig()
    target = target if target is not None else config.target
    adv = AdvancedAnalysis(target=target)
    adv.normality = normality_tests(df, config)
    adv.transforms = suggest_transforms(df, config)
    adv.vif = vif_scores(df, config)
    adv.pca = pca_summary(df, config)
    if target:
        adv.relevance, adv.task_type = feature_relevance(df, target, config)
    return adv
