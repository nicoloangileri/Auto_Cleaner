"""Extended statistics suite + orchestrator.

:func:`run_extended` runs the relevant modules for a dataset — always-on ones
(robust means, associations, distribution fitting, multivariate, Bayes factors,
survival, survey) plus archetype-gated ones (time-series diagnostics for a time
index; NLP / embeddings for text columns). Each module degrades gracefully, so a
missing optional backend never breaks a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from auto_cleaner.config import CleanConfig
from auto_cleaner.embeddings import EmbeddingReport, embedding_analysis
from auto_cleaner.stats.associations import AssociationsReport, associations
from auto_cleaner.stats.bayesian import BayesianReport, bayesian_analysis
from auto_cleaner.stats.distributions import DistributionFit, fit_distributions
from auto_cleaner.stats.forecast import ForecastResult, forecast_series
from auto_cleaner.stats.multivariate import MultivariateReport, multivariate_analysis
from auto_cleaner.stats.nlp import NLPColumnReport, nlp_analysis
from auto_cleaner.stats.robust import RobustResult, robust_summary
from auto_cleaner.stats.survey import SurveyReport, survey_analysis
from auto_cleaner.stats.survival import SurvivalReport, survival_analysis
from auto_cleaner.stats.timeseries import TimeSeriesResult, timeseries_analysis

__all__ = [
    "ExtendedAnalysis", "run_extended",
    "RobustResult", "robust_summary",
    "AssociationsReport", "associations",
    "TimeSeriesResult", "timeseries_analysis",
    "DistributionFit", "fit_distributions",
    "ForecastResult", "forecast_series",
    "MultivariateReport", "multivariate_analysis",
    "NLPColumnReport", "nlp_analysis",
    "BayesianReport", "bayesian_analysis",
    "SurvivalReport", "survival_analysis",
    "SurveyReport", "survey_analysis",
    "EmbeddingReport", "embedding_analysis",
]


@dataclass(slots=True)
class ExtendedAnalysis:
    """Aggregate container for the extended statistics suite."""

    robust: list[RobustResult] = field(default_factory=list)
    associations: AssociationsReport | None = None
    timeseries: list[TimeSeriesResult] = field(default_factory=list)
    distributions: list[DistributionFit] = field(default_factory=list)
    forecasts: list[ForecastResult] = field(default_factory=list)
    multivariate: MultivariateReport | None = None
    nlp: list[NLPColumnReport] = field(default_factory=list)
    bayesian: BayesianReport | None = None
    survival: SurvivalReport | None = None
    survey: SurveyReport | None = None
    embeddings: list[EmbeddingReport] = field(default_factory=list)


def run_extended(
    df: pl.DataFrame,
    config: CleanConfig | None = None,
    *,
    spec: Any = None,
    target: str | None = None,
    id_columns: list[str] | None = None,
) -> ExtendedAnalysis:
    """Run the full extended suite, gating archetype-specific modules on ``spec``."""
    config = config or CleanConfig()
    target = target if target is not None else config.target
    ext = ExtendedAnalysis()
    if not config.extended_stats:
        return ext

    ext.robust = robust_summary(df, config)
    ext.associations = associations(df, config)
    ext.distributions = fit_distributions(df, config)
    ext.multivariate = multivariate_analysis(df, config, id_columns=id_columns)
    ext.bayesian = bayesian_analysis(df, config, target=target, id_columns=id_columns)
    ext.survival = survival_analysis(df, config, id_columns=id_columns)
    ext.survey = survey_analysis(df, config, id_columns=id_columns)

    text_cols = list(getattr(spec, "text_columns", []) or [])
    if text_cols:
        ext.nlp = nlp_analysis(df, text_cols, config)
        ext.embeddings = embedding_analysis(df, text_cols, config)

    time_index = getattr(spec, "time_index", None)
    if time_index is not None:
        ext.timeseries = timeseries_analysis(df, time_index, config, id_columns=id_columns)
        if config.forecast:
            ext.forecasts = forecast_series(df, time_index, config, id_columns=id_columns)

    return ext
