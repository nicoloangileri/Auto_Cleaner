"""Central, immutable configuration for the cleaning pipeline.

Everything tunable lives in :class:`CleanConfig`. It is a *frozen* dataclass so
a config object can be safely shared across threads and embedded verbatim in the
EDA report for full reproducibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

# --- Strategy vocabularies (kept as Literals for static checking) ----------- #
OutlierMethod = Literal["iqr", "zscore", "isolation_forest"]
OutlierAction = Literal["flag", "cap", "drop", "none"]
NumericImputer = Literal["auto", "median", "mean", "knn", "none"]
CategoricalImputer = Literal["mode", "constant", "none"]
CategoricalCase = Literal["none", "lower", "upper", "title"]

__all__ = [
    "CleanConfig",
    "OutlierMethod",
    "OutlierAction",
    "NumericImputer",
    "CategoricalImputer",
    "CategoricalCase",
]


@dataclass(frozen=True, slots=True)
class CleanConfig:
    """Immutable bundle of every knob the pipeline exposes.

    Attributes are grouped by stage. All have production-sane defaults; the CLI
    and :func:`auto_cleaner.run_pipeline` accept overrides.
    """

    # -- Ingestion ---------------------------------------------------------- #
    detection_sample_rows: int = 4096
    """Rows sampled from the head of a file for separator/type sniffing."""
    csv_null_values: tuple[str, ...] = (
        "",
        "na",
        "n/a",
        "nan",
        "null",
        "none",
        "nil",
        "-",
        "?",
        "#n/a",
        "***",  # NASA GISTEMP-style missing marker
    )
    """Case-insensitive tokens treated as missing during CSV ingestion."""
    excel_sheet: str | None = None
    """Worksheet to read from an Excel workbook (default: the first sheet)."""

    # -- Memory optimisation (downcasting) ---------------------------------- #
    downcast: bool = True
    """Master switch for automatic dtype downcasting."""
    downcast_floats: bool = True
    """Downcast Float64 -> Float32 when the column round-trips within tolerance."""
    float32_rel_tolerance: float = 1e-6
    """Max relative error allowed when validating a Float64 -> Float32 downcast."""

    # -- Missing-data imputation -------------------------------------------- #
    impute_numeric: NumericImputer = "auto"
    """'auto' picks median for skewed / mean for symmetric columns."""
    impute_categorical: CategoricalImputer = "mode"
    categorical_fill_value: str = "Unknown"
    """Constant used when ``impute_categorical='constant'``."""
    skew_threshold: float = 1.0
    """|skewness| above this routes a column to median imputation under 'auto'."""
    detect_timeseries: bool = True
    """If a sorted datetime index is found, numeric NaNs are forward/back filled."""
    knn_neighbors: int = 5
    knn_max_rows: int = 20_000
    """Guardrail: above this row-count, KNN falls back to median (keeps it 'lightweight')."""

    # -- Outlier detection -------------------------------------------------- #
    outlier_methods: tuple[OutlierMethod, ...] = ("iqr",)
    """Any subset of {'iqr','zscore','isolation_forest'}."""
    outlier_action: OutlierAction = "flag"
    """flag = annotate only, cap = winsorize to bounds, drop = remove rows."""
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0
    iforest_contamination: float | Literal["auto"] = "auto"

    # -- Feature standardisation -------------------------------------------- #
    strip_whitespace: bool = True
    parse_datetimes: bool = True
    datetime_parse_min_success: float = 0.80
    """Min fraction of non-null values that must parse for a column to become datetime."""
    standardize_categoricals: bool = True
    categorical_case: CategoricalCase = "none"
    parse_numeric_strings: bool = True
    """Coerce strings like '$1,234.5' / '45%' into floats when the column is numeric-like."""
    numeric_string_min_success: float = 0.90

    # -- EDA / data-health thresholds --------------------------------------- #
    missing_warn_threshold: float = 0.20
    """Columns missing more than this fraction raise a health warning."""
    corr_threshold: float = 0.90
    """|correlation| above this between two features raises a collinearity warning."""
    high_cardinality_warn: int = 50
    """Categoricals with more unique levels than this raise a warning."""
    skew_warn_threshold: float = 2.0

    # -- Visualisation (interactive Plotly EDA charts) ---------------------- #
    make_charts: bool = True
    """Generate interactive charts (histograms, frequencies, SPLOM, boxplots, heatmap)."""
    export_png: bool = True
    """Also export each chart as a standalone PNG into a ``charts/`` folder."""
    make_pdf: bool = True
    """Also emit a polished per-dataset PDF report (``<output>_report.pdf``)."""
    make_json: bool = True
    """Also emit a machine-readable ``<output>_results.json`` (results + reproducibility manifest)."""
    save_model: bool = True
    """When a target is given, fit and serialise a model bundle (``<output>_model.joblib``)."""
    chart_max_numeric: int = 12
    """Cap on numeric columns charted as histograms/boxplots (keeps reports readable)."""
    chart_max_scatter_cols: int = 6
    """Cap on columns in the scatterplot matrix (SPLOM grows quadratically)."""
    chart_scatter_sample: int = 5_000
    """Row sample size for the scatterplot matrix (keeps it responsive)."""
    chart_top_categories: int = 15
    """Top-K levels shown in categorical frequency bars."""

    # -- Advanced analysis (research-grade) --------------------------------- #
    advanced: bool = True
    """Run normality tests, transform suggestions, VIF/PCA and feature relevance."""
    target: str | None = None
    """Optional target column → enables target-aware feature-relevance ranking."""
    apply_transforms: bool = False
    """If True, append power-transformed columns for skewed features to the output."""
    vif_warn: float = 10.0
    """VIF above this raises a severe-multicollinearity warning."""
    auto_specialize: bool = True
    """Detect dataset archetype(s) and auto-select which advanced modules to run."""
    inference: bool = True
    """Exploratory inference: bootstrap CIs, group tests, correlation significance, regression."""
    modeling: bool = True
    """Baseline cross-validated models when a target is provided."""
    fda: bool = True
    """Functional data analysis when a genuine time index is detected."""
    extended_stats: bool = True
    """Run the extended statistics suite (robust means, rank/partial/categorical
    associations, time-series diagnostics, distribution fitting, multivariate
    analysis, classical NLP, Bayes factors, survival, survey reliability)."""
    forecast: bool = True
    """Forecast numeric series ahead (ARIMA / Holt-Winters) when a time index is found."""
    forecast_horizon: int = 12
    """Number of steps to project ahead when forecasting."""
    tune: bool = False
    """Opt-in: Optuna hyper-parameter tuning of a gradient-boosting model."""
    tune_trials: int = 30
    """Number of Optuna trials when ``tune`` is enabled."""
    treatment: str | None = None
    """Treatment column → enables (opt-in) A/B test + causal IPW together with ``outcome``."""
    outcome: str | None = None
    """Outcome column for the A/B / causal analysis."""
    use_transformer_embeddings: bool = False
    """Opt-in neural text embeddings via sentence-transformers (heavy: downloads a model)."""
    embedding_model: str = "all-MiniLM-L6-v2"
    """Sentence-transformer model used when ``use_transformer_embeddings`` is on."""

    # -- Ingestion scale ---------------------------------------------------- #
    streaming: bool = False
    """Use polars streaming/lazy ingestion for larger-than-memory CSV/Parquet."""

    # -- Execution ---------------------------------------------------------- #
    random_seed: int = 7
    verbose: bool = True

    # ------------------------------------------------------------------ API #
    @classmethod
    def preset(cls, profile: str) -> "CleanConfig":
        """Named execution profiles trading depth for speed.

        - ``fast``     — clean + profile + reports only (seconds, any size);
        - ``standard`` — adds the advanced diagnostics an analyst reads daily
          (normality, transforms, VIF, relevance, inference, models, charts);
        - ``full``     — everything, including the extended statistics suite,
          forecasting and FDA (the current all-on defaults).
        """
        base = cls()
        if profile == "full":
            return base
        if profile == "standard":
            return base.with_overrides(extended_stats=False, forecast=False, fda=False)
        if profile == "fast":
            return base.with_overrides(
                advanced=False, inference=False, modeling=False, fda=False,
                extended_stats=False, forecast=False, auto_specialize=False,
                make_charts=False, export_png=False, make_pdf=False,
            )
        raise ValueError(f"Unknown profile '{profile}'; choose fast, standard or full")

    def with_overrides(self, **changes: Any) -> "CleanConfig":
        """Return a copy with ``changes`` applied (validated by dataclass)."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view, used for embedding the config in reports."""
        return asdict(self)
