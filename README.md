# auto_cleaner

**Autonomous, `polars`-native data preprocessing & automated EDA.**

Point it at any messy dataset — CSV, **Excel**, Parquet, JSON, or a SQL
database — and it returns a mathematically clean, ML-ready dataset plus a
comprehensive Exploratory Data Analysis report (HTML, Markdown **and** PDF),
opening with an **executive summary** of what matters and a review checklist.
No `pandas`: the engine is built end-to-end on [`polars`](https://pola.rs) for
multi-threaded, memory-efficient manipulation, with
[`duckdb`](https://duckdb.org) for SQL ingestion.

```bash
python -m auto_cleaner --input raw_data.csv --output clean_data.parquet
```

That single command auto-detects the format, fixes types, imputes missing
values, treats outliers, downcasts dtypes for memory, profiles the data, and
writes `clean_data_eda.html` / `clean_data_eda.md` beside the output.

---

## Why it exists

A Data Scientist or Quant should be able to *trust* the output as the input to
a model — because the evidence travels with it. Every transformation is
conservative by default (e.g. a `Float64 → Float32` downcast only happens when
it provably round-trips), every decision is logged, and **every
distribution-altering step is quantified**: for each imputed or treated column
the report shows cells changed, the mean shift in pre-cleaning standard
deviations, the two-sample Kolmogorov–Smirnov distance, and a
`negligible / minor / material` verdict — material distortions are escalated
to the executive summary. The EDA report also foregrounds **data-health
warnings** ("Feature X has 40% missing values", "High collinearity between Y
and Z") so nothing silently corrupts a downstream model.

## Design principles

- **Functional core, no god-objects.** Every step is a pure function
  `DataFrame -> (DataFrame, StepReport)`. The pipeline is literally their
  composition: `ingest → standardize → impute → outliers → downcast → profile`.
- **polars-first.** Vectorised expressions throughout; never a Python row-loop.
- **Rigorous typing & docstrings.** Full `typing` annotations; immutable,
  `slots`-based config and result objects.
- **Trustworthy defaults.** Safe enough to hand straight to scikit-learn.

## Architecture

```
auto-cleaner/
├── pyproject.toml            # installable package + `auto-cleaner` console script
├── requirements.txt
├── README.md
├── src/
│   └── auto_cleaner/
│       ├── __init__.py       # public API: run_pipeline, CleanConfig
│       ├── __main__.py       # CLI  → python -m auto_cleaner ...
│       ├── config.py         # immutable CleanConfig (every knob lives here)
│       ├── logging_utils.py  # zero-dependency logging + timing
│       ├── reporting.py      # StepReport accumulator
│       ├── pipeline.py       # functional orchestration + PipelineResult
│       ├── specialize.py     # auto-specialisation engine (archetype detection → routing)
│       ├── inference.py      # bootstrap CIs, group tests, BH correlations, OLS/logit
│       ├── modeling.py       # baseline AutoML (CV benchmarks + permutation importance)
│       ├── functional.py     # functional data analysis (smoothing + functional PCA)
│       ├── embeddings.py     # opt-in neural text embeddings (sentence-transformers)
│       ├── validate.py       # typed input/schema validation (hardening)
│       ├── pdfreport.py      # per-dataset PDF report (fpdf2)
│       ├── drift.py          # dataset drift / comparison (PSI + KS / chi-square)
│       ├── reproducibility.py# results.json + reproducibility manifest
│       ├── persistence.py    # exportable model bundle (joblib)
│       ├── predict.py        # `auto-cleaner-predict` scoring CLI
│       ├── tuning.py         # opt-in Optuna hyper-parameter tuning
│       ├── causal.py         # opt-in A/B test + causal IPW (heavily caveated)
│       ├── ingest/           # 1. Dynamic ingestion
│       │   ├── detect.py     #    format / encoding / delimiter / header sniffing
│       │   └── readers.py    #    polars + DuckDB readers
│       ├── clean/            # 2. Smart cleaning engine
│       │   ├── standardize.py#    whitespace, numeric strings, messy datetimes, casing
│       │   ├── impute.py     #    time-series ffill / median / mean / KNN / mode
│       │   ├── outliers.py   #    IQR, Z-score, Isolation Forest → flag / cap / drop
│       │   ├── impact.py     #    per-column footprint of each step (KS, Δmean, verdict)
│       │   └── dtypes.py     #    safe downcasting + Categorical encoding
│       ├── eda/              # 3. Automated statistical EDA + visualisation
│       │   ├── stats.py      #    skew, kurtosis, missingness, correlation/covariance
│       │   ├── summary.py    #    executive summary: findings ranked + review checklist
│       │   ├── report.py     #    self-contained HTML + Markdown report
│       │   └── visualize.py  #    interactive Plotly charts + standalone PNG export
│       ├── analyze/          # 4. Advanced analysis (research-grade)
│       │   ├── tests.py            # Shapiro / D'Agostino / Jarque-Bera / Anderson-Darling
│       │   ├── transforms.py       # Box-Cox / Yeo-Johnson normalising transforms
│       │   ├── multicollinearity.py# VIF + PCA explained variance
│       │   └── feature_relevance.py# mutual information, ANOVA F, target correlation
│       └── stats/            # 5. Extended statistics suite
│           ├── robust.py            # geometric/harmonic/trimmed/winsorized/MAD/Huber
│           ├── associations.py      # Spearman/Kendall/partial/Cramer's V/eta
│           ├── distributions.py     # distribution fitting + AIC/BIC
│           ├── timeseries.py        # ADF/KPSS, ACF/PACF, STL, Mann-Kendall, Ljung-Box
│           ├── forecast.py          # ARIMA / Holt-Winters forecasting + 95% intervals
│           ├── multivariate.py      # Mahalanobis, clustering, MANOVA, UMAP
│           ├── nlp.py               # LDA topics + VADER sentiment
│           ├── bayesian.py          # Bayes factors
│           ├── survival.py          # Kaplan-Meier + Cox PH
│           └── survey.py            # Cronbach's alpha + design weights
├── r/
│   └── eda_companion.R       # independent R cross-check (university companion)
├── sql/
│   ├── ingest.sql            # DuckDB ingestion recipes
│   └── profiling.sql         # in-database profiling (scales past RAM)
├── tests/                    # 100 tests across 12 files: unit, end-to-end,
│                             #   hostile-input, known-value statistics
└── examples/
    └── generate_raw.py       # materialises a real messy dataset for the demo
```

> **Note on layout.** The brief sketched `src/ingest`, `src/clean`, `src/eda`.
> This repo uses the **`src/auto_cleaner/...` package (src-layout)** instead —
> the professional standard that makes `python -m auto_cleaner` and the
> `auto-cleaner` console script work cleanly while keeping the exact module
> boundaries you asked for (`ingest`, `clean`, `eda`, `pipeline.py`).

## Installation

```bash
pip install -e .          # core (polars + numpy)
pip install -e ".[all]"   # + duckdb (SQL), scikit-learn (IForest/KNN), pyarrow
```

Or just install the pinned set: `pip install -r requirements.txt`.

## Usage

### CLI

```bash
# Minimal
python -m auto_cleaner -i raw_data.csv -o clean_data.parquet

# Tuned: KNN imputation, multivariate outliers, winsorize
python -m auto_cleaner -i sales.csv -o clean.parquet \
    --impute knn --outliers iqr,isolation_forest --outlier-action cap

# Excel: read the 'vendite' worksheet of a workbook
python -m auto_cleaner -i report.xlsx --sheet vendite -o clean.parquet

# Big file, just clean it: fast profile (no charts/advanced analysis, seconds)
python -m auto_cleaner -i big.csv -o clean.parquet --profile fast

# SQL ingestion via DuckDB
python -m auto_cleaner -i warehouse.duckdb --table trades -o clean.parquet
python -m auto_cleaner -i :memory: \
    --query "SELECT * FROM read_parquet('ticks/*.parquet')" -o clean.parquet

# Research-grade: target-aware feature relevance + apply power transforms
python -m auto_cleaner -i data.parquet -o clean.parquet \
    --target label --apply-transforms

# Astronomy FITS input + larger-than-memory streaming ingestion
python -m auto_cleaner -i survey.fits -o clean.parquet --streaming
```

Key advanced flags: `--profile {fast,standard,full}` (execution depth),
`--sheet <name>` (Excel worksheet), `--target <col>` (feature relevance),
`--apply-transforms` (append Box-Cox/Yeo-Johnson columns), `--no-advanced`
(skip the advanced section), `--streaming` (out-of-core ingestion).

Key flags: `--impute {auto,median,mean,knn,none}`,
`--outliers iqr,zscore,isolation_forest`, `--outlier-action {flag,cap,drop,none}`,
`--no-downcast`, `--categorical-case {none,lower,upper,title}`,
`--corr-threshold`, `--missing-threshold`. Full list: `python -m auto_cleaner -h`.

### Python API

```python
from auto_cleaner import run_pipeline, CleanConfig

cfg = CleanConfig().with_overrides(
    impute_numeric="auto",
    outlier_methods=("iqr", "isolation_forest"),
    outlier_action="flag",
)
result = run_pipeline("raw_data.csv", "clean_data.parquet", cfg)

print(result.summary())          # memory saved, rows, warnings, paths
clean_df = result.frame          # a polars DataFrame, ready for modelling
for w in result.profile.warnings:
    print(w)
```

Individual stages are independently importable and composable:

```python
import polars as pl
from auto_cleaner.clean import standardize, impute_missing, handle_outliers, downcast

df = pl.read_csv("raw.csv")
df, _ = standardize(df)
df, _ = impute_missing(df)
df, _ = handle_outliers(df)
df, _ = downcast(df)
```

## R & SQL companions

These re-derive the core statistics independently — handy for coursework, peer
review, or pushing profiling into the database.

```bash
# R: independent statistical cross-check of the cleaned data
Rscript r/eda_companion.R clean_data.parquet eda_report_R.md

# SQL: profile in-database with DuckDB (scales beyond memory)
duckdb mydb.duckdb < sql/profiling.sql
```

## Run the demo

```bash
python examples/generate_raw.py                       # real auto-mpg data → CSV
python -m auto_cleaner -i examples/data/raw_cars.csv \
       -o examples/output/clean_cars.parquet
open examples/output/clean_cars_eda.html              # interactive charts + tables
# standalone chart images are also written to examples/output/charts/*.png
```

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

## Feature checklist

| Stage | Capability |
|---|---|
| Ingestion | format + encoding + delimiter + header auto-detection; CSV/Excel/Parquet/JSON/NDJSON; DuckDB SQL; malformed-CSV recovery ladder with counted (never silent) truncations |
| Impact accounting | per-column footprint of every distribution-altering step: cells changed, Δmean in sd units, KS distance, `negligible/minor/material` verdict; material → escalated warning |
| Executive summary | report opens with prose findings ranked worst-first + a human review checklist |
| Profiles | `--profile fast` (clean+report), `standard` (+ diagnostics), `full` (everything) |
| Memory | smallest-fit integer downcast; provably-safe `Float64→Float32`; `Utf8→Categorical` |
| Imputation | time-series forward/back fill; median (skewed) / mean (symmetric) auto-routing; lightweight KNN; categorical mode/constant |
| Outliers | IQR, Z-score, Isolation Forest (multivariate) → flag / cap / drop |
| Standardisation | whitespace strip; messy datetime parsing; `$1,234` / `45%` → numeric; categorical casing |
| EDA | per-column profile, skewness, kurtosis, missingness, correlation & covariance, collinearity detection, health warnings, HTML + Markdown report |
| Visualisation | interactive Plotly charts embedded **offline** (histograms, frequency bars, scatterplot matrix, outlier boxplots, correlation heatmap, missingness) + standalone PNG export to `charts/` |
| Advanced | normality tests (Shapiro / D'Agostino / Jarque-Bera / Anderson-Darling), automatic Box-Cox / Yeo-Johnson transforms for skewed features, VIF + PCA multicollinearity, target-aware feature relevance (mutual information, ANOVA F, target correlation) |
| Auto-specialisation | inspects the dataset and detects archetype(s) — time-series, geospatial, text-heavy, high-dimensional/embeddings, survey, wide/omics, image-references — then **auto-routes** which advanced modules run |
| Inference | bootstrap confidence intervals, auto-selected group tests (t / Welch / Mann-Whitney / ANOVA / Kruskal / chi²) **with effect sizes** (Cohen's d / Cliff's delta / eta²), Benjamini-Hochberg-corrected correlation significance, OLS / logit regression with p-values + CIs |
| Modelling | baseline cross-validated benchmarks (dummy / linear / random forest / gradient boosting) with metrics, permutation importance, **SHAP** impact and a leakage check — a benchmark for a human to build on, **not** a deployable model |
| Functional (FDA) | smoothing + functional PCA of time-indexed curves, auto-triggered when a time index is found |
| Forecasting | per-series **ARIMA / Holt-Winters** (auto-selected by AIC) projecting ahead with 95% prediction intervals — "what might happen next" |
| Drift & monitoring | `--compare second.csv` → PSI + KS / chi-square per feature with a stability verdict (stable / moderate / major) |
| Quality & reproducibility | composite **0–100 data-quality score**; machine-readable `results.json` with a full reproducibility **manifest** (config, seed, library versions, dataset SHA-256); statistical-hygiene guard for multiple comparisons |
| Productionisation | saves an exportable model bundle (`*_model.joblib`) + an `auto-cleaner-predict` CLI for batch scoring of new data |
| Orchestration & contracts | YAML **data contracts** (`--contract`); plugs into Airflow/Dagster (example daily DAG in `examples/airflow_dag.py`) |
| Opt-in (advanced) | **Optuna** hyper-parameter tuning (`--tune`); **A/B test + observational causal IPW** (`--treatment`/`--outcome`, with prominent caveats — never auto-run); neural text embeddings |
| Extended statistics | robust means (geometric/harmonic/trimmed/winsorized/MAD/Huber); rank, partial & categorical associations (Spearman/Kendall/partial/Cramer's V/eta); distribution fitting (AIC/BIC); multivariate (Mahalanobis, clustering, MANOVA, UMAP); classical NLP (LDA topics + sentiment); Bayes factors; survival (Kaplan-Meier, Cox); survey reliability (Cronbach's α + design weights) |
| Reports & outputs | clean dataset (Parquet/CSV/Arrow); interactive offline **HTML** + **Markdown** + **PDF** reports; machine-readable **results.json**; standalone PNG charts; R companion + DuckDB SQL profiling |
| Hardening | typed input/schema validation, YAML data contracts, 108 tests (property-based ingestion fuzzing via Hypothesis + a golden-file report test) at 78% line coverage; a versioned pre-commit hook runs the suite before every commit (`git config core.hooksPath .githooks`); validated unmodified on 10 real-world datasets across 9 domains (see the paper) |
| Measured scale | ~500k rows/s sustained: 11.9M rows (4× NYC-taxi) cleaned + profiled in 26 s at 3.0 GB peak RSS (`examples/scale_probe.py`) |
| Formats & scale | CSV/TSV (any delimiter), **Excel** (.xlsx/.xlsm/.xls via calamine), Parquet, JSON/NDJSON, DuckDB SQL, **FITS** (astronomy), **netCDF** (climate, via xarray); optional streaming / out-of-core ingestion for larger-than-memory data |

## Scope & honest limitations

`auto_cleaner` is a strong **automated cleaning + analysis** tool — but it is not
a complete data-science platform, and it says so plainly:

- Modelling and inference outputs are **baselines and exploratory diagnostics for
  a human to validate**, never final causal claims or production models.
  Automating inference blindly invites p-hacking, leakage and false confidence.
- It is **domain-aware via heuristics**, not a domain expert: it routes by
  detected archetype but does not apply field-specific methodology (survey
  weighting, RNA-seq normalisation, climate de-seasonalisation, astrometric
  calibration, …).
- It handles **tabular** data (plus FITS/netCDF flattened to tables). It does not
  do computer vision, audio, graph learning, or deep-learning NLP — text support
  is basic features only.
- It is a clean, tested package, not a library hardened by years of production
  edge-cases.

In short: it takes you from a raw, messy file to a clean dataset and a rigorous,
well-structured *first* analysis — fast, and across domains. The domain-specific
and modelling steps are where a human takes over.

## Further reading

- [CASE_STUDY.md](CASE_STUDY.md) — portfolio-style write-up (problem, approach, real results).
- [ARCHITECTURE.md](ARCHITECTURE.md) — design + the honest single-machine ceiling and scaling path.

## Development note

This project was built with AI-assisted engineering. Its architecture,
statistical methods, and validation strategy were designed, reviewed, and are
fully understood and owned by the author — the AI served as an implementation
partner, it really helped me debugging and fix issues, not a substitute for judgement.

## License

MIT.
