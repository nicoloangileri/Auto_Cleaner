# auto_cleaner — Case Study

*A polars-native engine that takes a raw, messy dataset to a clean, ML-ready
table plus a rigorous automated analysis — in one command, across domains.*

## The problem

Every data project starts with the same unglamorous 60–80%: ingesting messy
files, fixing types, handling missing values and outliers, then doing enough
EDA to understand the data before any modelling. It is repetitive, error-prone,
and rarely reproducible. I wanted a single tool that does this first pass
**autonomously, fast, and honestly** — and that adapts to whatever kind of data
it is given.

## What it does

One command runs a functional pipeline:

```
ingest → validate → standardise → impute → outliers → downcast
       → profile → auto-specialise → {inference · modelling · forecasting · FDA · extended stats}
       → report (HTML · Markdown · PDF · JSON) + saved model
```

The differentiator is **auto-specialisation**: it inspects the schema and value
patterns, detects the dataset's archetype (time-series, geospatial, text-heavy,
high-dimensional/embeddings, survey, wide/omics…), and routes the analyses that
actually fit — so a climate series gets de-seasonalisation and forecasting while
a generic table gets baseline modelling.

## Technical choices (and why)

- **polars, not pandas** — multi-threaded, columnar, memory-efficient; the whole
  engine is vectorised with zero pandas in the data path.
- **DuckDB** for SQL ingestion and out-of-core profiling (data larger than RAM).
- **Functional core** — every step is a pure `DataFrame -> (DataFrame, Report)`
  function; the pipeline is their composition. Easy to test, reason about, reuse.
- **Graceful degradation** — heavy/optional backends (scikit-learn, statsmodels,
  astropy, sentence-transformers…) are imported lazily; a missing one downgrades
  a feature instead of breaking a run.
- **Honesty by construction** — modelling/inference are framed as *baselines and
  diagnostics for a human to validate*; a statistical-hygiene guard warns about
  multiple comparisons; every run emits a reproducibility manifest.
- **Engineering hygiene** — typed config, full type hints, 48 tests, GitHub
  Actions CI, data contracts, and an exportable model + `predict` CLI.

## Real results (proof of work)

Run on the public **auto-mpg** dataset (406×9), unedited:

- Memory **−44.7%** via safe downcasting; **data-quality score 91.9/100**.
- Correctly flagged severe multicollinearity (VIF: Displacement 20.0, Weight
  11.4, Cylinders 10.8) and Box-Cox normalised Horsepower (skew **+1.05 → +0.02**).
- Baseline 5-fold CV: Random Forest **R² ≈ 0.41** vs a −0.98 dummy, with SHAP +
  permutation importance.

On a daily climate series it auto-detected a **time-series** archetype and
forecast each variable 12 steps ahead (e.g. temperature 95% PI [14.4, 20.1]). A
drift comparison against a perturbed copy isolated the **one** shifted feature
(Horsepower, PSI 0.93, "major drift").

## Scope & honesty

It is an **automated analysis + cleaning accelerator** for tabular data up to
single-machine scale — not a distributed platform and not a replacement for
human judgement. Modelling and inference are starting points; domain-specific
methodology and production deployment are where a person takes over. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the honest scaling story.

## Skills demonstrated

Data engineering (ingestion, validation, contracts, reproducibility), applied
statistics (inference, effect sizes, distribution fitting, time-series, Bayesian,
survival), ML (baselines, SHAP, forecasting), performance engineering (polars,
DuckDB, streaming), software craft (functional design, typing, tests, CI,
packaging), and clear technical communication.
