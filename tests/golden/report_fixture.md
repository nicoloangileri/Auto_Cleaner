# Golden fixture report


## Executive Summary

> **TL;DR** — what matters in this dataset, worst news first.

- 🟡 Data-quality score **66.0/100**.
- ⚠️ Feature 'amount' is strongly skewed (skew=2.67) — consider a log/Box-Cox transform
- ⚠️ 30 duplicate row(s) detected
- 🔴 **impute materially changed `amount`** — 6 cell(s) (20.0%), mean shift 0.07 sd, KS 0.10. The cleaned column is *not* distribution-faithful.
- 🟡 3 row(s) flagged as outliers in `is_outlier` (kept in the data).

**Review checklist (human sign-off):**

- [ ] Decide whether the impute treatment of `amount` is acceptable, or re-run with a different strategy.
- [ ] Skim the flagged outlier rows before modelling.


## 1. Dataset Overview

- **Data-quality score:** 66.0/100  (completeness 100.0, validity 50.0, uniqueness 0.0)
- **Rows:** 30
- **Columns:** 4
- **In-memory size:** 278.0 B
- **Duplicate rows:** 30

## 2. Data-Health Warnings

- ⚠️ [impute] Cleaning materially changed 'amount' — 'amount': 6 cell(s) changed (20.0%), mean shift 0.074 sd, KS 0.100, → material. Review before trusting this column.
- ⚠️ Feature 'amount' is strongly skewed (skew=2.67) — consider a log/Box-Cox transform
- ⚠️ 30 duplicate row(s) detected

## 3. Visualisations

_Charts disabled (make_charts=False)._

## 4. Pipeline Actions

### ingest
- Ingested CSV → 30 rows × 3 cols (604.0 B in memory)

### standardize
- Stripped/collapsed whitespace on 1 text column(s)

### impute
- Imputed 'amount': 6 null(s) via median
- Imputed 'segment': 6 null(s) via mode 'corp'
- ⚠️ Cleaning materially changed 'amount' — 'amount': 6 cell(s) changed (20.0%), mean shift 0.074 sd, KS 0.100, → material. Review before trusting this column.

**Cleaning impact (impute)** — how much this step moved each column's distribution:

| Column | Cells changed | Mean before → after | Δmean (sd) | KS | Verdict |
|---|--:|--:|--:|--:|---|
| amount | 6 (20.0%) | 40.19 → 34.2 | 0.0739 | 0.1 | 🔴 material |

### outliers
- Flagged 3 row(s) in 'is_outlier'

### downcast
- Downcast 3 column(s): 632.0 B → 278.0 B (saved 56.0%)

## 5. Column Profiles

| Column | Dtype | Kind | Non-null | Null % | Unique | Mean | Std | Min | Median | Max | Skew | Kurtosis |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| amount | `Float32` | numeric | 30 | 0.0 | 8 | 34.2 | 73.17 | 9 | 10.25 | 250 | 2.666 | 5.11 |
| segment | `Categorical` | categorical | 30 | 0.0 | 2 | — | — | — | — | — | — | — |
| touches | `UInt8` | numeric | 30 | 0.0 | 3 | 1.8 | 0.7611 | 1 | 2 | 3 | 0.3436 | -1.153 |
| is_outlier | `Boolean` | boolean | 30 | 0.0 | 2 | — | — | — | — | — | — | — |

## 6. Correlation Highlights

_No feature pairs exceed the collinearity threshold._

## 7. Correlation Matrix (Pearson)

| | amount | touches |
|---|---|---|
| **amount** | +1.00 | -0.36 |
| **touches** | -0.36 | +1.00 |

## 8. Advanced Analysis

_Advanced analysis disabled._

## 9. Configuration

```json
{
  "detection_sample_rows": 4096,
  "csv_null_values": [
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
    "***"
  ],
  "excel_sheet": null,
  "downcast": true,
  "downcast_floats": true,
  "float32_rel_tolerance": 1e-06,
  "impute_numeric": "auto",
  "impute_categorical": "mode",
  "categorical_fill_value": "Unknown",
  "skew_threshold": 1.0,
  "detect_timeseries": true,
  "knn_neighbors": 5,
  "knn_max_rows": 20000,
  "outlier_methods": [
    "iqr"
  ],
  "outlier_action": "flag",
  "iqr_multiplier": 1.5,
  "zscore_threshold": 3.0,
  "iforest_contamination": "auto",
  "strip_whitespace": true,
  "parse_datetimes": true,
  "datetime_parse_min_success": 0.8,
  "standardize_categoricals": true,
  "categorical_case": "none",
  "parse_numeric_strings": true,
  "numeric_string_min_success": 0.9,
  "missing_warn_threshold": 0.2,
  "corr_threshold": 0.9,
  "high_cardinality_warn": 50,
  "skew_warn_threshold": 2.0,
  "make_charts": false,
  "export_png": true,
  "make_pdf": false,
  "make_json": false,
  "save_model": false,
  "chart_max_numeric": 12,
  "chart_max_scatter_cols": 6,
  "chart_scatter_sample": 5000,
  "chart_top_categories": 15,
  "advanced": false,
  "target": null,
  "apply_transforms": false,
  "vif_warn": 10.0,
  "auto_specialize": false,
  "inference": false,
  "modeling": false,
  "fda": false,
  "extended_stats": false,
  "forecast": false,
  "forecast_horizon": 12,
  "tune": false,
  "tune_trials": 30,
  "treatment": null,
  "outcome": null,
  "use_transformer_embeddings": false,
  "embedding_model": "all-MiniLM-L6-v2",
  "streaming": false,
  "random_seed": 7,
  "verbose": false
}
```
