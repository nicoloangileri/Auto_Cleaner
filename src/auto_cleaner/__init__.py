"""auto_cleaner — autonomous, polars-native data preprocessing & EDA.

A modular, high-performance toolkit that ingests messy data (CSV / Parquet /
JSON / SQL), produces a mathematically clean dataset, and emits a comprehensive
Exploratory Data Analysis report — all on top of ``polars`` (multi-threaded,
memory-efficient) with optional ``duckdb`` for SQL ingestion.

The public surface is intentionally tiny and functional::

    from auto_cleaner import run_pipeline, CleanConfig

    result = run_pipeline("raw_data.csv", "clean_data.parquet")
    print(result.summary())

Design principles
-----------------
* **Functional core.** Every transformation is a pure function
  ``DataFrame -> (DataFrame, StepReport)``. No hidden state, no god-objects.
* **polars-first.** Vectorised expressions everywhere; Python row-loops never.
* **Honest typing.** Rigorous ``typing`` annotations and docstrings throughout.
* **Trustworthy defaults.** Safe enough that a Data Scientist can hand the
  output straight to a model without re-checking the basics.
"""

from __future__ import annotations

from auto_cleaner.config import CleanConfig
from auto_cleaner.pipeline import PipelineResult, run_pipeline

__all__ = ["CleanConfig", "PipelineResult", "run_pipeline", "__version__"]
__version__ = "1.0.0"
