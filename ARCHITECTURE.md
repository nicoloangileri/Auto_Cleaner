# Architecture & Scaling — the honest version

This document is deliberately candid about what `auto_cleaner` is, where its
ceiling is, and how you would scale it. Pretending it does more than it does
would be the opposite of useful.

## Design

The engine is a **functional pipeline**: each stage is a pure function
`DataFrame -> (DataFrame, StepReport)`, and `run_pipeline` is their composition.
State lives only in the data flowing through and in the report accumulators.
This makes stages independently testable, reorderable, and reusable. Optional
backends are imported lazily so the core stays light and a missing dependency
degrades one feature rather than breaking the run.

```
ingest ─▶ validate ─▶ standardise ─▶ impute ─▶ outliers ─▶ downcast
       ─▶ profile ─▶ auto-specialise ─▶ analysis modules ─▶ reporters
```

## The ceiling: single machine

The compute engine is **polars (in-memory, multi-threaded)**. That is fast and
efficient, but fundamentally **single-machine**. The honest limits:

- **Bigger than RAM, one machine:** handled — polars streaming + DuckDB
  out-of-core profiling push well past memory on a single box.
- **Distributed / petabyte / cluster:** **not** handled, and not something you
  bolt on. That is a different engine (Spark / Dask / Ray). You would not "add
  Spark" to this — you would port the stage implementations onto it.
- **Real-time streaming** (Kafka/Flink): out of scope — a different paradigm
  (unbounded event streams vs bounded batch).

## How you would actually scale it

The pipeline's *shape* is portable even though the engine is not:

1. **Keep the stage interfaces** (`DataFrame -> (DataFrame, Report)`), swap the
   implementation to a distributed frame (Spark DataFrame, Dask, Polars-on-Ray).
   The orchestration, reporting, contracts and manifest layers stay.
2. **Orchestrate** with Airflow/Dagster/Prefect — see
   [`examples/airflow_dag.py`](examples/airflow_dag.py) for a daily DAG that
   cleans, enforces a data contract, and computes drift vs the previous run.
3. **Govern** with the existing pieces: `--contract` (YAML data contracts),
   `results.json` (reproducibility manifest + machine-readable results for
   lineage/observability hooks), and the data-quality score as an SLA signal.
4. **Operationalise** the model: the run saves a `*_model.joblib` bundle and
   ships an `auto-cleaner-predict` CLI for batch scoring of new data.

## What it is (and is not)

| It is | It is not |
|---|---|
| An automated cleaning + analysis accelerator | A distributed data platform |
| Single-machine (RAM + streaming + DuckDB) | Spark/petabyte-scale compute |
| Batch | Real-time streaming |
| Baselines + diagnostics for a human to validate | An autonomous decision/oracle |
| Orchestratable (example DAG provided) | An orchestrator itself |

That clarity is the point: use it for what it is excellent at — turning raw,
messy, single-machine-scale data into a clean dataset and a rigorous first
analysis, fast — and reach for the right heavier tool when you cross its ceiling.
