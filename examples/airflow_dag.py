"""Example Apache Airflow DAG that runs auto_cleaner on a daily schedule.

This demonstrates that the (single-machine) pipeline plugs cleanly into an
orchestrator: ingest -> clean -> analyse -> report, every morning, with a data
contract enforced on the way in. For cluster-scale data you would swap the
compute engine (Spark/Ray) while keeping this orchestration shape — see
ARCHITECTURE.md.

This file is illustrative (requires ``apache-airflow``) and is NOT imported by
the package.
"""

from __future__ import annotations

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:  # example only — keep importable without airflow installed
    DAG = None  # type: ignore[assignment]
    PythonOperator = None  # type: ignore[assignment]


def run_auto_cleaner(**_context) -> None:
    """Task body: clean + analyse today's drop, enforcing a data contract."""
    from auto_cleaner import CleanConfig, run_pipeline

    run_pipeline(
        "/data/raw/today.csv",
        "/data/clean/today.parquet",
        CleanConfig().with_overrides(target="label", verbose=False),
        contract="/data/contracts/today.yml",
        compare="/data/clean/yesterday.parquet",  # drift vs the previous run
    )


if DAG is not None:
    default_args = {"retries": 1, "retry_delay": timedelta(minutes=5)}
    with DAG(
        dag_id="auto_cleaner_daily",
        schedule="0 6 * * *",                 # every day at 06:00
        start_date=datetime(2024, 1, 1),
        catchup=False,
        default_args=default_args,
        tags=["data-quality", "auto_cleaner"],
    ) as dag:
        PythonOperator(task_id="clean_analyse_report", python_callable=run_auto_cleaner)
