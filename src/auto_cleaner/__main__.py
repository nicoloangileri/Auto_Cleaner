"""Command-line entry point.

Examples
--------
Minimal::

    python -m auto_cleaner --input raw_data.csv --output clean_data.parquet

Tuned::

    python -m auto_cleaner -i sales.csv -o clean.parquet \\
        --impute knn --outliers iqr,isolation_forest --outlier-action cap

SQL ingestion via DuckDB::

    python -m auto_cleaner -i warehouse.duckdb --table trades -o clean.parquet
    python -m auto_cleaner -i :memory: \\
        --query "SELECT * FROM read_parquet('ticks/*.parquet')" -o clean.parquet
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from auto_cleaner.config import CleanConfig
from auto_cleaner.pipeline import run_pipeline

_OUTLIER_CHOICES = {"iqr", "zscore", "isolation_forest"}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="auto_cleaner",
        description="Autonomous, polars-native data preprocessing & automated EDA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io = p.add_argument_group("input / output")
    io.add_argument("-i", "--input", required=True, help="Input path or ':memory:' for SQL.")
    io.add_argument("-o", "--output", help="Cleaned dataset path; extension picks the format.")
    io.add_argument("--report-html", help="Explicit HTML report path.")
    io.add_argument("--report-md", help="Explicit Markdown report path.")
    io.add_argument("--query", help="SQL query (DuckDB) when reading a database or ':memory:'.")
    io.add_argument("--table", help="Table to read from a DuckDB/SQLite database.")
    io.add_argument("--sheet", help="Worksheet to read from an Excel workbook (default: first).")
    io.add_argument("--compare", help="Second dataset to compute drift (PSI/KS) against.")
    io.add_argument("--contract", help="YAML data contract to validate the input against.")
    io.add_argument("--title", help="Report title.")
    io.add_argument(
        "--profile", choices=["fast", "standard", "full"], default="full",
        help="Execution profile: fast = clean+report only; standard = + advanced "
             "diagnostics; full = everything (default).",
    )

    mem = p.add_argument_group("memory optimisation")
    mem.add_argument("--no-downcast", action="store_true", help="Disable dtype downcasting.")

    imp = p.add_argument_group("imputation")
    imp.add_argument("--impute", choices=["auto", "median", "mean", "knn", "none"], default="auto")
    imp.add_argument(
        "--impute-categorical", choices=["mode", "constant", "none"], default="mode"
    )
    imp.add_argument("--knn-neighbors", type=int, default=5)

    out = p.add_argument_group("outliers")
    out.add_argument(
        "--outliers",
        default="iqr",
        help="Comma list of {iqr,zscore,isolation_forest}.",
    )
    out.add_argument(
        "--outlier-action", choices=["flag", "cap", "drop", "none"], default="flag"
    )
    out.add_argument("--iqr-multiplier", type=float, default=1.5)
    out.add_argument("--zscore-threshold", type=float, default=3.0)

    std = p.add_argument_group("standardisation")
    std.add_argument("--no-parse-dates", action="store_true", help="Do not parse datetime strings.")
    std.add_argument(
        "--categorical-case", choices=["none", "lower", "upper", "title"], default="none"
    )

    eda = p.add_argument_group("eda thresholds")
    eda.add_argument("--corr-threshold", type=float, default=0.90)
    eda.add_argument("--missing-threshold", type=float, default=0.20)

    viz = p.add_argument_group("visualisation")
    viz.add_argument("--no-charts", action="store_true", help="Disable interactive EDA charts.")
    viz.add_argument("--no-png", action="store_true", help="Do not export charts as PNG files.")
    viz.add_argument("--no-pdf", action="store_true", help="Do not emit the per-dataset PDF report.")
    viz.add_argument("--no-json", action="store_true", help="Do not emit the results.json export.")

    adv = p.add_argument_group("advanced analysis")
    adv.add_argument("--target", help="Target column → enables feature-relevance ranking.")
    adv.add_argument(
        "--apply-transforms", action="store_true",
        help="Append power-transformed (Box-Cox/Yeo-Johnson) columns for skewed features.",
    )
    adv.add_argument("--no-advanced", action="store_true", help="Skip the advanced-analysis section.")
    adv.add_argument("--no-inference", action="store_true", help="Skip statistical inference (CIs, tests, regression).")
    adv.add_argument("--no-modeling", action="store_true", help="Skip baseline cross-validated models.")
    adv.add_argument("--no-fda", action="store_true", help="Skip functional data analysis.")
    adv.add_argument("--no-specialize", action="store_true", help="Skip auto-specialisation routing.")
    adv.add_argument("--no-save-model", action="store_true", help="Do not save a model bundle when a target is given.")
    adv.add_argument("--tune", action="store_true", help="Opt-in: Optuna hyper-parameter tuning of the model.")
    adv.add_argument("--tune-trials", type=int, default=30, help="Number of Optuna trials when --tune is set.")
    adv.add_argument("--treatment", help="Treatment column → opt-in A/B test + causal analysis (needs --outcome).")
    adv.add_argument("--outcome", help="Outcome column for the A/B test + causal analysis.")
    adv.add_argument(
        "--streaming", action="store_true",
        help="Use polars streaming ingestion for larger-than-memory CSV/Parquet.",
    )

    p.add_argument("--seed", type=int, default=7)
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress step-by-step logs.")
    return p


def _parse_outliers(raw: str) -> tuple[str, ...]:
    methods = tuple(m.strip().lower() for m in raw.split(",") if m.strip())
    invalid = set(methods) - _OUTLIER_CHOICES
    if invalid:
        raise SystemExit(f"Unknown outlier method(s): {sorted(invalid)}; choose from {sorted(_OUTLIER_CHOICES)}")
    return methods or ("iqr",)


def _config_from_args(args: argparse.Namespace) -> CleanConfig:
    # Start from the chosen profile; --no-X flags can only turn things OFF on
    # top of it (so `--profile fast` is never silently re-enabled by defaults).
    base = CleanConfig.preset(args.profile)
    return base.with_overrides(
        downcast=not args.no_downcast,
        excel_sheet=args.sheet,
        impute_numeric=args.impute,
        impute_categorical=args.impute_categorical,
        knn_neighbors=args.knn_neighbors,
        outlier_methods=_parse_outliers(args.outliers),
        outlier_action=args.outlier_action,
        iqr_multiplier=args.iqr_multiplier,
        zscore_threshold=args.zscore_threshold,
        parse_datetimes=not args.no_parse_dates,
        categorical_case=args.categorical_case,
        corr_threshold=args.corr_threshold,
        missing_warn_threshold=args.missing_threshold,
        make_charts=base.make_charts and not args.no_charts,
        export_png=base.export_png and not args.no_png,
        make_pdf=base.make_pdf and not args.no_pdf,
        make_json=base.make_json and not args.no_json,
        save_model=base.save_model and not args.no_save_model,
        tune=args.tune,
        tune_trials=args.tune_trials,
        treatment=args.treatment,
        outcome=args.outcome,
        advanced=base.advanced and not args.no_advanced,
        target=args.target,
        apply_transforms=args.apply_transforms,
        inference=base.inference and not args.no_inference,
        modeling=base.modeling and not args.no_modeling,
        fda=base.fda and not args.no_fda,
        auto_specialize=base.auto_specialize and not args.no_specialize,
        streaming=args.streaming,
        random_seed=args.seed,
        verbose=not args.quiet,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI args, run the pipeline, return a process exit code."""
    args = _build_parser().parse_args(argv)
    config = _config_from_args(args)
    try:
        result = run_pipeline(
            args.input,
            args.output,
            config,
            query=args.query,
            table=args.table,
            compare=args.compare,
            contract=args.contract,
            report_html=args.report_html,
            report_markdown=args.report_md,
            title=args.title,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 — top-level guard for a clean CLI message
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.quiet:  # summary still printed once even in quiet mode
        print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
