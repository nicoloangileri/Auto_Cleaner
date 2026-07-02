"""End-to-end orchestration: messy source in, pristine data + EDA report out.

The pipeline is a *function composition*, not an object graph::

    ingest → standardize → impute → outliers → downcast → profile → report

Each arrow is a pure ``DataFrame -> (DataFrame, StepReport)`` function from the
sub-packages. :func:`run_pipeline` simply threads the frame through them and
collects the reports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from auto_cleaner.clean import downcast, handle_outliers, impute_missing, standardize
from auto_cleaner.config import CleanConfig
from auto_cleaner.eda import (
    build_charts,
    build_report,
    charts_to_html,
    export_pngs,
    profile_dataset,
    write_reports,
)
from auto_cleaner.analyze import apply_transforms, run_advanced
from auto_cleaner.functional import run_fda
from auto_cleaner.inference import run_inference
from auto_cleaner.modeling import run_baseline_models
from auto_cleaner.causal import causal_analysis
from auto_cleaner.specialize import detect_specialization
from auto_cleaner.stats import run_extended
from auto_cleaner.tuning import tune_model
from auto_cleaner.validate import validate_frame, validate_source
from auto_cleaner.eda.stats import DatasetProfile
from auto_cleaner.ingest import read_any
from auto_cleaner.logging_utils import human_bytes, log, timed
from auto_cleaner.reporting import StepReport

__all__ = ["PipelineResult", "run_pipeline"]


@dataclass(slots=True)
class PipelineResult:
    """Everything the pipeline produced — data handle, reports and metrics."""

    frame: pl.DataFrame
    profile: DatasetProfile
    step_reports: list[StepReport]
    source: str
    output_path: str | None
    report_paths: dict[str, str]
    rows_in: int
    rows_out: int
    memory_before: int
    memory_after: int
    elapsed_s: float
    advanced: Any = None
    drift: Any = None

    def summary(self) -> str:
        """Compact, human-readable run summary (for logs / CLI tail)."""
        saved = self.memory_before - self.memory_after
        pct = (saved / self.memory_before * 100.0) if self.memory_before else 0.0
        lines = [
            "─" * 56,
            "  auto_cleaner — run summary",
            "─" * 56,
            f"  source            : {self.source}",
            f"  rows  in → out    : {self.rows_in:,} → {self.rows_out:,}",
            f"  memory in → out   : {human_bytes(self.memory_before)} → "
            f"{human_bytes(self.memory_after)}  (−{pct:.1f}%)",
            f"  warnings          : {len(self.profile.warnings)}",
            f"  output            : {self.output_path or '(not written)'}",
        ]
        for fmt, path in self.report_paths.items():
            lines.append(f"  report ({fmt:<8}): {path}")
        lines.append(f"  elapsed           : {self.elapsed_s:.2f}s")
        lines.append("─" * 56)
        return "\n".join(lines)


def _write_output(df: pl.DataFrame, path: Path) -> None:
    """Dispatch on extension to the right polars writer (Parquet by default)."""
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix in {".csv", ".tsv"}:
        df.write_csv(path, separator="\t" if suffix == ".tsv" else ",")
    elif suffix in {".json", ".ndjson"}:
        df.write_ndjson(path)
    elif suffix in {".arrow", ".ipc", ".feather"}:
        df.write_ipc(path)
    else:  # .parquet / .pq / anything else → Parquet (columnar, compressed)
        df.write_parquet(path, compression="zstd")


def run_pipeline(
    source: str | Path,
    output: str | Path | None = None,
    config: CleanConfig | None = None,
    *,
    query: str | None = None,
    table: str | None = None,
    compare: str | Path | None = None,
    contract: str | Path | None = None,
    report_html: str | Path | None = None,
    report_markdown: str | Path | None = None,
    write_reports_to_disk: bool = True,
    title: str | None = None,
) -> PipelineResult:
    """Run the full preprocessing + EDA pipeline.

    Parameters
    ----------
    source:
        Path to the input (CSV/Parquet/JSON/DuckDB) or ``":memory:"`` for SQL.
    output:
        Where to write the cleaned dataset. Extension selects the format
        (``.parquet`` default, also ``.csv`` / ``.ndjson`` / ``.arrow``).
        ``None`` skips writing the data.
    config:
        A :class:`CleanConfig`; defaults are production-sane.
    query, table:
        Forwarded to the DuckDB SQL reader when ``source`` is a database.
    report_html, report_markdown:
        Explicit report destinations. If omitted, they default to
        ``<output stem>_eda.html`` / ``.md`` beside the output.
    title:
        Report title.

    Returns
    -------
    PipelineResult
        Cleaned frame, dataset profile, per-step reports and timing/memory metrics.
    """
    config = config or CleanConfig()
    source_str = str(source)
    title = title or f"Automated EDA Report — {Path(source_str).name}"
    start = time.perf_counter()
    log(f"auto_cleaner starting on '{source_str}'", "INFO", enabled=config.verbose)

    reports: list[StepReport] = []
    validate_source(source)

    with timed("ingest", enabled=config.verbose):
        df, ingest_rep = read_any(source, config, query=query, table=table)
    reports.append(ingest_rep)
    for issue in validate_frame(df, config).issues:
        log(issue, "WARN", enabled=config.verbose)
    if contract is not None:
        from auto_cleaner.validate import enforce_contract, load_contract

        for issue in enforce_contract(df, load_contract(contract), config).issues:
            log(f"[contract] {issue}", "WARN", enabled=config.verbose)
    rows_in = df.height
    memory_before = int(ingest_rep.metrics.get("memory_bytes", df.estimated_size()))

    with timed("standardize", enabled=config.verbose):
        df, rep = standardize(df, config)
    reports.append(rep)

    # polars frames are immutable, so pre-step snapshots are free references;
    # measure_impact quantifies the distributional footprint of each step.
    from auto_cleaner.clean.impact import measure_impact

    with timed("impute", enabled=config.verbose):
        before = df
        df, rep = impute_missing(df, config)
        measure_impact(before, df, rep)
    reports.append(rep)

    with timed("outliers", enabled=config.verbose):
        before = df
        df, rep = handle_outliers(df, config)
        measure_impact(before, df, rep)
    reports.append(rep)

    with timed("downcast", enabled=config.verbose):
        df, downcast_rep = downcast(df, config)
    reports.append(downcast_rep)
    memory_after = int(downcast_rep.metrics.get("memory_after", df.estimated_size()))

    with timed("eda-profile", enabled=config.verbose):
        profile = profile_dataset(df, config)

    # ---- advanced analysis (normality, transforms, VIF/PCA, relevance) -----
    advanced = None
    if config.advanced:
        with timed("advanced-analysis", enabled=config.verbose):
            advanced = run_advanced(df, config, target=config.target)
            if config.apply_transforms and advanced.transforms:
                df, transform_rep = apply_transforms(df, advanced.transforms, config)
                reports.append(transform_rep)
                profile = profile_dataset(df, config)  # re-profile with new columns

            # Auto-specialisation drives which extended modules run.
            spec = detect_specialization(df, config, target=config.target) if config.auto_specialize else None
            advanced.specialization = spec
            id_cols = spec.id_columns if spec is not None else None
            if config.inference:
                advanced.inference = run_inference(df, config, target=config.target, id_columns=id_cols)
            if config.modeling and config.target:
                advanced.modeling = run_baseline_models(df, config.target, config, id_columns=id_cols)
            if config.fda and spec is not None and spec.time_index is not None:
                advanced.fda = run_fda(df, spec.time_index, config, id_columns=id_cols)
            if config.extended_stats:
                advanced.extended = run_extended(
                    df, config, spec=spec, target=config.target, id_columns=id_cols
                )
            if config.tune and config.target:
                advanced.tuning = tune_model(df, config.target, config, id_columns=id_cols)
            if config.treatment and config.outcome:
                advanced.causal = causal_analysis(
                    df, config.treatment, config.outcome, config, id_columns=id_cols
                )
            if spec is not None:
                log(
                    f"Auto-specialisation: {spec.primary} → modules {spec.auto_modules}",
                    "OK", enabled=config.verbose,
                )

    # ---- write cleaned data -------------------------------------------------
    output_path: str | None = None
    if output is not None:
        out = Path(output)
        _write_output(df, out)
        output_path = str(out)
        log(f"Clean dataset written → {out}", "OK", enabled=config.verbose)

    # ---- build interactive charts + standalone PNGs ------------------------
    charts_head = charts_body = ""
    chart_pngs: list[tuple[str, str]] = []
    chart_pngs_abs: list[tuple[str, str]] = []
    if config.make_charts:
        with timed("charts", enabled=config.verbose):
            charts = build_charts(df, profile, config)
            charts_head, charts_body = charts_to_html(charts)
            asset_base: Path | None = None
            for candidate in (output, report_html, report_markdown):
                if candidate is not None:
                    asset_base = Path(candidate).parent
                    break
            if config.export_png and asset_base is not None and charts:
                png_map = export_pngs(charts, asset_base / "charts", config)
                chart_pngs = [
                    (ch.title, f"charts/{ch.name}.png") for ch in charts if ch.name in png_map
                ]
                chart_pngs_abs = [
                    (ch.title, png_map[ch.name]) for ch in charts if ch.name in png_map
                ]

    # ---- build & write reports ---------------------------------------------
    from auto_cleaner.eda.summary import build_summary

    summary_lines = build_summary(profile, reports, advanced)
    md, html_doc = build_report(
        profile, reports, title=title, source_name=source_str, config=config,
        charts_head=charts_head, charts_body=charts_body, chart_pngs=chart_pngs,
        advanced=advanced, summary_lines=summary_lines,
    )
    report_paths: dict[str, str] = {}
    if write_reports_to_disk:
        if report_html is None and report_markdown is None and output is not None:
            stem = Path(output).with_suffix("")
            report_html = f"{stem}_eda.html"
            report_markdown = f"{stem}_eda.md"
        report_paths = write_reports(
            md, html_doc, markdown_path=report_markdown, html_path=report_html
        )
        for fmt, path in report_paths.items():
            log(f"EDA report ({fmt}) written → {path}", "OK", enabled=config.verbose)

        if config.make_pdf and output is not None:
            from auto_cleaner.pdfreport import build_pdf

            stem = Path(output).with_suffix("")
            pdf_written = build_pdf(
                out_path=f"{stem}_report.pdf", title=title, source_name=source_str,
                profile=profile, advanced=advanced, chart_png_paths=chart_pngs_abs,
                config=config, summary_lines=summary_lines,
            )
            if pdf_written:
                report_paths["pdf"] = pdf_written
                log(f"PDF report written → {pdf_written}", "OK", enabled=config.verbose)

    # ---- machine-readable results + reproducibility manifest ---------------
    if config.make_json and output is not None:
        from auto_cleaner.reproducibility import build_results, write_results_json

        stem = Path(output).with_suffix("")
        data = build_results(
            profile=profile, advanced=advanced, config=config, source=source_str,
            rows_in=rows_in, rows_out=df.height, memory_before=memory_before,
            memory_after=memory_after, elapsed_s=time.perf_counter() - start,
        )
        jpath = write_results_json(f"{stem}_results.json", data)
        report_paths["json"] = jpath
        log(f"Results JSON written → {jpath}", "OK", enabled=config.verbose)

    # ---- exportable model bundle (productionisation) -----------------------
    if config.save_model and config.target and output is not None:
        from auto_cleaner.persistence import save_bundle, train_exportable

        id_cols2 = getattr(getattr(advanced, "specialization", None), "id_columns", None)
        bundle = train_exportable(df, config.target, config, id_columns=id_cols2)
        if bundle is not None:
            stem = Path(output).with_suffix("")
            mpath = save_bundle(bundle, f"{stem}_model.joblib")
            report_paths["model"] = mpath
            log(f"Model bundle saved → {mpath}", "OK", enabled=config.verbose)

    # ---- optional drift comparison against a second dataset ----------------
    drift = None
    if compare is not None:
        from auto_cleaner.drift import compute_drift, render_drift

        with timed("drift", enabled=config.verbose):
            df_b, _ = read_any(compare, config)
            df_b, _ = standardize(df_b, config)
            df_b, _ = impute_missing(df_b, config)
            df_b, _ = downcast(df_b, config)
            drift = compute_drift(df, df_b, config)
        if write_reports_to_disk and output is not None:
            stem = Path(output).with_suffix("")
            dmd, dhtml = render_drift(drift, source_str, str(compare))
            write_reports(dmd, dhtml, markdown_path=f"{stem}_drift.md", html_path=f"{stem}_drift.html")
            report_paths["drift_html"] = f"{stem}_drift.html"
            log(f"Drift report → {stem}_drift.html ({drift.n_drifted} features drifted)", "OK", enabled=config.verbose)

    elapsed = time.perf_counter() - start
    result = PipelineResult(
        frame=df,
        profile=profile,
        step_reports=reports,
        source=source_str,
        output_path=output_path,
        report_paths=report_paths,
        rows_in=rows_in,
        rows_out=df.height,
        memory_before=memory_before,
        memory_after=memory_after,
        elapsed_s=elapsed,
        advanced=advanced,
        drift=drift,
    )
    if config.verbose:
        print(result.summary())
    return result
