"""Readers that turn a detected source into a polars ``DataFrame``.

Supported: CSV/TSV (any delimiter), Parquet, JSON, NDJSON, and SQL via DuckDB
(database files *or* ad-hoc queries over CSV/Parquet). Every reader returns the
``DataFrame`` together with a :class:`StepReport` for the audit trail.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from auto_cleaner.config import CleanConfig
from auto_cleaner.ingest.detect import FileProfile, profile_source
from auto_cleaner.logging_utils import human_bytes, log
from auto_cleaner.reporting import StepReport

__all__ = ["read_any", "read_sql"]


def _null_token_variants(tokens: Iterable[str]) -> list[str]:
    """Expand null tokens into lower/upper/title-cased variants (CSV is literal)."""
    out: set[str] = set()
    for tok in tokens:
        out.update({tok, tok.lower(), tok.upper(), tok.capitalize()})
    return sorted(out)


def _read_text_bytes(profile: FileProfile) -> bytes:
    """Return file content as UTF-8 bytes, transcoding exotic encodings."""
    if profile.encoding in {"utf-8", "utf-8-sig"}:
        return profile.path.read_bytes()
    text = profile.path.read_text(encoding=profile.encoding, errors="replace")
    return text.encode("utf-8")


# Graceful-degradation ladder for malformed CSVs. Each rung re-reads with more
# tolerant options; the first success wins and anything past "string-fallback"
# is counted and surfaced as a warning — rows are never discarded silently.
_CSV_ATTEMPTS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("strict", {}),
    # Every column as Utf8 so mixed-type columns survive losslessly; the
    # cleaning stage re-parses numbers/dates deterministically.
    ("string-fallback", {"infer_schema_length": 0}),
    # Ragged lines: extra fields truncated, missing fields become null;
    # unparseable values become null instead of aborting the read.
    ("tolerant", {
        "infer_schema_length": 0,
        "truncate_ragged_lines": True,
        "ignore_errors": True,
    }),
    # Last resort for unclosed quotes: treat quote chars as literal text.
    ("tolerant-unquoted", {
        "infer_schema_length": 0,
        "truncate_ragged_lines": True,
        "ignore_errors": True,
        "quote_char": None,
    }),
)


def _report_csv_recovery(
    profile: FileProfile, df: pl.DataFrame, mode: str, report: StepReport
) -> None:
    """Count malformed source lines after a tolerant read and warn loudly."""
    malformed = dropped = 0
    n_data = df.height
    try:
        text = profile.path.read_text(encoding=profile.encoding, errors="replace")
        quoting = csv.QUOTE_NONE if mode == "tolerant-unquoted" else csv.QUOTE_MINIMAL
        reader = csv.reader(
            io.StringIO(text), delimiter=profile.separator or ",", quoting=quoting
        )
        counts = [len(row) for row in reader if row]
        if profile.has_header and counts:
            counts = counts[1:]
        n_data = len(counts)
        malformed = sum(1 for c in counts if c != df.width)
        dropped = max(0, n_data - df.height)
    except Exception:  # noqa: BLE001 — accounting must never break the read
        pass

    msg = (
        f"Malformed CSV recovered with tolerant parsing ('{mode}'): "
        f"{malformed} of {n_data:,} data line(s) had an unexpected field count "
        "(extra fields truncated, missing fields filled with null)"
    )
    if mode == "tolerant-unquoted":
        msg += "; quoting was disabled, quote characters are kept as literal text"
    msg += (
        f"; {dropped} row(s) could not be recovered and were dropped"
        if dropped
        else "; no rows were dropped"
    )
    report.warn(msg)
    report.measure("csv_malformed_lines", malformed)
    report.measure("csv_rows_dropped", dropped)
    log(msg, "WARN")


def _read_csv(profile: FileProfile, config: CleanConfig, report: StepReport) -> pl.DataFrame:
    """Read a delimited file, degrading through :data:`_CSV_ATTEMPTS` as needed."""
    source: str | bytes | Path
    if profile.encoding in {"utf-8", "utf-8-sig"}:
        source = profile.path
    else:
        source = _read_text_bytes(profile)  # already UTF-8 bytes

    common = dict(
        separator=profile.separator or ",",
        has_header=profile.has_header,
        null_values=_null_token_variants(config.csv_null_values),
        try_parse_dates=False,  # standardisation handles datetimes deterministically
        infer_schema_length=config.detection_sample_rows,
        ignore_errors=False,
        rechunk=True,
    )
    if config.streaming and isinstance(source, (str, Path)):
        try:
            df = _collect_streaming(
                pl.scan_csv(
                    source,
                    separator=profile.separator or ",",
                    has_header=profile.has_header,
                    null_values=_null_token_variants(config.csv_null_values),
                    try_parse_dates=False,
                    infer_schema_length=config.detection_sample_rows,
                    ignore_errors=False,
                )
            )
        except Exception:  # noqa: BLE001 — fall back to the eager ladder
            pass
        else:
            report.measure("csv_parse_mode", "strict-streaming")
            return df

    last_error: Exception | None = None
    for mode, overrides in _CSV_ATTEMPTS:
        try:
            df = pl.read_csv(source, **{**common, **overrides})
        except Exception as exc:  # noqa: BLE001 — try the next, more tolerant rung
            last_error = exc
            continue
        report.measure("csv_parse_mode", mode)
        if overrides.get("ignore_errors"):
            _report_csv_recovery(profile, df, mode, report)
        return df
    assert last_error is not None
    raise last_error


def _collect_streaming(lazy: "pl.LazyFrame") -> pl.DataFrame:
    """Collect a LazyFrame with the streaming engine (version-tolerant)."""
    try:
        return lazy.collect(engine="streaming")
    except TypeError:
        return lazy.collect(streaming=True)


def _read_excel(path: Path, config: CleanConfig, report: StepReport) -> pl.DataFrame:
    """Read one worksheet of an Excel workbook via the calamine engine.

    Multi-sheet workbooks are read from the first sheet by default, with a loud
    warning listing the alternatives — silently ignoring sheets is how numbers
    from the wrong tab end up in an analysis.
    """
    try:
        import fastexcel
    except ImportError as exc:  # pragma: no cover — exercised only without the extra
        raise ImportError(
            "Excel ingestion requires the 'fastexcel' package: "
            "pip install 'auto-cleaner[excel]'"
        ) from exc

    sheet_names = list(fastexcel.read_excel(str(path)).sheet_names)
    sheet = config.excel_sheet if config.excel_sheet is not None else sheet_names[0]
    if sheet not in sheet_names:
        raise ValueError(
            f"Worksheet '{sheet}' not found in {path.name}; available: {sheet_names}"
        )
    df = pl.read_excel(path, sheet_name=sheet, engine="calamine")
    report.measure("excel_sheet", sheet)
    report.measure("excel_sheets_available", sheet_names)
    if len(sheet_names) > 1 and config.excel_sheet is None:
        msg = (
            f"Workbook has {len(sheet_names)} sheets {sheet_names}; reading "
            f"'{sheet}' — pass --sheet (or CleanConfig.excel_sheet) to pick another"
        )
        report.warn(msg)
        log(msg, "WARN")
    return df


def _read_fits(path: Path) -> pl.DataFrame:
    """Read the first table HDU of a FITS (astronomy) file into polars via astropy."""
    from astropy.table import Table  # local import; astropy is an optional dependency

    table = Table.read(str(path))
    data: dict[str, list] = {}
    for name in table.colnames:
        col = table[name]
        if getattr(col, "ndim", 1) != 1:
            continue  # skip multi-dimensional / vector columns
        decoded = [
            (v.decode("utf-8", "replace") if isinstance(v, bytes) else v) for v in col.tolist()
        ]
        data[name] = decoded
    if not data:
        raise ValueError(f"No 1-D table columns found in FITS file: {path}")
    return pl.DataFrame(data)


def _read_netcdf(path: Path) -> pl.DataFrame:
    """Flatten a netCDF / HDF5 grid (climate, geoscience) into a long polars table.

    xarray + pandas are used only as a bridge here; the cleaned data continues
    through the polars engine as usual.
    """
    import xarray as xr  # optional dependency

    ds = xr.open_dataset(path)
    try:
        flat = ds.to_dataframe().reset_index()
    finally:
        ds.close()
    return pl.from_pandas(flat)


def read_sql(
    source: str | Path,
    *,
    query: str | None = None,
    table: str | None = None,
    config: CleanConfig | None = None,
) -> tuple[pl.DataFrame, StepReport]:
    """Execute a DuckDB query (or read a table) and return a polars frame.

    Parameters
    ----------
    source:
        Path to a DuckDB/SQLite database file, or ``":memory:"`` for ad-hoc
        queries over external files (``read_csv_auto`` / ``read_parquet``).
    query:
        Explicit SQL. Takes precedence over ``table``.
    table:
        Table name to ``SELECT *`` from when ``query`` is not given.

    Notes
    -----
    DuckDB streams the result straight into Arrow, which polars adopts
    zero-copy — so even large SQL results stay fast and memory-lean.
    """
    import duckdb  # local import keeps duckdb optional for non-SQL workflows

    config = config or CleanConfig()
    report = StepReport(step="ingest")
    con = duckdb.connect(str(source))
    try:
        if query is None:
            if table is None:
                tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
                if not tables:
                    raise ValueError(f"No tables found in DuckDB source: {source}")
                table = tables[0]
                if len(tables) > 1:
                    log(f"Multiple tables {tables}; defaulting to '{table}'.", "WARN")
            query = f'SELECT * FROM "{table}"'
        arrow_tbl = con.execute(query).arrow()
    finally:
        con.close()

    df = pl.from_arrow(arrow_tbl)
    if isinstance(df, pl.Series):  # single-column edge case
        df = df.to_frame()
    report.act(f"DuckDB query returned {df.height:,} rows × {df.width} cols")
    report.measure("source_format", "sql")
    report.measure("query", query)
    return df, report


def read_any(
    source: str | Path,
    config: CleanConfig | None = None,
    *,
    query: str | None = None,
    table: str | None = None,
) -> tuple[pl.DataFrame, StepReport]:
    """Open *any* supported source into a polars ``DataFrame``.

    The format, encoding, delimiter and header are auto-detected. ``query`` /
    ``table`` are forwarded to the SQL path for DuckDB sources.

    Returns
    -------
    (DataFrame, StepReport)
        The loaded frame and an ingestion report (shape, memory, format).
    """
    config = config or CleanConfig()

    # Ad-hoc SQL over a non-database file: source is ':memory:' + a query.
    if str(source) == ":memory:" or query is not None and Path(str(source)).suffix == "":
        return read_sql(source, query=query, table=table, config=config)

    profile: FileProfile = profile_source(source, sample_rows=config.detection_sample_rows)
    report = StepReport(step="ingest")
    report.measure("source_format", profile.fmt)

    if profile.fmt == "duckdb":
        df, sql_report = read_sql(profile.path, query=query, table=table, config=config)
        report.actions = sql_report.actions
        report.metrics.update(sql_report.metrics)
        report.metrics["source_format"] = "duckdb"
    elif profile.fmt == "parquet":
        df = (
            _collect_streaming(pl.scan_parquet(profile.path))
            if config.streaming
            else pl.read_parquet(profile.path)
        )
    elif profile.fmt == "excel":
        df = _read_excel(profile.path, config, report)
    elif profile.fmt == "fits":
        df = _read_fits(profile.path)
    elif profile.fmt == "netcdf":
        df = _read_netcdf(profile.path)
    elif profile.fmt == "ndjson":
        df = pl.read_ndjson(profile.path)
    elif profile.fmt == "json":
        df = pl.read_json(profile.path)
    else:  # csv / tsv / unknown-delimited
        df = _read_csv(profile, config, report)
        report.measure("separator", profile.separator)
        report.measure("encoding", profile.encoding)
        report.measure("has_header", profile.has_header)

    report.measure("rows", df.height)
    report.measure("cols", df.width)
    report.measure("memory_bytes", int(df.estimated_size()))
    report.act(
        f"Ingested {profile.fmt.upper()} → {df.height:,} rows × {df.width} cols "
        f"({human_bytes(df.estimated_size())} in memory)"
    )
    log(report.actions[-1], "OK", enabled=config.verbose)
    return df, report
