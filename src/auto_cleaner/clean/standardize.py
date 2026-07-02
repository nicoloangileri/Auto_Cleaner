"""Feature standardisation.

Turns raw text columns into clean, correctly-typed features:

* strip & collapse whitespace, empty-string → null;
* coerce numeric-looking strings (``"$1,234.50"``, ``"45%"``) into floats;
* parse messy datetime strings into native ``Date`` / ``Datetime`` dtypes;
* normalise categorical text casing.

All detection is fraction-based on a sample, so a single stray value never
forces (or blocks) a conversion.
"""

from __future__ import annotations

import polars as pl

from auto_cleaner.config import CleanConfig
from auto_cleaner.logging_utils import log
from auto_cleaner.reporting import StepReport

__all__ = ["standardize"]

# A value is "numeric-like" if it is an optionally-signed, optionally-currency,
# thousands-separated number with an optional percent sign.
_NUMERIC_RE = (
    r"^[+-]?[$€£]?\d{1,3}(,\d{3})+(\.\d+)?%?$"  # 1,234 / 1,234.56 / 12,000%
    r"|^[+-]?[$€£]?\d*\.?\d+%?$"                # 12 / 12.5 / .5 / 45%
)
# A value is "date-like" if it contains a date separator pattern or a month word.
_DATE_HINT_RE = (
    r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|(?i)(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
)
# Ordered: the first format that parses enough values wins, so put the least
# ambiguous (ISO) first.
_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d %b %Y",
    "%b %d, %Y",
    "%Y%m%d",
)


def _string_columns(df: pl.DataFrame) -> list[str]:
    return [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Utf8]


def _match_fraction(series: pl.Series, pattern: str, *, sample: int = 2000) -> float:
    """Fraction of non-null values (sampled) matching ``pattern``."""
    nn = series.drop_nulls()
    if nn.len() == 0:
        return 0.0
    head = nn.head(sample)
    return float(head.str.contains(pattern).sum()) / head.len()


def _strip_whitespace(df: pl.DataFrame, report: StepReport) -> pl.DataFrame:
    cols = _string_columns(df)
    if not cols:
        return df
    exprs = []
    for c in cols:
        cleaned = pl.col(c).str.strip_chars().str.replace_all(r"\s+", " ")
        exprs.append(
            pl.when(cleaned.str.len_chars() == 0).then(None).otherwise(cleaned).alias(c)
        )
    report.act(f"Stripped/collapsed whitespace on {len(cols)} text column(s)")
    return df.with_columns(exprs)


def _parse_numeric_strings(
    df: pl.DataFrame, config: CleanConfig, report: StepReport
) -> pl.DataFrame:
    converted: list[str] = []
    exprs = []
    for c in _string_columns(df):
        if _match_fraction(df.get_column(c), _NUMERIC_RE) >= config.numeric_string_min_success:
            base = pl.col(c).str.replace_all(r"[,$€£\s]", "")
            is_pct = pl.col(c).str.contains("%")
            num = base.str.replace_all("%", "").cast(pl.Float64, strict=False)
            exprs.append(
                pl.when(is_pct).then(num / 100.0).otherwise(num).alias(c)
            )
            converted.append(c)
    if exprs:
        df = df.with_columns(exprs)
        report.act(f"Parsed numeric strings → Float64: {converted}")
        report.measure("numeric_string_columns", converted)
    return df


def _best_datetime_format(series: pl.Series, min_success: float) -> tuple[str | None, bool]:
    """Return ``(format, has_time)`` for the first format that parses enough rows."""
    nn = series.drop_nulls()
    if nn.len() == 0:
        return None, False
    denom = nn.len()
    for fmt in _DATE_FORMATS:
        has_time = "%H" in fmt
        try:
            if has_time:
                parsed = nn.str.to_datetime(format=fmt, strict=False)
            else:
                parsed = nn.str.to_date(format=fmt, strict=False)
        except Exception:  # noqa: BLE001
            continue
        if parsed.is_not_null().sum() / denom >= min_success:
            return fmt, has_time
    return None, False


def _parse_datetimes(
    df: pl.DataFrame, config: CleanConfig, report: StepReport
) -> pl.DataFrame:
    converted: list[str] = []
    exprs = []
    for c in _string_columns(df):
        series = df.get_column(c)
        if _match_fraction(series, _DATE_HINT_RE) < 0.60:
            continue
        fmt, has_time = _best_datetime_format(series, config.datetime_parse_min_success)
        if fmt is None:
            continue
        if has_time:
            exprs.append(pl.col(c).str.to_datetime(format=fmt, strict=False).alias(c))
        else:
            exprs.append(pl.col(c).str.to_date(format=fmt, strict=False).alias(c))
        converted.append(f"{c} ({fmt})")
    if exprs:
        df = df.with_columns(exprs)
        report.act(f"Parsed datetimes: {converted}")
        report.measure("datetime_columns", converted)
    return df


def _standardize_categoricals(
    df: pl.DataFrame, config: CleanConfig, report: StepReport
) -> pl.DataFrame:
    if config.categorical_case == "none":
        return df
    transform = {
        "lower": lambda e: e.str.to_lowercase(),
        "upper": lambda e: e.str.to_uppercase(),
        "title": lambda e: e.str.to_titlecase(),
    }[config.categorical_case]
    cols = _string_columns(df)
    if not cols:
        return df
    df = df.with_columns([transform(pl.col(c)).alias(c) for c in cols])
    report.act(f"Normalised categorical casing → '{config.categorical_case}' on {len(cols)} col(s)")
    return df


def standardize(df: pl.DataFrame, config: CleanConfig | None = None) -> tuple[pl.DataFrame, StepReport]:
    """Run the full standardisation stage.

    Order matters: whitespace first (so detection sees clean tokens), then
    numeric coercion (removes numbers from the text pool), then datetime
    parsing, then categorical casing on whatever text remains.
    """
    config = config or CleanConfig()
    report = StepReport(step="standardize")

    if config.strip_whitespace:
        df = _strip_whitespace(df, report)
    if config.parse_numeric_strings:
        df = _parse_numeric_strings(df, config, report)
    if config.parse_datetimes:
        df = _parse_datetimes(df, config, report)
    if config.standardize_categoricals:
        df = _standardize_categoricals(df, config, report)

    for action in report.actions:
        log(action, "INFO", enabled=config.verbose)
    return df, report
