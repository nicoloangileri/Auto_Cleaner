"""Input & schema validation with typed, actionable errors.

Part of the production-hardening surface: fail fast and clearly on malformed
input, and surface non-fatal data issues as structured warnings rather than
letting them silently corrupt downstream analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = [
    "AutoCleanerError", "IngestionError", "ValidationError",
    "ValidationReport", "validate_source", "validate_frame", "enforce_schema",
    "load_contract", "enforce_contract",
]

_CONTRACT_NUMERIC = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


class AutoCleanerError(Exception):
    """Base class for all auto_cleaner errors."""


class IngestionError(AutoCleanerError):
    """Raised when a source cannot be opened or is unusable."""


class ValidationError(AutoCleanerError):
    """Raised when a DataFrame fails a fatal validation check."""


@dataclass(slots=True)
class ValidationReport:
    """Non-fatal data-quality issues surfaced before analysis."""

    issues: list[str] = field(default_factory=list)


def validate_source(path: str | Path) -> None:
    """Fail fast if a file source is missing, a directory, or empty."""
    if str(path) == ":memory:":
        return
    p = Path(path)
    if not p.exists():
        raise IngestionError(f"Input source does not exist: {p}")
    if p.is_dir():
        raise IngestionError(f"Input source is a directory, expected a file: {p}")
    if p.stat().st_size == 0:
        raise IngestionError(f"Input source is empty (0 bytes): {p}")


def validate_frame(df: pl.DataFrame, config: CleanConfig | None = None) -> ValidationReport:
    """Validate a loaded frame; raise on fatal problems, collect soft issues."""
    config = config or CleanConfig()
    if df.width == 0:
        raise ValidationError("Loaded dataset has no columns.")
    if df.height == 0:
        raise ValidationError("Loaded dataset has no rows.")

    report = ValidationReport()
    counts: dict[str, int] = {}
    for c in df.columns:
        counts[c] = counts.get(c, 0) + 1
    duplicate_names = sorted(name for name, n in counts.items() if n > 1)
    if duplicate_names:
        report.issues.append(f"Duplicate column names detected: {duplicate_names}")

    for c in df.columns:
        if df.get_column(c).null_count() == df.height:
            report.issues.append(f"Column '{c}' is entirely null.")

    if df.height == 1:
        report.issues.append("Dataset has a single row — most statistics will be undefined.")
    return report


def enforce_schema(df: pl.DataFrame, expected: dict[str, pl.DataType] | None) -> None:
    """Optionally enforce a column->dtype contract; raise on mismatch.

    ``expected`` maps column name to a polars dtype. Missing columns or dtype
    mismatches raise :class:`ValidationError` (a data-contract guard for
    pipelines that must not drift).
    """
    if not expected:
        return
    actual = dict(zip(df.columns, df.dtypes))
    missing = [c for c in expected if c not in actual]
    if missing:
        raise ValidationError(f"Schema mismatch — missing columns: {missing}")
    wrong = [
        f"{c}: expected {expected[c]}, got {actual[c]}"
        for c in expected
        if actual[c] != expected[c]
    ]
    if wrong:
        raise ValidationError("Schema mismatch — " + "; ".join(wrong))


def load_contract(path: str | Path) -> dict:
    """Load a YAML data contract (``columns: {name: {dtype, required, min, ...}}``)."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise IngestionError("Data contracts require PyYAML (pip install pyyaml).") from exc
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def enforce_contract(
    df: pl.DataFrame, contract: dict, config: CleanConfig | None = None
) -> ValidationReport:
    """Validate ``df`` against a data contract.

    Required-but-missing columns raise :class:`ValidationError`; dtype, null-rate,
    range and allowed-value violations are collected as soft issues.
    """
    config = config or CleanConfig()
    report = ValidationReport()
    columns = (contract or {}).get("columns", {})
    actual = dict(zip(df.columns, df.dtypes))
    for name, spec in columns.items():
        spec = spec or {}
        if name not in actual:
            if spec.get("required", True):
                raise ValidationError(f"Contract violation: required column '{name}' is missing")
            report.issues.append(f"Contract: optional column '{name}' is absent")
            continue
        s = df.get_column(name)
        expected = spec.get("dtype")
        if expected and str(actual[name]).lower() != str(expected).lower():
            report.issues.append(f"Contract: '{name}' dtype {actual[name]} != expected {expected}")
        max_null = spec.get("max_null_pct")
        if max_null is not None and df.height:
            null_pct = s.null_count() / df.height * 100
            if null_pct > float(max_null):
                report.issues.append(f"Contract: '{name}' is {null_pct:.1f}% null (> {max_null}% allowed)")
        if actual[name] in _CONTRACT_NUMERIC:
            if "min" in spec and s.min() is not None and s.min() < spec["min"]:
                report.issues.append(f"Contract: '{name}' min {s.min()} < {spec['min']}")
            if "max" in spec and s.max() is not None and s.max() > spec["max"]:
                report.issues.append(f"Contract: '{name}' max {s.max()} > {spec['max']}")
        allowed = spec.get("allowed")
        if allowed is not None:
            seen = {str(v) for v in s.drop_nulls().unique().to_list()}
            extra = sorted(seen - {str(a) for a in allowed})
            if extra:
                report.issues.append(f"Contract: '{name}' has unexpected values {extra[:5]}")
    return report
