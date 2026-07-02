"""Quantify how much a cleaning step distorted each column.

Imputation and outlier treatment *change the data*. Logging "imputed 6 nulls
via median" says what happened but not what it did to the distribution — and
that is the question an analyst must answer before trusting a cleaned column.
This module compares each numeric column before/after a step and reports:

- how many cells changed (and the share of the column),
- the mean/std shift, expressed in pre-cleaning standard deviations,
- a two-sample Kolmogorov–Smirnov statistic between the pre-cleaning
  (non-null) values and the post-cleaning values,
- a conservative verdict: ``negligible`` / ``minor`` / ``material``.

Material verdicts are escalated to data-health warnings so they cannot be
missed in the report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from auto_cleaner.reporting import StepReport

__all__ = ["ColumnImpact", "measure_impact"]

_NUMERIC_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)

# Verdict thresholds (shares / sd units / KS distance). Deliberately strict:
# it is safer to over-flag a shift than to bless a distorted column.
_MATERIAL_SHARE = 0.10
_MATERIAL_MEAN_SD = 0.10
_MATERIAL_KS = 0.15
_NEGLIGIBLE_SHARE = 0.01
_NEGLIGIBLE_MEAN_SD = 0.02
_NEGLIGIBLE_KS = 0.05


@dataclass(slots=True)
class ColumnImpact:
    """Distributional footprint of one cleaning step on one column."""

    column: str
    cells_changed: int
    change_share: float
    mean_before: float | None
    mean_after: float | None
    std_before: float | None
    std_after: float | None
    mean_shift_sd: float | None
    ks_stat: float | None
    verdict: str

    def describe(self) -> str:
        """One human line, e.g. for the report's impact table caption."""
        bits = [f"'{self.column}': {self.cells_changed} cell(s) changed "
                f"({self.change_share:.1%})"]
        if self.mean_shift_sd is not None:
            bits.append(f"mean shift {self.mean_shift_sd:.3f} sd")
        if self.ks_stat is not None:
            bits.append(f"KS {self.ks_stat:.3f}")
        bits.append(f"→ {self.verdict}")
        return ", ".join(bits)


def _cells_changed(before: pl.Series, after: pl.Series) -> int:
    filled = (before.is_null() & after.is_not_null()).sum()
    both = before.is_not_null() & after.is_not_null()
    altered = (both & (before != after).fill_null(False)).sum()
    return int(filled) + int(altered)


def _ks_stat(before: pl.Series, after: pl.Series) -> float | None:
    """Two-sample KS distance; None when scipy or the data are unavailable."""
    try:
        from scipy import stats
    except ImportError:
        return None
    b = before.drop_nulls().to_numpy()
    a = after.drop_nulls().to_numpy()
    if len(b) < 8 or len(a) < 8:
        return None
    return float(stats.ks_2samp(b, a).statistic)


def _verdict(share: float, mean_sd: float | None, ks: float | None) -> str:
    mean_sd = mean_sd or 0.0
    ks = ks or 0.0
    if share >= _MATERIAL_SHARE or mean_sd >= _MATERIAL_MEAN_SD or ks >= _MATERIAL_KS:
        return "material"
    if share < _NEGLIGIBLE_SHARE and mean_sd < _NEGLIGIBLE_MEAN_SD and ks < _NEGLIGIBLE_KS:
        return "negligible"
    return "minor"


def measure_impact(
    before: pl.DataFrame,
    after: pl.DataFrame,
    report: StepReport,
) -> list[ColumnImpact]:
    """Compare numeric columns across a cleaning step and record the results.

    Impacts land in ``report.metrics["impact"]`` (machine-readable) and any
    *material* distortion is escalated to ``report.warn`` so the report's
    Data-Health section surfaces it. Row-dropping steps are compared on the
    surviving distribution (heights may differ).
    """
    impacts: list[ColumnImpact] = []
    rows_dropped = before.height - after.height
    if rows_dropped:
        report.measure("rows_dropped", rows_dropped)

    for col, dtype in zip(before.columns, before.dtypes):
        if dtype not in _NUMERIC_DTYPES or col not in after.columns:
            continue
        b, a = before.get_column(col), after.get_column(col)
        same_height = b.len() == a.len()
        changed = _cells_changed(b, a) if same_height else rows_dropped
        if changed == 0:
            continue

        mean_b, mean_a = b.mean(), a.mean()
        std_b, std_a = b.std(), a.std()
        mean_shift = (
            abs(mean_a - mean_b) / std_b
            if mean_b is not None and mean_a is not None and std_b
            else None
        )
        share = changed / b.len() if b.len() else 0.0
        ks = _ks_stat(b, a)
        impact = ColumnImpact(
            column=col,
            cells_changed=changed,
            change_share=share,
            mean_before=mean_b, mean_after=mean_a,
            std_before=std_b, std_after=std_a,
            mean_shift_sd=mean_shift,
            ks_stat=ks,
            verdict=_verdict(share, mean_shift, ks),
        )
        impacts.append(impact)
        if impact.verdict == "material":
            report.warn(
                f"Cleaning materially changed '{col}' — {impact.describe()}. "
                "Review before trusting this column."
            )

    if impacts:
        report.metrics["impact"] = [asdict(i) for i in impacts]
    return impacts
