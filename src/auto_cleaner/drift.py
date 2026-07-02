"""Dataset drift / comparison between two snapshots.

Compares a baseline dataset against a second one (e.g. train vs serving, or two
time snapshots) and quantifies how much each feature has shifted: the Population
Stability Index (PSI) plus a distributional test (Kolmogorov-Smirnov for numeric,
chi-square for categorical). The headline a monitoring engineer cares about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = ["DriftResult", "DriftReport", "compute_drift", "render_drift"]

_NUMERIC = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


@dataclass(slots=True)
class DriftResult:
    feature: str
    kind: str
    psi: float
    test: str
    p_value: float | None
    level: str


@dataclass(slots=True)
class DriftReport:
    n_a: int
    n_b: int
    results: list[DriftResult] = field(default_factory=list)

    @property
    def n_drifted(self) -> int:
        return sum(1 for r in self.results if r.level != "stable")


def _level(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "moderate drift"
    return "major drift"


def _psi_numeric(a, b, bins: int = 10) -> float:
    import numpy as np

    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e, _ = np.histogram(a, bins=edges)
    f, _ = np.histogram(b, bins=edges)
    e = np.clip(e / max(e.sum(), 1), 1e-6, None)
    f = np.clip(f / max(f.sum(), 1), 1e-6, None)
    return float(np.sum((f - e) * np.log(f / e)))


def _psi_categorical(a, b) -> tuple[float, float | None]:
    import numpy as np
    import pandas as pd
    from scipy.stats import chi2_contingency

    sa = pd.Series(a).value_counts()
    sb = pd.Series(b).value_counts()
    levels = sorted(set(sa.index) | set(sb.index))
    ea = np.array([sa.get(l, 0) for l in levels], dtype=float)
    eb = np.array([sb.get(l, 0) for l in levels], dtype=float)
    pe = np.clip(ea / max(ea.sum(), 1), 1e-6, None)
    pf = np.clip(eb / max(eb.sum(), 1), 1e-6, None)
    psi = float(np.sum((pf - pe) * np.log(pf / pe)))
    p = None
    try:
        table = np.vstack([ea, eb])
        table = table[:, table.sum(axis=0) > 0]
        if table.shape[1] >= 2:
            p = float(chi2_contingency(table)[1])
    except Exception:  # noqa: BLE001
        pass
    return psi, p


def compute_drift(df_a: pl.DataFrame, df_b: pl.DataFrame, config: CleanConfig | None = None) -> DriftReport:
    """Compute per-feature drift between baseline ``df_a`` and comparison ``df_b``."""
    config = config or CleanConfig()
    report = DriftReport(n_a=df_a.height, n_b=df_b.height)
    try:
        import numpy as np
        from scipy import stats
    except ImportError:
        return report

    common = [c for c in df_a.columns if c in df_b.columns and c != "is_outlier"]
    for c in common:
        ta, tb = df_a.get_column(c).dtype, df_b.get_column(c).dtype
        if ta in _NUMERIC and tb in _NUMERIC:
            a = df_a.get_column(c).drop_nulls().to_numpy().astype(float)
            b = df_b.get_column(c).drop_nulls().to_numpy().astype(float)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if a.size < 5 or b.size < 5:
                continue
            psi = _psi_numeric(a, b)
            try:
                p = float(stats.ks_2samp(a, b).pvalue)
            except Exception:  # noqa: BLE001
                p = None
            report.results.append(DriftResult(c, "numeric", round(psi, 4), "KS", p, _level(psi)))
        elif ta in (pl.Utf8, pl.Categorical, pl.Boolean):
            a = df_a.get_column(c).drop_nulls().to_list()
            b = df_b.get_column(c).drop_nulls().to_list()
            if len(a) < 5 or len(b) < 5:
                continue
            psi, p = _psi_categorical(a, b)
            report.results.append(DriftResult(c, "categorical", round(psi, 4), "chi-square", p, _level(psi)))
    report.results.sort(key=lambda r: -r.psi)
    return report


def render_drift(report: DriftReport, source_a: str, source_b: str) -> tuple[str, str]:
    """Render a drift report to ``(markdown, html)``."""
    md = [
        "# Dataset Drift Report", "",
        f"*Baseline:* `{source_a}` ({report.n_a:,} rows)  vs  *Comparison:* `{source_b}` ({report.n_b:,} rows)",
        "", f"**Features drifted:** {report.n_drifted} / {len(report.results)}", "",
        "| Feature | Kind | PSI | Test p | Status |", "|---|---|--:|--:|---|",
    ]
    body_rows = []
    for r in report.results:
        md.append(f"| {r.feature} | {r.kind} | {r.psi:.4f} | {('-' if r.p_value is None else f'{r.p_value:.3g}')} | {r.level} |")
        color = "#16a34a" if r.level == "stable" else ("#b54708" if r.level == "major drift" else "#9a6700")
        body_rows.append(
            f"<tr><td>{r.feature}</td><td>{r.kind}</td><td>{r.psi:.4f}</td>"
            f"<td>{'-' if r.p_value is None else f'{r.p_value:.3g}'}</td>"
            f"<td style='color:{color};font-weight:600'>{r.level}</td></tr>"
        )
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Drift Report</title>
<style>body{{font-family:-apple-system,Segoe UI,Arial,sans-serif;margin:32px;color:#1a2233}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #eef2f7;padding:6px 9px;text-align:right}}
th{{background:#f8fafc}}td:first-child,th:first-child,td:nth-child(2){{text-align:left}}h1{{font-size:22px}}</style></head>
<body><h1>Dataset Drift Report</h1>
<p>Baseline: <code>{source_a}</code> ({report.n_a:,} rows) &nbsp;vs&nbsp; Comparison: <code>{source_b}</code> ({report.n_b:,} rows)<br>
<b>Features drifted:</b> {report.n_drifted} / {len(report.results)}</p>
<table><thead><tr><th>Feature</th><th>Kind</th><th>PSI</th><th>Test p</th><th>Status</th></tr></thead>
<tbody>{''.join(body_rows)}</tbody></table>
<p style="color:#667085;font-size:12px">PSI &lt; 0.1 stable · 0.1–0.25 moderate · &gt; 0.25 major drift.</p>
</body></html>"""
    return "\n".join(md), html
