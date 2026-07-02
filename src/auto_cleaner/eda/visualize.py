"""Interactive EDA visualisations (Plotly), with standalone PNG export.

Auto-generates a professional chart suite from a cleaned ``polars`` frame:

* **Missingness** — null-% per column;
* **Distributions** — a histogram per numeric feature (frequencies);
* **Frequencies** — top-K bar charts for categorical features;
* **Outliers** — standardised boxplots across numeric features;
* **Scatterplot matrix (SPLOM)** — pairwise numeric relationships, hued by a
  categorical when one is available;
* **Correlation heatmap** — Pearson matrix.

Plotly is imported lazily so the core engine never hard-depends on it. PNG
export uses ``kaleido`` and degrades gracefully if it is unavailable. Note
that kaleido 1.x no longer bundles Chromium: a headless Chrome must be
installed once (``plotly_get_chrome``) or every ``write_image`` call fails.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from auto_cleaner.config import CleanConfig
from auto_cleaner.eda.stats import DatasetProfile
from auto_cleaner.logging_utils import log

__all__ = ["Chart", "build_charts", "charts_to_html", "export_pngs"]

_NUMERIC_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)
_TEMPLATE = "plotly_white"
_COLORWAY = ["#2952cc", "#16a34a", "#b54708", "#9333ea", "#0891b2", "#dc2626"]


@dataclass(slots=True)
class Chart:
    """A named Plotly figure (``name`` doubles as the PNG filename slug)."""

    name: str
    title: str
    fig: Any  # plotly.graph_objects.Figure


def _numeric_columns(df: pl.DataFrame, *, exclude_constant: bool = True) -> list[str]:
    cols = [c for c, dt in zip(df.columns, df.dtypes) if dt in _NUMERIC_DTYPES]
    if exclude_constant:
        cols = [c for c in cols if df.get_column(c).n_unique() > 1]
    return cols


def _categorical_columns(df: pl.DataFrame) -> list[str]:
    return [c for c, dt in zip(df.columns, df.dtypes) if dt in (pl.Utf8, pl.Categorical)]


def _style(fig: Any, title: str) -> Any:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        template=_TEMPLATE,
        colorway=_COLORWAY,
        margin=dict(l=60, r=30, t=50, b=50),
        font=dict(family="-apple-system, Segoe UI, Roboto, Helvetica, Arial", size=12),
        plot_bgcolor="white",
    )
    return fig


# --------------------------------------------------------------------------- #
# Individual chart builders
# --------------------------------------------------------------------------- #
def _fig_missingness(profile: DatasetProfile) -> Chart | None:
    import plotly.graph_objects as go

    cols = [c.name for c in profile.columns]
    pcts = [c.null_pct for c in profile.columns]
    if not cols:
        return None
    fig = go.Figure(go.Bar(x=cols, y=pcts, marker_color="#b54708"))
    fig.update_yaxes(title="missing %", range=[0, max(100.0, max(pcts) if pcts else 0)])
    fig.update_xaxes(tickangle=-35)
    return Chart("missingness", "Missing values by column", _style(fig, "Missing values by column (%)"))


def _figs_histograms(df: pl.DataFrame, config: CleanConfig) -> list[Chart]:
    import plotly.graph_objects as go

    charts: list[Chart] = []
    for c in _numeric_columns(df)[: config.chart_max_numeric]:
        vals = df.get_column(c).drop_nulls().to_list()
        if not vals:
            continue
        fig = go.Figure(go.Histogram(x=vals, marker_color="#2952cc", opacity=0.85))
        fig.update_xaxes(title=c)
        fig.update_yaxes(title="frequency")
        charts.append(Chart(f"dist_{c}", f"Distribution — {c}", _style(fig, f"Distribution of {c}")))
    return charts


def _figs_frequencies(df: pl.DataFrame, config: CleanConfig) -> list[Chart]:
    import plotly.graph_objects as go

    charts: list[Chart] = []
    for c in _categorical_columns(df):
        s = df.get_column(c)
        if s.n_unique() > 50:  # high-cardinality → a bar chart isn't useful
            continue
        vc = s.drop_nulls().value_counts(sort=True).head(config.chart_top_categories)
        if vc.height == 0:
            continue
        levels = [str(v) for v in vc.get_column(c).to_list()]
        counts = vc.get_column("count").to_list()
        fig = go.Figure(go.Bar(x=levels, y=counts, marker_color="#16a34a"))
        fig.update_xaxes(title=c, tickangle=-35)
        fig.update_yaxes(title="count")
        charts.append(Chart(f"freq_{c}", f"Frequency — {c}", _style(fig, f"Frequency of {c}")))
    return charts


def _fig_boxplots(df: pl.DataFrame, config: CleanConfig) -> Chart | None:
    import plotly.graph_objects as go

    cols = _numeric_columns(df)[: config.chart_max_numeric]
    if not cols:
        return None
    fig = go.Figure()
    for c in cols:
        s = df.get_column(c).drop_nulls()
        mean, std = s.mean(), s.std()
        if std in (None, 0) or mean is None:
            continue
        z = ((s - mean) / std).to_list()  # standardise so features share an axis
        fig.add_trace(go.Box(y=z, name=c, boxpoints="outliers", marker_size=3))
    fig.update_yaxes(title="standardised value (z-score)")
    fig.update_xaxes(tickangle=-35)
    fig.update_layout(showlegend=False)
    return Chart("outlier_boxplots", "Outlier boxplots", _style(fig, "Outlier view — standardised boxplots"))


def _fig_scatter_matrix(df: pl.DataFrame, config: CleanConfig) -> Chart | None:
    import plotly.graph_objects as go

    cols = _numeric_columns(df)[: config.chart_max_scatter_cols]
    if len(cols) < 2:
        return None
    sample = df
    if df.height > config.chart_scatter_sample:
        sample = df.sample(n=config.chart_scatter_sample, seed=config.random_seed)

    marker = dict(size=3, opacity=0.6, line=dict(width=0))
    cat_cols = [c for c in _categorical_columns(df) if 2 <= df.get_column(c).n_unique() <= 10]
    color_note = ""
    if cat_cols:
        ccol = cat_cols[0]
        cats = [str(v) for v in sample.get_column(ccol).to_list()]
        uniq = list(dict.fromkeys(cats))
        code = {c: i for i, c in enumerate(uniq)}
        marker["color"] = [code[c] for c in cats]
        marker["colorscale"] = "Viridis"
        marker["showscale"] = True
        marker["colorbar"] = dict(
            title=ccol, tickvals=list(range(len(uniq))), ticktext=uniq, len=0.6
        )
        color_note = f" (hue: {ccol})"

    dims = [dict(label=c, values=sample.get_column(c).to_list()) for c in cols]
    fig = go.Figure(go.Splom(dimensions=dims, marker=marker, diagonal=dict(visible=True)))
    fig.update_layout(height=720)
    return Chart(
        "scatter_matrix",
        "Scatterplot matrix",
        _style(fig, f"Scatterplot matrix{color_note}"),
    )


def _fig_correlation(profile: DatasetProfile) -> Chart | None:
    import plotly.graph_objects as go

    if not profile.corr_matrix:
        return None
    labels = profile.corr_labels
    z = profile.corr_matrix
    show_text = len(labels) <= 10
    fig = go.Figure(
        go.Heatmap(
            z=z, x=labels, y=labels, zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
            text=[[f"{v:+.2f}" for v in row] for row in z] if show_text else None,
            texttemplate="%{text}" if show_text else None,
            colorbar=dict(title="r"),
        )
    )
    fig.update_layout(height=520)
    fig.update_yaxes(autorange="reversed")
    return Chart("correlation_heatmap", "Correlation heatmap", _style(fig, "Pearson correlation heatmap"))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_charts(
    df: pl.DataFrame, profile: DatasetProfile, config: CleanConfig | None = None
) -> list[Chart]:
    """Build the full chart suite. Returns ``[]`` if Plotly is unavailable."""
    config = config or CleanConfig()
    try:
        import plotly.graph_objects  # noqa: F401
    except ImportError:
        log("plotly not installed — skipping charts (pip install plotly)", "WARN", enabled=config.verbose)
        return []

    charts: list[Chart] = []
    if (m := _fig_missingness(profile)) is not None:
        charts.append(m)
    charts.extend(_figs_histograms(df, config))
    charts.extend(_figs_frequencies(df, config))
    if (b := _fig_boxplots(df, config)) is not None:
        charts.append(b)
    if (s := _fig_scatter_matrix(df, config)) is not None:
        charts.append(s)
    if (c := _fig_correlation(profile)) is not None:
        charts.append(c)
    log(f"Built {len(charts)} interactive chart(s)", "OK", enabled=config.verbose)
    return charts


def charts_to_html(charts: list[Chart]) -> tuple[str, str]:
    """Return ``(head_scripts, body_html)``.

    ``head_scripts`` embeds plotly.js **once** so the report is fully offline;
    each figure is rendered as a lightweight div in ``body_html``.
    """
    if not charts:
        return "", ""
    from plotly.offline import get_plotlyjs

    head = f'<script type="text/javascript">{get_plotlyjs()}</script>'
    blocks = []
    for ch in charts:
        div = ch.fig.to_html(
            full_html=False, include_plotlyjs=False, default_width="100%", default_height="430px"
        )
        blocks.append(f'<div class="chart">{div}</div>')
    return head, "\n".join(blocks)


_CHROME_HINT_EMITTED = False  # print the Chrome setup instruction at most once per process


def _is_missing_chrome(exc: Exception) -> bool:
    """True when kaleido failed because no Chrome/Chromium binary is available."""
    return type(exc).__name__ == "ChromeNotFoundError" or "chrome" in str(exc).lower()


def export_pngs(
    charts: list[Chart], charts_dir: str | Path, config: CleanConfig | None = None
) -> dict[str, str]:
    """Write each chart to ``charts_dir/<name>.png``; returns name → path."""
    global _CHROME_HINT_EMITTED
    config = config or CleanConfig()
    out: dict[str, str] = {}
    if not charts:
        return out
    directory = Path(charts_dir)
    directory.mkdir(parents=True, exist_ok=True)
    errors: list[Exception] = []
    for ch in charts:
        target = directory / f"{ch.name}.png"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ch.fig.write_image(str(target), width=1000, height=560, scale=2)
            out[ch.name] = str(target)
        except Exception as exc:  # noqa: BLE001 — PNG export is best-effort
            errors.append(exc)
            log(f"PNG export failed for '{ch.name}': {exc}", "WARN", enabled=config.verbose)
    if errors and not out:
        # Every export failed — surface the real cause even in non-verbose runs,
        # otherwise charts/ stays silently empty and the MD/PDF reports lose figures.
        log(
            f"PNG export failed for all {len(errors)} chart(s): {errors[0]} — "
            "Markdown/PDF reports will have no figures (the HTML report keeps interactive charts)",
            "WARN", enabled=True,
        )
        if any(_is_missing_chrome(e) for e in errors) and not _CHROME_HINT_EMITTED:
            _CHROME_HINT_EMITTED = True
            log(
                "kaleido 1.x no longer bundles Chromium — install a headless Chrome once with "
                "`plotly_get_chrome` (or `python -c \"import kaleido; kaleido.get_chrome_sync()\"`), "
                "then re-run",
                "WARN", enabled=True,
            )
    if out:
        log(f"Exported {len(out)} chart PNG(s) → {directory}", "OK", enabled=config.verbose)
    return out
