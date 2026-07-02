"""Render a :class:`DatasetProfile` into Markdown **and** self-contained HTML.

The HTML is dependency-free (inline CSS, no JS/CDN) so it opens anywhere and can
be emailed or archived. The Markdown mirrors it for diff-friendly version
control. Both foreground *data-health warnings* — the part a busy quant reads
first.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from auto_cleaner.config import CleanConfig
from auto_cleaner.eda.stats import ColumnProfile, DatasetProfile
from auto_cleaner.logging_utils import human_bytes
from auto_cleaner.reporting import StepReport

__all__ = ["build_report", "write_reports"]


def _fmt(value: Any, places: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if value != value:  # NaN
            return "—"
        return f"{value:,.{places}g}"
    return str(value)


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _corr_color(r: float) -> str:
    """Diverging colour: blue (+), red (−), intensity ∝ |r|."""
    if r >= 0:
        shade = int(255 * (1 - r))
        return f"rgb({shade},{shade},255)"
    shade = int(255 * (1 + r))
    return f"rgb(255,{shade},{shade})"


def _plotly_available() -> bool:
    from importlib.util import find_spec

    return find_spec("plotly") is not None


def _missing_charts_note(config: CleanConfig) -> str:
    """Explain *why* the Markdown report has no figures — the causes differ."""
    if not config.make_charts:
        return "Charts disabled (make_charts=False)."
    if not _plotly_available():
        return "Charts unavailable — plotly is not installed (`pip install plotly`)."
    if not config.export_png:
        return "PNG export disabled (export_png=False) — see the HTML report for interactive charts."
    return (
        "No chart PNGs available — the PNG export failed (see log; kaleido 1.x needs a "
        "headless Chrome: run `plotly_get_chrome` once). "
        "The HTML report still contains the interactive charts."
    )


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _md_columns_table(columns: Iterable[ColumnProfile]) -> str:
    head = (
        "| Column | Dtype | Kind | Non-null | Null % | Unique | Mean | Std | "
        "Min | Median | Max | Skew | Kurtosis |\n"
        "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n"
    )
    rows = []
    for c in columns:
        rows.append(
            f"| {c.name} | `{c.dtype}` | {c.kind} | {c.count:,} | {c.null_pct:.1f} | "
            f"{c.n_unique:,} | {_fmt(c.mean)} | {_fmt(c.std)} | {_fmt(c.minimum)} | "
            f"{_fmt(c.median)} | {_fmt(c.maximum)} | {_fmt(c.skewness)} | {_fmt(c.kurtosis)} |"
        )
    return head + "\n".join(rows)


def render_markdown(
    profile: DatasetProfile,
    step_reports: list[StepReport],
    *,
    title: str,
    source_name: str,
    config: CleanConfig,
    chart_pngs: list[tuple[str, str]] | None = None,
    advanced: Any = None,
    summary_lines: list[str] | None = None,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out: list[str] = [f"# {title}", ""]
    out.append(f"*Source:* `{source_name}`  •  *Generated:* {ts}  •  *Engine:* polars")
    if summary_lines:
        out += ["", "## Executive Summary", ""]
        out += summary_lines
    out += ["", "## 1. Dataset Overview", ""]
    out.append(f"- **Data-quality score:** {profile.quality_score}/100  "
               f"(completeness {profile.quality_components.get('completeness', '-')}, "
               f"validity {profile.quality_components.get('validity', '-')}, "
               f"uniqueness {profile.quality_components.get('uniqueness', '-')})")
    out.append(f"- **Rows:** {profile.n_rows:,}")
    out.append(f"- **Columns:** {profile.n_cols}")
    out.append(f"- **In-memory size:** {human_bytes(profile.memory_bytes)}")
    out.append(f"- **Duplicate rows:** {profile.duplicate_rows:,}")

    out += ["", "## 2. Data-Health Warnings", ""]
    step_warns = [f"[{rep.step}] {w}" for rep in step_reports for w in rep.warnings]
    if step_warns or profile.warnings:
        out += [f"- ⚠️ {w}" for w in step_warns + profile.warnings]
    else:
        out.append("- ✅ No critical data-health issues detected.")

    out += ["", "## 3. Visualisations", ""]
    if chart_pngs:
        for caption, rel in chart_pngs:
            out.append(f"![{caption}]({rel})")
            out.append("")
    else:
        out.append(f"_{_missing_charts_note(config)}_")
        out.append("")

    out += ["## 4. Pipeline Actions", ""]
    for rep in step_reports:
        out.append(f"### {rep.step}")
        for a in rep.actions:
            out.append(f"- {a}")
        for w in rep.warnings:
            out.append(f"- ⚠️ {w}")
        out.append("")
        impacts = rep.metrics.get("impact")
        if impacts:
            out.append(f"**Cleaning impact ({rep.step})** — how much this step "
                       "moved each column's distribution:")
            out.append("")
            out.append("| Column | Cells changed | Mean before → after | Δmean (sd) | KS | Verdict |")
            out.append("|---|--:|--:|--:|--:|---|")
            for imp in impacts:
                mean_bits = (
                    f"{_fmt(imp['mean_before'])} → {_fmt(imp['mean_after'])}"
                    if imp["mean_before"] is not None else "—"
                )
                out.append(
                    f"| {imp['column']} | {imp['cells_changed']} ({imp['change_share']:.1%}) "
                    f"| {mean_bits} "
                    f"| {_fmt(imp['mean_shift_sd'], 3) if imp['mean_shift_sd'] is not None else '—'} "
                    f"| {_fmt(imp['ks_stat'], 3) if imp['ks_stat'] is not None else '—'} "
                    f"| {'🔴' if imp['verdict'] == 'material' else ('🟡' if imp['verdict'] == 'minor' else '🟢')} {imp['verdict']} |"
                )
            out.append("")

    out += ["## 5. Column Profiles", "", _md_columns_table(profile.columns), ""]

    out += ["## 6. Correlation Highlights", ""]
    if profile.collinear_pairs:
        out.append("| Feature A | Feature B | Pearson r |")
        out.append("|---|---|--:|")
        for a, b, r in sorted(profile.collinear_pairs, key=lambda x: -abs(x[2])):
            out.append(f"| {a} | {b} | {r:+.3f} |")
    else:
        out.append("_No feature pairs exceed the collinearity threshold._")
    out.append("")

    if profile.corr_matrix:
        out += ["## 7. Correlation Matrix (Pearson)", ""]
        labels = profile.corr_labels
        out.append("| | " + " | ".join(labels) + " |")
        out.append("|---" * (len(labels) + 1) + "|")
        for i, row in enumerate(profile.corr_matrix):
            out.append(f"| **{labels[i]}** | " + " | ".join(f"{v:+.2f}" for v in row) + " |")
        out.append("")

    out += ["## 8. Advanced Analysis", ""]
    out += _advanced_md(advanced)

    out += ["## 9. Configuration", "", "```json"]
    out.append(_json_like(config.to_dict()))
    out += ["```", ""]
    return "\n".join(out)


def _json_like(d: dict[str, Any]) -> str:
    import json

    return json.dumps(d, indent=2, default=str)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
_CSS = """
:root { --fg:#1a2233; --muted:#5b667a; --line:#e3e8f0; --accent:#2952cc; --warn:#b54708; }
* { box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
       color:#1a2233; margin:0; padding:32px; background:#f7f9fc; line-height:1.5; }
.wrap { max-width:1100px; margin:0 auto; background:#fff; border:1px solid #e3e8f0;
        border-radius:14px; padding:36px 40px; box-shadow:0 1px 3px rgba(16,24,40,.06); }
h1 { font-size:26px; margin:0 0 4px; letter-spacing:-.02em; }
h2 { font-size:18px; margin:34px 0 12px; padding-bottom:6px; border-bottom:2px solid #eef2f7; }
h3 { font-size:14px; margin:18px 0 6px; color:#344054; text-transform:uppercase; letter-spacing:.04em; }
.meta { color:#667085; font-size:13px; margin-bottom:8px; }
.cards { display:flex; flex-wrap:wrap; gap:14px; margin:8px 0 4px; }
.card { flex:1 1 160px; background:#f8fafc; border:1px solid #eef2f7; border-radius:10px; padding:14px 16px; }
.card .k { font-size:12px; color:#667085; text-transform:uppercase; letter-spacing:.04em; }
.card .v { font-size:22px; font-weight:650; margin-top:4px; }
ul.warn { list-style:none; padding:0; margin:0; }
ul.warn li { background:#fffaf0; border:1px solid #fde9c8; border-left:4px solid #f79009;
             padding:9px 13px; border-radius:8px; margin-bottom:8px; font-size:13.5px; }
ul.ok li { background:#f0fdf4; border:1px solid #bbf7d0; border-left:4px solid #16a34a; }
table { border-collapse:collapse; width:100%; font-size:12.5px; margin-top:6px; }
th,td { border:1px solid #eef2f7; padding:6px 9px; text-align:right; }
th { background:#f8fafc; color:#344054; font-weight:650; position:sticky; top:0; }
td.l,th.l { text-align:left; }
code { background:#f1f5f9; padding:1px 5px; border-radius:5px; font-size:12px; }
.scroll { overflow-x:auto; }
details { margin-top:8px; } summary { cursor:pointer; font-weight:600; color:#2952cc; }
pre { background:#0f172a; color:#e2e8f0; padding:16px; border-radius:10px; overflow:auto; font-size:12px; }
.actions { font-size:13px; } .actions li { margin-bottom:3px; }
.footer { margin-top:28px; color:#98a2b3; font-size:12px; text-align:center; }
.chart { margin:16px 0; border:1px solid #eef2f7; border-radius:10px; padding:8px 8px 2px; background:#fff; }
"""


def _html_cards(profile: DatasetProfile, n_step_warnings: int = 0) -> str:
    items = [
        ("Quality score", f"{profile.quality_score}/100"),
        ("Rows", f"{profile.n_rows:,}"),
        ("Columns", str(profile.n_cols)),
        ("Memory", human_bytes(profile.memory_bytes)),
        ("Duplicate rows", f"{profile.duplicate_rows:,}"),
        ("Warnings", str(len(profile.warnings) + n_step_warnings)),
    ]
    cells = "".join(
        f'<div class="card"><div class="k">{_esc(k)}</div><div class="v">{_esc(v)}</div></div>'
        for k, v in items
    )
    return f'<div class="cards">{cells}</div>'


def _html_columns_table(columns: Iterable[ColumnProfile]) -> str:
    headers = [
        ("Column", "l"), ("Dtype", "l"), ("Kind", "l"), ("Non-null", ""), ("Null %", ""),
        ("Unique", ""), ("Mean", ""), ("Std", ""), ("Min", ""), ("Median", ""),
        ("Max", ""), ("Skew", ""), ("Kurt", ""),
    ]
    thead = "".join(f'<th class="{cls}">{_esc(h)}</th>' for h, cls in headers)
    body = []
    for c in columns:
        null_style = ' style="color:#b54708;font-weight:650"' if c.null_pct > 0 else ""
        body.append(
            "<tr>"
            f'<td class="l"><b>{_esc(c.name)}</b></td>'
            f'<td class="l"><code>{_esc(c.dtype)}</code></td>'
            f'<td class="l">{_esc(c.kind)}</td>'
            f"<td>{c.count:,}</td>"
            f'<td{null_style}>{c.null_pct:.1f}</td>'
            f"<td>{c.n_unique:,}</td>"
            f"<td>{_fmt(c.mean)}</td><td>{_fmt(c.std)}</td><td>{_fmt(c.minimum)}</td>"
            f"<td>{_fmt(c.median)}</td><td>{_fmt(c.maximum)}</td>"
            f"<td>{_fmt(c.skewness)}</td><td>{_fmt(c.kurtosis)}</td>"
            "</tr>"
        )
    return f'<div class="scroll"><table><thead><tr>{thead}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _html_corr_matrix(profile: DatasetProfile) -> str:
    if not profile.corr_matrix:
        return "<p><em>Not enough numeric columns for a correlation matrix.</em></p>"
    labels = profile.corr_labels
    head = '<th class="l"></th>' + "".join(f'<th>{_esc(x)}</th>' for x in labels)
    rows = []
    for i, row in enumerate(profile.corr_matrix):
        cells = [f'<td class="l"><b>{_esc(labels[i])}</b></td>']
        for v in row:
            cells.append(f'<td style="background:{_corr_color(v)}">{v:+.2f}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _spec_html(spec: Any) -> str:
    if spec is None:
        return ""
    arche = ", ".join(f"{name} ({conf:.0%})" for name, conf in spec.archetypes)
    out = (
        "<h3>Auto-specialisation &mdash; detected archetype(s)</h3>"
        f"<p><b>{_esc(arche)}</b> &nbsp;|&nbsp; auto-modules: "
        f"{_esc(', '.join(spec.auto_modules) or 'none')}</p>"
    )
    if spec.recommendations:
        out += "<ul class='warn'>" + "".join(f"<li>{_esc(r)}</li>" for r in spec.recommendations) + "</ul>"
    tf = (spec.signals or {}).get("text_features")
    if tf:
        rows = "".join(
            f"<tr><td class='l'>{_esc(c)}</td><td>{d['mean_chars']}</td><td>{d['mean_words']}</td>"
            f"<td class='l'>{_esc(', '.join(d['top_terms']))}</td></tr>"
            for c, d in tf.items()
        )
        out += (
            "<h3>Text features</h3><div class='scroll'><table><thead><tr>"
            "<th class='l'>Column</th><th>Avg chars</th><th>Avg words</th>"
            f"<th class='l'>Top TF-IDF terms</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )
    return out


def _inference_html(inf: Any) -> str:
    if inf is None:
        return ""
    parts: list[str] = []
    if inf.cis:
        rows = "".join(
            f"<tr><td class='l'>{_esc(r.feature)}</td><td>{_esc(r.statistic)}</td>"
            f"<td>{_fmt(r.point)}</td><td>[{_fmt(r.lo)}, {_fmt(r.hi)}]</td></tr>"
            for r in inf.cis[:20]
        )
        parts.append(
            "<h3>Bootstrap 95% confidence intervals</h3><div class='scroll'><table><thead><tr>"
            "<th class='l'>Feature</th><th>Statistic</th><th>Estimate</th><th>95% CI</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>"
        )
    if inf.group_tests:
        rows = ""
        for g in inf.group_tests:
            colr = '#16a34a' if g.significant else '#667085'
            res = 'significant' if g.significant else 'n.s.'
            eff = f"{_esc(g.effect_name)} {_fmt(g.effect_value)}" if g.effect_name else "&mdash;"
            rows += (
                f"<tr><td class='l'>{_esc(g.value)}</td><td class='l'>{_esc(g.group)}</td>"
                f"<td class='l'>{_esc(g.test)}</td><td>{_fmt(g.statistic)}</td><td>{_fmt(g.p_value)}</td>"
                f"<td>{eff}</td><td style='color:{colr}'>{res}</td></tr>"
            )
        parts.append(
            "<h3>Group comparison tests (auto-selected, with effect size)</h3><div class='scroll'>"
            "<table><thead><tr><th class='l'>Value</th><th class='l'>Group</th><th class='l'>Test</th>"
            "<th>Statistic</th><th>p-value</th><th>Effect</th><th>Result</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>"
        )
    sig = [c for c in inf.corr_sig if c.significant][:15]
    if sig:
        rows = "".join(
            f"<tr><td class='l'>{_esc(c.a)}</td><td class='l'>{_esc(c.b)}</td>"
            f"<td>{c.r:+.3f}</td><td>{_fmt(c.p_value)}</td><td>{_fmt(c.p_adj)}</td></tr>"
            for c in sig
        )
        parts.append(
            "<h3>Significant correlations (Benjamini-Hochberg)</h3><div class='scroll'><table><thead><tr>"
            "<th class='l'>A</th><th class='l'>B</th><th>r</th><th>p</th><th>p (adj)</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>"
        )
    reg = inf.regression
    if reg is not None and reg.terms:
        rows = "".join(
            f"<tr><td class='l'>{_esc(name)}</td><td>{_fmt(coef)}</td><td>{_fmt(se)}</td>"
            f"<td>{_fmt(p)}</td><td>[{_fmt(lo)}, {_fmt(hi)}]</td></tr>"
            for (name, coef, se, p, lo, hi) in reg.terms[:20]
        )
        parts.append(
            f"<h3>{_esc(reg.kind)} regression on '<code>{_esc(reg.target)}</code>' "
            f"(R&sup2;={_fmt(reg.r2)}, n={reg.n})</h3><div class='scroll'><table><thead><tr>"
            "<th class='l'>Term</th><th>Coef</th><th>Std err</th><th>p-value</th><th>95% CI</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>"
        )
    elif reg is not None and reg.note:
        parts.append(f"<p><em>{_esc(reg.note)}</em></p>")
    if parts:
        caveats = "".join(f"<li>{_esc(c)}</li>" for c in inf.caveats)
        parts.append(
            "<p style='color:#667085;font-size:12px;margin-bottom:2px'><b>Inference caveats:</b></p>"
            f"<ul class='actions' style='color:#667085'>{caveats}</ul>"
        )
    return "".join(parts)


def _model_html(m: Any) -> str:
    if m is None:
        return ""
    if not m.scores:
        return f"<h3>Baseline models</h3><p><em>{_esc(m.note or 'unavailable')}</em></p>"
    keys = list(m.scores[0].metrics.keys())
    head = "".join(f"<th>{_esc(k)}</th>" for k in keys)
    rows = ""
    for s in m.scores:
        mark = ' &starf;' if s.name == m.best else ''
        cells = "".join(f"<td>{_fmt(s.metrics.get(k))}</td>" for k in keys)
        rows += f"<tr><td class='l'>{_esc(s.name)}{mark}</td>{cells}</tr>"
    out = (
        f"<h3>Baseline models &mdash; {_esc(m.task)} on '<code>{_esc(m.target)}</code>' "
        f"(n={m.n}, 5-fold CV)</h3><div class='scroll'><table><thead><tr><th class='l'>Model</th>"
        f"{head}</tr></thead><tbody>{rows}</tbody></table></div>"
    )
    if m.leakage:
        out += "<ul class='warn'>" + "".join(f"<li>Possible leakage: {_esc(x)}</li>" for x in m.leakage) + "</ul>"
    if m.importances:
        irows = "".join(f"<tr><td class='l'>{_esc(f)}</td><td>{_fmt(v)}</td></tr>" for f, v in m.importances)
        out += (
            "<h3>Permutation importance (best model)</h3><div class='scroll'><table><thead><tr>"
            f"<th class='l'>Feature</th><th>Importance</th></tr></thead><tbody>{irows}</tbody></table></div>"
        )
    if getattr(m, "shap_importances", None):
        srows = "".join(f"<tr><td class='l'>{_esc(f)}</td><td>{_fmt(v)}</td></tr>" for f, v in m.shap_importances)
        out += (
            "<h3>SHAP feature impact (best model)</h3><div class='scroll'><table><thead><tr>"
            f"<th class='l'>Feature</th><th>mean |SHAP|</th></tr></thead><tbody>{srows}</tbody></table></div>"
        )
    caveats = "".join(f"<li>{_esc(c)}</li>" for c in m.caveats)
    out += (
        "<p style='color:#667085;font-size:12px;margin-bottom:2px'><b>Modelling caveats:</b></p>"
        f"<ul class='actions' style='color:#667085'>{caveats}</ul>"
    )
    return out


def _fda_html(f: Any) -> str:
    if f is None:
        return ""
    evr = ", ".join(f"{x:.0%}" for x in f.variance_ratio[:6])
    return (
        f"<h3>Functional data analysis (index '<code>{_esc(f.time_index)}</code>')</h3>"
        f"<p>{f.n_curves} curves &times; {f.n_points} points &middot; smoothing: {_esc(f.smoothing)}<br>"
        f"Functional PCA &mdash; modes for 90% variance: <b>{f.n_modes_90}</b> &middot; "
        f"top-mode variance: {evr}</p>"
    )


def _extended_html(ext: Any) -> str:  # noqa: C901 — many independent sub-tables
    if ext is None:
        return ""
    P: list[str] = []

    if getattr(ext, "robust", None):
        rows = "".join(
            f"<tr><td class='l'><b>{_esc(r.feature)}</b></td><td>{_fmt(r.arithmetic_mean)}</td>"
            f"<td>{_fmt(r.geometric_mean)}</td><td>{_fmt(r.harmonic_mean)}</td><td>{_fmt(r.trimmed_mean_10)}</td>"
            f"<td>{_fmt(r.winsorized_mean_10)}</td><td>{_fmt(r.median)}</td><td>{_fmt(r.mad)}</td>"
            f"<td>{_fmt(r.huber_location)}</td></tr>"
            for r in ext.robust[:30]
        )
        P.append(
            "<h3>Robust location estimators</h3><div class='scroll'><table><thead><tr>"
            "<th class='l'>Feature</th><th>Mean</th><th>Geom</th><th>Harm</th><th>Trim 10%</th>"
            "<th>Winsor 10%</th><th>Median</th><th>MAD</th><th>Huber</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>"
        )

    a = getattr(ext, "associations", None)
    if a is not None and (a.spearman or a.partial or a.cramers_v or a.eta):
        sub = []
        if a.spearman:
            rows = "".join(f"<tr><td class='l'>{_esc(x)}</td><td class='l'>{_esc(y)}</td><td>{r:+.3f}</td><td>{_fmt(p)}</td></tr>" for x, y, r, p in a.spearman[:12])
            sub.append("<h3>Spearman rank correlations</h3><div class='scroll'><table><thead><tr><th class='l'>A</th><th class='l'>B</th><th>rho</th><th>p</th></tr></thead><tbody>" + rows + "</tbody></table></div>")
        if a.partial:
            rows = "".join(f"<tr><td class='l'>{_esc(x)}</td><td class='l'>{_esc(y)}</td><td>{r:+.3f}</td></tr>" for x, y, r in a.partial[:12])
            sub.append("<h3>Partial correlations (controlling for others)</h3><div class='scroll'><table><thead><tr><th class='l'>A</th><th class='l'>B</th><th>partial r</th></tr></thead><tbody>" + rows + "</tbody></table></div>")
        if a.cramers_v:
            rows = "".join(f"<tr><td class='l'>{_esc(x)}</td><td class='l'>{_esc(y)}</td><td>{v:.3f}</td></tr>" for x, y, v in a.cramers_v[:12])
            sub.append("<h3>Categorical association (Cramer's V)</h3><div class='scroll'><table><thead><tr><th class='l'>A</th><th class='l'>B</th><th>V</th></tr></thead><tbody>" + rows + "</tbody></table></div>")
        if a.eta:
            rows = "".join(f"<tr><td class='l'>{_esc(c)}</td><td class='l'>{_esc(n)}</td><td>{e:.3f}</td></tr>" for c, n, e in a.eta[:12])
            sub.append("<h3>Correlation ratio (categorical &rarr; numeric)</h3><div class='scroll'><table><thead><tr><th class='l'>Category</th><th class='l'>Numeric</th><th>eta</th></tr></thead><tbody>" + rows + "</tbody></table></div>")
        P.append("".join(sub))

    if getattr(ext, "distributions", None):
        rows = "".join(f"<tr><td class='l'><b>{_esc(d.feature)}</b></td><td class='l'>{_esc(d.best_distribution)}</td><td>{_fmt(d.aic)}</td><td>{_fmt(d.bic)}</td><td>{_fmt(d.ks_p)}</td></tr>" for d in ext.distributions[:30])
        P.append("<h3>Best-fit distributions (by AIC)</h3><div class='scroll'><table><thead><tr><th class='l'>Feature</th><th class='l'>Best fit</th><th>AIC</th><th>BIC</th><th>KS p</th></tr></thead><tbody>" + rows + "</tbody></table></div>")

    if getattr(ext, "timeseries", None):
        rows = ""
        for r in ext.timeseries:
            stat = "stationary" if r.stationary else "non-stationary"
            rows += (f"<tr><td class='l'><b>{_esc(r.feature)}</b></td><td>{_fmt(r.adf_p)}</td><td>{_fmt(r.kpss_p)}</td>"
                     f"<td>{stat}</td><td>{_esc(r.mk_trend)} ({_fmt(r.mk_p)})</td><td>{_esc(r.seasonal_period)}</td>"
                     f"<td>{_fmt(r.seasonal_strength)}</td><td>{_fmt(r.trend_strength)}</td></tr>")
        P.append("<h3>Time-series diagnostics</h3><div class='scroll'><table><thead><tr><th class='l'>Series</th><th>ADF p</th><th>KPSS p</th><th>Stationarity</th><th>Mann-Kendall</th><th>Period</th><th>Seasonal str.</th><th>Trend str.</th></tr></thead><tbody>" + rows + "</tbody></table></div>")

    if getattr(ext, "forecasts", None):
        rows = ""
        for f in ext.forecasts:
            nxt = f.forecast[0] if f.forecast else None
            end = f.forecast[-1] if f.forecast else None
            lo = f.lower[-1] if f.lower else None
            hi = f.upper[-1] if f.upper else None
            rows += (
                f"<tr><td class='l'><b>{_esc(f.feature)}</b></td><td class='l'>{_esc(f.model)}</td>"
                f"<td>{_fmt(f.last_value)}</td><td>{_fmt(nxt)}</td>"
                f"<td>{_fmt(end)} [{_fmt(lo)}, {_fmt(hi)}]</td></tr>"
            )
        P.append(
            f"<h3>Forecasts &mdash; next {ext.forecasts[0].horizon} steps</h3><div class='scroll'>"
            "<table><thead><tr><th class='l'>Series</th><th class='l'>Model</th><th>Last value</th>"
            "<th>Next</th><th>Horizon end (95% CI)</th></tr></thead><tbody>"
            f"{rows}</tbody></table></div>"
        )

    m = getattr(ext, "multivariate", None)
    if m is not None:
        bits = []
        if m.mahalanobis_outliers is not None:
            bits.append(f"Mahalanobis outliers: <b>{m.mahalanobis_outliers}</b> (&chi;&sup2; &gt; {_fmt(m.mahalanobis_threshold)})")
        if m.best_k is not None:
            bits.append(f"clusters: <b>{m.best_k}</b> (silhouette {_fmt(m.silhouette)}, sizes {m.cluster_sizes})")
        if m.pca_2d_variance is not None:
            bits.append(f"2-D PCA variance: <b>{m.pca_2d_variance:.0%}</b>")
        if m.manova_p is not None:
            bits.append(f"MANOVA Wilks p: <b>{_fmt(m.manova_p)}</b>")
        if m.umap_silhouette is not None:
            bits.append(f"UMAP silhouette: <b>{_fmt(m.umap_silhouette)}</b>")
        if bits:
            P.append("<h3>Multivariate analysis</h3><p>" + " &nbsp;&middot;&nbsp; ".join(bits) + "</p>")

    if getattr(ext, "nlp", None):
        sub = []
        for r in ext.nlp:
            topics = "".join(f"<li>{_esc(t)}</li>" for t in r.topics)
            sent = f"sentiment mean {_fmt(r.sentiment_mean)} &middot; pos {_fmt(r.positive_pct)}% &middot; neg {_fmt(r.negative_pct)}%"
            sub.append(f"<h3>Text &mdash; '<code>{_esc(r.column)}</code>' ({r.n_docs} docs)</h3><p>{sent}</p><ul class='actions'>{topics}</ul>")
        P.append("".join(sub))

    if getattr(ext, "embeddings", None):
        rows = "".join(f"<tr><td class='l'>{_esc(e.column)}</td><td class='l'>{_esc(e.model)}</td><td>{e.dim}</td><td>{_fmt(e.pca_2d_variance)}</td><td>{_fmt(e.silhouette)}</td></tr>" for e in ext.embeddings)
        P.append("<h3>Neural embeddings</h3><div class='scroll'><table><thead><tr><th class='l'>Column</th><th class='l'>Model</th><th>Dim</th><th>2-D var</th><th>Silhouette</th></tr></thead><tbody>" + rows + "</tbody></table></div>")

    b = getattr(ext, "bayesian", None)
    if b is not None and b.factors:
        rows = "".join(f"<tr><td class='l'>{_esc(f.value)}</td><td class='l'>{_esc(f.group)}</td><td>{_fmt(f.bf10)}</td><td class='l'>{_esc(f.interpretation)}</td></tr>" for f in b.factors[:12])
        P.append("<h3>Bayesian comparison (Bayes factors)</h3><div class='scroll'><table><thead><tr><th class='l'>Value</th><th class='l'>Group</th><th>BF10</th><th class='l'>Interpretation</th></tr></thead><tbody>" + rows + "</tbody></table></div>")

    s = getattr(ext, "survival", None)
    if s is not None:
        terms = "".join(f"<tr><td class='l'>{_esc(n)}</td><td>{_fmt(coef)}</td><td>{_fmt(p)}</td><td>{_fmt(hr)}</td></tr>" for n, coef, p, hr in s.cox_terms[:12])
        P.append(
            f"<h3>Survival analysis (duration '<code>{_esc(s.duration_col)}</code>', event '<code>{_esc(s.event_col)}</code>')</h3>"
            f"<p>n={s.n}, events={s.n_events}, median survival={_fmt(s.median_survival)}, Cox C-index={_fmt(s.cox_concordance)}</p>"
            + (f"<div class='scroll'><table><thead><tr><th class='l'>Covariate</th><th>coef</th><th>p</th><th>HR</th></tr></thead><tbody>{terms}</tbody></table></div>" if terms else "")
        )

    sv = getattr(ext, "survey", None)
    if sv is not None:
        bits = []
        if sv.cronbach_alpha is not None:
            bits.append(f"Cronbach's &alpha;: <b>{_fmt(sv.cronbach_alpha)}</b> (CI {sv.cronbach_ci}) over {len(sv.cronbach_items)} items")
        if sv.weight_column:
            bits.append(f"design weight: '<code>{_esc(sv.weight_column)}</code>'")
        wm = ""
        if sv.weighted_means:
            wrows = "".join(f"<tr><td class='l'>{_esc(c)}</td><td>{_fmt(w)}</td><td>{_fmt(u)}</td></tr>" for c, w, u in sv.weighted_means[:12])
            wm = "<div class='scroll'><table><thead><tr><th class='l'>Column</th><th>Weighted mean</th><th>Unweighted</th></tr></thead><tbody>" + wrows + "</tbody></table></div>"
        if bits or wm:
            P.append("<h3>Survey methodology</h3><p>" + " &nbsp;&middot;&nbsp; ".join(bits) + "</p>" + wm)

    return "\n".join(p for p in P if p)


def _extended_md(ext: Any) -> list[str]:
    if ext is None:
        return []
    out: list[str] = []
    if getattr(ext, "robust", None):
        out += ["**Robust location estimators**", "",
                "| Feature | Mean | Geom | Harm | Trim 10% | Winsor 10% | Median | MAD | Huber |",
                "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for r in ext.robust[:30]:
            out.append(f"| {r.feature} | {_fmt(r.arithmetic_mean)} | {_fmt(r.geometric_mean)} | {_fmt(r.harmonic_mean)} | {_fmt(r.trimmed_mean_10)} | {_fmt(r.winsorized_mean_10)} | {_fmt(r.median)} | {_fmt(r.mad)} | {_fmt(r.huber_location)} |")
        out.append("")
    a = getattr(ext, "associations", None)
    if a is not None and a.spearman:
        out += ["**Spearman rank correlations**", "", "| A | B | rho | p |", "|---|---|--:|--:|"]
        for x, y, r, p in a.spearman[:12]:
            out.append(f"| {x} | {y} | {r:+.3f} | {_fmt(p)} |")
        out.append("")
    if a is not None and a.partial:
        out += ["**Partial correlations**", "", "| A | B | partial r |", "|---|---|--:|"]
        for x, y, r in a.partial[:12]:
            out.append(f"| {x} | {y} | {r:+.3f} |")
        out.append("")
    if a is not None and a.cramers_v:
        out += ["**Cramer's V (categorical association)**", "", "| A | B | V |", "|---|---|--:|"]
        for x, y, v in a.cramers_v[:12]:
            out.append(f"| {x} | {y} | {v:.3f} |")
        out.append("")
    if getattr(ext, "distributions", None):
        out += ["**Best-fit distributions (AIC)**", "", "| Feature | Best fit | AIC | BIC | KS p |", "|---|---|--:|--:|--:|"]
        for d in ext.distributions[:30]:
            out.append(f"| {d.feature} | {d.best_distribution} | {_fmt(d.aic)} | {_fmt(d.bic)} | {_fmt(d.ks_p)} |")
        out.append("")
    if getattr(ext, "timeseries", None):
        out += ["**Time-series diagnostics**", "", "| Series | ADF p | KPSS p | Stationary | Mann-Kendall | Period | Seasonal str. | Trend str. |", "|---|--:|--:|---|---|--:|--:|--:|"]
        for r in ext.timeseries:
            out.append(f"| {r.feature} | {_fmt(r.adf_p)} | {_fmt(r.kpss_p)} | {'yes' if r.stationary else 'no'} | {r.mk_trend} ({_fmt(r.mk_p)}) | {r.seasonal_period} | {_fmt(r.seasonal_strength)} | {_fmt(r.trend_strength)} |")
        out.append("")
    if getattr(ext, "forecasts", None):
        out += [f"**Forecasts (next {ext.forecasts[0].horizon} steps)**", "",
                "| Series | Model | Last | Next | Horizon end (95% CI) |", "|---|---|--:|--:|---|"]
        for f in ext.forecasts:
            nxt = f.forecast[0] if f.forecast else None
            end = f.forecast[-1] if f.forecast else None
            lo = f.lower[-1] if f.lower else None
            hi = f.upper[-1] if f.upper else None
            out.append(f"| {f.feature} | {f.model} | {_fmt(f.last_value)} | {_fmt(nxt)} | {_fmt(end)} [{_fmt(lo)}, {_fmt(hi)}] |")
        out.append("")
    m = getattr(ext, "multivariate", None)
    if m is not None:
        bits = []
        if m.mahalanobis_outliers is not None:
            bits.append(f"Mahalanobis outliers {m.mahalanobis_outliers}")
        if m.best_k is not None:
            bits.append(f"clusters {m.best_k} (silhouette {_fmt(m.silhouette)})")
        if m.pca_2d_variance is not None:
            bits.append(f"2D PCA var {m.pca_2d_variance}")
        if m.manova_p is not None:
            bits.append(f"MANOVA p {_fmt(m.manova_p)}")
        if bits:
            out += ["**Multivariate:** " + "; ".join(bits), ""]
    if getattr(ext, "nlp", None):
        for r in ext.nlp:
            out += [f"**Text `{r.column}`** ({r.n_docs} docs) — sentiment {_fmt(r.sentiment_mean)}", ""]
            for t in r.topics:
                out.append(f"- topic: {t}")
            out.append("")
    b = getattr(ext, "bayesian", None)
    if b is not None and b.factors:
        out += ["**Bayes factors**", "", "| Value | Group | BF10 | Interpretation |", "|---|---|--:|---|"]
        for f in b.factors[:12]:
            out.append(f"| {f.value} | {f.group} | {_fmt(f.bf10)} | {f.interpretation} |")
        out.append("")
    s = getattr(ext, "survival", None)
    if s is not None:
        out += [f"**Survival** (duration `{s.duration_col}`, event `{s.event_col}`) — n={s.n}, events={s.n_events}, median={_fmt(s.median_survival)}, Cox C={_fmt(s.cox_concordance)}", ""]
    sv = getattr(ext, "survey", None)
    if sv is not None and sv.cronbach_alpha is not None:
        out += [f"**Survey reliability** — Cronbach's alpha {_fmt(sv.cronbach_alpha)} over {len(sv.cronbach_items)} items", ""]
    return out


def _tuning_html(t: Any) -> str:
    if t is None:
        return ""
    verdict = "improved" if t.improvement > 0 else "no improvement over baseline"
    params = ", ".join(f"{k}={v}" for k, v in t.best_params.items())
    return (
        f"<h3>Hyper-parameter tuning (Optuna, {t.n_trials} trials)</h3>"
        f"<p>Metric {_esc(t.metric)}: baseline {_fmt(t.baseline_cv)} &rarr; tuned <b>{_fmt(t.best_cv)}</b> "
        f"(&Delta; {t.improvement:+.4f}, {verdict}).<br>"
        f"<span style='color:#667085'>Best params: {_esc(params)}</span></p>"
    )


def _causal_html(c: Any) -> str:
    if c is None:
        return ""
    warn = "<ul class='warn'>" + "".join(f"<li>{_esc(x)}</li>" for x in c.caveats) + "</ul>"
    es = f" &middot; {_esc(c.effect_size_name)} {_fmt(c.effect_size)}" if c.effect_size_name else ""
    ipw = "" if c.ipw_ate is None else (
        f"<br>Propensity-IPW causal ATE: <b>{_fmt(c.ipw_ate)}</b> "
        f"(naive diff {_fmt(c.naive_diff)}, propensity AUC {_fmt(c.propensity_auc)})"
    )
    return (
        f"<h3>A/B &amp; causal &mdash; '<code>{_esc(c.treatment)}</code>' &rarr; '<code>{_esc(c.outcome)}</code>'</h3>"
        f"<p>{c.n_treated} treated vs {c.n_control} control &middot; {_esc(c.effect_name)}: "
        f"<b>{_fmt(c.effect)}</b> [{_fmt(c.ci_low)}, {_fmt(c.ci_high)}] &middot; {_esc(c.test)} "
        f"p={_fmt(c.p_value)}{es}{ipw}</p>{warn}"
    )


def _count_tests(advanced: Any) -> int:
    """Count auto-computed hypothesis tests/associations (for the hygiene guard)."""
    n = 0
    inf = getattr(advanced, "inference", None)
    if inf is not None:
        n += len(inf.group_tests) + len(inf.corr_sig)
    ext = getattr(advanced, "extended", None)
    if ext is not None:
        a = getattr(ext, "associations", None)
        if a is not None:
            n += len(a.spearman) + len(a.kendall) + len(a.cramers_v)
        b = getattr(ext, "bayesian", None)
        if b is not None:
            n += len(b.factors)
        n += len(getattr(ext, "timeseries", []) or [])
    return n


def _advanced_html(advanced: Any) -> str:
    """Render the AdvancedAnalysis container into an HTML section body."""
    if advanced is None:
        return "<p><em>Advanced analysis disabled.</em></p>"
    parts: list[str] = []

    n_tests = _count_tests(advanced)
    if n_tests >= 10:
        parts.append(
            "<ul class='warn'><li><b>Statistical hygiene:</b> "
            f"{n_tests} hypothesis tests/associations were computed automatically. "
            "Unadjusted p-values should be read cautiously (multiple comparisons); "
            "confirm key findings on a held-out sample.</li></ul>"
        )

    spec_block = _spec_html(getattr(advanced, "specialization", None))
    if spec_block:
        parts.append(spec_block)

    if getattr(advanced, "normality", None):
        rows = "".join(
            "<tr>"
            f'<td class="l"><b>{_esc(r.feature)}</b></td><td>{r.n:,}</td>'
            f"<td>{_fmt(r.shapiro_p)}</td><td>{_fmt(r.dagostino_p)}</td><td>{_fmt(r.jarque_bera_p)}</td>"
            f"<td>{_fmt(r.anderson_stat)} / {_fmt(r.anderson_crit_5pct)}</td>"
            f'<td style="color:{"#16a34a" if r.is_normal else "#b54708"}">'
            f'{"Normal" if r.is_normal else "Non-normal"}</td></tr>'
            for r in advanced.normality
        )
        parts.append(
            "<h3>Normality tests</h3><div class='scroll'><table><thead><tr>"
            "<th class='l'>Feature</th><th>n</th><th>Shapiro p</th><th>D'Agostino p</th>"
            "<th>Jarque-Bera p</th><th>Anderson stat/crit(5%)</th><th>Verdict</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>"
        )

    if getattr(advanced, "transforms", None):
        rows = "".join(
            f'<tr><td class="l"><b>{_esc(t.feature)}</b></td><td class="l">{_esc(t.method)}</td>'
            f"<td>{t.skew_before:+.3f}</td><td>{t.skew_after:+.3f}</td><td>{_fmt(t.lam)}</td></tr>"
            for t in advanced.transforms
        )
        parts.append(
            "<h3>Recommended normalising transforms</h3><div class='scroll'><table><thead><tr>"
            "<th class='l'>Feature</th><th class='l'>Method</th><th>Skew before</th>"
            f"<th>Skew after</th><th>lambda</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )

    if getattr(advanced, "vif", None):
        rows = "".join(
            f'<tr><td class="l"><b>{_esc(v.feature)}</b></td><td>{v.vif:,.2f}</td>'
            f'<td style="color:{"#b54708" if v.vif > 10 else ("#9a6700" if v.vif > 5 else "#16a34a")}">'
            f'{"severe" if v.vif > 10 else ("moderate" if v.vif > 5 else "ok")}</td></tr>'
            for v in advanced.vif
        )
        parts.append(
            "<h3>Multicollinearity &mdash; VIF</h3><div class='scroll'><table><thead><tr>"
            "<th class='l'>Feature</th><th>VIF</th><th>Severity</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>"
        )

    if getattr(advanced, "pca", None) is not None:
        p = advanced.pca
        evr = ", ".join(f"{x:.0%}" for x in p.explained_variance_ratio[:6])
        parts.append(
            f"<h3>PCA</h3><p>Components for 90% variance: <b>{p.n_components_90}</b> &nbsp;&bull;&nbsp; "
            f"for 95%: <b>{p.n_components_95}</b><br><span style='color:#667085'>"
            f"Top explained variance: {evr}</span></p>"
        )

    if getattr(advanced, "relevance", None):
        rows = "".join(
            f'<tr><td>{r.rank}</td><td class="l"><b>{_esc(r.feature)}</b></td>'
            f"<td>{_fmt(r.mutual_info)}</td><td>{_fmt(r.f_stat)}</td><td>{_fmt(r.p_value)}</td>"
            f"<td>{_fmt(r.target_corr)}</td></tr>"
            for r in advanced.relevance
        )
        parts.append(
            f"<h3>Feature relevance &mdash; target '<code>{_esc(advanced.target)}</code>' "
            f"({_esc(advanced.task_type)})</h3><div class='scroll'><table><thead><tr>"
            "<th>Rank</th><th class='l'>Feature</th><th>Mutual info</th><th>F</th>"
            f"<th>p-value</th><th>|corr|</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )

    for extra in (
        _inference_html(getattr(advanced, "inference", None)),
        _model_html(getattr(advanced, "modeling", None)),
        _fda_html(getattr(advanced, "fda", None)),
        _extended_html(getattr(advanced, "extended", None)),
        _tuning_html(getattr(advanced, "tuning", None)),
        _causal_html(getattr(advanced, "causal", None)),
    ):
        if extra:
            parts.append(extra)

    if not parts:
        return "<p><em>No advanced diagnostics available (needs &ge;2 numeric features and scipy).</em></p>"
    return "\n".join(parts)


def _spec_md(spec: Any) -> list[str]:
    if spec is None:
        return []
    arche = ", ".join(f"{name} ({conf:.0%})" for name, conf in spec.archetypes)
    out = [f"**Auto-specialisation:** {arche}  |  auto-modules: {', '.join(spec.auto_modules) or 'none'}", ""]
    for r in spec.recommendations:
        out.append(f"- {r}")
    out.append("")
    tf = (spec.signals or {}).get("text_features")
    if tf:
        out += ["**Text features**", "", "| Column | Avg chars | Avg words | Top TF-IDF terms |", "|---|--:|--:|---|"]
        for c, d in tf.items():
            out.append(f"| {c} | {d['mean_chars']} | {d['mean_words']} | {', '.join(d['top_terms'])} |")
        out.append("")
    return out


def _inference_md(inf: Any) -> list[str]:
    if inf is None:
        return []
    out: list[str] = []
    if inf.cis:
        out += ["**Bootstrap 95% confidence intervals**", "", "| Feature | Statistic | Estimate | 95% CI |", "|---|---|--:|---|"]
        for r in inf.cis[:20]:
            out.append(f"| {r.feature} | {r.statistic} | {_fmt(r.point)} | [{_fmt(r.lo)}, {_fmt(r.hi)}] |")
        out.append("")
    if inf.group_tests:
        out += ["**Group comparison tests (with effect size)**", "", "| Value | Group | Test | Statistic | p-value | Effect | Result |", "|---|---|---|--:|--:|---|---|"]
        for g in inf.group_tests:
            eff = f"{g.effect_name} {_fmt(g.effect_value)}" if g.effect_name else "—"
            out.append(f"| {g.value} | {g.group} | {g.test} | {_fmt(g.statistic)} | {_fmt(g.p_value)} | {eff} | {'significant' if g.significant else 'n.s.'} |")
        out.append("")
    sig = [c for c in inf.corr_sig if c.significant][:15]
    if sig:
        out += ["**Significant correlations (Benjamini-Hochberg)**", "", "| A | B | r | p | p (adj) |", "|---|---|--:|--:|--:|"]
        for c in sig:
            out.append(f"| {c.a} | {c.b} | {c.r:+.3f} | {_fmt(c.p_value)} | {_fmt(c.p_adj)} |")
        out.append("")
    reg = inf.regression
    if reg is not None and reg.terms:
        out += [f"**{reg.kind} regression on `{reg.target}`** (R2={_fmt(reg.r2)}, n={reg.n})", "",
                "| Term | Coef | Std err | p-value | 95% CI |", "|---|--:|--:|--:|---|"]
        for (name, coef, se, p, lo, hi) in reg.terms[:20]:
            out.append(f"| {name} | {_fmt(coef)} | {_fmt(se)} | {_fmt(p)} | [{_fmt(lo)}, {_fmt(hi)}] |")
        out.append("")
    elif reg is not None and reg.note:
        out += [f"_{reg.note}_", ""]
    if out:
        out += ["_Caveats: " + " ".join(inf.caveats) + "_", ""]
    return out


def _model_md(m: Any) -> list[str]:
    if m is None:
        return []
    if not m.scores:
        return [f"**Baseline models:** {m.note or 'unavailable'}", ""]
    keys = list(m.scores[0].metrics.keys())
    out = [f"**Baseline models — {m.task} on `{m.target}`** (n={m.n}, 5-fold CV)", "",
           "| Model | " + " | ".join(keys) + " |", "|---" + "|--:" * len(keys) + "|"]
    for s in m.scores:
        mark = " (best)" if s.name == m.best else ""
        out.append("| " + s.name + mark + " | " + " | ".join(_fmt(s.metrics.get(k)) for k in keys) + " |")
    out.append("")
    for x in m.leakage:
        out.append(f"- Possible leakage: {x}")
    if m.leakage:
        out.append("")
    if m.importances:
        out += ["**Permutation importance (best model)**", "", "| Feature | Importance |", "|---|--:|"]
        for f, v in m.importances:
            out.append(f"| {f} | {_fmt(v)} |")
        out.append("")
    if getattr(m, "shap_importances", None):
        out += ["**SHAP feature impact (best model)**", "", "| Feature | mean |SHAP| |", "|---|--:|"]
        for f, v in m.shap_importances:
            out.append(f"| {f} | {_fmt(v)} |")
        out.append("")
    out += ["_Caveats: " + " ".join(m.caveats) + "_", ""]
    return out


def _fda_md(f: Any) -> list[str]:
    if f is None:
        return []
    evr = ", ".join(f"{x:.0%}" for x in f.variance_ratio[:6])
    return [
        f"**Functional data analysis** (index `{f.time_index}`) — {f.n_curves} curves "
        f"× {f.n_points} points, smoothing: {f.smoothing}; modes for 90% variance: "
        f"{f.n_modes_90}; top-mode variance: {evr}",
        "",
    ]


def _tuning_md(t: Any) -> list[str]:
    if t is None:
        return []
    return [
        f"**Hyper-parameter tuning** ({t.metric}, {t.n_trials} trials): baseline {_fmt(t.baseline_cv)} "
        f"→ tuned {_fmt(t.best_cv)} (Δ {t.improvement:+.4f})",
        f"- best params: {t.best_params}", "",
    ]


def _causal_md(c: Any) -> list[str]:
    if c is None:
        return []
    out = [
        f"**A/B & causal — `{c.treatment}` → `{c.outcome}`**", "",
        f"- {c.n_treated} treated vs {c.n_control} control",
        f"- {c.effect_name}: {_fmt(c.effect)} [{_fmt(c.ci_low)}, {_fmt(c.ci_high)}], {c.test} p={_fmt(c.p_value)}",
    ]
    if c.effect_size_name:
        out.append(f"- {c.effect_size_name}: {_fmt(c.effect_size)}")
    if c.ipw_ate is not None:
        out.append(f"- propensity-IPW ATE: {_fmt(c.ipw_ate)} (naive {_fmt(c.naive_diff)}, propensity AUC {_fmt(c.propensity_auc)})")
    out += ["- ⚠️ " + " ".join(c.caveats), ""]
    return out


def _advanced_md(advanced: Any) -> list[str]:
    """Render the AdvancedAnalysis container into Markdown lines."""
    if advanced is None:
        return ["_Advanced analysis disabled._", ""]
    out: list[str] = []
    n_tests = _count_tests(advanced)
    if n_tests >= 10:
        out += [f"> **Statistical hygiene:** {n_tests} tests/associations computed automatically; "
                "treat unadjusted p-values cautiously (multiple comparisons).", ""]
    out += _spec_md(getattr(advanced, "specialization", None))
    if getattr(advanced, "normality", None):
        out += [
            "**Normality tests**", "",
            "| Feature | n | Shapiro p | D'Agostino p | Jarque-Bera p | Anderson stat/crit(5%) | Verdict |",
            "|---|--:|--:|--:|--:|--:|---|",
        ]
        for r in advanced.normality:
            out.append(
                f"| {r.feature} | {r.n:,} | {_fmt(r.shapiro_p)} | {_fmt(r.dagostino_p)} | "
                f"{_fmt(r.jarque_bera_p)} | {_fmt(r.anderson_stat)} / {_fmt(r.anderson_crit_5pct)} | "
                f"{'Normal' if r.is_normal else 'Non-normal'} |"
            )
        out.append("")
    if getattr(advanced, "transforms", None):
        out += ["**Recommended transforms**", "",
                "| Feature | Method | Skew before | Skew after | lambda |", "|---|---|--:|--:|--:|"]
        for t in advanced.transforms:
            out.append(f"| {t.feature} | {t.method} | {t.skew_before:+.3f} | {t.skew_after:+.3f} | {_fmt(t.lam)} |")
        out.append("")
    if getattr(advanced, "vif", None):
        out += ["**Multicollinearity (VIF)**", "", "| Feature | VIF | Severity |", "|---|--:|---|"]
        for v in advanced.vif:
            sev = "severe" if v.vif > 10 else ("moderate" if v.vif > 5 else "ok")
            out.append(f"| {v.feature} | {v.vif:,.2f} | {sev} |")
        out.append("")
    if getattr(advanced, "pca", None) is not None:
        p = advanced.pca
        out += [f"**PCA** — components for 90% variance: {p.n_components_90}; for 95%: {p.n_components_95}", ""]
    if getattr(advanced, "relevance", None):
        out += [f"**Feature relevance** — target `{advanced.target}` ({advanced.task_type})", "",
                "| Rank | Feature | Mutual info | F | p-value | corr |", "|--:|---|--:|--:|--:|--:|"]
        for r in advanced.relevance:
            out.append(
                f"| {r.rank} | {r.feature} | {_fmt(r.mutual_info)} | {_fmt(r.f_stat)} | "
                f"{_fmt(r.p_value)} | {_fmt(r.target_corr)} |"
            )
        out.append("")
    out += _inference_md(getattr(advanced, "inference", None))
    out += _model_md(getattr(advanced, "modeling", None))
    out += _fda_md(getattr(advanced, "fda", None))
    out += _extended_md(getattr(advanced, "extended", None))
    out += _tuning_md(getattr(advanced, "tuning", None))
    out += _causal_md(getattr(advanced, "causal", None))
    if not out:
        out = ["_No advanced diagnostics available._", ""]
    return out


def _impact_html(rep: StepReport) -> str:
    impacts = rep.metrics.get("impact")
    if not impacts:
        return ""
    rows = []
    for imp in impacts:
        dot = {"material": "🔴", "minor": "🟡"}.get(imp["verdict"], "🟢")
        mean_bits = (
            f"{_fmt(imp['mean_before'])} → {_fmt(imp['mean_after'])}"
            if imp["mean_before"] is not None else "—"
        )
        rows.append(
            f'<tr><td class="l">{_esc(imp["column"])}</td>'
            f'<td>{imp["cells_changed"]} ({imp["change_share"]:.1%})</td>'
            f"<td>{mean_bits}</td>"
            f"<td>{_fmt(imp['mean_shift_sd'], 3) if imp['mean_shift_sd'] is not None else '—'}</td>"
            f"<td>{_fmt(imp['ks_stat'], 3) if imp['ks_stat'] is not None else '—'}</td>"
            f'<td class="l">{dot} {imp["verdict"]}</td></tr>'
        )
    return (
        f"<p><em>Cleaning impact ({_esc(rep.step)}) — how much this step moved "
        "each column's distribution:</em></p>"
        '<table><thead><tr><th class="l">Column</th><th>Cells changed</th>'
        "<th>Mean before → after</th><th>Δmean (sd)</th><th>KS</th>"
        '<th class="l">Verdict</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _summary_html(summary_lines: list[str] | None) -> str:
    if not summary_lines:
        return ""
    import re as _re

    items = []
    for ln in summary_lines:
        text = ln.strip()
        if not text or text.startswith(">"):
            continue
        text = text.lstrip("- ").replace("[ ] ", "☐ ")
        text = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", _esc(text))
        text = _re.sub(r"`(.+?)`", r"<code>\1</code>", text)
        if text.startswith("<b>Review checklist"):
            items.append(f"<p class='meta'>{text}</p>")
        else:
            items.append(f"<li>{text}</li>")
    return (
        "<h2>Executive Summary</h2>"
        "<div class='meta'>What matters in this dataset, worst news first.</div>"
        f"<ul class='warn'>{''.join(items)}</ul>"
    )


def render_html(
    profile: DatasetProfile,
    step_reports: list[StepReport],
    *,
    title: str,
    source_name: str,
    config: CleanConfig,
    charts_head: str = "",
    charts_body: str = "",
    advanced: Any = None,
    summary_lines: list[str] | None = None,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    step_warns = [f"[{rep.step}] {w}" for rep in step_reports for w in rep.warnings]
    all_warns = step_warns + profile.warnings
    if all_warns:
        warn_html = '<ul class="warn">' + "".join(f"<li>{_esc(w)}</li>" for w in all_warns) + "</ul>"
    else:
        warn_html = '<ul class="warn ok"><li>✅ No critical data-health issues detected.</li></ul>'

    steps_html = []
    for rep in step_reports:
        items = "".join(f"<li>{_esc(a)}</li>" for a in rep.actions)
        items += "".join(f'<li style="color:#b54708">⚠️ {_esc(w)}</li>' for w in rep.warnings)
        steps_html.append(
            f"<h3>{_esc(rep.step)}</h3><ul class='actions'>{items}</ul>" + _impact_html(rep)
        )

    if profile.collinear_pairs:
        pair_rows = "".join(
            f'<tr><td class="l">{_esc(a)}</td><td class="l">{_esc(b)}</td><td>{r:+.3f}</td></tr>'
            for a, b, r in sorted(profile.collinear_pairs, key=lambda x: -abs(x[2]))
        )
        pairs_html = (
            '<table><thead><tr><th class="l">Feature A</th><th class="l">Feature B</th>'
            f"<th>Pearson r</th></tr></thead><tbody>{pair_rows}</tbody></table>"
        )
    else:
        pairs_html = "<p><em>No feature pairs exceed the collinearity threshold.</em></p>"

    config_json = _esc(_json_like(config.to_dict()))
    if charts_body:
        viz_body = charts_body
    elif not config.make_charts:
        viz_body = "<p><em>Charts disabled (make_charts=False).</em></p>"
    elif not _plotly_available():
        viz_body = (
            "<p><em>Charts unavailable — install plotly to enable interactive "
            "visualisations.</em></p>"
        )
    else:
        viz_body = "<p><em>No charts were generated for this dataset.</em></p>"
    advanced_html = _advanced_html(advanced)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title><style>{_CSS}</style>{charts_head}</head>
<body><div class="wrap">
<h1>{_esc(title)}</h1>
<div class="meta">Source: <code>{_esc(source_name)}</code> &nbsp;•&nbsp; Generated: {ts} &nbsp;•&nbsp; Engine: polars + plotly</div>
{_summary_html(summary_lines)}
<h2>1 · Dataset Overview</h2>
{_html_cards(profile, len(step_warns))}
<h2>2 · Data-Health Warnings</h2>
{warn_html}
<h2>3 · Interactive Visualisations</h2>
{viz_body}
<h2>4 · Column Profiles</h2>
{_html_columns_table(profile.columns)}
<h2>5 · Correlation Highlights</h2>
{pairs_html}
<h2>6 · Correlation Matrix (Pearson)</h2>
{_html_corr_matrix(profile)}
<h2>7 · Pipeline Actions</h2>
{''.join(steps_html)}
<h2>8 · Advanced Analysis</h2>
{advanced_html}
<h2>9 · Configuration</h2>
<details><summary>Show pipeline configuration</summary><pre>{config_json}</pre></details>
<div class="footer">Generated by auto_cleaner · polars-native autonomous data preprocessing &amp; EDA</div>
</div></body></html>"""


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_report(
    profile: DatasetProfile,
    step_reports: list[StepReport],
    *,
    title: str = "Automated EDA Report",
    source_name: str = "dataset",
    config: CleanConfig | None = None,
    charts_head: str = "",
    charts_body: str = "",
    chart_pngs: list[tuple[str, str]] | None = None,
    advanced: Any = None,
    summary_lines: list[str] | None = None,
) -> tuple[str, str]:
    """Return ``(markdown, html)`` strings for the given profile.

    ``charts_head`` / ``charts_body`` carry the interactive Plotly payload for
    the HTML report; ``chart_pngs`` is a list of ``(caption, relative_png_path)``
    embedded as images in the Markdown report; ``advanced`` is an
    ``AdvancedAnalysis`` rendered into the Advanced Analysis section.
    """
    config = config or CleanConfig()
    md = render_markdown(
        profile, step_reports, title=title, source_name=source_name,
        config=config, chart_pngs=chart_pngs, advanced=advanced,
        summary_lines=summary_lines,
    )
    html_doc = render_html(
        profile, step_reports, title=title, source_name=source_name,
        config=config, charts_head=charts_head, charts_body=charts_body, advanced=advanced,
        summary_lines=summary_lines,
    )
    return md, html_doc


def write_reports(
    markdown: str,
    html_doc: str,
    *,
    markdown_path: str | Path | None = None,
    html_path: str | Path | None = None,
) -> dict[str, str]:
    """Persist the report(s) to disk; returns a mapping of format → path."""
    written: dict[str, str] = {}
    if markdown_path is not None:
        p = Path(markdown_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown, encoding="utf-8")
        written["markdown"] = str(p)
    if html_path is not None:
        p = Path(html_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html_doc, encoding="utf-8")
        written["html"] = str(p)
    return written
