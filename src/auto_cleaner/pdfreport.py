"""Auto-generated per-dataset PDF report (via fpdf2).

Renders a polished, static PDF of a run: overview, data-health warnings,
auto-specialisation, column profiles, the exported charts, and a compact summary
of the advanced findings. Degrades to a no-op (returns ``None``) if fpdf2 is not
installed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from auto_cleaner.eda.stats import DatasetProfile
from auto_cleaner.logging_utils import human_bytes

__all__ = ["build_pdf"]

_INK = (26, 34, 51)
_ACCENT = (41, 82, 204)
_MUTED = (102, 112, 133)
_WARN = (181, 71, 8)


def _s(text: Any) -> str:
    """Latin-1-safe string for fpdf2 core fonts."""
    return str("" if text is None else text).encode("latin-1", "replace").decode("latin-1")


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "-" if v != v else f"{v:,.4g}"
    return str(v)


def build_pdf(
    *,
    out_path: str | Path,
    title: str,
    source_name: str,
    profile: DatasetProfile,
    advanced: Any,
    chart_png_paths: list[tuple[str, str]],
    config: Any = None,
    summary_lines: list[str] | None = None,
) -> str | None:
    """Build the per-dataset PDF; returns the path or ``None`` if fpdf2 missing."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    epw = pdf.epw  # effective page width

    def heading(text: str, size: int = 14) -> None:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", size)
        pdf.set_text_color(*_ACCENT)
        pdf.cell(0, 8, _s(text), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*_INK)

    def body(text: str, size: int = 10) -> None:
        pdf.set_font("Helvetica", "", size)
        pdf.multi_cell(0, 5, _s(text), new_x="LMARGIN", new_y="NEXT")

    def table(headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(243, 245, 249)
        for h, w in zip(headers, widths):
            pdf.cell(w, 6, _s(h), border=1, fill=True)
        pdf.ln()
        pdf.set_font("Helvetica", "", 8)
        for row in rows:
            for cell, w in zip(row, widths):
                txt = _s(cell)
                if len(txt) > int(w / 1.7):
                    txt = txt[: int(w / 1.7) - 1] + "."
                pdf.cell(w, 5.2, txt, border=1)
            pdf.ln()

    # --- Cover -------------------------------------------------------------- #
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.multi_cell(0, 9, _s(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*_MUTED)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf.multi_cell(
        0, 5,
        _s(f"Source: {source_name}   |   Generated: {ts}   |   Engine: polars + auto_cleaner"),
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.set_text_color(*_INK)

    # --- Executive summary ---------------------------------------------------#
    if summary_lines:
        heading("Executive summary")
        for ln in summary_lines:
            text = ln.strip()
            if not text or text.startswith(">"):
                continue
            text = text.lstrip("- ").replace("**", "").replace("`", "")
            if text.startswith("[ ] "):
                text = "[ ] " + text[4:]
                body(f"   {text}", size=9)
            elif text.startswith("Review checklist"):
                pdf.ln(1)
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 6, _s("Review checklist (human sign-off):"),
                         new_x="LMARGIN", new_y="NEXT")
            else:
                body(f"- {text}")

    # --- Overview ----------------------------------------------------------- #
    heading("1. Dataset overview")
    body(f"Data-quality score: {profile.quality_score}/100   (components: {profile.quality_components})")
    body(
        f"Rows: {profile.n_rows:,}    Columns: {profile.n_cols}    "
        f"In-memory: {human_bytes(profile.memory_bytes)}    "
        f"Duplicate rows: {profile.duplicate_rows:,}    Warnings: {len(profile.warnings)}"
    )

    # --- Warnings ----------------------------------------------------------- #
    heading("2. Data-health warnings")
    if profile.warnings:
        for w in profile.warnings:
            body(f"- {w}")
    else:
        body("No critical data-health issues detected.")

    # --- Specialisation ----------------------------------------------------- #
    spec = getattr(advanced, "specialization", None)
    if spec is not None:
        heading("3. Auto-specialisation")
        arche = ", ".join(f"{n} ({c:.0%})" for n, c in spec.archetypes)
        body(f"Detected archetype(s): {arche}")
        body(f"Auto-modules: {', '.join(spec.auto_modules) or 'none'}")
        for r in spec.recommendations:
            body(f"- {r}")

    # --- Column profiles ---------------------------------------------------- #
    heading("4. Column profiles")
    widths = [epw * x for x in (0.26, 0.14, 0.12, 0.10, 0.12, 0.13, 0.13)]
    rows = [
        [c.name, c.dtype, c.kind, f"{c.null_pct:.1f}%", f"{c.n_unique:,}", _fmt(c.mean), _fmt(c.median)]
        for c in profile.columns[:40]
    ]
    table(["Feature", "Dtype", "Kind", "Null%", "Unique", "Mean", "Median"], rows, widths)

    # --- Advanced highlights ------------------------------------------------ #
    heading("5. Advanced analysis highlights")
    _advanced_highlights(body, advanced)

    # --- Charts ------------------------------------------------------------- #
    if chart_png_paths:
        pdf.add_page()
        heading("6. Visualisations")
        for caption, path in chart_png_paths:
            if not Path(path).exists():
                continue
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*_MUTED)
            pdf.cell(0, 5, _s(caption), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(*_INK)
            try:
                pdf.image(path, w=epw * 0.92)
            except Exception:  # noqa: BLE001
                pass
            pdf.ln(2)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out))
    return str(out)


def _advanced_highlights(body, advanced: Any) -> None:
    """Write a compact textual summary of the advanced results into the PDF."""
    if advanced is None:
        body("Advanced analysis disabled.")
        return
    inf = getattr(advanced, "inference", None)
    if inf is not None and inf.corr_sig:
        sig = [c for c in inf.corr_sig if c.significant][:3]
        if sig:
            body("Significant correlations: " + "; ".join(f"{c.a}~{c.b} r={c.r:+.2f}" for c in sig))
    if inf is not None and inf.regression is not None and inf.regression.terms:
        reg = inf.regression
        body(f"{reg.kind} regression on '{reg.target}': R2={_fmt(reg.r2)} (n={reg.n}).")
    mdl = getattr(advanced, "modeling", None)
    if mdl is not None and mdl.scores and mdl.best:
        best = next((s for s in mdl.scores if s.name == mdl.best), None)
        if best:
            metrics = ", ".join(f"{k}={_fmt(v)}" for k, v in best.metrics.items())
            body(f"Best baseline model: {mdl.best} ({metrics}).")
    ext = getattr(advanced, "extended", None)
    if ext is not None:
        if getattr(ext, "distributions", None):
            d = ext.distributions[0]
            body(f"Distribution fit (example): '{d.feature}' best ~ {d.best_distribution} (AIC {_fmt(d.aic)}).")
        m = getattr(ext, "multivariate", None)
        if m is not None and m.best_k:
            body(f"Clustering: {m.best_k} clusters (silhouette {_fmt(m.silhouette)}); Mahalanobis outliers: {m.mahalanobis_outliers}.")
        if getattr(ext, "timeseries", None):
            r = ext.timeseries[0]
            body(f"Time-series (example '{r.feature}'): stationary={r.stationary}, trend={r.mk_trend}, seasonal strength={_fmt(r.seasonal_strength)}.")
    fda = getattr(advanced, "fda", None)
    if fda is not None:
        body(f"Functional PCA: {fda.n_modes_90} mode(s) capture 90% of curve variance.")
