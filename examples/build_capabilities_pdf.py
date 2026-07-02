"""Generate the auto_cleaner *capabilities* PDF.

Two parts:
  1. **General** — what the tool does (pipeline, full capability matrix, formats,
     honest scope).
  2. **Proof of work** — real numbers from an actual run on the public auto-mpg
     ("cars") dataset, so the claims are demonstrated, not asserted.

Run:  python examples/build_capabilities_pdf.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from auto_cleaner import CleanConfig, run_pipeline

OUT = Path(__file__).parent / "output" / "auto_cleaner_capabilities.pdf"
RAW = Path(__file__).parent / "data" / "raw_cars.csv"

_ACCENT = (41, 82, 204)
_INK = (26, 34, 51)
_MUTED = (102, 112, 133)


def _s(text) -> str:
    return str("" if text is None else text).encode("latin-1", "replace").decode("latin-1")


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "-" if v != v else f"{v:,.4g}"
    return str(v)


def main() -> None:
    from fpdf import FPDF

    if not RAW.exists():
        from examples.generate_raw import main as gen  # type: ignore
        gen()

    # Run the real pipeline (quiet, in-memory) to harvest genuine numbers —
    # no side-effect files written.
    result = run_pipeline(
        RAW, None,
        CleanConfig().with_overrides(
            verbose=False, target="Miles_per_Gallon", make_pdf=False, make_charts=False
        ),
        write_reports_to_disk=False,
    )
    prof, adv = result.profile, result.advanced

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)

    def h1(t):
        pdf.set_font("Helvetica", "B", 22); pdf.set_text_color(*_INK)
        pdf.multi_cell(0, 10, _s(t), new_x="LMARGIN", new_y="NEXT")

    def h2(t):
        pdf.ln(3); pdf.set_font("Helvetica", "B", 14); pdf.set_text_color(*_ACCENT)
        pdf.multi_cell(0, 8, _s(t), new_x="LMARGIN", new_y="NEXT"); pdf.set_text_color(*_INK)

    def p(t, size=10):
        pdf.set_font("Helvetica", "", size); pdf.set_text_color(*_INK)
        pdf.multi_cell(0, 5, _s(t), new_x="LMARGIN", new_y="NEXT")

    def kv(area, desc):
        pdf.set_font("Helvetica", "B", 10); pdf.set_text_color(*_INK); pdf.write(5, _s(area + ": "))
        pdf.set_font("Helvetica", "", 10); pdf.write(5, _s(desc)); pdf.ln(6)

    def muted(t):
        pdf.set_font("Helvetica", "I", 9); pdf.set_text_color(*_MUTED)
        pdf.multi_cell(0, 4.5, _s(t), new_x="LMARGIN", new_y="NEXT"); pdf.set_text_color(*_INK)

    # ===================== PART 1 — GENERAL ============================= #
    pdf.add_page()
    h1("auto_cleaner")
    pdf.set_font("Helvetica", "", 12); pdf.set_text_color(*_MUTED)
    pdf.multi_cell(0, 6, _s("Capabilities & Proof of Work"), new_x="LMARGIN", new_y="NEXT")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, _s(f"Generated {ts}  |  polars-native autonomous data preprocessing & analysis"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*_INK)

    h2("What it is")
    p("auto_cleaner ingests any messy tabular dataset (CSV, Parquet, JSON, SQL, FITS, netCDF), "
      "produces a mathematically clean, ML-ready dataset, and emits a comprehensive analysis "
      "report (interactive HTML, Markdown and PDF). It is built end-to-end on polars (no pandas "
      "in the engine) with DuckDB for SQL. Crucially, it inspects each dataset and AUTO-SPECIALISES: "
      "it detects the data's archetype and runs the analyses that actually fit.")

    h2("Pipeline")
    p("ingest  ->  validate  ->  standardise  ->  impute  ->  outliers  ->  downcast  ->  "
      "profile  ->  auto-specialise  ->  inference / modelling / FDA / extended stats  ->  report.")

    h2("Capability matrix")
    kv("Ingestion", "CSV/TSV/Parquet/JSON/NDJSON, DuckDB SQL, FITS (astronomy), netCDF (climate); "
                    "auto format/delimiter/encoding/header detection; streaming/out-of-core.")
    kv("Cleaning", "safe dtype downcasting; imputation (median/mean/KNN/time-series ffill); outliers "
                   "(IQR, Z-score, Isolation Forest); datetime/numeric-string/whitespace/categorical standardisation.")
    kv("EDA", "per-column profile, skewness, kurtosis, correlation & covariance, missingness, "
              "data-health warnings; HTML + Markdown + PDF reports.")
    kv("Visualisation", "interactive offline Plotly charts (histograms, frequency bars, scatter matrix, "
                        "boxplots, correlation heatmap, missingness) + standalone PNG export.")
    kv("Auto-specialisation", "detects time-series, geospatial, text-heavy, high-dimensional/embeddings, "
                              "survey, wide/omics, image-reference archetypes and routes modules accordingly.")
    kv("Advanced", "normality tests (Shapiro/D'Agostino/Jarque-Bera/Anderson-Darling), Box-Cox/Yeo-Johnson "
                   "transforms, VIF + PCA, target-aware feature relevance (mutual information, ANOVA F).")
    kv("Inference", "bootstrap confidence intervals; auto-selected group tests (t/Welch/Mann-Whitney/ANOVA/"
                    "Kruskal/chi2) WITH effect sizes (Cohen's d, Cliff's delta, eta^2); BH-corrected correlation "
                    "significance; OLS/logit regression with p-values and CIs.")
    kv("Modelling", "cross-validated baselines (dummy/linear/forest/boosting) with metrics, permutation "
                    "importance and a leakage check - a benchmark for a human, not a deployable model.")
    kv("Extended statistics", "robust means (geometric/harmonic/trimmed/winsorized/MAD/Huber); rank, partial & "
                              "categorical associations (Spearman/Kendall/partial/Cramer's V/eta); distribution "
                              "fitting (AIC/BIC); multivariate (Mahalanobis, clustering, MANOVA, UMAP); classical "
                              "NLP (LDA topics, sentiment); Bayes factors; survival (KM, Cox); survey (Cronbach, weights).")
    kv("Functional (FDA)", "smoothing + functional PCA of time-indexed curves.")
    kv("Hardening", "typed validation + schema enforcement, expanded test suite, GitHub Actions CI.")

    h2("Honest scope & limitations")
    p("- Modelling and inference are baselines/diagnostics for a human to validate, not final causal claims "
      "or production models.\n"
      "- Domain-aware via heuristics, not a domain expert (no automatic survey weighting design, omics "
      "normalisation, astrometric calibration).\n"
      "- Tabular focus; no computer vision / audio / graph deep learning. Neural text embeddings are opt-in.\n"
      "- A clean, tested package - not a library hardened by years of production edge-cases.")

    # ===================== PART 2 — PROOF OF WORK ======================= #
    pdf.add_page()
    h1("Proof of work")
    muted("Every number below comes from a real auto_cleaner run on the public auto-mpg ('cars') dataset "
          "(406 rows x 9 columns), not from hand-picked examples.")

    h2("Run summary")
    saved = result.memory_before - result.memory_after
    pct = (saved / result.memory_before * 100) if result.memory_before else 0
    p(f"Rows in -> out: {result.rows_in:,} -> {result.rows_out:,}")
    p(f"Memory in -> out: {result.memory_before/1024:.1f} KB -> {result.memory_after/1024:.1f} KB  (-{pct:.1f}%)")
    p(f"Data-health warnings raised: {len(prof.warnings)}    Elapsed: {result.elapsed_s:.2f}s")

    spec = getattr(adv, "specialization", None)
    if spec is not None:
        h2("Auto-specialisation")
        p("Detected: " + ", ".join(f"{n} ({c:.0%})" for n, c in spec.archetypes))
        p("Auto-modules run: " + (", ".join(spec.auto_modules) or "none"))

    if prof.warnings:
        h2("Data-health warnings (verbatim)")
        for w in prof.warnings:
            p(f"- {w}")

    mdl = getattr(adv, "modeling", None)
    if mdl is not None and mdl.scores:
        h2("Baseline models (5-fold cross-validation)")
        for s in mdl.scores:
            mark = "  <- best" if s.name == mdl.best else ""
            p(f"- {s.name}: " + ", ".join(f"{k}={_fmt(v)}" for k, v in s.metrics.items()) + mark)

    inf = getattr(adv, "inference", None)
    if inf is not None:
        sig = [c for c in inf.corr_sig if c.significant][:5]
        if sig:
            h2("Significant correlations (Benjamini-Hochberg corrected)")
            for c in sig:
                p(f"- {c.a} ~ {c.b}: r={c.r:+.3f}, p(adj)={_fmt(c.p_adj)}")
        if inf.regression is not None and inf.regression.terms:
            reg = inf.regression
            h2(f"{reg.kind} regression on '{reg.target}' (R2={_fmt(reg.r2)}, n={reg.n})")
            for name, coef, se, pv, lo, hi in reg.terms[:6]:
                p(f"- {name}: coef={_fmt(coef)}, p={_fmt(pv)}, 95% CI [{_fmt(lo)}, {_fmt(hi)}]")

    ext = getattr(adv, "extended", None)
    if ext is not None and getattr(ext, "distributions", None):
        h2("Best-fit distributions (lowest AIC)")
        for d in ext.distributions[:6]:
            p(f"- {d.feature}: {d.best_distribution} (AIC {_fmt(d.aic)}, KS p {_fmt(d.ks_p)})")
    if ext is not None and getattr(ext, "multivariate", None) is not None:
        m = ext.multivariate
        h2("Multivariate")
        p(f"Mahalanobis outliers: {m.mahalanobis_outliers}; clusters: {m.best_k} "
          f"(silhouette {_fmt(m.silhouette)}); 2-D PCA variance: {_fmt(m.pca_2d_variance)}")

    muted("Alongside this document, each run also produces an interactive HTML report, a Markdown report, "
          "a per-dataset PDF, and standalone PNG charts.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
