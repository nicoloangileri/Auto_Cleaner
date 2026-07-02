"""A/B testing + observational causal inference (strictly opt-in).

Runs ONLY when the user explicitly names a ``treatment`` and an ``outcome`` — it
is never auto-triggered, by design. Provides:

* an **A/B test** (group means/rates, effect, 95% CI, test, effect size), and
* an **observational causal ATE** via propensity-score inverse-probability
  weighting (IPW), with a balance/overlap diagnostic.

Every result ships with prominent caveats. Automating causal inference carelessly
produces confident-but-wrong conclusions — these outputs are hypothesis-generating,
not proof of causation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = ["CausalReport", "causal_analysis"]

_NUMERIC = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)
_CAVEATS = (
    "Observational causal estimates assume NO unmeasured confounders (ignorability) — usually unverifiable.",
    "They require overlap/positivity; a propensity AUC near 1.0 signals poor overlap and an unreliable estimate.",
    "A/B results are valid as causal only if the treatment was actually randomised.",
    "Treat all of this as hypothesis-generating, not proof of causation.",
)


@dataclass(slots=True)
class CausalReport:
    treatment: str
    outcome: str
    outcome_kind: str
    n_treated: int
    n_control: int
    effect_name: str
    effect: float
    ci_low: float
    ci_high: float
    p_value: float
    test: str
    effect_size: float | None = None
    effect_size_name: str | None = None
    ipw_ate: float | None = None
    naive_diff: float | None = None
    propensity_auc: float | None = None
    caveats: tuple[str, ...] = _CAVEATS


def _binary_codes(series: pl.Series):
    """Map a 2-level treatment to 0/1 (sorted order); return (codes, ok)."""
    import numpy as np

    if series.dtype == pl.Boolean:
        return series.cast(pl.Int64).to_numpy(), True
    levels = sorted(str(v) for v in series.drop_nulls().unique().to_list())
    if len(levels) != 2:
        return None, False
    mapping = {levels[0]: 0, levels[1]: 1}
    return np.array([mapping[str(v)] for v in series.to_list()], dtype=int), True


def _covariate_matrix(frame: pl.DataFrame, exclude: set[str]):
    import numpy as np

    cols = []
    for c, dt in zip(frame.columns, frame.dtypes):
        if c in exclude:
            continue
        s = frame.get_column(c)
        if dt in _NUMERIC:
            cols.append(s.cast(pl.Float64).fill_null(s.median()).to_numpy())
        elif dt in (pl.Utf8, pl.Categorical, pl.Boolean) and s.n_unique() <= 50:
            codes = (s.cast(pl.Int8) if dt == pl.Boolean else s.cast(pl.Categorical).to_physical()).fill_null(-1).to_numpy()
            cols.append(codes.astype(float))
    return np.column_stack(cols) if cols else None


def causal_analysis(
    df: pl.DataFrame, treatment: str, outcome: str, config: CleanConfig | None = None,
    *, id_columns: list[str] | None = None,
) -> CausalReport | None:
    """A/B test + propensity-IPW ATE for ``treatment`` -> ``outcome``."""
    config = config or CleanConfig()
    if treatment not in df.columns or outcome not in df.columns:
        return None
    try:
        import numpy as np
        from scipy import stats
    except ImportError:
        return None

    frame = df.drop_nulls(subset=[treatment, outcome])
    t, ok = _binary_codes(frame.get_column(treatment))
    if not ok:
        return None

    out_series = frame.get_column(outcome)
    binary_outcome = out_series.n_unique() == 2 and out_series.dtype != pl.Float64
    if binary_outcome:
        y, _ = _binary_codes(out_series)
        y = y.astype(float)
        kind = "binary"
    else:
        y = out_series.cast(pl.Float64).to_numpy().astype(float)
        kind = "numeric"

    mask = np.isfinite(y)
    t, y = t[mask], y[mask]
    yt, yc = y[t == 1], y[t == 0]
    if yt.size < 5 or yc.size < 5:
        return None

    if kind == "numeric":
        diff = float(yt.mean() - yc.mean())
        se = float(np.sqrt(yt.var(ddof=1) / yt.size + yc.var(ddof=1) / yc.size))
        p = float(stats.ttest_ind(yt, yc, equal_var=False).pvalue)
        pooled = float(np.sqrt(((yt.size - 1) * yt.var(ddof=1) + (yc.size - 1) * yc.var(ddof=1)) / (yt.size + yc.size - 2)))
        eff_size = None if pooled == 0 else round(diff / pooled, 3)
        report = CausalReport(
            treatment=treatment, outcome=outcome, outcome_kind=kind,
            n_treated=int(yt.size), n_control=int(yc.size),
            effect_name="mean difference (treated - control)", effect=round(diff, 4),
            ci_low=round(diff - 1.96 * se, 4), ci_high=round(diff + 1.96 * se, 4),
            p_value=p, test="Welch t-test", effect_size=eff_size, effect_size_name="Cohen's d",
        )
    else:
        pt, pc = float(yt.mean()), float(yc.mean())
        rd = pt - pc
        se = float(np.sqrt(pt * (1 - pt) / yt.size + pc * (1 - pc) / yc.size))
        p_pool = (yt.sum() + yc.sum()) / (yt.size + yc.size)
        se_pool = float(np.sqrt(p_pool * (1 - p_pool) * (1 / yt.size + 1 / yc.size)))
        z = rd / se_pool if se_pool > 0 else 0.0
        p = float(2 * (1 - stats.norm.cdf(abs(z))))
        rr = round(pt / pc, 3) if pc > 0 else None
        report = CausalReport(
            treatment=treatment, outcome=outcome, outcome_kind=kind,
            n_treated=int(yt.size), n_control=int(yc.size),
            effect_name="risk difference (treated - control)", effect=round(rd, 4),
            ci_low=round(rd - 1.96 * se, 4), ci_high=round(rd + 1.96 * se, 4),
            p_value=p, test="two-proportion z-test", effect_size=rr, effect_size_name="relative risk",
        )
    report.naive_diff = round(float(yt.mean() - yc.mean()), 4)

    # Propensity-score IPW for the observational causal estimate.
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        exclude = set(id_columns or []) | {"is_outlier", treatment, outcome}
        x = _covariate_matrix(frame, exclude)
        if x is not None and x.shape[1] >= 1:
            x = StandardScaler().fit_transform(x)[mask]
            model = LogisticRegression(max_iter=1000).fit(x, t)
            e = np.clip(model.predict_proba(x)[:, 1], 0.05, 0.95)
            report.ipw_ate = round(float(np.mean(t * y / e - (1 - t) * y / (1 - e))), 4)
            report.propensity_auc = round(float(roc_auc_score(t, e)), 3)
    except Exception:  # noqa: BLE001
        pass

    return report
