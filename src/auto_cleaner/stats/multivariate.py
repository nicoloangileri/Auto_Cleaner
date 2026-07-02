"""Multivariate analysis: Mahalanobis outliers, clustering, MANOVA, projection.

* **Mahalanobis distance** flags genuinely multivariate outliers (vs the
  per-column view of IQR/Z-score).
* **Clustering** picks k by silhouette and reports cluster structure.
* **MANOVA** tests whether numeric means differ across a categorical factor.
* **Projection** reports 2-D PCA variance and (if available) a UMAP embedding's
  cluster separability — the natural lens for embeddings / high-dimensional data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = ["MultivariateReport", "multivariate_analysis"]

_NUMERIC = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


@dataclass(slots=True)
class MultivariateReport:
    n_numeric: int
    mahalanobis_outliers: int | None = None
    mahalanobis_threshold: float | None = None
    best_k: int | None = None
    silhouette: float | None = None
    cluster_sizes: list[int] = field(default_factory=list)
    pca_2d_variance: float | None = None
    manova_p: float | None = None
    umap_silhouette: float | None = None
    projection_note: str | None = None


def multivariate_analysis(
    df: pl.DataFrame, config: CleanConfig | None = None, *, id_columns: list[str] | None = None
) -> MultivariateReport | None:
    """Run the multivariate diagnostic suite on the numeric feature space."""
    config = config or CleanConfig()
    exclude = set(id_columns or []) | {"is_outlier"}
    numeric = [c for c, dt in zip(df.columns, df.dtypes) if dt in _NUMERIC and c not in exclude and df.get_column(c).n_unique() > 2]
    if len(numeric) < 2:
        return None
    try:
        import numpy as np
        from scipy import stats
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None

    sub = df.select(numeric).drop_nulls()
    if sub.height < max(20, len(numeric) + 5):
        return None
    rep = MultivariateReport(n_numeric=len(numeric))
    X = StandardScaler().fit_transform(sub.to_numpy().astype(float))
    n = X.shape[0]
    seed = config.random_seed
    rng = np.random.default_rng(seed)
    Xs = X if n <= 3000 else X[rng.choice(n, 3000, replace=False)]

    # Mahalanobis outliers
    try:
        cov = np.cov(X, rowvar=False)
        inv = np.linalg.pinv(cov)
        diff = X - X.mean(axis=0)
        d2 = np.einsum("ij,ij->i", diff @ inv, diff)
        thresh = float(stats.chi2.ppf(0.975, df=len(numeric)))
        rep.mahalanobis_outliers = int((d2 > thresh).sum())
        rep.mahalanobis_threshold = round(thresh, 2)
    except Exception:  # noqa: BLE001
        pass

    # KMeans clustering, k chosen by silhouette
    try:
        best = (-1.0, None, None)
        for k in range(2, min(7, Xs.shape[0] - 1)):
            labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xs)
            score = silhouette_score(Xs, labels)
            if score > best[0]:
                best = (score, k, labels)
        if best[1] is not None:
            rep.best_k = int(best[1])
            rep.silhouette = round(float(best[0]), 3)
            _, counts = np.unique(best[2], return_counts=True)
            rep.cluster_sizes = [int(x) for x in counts]
    except Exception:  # noqa: BLE001
        pass

    # 2-D PCA variance
    try:
        from sklearn.decomposition import PCA

        rep.pca_2d_variance = round(float(PCA(n_components=2).fit(X).explained_variance_ratio_.sum()), 3)
    except Exception:  # noqa: BLE001
        pass

    # MANOVA across a categorical factor, if one exists
    try:
        import pandas as pd
        from statsmodels.multivariate.manova import MANOVA

        factor = None
        for c, dt in zip(df.columns, df.dtypes):
            if c in exclude:
                continue
            if dt in (pl.Utf8, pl.Categorical, pl.Boolean) and 2 <= df.get_column(c).n_unique() <= 6:
                factor = c
                break
        if factor is not None:
            cols = numeric[:6]
            pdf = df.select(cols + [factor]).drop_nulls().to_pandas()
            safe = {c: c.replace(" ", "_") for c in cols}
            pdf = pdf.rename(columns=safe)
            formula = " + ".join(safe.values()) + " ~ C(Q('" + factor + "'))"
            m = MANOVA.from_formula(formula, data=pdf)
            wilks_p = m.mv_test().results["C(Q('" + factor + "'))"]["stat"].loc["Wilks' lambda", "Pr > F"]
            rep.manova_p = float(wilks_p)
    except Exception:  # noqa: BLE001
        pass

    # UMAP projection separability (optional / heavy)
    try:
        import umap  # type: ignore

        if Xs.shape[0] >= 30:
            emb = umap.UMAP(n_components=2, random_state=seed).fit_transform(Xs)
            if rep.best_k:
                lab = KMeans(n_clusters=rep.best_k, n_init=10, random_state=seed).fit_predict(emb)
                rep.umap_silhouette = round(float(silhouette_score(emb, lab)), 3)
            rep.projection_note = "UMAP 2-D embedding computed"
    except Exception:  # noqa: BLE001
        pass

    return rep
