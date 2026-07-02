"""Optional neural text embeddings (sentence-transformers).

Heavy and opt-in (``CleanConfig.use_transformer_embeddings``): downloads a model
and encodes text, then summarises the embedding geometry (2-D PCA variance and
cluster separability). Degrades silently if the library/model is unavailable, so
it never blocks a run.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = ["EmbeddingReport", "embedding_analysis"]


@dataclass(slots=True)
class EmbeddingReport:
    column: str
    model: str
    n_docs: int
    dim: int
    pca_2d_variance: float | None = None
    best_k: int | None = None
    silhouette: float | None = None


def embedding_analysis(
    df: pl.DataFrame, text_columns: list[str], config: CleanConfig | None = None
) -> list[EmbeddingReport]:
    """Encode text columns with a sentence-transformer and summarise geometry."""
    config = config or CleanConfig()
    if not config.use_transformer_embeddings or not text_columns:
        return []
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
    except Exception:  # noqa: BLE001 — torch/model may be missing
        return []

    model_name = config.embedding_model
    try:
        model = SentenceTransformer(model_name)
    except Exception:  # noqa: BLE001 — offline / download blocked
        return []

    out: list[EmbeddingReport] = []
    for c in text_columns:
        texts = [t for t in df.get_column(c).drop_nulls().to_list() if isinstance(t, str) and t.strip()]
        if len(texts) < 10:
            continue
        try:
            emb = np.asarray(model.encode(texts[:2000], show_progress_bar=False))
        except Exception:  # noqa: BLE001
            continue
        rep = EmbeddingReport(column=c, model=model_name, n_docs=int(emb.shape[0]), dim=int(emb.shape[1]))
        try:
            rep.pca_2d_variance = round(float(PCA(n_components=2).fit(emb).explained_variance_ratio_.sum()), 3)
        except Exception:  # noqa: BLE001
            pass
        try:
            best = (-1.0, None)
            for k in range(2, min(7, emb.shape[0] - 1)):
                labels = KMeans(n_clusters=k, n_init=10, random_state=config.random_seed).fit_predict(emb)
                score = silhouette_score(emb, labels)
                if score > best[0]:
                    best = (score, k)
            if best[1] is not None:
                rep.silhouette = round(float(best[0]), 3)
                rep.best_k = int(best[1])
        except Exception:  # noqa: BLE001
            pass
        out.append(rep)
    return out
