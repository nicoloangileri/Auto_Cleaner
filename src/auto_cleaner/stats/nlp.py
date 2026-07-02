"""Classical NLP for free-text columns: topic modelling + sentiment.

* **LDA topic modelling** (bag-of-words) surfaces the latent themes,
* **VADER sentiment** gives a lexicon-based polarity distribution.

This is deliberately classical and fast (no neural networks). Transformer
embeddings live in :mod:`auto_cleaner.embeddings` and are opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = ["NLPColumnReport", "nlp_analysis"]


@dataclass(slots=True)
class NLPColumnReport:
    column: str
    n_docs: int
    topics: list[str] = field(default_factory=list)
    sentiment_mean: float | None = None
    positive_pct: float | None = None
    negative_pct: float | None = None


def _topics(texts: list[str], seed: int, n_topics: int = 5, n_words: int = 8) -> list[str]:
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(max_features=500, stop_words="english", min_df=2)
    dtm = vec.fit_transform(texts)
    if dtm.shape[1] < n_words:
        return []
    k = min(n_topics, max(2, dtm.shape[0] // 10))
    lda = LatentDirichletAllocation(n_components=k, random_state=seed, max_iter=10)
    lda.fit(dtm)
    vocab = vec.get_feature_names_out()
    out = []
    for comp in lda.components_:
        top = [vocab[i] for i in comp.argsort()[::-1][:n_words]]
        out.append(", ".join(top))
    return out


def nlp_analysis(
    df: pl.DataFrame, text_columns: list[str], config: CleanConfig | None = None
) -> list[NLPColumnReport]:
    """Topic-model and sentiment-score each free-text column."""
    config = config or CleanConfig()
    out: list[NLPColumnReport] = []
    for c in text_columns:
        texts = [t for t in df.get_column(c).drop_nulls().to_list() if isinstance(t, str) and t.strip()]
        if len(texts) < 10:
            continue
        rep = NLPColumnReport(column=c, n_docs=len(texts))
        try:
            rep.topics = _topics(texts[:5000], config.random_seed)
        except Exception:  # noqa: BLE001
            pass
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            sia = SentimentIntensityAnalyzer()
            comps = [sia.polarity_scores(t)["compound"] for t in texts[:5000]]
            if comps:
                import numpy as np

                arr = np.asarray(comps)
                rep.sentiment_mean = round(float(arr.mean()), 3)
                rep.positive_pct = round(float((arr > 0.05).mean() * 100), 1)
                rep.negative_pct = round(float((arr < -0.05).mean() * 100), 1)
        except Exception:  # noqa: BLE001
            pass
        out.append(rep)
    return out
