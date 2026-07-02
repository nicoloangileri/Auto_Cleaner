"""Auto-specialisation engine — let the *data* decide how to analyse itself.

Instead of treating every dataset identically, this inspects the schema, dtypes
and value patterns to infer one or more **archetypes** (time-series, geospatial,
text-heavy, high-dimensional/embeddings, survey, wide/omics, image-references,
generic tabular). From those it emits:

* ``signals``        — the concrete facts it found,
* ``recommendations``— tailored, honest guidance, and
* ``auto_modules``   — which advanced modules the pipeline should auto-run.

It is heuristic and transparent by design: every decision is reported, so a
human can see *why* a specialisation fired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = ["Specialization", "detect_specialization"]

_INT_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
)
_NUMERIC = _INT_DTYPES + (pl.Float32, pl.Float64)
_LAT_NAMES = {"lat", "latitude", "ylat", "y_lat", "declination", "dec"}
_LON_NAMES = {"lon", "long", "lng", "longitude", "xlon", "x_lon", "ra"}
_TIME_HINTS = ("date", "time", "timestamp", "datetime", "year", "month", "day", "epoch", "mjd")
_ID_HINTS = ("id", "uuid", "guid", "key", "index", "_id")
_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".fits", ".fit", ".bmp", ".gif")


@dataclass(slots=True)
class Specialization:
    """The data-driven analysis plan."""

    archetypes: list[tuple[str, float]] = field(default_factory=list)  # (name, confidence)
    signals: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    auto_modules: list[str] = field(default_factory=list)
    time_index: str | None = None
    geo: dict[str, str] | None = None
    text_columns: list[str] = field(default_factory=list)
    id_columns: list[str] = field(default_factory=list)
    embedding_columns: list[str] = field(default_factory=list)

    @property
    def primary(self) -> str:
        return self.archetypes[0][0] if self.archetypes else "generic tabular"


def _kinds(df: pl.DataFrame) -> dict[str, list[str]]:
    numeric, categorical, boolean, datetime, text = [], [], [], [], []
    for c, dt in zip(df.columns, df.dtypes):
        if dt in _NUMERIC:
            numeric.append(c)
        elif dt == pl.Boolean:
            boolean.append(c)
        elif dt in (pl.Date, pl.Datetime, pl.Time):
            datetime.append(c)
        elif dt in (pl.Utf8, pl.Categorical):
            categorical.append(c)
    return {"numeric": numeric, "categorical": categorical, "boolean": boolean, "datetime": datetime}


def _detect_time_index(df: pl.DataFrame, datetime_cols: list[str]) -> str | None:
    for c in datetime_cols:
        s = df.get_column(c)
        if s.null_count() == 0 and s.n_unique() / max(s.len(), 1) >= 0.9:
            diffs = s.to_physical().diff().drop_nulls()
            if diffs.len() and bool((diffs >= 0).all()):
                return c
    # name-hinted monotonic numeric (e.g. integer 'year')
    for c, dt in zip(df.columns, df.dtypes):
        if dt in _NUMERIC and any(h in c.lower() for h in _TIME_HINTS):
            s = df.get_column(c).drop_nulls()
            if s.len() > 2 and bool((s.diff().drop_nulls() >= 0).all()):
                return c
    return None


def _detect_geo(df: pl.DataFrame, numeric: list[str]) -> dict[str, str] | None:
    lat = lon = None
    for c in numeric:
        lo = c.lower()
        s = df.get_column(c)
        mn, mx = s.min(), s.max()
        if mn is None or mx is None:
            continue
        if lat is None and lo in _LAT_NAMES and -90.0 <= mn and mx <= 90.0:
            lat = c
        elif lon is None and lo in _LON_NAMES and -180.0 <= mn and mx <= 360.0:
            lon = c
    return {"lat": lat, "lon": lon} if lat and lon else None


def _detect_text(df: pl.DataFrame) -> list[str]:
    out = []
    for c, dt in zip(df.columns, df.dtypes):
        if dt != pl.Utf8:
            continue
        s = df.get_column(c).drop_nulls()
        if s.len() == 0:
            continue
        mean_len = df.select(pl.col(c).str.len_chars().mean()).item() or 0.0
        uniq_ratio = s.n_unique() / s.len()
        space_frac = df.select(pl.col(c).str.contains(" ").mean()).item() or 0.0
        has_spaces = space_frac > 0.5
        if mean_len > 40 or (uniq_ratio > 0.8 and mean_len > 15 and has_spaces):
            out.append(c)
    return out


def _detect_ids(df: pl.DataFrame) -> list[str]:
    """Identifier columns: integer/text only (never continuous floats), and either
    name-hinted or a contiguous integer sequence / fully-unique text key."""
    out = []
    n = df.height
    for c, dt in zip(df.columns, df.dtypes):
        if dt not in _INT_DTYPES and dt != pl.Utf8:  # floats/dates are never IDs
            continue
        s = df.get_column(c)
        ratio = s.n_unique() / max(n, 1)
        if ratio <= 0.98:
            continue
        named = any(h == c.lower() or c.lower().endswith(h) for h in _ID_HINTS)
        is_sequence = False
        if dt in _INT_DTYPES:
            d = s.drop_nulls().sort().diff().drop_nulls()
            is_sequence = d.len() > 0 and bool((d == 1).all())
        if named or (dt == pl.Utf8 and ratio == 1.0) or is_sequence:
            out.append(c)
    return out


def _detect_image_paths(df: pl.DataFrame) -> list[str]:
    out = []
    for c, dt in zip(df.columns, df.dtypes):
        if dt != pl.Utf8:
            continue
        s = df.get_column(c).drop_nulls().head(500)
        if s.len() == 0:
            continue
        hits = sum(any(str(v).lower().endswith(ext) for ext in _IMAGE_EXT) for v in s.to_list())
        if hits / s.len() > 0.5:
            out.append(c)
    return out


def _text_features(df: pl.DataFrame, columns: list[str], config: CleanConfig) -> dict[str, Any]:
    """Basic NLP descriptors per free-text column (length, words, TF-IDF top terms)."""
    out: dict[str, Any] = {}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        TfidfVectorizer = None  # type: ignore[assignment]
    for c in columns:
        s = df.get_column(c).drop_nulls()
        if s.len() == 0:
            continue
        mean_chars = df.select(pl.col(c).str.len_chars().mean()).item() or 0.0
        mean_words = df.select(pl.col(c).str.split(" ").list.len().mean()).item() or 0.0
        top_terms: list[str] = []
        if TfidfVectorizer is not None and s.len() >= 5:
            try:
                vec = TfidfVectorizer(max_features=200, stop_words="english")
                tfidf = vec.fit_transform(s.head(5000).to_list())
                scores = tfidf.mean(axis=0).A1
                terms = vec.get_feature_names_out()
                top_terms = [terms[i] for i in scores.argsort()[::-1][:6]]
            except Exception:  # noqa: BLE001
                pass
        out[c] = {
            "mean_chars": round(float(mean_chars), 1),
            "mean_words": round(float(mean_words), 1),
            "top_terms": top_terms,
        }
    return out


def detect_specialization(
    df: pl.DataFrame, config: CleanConfig | None = None, *, target: str | None = None
) -> Specialization:
    """Infer dataset archetype(s) and the analysis plan they imply."""
    config = config or CleanConfig()
    spec = Specialization()
    k = _kinds(df)
    n_rows, n_cols = df.height, df.width
    n_num = len(k["numeric"])

    spec.time_index = _detect_time_index(df, k["datetime"])
    spec.geo = _detect_geo(df, k["numeric"])
    spec.text_columns = _detect_text(df)
    spec.id_columns = _detect_ids(df)
    image_cols = _detect_image_paths(df)

    scores: dict[str, float] = {}

    # Time-series
    if spec.time_index is not None and n_num >= 1:
        scores["time-series"] = 0.9
        spec.recommendations.append(
            f"Time index '{spec.time_index}' detected → time-series handling enabled "
            "(ordered forward-fill, functional analysis of trajectories)."
        )
    # Geospatial
    if spec.geo is not None:
        scores["geospatial"] = 0.85
        spec.recommendations.append(
            f"Geospatial coordinates ('{spec.geo['lat']}', '{spec.geo['lon']}') detected → "
            "spatial clustering/joins recommended (not auto-applied)."
        )
    # High-dimensional / embeddings
    if n_num >= 50:
        spec.embedding_columns = k["numeric"]
        scores["high-dimensional / embeddings"] = min(0.6 + n_num / 500.0, 0.95)
        spec.recommendations.append(
            f"High-dimensional numeric block ({n_num} cols) → PCA summary applied; "
            "consider UMAP/t-SNE and regularised models."
        )
    # Wide (p > n) → omics-like
    if n_cols > max(n_rows * 1.5, n_rows + 20):
        scores["wide / high-dimensional (omics-like)"] = 0.8
        spec.recommendations.append(
            f"Wide dataset (p={n_cols} > n={n_rows}) → regularisation essential and "
            "multiple-testing correction applied to comparisons."
        )
    # Survey / questionnaire (many low-cardinality / Likert columns)
    likert = [
        c for c in k["numeric"]
        if df.get_column(c).n_unique() <= 7 and (df.get_column(c).min() or 0) >= 0
    ]
    if len(likert) >= max(3, n_cols // 2) and n_rows > 30:
        scores["survey / questionnaire"] = 0.7
        spec.recommendations.append(
            f"{len(likert)} Likert-like ordinal columns → treat as ordinal; consider survey "
            "weighting and reliability (Cronbach's α) — not auto-applied."
        )
    # Text-heavy
    if spec.text_columns:
        scores["text-heavy"] = 0.75
        spec.recommendations.append(
            f"Free-text column(s) {spec.text_columns} → basic NLP features extracted; "
            "deep NLP/LLM embeddings not automated."
        )
    # Image references
    if image_cols:
        scores["image references"] = 0.7
        spec.recommendations.append(
            f"Image-path column(s) {image_cols} → flagged; pixel/CV analysis is out of scope."
        )
    if not scores:
        scores["generic tabular"] = 0.6

    spec.archetypes = sorted(scores.items(), key=lambda kv: -kv[1])

    # ---- decide which advanced modules to auto-run -------------------------
    modules: list[str] = []
    if n_num >= 1:
        modules.append("inference")
    if spec.time_index is not None and n_num >= 3:
        modules.append("fda")
    if target is not None or config.target is not None:
        modules.append("model")
    if spec.text_columns:
        modules.append("text")
    if spec.geo is not None:
        modules.append("geospatial")
    spec.auto_modules = modules

    spec.signals = {
        "rows": n_rows,
        "cols": n_cols,
        "numeric": n_num,
        "categorical": len(k["categorical"]),
        "datetime": len(k["datetime"]),
        "time_index": spec.time_index,
        "geo": spec.geo,
        "text_columns": spec.text_columns,
        "id_columns": spec.id_columns,
        "image_columns": image_cols,
        "wide_p_gt_n": n_cols > n_rows,
    }
    if spec.id_columns:
        spec.recommendations.append(
            f"Identifier-like column(s) {spec.id_columns} → excluded from modelling/inference."
        )
    return spec
