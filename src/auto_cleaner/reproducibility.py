"""Reproducibility manifest + machine-readable results export.

Writes a ``results.json`` capturing both *what was found* (the full analysis tree)
and *how it was produced* (config, seed, library versions, dataset hash,
timestamp) — so a run is auditable and reproducible, and downstream systems can
consume the results without scraping HTML.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from auto_cleaner.config import CleanConfig
from auto_cleaner.eda.stats import DatasetProfile

__all__ = ["library_versions", "dataset_hash", "build_results", "write_results_json"]

_TRACKED = ("polars", "numpy", "scipy", "scikit-learn", "statsmodels", "duckdb", "plotly", "pingouin", "lifelines")


def library_versions() -> dict[str, str]:
    """Best-effort version map of the key analytical libraries."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {"python": platform.python_version()}
    for pkg in _TRACKED:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return out


def dataset_hash(source: str | Path, max_bytes: int = 64 * 1024 * 1024) -> str | None:
    """SHA-256 of the source file (capped) for provenance; ``None`` for non-files."""
    p = Path(str(source))
    if str(source) == ":memory:" or not p.exists() or p.is_dir():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        h.update(fh.read(max_bytes))
    return h.hexdigest()


def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    return obj


def build_results(
    *,
    profile: DatasetProfile,
    advanced: Any,
    config: CleanConfig,
    source: str,
    rows_in: int,
    rows_out: int,
    memory_before: int,
    memory_after: int,
    elapsed_s: float,
) -> dict[str, Any]:
    """Assemble the JSON-serialisable results document."""
    manifest = {
        "tool": "auto_cleaner",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "dataset_sha256": dataset_hash(source),
        "random_seed": config.random_seed,
        "library_versions": library_versions(),
        "config": config.to_dict(),
    }
    summary = {
        "rows_in": rows_in,
        "rows_out": rows_out,
        "memory_before_bytes": memory_before,
        "memory_after_bytes": memory_after,
        "memory_saved_pct": round((memory_before - memory_after) / memory_before * 100, 2) if memory_before else 0.0,
        "quality_score": profile.quality_score,
        "quality_components": profile.quality_components,
        "n_warnings": len(profile.warnings),
        "warnings": profile.warnings,
        "elapsed_s": round(elapsed_s, 3),
    }
    columns = [
        {
            "name": c.name, "dtype": c.dtype, "kind": c.kind, "null_pct": c.null_pct,
            "n_unique": c.n_unique, "mean": c.mean, "median": c.median,
            "skewness": c.skewness, "kurtosis": c.kurtosis,
        }
        for c in profile.columns
    ]
    return {
        "manifest": manifest,
        "summary": summary,
        "columns": columns,
        "advanced": _to_jsonable(advanced) if advanced is not None else None,
    }


def write_results_json(path: str | Path, data: dict[str, Any]) -> str:
    """Write ``data`` as pretty JSON; returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return str(p)
