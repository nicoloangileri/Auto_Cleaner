"""Materialise a *real* messy dataset for the demo.

Pulls the classic **auto-mpg ("cars")** dataset — bundled locally in
``vega_datasets`` so no network is required — and writes it as a raw,
semicolon-delimited CSV. This is genuine real-world data: it ships with missing
``Miles_per_Gallon`` / ``Horsepower`` values, a date column, a categorical
``Origin``, and strongly collinear engine measurements — perfect for exercising
the whole pipeline.

Run::

    python examples/generate_raw.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

OUT = Path(__file__).parent / "data" / "raw_cars.csv"


def main() -> None:
    try:
        from vega_datasets import data  # optional: bundled auto-mpg dataset
    except ImportError:
        # vega_datasets is an example-only convenience. A copy of the raw CSV
        # already ships in the repo, so if the package is absent we keep the
        # committed file rather than failing (this is what CI relies on).
        if OUT.exists():
            print(f"vega_datasets not installed — using committed dataset at {OUT}")
            return
        raise SystemExit(
            "vega_datasets not installed and no committed dataset found; "
            "install it with `pip install vega-datasets` to regenerate."
        )

    df = pl.from_pandas(data.cars())
    # Keep the date as a clean ISO *date* string in the raw export (no time part).
    df = df.with_columns(pl.col("Year").dt.date())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Semicolon separator → exercises auto delimiter detection on ingest.
    df.write_csv(OUT, separator=";")
    print(f"Wrote {df.height} rows × {df.width} cols → {OUT}")
    print("Null counts:", df.null_count().to_dicts()[0])


if __name__ == "__main__":
    main()
