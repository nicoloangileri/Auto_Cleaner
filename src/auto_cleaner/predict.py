"""CLI: score new data with a saved auto_cleaner model bundle.

    python -m auto_cleaner.predict --model clean_model.joblib \\
        --input new_rows.csv --output scored.parquet

The output is the input data plus a ``<target>_prediction`` column.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import polars as pl

from auto_cleaner.config import CleanConfig
from auto_cleaner.ingest import read_any
from auto_cleaner.persistence import load_bundle, predict_frame


def main(argv: Sequence[str] | None = None) -> int:
    """Load a model bundle, score the input, and write predictions."""
    parser = argparse.ArgumentParser(
        prog="auto_cleaner.predict",
        description="Score new data with a saved auto_cleaner model bundle.",
    )
    parser.add_argument("-m", "--model", required=True, help="Path to the .joblib model bundle.")
    parser.add_argument("-i", "--input", required=True, help="New data to score (CSV/Parquet/JSON/...).")
    parser.add_argument("-o", "--output", required=True, help="Where to write the scored data.")
    args = parser.parse_args(argv)

    try:
        bundle = load_bundle(args.model)
        df, _ = read_any(args.input, CleanConfig().with_overrides(verbose=False))
        predictions = predict_frame(bundle, df)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    out = df.with_columns(predictions)
    path = Path(args.output)
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix in {".parquet", ".pq"}:
        out.write_parquet(path)
    elif suffix in {".json", ".ndjson"}:
        out.write_ndjson(path)
    else:
        out.write_csv(path)
    print(f"Scored {out.height:,} rows → {path}  (target: {bundle['target']}, task: {bundle['task']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
