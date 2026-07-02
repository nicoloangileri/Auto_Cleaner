"""Measure — don't estimate — the engine's single-machine scale envelope.

Concatenates k copies of the real NYC-taxi month (2.96M rows x 20 cols) and
runs the fast profile with streaming ingestion, recording wall time, rows/s
and peak RSS. Results land in ``examples/output/scale_probe.json`` and are
cited by the paper builder.

Run:  python examples/scale_probe.py [k1 k2 ...]   (default: 1 2 4)
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import polars as pl

HERE = Path(__file__).parent
TAXI = HERE / "data" / "real" / "yellow_tripdata_2024-01.parquet"
OUT = HERE / "output" / "scale_probe.json"

sys.path.insert(0, str(HERE))


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3  # macOS: bytes


def probe(k: int) -> dict:
    from auto_cleaner import CleanConfig, run_pipeline

    base = pl.read_parquet(TAXI)
    df = pl.concat([base] * k, rechunk=True)
    src = HERE / "output" / f"_scale_{k}x.parquet"
    df.write_parquet(src)
    rows = df.height
    del base, df

    cfg = CleanConfig.preset("fast").with_overrides(
        verbose=False, streaming=True, make_json=False, save_model=False,
    )
    t0 = time.perf_counter()
    run_pipeline(src, None, cfg, write_reports_to_disk=False)
    elapsed = time.perf_counter() - t0
    src.unlink(missing_ok=True)
    return {
        "copies": k,
        "rows": rows,
        "elapsed_s": round(elapsed, 2),
        "rows_per_s": int(rows / elapsed),
        "peak_rss_gb": round(_peak_rss_gb(), 2),
    }


def main() -> None:
    if not TAXI.exists():
        print("taxi parquet missing — run examples/fetch_real_datasets.py first",
              file=sys.stderr)
        raise SystemExit(1)
    ks = [int(a) for a in sys.argv[1:]] or [1, 2, 4]
    results = []
    for k in ks:
        print(f"[•] {k}x taxi ...", flush=True)
        try:
            r = probe(k)
        except MemoryError:
            r = {"copies": k, "error": "MemoryError — beyond the single-machine envelope"}
        results.append(r)
        print(f"    {r}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
