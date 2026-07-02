"""Run auto_cleaner over every real-world benchmark dataset and harvest metrics.

Uses the registry in :mod:`fetch_real_datasets` (download first if needed) and
writes ``examples/output/real_benchmarks.json`` — the file the paper builder
cites, so every number in the PDF traces back to an actual run.

Run:  python examples/run_real_benchmarks.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from fetch_real_datasets import DATASETS, RealDataset, fetch, path_for  # noqa: E402

from auto_cleaner import CleanConfig, run_pipeline  # noqa: E402

OUT_JSON = HERE / "output" / "real_benchmarks.json"


def _config_for(ds: RealDataset) -> CleanConfig:
    profile = "fast" if ds.scale_benchmark else "standard"
    cfg = CleanConfig.preset(profile).with_overrides(
        verbose=False, make_charts=False, make_pdf=False, make_json=False,
        save_model=False,
    )
    if ds.target:
        cfg = cfg.with_overrides(target=ds.target)
    if ds.scale_benchmark:
        cfg = cfg.with_overrides(streaming=True)
    return cfg


def _bench_one(ds: RealDataset) -> dict:
    src = path_for(ds)
    t0 = time.perf_counter()
    result = run_pipeline(src, None, _config_for(ds), write_reports_to_disk=False)
    elapsed = time.perf_counter() - t0

    ingest = next(r for r in result.step_reports if r.step == "ingest")
    impacts = [
        imp for rep in result.step_reports for imp in rep.metrics.get("impact", [])
    ]
    worst = max(impacts, key=lambda i: i["ks_stat"] or 0.0) if impacts else None
    saved = (
        (result.memory_before - result.memory_after) / result.memory_before * 100.0
        if result.memory_before else 0.0
    )
    return {
        "key": ds.key,
        "domain": ds.domain,
        "citation": ds.citation,
        "rows": result.rows_in,
        "cols": result.profile.n_cols,
        "quality": result.profile.quality_score,
        "warnings": len(result.profile.warnings),
        "elapsed_s": round(elapsed, 2),
        "memory_saved_pct": round(saved, 1),
        "parse_mode": ingest.metrics.get("csv_parse_mode"),
        "preamble_lines": ingest.metrics.get("csv_preamble_lines", 0),
        "excel_sheet": ingest.metrics.get("excel_sheet"),
        "source_format": ingest.metrics.get("source_format"),
        "worst_impact": (
            {
                "column": worst["column"],
                "ks": worst["ks_stat"],
                "share": worst["change_share"],
                "verdict": worst["verdict"],
            }
            if worst else None
        ),
        "profile_used": "fast" if ds.scale_benchmark else "standard",
    }


def main() -> None:
    fetch()
    rows: list[dict] = []
    for ds in DATASETS:
        src = path_for(ds)
        if not src.exists():
            print(f"[!] {ds.key}: missing (fetch failed?) — skipped", file=sys.stderr)
            continue
        print(f"[•] {ds.key:<15} ({ds.domain}) ...", flush=True)
        try:
            rows.append(_bench_one(ds))
            r = rows[-1]
            print(f"    {r['rows']:,} rows x {r['cols']} cols  "
                  f"quality {r['quality']}/100  {r['elapsed_s']}s")
        except Exception as exc:  # noqa: BLE001 — one bad dataset must not kill the sweep
            print(f"    FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            rows.append({"key": ds.key, "domain": ds.domain,
                         "citation": ds.citation, "error": f"{type(exc).__name__}: {exc}"})

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    ok = sum(1 for r in rows if "error" not in r)
    print(f"\n{ok}/{len(rows)} datasets cleaned successfully → {OUT_JSON}")


if __name__ == "__main__":
    main()
