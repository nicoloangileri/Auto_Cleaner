"""Fetch the real-world benchmark datasets into ``examples/data/real/``.

Every dataset is public, fetched from its primary source with no
authentication, and listed in :data:`DATASETS` together with its citation —
the same registry the paper builder uses, so the PDF's dataset table and the
downloads can never drift apart.

Run:  python examples/fetch_real_datasets.py
"""

from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REAL_DIR = Path(__file__).parent / "data" / "real"


@dataclass(frozen=True)
class RealDataset:
    key: str
    domain: str
    filename: str
    url: str
    citation: str
    target: str | None = None      # optional supervised target for the benchmark
    scale_benchmark: bool = False  # large file: fast profile + timing focus


DATASETS: tuple[RealDataset, ...] = (
    RealDataset(
        key="credit_default", domain="Finance (credit risk)",
        filename="credit_default.xls",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
        citation="I-Cheng Yeh, C. Lien. Default of Credit Card Clients. UCI Machine Learning Repository, 2016.",
    ),
    RealDataset(
        key="eurusd", domain="Finance (FX markets)",
        filename="eurusd_daily.csv",
        url="https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?format=csvdata",
        citation="European Central Bank. US dollar/Euro reference exchange rate, daily (EXR.D.USD.EUR.SP00.A). ECB Data Portal, retrieved 2026.",
    ),
    RealDataset(
        key="auto_mpg", domain="Automotive",
        filename="../raw_cars.csv",  # already shipped with the repo
        url="",
        citation="R. Quinlan. Auto MPG. UCI Machine Learning Repository, 1993 (via vega-datasets).",
        target="Miles_per_Gallon",
    ),
    RealDataset(
        key="ai_models", domain="Artificial intelligence",
        filename="notable_ai_models.csv",
        url="https://epoch.ai/data/notable_ai_models.csv",
        citation="Epoch AI. Notable AI Models database. epoch.ai/data, retrieved 2026.",
    ),
    RealDataset(
        key="gistemp", domain="Climate",
        filename="gistemp_global.csv",
        url="https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
        citation="GISTEMP Team. GISS Surface Temperature Analysis v4. NASA GISS, retrieved 2026.",
    ),
    RealDataset(
        key="adult", domain="Social / census",
        filename="adult_census.csv",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        citation="B. Becker, R. Kohavi. Adult (Census Income). UCI Machine Learning Repository, 1996.",
    ),
    RealDataset(
        key="heart", domain="Medicine (cardiology)",
        filename="heart_cleveland.csv",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        citation="A. Janosi et al. Heart Disease (Cleveland). UCI Machine Learning Repository, 1988.",
    ),
    RealDataset(
        key="penguins", domain="Biology (field data)",
        filename="penguins.csv",
        url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv",
        citation="K. Gorman, A. Horst, A. Hill. Palmer Archipelago Penguins. PLoS ONE 9(3), 2014.",
        target="species",
    ),
    RealDataset(
        key="wine", domain="Food chemistry",
        filename="winequality_red.csv",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        citation="P. Cortez et al. Wine Quality. UCI Machine Learning Repository / Decision Support Systems 47(4), 2009.",
        target="quality",
    ),
    RealDataset(
        key="nyc_taxi", domain="Transport (urban mobility)",
        filename="yellow_tripdata_2024-01.parquet",
        url="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet",
        citation="NYC Taxi & Limousine Commission. Yellow Taxi Trip Records, January 2024. nyc.gov/tlc.",
        scale_benchmark=True,
    ),
)


def path_for(ds: RealDataset) -> Path:
    return (REAL_DIR / ds.filename).resolve()


def fetch(force: bool = False) -> list[RealDataset]:
    """Download every remote dataset that is not already on disk."""
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    got: list[RealDataset] = []
    for ds in DATASETS:
        dest = path_for(ds)
        if not ds.url:  # shipped with the repository
            got.append(ds)
            continue
        if dest.exists() and not force:
            print(f"  [=] {ds.key:<15} already present ({dest.stat().st_size:,} B)")
            got.append(ds)
            continue
        print(f"  [↓] {ds.key:<15} {ds.url[:70]}")
        req = urllib.request.Request(ds.url, headers={"User-Agent": "auto-cleaner-benchmarks/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as fh:
                fh.write(resp.read())
            got.append(ds)
        except Exception as exc:  # noqa: BLE001 — a dead mirror must not kill the run
            print(f"  [!] {ds.key}: download failed ({exc}); skipping", file=sys.stderr)
    return got


if __name__ == "__main__":
    fetched = fetch(force="--force" in sys.argv)
    print(f"{len(fetched)}/{len(DATASETS)} datasets available under {REAL_DIR}")
