#!/usr/bin/env bash
# One-command setup: install everything, then run the demos end-to-end.
set -euo pipefail

echo "==> Installing dependencies + package (editable)"
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo "==> Generating the real demo dataset (auto-mpg)"
python examples/generate_raw.py

echo "==> Running the full pipeline (clean + analyse + report)"
python -m auto_cleaner -i examples/data/raw_cars.csv \
  -o examples/output/clean_cars.parquet --target Miles_per_Gallon

echo "==> Building the capabilities PDF (general + proof-of-work)"
python examples/build_capabilities_pdf.py

echo ""
echo "Done. Outputs are in examples/output/:"
echo "  - clean_cars.parquet           (clean dataset)"
echo "  - clean_cars_eda.html          (interactive report)"
echo "  - clean_cars_report.pdf        (per-dataset PDF)"
echo "  - clean_cars_results.json      (machine-readable results + manifest)"
echo "  - charts/*.png                 (standalone charts)"
echo "  - auto_cleaner_capabilities.pdf"
