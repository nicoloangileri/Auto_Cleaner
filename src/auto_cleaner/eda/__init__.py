"""Automated statistical EDA: profile the data, then report on its health."""

from __future__ import annotations

from auto_cleaner.eda.report import build_report, write_reports
from auto_cleaner.eda.stats import ColumnProfile, DatasetProfile, profile_dataset
from auto_cleaner.eda.visualize import Chart, build_charts, charts_to_html, export_pngs

__all__ = [
    "ColumnProfile",
    "DatasetProfile",
    "profile_dataset",
    "build_report",
    "write_reports",
    "Chart",
    "build_charts",
    "charts_to_html",
    "export_pngs",
]
