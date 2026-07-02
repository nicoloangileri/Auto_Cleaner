"""Dynamic ingestion: detect, then read — any messy source into polars."""

from __future__ import annotations

from auto_cleaner.ingest.detect import FileProfile, detect_format, profile_source
from auto_cleaner.ingest.readers import read_any, read_sql

__all__ = [
    "FileProfile",
    "detect_format",
    "profile_source",
    "read_any",
    "read_sql",
]
