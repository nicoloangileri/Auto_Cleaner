"""Smart cleaning engine: standardise → impute → outliers → downcast."""

from __future__ import annotations

from auto_cleaner.clean.dtypes import downcast
from auto_cleaner.clean.impute import impute_missing
from auto_cleaner.clean.outliers import handle_outliers
from auto_cleaner.clean.standardize import standardize

__all__ = ["standardize", "impute_missing", "handle_outliers", "downcast"]
