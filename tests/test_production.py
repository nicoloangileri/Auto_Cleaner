"""Tests for model persistence / predict and YAML data contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from auto_cleaner import CleanConfig
from auto_cleaner.persistence import load_bundle, predict_frame, save_bundle, train_exportable
from auto_cleaner.validate import ValidationError, enforce_contract


def _frame(n: int = 120) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, n)
    y = x * 2.0 + rng.normal(0.0, 0.3, n)
    cat = np.where(x > 0, "A", "B")
    return pl.DataFrame({"x": x, "cat": list(cat), "y": y})


# --- Persistence / predict -------------------------------------------------- #
def test_train_save_predict_roundtrip(tmp_path: Path):
    df = _frame()
    bundle = train_exportable(df, "y", CleanConfig())
    assert bundle is not None and bundle["task"] == "regression"
    path = save_bundle(bundle, tmp_path / "m.joblib")
    preds = predict_frame(load_bundle(path), df)
    assert preds.len() == df.height
    r = np.corrcoef(preds.to_numpy().astype(float), df["y"].to_numpy())[0, 1]
    assert r > 0.8  # predictions track the target


def test_classification_bundle(tmp_path: Path):
    df = _frame().with_columns((pl.col("x") > 0).cast(pl.Int8).alias("label"))
    bundle = train_exportable(df, "label", CleanConfig())
    assert bundle is not None and bundle["task"] == "classification"
    preds = predict_frame(bundle, df)
    assert preds.n_unique() <= 2


# --- Data contracts --------------------------------------------------------- #
def test_contract_pass_is_clean():
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "g": ["x", "y", "x"]})
    contract = {"columns": {"a": {"dtype": "Float64", "min": 0}, "g": {"allowed": ["x", "y"]}}}
    assert enforce_contract(df, contract, CleanConfig()).issues == []


def test_contract_flags_violations():
    df = pl.DataFrame({"a": [-1.0, 2.0, 3.0], "g": ["x", "z", "x"]})
    contract = {"columns": {"a": {"min": 0}, "g": {"allowed": ["x", "y"]}}}
    issues = enforce_contract(df, contract, CleanConfig()).issues
    assert any("min" in i for i in issues)
    assert any("unexpected" in i for i in issues)


def test_contract_required_missing_raises():
    df = pl.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValidationError):
        enforce_contract(df, {"columns": {"b": {"required": True}}}, CleanConfig())
