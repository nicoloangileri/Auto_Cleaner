"""Hostile-input ingestion tests: real-world malformed CSVs must be recovered,
counted and surfaced — never crash the pipeline, never drop rows silently.

Run with:  pytest -q
"""

from __future__ import annotations

import pytest

from auto_cleaner import CleanConfig
from auto_cleaner.eda import build_report, profile_dataset
from auto_cleaner.ingest import read_any
from auto_cleaner.ingest.detect import detect_delimiter, detect_encoding, profile_source

CFG = CleanConfig().with_overrides(verbose=False)


# --------------------------------------------------------------------------- #
# Ragged / malformed structure
# --------------------------------------------------------------------------- #
def test_ragged_rows_recovered_and_counted(tmp_path):
    p = tmp_path / "ragged.csv"
    p.write_text(
        "id,name,score\n"
        "1,alpha,10\n"
        "2,beta,20,EXTRA,MORE\n"  # too many fields → truncated
        "3,gamma,30\n"
        "4,delta\n"  # too few fields → null-padded
        "5,epsilon,50\n"
    )
    df, rep = read_any(p, CFG)
    assert (df.height, df.width) == (5, 3)  # every row survives
    assert df["name"].to_list() == ["alpha", "beta", "gamma", "delta", "epsilon"]
    assert df["score"][3] is None  # short row padded, not dropped
    assert rep.metrics["csv_parse_mode"] == "tolerant"
    assert rep.metrics["csv_malformed_lines"] == 2
    assert rep.metrics["csv_rows_dropped"] == 0
    assert any("tolerant parsing" in w for w in rep.warnings)


def test_unclosed_quote_recovered(tmp_path):
    p = tmp_path / "unclosed.csv"
    p.write_text('id,comment\n1,"fine"\n2,"broken quote here\n3,ok\n')
    df, rep = read_any(p, CFG)
    assert (df.height, df.width) == (3, 2)  # the bad quote must not swallow rows
    assert df["comment"].to_list() == ['"fine"', '"broken quote here', "ok"]
    assert rep.metrics["csv_parse_mode"] == "tolerant-unquoted"
    assert any("quoting was disabled" in w for w in rep.warnings)


def test_mixed_delimiters_in_header(tmp_path):
    # Header contains both ';' and ',': the consistent splitter (';') must win.
    p = tmp_path / "mixed.csv"
    p.write_text("id;name, surname;score\n1;anna, verdi;10\n2;luca, bianchi;20\n")
    prof = profile_source(p)
    assert prof.separator == ";"
    df, _ = read_any(p, CFG)
    assert (df.height, df.width) == (2, 3)
    assert df["name, surname"].to_list() == ["anna, verdi", "luca, bianchi"]


def test_clean_csv_stays_strict(tmp_path):
    p = tmp_path / "clean.csv"
    p.write_text("alpha,beta\n1,2\n3,4\n")
    df, rep = read_any(p, CFG)
    assert (df.height, df.width) == (2, 2)
    assert rep.metrics["csv_parse_mode"] == "strict"
    assert rep.warnings == []


# --------------------------------------------------------------------------- #
# Encodings: UTF-16 / UTF-32, with and without BOM
# --------------------------------------------------------------------------- #
def test_utf16_le_bom(tmp_path):
    p = tmp_path / "utf16.csv"
    p.write_bytes("città,valore\nRoma,1\nMünchen,2\n".encode("utf-16"))  # LE + BOM
    assert detect_encoding(p) == "utf-16"
    df, _ = read_any(p, CFG)
    assert (df.height, df.width) == (2, 2)
    assert df.columns == ["città", "valore"]
    assert df["città"].to_list() == ["Roma", "München"]


def test_utf16_be_bom(tmp_path):
    p = tmp_path / "utf16be.csv"
    p.write_bytes(b"\xfe\xff" + "a,b\n1,2\n3,4\n".encode("utf-16-be"))
    assert detect_encoding(p) == "utf-16"
    df, _ = read_any(p, CFG)
    assert df.columns == ["a", "b"]
    assert df["a"].to_list() == [1, 3]


def test_utf16_le_without_bom(tmp_path):
    p = tmp_path / "utf16_nobom.csv"
    p.write_bytes("città,valore\nRoma,10\nMünchen,20\n".encode("utf-16-le"))
    assert detect_encoding(p) == "utf-16-le"
    df, _ = read_any(p, CFG)
    assert df.columns == ["città", "valore"]  # no stray NUL bytes in names
    assert df["città"].to_list() == ["Roma", "München"]
    assert df["valore"].to_list() == [10, 20]


def test_utf32_le_bom(tmp_path):
    p = tmp_path / "utf32.csv"
    p.write_bytes("a,b\n1,2\n3,4\n".encode("utf-32"))  # LE + BOM
    assert detect_encoding(p) == "utf-32"
    df, _ = read_any(p, CFG)
    assert df.columns == ["a", "b"]
    assert df["b"].to_list() == [2, 4]


def test_utf8_still_detected(tmp_path):
    p = tmp_path / "plain.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    assert detect_encoding(p) == "utf-8"


def test_detect_delimiter_unchanged_on_clean_samples():
    # Pinned behaviour: the quote-aware scorer must not regress simple cases.
    assert detect_delimiter("a;b;c\n1;2;3\n4;5;6") == ";"
    assert detect_delimiter("a,b,c\n1,2,3\n4,5,6") == ","
    assert detect_delimiter('name,desc\n"x","a, b, c"\n"y","d, e"') == ","


# --------------------------------------------------------------------------- #
# Recovery warnings must reach the EDA report, not just the StepReport
# --------------------------------------------------------------------------- #
def test_ingest_warning_visible_in_eda_report(tmp_path):
    p = tmp_path / "ragged.csv"
    p.write_text("a,b\n1,2\n3,4,SPURIOUS\n5,6\n")
    df, rep = read_any(p, CFG)
    profile = profile_dataset(df, CFG)
    md, html = build_report(profile, [rep], config=CFG)

    md_health = md.split("## 2. Data-Health Warnings")[1].split("## 3.")[0]
    assert "[ingest]" in md_health and "tolerant parsing" in md_health

    html_health = html.split("2 · Data-Health Warnings")[1].split("<h2>")[0]
    assert "[ingest]" in html_health and "tolerant parsing" in html_health
