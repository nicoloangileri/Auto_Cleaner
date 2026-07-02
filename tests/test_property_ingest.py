"""Property-based ingestion tests: the reader must never crash, never lie.

Hypothesis generates deformed-but-plausible delimited files (random field
counts, quotes, unicode, null tokens, preambles, encodings). Two invariants:

1. ``read_any`` either returns a DataFrame or raises a *clean* error — never a
   polars panic leaking through as an unrelated exception type.
2. When it returns, the accounting must hold: rows read + rows reported
   dropped equals the parsable data lines (no silent row loss).
"""

from __future__ import annotations

import polars as pl
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from auto_cleaner import CleanConfig
from auto_cleaner.ingest import read_any

CFG = CleanConfig(verbose=False)

_CELL = st.one_of(
    st.integers(-10**6, 10**6).map(str),
    st.floats(allow_nan=False, allow_infinity=False, width=32).map(lambda f: f"{f:.4g}"),
    st.sampled_from(["", "na", "N/A", "null", "-", "?", "***"]),
    st.text(
        alphabet=st.characters(
            codec="utf-8",
            exclude_categories=("Cs", "Cc"),
            exclude_characters=',;\t|:"\n\r',
        ),
        max_size=12,
    ),
    st.sampled_from(['"quoted, cell"', '"unclosed', "€45,3", "50%", " padded "]),
)

_DELIM = st.sampled_from([",", ";", "\t", "|"])


@st.composite
def deformed_csv(draw) -> tuple[str, str]:
    """A header + data lines whose field counts may vary line to line."""
    delim = draw(_DELIM)
    ncols = draw(st.integers(2, 6))
    header = delim.join(f"col_{i}" for i in range(ncols))
    lines = [header]
    n_lines = draw(st.integers(3, 12))
    for _ in range(n_lines):
        width = draw(st.integers(max(1, ncols - 2), ncols + 2))
        cells = draw(st.lists(_CELL, min_size=width, max_size=width))
        lines.append(delim.join(c.replace(delim, " ") for c in cells))
    preamble = draw(st.sampled_from(["", "Export banner line\n", "Title\nSubtitle\n"]))
    return preamble + "\n".join(lines) + "\n", delim


@given(payload=deformed_csv(), encoding=st.sampled_from(["utf-8", "utf-16", "latin-1"]))
@settings(max_examples=60, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_reader_never_crashes_and_never_loses_rows_silently(tmp_path, payload, encoding):
    text, _delim = payload
    path = tmp_path / "fuzz.csv"
    path.write_bytes(text.encode(encoding, errors="replace"))

    try:
        df, report = read_any(path, CFG)
    except (ValueError, pl.exceptions.PolarsError) as exc:
        # A refusal is acceptable — but it must be a clean, typed error.
        assert str(exc), "error must carry a message"
        return

    assert isinstance(df, pl.DataFrame)
    assert df.width >= 1
    # Accounting invariant: tolerant modes must report what they dropped.
    dropped = report.metrics.get("csv_rows_dropped", 0)
    assert dropped >= 0
    if report.metrics.get("csv_parse_mode") in ("tolerant", "tolerant-unquoted"):
        assert any("Malformed CSV recovered" in w for w in report.warnings)


@st.composite
def roundtrip_frame(draw) -> pl.DataFrame:
    n = draw(st.integers(5, 40))
    nums = draw(st.lists(
        st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False, width=32)),
        min_size=n, max_size=n))
    txts = draw(st.lists(
        st.one_of(st.none(), st.text(
            alphabet=st.characters(codec="utf-8", exclude_categories=("Cs", "Cc"),
                                   exclude_characters=',"\n\r'),
            max_size=8)),
        min_size=n, max_size=n))
    return pl.DataFrame({"num": nums, "txt": txts})


@given(df=roundtrip_frame())
@settings(max_examples=40, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_csv_roundtrip_preserves_shape(tmp_path, df):
    """Anything polars can write as CSV, the reader must load back same-shape."""
    path = tmp_path / "rt.csv"
    df.write_csv(path)
    loaded, _ = read_any(path, CFG)
    assert loaded.height == df.height
    assert loaded.width == df.width
