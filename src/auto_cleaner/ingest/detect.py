"""Source sniffing: format, encoding, delimiter and header detection.

No third-party dependencies — pure stdlib heuristics that are fast and
deterministic. The goal is to make :func:`auto_cleaner.ingest.read_any` able to
open an unknown file without the caller specifying anything.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SourceFormat = Literal[
    "csv", "parquet", "json", "ndjson", "duckdb", "sql", "fits", "netcdf", "excel"
]

_DELIMITER_CANDIDATES: tuple[str, ...] = (",", ";", "\t", "|", ":")
_PARQUET_MAGIC = b"PAR1"
_DUCKDB_MAGIC = b"DUCK"  # appears at byte offset 8 of a DuckDB database file
_FITS_MAGIC = b"SIMPLE  ="  # FITS (astronomy) files open with this 80-char card
_NETCDF_MAGIC = b"CDF"      # classic netCDF; netCDF4 is HDF5-based (matched by suffix)
_ZIP_MAGIC = b"PK\x03\x04"  # xlsx/xlsm are ZIP containers
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # legacy .xls compound document

__all__ = ["FileProfile", "SourceFormat", "detect_format", "profile_source"]


@dataclass(frozen=True, slots=True)
class FileProfile:
    """Everything the readers need to open a delimited/columnar source."""

    path: Path
    fmt: SourceFormat
    encoding: str = "utf-8"
    separator: str | None = None
    has_header: bool = True
    skip_rows: int = 0
    """Junk preamble lines (titles, export banners) before the real header."""


def _read_head_bytes(path: Path, n: int = 16) -> bytes:
    with path.open("rb") as fh:
        return fh.read(n)


def _guess_wide_encoding(chunk: bytes) -> str | None:
    """Detect BOM-less UTF-16/UTF-32 from the NUL-byte pattern of the sample.

    ASCII-heavy text encoded as UTF-16 is ~50% NUL bytes (every other byte);
    as UTF-32 it is ~75% (three of every four). Ordinary 8-bit text virtually
    never contains NULs, so the share alone separates the families and the
    *position* of the NULs gives the byte order.
    """
    n = len(chunk) - (len(chunk) % 4)
    if n < 16:
        return None
    chunk = chunk[:n]
    nul_share = chunk.count(0) / n
    if nul_share < 0.25:
        return None
    if nul_share > 0.6:  # UTF-32 territory
        lead_nuls = chunk[0::4].count(0)
        tail_nuls = chunk[3::4].count(0)
        return "utf-32-le" if lead_nuls < tail_nuls else "utf-32-be"
    even_nuls = chunk[0::2].count(0)
    odd_nuls = chunk[1::2].count(0)
    return "utf-16-le" if odd_nuls > even_nuls else "utf-16-be"


def detect_encoding(path: Path) -> str:
    """Return a best-effort text encoding via BOM inspection + decode probe."""
    head = _read_head_bytes(path, 4)
    # UTF-32 BOMs first: the UTF-32-LE BOM begins with the UTF-16-LE one.
    if head.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return "utf-32"
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    # Probe a larger chunk; fall back to latin-1 which never raises.
    with path.open("rb") as fh:
        chunk = fh.read(65_536)
    wide = _guess_wide_encoding(chunk)
    if wide is not None:
        return wide
    try:
        chunk.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def detect_format(path: Path) -> SourceFormat:
    """Infer the source format from magic bytes first, extension second.

    Binary signatures win over extensions because real-world files are often
    mislabelled (a Parquet file named ``data.csv`` is not hypothetical).
    """
    suffix = path.suffix.lower().lstrip(".")
    head = _read_head_bytes(path, 16)

    if head[:4] == _PARQUET_MAGIC:
        return "parquet"
    if head[:9] == _FITS_MAGIC:
        return "fits"
    if head[:3] == _NETCDF_MAGIC:
        return "netcdf"
    if len(head) >= 12 and head[8:12] == _DUCKDB_MAGIC:
        return "duckdb"
    # ZIP container: an .xlsx even when mislabelled (e.g. exported as .csv).
    if head[:4] == _ZIP_MAGIC or head[:8] == _OLE2_MAGIC:
        return "excel"
    if suffix in {"xlsx", "xlsm", "xls"}:
        return "excel"
    if suffix in {"fits", "fit", "fts"}:
        return "fits"
    if suffix in {"nc", "nc4", "cdf", "netcdf"}:
        return "netcdf"
    if suffix in {"parquet", "pq"}:
        return "parquet"
    if suffix in {"duckdb", "ddb"}:
        return "duckdb"
    if suffix in {"db", "sqlite"}:
        return "duckdb"
    if suffix in {"ndjson", "jsonl"}:
        return "ndjson"
    if suffix == "json":
        # Newline-delimited JSON starts with an object on every line; a JSON
        # document starts with '[' or '{' and is parsed whole.
        stripped = head.lstrip()
        return "json" if stripped[:1] in (b"[", b"{") else "ndjson"
    if suffix in {"sql"}:
        return "sql"
    # Default: treat as delimited text.
    return "csv"


def _delimiter_score(lines: list[str], cand: str) -> float:
    """Consistency score for one candidate: high, stable field counts win.

    Quote-aware (a comma inside ``"a, b"`` is not a delimiter), which keeps the
    score honest on samples that mix delimiters inside quoted fields.
    """
    counts: list[int] = []
    for ln in lines:
        try:
            row = next(csv.reader([ln], delimiter=cand), [])
        except csv.Error:
            row = ln.split(cand)
        counts.append(len(row))
    if not counts or max(counts) < 2:
        return -1.0
    median = statistics.median(counts)
    spread = statistics.pstdev(counts) if len(counts) > 1 else 0.0
    return median - spread


def detect_delimiter(sample: str) -> str:
    """Pick the most likely column delimiter from a text sample.

    Strategy: score every candidate by how *consistently* it splits lines into
    the same number of fields, and accept :class:`csv.Sniffer`'s verdict only
    when it scores at least as well — headers that mix delimiters (e.g.
    ``id;name, surname;score``) routinely fool the sniffer.
    """
    sample = sample.strip("\n")
    if not sample:
        return ","
    lines = [ln for ln in sample.splitlines() if ln][:50]
    scores = {cand: _delimiter_score(lines, cand) for cand in _DELIMITER_CANDIDATES}
    best = max(_DELIMITER_CANDIDATES, key=scores.__getitem__)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(_DELIMITER_CANDIDATES))
        sniffed = dialect.delimiter
        if sniffed in _DELIMITER_CANDIDATES and scores[sniffed] >= scores[best]:
            return sniffed
    except csv.Error:
        pass
    return best if scores[best] > -1.0 else ","


def _numeric_share(row: list[str]) -> float:
    if not row:
        return 0.0
    hits = 0
    for cell in row:
        try:
            float(cell.replace(",", ""))
            hits += 1
        except ValueError:
            pass
    return hits / len(row)


def detect_header(sample: str, delimiter: str) -> bool:
    """Heuristic: does the first row look like names rather than data?

    The numeric-share signal is checked *before* :class:`csv.Sniffer`: an
    entirely non-numeric first row over a partly numeric second row is a header
    even when the Sniffer's type/length voting says otherwise (it is famously
    fooled by short names over uniform-width string columns, e.g. ``a,b``).
    """
    rows = [r for r in csv.reader(sample.splitlines(), delimiter=delimiter) if r]
    if len(rows) >= 2:
        first, second = rows[0], rows[1]
        if _numeric_share(first) == 0.0 and _numeric_share(second) > 0.0:
            return True
    try:
        return csv.Sniffer().has_header(sample)
    except csv.Error:
        pass
    if len(rows) < 2:
        return True
    first, second = rows[0], rows[1]
    # Header likely when the first row is far less numeric than the next row.
    return _numeric_share(first) + 0.25 < _numeric_share(second) or _numeric_share(first) == 0.0


_MAX_PREAMBLE_LINES = 10


def detect_preamble(sample: str, delimiter: str) -> int:
    """Count junk lines (titles, export banners) before the real table.

    Real exports often open with a caption line — e.g. NASA's GISTEMP CSV
    starts with ``Land-Ocean: Global Means`` before the header. Reading that
    line as the header would collapse the table to one misnamed column. A line
    is preamble when its (quote-aware) field count differs from the *modal*
    field count of the sample; the table starts at the first modal-width line.
    """
    lines = [ln for ln in sample.splitlines() if ln.strip()][: 200]
    if len(lines) < 3:
        return 0
    counts = []
    for ln in lines:
        try:
            counts.append(len(next(csv.reader([ln], delimiter=delimiter), [])))
        except csv.Error:
            counts.append(len(ln.split(delimiter)))
    modal = statistics.mode(counts)
    if modal < 2:
        return 0
    for i, c in enumerate(counts[:_MAX_PREAMBLE_LINES + 1]):
        if c == modal:
            return i
    return 0


def profile_source(path: str | Path, sample_rows: int = 4096) -> FileProfile:
    """Produce a :class:`FileProfile` describing how to open ``path``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input source does not exist: {p}")

    fmt = detect_format(p)
    if fmt in {"parquet", "duckdb", "json", "ndjson", "sql", "fits", "netcdf", "excel"}:
        return FileProfile(path=p, fmt=fmt)

    encoding = detect_encoding(p)
    with p.open("r", encoding=encoding, errors="replace", newline="") as fh:
        sample = "".join(next(fh, "") for _ in range(min(sample_rows, 200)))
    delimiter = detect_delimiter(sample)
    skip_rows = detect_preamble(sample, delimiter)
    if skip_rows:
        sample = "\n".join(sample.splitlines()[skip_rows:])
    has_header = detect_header(sample, delimiter)
    return FileProfile(
        path=p,
        fmt="csv",
        encoding=encoding,
        separator=delimiter,
        has_header=has_header,
        skip_rows=skip_rows,
    )
