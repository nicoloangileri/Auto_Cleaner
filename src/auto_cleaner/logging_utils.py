"""Zero-dependency structured logging + timing helpers.

We avoid pulling in ``rich``/``loguru`` to keep the install slim and the latency
predictable. Output goes to *stderr* so it never contaminates piped data.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Iterator

_LEVEL_TAG = {
    "DEBUG": "·",
    "INFO": "•",
    "WARN": "!",
    "ERROR": "✗",
    "OK": "✓",
}


def log(message: str, level: str = "INFO", *, enabled: bool = True) -> None:
    """Emit a single, prefixed log line to stderr.

    Parameters
    ----------
    message:
        Human-readable text.
    level:
        One of ``DEBUG``/``INFO``/``WARN``/``ERROR``/``OK``.
    enabled:
        When ``False`` the call is a no-op (wired to ``CleanConfig.verbose``).
    """
    if not enabled:
        return
    tag = _LEVEL_TAG.get(level, "•")
    print(f"  [{tag}] {message}", file=sys.stderr, flush=True)


@contextmanager
def timed(label: str, *, enabled: bool = True) -> Iterator[None]:
    """Context manager that logs the wall-clock duration of a block.

    Example
    -------
    >>> with timed("ingest"):
    ...     df = read_any(path)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1_000.0
        log(f"{label} finished in {elapsed_ms:,.1f} ms", level="OK", enabled=enabled)


def human_bytes(n: int | float) -> str:
    """Render a byte count as a human-readable string (``1.5 MB``)."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:,.1f} {unit}"
        size /= 1024.0
    return f"{size:,.1f} TB"
