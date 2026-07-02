"""PNG-export failure handling: aggregate warning + honest report messages.

kaleido 1.x no longer bundles Chromium, so on a fresh machine every
``write_image`` call fails. These tests simulate that failure (the real one
can't be reproduced here — Chrome is installed) and check that:

* ``export_pngs`` emits a clearly visible aggregate warning (not verbose-only)
  with the real cause, plus a one-time Chrome setup hint;
* the Markdown report says "PNG export failed", not "install plotly", when
  plotly is present.
"""

from __future__ import annotations

import polars as pl

import auto_cleaner.eda.report as report_mod
import auto_cleaner.eda.visualize as viz
from auto_cleaner.config import CleanConfig
from auto_cleaner.eda.report import render_markdown
from auto_cleaner.eda.stats import profile_dataset
from auto_cleaner.eda.visualize import Chart, export_pngs


class _BrokenFig:
    """Stands in for a Plotly figure on a machine without headless Chrome."""

    def write_image(self, *args, **kwargs):
        raise RuntimeError("Kaleido requires Google Chrome to be installed")


def _broken_charts(n: int = 2) -> list[Chart]:
    return [Chart(f"c{i}", f"Chart {i}", _BrokenFig()) for i in range(n)]


def test_all_failures_emit_visible_aggregate_warning(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(viz, "_CHROME_HINT_EMITTED", False)
    out = export_pngs(_broken_charts(), tmp_path / "charts", CleanConfig(verbose=False))
    err = capsys.readouterr().err
    assert out == {}
    assert list((tmp_path / "charts").iterdir()) == []
    # visible even with verbose=False, and carries the real cause
    assert "PNG export failed for all 2 chart(s)" in err
    assert "Chrome" in err


def test_chrome_hint_printed_once_per_process(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(viz, "_CHROME_HINT_EMITTED", False)
    export_pngs(_broken_charts(), tmp_path / "a", CleanConfig(verbose=False))
    export_pngs(_broken_charts(), tmp_path / "b", CleanConfig(verbose=False))
    err = capsys.readouterr().err
    assert err.count("plotly_get_chrome") == 1


def test_partial_failure_does_not_trigger_aggregate_warning(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(viz, "_CHROME_HINT_EMITTED", False)

    class _OkFig:
        def write_image(self, path, *args, **kwargs):
            from pathlib import Path

            Path(path).write_bytes(b"png")

    charts = [Chart("good", "Good", _OkFig()), *_broken_charts(1)]
    out = export_pngs(charts, tmp_path / "charts", CleanConfig(verbose=False))
    err = capsys.readouterr().err
    assert set(out) == {"good"}
    assert "PNG export failed for all" not in err


def test_markdown_distinguishes_failed_export_from_missing_plotly():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 4.0]})
    config = CleanConfig()
    profile = profile_dataset(df, config)
    md = render_markdown(
        profile, [], title="t", source_name="s", config=config, chart_pngs=[]
    )
    # plotly IS installed in the test env: never blame the dependency
    assert "install plotly" not in md
    assert "PNG export failed" in md
    assert "plotly_get_chrome" in md


def test_markdown_message_when_plotly_missing(monkeypatch):
    monkeypatch.setattr(report_mod, "_plotly_available", lambda: False)
    df = pl.DataFrame({"a": [1, 2, 3]})
    config = CleanConfig()
    profile = profile_dataset(df, config)
    md = render_markdown(
        profile, [], title="t", source_name="s", config=config, chart_pngs=None
    )
    assert "plotly is not installed" in md
    assert "PNG export failed" not in md
