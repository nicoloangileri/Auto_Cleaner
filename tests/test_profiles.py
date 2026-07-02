"""Execution profiles: --profile fast/standard/full must actually gate work."""

import pytest

from auto_cleaner import CleanConfig
from auto_cleaner.__main__ import _build_parser, _config_from_args


def test_preset_fast_disables_heavy_stages():
    cfg = CleanConfig.preset("fast")
    assert not cfg.advanced and not cfg.extended_stats and not cfg.forecast
    assert not cfg.make_charts and not cfg.make_pdf
    assert cfg.downcast  # cleaning itself stays on


def test_preset_standard_keeps_advanced_but_trims_extended():
    cfg = CleanConfig.preset("standard")
    assert cfg.advanced and cfg.inference and cfg.modeling
    assert not cfg.extended_stats and not cfg.forecast and not cfg.fda


def test_preset_full_is_the_default_config():
    assert CleanConfig.preset("full") == CleanConfig()


def test_preset_unknown_raises():
    with pytest.raises(ValueError, match="Unknown profile"):
        CleanConfig.preset("turbo")


def test_cli_profile_fast_is_not_reenabled_by_flag_defaults():
    args = _build_parser().parse_args(["-i", "x.csv", "--profile", "fast"])
    cfg = _config_from_args(args)
    # Without --no-charts on the command line, the fast profile's make_charts=False
    # must survive the CLI override merge.
    assert not cfg.make_charts and not cfg.make_pdf and not cfg.advanced


def test_cli_no_flags_still_disable_on_top_of_profile():
    args = _build_parser().parse_args(
        ["-i", "x.csv", "--profile", "standard", "--no-modeling"]
    )
    cfg = _config_from_args(args)
    assert cfg.advanced and not cfg.modeling
