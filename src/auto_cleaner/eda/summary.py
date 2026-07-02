"""Executive summary: the report's findings in prose, ranked by importance.

A 280-line report is an archive, not a briefing. This module distils the
profile, the cleaning impact and the advanced diagnostics into a handful of
plain-English bullets — worst news first — plus a short review checklist of
the decisions a human must sign off before trusting the data. It only states
what was actually measured; when a section did not run, it says nothing.
"""

from __future__ import annotations

from typing import Any

from auto_cleaner.reporting import StepReport

__all__ = ["build_summary"]

_MAX_HEADLINES = 7


def _impact_bullets(step_reports: list[StepReport]) -> tuple[list[str], list[str]]:
    """(headlines, checklist) drawn from the measured cleaning impact."""
    heads: list[str] = []
    checks: list[str] = []
    material = []
    total_changed = 0
    for rep in step_reports:
        for imp in rep.metrics.get("impact", []):
            total_changed += imp["cells_changed"]
            if imp["verdict"] == "material":
                material.append((rep.step, imp))
    if material:
        for step, imp in material:
            heads.append(
                f"🔴 **{step} materially changed `{imp['column']}`** — "
                f"{imp['cells_changed']} cell(s) ({imp['change_share']:.1%}), "
                f"mean shift {imp['mean_shift_sd']:.2f} sd"
                + (f", KS {imp['ks_stat']:.2f}" if imp["ks_stat"] is not None else "")
                + ". The cleaned column is *not* distribution-faithful."
            )
            checks.append(
                f"Decide whether the {step} treatment of `{imp['column']}` is "
                "acceptable, or re-run with a different strategy."
            )
    elif total_changed:
        heads.append(
            f"🟢 Cleaning changed {total_changed} cell(s) overall with no material "
            "distribution shift (all columns negligible/minor — see Cleaning Impact)."
        )
    return heads, checks


def _outlier_bullets(step_reports: list[StepReport]) -> tuple[list[str], list[str]]:
    heads: list[str] = []
    checks: list[str] = []
    for rep in step_reports:
        if rep.step != "outliers":
            continue
        flagged = rep.metrics.get("rows_flagged_total")
        dropped = rep.metrics.get("rows_dropped")
        if dropped:
            heads.append(f"🟠 {dropped} row(s) were **dropped** as outliers.")
            checks.append("Confirm the dropped outlier rows were noise, not signal.")
        elif flagged:
            heads.append(
                f"🟡 {flagged} row(s) flagged as outliers in `is_outlier` "
                "(kept in the data)."
            )
            checks.append("Skim the flagged outlier rows before modelling.")
    return heads, checks


def _profile_bullets(profile: Any) -> tuple[list[str], list[str]]:
    heads: list[str] = []
    checks: list[str] = []
    score = getattr(profile, "quality_score", None)
    if score is not None:
        face = "🟢" if score >= 85 else ("🟡" if score >= 60 else "🔴")
        heads.append(f"{face} Data-quality score **{score}/100**.")
    warns = list(getattr(profile, "warnings", []) or [])
    for w in warns[:3]:
        heads.append(f"⚠️ {w}")
        low = w.lower()
        if "missing" in low:
            checks.append("Judge whether the high-missingness feature(s) are usable at all.")
        elif "collinearity" in low:
            checks.append("Pick one feature per collinear pair before regression/importance.")
        elif "cardinality" in low:
            checks.append("Decide an encoding (or drop) for the high-cardinality feature(s).")
    if len(warns) > 3:
        heads.append(f"⚠️ …and {len(warns) - 3} more warning(s) in section 2.")
    return heads, checks


def build_summary(
    profile: Any,
    step_reports: list[StepReport],
    advanced: Any = None,
) -> list[str]:
    """Markdown lines for the Executive Summary block (headlines + checklist)."""
    headlines: list[str] = []
    checklist: list[str] = []

    h, c = _profile_bullets(profile)
    headlines += h
    checklist += c
    h, c = _impact_bullets(step_reports)
    headlines += h
    checklist += c
    h, c = _outlier_bullets(step_reports)
    headlines += h
    checklist += c

    n_tests = getattr(advanced, "normality", None)
    if n_tests:
        non_normal = [t for t in n_tests if not getattr(t, "is_normal", True)]
        if non_normal:
            names = ", ".join(f"`{t.feature}`" for t in non_normal[:4])
            headlines.append(
                f"📐 {len(non_normal)} feature(s) reject normality ({names}"
                + ("…" if len(non_normal) > 4 else "") +
                ") — prefer rank/robust methods or the suggested transforms."
            )

    lines = ["> **TL;DR** — what matters in this dataset, worst news first.", ""]
    lines += [f"- {h}" for h in headlines[:_MAX_HEADLINES]]
    if checklist:
        lines += ["", "**Review checklist (human sign-off):**", ""]
        lines += [f"- [ ] {c}" for c in dict.fromkeys(checklist)]  # dedup, keep order
    lines.append("")
    return lines
