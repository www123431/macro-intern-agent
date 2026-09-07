"""Tests for scripts/reports/report_belief_coverage.py.

Tests the pure compute_report() function + render_markdown() against
synthetic autopsy data so the test doesn't depend on the live store.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make scripts/reports importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "reports"))

import report_belief_coverage as rbc


def _write_autopsies(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _row(family: str, verdict: str, *, n_obs_months: int = 360) -> dict:
    return {
        "strategy_family": family,
        "actual_verdict":  verdict,
        "n_obs_months":    n_obs_months,
    }


def test_load_autopsies_returns_empty_when_missing(tmp_path):
    """Don't crash if the live autopsies file is missing."""
    assert rbc._load_autopsies(tmp_path / "missing.jsonl") == []


def test_load_autopsies_skips_malformed_rows(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text(
        '{"strategy_family": "X", "actual_verdict": "GREEN"}\n'
        'not-valid-json\n'
        '{"strategy_family": "X", "actual_verdict": "RED"}\n',
        encoding="utf-8",
    )
    out = rbc._load_autopsies(p)
    assert len(out) == 2


def test_family_breakdown_counts_verdicts():
    rows = [
        _row("A", "GREEN"),
        _row("A", "GREEN"),
        _row("A", "RED"),
        _row("B", "MARGINAL"),
    ]
    out = rbc._family_breakdown(rows)
    assert out["A"]["GREEN"] == 2
    assert out["A"]["RED"] == 1
    assert out["A"]["n"] == 3
    assert out["B"]["MARGINAL"] == 1
    assert out["B"]["n"] == 1


def test_family_breakdown_skips_superseded():
    rows = [
        _row("A", "GREEN"),
        {**_row("A", "RED"), "superseded_by": "corr_2026_06_28"},
        _row("A", "MARGINAL"),
    ]
    out = rbc._family_breakdown(rows)
    assert out["A"]["n"] == 2          # superseded RED dropped
    assert out["A"]["RED"] == 0
    assert out["A"]["GREEN"] == 1
    assert out["A"]["MARGINAL"] == 1


def test_compute_report_against_synthetic_corpus(tmp_path, monkeypatch):
    """End-to-end with a small synthetic corpus — verify coverage math
    + per-family entries shape."""
    autopsies = tmp_path / "autopsies.jsonl"
    _write_autopsies(autopsies, [
        # Family with N=5 — eligible
        _row("FAM_BIG", "GREEN"),
        _row("FAM_BIG", "GREEN"),
        _row("FAM_BIG", "RED"),
        _row("FAM_BIG", "RED"),
        _row("FAM_BIG", "MARGINAL"),
        # Family with N=3 — v19 eligible
        _row("FAM_THREE", "RED"),
        _row("FAM_THREE", "RED"),
        _row("FAM_THREE", "RED"),
        # Family with N=2 — still below cutoff
        _row("FAM_TWO", "RED"),
        _row("FAM_TWO", "RED"),
    ])

    monkeypatch.setattr(rbc, "_AUTOPSIES_PATH", autopsies)
    # Also patch the belief_prior_calibration module path so it sees
    # the same file (in case calibrated_family_prior is invoked).
    import engine.research.belief_prior_calibration as bpc
    monkeypatch.setattr(bpc, "AUTOPSIES_PATH", autopsies)

    rep = rbc.compute_report()
    s = rep["summary"]

    assert s["n_autopsies"] == 10
    assert s["n_families"] == 3
    assert s["min_autopsies_for_override"] == 3
    assert s["eligible_families"] == 2          # BIG + THREE
    assert s["eligible_autopsies"] == 8          # 5 + 3
    assert abs(s["coverage_pct"] - 80.0) < 0.01

    # FAM_TWO should be flagged as not eligible
    f_two = next(f for f in rep["families"] if f["family"] == "FAM_TWO")
    assert f_two["eligible"] is False
    assert f_two["calibrated_prior"] is None
    assert f_two["kl_from_override"] is None

    # FAM_THREE should be eligible with a calibrated prior shifted
    # toward RED (since 100% of its autopsies were RED)
    f_three = next(f for f in rep["families"] if f["family"] == "FAM_THREE")
    assert f_three["eligible"] is True
    assert f_three["calibrated_prior"] is not None
    assert f_three["calibrated_prior"]["RED"] > 0.55
    assert f_three["kl_from_override"] is not None and \
           f_three["kl_from_override"] > 0


def test_render_markdown_smoke(tmp_path, monkeypatch):
    """Markdown report renders without error + contains key sections."""
    autopsies = tmp_path / "autopsies.jsonl"
    _write_autopsies(autopsies, [
        _row("FOO", "GREEN") for _ in range(5)
    ])
    monkeypatch.setattr(rbc, "_AUTOPSIES_PATH", autopsies)
    import engine.research.belief_prior_calibration as bpc
    monkeypatch.setattr(bpc, "AUTOPSIES_PATH", autopsies)

    rep = rbc.compute_report()
    md = rbc.render_markdown(rep)

    assert "# Belief Layer Phase 4 — Coverage Report" in md
    assert "## Headline" in md
    assert "## Per-family breakdown" in md
    assert "FOO" in md
    assert "Override G" in md
    assert "Calibrated G" in md


def test_kl_div_returns_zero_on_degenerate():
    """KL undefined when either p or q has 0 — implementation skips
    those terms rather than returning inf or raising."""
    p = {"GREEN": 1.0, "MARGINAL": 0.0, "RED": 0.0}
    q = {"GREEN": 0.5, "MARGINAL": 0.5, "RED": 0.0}
    # Should not raise — RED term is skipped (both 0); GREEN term computed
    kl = rbc._kl_div(p, q)
    assert kl > 0
    assert kl < float("inf")


def test_kl_div_zero_for_identical_distributions():
    p = {"GREEN": 0.3, "MARGINAL": 0.4, "RED": 0.3}
    assert abs(rbc._kl_div(p, p)) < 1e-9
