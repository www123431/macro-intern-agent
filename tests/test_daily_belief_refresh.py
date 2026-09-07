"""Tests for scripts/cron/daily_belief_refresh.py (v23 extension).

Covers the two v23-added steps + the STEPS registry contract that
main() consumes. The three pre-existing steps (autopsy_backfill /
track_record_report / rigor_report) already had regression coverage
in test_belief_track_record.py; here we only test what v23 added
plus the aggregation invariant.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "cron"))

import daily_belief_refresh as dbr


# ── STEPS registry contract ──────────────────────────────────────────


def test_steps_registry_includes_v23_additions():
    """Adding a step is a schema change to what the AgentHealth tile /
    ops_watchdog reads. Regression-lock the labels."""
    labels = [label for label, _fn in dbr.STEPS]
    assert "autopsy_backfill" in labels
    assert "track_record_report" in labels
    assert "rigor_report" in labels
    assert "coverage_report" in labels           # v23 addition
    assert "temperature_loocv" in labels         # v23 addition
    assert "events_integrity" in labels          # v24 addition


def test_steps_are_ordered_pre_v23_first():
    """coverage_report depends on autopsy_backfill running first (it
    reads data/research/autopsies.jsonl). Same for temperature_loocv.
    So both must be AFTER autopsy_backfill in the STEPS list."""
    labels = [label for label, _fn in dbr.STEPS]
    ab_idx = labels.index("autopsy_backfill")
    cov_idx = labels.index("coverage_report")
    temp_idx = labels.index("temperature_loocv")
    assert ab_idx < cov_idx
    assert ab_idx < temp_idx


def test_all_steps_are_callable():
    for _label, fn in dbr.STEPS:
        assert callable(fn)


# ── step_coverage_report ─────────────────────────────────────────────


def test_coverage_report_step_returns_ok_on_live_corpus():
    """End-to-end: the shipped v19 coverage report is idempotent + fast.
    Running it in a test should succeed against the live autopsies file
    (or produce a graceful failure message we can inspect)."""
    ok, msg = dbr.step_coverage_report()
    assert isinstance(ok, bool)
    assert isinstance(msg, str)
    if ok:
        assert "coverage_report OK" in msg
    else:
        # Failure surface must include the failing script name so ops
        # digest can attribute the alert
        assert "coverage_report" in msg


def test_coverage_report_step_handles_missing_script(monkeypatch, tmp_path):
    """If the coverage report script goes missing (rename / rm), the
    step should FAIL gracefully with a non-zero-exit message, not
    raise."""
    # Point the module's REPO_ROOT at an empty tmp dir so the script
    # path resolves to something that doesn't exist
    monkeypatch.setattr(dbr, "REPO_ROOT", tmp_path)
    ok, msg = dbr.step_coverage_report()
    assert ok is False
    assert "coverage_report" in msg


# ── step_temperature_loocv ───────────────────────────────────────────


def test_temperature_loocv_step_returns_ok_on_live_corpus():
    ok, msg = dbr.step_temperature_loocv()
    assert isinstance(ok, bool)
    assert isinstance(msg, str)
    if ok:
        assert "temperature_loocv OK" in msg
    else:
        assert "temperature_loocv" in msg


def test_temperature_loocv_step_handles_missing_script(monkeypatch, tmp_path):
    monkeypatch.setattr(dbr, "REPO_ROOT", tmp_path)
    ok, msg = dbr.step_temperature_loocv()
    assert ok is False
    assert "temperature_loocv" in msg


# ── main() ───────────────────────────────────────────────────────────


def test_main_returns_zero_even_when_a_step_fails(monkeypatch, capsys):
    """Defensive cron contract: main() MUST return 0 even if a step
    fails, so schtasks doesn't treat the day's run as a total failure
    (next day retries naturally). Locked in the module docstring."""
    def _fail_step():
        return False, "synthetic step failure"

    monkeypatch.setattr(dbr, "STEPS", [
        ("only_step_that_fails", _fail_step),
    ])
    monkeypatch.setattr(sys, "argv", ["daily_belief_refresh.py"])
    rc = dbr.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "FAIL" in out


def test_main_prints_summary_line_with_totals(monkeypatch, capsys):
    """Regression: the summary line format is consumed by ops_watchdog
    and the AgentHealth digest. Keep the '<ok>/<total> OK' format."""
    def _ok_step():
        return True, "ok"

    monkeypatch.setattr(dbr, "STEPS", [
        ("step_a", _ok_step),
        ("step_b", _ok_step),
        ("step_c", _ok_step),
    ])
    monkeypatch.setattr(sys, "argv", ["daily_belief_refresh.py"])
    dbr.main()
    out = capsys.readouterr().out
    assert "3/3 OK" in out
    assert "0/3 failed" in out


def test_main_quiet_flag_suppresses_output(monkeypatch, capsys):
    def _ok_step():
        return True, "ok"

    monkeypatch.setattr(dbr, "STEPS", [("s", _ok_step)])
    monkeypatch.setattr(sys, "argv", ["daily_belief_refresh.py", "--quiet"])
    dbr.main()
    out = capsys.readouterr().out
    assert out == ""
