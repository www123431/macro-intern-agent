"""Tests for scripts/cron_fegd_scan.py — the weekly FEGD-scan cron
wrapper that closes the YAML-curated-knowledge blindspot loop.

Covers:
  - run_once(dry_run=True) returns a summary without writing rows
  - run_once(dry_run=False) actually writes new FEGD rows
  - Health record is emitted on success
  - Health record is emitted on failure (with error string)
  - End-to-end closure check: FEGD-emitted rows ARE picked up by
    burndown_ranker.load_demand_families (the whole point of the cron)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import cron_fegd_scan  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────


def _stub_sleeves_with_predictable_pnl(monkeypatch, tmp_path):
    """Replace the heavy combined_book builders with cheap, fixed
    synthetic PnL series. Three distinct series (one per sleeve) so
    each sleeve emits its own gap signatures.

    CRITICAL: each builder must return THE SAME series every time it
    is called — idempotency tests re-invoke run_once and the SAME
    sleeve PnL must produce the SAME factor regression → same
    signatures → no new writes. Using a shared `rng` closure here
    would advance state between calls and silently change the PnL."""
    idx = pd.date_range("2018-01-31", periods=96, freq="ME")
    # Independent seeds per sleeve so they get distinct (but each
    # individually deterministic) series.
    equity_pnl = pd.Series(
        np.random.default_rng(42).normal(0, 0.04, 96), index=idx,
    )
    carry_pnl = pd.Series(
        np.random.default_rng(43).normal(0, 0.04, 96), index=idx,
    )
    tsmom_pnl = pd.Series(
        np.random.default_rng(44).normal(0, 0.04, 96), index=idx,
    )

    monkeypatch.setattr(
        "engine.portfolio.combined_book.build_equity_book",
        lambda: equity_pnl.copy(),
    )
    monkeypatch.setattr(
        "engine.portfolio.combined_book.build_carry_book",
        lambda: carry_pnl.copy(),
    )
    monkeypatch.setattr(
        "engine.portfolio.combined_book.build_tsmom_book",
        lambda: tsmom_pnl.copy(),
    )

    # Redirect capability_gaps + health output to tmp_path so tests
    # don't pollute the live store. FEGD imports DEFAULT_GAPS_PATH from
    # deployment_demand_emitter (single source of truth for the gap
    # ledger path), so patching there is sufficient — emit_fegd_demand's
    # `out_path = gaps_path or _DEFAULT_GAPS_PATH` resolves the patched
    # default at call time since it's a local import in that function.
    gaps_path = tmp_path / "capability_gaps.jsonl"

    from engine.research import deployment_demand_emitter
    monkeypatch.setattr(deployment_demand_emitter, "DEFAULT_GAPS_PATH",
                          gaps_path)
    from engine.research import burndown_ranker
    monkeypatch.setattr(burndown_ranker, "DEFAULT_GAPS_PATH", gaps_path)

    # Health log → tmp
    monkeypatch.setattr(cron_fegd_scan, "HEALTH_PATH",
                          tmp_path / "fegd_scan.jsonl")

    return gaps_path


# ── run_once ──────────────────────────────────────────────────────────


def test_run_once_dry_run_returns_summary(monkeypatch, tmp_path):
    """Smoke test: dry-run produces a structured summary with the
    expected per-sleeve breakdown + no side effects on the gap ledger."""
    gaps_path = _stub_sleeves_with_predictable_pnl(monkeypatch, tmp_path)

    summary = cron_fegd_scan.run_once(dry_run=True)

    assert summary["dry_run"] is True
    assert summary["total_parsed"] >= 0
    assert summary["total_written"] == 0           # dry-run never writes
    assert len(summary["per_sleeve"]) == 3
    expected_sleeves = {"equity_book", "cross_asset_carry",
                         "cross_asset_tsmom"}
    actual_sleeves = {ps["sleeve_id"] for ps in summary["per_sleeve"]}
    assert actual_sleeves == expected_sleeves
    # Critical: dry-run did NOT touch the gap ledger
    assert not gaps_path.exists() or gaps_path.stat().st_size == 0


def test_run_once_write_mode_writes_rows(monkeypatch, tmp_path):
    """Write mode should emit at least one capability_gaps row to disk
    when the synthetic PnL has gap-factor t-stats below threshold
    (pure noise sleeve → all factors are gaps by construction)."""
    gaps_path = _stub_sleeves_with_predictable_pnl(monkeypatch, tmp_path)

    summary = cron_fegd_scan.run_once(dry_run=False)

    assert summary["dry_run"] is False
    assert summary["total_written"] >= 1, (
        "Pure-noise sleeves should detect at least one gap with t<1.65"
    )
    assert gaps_path.exists()
    rows = [
        json.loads(ln)
        for ln in gaps_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(rows) >= 1
    # Every row should carry the FEGD source tag for audit trail
    assert all(r.get("source") == "fegd_factor_gap" for r in rows)
    # Every row should have a family + gap_factor — these are the
    # ranker's join keys
    assert all(r.get("family") for r in rows)
    assert all(r.get("gap_factor") for r in rows)


def test_run_once_is_idempotent(monkeypatch, tmp_path):
    """Re-running the same scan should be a no-op (already_present > 0,
    written == 0). This is the safety guarantee that lets the cron
    run weekly without piling duplicates."""
    _stub_sleeves_with_predictable_pnl(monkeypatch, tmp_path)

    first = cron_fegd_scan.run_once(dry_run=False)
    assert first["total_written"] >= 1

    second = cron_fegd_scan.run_once(dry_run=False)
    assert second["total_written"] == 0, (
        "Second run should write nothing — signatures already present"
    )
    assert second["total_present"] == first["total_written"]


# ── main() health-row emission ────────────────────────────────────────


def test_main_records_ok_health_row(monkeypatch, tmp_path, capsys):
    """main() returns 0 on success AND appends an 'ok' status row to
    the health ledger so AgentHealth tile can display 'FEGD ran X
    ago, N gaps detected'."""
    _stub_sleeves_with_predictable_pnl(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["cron_fegd_scan.py"])

    rc = cron_fegd_scan.main()
    assert rc == 0
    assert cron_fegd_scan.HEALTH_PATH.exists()
    rows = [
        json.loads(ln)
        for ln in cron_fegd_scan.HEALTH_PATH.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["agent_id"] == "fegd_scan"
    assert rows[0]["status"] == "ok"
    assert "elapsed_s" in rows[0]
    assert rows[0]["parsed"] >= 0


def test_main_records_error_health_row(monkeypatch, tmp_path):
    """If run_once raises, main() returns 1 AND emits 'error' row with
    the exception class + message so ops_watchdog can alert."""
    monkeypatch.setattr(
        cron_fegd_scan, "HEALTH_PATH", tmp_path / "fegd_scan.jsonl",
    )

    def _boom(dry_run: bool):
        raise RuntimeError("synthetic FEGD failure for test")

    monkeypatch.setattr(cron_fegd_scan, "run_once", _boom)
    monkeypatch.setattr(sys, "argv", ["cron_fegd_scan.py"])

    rc = cron_fegd_scan.main()
    assert rc == 1
    rows = [
        json.loads(ln)
        for ln in cron_fegd_scan.HEALTH_PATH.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert "RuntimeError" in rows[0]["error"]
    assert "synthetic FEGD failure" in rows[0]["error"]


# ── End-to-end loop closure (the whole point of the cron) ────────────


def test_emitted_rows_picked_up_by_ranker(monkeypatch, tmp_path):
    """The architectural justification for this cron: FEGD-emitted
    rows MUST be visible to burndown_ranker.load_demand_families so
    the ranker's ×1.5 multiplier kicks in. This regression test locks
    that contract."""
    gaps_path = _stub_sleeves_with_predictable_pnl(monkeypatch, tmp_path)

    cron_fegd_scan.run_once(dry_run=False)
    assert gaps_path.exists()

    from engine.research.burndown_ranker import load_demand_families
    fams = load_demand_families(gaps_path=gaps_path)

    assert len(fams) >= 1, (
        "Ranker should see at least one family from the FEGD rows. "
        "If empty, the source tag / family field schema drifted — "
        "broken the closed loop deploy→research."
    )
