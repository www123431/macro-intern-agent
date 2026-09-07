"""Tests for engine.auto_audit_rules.rule_stuck_severe_circuit_breaker (v29).

Locks the severity ladder against synthetic CB state + reset ledger
timestamps. Also verifies non-severe CB → no finding, and that the
rule handles missing / broken CB state gracefully.
"""
from __future__ import annotations

import datetime as _dt
from types import SimpleNamespace

import pytest

from engine import auto_audit_rules as aar


# ── Helpers ──────────────────────────────────────────────────────────


def _stub_cb(monkeypatch, *, level: str = "severe",
              triggered_at: str = None, reason: str = "test"):
    def _fake_status():
        return SimpleNamespace(level=level, reason=reason,
                                 triggered_at=triggered_at,
                                 auto_reset=False)
    monkeypatch.setattr("engine.circuit_breaker.get_status", _fake_status)


def _stub_reset_at(monkeypatch, ts):
    def _fake():
        return ts
    monkeypatch.setattr("engine.circuit_breaker.latest_reset_at", _fake)


def _iso_hours_ago(hours: float) -> str:
    ts = (_dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)
          - _dt.timedelta(hours=hours))
    return ts.isoformat()


# ── Not-severe cases return None ────────────────────────────────────


def test_returns_none_when_cb_level_is_none(monkeypatch):
    _stub_cb(monkeypatch, level="none")
    _stub_reset_at(monkeypatch, None)
    assert aar.rule_stuck_severe_circuit_breaker() is None


def test_returns_none_when_cb_level_is_light(monkeypatch):
    _stub_cb(monkeypatch, level="light", triggered_at=_iso_hours_ago(1))
    _stub_reset_at(monkeypatch, None)
    assert aar.rule_stuck_severe_circuit_breaker() is None


def test_returns_none_when_cb_level_is_medium(monkeypatch):
    _stub_cb(monkeypatch, level="medium", triggered_at=_iso_hours_ago(1))
    _stub_reset_at(monkeypatch, None)
    assert aar.rule_stuck_severe_circuit_breaker() is None


# ── Severity ladder ─────────────────────────────────────────────────


def test_low_when_cb_severe_less_than_24h(monkeypatch):
    _stub_cb(monkeypatch, triggered_at=_iso_hours_ago(2))
    _stub_reset_at(monkeypatch, None)
    r = aar.rule_stuck_severe_circuit_breaker()
    assert r["severity"] == "LOW"
    assert r["snapshot"]["age_hours"] == pytest.approx(2.0, abs=0.1)


def test_low_at_exact_24h_boundary(monkeypatch):
    """24h itself is inclusive of LOW."""
    _stub_cb(monkeypatch, triggered_at=_iso_hours_ago(24))
    _stub_reset_at(monkeypatch, None)
    r = aar.rule_stuck_severe_circuit_breaker()
    assert r["severity"] == "LOW"


def test_mid_when_cb_severe_between_24h_and_3days(monkeypatch):
    _stub_cb(monkeypatch, triggered_at=_iso_hours_ago(48))
    _stub_reset_at(monkeypatch, None)
    r = aar.rule_stuck_severe_circuit_breaker()
    assert r["severity"] == "MID"


def test_mid_at_exact_72h_boundary(monkeypatch):
    _stub_cb(monkeypatch, triggered_at=_iso_hours_ago(72))
    _stub_reset_at(monkeypatch, None)
    r = aar.rule_stuck_severe_circuit_breaker()
    assert r["severity"] == "MID"


def test_high_when_cb_severe_more_than_3days(monkeypatch):
    _stub_cb(monkeypatch, triggered_at=_iso_hours_ago(96))
    _stub_reset_at(monkeypatch, None)
    r = aar.rule_stuck_severe_circuit_breaker()
    assert r["severity"] == "HIGH"


def test_high_reproduces_v26_14day_scenario(monkeypatch):
    """The exact 2026-06-17 → 2026-07-01 timeline that motivated v29.
    14 days = 336h → HIGH."""
    _stub_cb(monkeypatch, triggered_at=_iso_hours_ago(24 * 14))
    _stub_reset_at(monkeypatch, None)
    r = aar.rule_stuck_severe_circuit_breaker()
    assert r["severity"] == "HIGH"
    assert r["snapshot"]["age_hours"] > 300


# ── Reset-ledger interaction ───────────────────────────────────────


def test_uses_reset_at_when_it_is_after_triggered(monkeypatch):
    """If the CB was triggered, then RESET, then triggered AGAIN, the
    clock starts at the newer trigger — but if triggered_at itself
    was reset then the reset timestamp is the clock start."""
    # Trigger 10 days ago, reset 5 days ago → clock starts at
    # max(-10d, -5d) = -5d → age 120h → HIGH
    _stub_cb(monkeypatch, triggered_at=_iso_hours_ago(24 * 10))
    _stub_reset_at(monkeypatch, _iso_hours_ago(24 * 5))
    r = aar.rule_stuck_severe_circuit_breaker()
    assert r["severity"] == "HIGH"
    assert r["snapshot"]["age_hours"] == pytest.approx(120, abs=1.0)


def test_recent_reset_makes_new_severe_low_even_if_old_trigger_ts(monkeypatch):
    """Trigger recorded 14 days ago, but there was a reset 1h ago
    (implies someone reset THEN CB re-triggered). Age counts from
    reset, so severity is LOW."""
    _stub_cb(monkeypatch, triggered_at=_iso_hours_ago(24 * 14))
    _stub_reset_at(monkeypatch, _iso_hours_ago(1))
    r = aar.rule_stuck_severe_circuit_breaker()
    assert r["severity"] == "LOW"
    assert r["snapshot"]["age_hours"] == pytest.approx(1.0, abs=0.1)


def test_returns_low_when_triggered_missing_and_no_reset(monkeypatch):
    """Defensive: if we can't compute age (no timestamps), default to
    LOW so we don't fire spurious HIGH pages. State is still surfaced."""
    _stub_cb(monkeypatch, triggered_at=None)
    _stub_reset_at(monkeypatch, None)
    r = aar.rule_stuck_severe_circuit_breaker()
    assert r["severity"] == "LOW"
    assert r["snapshot"]["age_hours"] == 0.0


# ── Failure paths ──────────────────────────────────────────────────


def test_gracefully_handles_get_status_exception(monkeypatch):
    def _boom():
        raise RuntimeError("cb probe broken")
    monkeypatch.setattr("engine.circuit_breaker.get_status", _boom)
    _stub_reset_at(monkeypatch, None)
    r = aar.rule_stuck_severe_circuit_breaker()
    assert r["severity"] == "LOW"
    assert r["snapshot"]["kind"] == "circuit_breaker_probe_failed"


def test_gracefully_handles_bad_timestamp_strings(monkeypatch):
    """Unparseable triggered_at should not raise — treat as 0 age."""
    _stub_cb(monkeypatch, triggered_at="not-an-iso-date")
    _stub_reset_at(monkeypatch, None)
    r = aar.rule_stuck_severe_circuit_breaker()
    assert r["severity"] == "LOW"


# ── Snapshot contract ──────────────────────────────────────────────


def test_snapshot_includes_expected_keys(monkeypatch):
    """Cockpit + auto-repair consume specific keys — regression-lock."""
    _stub_cb(monkeypatch, triggered_at=_iso_hours_ago(48),
              reason="ss_sp500 sleeve drift 25%")
    _stub_reset_at(monkeypatch, None)
    r = aar.rule_stuck_severe_circuit_breaker()
    snap = r["snapshot"]
    for key in ("kind", "level", "age_hours", "triggered_at",
                 "last_reset_at", "reason", "context", "fix_hint"):
        assert key in snap, f"missing snapshot key: {key}"
    assert snap["kind"] == "stuck_severe_circuit_breaker"
    assert snap["reason"] == "ss_sp500 sleeve drift 25%"


# ── Registry ────────────────────────────────────────────────────────


def test_rule_registered_in_watchdog_rules():
    from engine.auto_audit_rules import WATCHDOG_RULES
    assert aar.rule_stuck_severe_circuit_breaker in WATCHDOG_RULES
