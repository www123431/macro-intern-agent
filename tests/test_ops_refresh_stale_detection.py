"""Tests for v27 (2026-07-01) stale-detection in ops_refresh_status.

When circuit_breaker gets manually reset AFTER the last refresh's
exit_code was 4 (SEVERE halt), the cached _REFRESH_STATE should be
treated as invalidated by /api/ops/refresh so the UI stops showing
the pre-reset halt message.
"""
from __future__ import annotations

import pytest

from api import main as api_main


@pytest.fixture(autouse=True)
def _reset_refresh_state():
    """Every test starts with a clean cache."""
    with api_main._REFRESH_LOCK:
        api_main._REFRESH_STATE.update({
            "running": False, "trigger": None, "started_at": None,
            "finished_at": None, "exit_code": None, "ok": None,
            "message": None, "log_tail": None,
        })
    yield
    with api_main._REFRESH_LOCK:
        api_main._REFRESH_STATE.update({
            "running": False, "trigger": None, "started_at": None,
            "finished_at": None, "exit_code": None, "ok": None,
            "message": None, "log_tail": None,
        })


# ── _cached_state_invalidated_by_cb_reset ───────────────────────────


def test_not_invalidated_when_no_exit_code_yet(monkeypatch):
    """Fresh cache (never run) → not stale."""
    state = {"running": False, "exit_code": None}
    _stub_cb_status(monkeypatch, "none")
    assert api_main._cached_state_invalidated_by_cb_reset(state) is False


def test_not_invalidated_when_running(monkeypatch):
    """A refresh in flight — never invalidate the running state."""
    state = {"running": True, "exit_code": 4}
    _stub_cb_status(monkeypatch, "none")
    assert api_main._cached_state_invalidated_by_cb_reset(state) is False


def test_not_invalidated_when_exit_code_not_4(monkeypatch):
    """Only exit_code=4 (CB SEVERE halt) is subject to CB-reset
    invalidation. Other halts (5=Risk Manager, 6=DQ) are not."""
    _stub_cb_status(monkeypatch, "none")
    for code in (0, 1, 2, 3, 5, 6, -1, -2):
        state = {"running": False, "exit_code": code}
        assert api_main._cached_state_invalidated_by_cb_reset(state) is False, (
            f"exit_code={code} shouldn't invalidate on CB reset"
        )


def test_invalidated_when_exit_4_and_cb_now_none(monkeypatch):
    """The 2026-07-01 scenario: cache says exit=4 from before reset;
    CB has been reset to none → invalidate."""
    state = {"running": False, "exit_code": 4}
    _stub_cb_status(monkeypatch, "none")
    assert api_main._cached_state_invalidated_by_cb_reset(state) is True


def test_not_invalidated_when_cb_still_severe(monkeypatch):
    """If CB is still SEVERE, the cached halt message is accurate —
    don't invalidate."""
    state = {"running": False, "exit_code": 4}
    _stub_cb_status(monkeypatch, "severe")
    assert api_main._cached_state_invalidated_by_cb_reset(state) is False


def test_gracefully_handles_cb_import_failure(monkeypatch):
    """If we can't read CB state for any reason, DO NOT falsely
    invalidate (safer to show the halt than mask it)."""
    def _boom():
        raise RuntimeError("cb probe broken")
    monkeypatch.setattr(
        "engine.circuit_breaker.get_status", _boom,
    )
    state = {"running": False, "exit_code": 4}
    assert api_main._cached_state_invalidated_by_cb_reset(state) is False


# ── ops_refresh_status endpoint ────────────────────────────────────


def test_endpoint_returns_raw_state_when_not_stale(monkeypatch):
    """Fresh success — endpoint returns state as-is (no stale_reason)."""
    with api_main._REFRESH_LOCK:
        api_main._REFRESH_STATE.update({
            "running": False, "exit_code": 0, "ok": True,
            "message": "Refresh complete", "log_tail": "ok",
        })
    _stub_cb_status(monkeypatch, "none")
    r = api_main.ops_refresh_status()
    assert r["exit_code"] == 0
    assert r["ok"] is True
    assert "stale_reason" not in r


def test_endpoint_masks_stale_severe_halt(monkeypatch):
    """The core 2026-07-01 fix: cached exit=4 + CB=none → endpoint
    returns invalidated view. The UI sees exit_code=None (nothing
    to render as halt) plus a stale_reason string for debug."""
    with api_main._REFRESH_LOCK:
        api_main._REFRESH_STATE.update({
            "running": False, "exit_code": 4, "ok": False,
            "message": "Halted: circuit breaker SEVERE — manual reset required "
                        "before a refresh can run.",
            "log_tail": "cb halt",
        })
    _stub_cb_status(monkeypatch, "none")
    r = api_main.ops_refresh_status()
    assert r["exit_code"] is None
    assert r["ok"] is None
    assert r["message"] is None
    assert r["log_tail"] is None
    assert "stale_reason" in r
    assert "reset" in r["stale_reason"].lower()


def test_endpoint_preserves_severe_halt_when_cb_still_broken(monkeypatch):
    """If CB is still SEVERE, the halt message IS accurate — the UI
    should keep showing it (no invalidation)."""
    with api_main._REFRESH_LOCK:
        api_main._REFRESH_STATE.update({
            "running": False, "exit_code": 4, "ok": False,
            "message": "Halted: circuit breaker SEVERE",
            "log_tail": "halt",
        })
    _stub_cb_status(monkeypatch, "severe")
    r = api_main.ops_refresh_status()
    assert r["exit_code"] == 4
    assert r["ok"] is False
    assert "stale_reason" not in r


def test_endpoint_leaves_running_refresh_alone(monkeypatch):
    """Even if CB is none, an in-flight refresh must be reported
    faithfully — don't clobber `running=True`."""
    with api_main._REFRESH_LOCK:
        api_main._REFRESH_STATE.update({
            "running": True, "trigger": "manual",
            "started_at": "2026-07-01T03:00:00",
            "exit_code": None, "ok": None,
            "message": None, "log_tail": None,
        })
    _stub_cb_status(monkeypatch, "none")
    r = api_main.ops_refresh_status()
    assert r["running"] is True
    assert r["trigger"] == "manual"


# ── helpers ─────────────────────────────────────────────────────────


def _stub_cb_status(monkeypatch, level: str) -> None:
    """Fake engine.circuit_breaker.get_status() returning an object
    with .level == level. Matches the shape produced by
    CircuitBreakerState (which the invalidator only reads .level on)."""
    from types import SimpleNamespace
    def _fake():
        return SimpleNamespace(level=level, reason=None, triggered_at=None)
    monkeypatch.setattr("engine.circuit_breaker.get_status", _fake)
