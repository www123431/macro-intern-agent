"""Tests for v28 (2026-07-01) circuit_breaker reset audit ledger.

Locks the observable contract that manual_reset() writes a row to
data/state/cb_reset_events.jsonl and that latest_reset_at() reads
back the most recent timestamp — used by /api/ops/refresh and any
future consumer that needs to know "did a reset happen since my
cached data?"
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest

from engine import circuit_breaker as cb


# ── Helpers ──────────────────────────────────────────────────────────


@pytest.fixture
def tmp_ledger(tmp_path, monkeypatch):
    """Redirect the reset ledger + persistent CB state file to tmp so
    tests don't collide with each other or the live audit trail."""
    ledger = tmp_path / "cb_reset_events.jsonl"
    state_file = tmp_path / "circuit_breaker.json"
    monkeypatch.setattr(cb, "_RESET_LEDGER", ledger)
    monkeypatch.setattr(cb, "_STATE_FILE", state_file)
    yield ledger


def _mk_severe(reason: str = "test") -> cb.CircuitBreakerState:
    return cb.CircuitBreakerState(
        level="severe", reason=reason,
        triggered_at=_dt.datetime.utcnow().replace(
            tzinfo=_dt.timezone.utc,
        ).isoformat(),
        auto_reset=False,
    )


# ── latest_reset_at ──────────────────────────────────────────────────


def test_latest_reset_at_returns_none_when_ledger_missing(tmp_ledger):
    # Ledger not created yet
    assert cb.latest_reset_at() is None


def test_latest_reset_at_returns_none_when_ledger_empty(tmp_ledger):
    tmp_ledger.parent.mkdir(parents=True, exist_ok=True)
    tmp_ledger.write_text("", encoding="utf-8")
    assert cb.latest_reset_at() is None


def test_latest_reset_at_returns_last_row_ts(tmp_ledger):
    tmp_ledger.parent.mkdir(parents=True, exist_ok=True)
    with tmp_ledger.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"ts": "2026-06-30T00:00:00+00:00",
                              "reason": "old"}) + "\n")
        f.write(json.dumps({"ts": "2026-07-01T03:30:00+00:00",
                              "reason": "newest"}) + "\n")
    assert cb.latest_reset_at() == "2026-07-01T03:30:00+00:00"


def test_latest_reset_at_tolerates_malformed_rows(tmp_ledger):
    tmp_ledger.parent.mkdir(parents=True, exist_ok=True)
    with tmp_ledger.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"ts": "2026-07-01T00:00:00+00:00"}) + "\n")
        f.write("not-valid-json\n")
        f.write(json.dumps({"ts": "2026-07-01T02:00:00+00:00"}) + "\n")
    # Skips the bad line, returns the last GOOD row's ts
    assert cb.latest_reset_at() == "2026-07-01T02:00:00+00:00"


def test_latest_reset_at_tolerates_missing_ts_field(tmp_ledger):
    tmp_ledger.parent.mkdir(parents=True, exist_ok=True)
    with tmp_ledger.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"ts": "2026-07-01T00:00:00+00:00"}) + "\n")
        f.write(json.dumps({"reason": "no-ts row"}) + "\n")
    assert cb.latest_reset_at() == "2026-07-01T00:00:00+00:00"


# ── manual_reset audit-log side effect ───────────────────────────────


def test_manual_reset_writes_row(tmp_ledger):
    cb._save_persistent(_mk_severe(reason="synthetic"))
    cb.manual_reset(reason="test reason")
    assert tmp_ledger.is_file()
    rows = [
        json.loads(ln)
        for ln in tmp_ledger.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["reason"] == "test reason"
    assert rows[0]["prior_level"] == "severe"
    assert rows[0]["prior_reason"] == "synthetic"
    assert "ts" in rows[0]


def test_manual_reset_captures_none_prior_when_never_persisted(tmp_ledger):
    """Resetting from a clean state (no persisted severe) still records
    the audit event — the prior_level just reads as 'none'."""
    cb.manual_reset(reason="idempotent reset")
    rows = [
        json.loads(ln)
        for ln in tmp_ledger.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["prior_level"] == "none"
    assert rows[0]["prior_reason"] is None


def test_manual_reset_reason_empty_string_records_empty_not_missing(tmp_ledger):
    """A caller omitting reason should still get a valid row with an
    empty-string reason — makes downstream parsing simpler."""
    cb.manual_reset()
    rows = [
        json.loads(ln)
        for ln in tmp_ledger.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert rows[0]["reason"] == ""


def test_manual_reset_is_append_only(tmp_ledger):
    """Two resets should produce two rows — never overwrite."""
    cb._save_persistent(_mk_severe(reason="first severe"))
    cb.manual_reset(reason="first reset")
    cb._save_persistent(_mk_severe(reason="second severe"))
    cb.manual_reset(reason="second reset")

    rows = [
        json.loads(ln)
        for ln in tmp_ledger.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(rows) == 2
    assert rows[0]["reason"] == "first reset"
    assert rows[1]["reason"] == "second reset"


def test_manual_reset_survives_ledger_io_failure(tmp_ledger, monkeypatch):
    """Best-effort contract: if the audit write fails for any reason,
    the reset itself must still succeed — a broken audit path cannot
    block admin operations."""
    cb._save_persistent(_mk_severe())

    # Point the ledger at a path whose parent already exists as a
    # FILE — mkdir(exist_ok=True) will raise FileExistsError → the
    # audit write path errors before it can open the file. Bytes ARE
    # in the "parent" so the append side has nowhere valid to go.
    bad_parent = tmp_ledger.parent / "not_a_dir"
    bad_parent.write_bytes(b"blocker file, not a directory")
    bad_ledger_path = bad_parent / "cb_reset_events.jsonl"
    monkeypatch.setattr(cb, "_RESET_LEDGER", bad_ledger_path)

    # This should NOT raise
    cb.manual_reset(reason="io-failure test")

    # And CB state should still be cleared
    assert cb._load_persistent() is None


# ── latest_reset_at + manual_reset integration ──────────────────────


def test_latest_reset_at_advances_after_manual_reset(tmp_ledger):
    assert cb.latest_reset_at() is None

    cb._save_persistent(_mk_severe())
    cb.manual_reset(reason="first")
    t1 = cb.latest_reset_at()
    assert t1 is not None

    # Small sleep to guarantee timestamp advances
    import time
    time.sleep(0.01)

    cb._save_persistent(_mk_severe())
    cb.manual_reset(reason="second")
    t2 = cb.latest_reset_at()
    assert t2 is not None
    assert t2 > t1
