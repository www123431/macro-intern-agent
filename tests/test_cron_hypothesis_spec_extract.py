"""Tests for scripts/cron_hypothesis_spec_extract.py (v32, 2026-07-01).

The v32 wrapper adds cadence + cost cap + health telemetry around the
existing scripts/backfill_hypothesis_specs.py logic. Tests focus on
the wrapper contract (pre-flight count, skip-when-done, health rows,
cost cap, subprocess error handling) — NOT the underlying LLM call,
which has its own coverage in test_hypothesis_spec_extractor.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import cron_hypothesis_spec_extract as csx


# ── Helpers ──────────────────────────────────────────────────────────


@pytest.fixture
def tmp_health(tmp_path, monkeypatch):
    """Redirect the health ledger to tmp so tests don't pollute the
    live audit trail."""
    p = tmp_path / "hypothesis_spec_extract.jsonl"
    monkeypatch.setattr(csx, "HEALTH_PATH", p)
    return p


# ── Pre-flight count ────────────────────────────────────────────────


def test_count_ready_hypotheses_returns_tuple(monkeypatch):
    """Basic shape: pre-flight function returns (total, already_speced)
    integers. Uses whatever the live corpus has — we're just contract-
    testing the return shape."""
    total, already = csx._count_ready_hypotheses()
    assert isinstance(total, int)
    assert isinstance(already, int)
    assert 0 <= already <= total


def test_count_ready_hypotheses_deduplicates_by_version(monkeypatch):
    """Multiple versions of the same hypothesis_id should count once —
    matches the underlying backfill script's latest_by_id logic."""
    from dataclasses import dataclass

    @dataclass
    class _FakeHyp:
        hypothesis_id: str
        version:       int

    def _fake_load():
        return [
            _FakeHyp("h1", 1), _FakeHyp("h1", 2), _FakeHyp("h1", 3),
            _FakeHyp("h2", 1),
        ]
    monkeypatch.setattr(
        "engine.research_store.hypothesis.load_hypotheses", _fake_load,
    )
    monkeypatch.setattr(
        "engine.hypothesis_spec.store.latest_for",
        lambda hid: None,
    )
    total, already = csx._count_ready_hypotheses()
    assert total == 2       # h1 + h2, not 4
    assert already == 0


# ── main() branches ────────────────────────────────────────────────


def test_main_exits_zero_when_no_backlog(monkeypatch, capsys, tmp_health):
    """When every hypothesis already has a spec, wrapper exits 0 without
    starting the LLM subprocess. Health row emits status=ok, extracted=0."""
    monkeypatch.setattr(csx, "_count_ready_hypotheses",
                          lambda: (100, 100))
    subprocess_calls: list = []
    monkeypatch.setattr(subprocess, "run",
                          lambda *a, **kw: subprocess_calls.append(a) or 1/0)
    monkeypatch.setattr(sys, "argv", ["cron_hypothesis_spec_extract.py"])

    rc = csx.main()
    assert rc == 0
    assert subprocess_calls == [], "subprocess must NOT run on empty backlog"
    out = capsys.readouterr().out
    assert "up-to-date" in out
    # Health row
    rows = [json.loads(ln) for ln in tmp_health.read_text(encoding="utf-8")
                                        .splitlines() if ln.strip()]
    assert rows[0]["status"] == "ok"
    assert rows[0]["extracted"] == 0
    assert rows[0]["to_extract"] == 0


def test_main_caps_extraction_at_limit(monkeypatch, capsys, tmp_health):
    """When backlog > --limit, wrapper caps the subprocess at --limit
    to protect the LLM budget."""
    monkeypatch.setattr(csx, "_count_ready_hypotheses",
                          lambda: (300, 50))  # 250 backlog

    ran: list = []
    class _FakeProc:
        returncode = 0
        stderr = ""
        stdout = "ok"
    def _fake_run(cmd, **kw):
        ran.append(cmd)
        return _FakeProc()
    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(sys, "argv",
                          ["cron_hypothesis_spec_extract.py", "--limit", "20"])

    rc = csx.main()
    assert rc == 0
    # Subprocess got --limit 20 (not 250)
    assert "--limit" in ran[0]
    idx = ran[0].index("--limit")
    assert ran[0][idx + 1] == "20"


def test_main_uses_full_backlog_when_smaller_than_limit(monkeypatch, tmp_health):
    """--limit 20, backlog 5 → subprocess gets --limit 5 (the smaller
    of the two). Prevents wasted subprocess overhead + accurate
    reporting."""
    monkeypatch.setattr(csx, "_count_ready_hypotheses",
                          lambda: (100, 95))    # 5 backlog

    ran: list = []
    class _FakeProc:
        returncode = 0
        stderr = ""
        stdout = ""
    def _fake_run(cmd, **kw):
        ran.append(cmd)
        return _FakeProc()
    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(sys, "argv",
                          ["cron_hypothesis_spec_extract.py", "--limit", "20"])
    csx.main()
    idx = ran[0].index("--limit")
    assert ran[0][idx + 1] == "5"


def test_dry_run_skips_subprocess(monkeypatch, capsys, tmp_health):
    """--dry-run reports work but does not spawn the LLM subprocess."""
    monkeypatch.setattr(csx, "_count_ready_hypotheses",
                          lambda: (300, 100))    # 200 backlog

    ran = []
    monkeypatch.setattr(subprocess, "run",
                          lambda *a, **kw: ran.append(a) or 1/0)
    monkeypatch.setattr(sys, "argv",
                          ["cron_hypothesis_spec_extract.py",
                           "--dry-run", "--limit", "20"])

    rc = csx.main()
    assert rc == 0
    assert ran == []
    rows = [json.loads(ln) for ln in tmp_health.read_text(encoding="utf-8")
                                        .splitlines() if ln.strip()]
    assert rows[0]["status"] == "dry_run"


def test_main_records_error_on_subprocess_failure(monkeypatch, tmp_health):
    """Non-zero exit from backfill subprocess → wrapper exits 1 +
    health row status='error' with stderr excerpt."""
    call_count = {"n": 0}
    def _stub(): call_count["n"] += 1; return (100, 50)
    monkeypatch.setattr(csx, "_count_ready_hypotheses", _stub)

    class _FailProc:
        returncode = 7
        stderr = "extract failed: connection reset"
        stdout = ""
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _FailProc())
    monkeypatch.setattr(sys, "argv", ["cron_hypothesis_spec_extract.py"])

    rc = csx.main()
    assert rc == 1
    rows = [json.loads(ln) for ln in tmp_health.read_text(encoding="utf-8")
                                        .splitlines() if ln.strip()]
    assert rows[0]["status"] == "error"
    assert "connection reset" in rows[0]["error"]


# ── v34 (2026-07-01): timeout scaling + partial progress accounting ─


def test_main_timeout_scales_with_cap(monkeypatch, tmp_health):
    """Regression against pre-v34 bug: hard-coded 600s timeout was
    too short for --limit=222 (real 2026-07-01 run hit it). v34
    scales the ceiling with the cost cap."""
    monkeypatch.setattr(csx, "_count_ready_hypotheses",
                          lambda: (300, 78))    # 222 backlog

    captured_timeout = {}
    class _OKProc:
        returncode = 0
        stderr = ""
        stdout = ""
    def _fake_run(cmd, **kw):
        captured_timeout["value"] = kw.get("timeout")
        return _OKProc()
    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(sys, "argv",
                          ["cron_hypothesis_spec_extract.py",
                           "--limit", "222"])
    csx.main()
    # v34: 222 × 30s/hyp = 6660s, well above the pre-v34 600s ceiling
    assert captured_timeout["value"] >= 222 * 20


def test_main_timeout_respects_floor_for_tiny_runs(monkeypatch, tmp_health):
    """--limit=1 shouldn't get a 30s timeout — subprocess startup +
    engine import take that long by themselves. Floor keeps small
    runs viable."""
    monkeypatch.setattr(csx, "_count_ready_hypotheses",
                          lambda: (100, 99))    # 1 backlog

    captured = {}
    class _OKProc:
        returncode = 0
        stderr = ""
        stdout = ""
    def _fake_run(cmd, **kw):
        captured["timeout"] = kw.get("timeout")
        return _OKProc()
    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(sys, "argv",
                          ["cron_hypothesis_spec_extract.py"])
    csx.main()
    assert captured["timeout"] >= 60    # floor


def test_main_timeout_reports_partial_progress_exit_0(monkeypatch,
                                                        tmp_health):
    """v34: when subprocess times out but PARTIAL work landed on disk,
    wrapper exits 0 (not 1) — treated as a warning; cron picks up
    remaining work tomorrow. Health row status='timeout' with
    accurate `extracted` count."""
    call = {"n": 0}
    def _stub():
        call["n"] += 1
        # Pre-flight: 78 speced. Post-timeout: 131 (53 landed before
        # kill, matching the real 2026-07-01 measurement).
        return (300, 131) if call["n"] > 1 else (300, 78)
    monkeypatch.setattr(csx, "_count_ready_hypotheses", _stub)

    def _timeout_run(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="backfill", timeout=600)
    monkeypatch.setattr(subprocess, "run", _timeout_run)
    monkeypatch.setattr(sys, "argv",
                          ["cron_hypothesis_spec_extract.py",
                           "--limit", "222"])

    rc = csx.main()
    assert rc == 0, "partial-progress timeout should be a warning, not error"
    rows = [json.loads(ln) for ln in tmp_health.read_text(encoding="utf-8")
                                        .splitlines() if ln.strip()]
    assert rows[0]["status"] == "timeout"
    assert rows[0]["extracted"] == 53    # 131 - 78
    assert "partial progress" in rows[0]["error"]


def test_main_timeout_with_zero_progress_exits_1(monkeypatch, tmp_health):
    """v34 boundary: if the subprocess timed out with ZERO extraction
    progress, something is genuinely stuck (import loop / LLM never
    responds / subprocess hung). Exit 1 so ops_watchdog surfaces it."""
    call = {"n": 0}
    def _stub():
        call["n"] += 1
        return (100, 50)   # never changes → 0 extracted
    monkeypatch.setattr(csx, "_count_ready_hypotheses", _stub)

    def _timeout_run(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="backfill", timeout=600)
    monkeypatch.setattr(subprocess, "run", _timeout_run)
    monkeypatch.setattr(sys, "argv", ["cron_hypothesis_spec_extract.py"])

    rc = csx.main()
    assert rc == 1
    rows = [json.loads(ln) for ln in tmp_health.read_text(encoding="utf-8")
                                        .splitlines() if ln.strip()]
    assert rows[0]["extracted"] == 0
    assert rows[0]["status"] == "timeout"


def test_main_records_partial_progress_on_wrapper_exception(monkeypatch,
                                                              tmp_health):
    """v34: even if the wrapper itself crashes AFTER some backfill
    work landed on disk, the health row should reflect the actual
    progress — not fall back to extracted=0 (the pre-v34 bug)."""
    call = {"n": 0}
    def _stub():
        call["n"] += 1
        if call["n"] == 1:
            return (100, 30)   # pre-flight
        # Wrapper crashes after subprocess. Second call: 10 more speced.
        raise RuntimeError("simulated: subprocess.run raised")
    monkeypatch.setattr(csx, "_count_ready_hypotheses", _stub)
    monkeypatch.setattr(sys, "argv", ["cron_hypothesis_spec_extract.py"])

    rc = csx.main()
    assert rc == 1
    rows = [json.loads(ln) for ln in tmp_health.read_text(encoding="utf-8")
                                        .splitlines() if ln.strip()]
    # Even though we could not measure the delta (recount also
    # raised), the health row must still exist AND be classified
    # error. extracted=0 in this pathological case is honest —
    # we don't know how much landed.
    assert rows[0]["status"] == "error"
    assert "RuntimeError" in rows[0]["error"]


def test_main_records_extraction_delta(monkeypatch, capsys, tmp_health):
    """After a successful subprocess run, the wrapper re-checks the
    store and records how many NEW specs got persisted. This is the
    "did anything actually happen" telemetry."""
    call_count = {"n": 0}
    def _count_stub():
        call_count["n"] += 1
        # Pre-flight: 50 already speced. Post-run: 55 (5 new).
        return (100, 55) if call_count["n"] > 1 else (100, 50)

    monkeypatch.setattr(csx, "_count_ready_hypotheses", _count_stub)
    class _OKProc:
        returncode = 0
        stderr = ""
        stdout = ""
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _OKProc())
    monkeypatch.setattr(sys, "argv",
                          ["cron_hypothesis_spec_extract.py", "--limit", "10"])

    rc = csx.main()
    assert rc == 0
    rows = [json.loads(ln) for ln in tmp_health.read_text(encoding="utf-8")
                                        .splitlines() if ln.strip()]
    assert rows[0]["extracted"] == 5    # 55 - 50


def test_main_catches_unhandled_exceptions(monkeypatch, tmp_health):
    """Any exception in the wrapper itself gets caught + logged +
    exit 1. Cron must never surface a stack trace to the operator."""
    def _boom():
        raise RuntimeError("synthetic wrapper failure")
    monkeypatch.setattr(csx, "_count_ready_hypotheses", _boom)
    monkeypatch.setattr(sys, "argv", ["cron_hypothesis_spec_extract.py"])

    rc = csx.main()
    assert rc == 1
    rows = [json.loads(ln) for ln in tmp_health.read_text(encoding="utf-8")
                                        .splitlines() if ln.strip()]
    assert rows[0]["status"] == "error"
    assert "RuntimeError" in rows[0]["error"]
    assert "synthetic wrapper failure" in rows[0]["error"]
