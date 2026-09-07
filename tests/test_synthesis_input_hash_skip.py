"""Tests for v31 (2026-07-01) input-hash Sonnet skip in synthesis_runner.

The core invariant: consecutive runs with an unchanged SynthesisInput
must not call Sonnet twice. When the hash matches the last-emitted
run's persisted `input_hash`, the second run short-circuits and
emits an audit event with `skipped_reason=input_unchanged_since_last_run`.
"""
from __future__ import annotations

import dataclasses as _dc
import hashlib
import json
from typing import Any

import pytest

from engine.agents.papers_curator import synthesis_runner as sr


# ── _synthesis_input_hash ───────────────────────────────────────────


@_dc.dataclass(frozen=True)
class _StubSI:
    """Minimal SynthesisInput stand-in — only the fields the hash reads."""
    recent_summaries:     tuple
    deployed_sleeves:     tuple
    recent_events:        tuple
    doctrine_snippets:    tuple
    anchor_library:       tuple = ()
    belief_layer_summary: tuple = ()
    # snapshot_ts is on the real SI but INTENTIONALLY not in the hash
    snapshot_ts:          str   = ""


@_dc.dataclass(frozen=True)
class _StubRow:
    """Tiny dataclass so asdict() has something to walk."""
    key:   str
    value: str = "v"


def test_hash_stable_across_repeated_calls():
    si = _StubSI(
        recent_summaries  = (_StubRow("s1"), _StubRow("s2")),
        deployed_sleeves  = (_StubRow("sleeve_a"),),
        recent_events     = (_StubRow("e1"), _StubRow("e2"), _StubRow("e3")),
        doctrine_snippets = (_StubRow("d1"),),
    )
    h1 = sr._synthesis_input_hash(si)
    h2 = sr._synthesis_input_hash(si)
    assert h1 == h2
    assert len(h1) == 64            # sha256 hex


def test_hash_changes_when_summaries_change():
    base = _StubSI(recent_summaries=(_StubRow("s1"),),
                    deployed_sleeves=(), recent_events=(),
                    doctrine_snippets=())
    changed = _dc.replace(base, recent_summaries=(_StubRow("s2"),))
    assert sr._synthesis_input_hash(base) != sr._synthesis_input_hash(changed)


def test_hash_changes_when_events_change():
    base = _StubSI(recent_summaries=(), deployed_sleeves=(),
                    recent_events=(_StubRow("e1"),),
                    doctrine_snippets=())
    changed = _dc.replace(base, recent_events=(_StubRow("e1"), _StubRow("e2")))
    assert sr._synthesis_input_hash(base) != sr._synthesis_input_hash(changed)


def test_hash_ignores_snapshot_ts():
    """snapshot_ts is per-run wall-clock — if it were in the hash we'd
    never skip anything. Verify explicitly."""
    base = _StubSI(recent_summaries=(_StubRow("s1"),),
                    deployed_sleeves=(), recent_events=(),
                    doctrine_snippets=(), snapshot_ts="2026-01-01T00:00:00Z")
    later = _dc.replace(base, snapshot_ts="2026-12-31T23:59:59Z")
    assert sr._synthesis_input_hash(base) == sr._synthesis_input_hash(later)


def test_hash_changes_when_sleeves_change():
    base = _StubSI(recent_summaries=(), deployed_sleeves=(_StubRow("a"),),
                    recent_events=(), doctrine_snippets=())
    changed = _dc.replace(base, deployed_sleeves=(_StubRow("b"),))
    assert sr._synthesis_input_hash(base) != sr._synthesis_input_hash(changed)


def test_hash_changes_when_doctrine_changes():
    base = _StubSI(recent_summaries=(), deployed_sleeves=(),
                    recent_events=(), doctrine_snippets=(_StubRow("d1"),))
    changed = _dc.replace(base, doctrine_snippets=(_StubRow("d2"),))
    assert sr._synthesis_input_hash(base) != sr._synthesis_input_hash(changed)


# ── _last_synthesis_input_hash ──────────────────────────────────────


def test_last_hash_none_when_store_empty(monkeypatch):
    def _empty(*_a, **_kw):
        return []
    monkeypatch.setattr("engine.research_store.store.filter_events", _empty)
    assert sr._last_synthesis_input_hash() is None


def test_last_hash_extracts_from_metrics(monkeypatch):
    """The lookup returns the string persisted at metrics.input_hash."""
    class _FakeEvent:
        metrics = {"input_hash": "abc123def456", "n_candidates": 0}
        ts = "2026-07-01T00:00:00Z"
    def _one(*_a, **_kw):
        return [_FakeEvent()]
    monkeypatch.setattr("engine.research_store.store.filter_events", _one)
    assert sr._last_synthesis_input_hash() == "abc123def456"


def test_last_hash_none_when_field_missing(monkeypatch):
    """Pre-v31 events have no input_hash → lookup returns None so caller
    doesn't accidentally match against '' + skip."""
    class _FakeEvent:
        metrics = {"n_candidates": 3}    # no input_hash field
        ts = "2026-07-01T00:00:00Z"
    monkeypatch.setattr(
        "engine.research_store.store.filter_events",
        lambda *_a, **_kw: [_FakeEvent()],
    )
    assert sr._last_synthesis_input_hash() is None


def test_last_hash_swallows_store_exceptions(monkeypatch):
    """Store lookup failing should never block the pipeline — the
    caller falls back to running Sonnet."""
    def _boom(*_a, **_kw):
        raise IOError("store unavailable")
    monkeypatch.setattr("engine.research_store.store.filter_events", _boom)
    assert sr._last_synthesis_input_hash() is None


# ── Skip integration ────────────────────────────────────────────────


def test_pipeline_skips_sonnet_when_hash_matches(monkeypatch):
    """Contract lock: if _last_synthesis_input_hash returns the current
    hash, run_synthesis MUST NOT be called."""
    # Stub build_synthesis_input to return a deterministic SI
    fake_si = _StubSI(
        recent_summaries=(_StubRow("paper1"),),
        deployed_sleeves=(),
        recent_events=(),
        doctrine_snippets=(),
        snapshot_ts="2026-07-01T00:00:00Z",
    )
    expected_hash = sr._synthesis_input_hash(fake_si)

    monkeypatch.setattr(sr, "build_synthesis_input",
                          lambda **_kw: fake_si)
    monkeypatch.setattr(sr, "_last_synthesis_input_hash",
                          lambda: expected_hash)

    sonnet_called = {"n": 0}
    def _fake_run_synthesis(_si):
        sonnet_called["n"] += 1
        return []
    monkeypatch.setattr(sr, "run_synthesis", _fake_run_synthesis)

    # Stub emit so we don't touch the real store
    emit_calls = []
    class _FakeEmitModule:
        @staticmethod
        def papers_curator_synthesis_run(**kw):
            emit_calls.append(kw)
            return "test_event_id"
    monkeypatch.setattr(
        "engine.research_store.emit",
        _FakeEmitModule,
    )

    r = sr.run_synthesis_pipeline(dry_run=True)

    assert sonnet_called["n"] == 0, "Sonnet must NOT be called on hash match"
    assert r["skipped_reason"] == "input_unchanged_since_last_run"
    assert r["n_candidates"] == 0
    assert r["event_id"] == "test_event_id"
    assert emit_calls[0]["snapshot"]["input_hash"] == expected_hash
    assert emit_calls[0]["snapshot"]["skipped_reason"] == \
        "input_unchanged_since_last_run"


def test_pipeline_runs_sonnet_when_no_prior_hash(monkeypatch):
    """First-ever run (no prior event to compare against) must call
    Sonnet as usual — the skip is a NEXT-run optimization."""
    fake_si = _StubSI(
        recent_summaries=(_StubRow("p"),),
        deployed_sleeves=(), recent_events=(),
        doctrine_snippets=(),
        snapshot_ts="2026-07-01T00:00:00Z",
    )
    monkeypatch.setattr(sr, "build_synthesis_input",
                          lambda **_kw: fake_si)
    monkeypatch.setattr(sr, "_last_synthesis_input_hash",
                          lambda: None)   # empty store

    sonnet_called = {"n": 0}
    def _fake_run_synthesis(_si):
        sonnet_called["n"] += 1
        return []
    monkeypatch.setattr(sr, "run_synthesis", _fake_run_synthesis)
    monkeypatch.setattr(
        sr, "_enrich_with_citation_checks",
        lambda cands: (cands, []),
    )

    class _FakeEmitModule:
        @staticmethod
        def papers_curator_synthesis_run(**kw):
            return "eid"
    monkeypatch.setattr("engine.research_store.emit", _FakeEmitModule)

    r = sr.run_synthesis_pipeline(dry_run=True)
    assert sonnet_called["n"] == 1, "First-ever run must call Sonnet"
    assert r.get("skipped_reason") is None


def test_pipeline_runs_sonnet_when_hash_differs(monkeypatch):
    """Different hash → real input change → must call Sonnet."""
    fake_si = _StubSI(
        recent_summaries=(_StubRow("new_paper"),),
        deployed_sleeves=(), recent_events=(),
        doctrine_snippets=(),
        snapshot_ts="2026-07-01T00:00:00Z",
    )
    monkeypatch.setattr(sr, "build_synthesis_input",
                          lambda **_kw: fake_si)
    monkeypatch.setattr(sr, "_last_synthesis_input_hash",
                          lambda: "different_stale_hash_from_yesterday")

    sonnet_called = {"n": 0}
    def _fake_run_synthesis(_si):
        sonnet_called["n"] += 1
        return []
    monkeypatch.setattr(sr, "run_synthesis", _fake_run_synthesis)
    monkeypatch.setattr(
        sr, "_enrich_with_citation_checks",
        lambda cands: (cands, []),
    )

    class _FakeEmitModule:
        @staticmethod
        def papers_curator_synthesis_run(**kw):
            return "eid"
    monkeypatch.setattr("engine.research_store.emit", _FakeEmitModule)

    r = sr.run_synthesis_pipeline(dry_run=True)
    assert sonnet_called["n"] == 1, "Sonnet must be called when hash differs"
    assert r.get("skipped_reason") is None


def test_snapshot_always_contains_input_hash(monkeypatch):
    """The persisted event's snapshot dict must always carry input_hash
    so the NEXT run has something to compare against."""
    fake_si = _StubSI(recent_summaries=(), deployed_sleeves=(),
                       recent_events=(), doctrine_snippets=(),
                       snapshot_ts="2026-07-01T00:00:00Z")
    monkeypatch.setattr(sr, "build_synthesis_input",
                          lambda **_kw: fake_si)
    monkeypatch.setattr(sr, "_last_synthesis_input_hash",
                          lambda: None)
    monkeypatch.setattr(sr, "run_synthesis", lambda _si: [])
    monkeypatch.setattr(
        sr, "_enrich_with_citation_checks",
        lambda cands: (cands, []),
    )

    captured_snapshot = {}
    class _FakeEmitModule:
        @staticmethod
        def papers_curator_synthesis_run(**kw):
            captured_snapshot.update(kw["snapshot"])
            return "eid"
    monkeypatch.setattr("engine.research_store.emit", _FakeEmitModule)

    r = sr.run_synthesis_pipeline(dry_run=True)
    assert "input_hash" in captured_snapshot
    assert len(captured_snapshot["input_hash"]) == 64   # sha256 hex


# ── emit persistence ───────────────────────────────────────────────


def test_emit_persists_input_hash_into_metrics(monkeypatch):
    """After v31, emit.papers_curator_synthesis_run must include
    input_hash + skipped_reason in the persisted metrics dict.

    Intercepts store.append at the emit boundary so we don't touch
    the real events.jsonl."""
    from engine.research_store import emit as emit_module

    captured: dict[str, Any] = {}
    def _fake_append(event):
        captured["event"] = event
    monkeypatch.setattr("engine.research_store.store.append", _fake_append)
    # Also stub the registry check because the tmp store won't have
    # 'papers_curator' subject registered.
    monkeypatch.setattr(
        "engine.research_store.registry.assert_subject_exists",
        lambda *a, **kw: None, raising=False,
    )

    emit_module.papers_curator_synthesis_run(
        n_candidates = 0,
        n_written    = 0,
        snapshot     = {
            "recent_summaries":  0,
            "deployed_sleeves":  5,
            "recent_events":     40,
            "doctrine_snippets": 5,
            "snapshot_ts":       "2026-07-01T00:00:00Z",
            "input_hash":        "aabbccddeeff0011",
            "skipped_reason":    "input_unchanged_since_last_run",
        },
        candidates = [],
    )

    ev = captured["event"]
    m = ev.metrics
    assert m["input_hash"]      == "aabbccddeeff0011"
    assert m["skipped_reason"] == "input_unchanged_since_last_run"
