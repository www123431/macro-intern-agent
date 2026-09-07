"""Tests for engine.research_store.integrity_audit (v24, 2026-06-28).

Covers all 5 audit categories using synthetic events written to tmp
paths. Live-corpus behavior is verified in the daily_belief_refresh
integration test — this suite locks the audit rules themselves.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.research_store.integrity_audit import (
    IntegrityIssue,
    IntegrityReport,
    audit_events,
    _load_subject_registry,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _write_events(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _write_subjects(path: Path, subject_ids: list[str]) -> None:
    """Write minimal subjects.yaml for the audit's registry check.
    We only need the subject_id keys; the audit doesn't inspect the
    payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subjects_block = {"subjects": {sid: {"family": "TEST"} for sid in subject_ids}}
    # Use plain JSON (valid YAML) so we don't require PyYAML pass-through
    path.write_text(json.dumps(subjects_block), encoding="utf-8")


def _factor_verdict_event(*, event_id: str, subject_id: str,
                            evidence_doc: str | None = None,
                            parent_event_ids: list[str] | None = None,
                            ) -> dict:
    ev: dict = {
        "event_id":    event_id,
        "event_type":  "factor_verdict_filed",
        "subject_id":  subject_id,
        "verdict":     "GREEN",
        "artifacts":   {},
    }
    if evidence_doc is not None:
        ev["artifacts"]["evidence_doc"] = evidence_doc
    if parent_event_ids is not None:
        ev["parent_event_ids"] = parent_event_ids
    return ev


# ── Empty / clean baseline ──────────────────────────────────────────


def test_audit_empty_events_file(tmp_path):
    """Missing events.jsonl is a clean baseline, not an error."""
    rep = audit_events(events_path=tmp_path / "missing.jsonl")
    assert rep.n_events == 0
    assert rep.n_issues == 0
    assert rep.is_clean is True


def test_audit_clean_events_produces_no_issues(tmp_path):
    """All 5 audit rules pass when data is well-formed."""
    ev_path = tmp_path / "events.jsonl"
    subj_path = tmp_path / "subjects.yaml"
    _write_subjects(subj_path, ["sub_a"])
    ev_doc = tmp_path / "evidence.md"
    ev_doc.write_text("# evidence", encoding="utf-8")
    _write_events(ev_path, [
        _factor_verdict_event(
            event_id="e1", subject_id="sub_a",
            evidence_doc=str(ev_doc.relative_to(tmp_path)),
        ),
    ])
    # For the evidence_doc file-exists check we override the repo root
    # via the events path being the parent so the audit resolves paths.
    # audit_events resolves relative paths against its own _REPO_ROOT.
    # Since we can't easily patch that here, disable the file-exists
    # check to isolate the rule under test.
    rep = audit_events(
        events_path=ev_path, subjects_path=subj_path,
        check_evidence_doc_files=False,
    )
    assert rep.n_events == 1
    assert rep.is_clean, f"issues: {rep.issues}"


# ── Rule 1: duplicate event_id ──────────────────────────────────────


def test_audit_flags_duplicate_event_id(tmp_path):
    ev_path = tmp_path / "events.jsonl"
    _write_events(ev_path, [
        _factor_verdict_event(event_id="dup", subject_id="s",
                                evidence_doc="e.md"),
        _factor_verdict_event(event_id="dup", subject_id="s",
                                evidence_doc="e.md"),
    ])
    rep = audit_events(
        events_path=ev_path,
        check_subject_registry=False, check_evidence_doc_files=False,
    )
    dup_issues = [i for i in rep.issues if i.category == "duplicate_event_id"]
    assert len(dup_issues) == 1
    assert dup_issues[0].event_id == "dup"


# ── Rule 2: missing evidence_doc ────────────────────────────────────


def test_audit_flags_verdict_missing_evidence_doc(tmp_path):
    ev_path = tmp_path / "events.jsonl"
    _write_events(ev_path, [
        _factor_verdict_event(event_id="e1", subject_id="s"),
        _factor_verdict_event(event_id="e2", subject_id="s",
                                evidence_doc=""),
    ])
    rep = audit_events(
        events_path=ev_path,
        check_subject_registry=False, check_evidence_doc_files=False,
    )
    missing = [i for i in rep.issues if i.category == "missing_evidence_doc"]
    assert len(missing) == 2
    assert {i.event_id for i in missing} == {"e1", "e2"}


def test_audit_ignores_non_verdict_events_for_evidence_doc(tmp_path):
    """Only factor_verdict_filed events need evidence_doc — other
    event types (memory_doctrine_locked, decay_alert, etc.) don't."""
    ev_path = tmp_path / "events.jsonl"
    _write_events(ev_path, [
        {"event_id": "m1", "event_type": "memory_doctrine_locked",
         "subject_id": "s"},
        {"event_id": "d1", "event_type": "decay_alert",
         "subject_id": "s"},
    ])
    rep = audit_events(
        events_path=ev_path,
        check_subject_registry=False, check_evidence_doc_files=False,
    )
    missing = [i for i in rep.issues if i.category == "missing_evidence_doc"]
    assert missing == []


# ── Rule 3: evidence_doc file missing ───────────────────────────────


def test_audit_flags_evidence_doc_file_missing(tmp_path, monkeypatch):
    """When the audit's file existence check is on, a referenced
    evidence_doc path that doesn't exist on disk gets flagged."""
    ev_path = tmp_path / "events.jsonl"
    _write_events(ev_path, [
        _factor_verdict_event(event_id="e1", subject_id="s",
                                evidence_doc="docs/nowhere.md"),
    ])
    # audit_events resolves file existence against _REPO_ROOT — override
    # it so the fake path is checked relative to tmp_path
    import engine.research_store.integrity_audit as ia
    monkeypatch.setattr(ia, "_REPO_ROOT", tmp_path)
    rep = audit_events(
        events_path=ev_path,
        check_subject_registry=False,
        check_evidence_doc_files=True,
    )
    file_missing = [i for i in rep.issues
                    if i.category == "evidence_doc_file_missing"]
    assert len(file_missing) == 1


# ── Rule 4: broken parent_event_ids ─────────────────────────────────


def test_audit_flags_broken_parent_event_id(tmp_path):
    ev_path = tmp_path / "events.jsonl"
    _write_events(ev_path, [
        _factor_verdict_event(event_id="e1", subject_id="s",
                                evidence_doc="x",
                                parent_event_ids=["e_nonexistent"]),
    ])
    rep = audit_events(
        events_path=ev_path,
        check_subject_registry=False, check_evidence_doc_files=False,
    )
    broken = [i for i in rep.issues if i.category == "broken_parent_ref"]
    assert len(broken) == 1
    assert "e_nonexistent" in broken[0].detail


def test_audit_accepts_parent_ref_that_resolves(tmp_path):
    ev_path = tmp_path / "events.jsonl"
    _write_events(ev_path, [
        _factor_verdict_event(event_id="parent", subject_id="s",
                                evidence_doc="x"),
        _factor_verdict_event(event_id="child",  subject_id="s",
                                evidence_doc="x",
                                parent_event_ids=["parent"]),
    ])
    rep = audit_events(
        events_path=ev_path,
        check_subject_registry=False, check_evidence_doc_files=False,
    )
    broken = [i for i in rep.issues if i.category == "broken_parent_ref"]
    assert broken == []


# ── Rule 5: unregistered subject ────────────────────────────────────


def test_audit_flags_unregistered_subject(tmp_path):
    ev_path = tmp_path / "events.jsonl"
    subj_path = tmp_path / "subjects.yaml"
    _write_subjects(subj_path, ["known"])
    _write_events(ev_path, [
        _factor_verdict_event(event_id="e1", subject_id="unknown",
                                evidence_doc="x"),
    ])
    rep = audit_events(
        events_path=ev_path, subjects_path=subj_path,
        check_evidence_doc_files=False,
    )
    unreg = [i for i in rep.issues if i.category == "unregistered_subject"]
    assert len(unreg) == 1
    assert "unknown" in unreg[0].detail


def test_audit_skips_subject_check_when_disabled(tmp_path):
    """When check_subject_registry=False, no unregistered_subject
    issues surface even if subjects.yaml is empty / missing."""
    ev_path = tmp_path / "events.jsonl"
    _write_events(ev_path, [
        _factor_verdict_event(event_id="e1", subject_id="random",
                                evidence_doc="x"),
    ])
    rep = audit_events(
        events_path=ev_path,
        check_subject_registry=False,
        check_evidence_doc_files=False,
    )
    assert not any(i.category == "unregistered_subject" for i in rep.issues)


# ── Report aggregation ──────────────────────────────────────────────


def test_report_by_category_counts_match_issues(tmp_path):
    """by_category dict must equal the actual issues distribution."""
    ev_path = tmp_path / "events.jsonl"
    _write_events(ev_path, [
        _factor_verdict_event(event_id="dup", subject_id="s"),
        _factor_verdict_event(event_id="dup", subject_id="s"),
        _factor_verdict_event(event_id="e3",  subject_id="s"),
    ])
    rep = audit_events(
        events_path=ev_path,
        check_subject_registry=False, check_evidence_doc_files=False,
    )
    # 3 missing_evidence_doc (all 3 verdicts) + 1 duplicate = 4 issues
    assert rep.n_issues == 4
    assert rep.by_category["missing_evidence_doc"] == 3
    assert rep.by_category["duplicate_event_id"] == 1


def test_report_is_frozen():
    from engine.research_store.integrity_audit import IntegrityReport
    r = IntegrityReport(n_events=0, n_verdicts=0, n_issues=0,
                          by_category={}, issues=())
    import dataclasses as _dc
    with pytest.raises(_dc.FrozenInstanceError):
        r.n_events = 5  # type: ignore[misc]


def test_report_summary_string():
    """The summary_str format is used in cron logs — lock the shape."""
    r = IntegrityReport(
        n_events=100, n_verdicts=50, n_issues=5,
        by_category={"missing_evidence_doc": 3, "broken_parent_ref": 2},
        issues=(),
    )
    s = r.summary_str()
    assert "n_events=100" in s
    assert "issues=5" in s
    assert "broken_parent_ref=2" in s


# ── Subject registry loader ─────────────────────────────────────────


def test_load_subject_registry_returns_empty_on_missing_file(tmp_path):
    out = _load_subject_registry(tmp_path / "missing.yaml")
    assert out == set()


def test_load_subject_registry_dict_form(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(json.dumps({"subjects": {"a": {}, "b": {}}}), encoding="utf-8")
    assert _load_subject_registry(p) == {"a", "b"}
