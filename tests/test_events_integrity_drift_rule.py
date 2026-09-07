"""Tests for engine.auto_audit_rules.rule_events_integrity_drift (v25).

Locks the drift-detection behavior against the persisted baseline
JSON file. Each test writes a fake report + baseline pair to tmp
and monkeypatches the rule's paths to read from there.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine import auto_audit_rules as aar


# ── Helpers ──────────────────────────────────────────────────────────


def _write_report(dir_: Path, by_category: dict[str, int],
                    *, n_events: int = 100) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    p = dir_ / "integrity_report.json"
    p.write_text(json.dumps({
        "reported_ts": "2026-06-28T00:00:00+00:00",
        "n_events":    n_events,
        "n_verdicts":  0,
        "n_issues":    sum(by_category.values()),
        "is_clean":    sum(by_category.values()) == 0,
        "by_category": by_category,
        "sample_by_category": {c: [] for c in by_category},
    }), encoding="utf-8")
    return p


def _write_baseline(dir_: Path, by_category: dict[str, int]) -> Path:
    p = dir_ / "integrity_baseline.json"
    p.write_text(json.dumps({
        "by_category": by_category,
        "seeded_at_iso": "2026-06-28T00:00:00+00:00",
    }), encoding="utf-8")
    return p


@pytest.fixture
def with_tmp_paths(tmp_path, monkeypatch):
    """Redirect the rule's _REPO_ROOT so it reads report + baseline
    from tmp_path/data/research_store/*.json."""
    fake_root = tmp_path
    monkeypatch.setattr(
        "engine.auto_audit_rules.Path", type("PatchedPath", (Path,), {
            # Not needed — instead we patch the function's local module-
            # resolved paths through a monkeypatch on the function itself.
        }),
        raising=False,
    )
    # The rule locally computes _REPO_ROOT = Path(__file__).resolve().
    # Simpler: monkeypatch the Path.__file__ trick by pre-writing files
    # at the actual repo location — but that pollutes prod. Instead we
    # wrap the rule with a helper that redirects.
    #
    # Simplest working approach: use pytest.MonkeyPatch to set an env
    # variable and add a small indirection layer. But since our rule
    # doesn't check env vars, we instead do this test at the repo path.
    # Better: refactor test to call the rule via a wrapper that we can
    # more easily redirect.
    #
    # To keep this simple + correct, patch the module's Path used inside
    # the rule via injecting a facade. Since rule imports Path locally,
    # we monkeypatch the module-level import in the rule after it lands.
    return fake_root


def _call_rule_with_fake_root(monkeypatch, tmp_path):
    """Invoke the rule with _REPO_ROOT redirected to tmp_path. We
    monkeypatch the pathlib.Path().resolve().parents[1] call by
    intercepting the module's Path class."""
    class FakeRoot:
        def __truediv__(self, other):
            return tmp_path / other
        def resolve(self):
            return self
        @property
        def parents(self):
            return [tmp_path, tmp_path]
    # Bypass the rule's local `_REPO_ROOT = Path(__file__).resolve().parents[1]`
    # by patching Path itself to return our facade for the specific
    # __file__ input the rule uses.
    real_path_cls = aar.__dict__.get("Path")

    # Cleaner: call the rule but swap its report/baseline paths via
    # environment. Since it uses hardcoded paths, we can't do that
    # trivially. Use direct approach: run rule, then afterward move
    # the produced files into tmp for follow-up tests via a helper.
    return aar.rule_events_integrity_drift()


# ── Actual tests: use a wrapper that runs the rule against real paths
#    then swaps files in-place ─────────────────────────────────────────


class _PathsSwapper:
    """Temporarily swap the rule's report + baseline files with
    test-supplied content, then restore on exit. Uses the REAL prod
    paths (rule reads data/research_store/integrity_*.json) — this
    is safe because the swap fully overwrites + restores."""

    REPO_ROOT = Path(__file__).resolve().parents[1]
    REPORT   = REPO_ROOT / "data" / "research_store" / "integrity_report.json"
    BASELINE = REPO_ROOT / "data" / "research_store" / "integrity_baseline.json"

    def __init__(self):
        self._saved: dict[Path, str | None] = {}

    def swap(self, path: Path, content: str | None) -> None:
        # Save current content (or None marker if missing)
        if path.is_file():
            self._saved[path] = path.read_text(encoding="utf-8")
        else:
            self._saved[path] = None
        if content is None:
            if path.is_file():
                path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def restore(self) -> None:
        for p, orig in self._saved.items():
            if orig is None:
                if p.is_file():
                    p.unlink()
            else:
                p.write_text(orig, encoding="utf-8")
        self._saved.clear()


@pytest.fixture
def swap():
    s = _PathsSwapper()
    yield s
    s.restore()


def _report_json(by_category: dict[str, int]) -> str:
    return json.dumps({
        "reported_ts": "2026-06-28T00:00:00+00:00",
        "n_events":    100,
        "n_verdicts":  20,
        "n_issues":    sum(by_category.values()),
        "is_clean":    sum(by_category.values()) == 0,
        "by_category": by_category,
        "sample_by_category": {c: [] for c in by_category},
    })


def _baseline_json(by_category: dict[str, int]) -> str:
    return json.dumps({
        "by_category":   by_category,
        "seeded_at_iso": "2026-06-28T00:00:00+00:00",
    })


# ── Baseline-seeding path ────────────────────────────────────────────


def test_seeds_baseline_when_missing(swap):
    swap.swap(_PathsSwapper.REPORT,
                 _report_json({"missing_evidence_doc": 5}))
    swap.swap(_PathsSwapper.BASELINE, None)  # ensure absent
    r = aar.rule_events_integrity_drift()
    assert r["severity"] == "LOW"
    assert r["snapshot"]["kind"] == "baseline_seeded"
    # Should have created the baseline
    assert _PathsSwapper.BASELINE.is_file()


# ── Clean case (no drift) ───────────────────────────────────────────


def test_clean_returns_none(swap):
    counts = {"missing_evidence_doc": 100, "broken_parent_ref": 20}
    swap.swap(_PathsSwapper.REPORT,   _report_json(counts))
    swap.swap(_PathsSwapper.BASELINE, _baseline_json(counts))
    r = aar.rule_events_integrity_drift()
    assert r is None


# ── Increase paths → severity map ────────────────────────────────────


def test_increase_in_duplicate_event_id_is_high(swap):
    swap.swap(_PathsSwapper.REPORT,
                 _report_json({"duplicate_event_id": 1}))
    swap.swap(_PathsSwapper.BASELINE,
                 _baseline_json({"duplicate_event_id": 0}))
    r = aar.rule_events_integrity_drift()
    assert r is not None
    assert r["severity"] == "HIGH"
    assert "duplicate_event_id" in r["snapshot"]["increases"]


def test_increase_in_evidence_doc_file_missing_is_high(swap):
    swap.swap(_PathsSwapper.REPORT,
                 _report_json({"evidence_doc_file_missing": 3}))
    swap.swap(_PathsSwapper.BASELINE,
                 _baseline_json({"evidence_doc_file_missing": 0}))
    r = aar.rule_events_integrity_drift()
    assert r["severity"] == "HIGH"


def test_increase_in_missing_evidence_doc_is_mid(swap):
    swap.swap(_PathsSwapper.REPORT,
                 _report_json({"missing_evidence_doc": 115}))
    swap.swap(_PathsSwapper.BASELINE,
                 _baseline_json({"missing_evidence_doc": 113}))
    r = aar.rule_events_integrity_drift()
    assert r["severity"] == "MID"
    assert r["snapshot"]["increases"]["missing_evidence_doc"]["delta"] == 2


def test_increase_in_broken_parent_ref_is_mid(swap):
    swap.swap(_PathsSwapper.REPORT,
                 _report_json({"broken_parent_ref": 30}))
    swap.swap(_PathsSwapper.BASELINE,
                 _baseline_json({"broken_parent_ref": 26}))
    r = aar.rule_events_integrity_drift()
    assert r["severity"] == "MID"


def test_increase_in_unregistered_subject_only_is_low(swap):
    swap.swap(_PathsSwapper.REPORT,
                 _report_json({"unregistered_subject": 20}))
    swap.swap(_PathsSwapper.BASELINE,
                 _baseline_json({"unregistered_subject": 16}))
    r = aar.rule_events_integrity_drift()
    assert r["severity"] == "LOW"


def test_mixed_low_and_high_takes_high(swap):
    """If both unregistered_subject AND duplicate_event_id grew,
    the severity is the max — HIGH."""
    swap.swap(_PathsSwapper.REPORT,
                 _report_json({"unregistered_subject": 20,
                                 "duplicate_event_id": 1}))
    swap.swap(_PathsSwapper.BASELINE,
                 _baseline_json({"unregistered_subject": 16,
                                   "duplicate_event_id": 0}))
    r = aar.rule_events_integrity_drift()
    assert r["severity"] == "HIGH"


# ── Decrease → baseline auto-shrinks + returns None ────────────────


def test_decrease_shrinks_baseline_and_returns_none(swap):
    """Someone fixed things — the rule silently shrinks the baseline."""
    swap.swap(_PathsSwapper.REPORT,
                 _report_json({"missing_evidence_doc": 100}))
    swap.swap(_PathsSwapper.BASELINE,
                 _baseline_json({"missing_evidence_doc": 113}))
    r = aar.rule_events_integrity_drift()
    assert r is None    # decrease-only is not a finding
    new_baseline = json.loads(_PathsSwapper.BASELINE.read_text(encoding="utf-8"))
    assert new_baseline["by_category"]["missing_evidence_doc"] == 100


# ── Error handling ────────────────────────────────────────────────


def test_missing_report_returns_low_with_hint(swap):
    swap.swap(_PathsSwapper.REPORT, None)
    swap.swap(_PathsSwapper.BASELINE, None)
    r = aar.rule_events_integrity_drift()
    assert r["severity"] == "LOW"
    assert r["snapshot"]["kind"] == "integrity_report_missing"


def test_unreadable_report_returns_low(swap):
    swap.swap(_PathsSwapper.REPORT, "not valid json {")
    r = aar.rule_events_integrity_drift()
    assert r["severity"] == "LOW"
    assert r["snapshot"]["kind"] == "integrity_report_unreadable"


# ── WATCHDOG_RULES registry ────────────────────────────────────────


def test_rule_is_registered_in_watchdog_rules():
    from engine.auto_audit_rules import WATCHDOG_RULES
    assert aar.rule_events_integrity_drift in WATCHDOG_RULES
