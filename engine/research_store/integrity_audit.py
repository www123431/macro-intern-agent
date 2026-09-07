"""engine.research_store.integrity_audit — audit the canonical events log.

Verifies the CLAUDE.md "Research Event Emission Doctrine" invariants
against the live data/research_store/events.jsonl:

  1. NO duplicate event_ids (the log is append-only + UUIDs SHOULD be
     unique per emit call; a duplicate means either an emit bug or a
     manual concatenation slip)

  2. Every `factor_verdict_filed` event has an `artifacts.evidence_doc`
     path (v15 doctrine — emit-time enforcement lives in emit.py, this
     audit catches PRE-v15 legacy events + manual writes that bypassed
     the API)

  3. Every referenced `evidence_doc` file exists on disk

  4. Every `parent_event_ids` reference resolves to an event_id present
     in the log (broken references mean either an intentionally deleted
     ancestor or corruption)

  5. Every event carries a `subject_id` that resolves in the subjects
     registry (subjects registry lives in data/research_store/
     subjects.yaml; the registry is single-source-of-truth for what
     subjects the store knows about)

Not an enforcement — this is a diagnostic that emits a report.
Enforcement happens at emit boundaries (see emit.py + shadow_emit.py).
Fixing legacy issues is human-in-the-loop work: sometimes a "broken"
parent_event_ids just means the ancestor was legitimately superseded
+ removed from the working set.

Air-gap doctrine: this module reads events but never writes them, so
it can be called from any surface (cron / UI / test) safely.
"""
from __future__ import annotations

import dataclasses as _dc
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVENTS_PATH   = _REPO_ROOT / "data" / "research_store" / "events.jsonl"
_SUBJECTS_PATH = _REPO_ROOT / "data" / "research_store" / "subjects.yaml"


@_dc.dataclass(frozen=True)
class IntegrityIssue:
    """One flagged issue. The category is stable so downstream digests
    can group + filter."""
    category:  str        # "duplicate_event_id" / "missing_evidence_doc" /
                          # "evidence_doc_file_missing" / "broken_parent_ref" /
                          # "unregistered_subject"
    event_id:  str        # the offending event
    detail:    str        # short human-readable rationale


@_dc.dataclass(frozen=True)
class IntegrityReport:
    """Aggregate audit result. Empty issues list = clean store."""
    n_events:      int
    n_verdicts:    int
    n_issues:      int
    by_category:   dict[str, int]
    issues:        tuple[IntegrityIssue, ...]

    @property
    def is_clean(self) -> bool:
        return self.n_issues == 0

    def summary_str(self) -> str:
        parts = [f"{c}={n}" for c, n in sorted(self.by_category.items())]
        return f"n_events={self.n_events} issues={self.n_issues}: " + " ".join(parts)


def _iter_events(path: Path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as fh:
        for ln_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("integrity_audit: %s line %d malformed",
                                 path.name, ln_no)


def _load_subject_registry(path: Path) -> set[str]:
    """Read the subjects registry (yaml). Returns the set of known
    subject_ids. Silently tolerates missing PyYAML / missing file."""
    if not path.is_file():
        return set()
    try:
        import yaml  # type: ignore
    except ImportError:
        return set()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    subs = data.get("subjects") if isinstance(data, dict) else data
    if isinstance(subs, dict):
        return {str(k) for k in subs.keys()}
    if isinstance(subs, list):
        return {
            str(r.get("subject_id") or r.get("id") or "")
            for r in subs if isinstance(r, dict)
        } - {""}
    return set()


def audit_events(
    *,
    events_path:   Optional[Path] = None,
    subjects_path: Optional[Path] = None,
    check_subject_registry: bool = True,
    check_evidence_doc_files: bool = True,
) -> IntegrityReport:
    """Run all five checks against the events log.

    `check_subject_registry` can be disabled when the subjects file is
    stale mid-migration; `check_evidence_doc_files` can be disabled
    when running from a snapshot without the docs dir.
    """
    events_path   = events_path   or _EVENTS_PATH
    subjects_path = subjects_path or _SUBJECTS_PATH

    known_subjects: set[str] = (
        _load_subject_registry(subjects_path)
        if check_subject_registry else set()
    )

    # Two-pass: first pass builds event_id set + collects issues that
    # can be judged in isolation; second pass checks parent_event_ids
    # references against the full id set.
    all_events: list[dict] = []
    seen_ids: dict[str, int] = {}   # event_id → occurrence count
    issues: list[IntegrityIssue] = []
    verdict_count = 0

    for ev in _iter_events(events_path):
        eid = ev.get("event_id") or ""
        all_events.append(ev)
        if eid:
            seen_ids[eid] = seen_ids.get(eid, 0) + 1

        et = ev.get("event_type") or ""
        if et == "factor_verdict_filed":
            verdict_count += 1
            artifacts = ev.get("artifacts") or {}
            evidence_doc = str(artifacts.get("evidence_doc") or "").strip()
            if not evidence_doc:
                issues.append(IntegrityIssue(
                    category = "missing_evidence_doc",
                    event_id = eid,
                    detail   = (f"factor_verdict_filed for subject={ev.get('subject_id')} "
                                  f"has no artifacts.evidence_doc (pre-v15 legacy or "
                                  f"manual write bypassing emit.py)"),
                ))
            elif check_evidence_doc_files:
                p = _REPO_ROOT / evidence_doc
                if not p.is_file():
                    issues.append(IntegrityIssue(
                        category = "evidence_doc_file_missing",
                        event_id = eid,
                        detail   = f"evidence_doc references '{evidence_doc}' but file not on disk",
                    ))

        # Subject registry
        if check_subject_registry and known_subjects:
            sid = str(ev.get("subject_id") or "")
            if sid and sid not in known_subjects:
                issues.append(IntegrityIssue(
                    category = "unregistered_subject",
                    event_id = eid,
                    detail   = f"subject_id '{sid}' not in subjects registry",
                ))

    # Duplicates
    for eid, count in seen_ids.items():
        if count > 1:
            issues.append(IntegrityIssue(
                category = "duplicate_event_id",
                event_id = eid,
                detail   = f"event_id '{eid}' appears {count} times in log",
            ))

    # Broken parent references
    id_set = set(seen_ids.keys())
    for ev in all_events:
        parents = ev.get("parent_event_ids") or []
        for pid in parents:
            if pid and pid not in id_set:
                issues.append(IntegrityIssue(
                    category = "broken_parent_ref",
                    event_id = ev.get("event_id") or "",
                    detail   = (f"parent_event_ids references '{pid}' "
                                  f"which is not present in the log"),
                ))

    # Aggregate
    by_cat: dict[str, int] = {}
    for iss in issues:
        by_cat[iss.category] = by_cat.get(iss.category, 0) + 1

    return IntegrityReport(
        n_events    = len(all_events),
        n_verdicts  = verdict_count,
        n_issues    = len(issues),
        by_category = dict(sorted(by_cat.items())),
        issues      = tuple(issues),
    )
