"""Emit a daily integrity report for the research_store events log.

Runs the 5 checks in engine.research_store.integrity_audit against
the live events.jsonl + subjects.yaml and writes Markdown + JSON to
data/research_store/integrity_report.{md,json}. The audit is a
diagnostic, not an enforcement — fixing legacy issues is manual.

Usage:
    python scripts/reports/report_events_integrity.py
    python scripts/reports/report_events_integrity.py --json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from engine.research_store.integrity_audit import audit_events

_DEFAULT_OUT  = _REPO_ROOT / "data" / "research_store" / "integrity_report.md"
_DEFAULT_JSON = _REPO_ROOT / "data" / "research_store" / "integrity_report.json"

_CATEGORY_EXPLAIN = {
    "duplicate_event_id": (
        "Event log contains the same event_id twice. This should be "
        "impossible under the emit.py contract (auto-generated UUID "
        "per call). A hit here means either a bug in emit.py or a "
        "manual concatenation slip."
    ),
    "missing_evidence_doc": (
        "A factor_verdict_filed event has no artifacts.evidence_doc "
        "path. v15 (2026-06-26) added emit-time enforcement, so these "
        "are pre-v15 legacy events. Not a bug — historical artifact — "
        "but they will fail S7 PROMOTE Gate 3 (PIT-clean) if they "
        "ever reach the promote pipeline."
    ),
    "evidence_doc_file_missing": (
        "An event references an evidence_doc path that isn't on disk. "
        "Usually means the doc was deleted (intentional cleanup) or "
        "the path was mis-typed at emit time. Recover via git history."
    ),
    "broken_parent_ref": (
        "parent_event_ids points at an event_id that isn't in the "
        "log. Sometimes intentional (superseded ancestor removed); "
        "often a sign of a partial migration or a bug."
    ),
    "unregistered_subject": (
        "Event's subject_id doesn't resolve in subjects.yaml. "
        "The subjects registry is single-source-of-truth; unregistered "
        "subjects break registry-driven queries (UI, capability_gaps)."
    ),
}


def compute_report() -> dict:
    rep = audit_events()
    return {
        "reported_ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "n_events":    rep.n_events,
        "n_verdicts":  rep.n_verdicts,
        "n_issues":    rep.n_issues,
        "is_clean":    rep.is_clean,
        "by_category": rep.by_category,
        "sample_by_category": {
            cat: [
                {"event_id": iss.event_id, "detail": iss.detail}
                for iss in rep.issues if iss.category == cat
            ][:5]
            for cat in rep.by_category
        },
    }


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# research_store Events — Integrity Report")
    lines.append("")
    lines.append(f"_Generated: {report['reported_ts']}_")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    if report["is_clean"]:
        lines.append(f"- **CLEAN** — {report['n_events']} events, no issues.")
    else:
        lines.append(f"- **{report['n_issues']} issues** across "
                     f"{report['n_events']} events ("
                     f"{report['n_verdicts']} factor_verdict_filed).")
        lines.append("")
        lines.append("### Category breakdown")
        lines.append("")
        lines.append("| category | count |")
        lines.append("|---|---:|")
        for cat, n in sorted(report["by_category"].items(),
                              key=lambda x: -x[1]):
            lines.append(f"| `{cat}` | {n} |")

    lines.append("")
    lines.append("## Category explanations")
    lines.append("")
    for cat in sorted(report["by_category"].keys()):
        lines.append(f"### `{cat}`")
        lines.append("")
        lines.append(_CATEGORY_EXPLAIN.get(cat, "(no explanation available)"))
        lines.append("")
        samples = report["sample_by_category"].get(cat) or []
        if samples:
            lines.append(f"Sample (first {len(samples)}):")
            for s in samples:
                lines.append(f"- `{s['event_id'][:20]}` — {s['detail'][:140]}")
            lines.append("")

    if not report["is_clean"]:
        lines.append("## Remediation")
        lines.append("")
        lines.append("The audit is a diagnostic, not enforcement. Fixing "
                     "issues is human-in-the-loop:")
        lines.append("- **missing_evidence_doc**: safe to leave for legacy; "
                     "new verdicts are guarded by v15 emit contract")
        lines.append("- **evidence_doc_file_missing**: `git log --diff-filter=D "
                     "--follow docs/capability_evidence/<path>` to recover")
        lines.append("- **broken_parent_ref**: check if the ancestor was "
                     "intentionally superseded")
        lines.append("- **unregistered_subject**: add subject via "
                     "`engine.research_store.registry.register_subject`")
        lines.append("- **duplicate_event_id**: never expected — investigate "
                     "emit.py + shadow_emit.py callers")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",     action="store_true",
                     help="Print JSON to stdout instead of writing Markdown.")
    ap.add_argument("--out",      type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path, default=_DEFAULT_JSON)
    args = ap.parse_args()

    report = compute_report()

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    md = render_markdown(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")

    print(f"Wrote {args.out}")
    print(f"  n_events={report['n_events']} "
          f"n_issues={report['n_issues']} "
          f"is_clean={report['is_clean']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
