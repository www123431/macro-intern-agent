"""engine.agents.papers_curator.synthesis_runner — Phase 2.0 step 5a + 4c.

Thin orchestration over the 3 pieces shipped in steps 3 / 3b / 4b:

  build_synthesis_input()   →  run_synthesis()   →  write_synthesized_candidates()
       (gatherer)                  (LLM call)             (persistence)
                                                                 ↓
                                              emit.papers_curator_synthesis_run
                                                       (audit event, step 4c)

Lives in engine/ NOT scripts/ so it's reusable from:

  - scripts/run_papers_curator_synthesis.py  (CLI / cron entry)
  - api/routes_papers_curator.py             (UI button — step 5b)
  - chief_of_staff orchestrator              (step 14, later)

Returns a structured dict (not the raw dataclasses) so the JSON API
+ UI can render it directly. Result shape is the contract.

Step 4c (2026-06-06): after each run, emit a papers_curator_synthesis_run
event. The event captures generation metadata the hypotheses.jsonl
writer drops (cochrane_frame / novelty / conflicts / prior) + the
snapshot the LLM read + the cost/latency facts. Failure to emit is
non-fatal — the result dict still returns the candidates; emit errors
land in result["errors"] like any other partial failure.
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import hashlib as _hashlib
import json as _json
import logging
from pathlib import Path
from typing import Optional

from engine.agents.papers_curator.synthesis import run_synthesis, SynthesizedCandidate
from engine.agents.papers_curator.synthesis_context import build_synthesis_input
from engine.agents.papers_curator.synthesis_writer import write_synthesized_candidates

logger = logging.getLogger(__name__)


# ── v31 (2026-07-01): Sonnet-skip when nothing changed ────────────────


def _synthesis_input_hash(si) -> str:
    """Deterministic content hash of a SynthesisInput. Excludes
    snapshot_ts (which changes every run — including it would defeat
    the whole cache). Includes every content-bearing field, so any
    genuine change (new paper summary, new event, new doctrine hit,
    new sleeve status, new anchor, new belief-family row) shifts the
    hash → Sonnet gets called on real change; skipped on redundant
    daily re-runs.

    Determinism verified: two consecutive build_synthesis_input()
    invocations on the same corpus yield the same 64-char sha256.
    """
    parts = {
        "summaries":   [_dc.asdict(s) for s in si.recent_summaries],
        "sleeves":     [_dc.asdict(s) for s in si.deployed_sleeves],
        "events":      [_dc.asdict(s) for s in si.recent_events],
        "doctrine":    [_dc.asdict(d) for d in si.doctrine_snippets],
        "anchors":     [_dc.asdict(a) for a in si.anchor_library],
        # belief_layer_summary is a tuple of plain dicts/tuples per
        # build_belief_summary — pass through as-is
        "belief":      list(si.belief_layer_summary),
    }
    blob = _json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return _hashlib.sha256(blob).hexdigest()


def _last_synthesis_input_hash() -> Optional[str]:
    """Look up the input_hash on the most recent
    papers_curator_synthesis_run event. Returns None if none exists
    or none carried the field (pre-v31 events, or first-ever run).

    Failure-safe: any exception (store missing, corrupted rows,
    permission error) returns None → v31 falls back to running Sonnet.
    Never blocks the pipeline on this lookup."""
    try:
        from engine.research_store import store
        evs = store.filter_events(
            event_type = "papers_curator_synthesis_run",
            limit      = 1,
        )
        if not evs:
            return None
        ev = evs[-1]
        metrics = getattr(ev, "metrics", None) or {}
        hv = metrics.get("input_hash")
        return str(hv) if hv else None
    except Exception:
        return None


def _candidate_to_payload(c: SynthesizedCandidate) -> dict:
    """Hand-rolled because asdict() returns Tuple[str,...] as list which
    is what we want; just keeps the spelling consistent + drops nothing."""
    return _dc.asdict(c)


def _enrich_with_citation_checks(
    candidates: list[SynthesizedCandidate],
) -> tuple[list[SynthesizedCandidate], list[str]]:
    """Phase 2.2b: after run_synthesis returns, verify each candidate's
    cited papers via citation_verifier. Returns (enriched_candidates,
    errors). Failures on individual candidates degrade the candidate
    to citation_quality=None + carry an error — do NOT drop the candidate."""
    if not candidates:
        return [], []
    try:
        from engine.agents.papers_curator.citation_verifier import (
            verify_citations, aggregate_citation_quality,
        )
    except Exception as exc:
        return list(candidates), [f"citation_verifier_import: {exc}"]

    errors: list[str] = []
    enriched: list[SynthesizedCandidate] = []
    for c in candidates:
        try:
            checks = verify_citations(
                claim     = c.claim,
                paper_ids = tuple(c.synthesizes_paper_ids),
            )
            quality = aggregate_citation_quality(checks)
            enriched.append(_dc.replace(
                c,
                citation_verifications = checks,
                citation_quality       = quality,
            ))
        except Exception as exc:
            logger.exception("citation_verify failed for candidate '%s'",
                              c.claim[:60])
            errors.append(f"citation:{c.claim[:60]}: {exc}")
            # Keep the candidate unchanged — better to write the
            # un-verified row than to drop it
            enriched.append(c)
    return enriched, errors


def run_synthesis_pipeline(
    *,
    dry_run: bool = False,
    summaries_days: int = 14,
    events_days: int = 30,
    created_by: str = "papers_curator_synthesis",
    extra_tags: tuple[str, ...] = (),
    hypotheses_path: Optional[Path] = None,
) -> dict:
    """End-to-end: gather → synthesize → persist.

    Args:
      dry_run:           if True, skip the writer step (still runs the
                         LLM call). Use for "preview what synthesis
                         would propose" UI calls.
      summaries_days:    recency window for paper summaries (default 14).
      events_days:       recency window for events (default 30).
      created_by:        actor tag on persisted hypotheses.
      extra_tags:        appended to ("synthesis",) on persisted rows.
      hypotheses_path:   override the jsonl path (tests).

    Returns:
      {
        "run_ts":              ISO-8601 UTC,
        "dry_run":             bool,
        "snapshot": {
          "snapshot_ts":         iso,
          "recent_summaries":    int,
          "deployed_sleeves":    int,
          "recent_events":       int,
          "doctrine_snippets":   int,
        },
        "candidates":          [SynthesizedCandidate-as-dict, ...],
        "n_candidates":        int,
        "written_hypothesis_ids": [str, ...],   # empty when dry_run
        "n_written":           int,
        "errors":              [str, ...],      # graceful — never raises
        "event_id":            str | None,      # step 4c: audit event_id
                                                # (None if emit failed or
                                                # snapshot stage failed before
                                                # the emit point)
      }

    Fail-safe: any exception in synthesize or write is caught,
    appended to errors, and a partial result is returned. The caller
    (cron / API / orchestrator) decides what to do with the errors.
    """
    run_ts = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    result: dict = {
        "run_ts":                run_ts,
        "dry_run":               dry_run,
        "snapshot":              {},
        "candidates":            [],
        "n_candidates":          0,
        "written_hypothesis_ids": [],
        "n_written":             0,
        "errors":                [],
        "event_id":              None,   # step 4c
    }

    try:
        si = build_synthesis_input(
            summaries_days = summaries_days,
            events_days    = events_days,
        )
        result["snapshot"] = {
            "snapshot_ts":       si.snapshot_ts,
            "recent_summaries":  len(si.recent_summaries),
            "deployed_sleeves":  len(si.deployed_sleeves),
            "recent_events":     len(si.recent_events),
            "doctrine_snippets": len(si.doctrine_snippets),
        }
        # v31 (2026-07-01): input_hash lets the next run know whether
        # the SI content has actually changed, so we can skip Sonnet
        # (~$0.05/call) on a redundant no-change cron. Deterministic
        # across the SI content minus snapshot_ts. See
        # _synthesis_input_hash docstring for the fields hashed.
        current_input_hash = _synthesis_input_hash(si)
        result["snapshot"]["input_hash"] = current_input_hash
        # Capture doctrine snippet IDs A actually saw — feeds the
        # Layer 4 attribution rollup ("which doctrine entries lead to
        # GREEN candidates?"). See attribution module.
        doctrine_snippet_ids = [
            getattr(d, "memory_file_id", "")
            for d in (si.doctrine_snippets or ())
            if getattr(d, "memory_file_id", "")
        ]
        result["doctrine_snippet_ids"] = doctrine_snippet_ids
    except Exception as exc:
        logger.exception("synthesis_runner: gather failed")
        result["errors"].append(f"gather: {exc}")
        return result

    # v31: skip Sonnet if the input hash matches the last emitted run.
    # The audit event still fires (n_candidates=0 + skipped_reason)
    # so ops can see "A ran; input unchanged, no LLM call". Real work
    # happens as soon as summaries.jsonl / events.jsonl grows.
    last_hash = _last_synthesis_input_hash()
    if last_hash and last_hash == current_input_hash:
        result["skipped_reason"] = "input_unchanged_since_last_run"
        # Stash skipped_reason into snapshot dict so the emit call
        # persists it into event metrics (see emit.papers_curator_
        # synthesis_run v31 pass-through fields).
        result["snapshot"]["skipped_reason"] = result["skipped_reason"]
        # Emit the audit event so the run is still visible to
        # ops_watchdog + observability (step 4c invariant preserved).
        try:
            from engine.research_store import emit
            event_id = emit.papers_curator_synthesis_run(
                n_candidates        = 0,
                n_written           = 0,
                snapshot            = result["snapshot"],
                candidates          = [],
                errors              = [],
                dry_run             = dry_run,
                doctrine_snippet_ids= result.get("doctrine_snippet_ids", []),
            )
            result["event_id"] = event_id
        except Exception as exc:
            logger.exception("synthesis_runner: emit on skip failed")
            result["errors"].append(f"emit_on_skip: {exc}")
        logger.info(
            "synthesis_runner: skipped Sonnet (input_hash=%s unchanged); "
            "emit event_id=%s", current_input_hash[:12], result["event_id"],
        )
        return result

    try:
        candidates = run_synthesis(si)
    except Exception as exc:
        # run_synthesis() is itself fail-safe (returns []) but belt-
        # and-braces in case a future refactor breaks that contract.
        logger.exception("synthesis_runner: run_synthesis raised")
        result["errors"].append(f"synthesize: {exc}")
        return result

    # Phase 2.2b: enrich each candidate with citation_verifications +
    # aggregate citation_quality. Failures degrade individual
    # candidates but never drop them.
    candidates, citation_errors = _enrich_with_citation_checks(candidates)
    for e in citation_errors:
        result["errors"].append(e)

    result["candidates"]   = [_candidate_to_payload(c) for c in candidates]
    result["n_candidates"] = len(candidates)

    if not dry_run and candidates:
        try:
            written = write_synthesized_candidates(
                candidates,
                created_by      = created_by,
                extra_tags      = extra_tags,
                path            = hypotheses_path,
                validate_strict = False,  # don't kill a batch over one bad row
            )
            result["written_hypothesis_ids"] = [h.hypothesis_id for h in written]
            result["n_written"]              = len(written)
        except Exception as exc:
            logger.exception("synthesis_runner: writer failed")
            result["errors"].append(f"write: {exc}")

    # Step 4c: audit-trail event. Always attempt, even on dry_run /
    # empty candidates — those outcomes ARE the audit data ("A ran
    # this week, returned 0", "A ran, proposed 2, conflicts blocked
    # 1"). Failure here is non-fatal — emit errors land in result
    # errors[] like any other partial failure.
    try:
        from engine.research_store import emit
        event_id = emit.papers_curator_synthesis_run(
            n_candidates         = result["n_candidates"],
            n_written            = result["n_written"],
            snapshot             = result["snapshot"],
            candidates           = result["candidates"],
            errors               = result["errors"],
            dry_run              = dry_run,
            doctrine_snippet_ids = result.get("doctrine_snippet_ids", []),
        )
        result["event_id"] = event_id
    except Exception as exc:
        logger.exception("synthesis_runner: emit failed")
        result["errors"].append(f"emit: {exc}")

    return result
