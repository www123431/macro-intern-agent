"""scripts/cron_hypothesis_spec_extract.py — daily hypothesis→spec cron.

Closes the pipeline chain identified in the 2026-07-01 audit:
  papers_curator (08:30) → hypotheses.jsonl
  ↓
  (this cron, 08:55) → hypothesis_specs.jsonl
  ↓
  burndown (09:00) → factor_verdict_filed events

Before v32, hypothesis→spec conversion (backfill_hypothesis_specs.py)
was manual. hypothesis_specs.jsonl stopped growing 2026-06-04. As a
result autopilot F14b saw "0 ready FACTOR_HYPOTHESIS specs" every
morning and produced no candidates for the dispatcher. Verdicts fell
off, then laptop-offline gap compounded the drought — last GREEN
was 2026-06-22, and none since.

This wrapper runs the extractor in idempotent skip-done mode with a
per-day cost cap so a huge backlog doesn't blow through the LLM
budget in one shot:

  - default --limit=20 → ~$0.10 per day
  - if no work → exit 0 immediately (no LLM cost)
  - health row emitted so AgentHealth tile picks it up

Extend the limit safely: this cron is fully idempotent (backfill
skips hypotheses whose spec already exists), so bumping --limit is
safe. The ceiling is the current-day LLM budget guarded by
engine.llm_budget.

Cadence: DAILY 08:55 SGT — between papers_curator (08:30) and
burndown (09:00) so the daily paper→hypothesis→spec→dispatch chain
completes in one morning cycle.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HEALTH_PATH = REPO_ROOT / "data" / "agents" / "_health" / "hypothesis_spec_extract.jsonl"

# v34 (2026-07-01): per-hypothesis budget. Empirically each spec
# extraction takes ~3-10 seconds on Sonnet 4.6 (a few seconds LLM
# call + JSON validation + persistence). 30s/hyp gives generous
# headroom for slow LLM responses without letting a runaway process
# eat the whole day. Multiplied by --limit + a 60s floor for tiny
# runs so subprocess.run() has time to start Python + import engine.
_TIMEOUT_SECONDS_PER_HYP: int = 30
_TIMEOUT_FLOOR_SECONDS:   int = 60


def _record(status: str, *, elapsed_s: float, to_extract: int = 0,
             skipped: int = 0, extracted: int = 0,
             error: str | None = None) -> None:
    HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "agent_id":     "hypothesis_spec_extract",
        "ts":           _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "status":       status,
        "elapsed_s":    round(elapsed_s, 2),
        "date_key":     _dt.date.today().isoformat(),
        "to_extract":   to_extract,
        "skipped":      skipped,
        "extracted":    extracted,
    }
    if error:
        row["error"] = error[:500]
    with HEALTH_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _count_ready_hypotheses() -> tuple[int, int]:
    """Pre-flight count: how many hypotheses are waiting for spec
    extraction? Returns (total_hyps, already_speced) so the wrapper
    can report expected work + exit early if nothing to do."""
    from engine.research_store.hypothesis import load_hypotheses
    from engine.hypothesis_spec.store import latest_for

    hyps_raw = load_hypotheses()
    latest_by_id: dict = {}
    for h in hyps_raw:
        prior = latest_by_id.get(h.hypothesis_id)
        if prior is None or h.version > prior.version:
            latest_by_id[h.hypothesis_id] = h

    total = len(latest_by_id)
    already = sum(
        1 for hid in latest_by_id if latest_for(hid) is not None
    )
    return total, already


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20,
                     help="max hypotheses to extract this run (cost cap; "
                          "default 20 = ~$0.10/day)")
    ap.add_argument("--dry-run", action="store_true",
                     help="pre-flight only — report expected work, do not "
                          "spend LLM cost")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = _dt.datetime.utcnow()

    try:
        total, already = _count_ready_hypotheses()
        to_extract = total - already
        if to_extract <= 0:
            elapsed = (_dt.datetime.utcnow() - t0).total_seconds()
            _record("ok", elapsed_s=elapsed, to_extract=0, skipped=already,
                     extracted=0)
            print(f"[cron_hypothesis_spec_extract] up-to-date "
                  f"({already}/{total} already spec'd, {elapsed:.1f}s)")
            return 0

        cap = min(to_extract, args.limit)
        print(f"[cron_hypothesis_spec_extract] "
              f"backlog={to_extract} (of {total} total) "
              f"→ extracting up to {cap} this run")

        if args.dry_run:
            elapsed = (_dt.datetime.utcnow() - t0).total_seconds()
            _record("dry_run", elapsed_s=elapsed, to_extract=cap,
                     skipped=already, extracted=0)
            return 0

        # v34: scale timeout with the cost cap. A 600s constant was
        # too short for --limit=222 (real 2026-07-01 run: 53 specs
        # extracted before TimeoutExpired at 601s; wrapper then
        # falsely reported extracted=0). Per-hyp budget grows the
        # ceiling linearly with work while keeping a floor for
        # subprocess/import overhead on tiny runs.
        timeout_seconds = max(_TIMEOUT_FLOOR_SECONDS,
                                cap * _TIMEOUT_SECONDS_PER_HYP)

        # Delegate to the existing backfill script, which owns the
        # LLM call + validation + persistence logic. We just add
        # cadence + cost cap + health telemetry.
        try:
            proc = subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts" /
                                       "backfill_hypothesis_specs.py"),
                 "--limit", str(cap)],
                capture_output=True, text=True,
                timeout=timeout_seconds,
            )
            exit_code = proc.returncode
            stderr    = proc.stderr or ""
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            # v34: partial work is still work. Record it accurately
            # rather than falsely reporting extracted=0 (as pre-v34
            # did). The backfill script is append-only, so anything
            # written before the kill lands on disk.
            exit_code = None
            stderr    = f"TimeoutExpired after {timeout_seconds}s"
            timed_out = True

        elapsed = (_dt.datetime.utcnow() - t0).total_seconds()

        # Always re-count after the subprocess — success, non-zero
        # exit, OR timeout. This is the "what actually happened"
        # signal, not "what did we ask for" (cap).
        _, new_already = _count_ready_hypotheses()
        extracted = new_already - already

        if timed_out:
            _record("timeout", elapsed_s=elapsed, to_extract=cap,
                     skipped=already, extracted=extracted,
                     error=(f"{stderr}; partial progress: {extracted} "
                              f"of {cap} extracted before kill"))
            print(f"[cron_hypothesis_spec_extract] "
                  f"TIMEOUT after {timeout_seconds}s — "
                  f"partial: {extracted}/{cap} extracted", file=sys.stderr)
            # Timeout with partial progress is treated as a WARNING
            # (exit 0) since the cron will pick up remaining work
            # tomorrow. Zero progress → exit 1 (something is really
            # broken).
            return 0 if extracted > 0 else 1

        if exit_code != 0:
            _record("error", elapsed_s=elapsed, to_extract=cap,
                     skipped=already, extracted=extracted,
                     error=stderr[:500])
            print(f"[cron_hypothesis_spec_extract] "
                  f"backfill exit={exit_code}", file=sys.stderr)
            print(stderr[:500], file=sys.stderr)
            return 1

        _record("ok", elapsed_s=elapsed, to_extract=cap,
                 skipped=already, extracted=extracted)
        print(f"[cron_hypothesis_spec_extract] "
              f"extracted={extracted}/{cap} in {elapsed:.1f}s")
        return 0
    except Exception as exc:
        elapsed = (_dt.datetime.utcnow() - t0).total_seconds()
        # Even on wrapper-side failure, re-check the store so we
        # don't undercount progress that landed before the crash.
        try:
            _, new_already = _count_ready_hypotheses()
            extracted = new_already - already
        except Exception:
            extracted = 0
        _record("error", elapsed_s=elapsed, extracted=extracted,
                 error=f"{type(exc).__name__}: {exc}")
        logger.exception("cron_hypothesis_spec_extract failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
