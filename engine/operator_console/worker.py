"""Async station execution worker + in-process SSE event bus.

Per design doc D2: SSE for live progress. Per D1: async execution
via existing workflow_executor pattern; this module is the worker
that pulls a triggered job, instantiates the station, runs execute()
with an SSEEmitter that pushes events to a per-job queue, then marks
the job terminal.

In-process queue model:
    JOB_QUEUES: dict[job_id, asyncio.Queue]
    Each SSE endpoint subscriber dequeues events for its job_id.
    Worker pushes; subscriber pulls; queue is created lazily on the
    worker side (`get_or_create_queue`). Subscribers MUST use
    `subscribe_queue` instead, which is query-only — see "Orphan
    queue leak" below.

Server-restart caveat (R6): jobs running when uvicorn restarts are
orphaned — queue lost, worker died. routes_operator_console restart
scan marks them RECOVERED_UNKNOWN. Out-of-scope to persist queue
state in MVP; see docs/architecture/operator_console.md Risk #5.

Orphan queue leak (v18 fix, 2026-06-28):
    Pre-v18, SSE subscribers called `get_or_create_queue` directly.
    If a subscriber connected AFTER the worker finished + called
    `cleanup_job`, the call would create a NEW empty queue that no
    producer would ever write to. Subscriber would timeout at 60s,
    check terminal state, return — but the orphan queue stayed in
    JOB_QUEUES forever (small leak per late subscriber).

    v18 splits the accessor: workers use `get_or_create_queue`
    (creates on demand), subscribers use `subscribe_queue` (returns
    None if absent). When subscribe_queue returns None, SSE
    endpoint emits terminal snapshot directly and exits without
    ever creating a queue. Belt-and-suspenders: `sweep_stale_queues`
    periodically removes any queue whose corresponding job has been
    terminal for > 5 minutes.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from engine.operator_console import emit as opcon_emit
from engine.operator_console import registry, store
from engine.operator_console.pipeline_station import SSEEmitter
from engine.operator_console.schema import (
    CancellationToken,
    JobState,
    StationResult,
)


logger = logging.getLogger(__name__)


# ── Per-job queue + cancellation registry ────────────────────────


# job_id → asyncio.Queue of SSE event dicts
JOB_QUEUES: dict[str, asyncio.Queue[dict[str, Any]]] = {}

# job_id → CancellationToken (so /cancel API can flip the flag)
JOB_CANCELLATIONS: dict[str, CancellationToken] = {}


# Job states that mean "worker has finished, nothing more will be
# pushed to this queue". Sweep uses this to identify candidates for
# eviction. RUNNING / QUEUED stay protected because the worker is
# still (or may still be) writing.
_TERMINAL_STATES: frozenset[str] = frozenset({
    JobState.COMPLETED.value,
    JobState.FAILED.value,
    JobState.CANCELLED.value,
    JobState.HALTED_COST_CAP.value,
    JobState.RECOVERED_UNKNOWN.value,
})


# Default grace period before a terminal-state job's queue is swept.
# Long enough for a slow SSE subscriber that connected RIGHT as the
# worker finished to drain any pre-cleanup events; short enough that
# we don't pile up dead queues in a long-running process.
SWEEP_GRACE_SECONDS: int = 300                    # 5 min


def get_or_create_queue(job_id: str) -> asyncio.Queue[dict[str, Any]]:
    """Worker-side accessor. Lazy queue creation — worker calls this
    when starting a job, so the queue exists before any subscriber
    can hit `subscribe_queue` and pull events.

    Subscribers MUST use `subscribe_queue` instead. See module
    docstring "Orphan queue leak" for why this matters.
    """
    q = JOB_QUEUES.get(job_id)
    if q is None:
        q = asyncio.Queue(maxsize=200)
        JOB_QUEUES[job_id] = q
    return q


def subscribe_queue(job_id: str) -> Optional[asyncio.Queue[dict[str, Any]]]:
    """Subscriber-side accessor. Returns the existing queue if the
    worker is still around to push events; returns None if no queue
    exists (worker already finished + cleaned up, or never started).

    Subscribers MUST handle the None case by snapshotting the
    terminal job state and exiting — they should NOT create an
    orphan queue that no producer will ever write to.
    """
    return JOB_QUEUES.get(job_id)


def sweep_stale_queues(*, grace_seconds: int = SWEEP_GRACE_SECONDS,
                        now: Optional[_dt.datetime] = None) -> int:
    """Belt-and-suspenders sweep: drop queues whose corresponding job
    has been in a terminal state for longer than `grace_seconds`.

    Defensive against:
      - Subscribers that bypass `subscribe_queue` and call
        `get_or_create_queue` directly (legacy code paths)
      - Worker crashes between writing terminal state and calling
        `cleanup_job` (cleanup_job is called from a finally block
        so this should be rare, but the sweep catches it anyway)
      - Race conditions where a subscriber connects RIGHT as the
        worker finishes — its queue reference stays alive in the
        subscriber's coroutine until that coroutine returns; we
        give the grace period for the coroutine to drain & exit

    Returns: number of queues evicted.
    """
    if now is None:
        now = _dt.datetime.now(_dt.timezone.utc)
    cutoff = now - _dt.timedelta(seconds=grace_seconds)

    evicted = 0
    # Snapshot keys so we can mutate dict while iterating
    for job_id in list(JOB_QUEUES.keys()):
        job = store.get_job(job_id)
        if job is None:
            # Job row vanished from store — definitely stale
            JOB_QUEUES.pop(job_id, None)
            JOB_CANCELLATIONS.pop(job_id, None)
            evicted += 1
            continue

        state = str(job.get("state") or "")
        if state not in _TERMINAL_STATES:
            continue  # worker may still be writing

        # Parse updated_ts — schema uses isoformat with trailing 'Z'
        ts_raw = str(job.get("updated_ts") or "").rstrip("Z")
        if not ts_raw:
            # Terminal state but no timestamp — sweep conservatively
            JOB_QUEUES.pop(job_id, None)
            JOB_CANCELLATIONS.pop(job_id, None)
            evicted += 1
            continue
        try:
            ts = _dt.datetime.fromisoformat(ts_raw)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_dt.timezone.utc)
        except ValueError:
            logger.warning("worker.sweep: unparseable updated_ts=%r for job=%s",
                            ts_raw, job_id)
            continue

        if ts < cutoff:
            JOB_QUEUES.pop(job_id, None)
            JOB_CANCELLATIONS.pop(job_id, None)
            evicted += 1

    if evicted:
        logger.info("worker.sweep: evicted %d stale queue(s) "
                    "(grace=%ds, remaining=%d)",
                    evicted, grace_seconds, len(JOB_QUEUES))
    return evicted


async def periodic_sweep_loop(*, interval_seconds: int = 60,
                               grace_seconds: int = SWEEP_GRACE_SECONDS) -> None:
    """Long-running coroutine that calls `sweep_stale_queues` every
    `interval_seconds`. Designed to be spawned via
    `asyncio.create_task` from the FastAPI lifespan startup hook.

    Exits gracefully on CancelledError (lifespan shutdown).
    """
    logger.info("worker.sweep: starting periodic_sweep_loop "
                "(interval=%ds, grace=%ds)", interval_seconds, grace_seconds)
    try:
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                sweep_stale_queues(grace_seconds=grace_seconds)
            except Exception as e:  # noqa: BLE001 — never let sweep crash lifespan
                logger.exception("worker.sweep: tick failed: %s", e)
    except asyncio.CancelledError:
        logger.info("worker.sweep: periodic_sweep_loop cancelled (shutdown)")
        raise


def get_or_create_cancellation(job_id: str) -> CancellationToken:
    tok = JOB_CANCELLATIONS.get(job_id)
    if tok is None:
        tok = CancellationToken()
        JOB_CANCELLATIONS[job_id] = tok
    return tok


def request_cancellation(job_id: str) -> bool:
    """Flip the cancellation flag for an in-flight job. Returns True
    if a token existed (job was active), False if not (job already
    terminal or never started). Honored at next stage boundary per R3."""
    tok = JOB_CANCELLATIONS.get(job_id)
    if tok is None:
        return False
    tok.cancel()
    return True


def cleanup_job(job_id: str) -> None:
    """Drop the queue + cancellation token after the job reaches
    terminal state. Called by worker after marking job done."""
    JOB_QUEUES.pop(job_id, None)
    JOB_CANCELLATIONS.pop(job_id, None)


# ── SSE emitter (worker-side) ────────────────────────────────────


@dataclass
class QueueSSEEmitter:
    """Pushes events into the per-job queue. Implements the SSEEmitter
    Protocol used by station.execute().

    Each named method maps to an SSE event type that the frontend
    StationProgressStream consumer understands."""

    job_id: str
    queue:  asyncio.Queue[dict[str, Any]]

    def _put(self, event: str, payload: dict) -> None:
        try:
            self.queue.put_nowait({"event": event, "data": json.dumps(payload, ensure_ascii=False)})
        except asyncio.QueueFull:
            logger.warning("worker: SSE queue full for job_id=%s; dropping event=%s",
                           self.job_id, event)

    def stage_started(self, stage: str, expected_seconds: int = 0) -> None:
        self._put("stage_started", {
            "stage":            stage,
            "expected_seconds": expected_seconds,
        })

    def stage_progress(self, stage: str, pct: int, current: str = "") -> None:
        self._put("stage_progress", {
            "stage":   stage,
            "pct":     pct,
            "current": current,
        })

    def stage_completed(self, stage: str, result: dict[str, Any]) -> None:
        self._put("stage_completed", {
            "stage":  stage,
            "result": result,
        })

    def stage_failed(self, stage: str, error: str) -> None:
        self._put("stage_failed", {
            "stage": stage,
            "error": error,
        })

    def log_line(self, line: str) -> None:
        self._put("log", {"line": line})

    def terminal(self, state: str) -> None:
        """Emit the job_terminal SSE event signalling the stream is
        done. Public hook so callers (run_job) don't reach into _put."""
        self._put("job_terminal", {"job_id": self.job_id, "state": state})


# ── Worker ───────────────────────────────────────────────────────


@dataclass
class _SessionView:
    """Minimal Session struct passed to station.execute(). Real
    sessions API returns rich SessionRow; for now we pass the
    session_id + actor_id + a (best-effort) type."""
    session_id:   str
    session_type: str = ""
    actor_id:     str = "principal"


async def run_job(job_id: str) -> None:
    """Worker entry point. Looks up job, instantiates station, runs
    execute() with queued SSE emitter, marks terminal state.

    Called from the trigger endpoint via FastAPI BackgroundTasks
    (or asyncio.create_task). Returns when the job reaches a
    terminal state — no exceptions propagate (all caught + recorded
    as job_failed)."""
    job = store.get_job(job_id)
    if job is None:
        logger.error("worker: job_id=%s not found", job_id)
        return

    station_cls = registry.get(job["station_id"])
    if station_cls is None:
        store.update_job_state(job_id, state=JobState.FAILED,
                               error=f"station '{job['station_id']}' not registered at execute time")
        cleanup_job(job_id)
        return

    queue = get_or_create_queue(job_id)
    cancellation = get_or_create_cancellation(job_id)
    emitter = QueueSSEEmitter(job_id=job_id, queue=queue)
    session_view = _SessionView(
        session_id   = job.get("session_id", ""),
        actor_id     = job.get("actor_id", "principal"),
    )

    station = station_cls()
    try:
        result: StationResult = await station.execute(
            session      = session_view,
            config       = job.get("config", {}),
            emitter      = emitter,
            cancellation = cancellation,
        )
    except Exception as e:
        logger.exception("worker: job_id=%s station.execute() raised", job_id)
        emitter.stage_failed("execute", str(e)[:300])
        store.update_job_state(job_id, state=JobState.FAILED, error=str(e)[:1000])
        try:
            opcon_emit.station_failed(
                session_id   = session_view.session_id,
                actor_id     = session_view.actor_id,
                job_id       = job_id,
                station_id   = job["station_id"],
                stage_failed = "execute",
                error        = str(e),
            )
        except Exception:
            logger.exception("worker: failed to emit station_failed event")
        # Final terminal SSE event so subscribers close cleanly
        emitter.terminal(JobState.FAILED.value)
        cleanup_job(job_id)
        return

    # Determine terminal state from result
    if cancellation.cancelled and not result.success:
        terminal = JobState.CANCELLED
    elif result.success:
        terminal = JobState.COMPLETED
    else:
        terminal = JobState.FAILED

    store.update_job_state(job_id, state=terminal, result=result,
                           error=result.error_message or None)

    emitter.terminal(terminal.value)

    # Defer cleanup briefly so any in-flight SSE subscriber drains
    await asyncio.sleep(0.1)
    cleanup_job(job_id)
