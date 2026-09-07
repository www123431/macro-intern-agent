"""Worker queue-leak regression tests (v18 fix, 2026-06-28).

Pre-v18, `stream_job` SSE endpoint called `get_or_create_queue` which
created an orphan queue when subscriber connected AFTER worker had
cleaned up. The orphan queue had no producer and no eviction path,
causing a small per-late-subscriber memory leak.

These tests lock the v18 fix in:
  - subscribe_queue returns None for jobs without an active queue
  - subscribe_queue returns existing queue when worker is around
  - sweep_stale_queues evicts queues whose job has been terminal for
    longer than the grace period
  - sweep_stale_queues leaves running / recently-finished queues alone
  - sweep_stale_queues handles edge cases (missing job, bad timestamp)
"""
from __future__ import annotations

import asyncio
import datetime as _dt

import pytest

from engine.operator_console import worker, store
from engine.operator_console.schema import JobState


@pytest.fixture(autouse=True)
def _reset_worker_state():
    """Worker module-level dicts must be empty at the start of each
    test so cases don't bleed into each other."""
    worker.JOB_QUEUES.clear()
    worker.JOB_CANCELLATIONS.clear()
    yield
    worker.JOB_QUEUES.clear()
    worker.JOB_CANCELLATIONS.clear()


# ── subscribe_queue ──────────────────────────────────────────────


def test_subscribe_queue_returns_none_for_unknown_job():
    """v18 contract: subscriber MUST NOT create a queue. Bypassing
    this would re-introduce the orphan leak."""
    assert worker.subscribe_queue("nonexistent_job_id") is None
    # Critical: confirm the call did NOT create an entry
    assert "nonexistent_job_id" not in worker.JOB_QUEUES


def test_subscribe_queue_returns_existing_queue():
    """When the worker has called get_or_create_queue, a subsequent
    subscribe_queue call returns the SAME queue instance."""
    job_id = "job_abc123"
    q_worker = worker.get_or_create_queue(job_id)
    q_subscriber = worker.subscribe_queue(job_id)
    assert q_subscriber is q_worker


def test_subscribe_queue_after_cleanup_returns_none():
    """The exact pre-v18 leak scenario: worker creates queue, runs
    job to terminal, cleanup_job drops it. Late subscriber arrives —
    must get None, NOT a freshly-created orphan."""
    job_id = "job_leak_repro"
    worker.get_or_create_queue(job_id)  # worker side
    worker.cleanup_job(job_id)          # terminal cleanup
    assert worker.subscribe_queue(job_id) is None
    # Critical: dict stays empty
    assert job_id not in worker.JOB_QUEUES


# ── sweep_stale_queues ───────────────────────────────────────────


def _make_terminal_job_row(job_id: str, state: str, updated_ts: str,
                            tmp_jobs_path):
    """Helper: write a job row to a tmp jobs.jsonl path that the
    worker's store.get_job will read from. Returns the path."""
    import json
    with tmp_jobs_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "schema_version": "1.0.0",
            "job_id":         job_id,
            "state":          state,
            "updated_ts":     updated_ts,
        }) + "\n")
    return tmp_jobs_path


def test_sweep_evicts_long_terminal_queues(tmp_path, monkeypatch):
    """Job terminal > grace seconds → queue evicted."""
    jobs_path = tmp_path / "jobs.jsonl"
    monkeypatch.setattr(store, "_JOBS_PATH", jobs_path)

    job_id = "job_long_done"
    worker.get_or_create_queue(job_id)
    old_ts = "2026-06-28T00:00:00.000000"  # very old
    _make_terminal_job_row(job_id, JobState.COMPLETED.value, old_ts, jobs_path)

    n = worker.sweep_stale_queues(grace_seconds=300)
    assert n == 1
    assert job_id not in worker.JOB_QUEUES


def test_sweep_preserves_running_jobs(tmp_path, monkeypatch):
    """Job RUNNING → queue preserved no matter the timestamp.
    Worker may still be writing."""
    jobs_path = tmp_path / "jobs.jsonl"
    monkeypatch.setattr(store, "_JOBS_PATH", jobs_path)

    job_id = "job_still_running"
    worker.get_or_create_queue(job_id)
    very_old_ts = "2025-01-01T00:00:00.000000"
    _make_terminal_job_row(job_id, JobState.RUNNING.value, very_old_ts, jobs_path)

    n = worker.sweep_stale_queues(grace_seconds=300)
    assert n == 0
    assert job_id in worker.JOB_QUEUES


def test_sweep_preserves_recently_terminal_queues(tmp_path, monkeypatch):
    """Job COMPLETED but only 30s ago → still within grace, preserved.
    Protects slow subscribers that connected RIGHT as worker finished."""
    jobs_path = tmp_path / "jobs.jsonl"
    monkeypatch.setattr(store, "_JOBS_PATH", jobs_path)

    now = _dt.datetime.now(_dt.timezone.utc)
    recent = (now - _dt.timedelta(seconds=30)).isoformat()

    job_id = "job_just_done"
    worker.get_or_create_queue(job_id)
    _make_terminal_job_row(job_id, JobState.COMPLETED.value, recent, jobs_path)

    n = worker.sweep_stale_queues(grace_seconds=300, now=now)
    assert n == 0
    assert job_id in worker.JOB_QUEUES


def test_sweep_evicts_when_job_row_missing(tmp_path, monkeypatch):
    """Queue exists but corresponding job row vanished from store
    (e.g. data dir wipe / test fixture mismatch) — evict."""
    jobs_path = tmp_path / "jobs.jsonl"
    monkeypatch.setattr(store, "_JOBS_PATH", jobs_path)

    job_id = "job_ghost"
    worker.get_or_create_queue(job_id)
    worker.get_or_create_cancellation(job_id)
    # Don't write a row at all — store.get_job returns None
    n = worker.sweep_stale_queues(grace_seconds=300)
    assert n == 1
    assert job_id not in worker.JOB_QUEUES
    assert job_id not in worker.JOB_CANCELLATIONS


def test_sweep_handles_terminal_states_completed_failed_cancelled(
    tmp_path, monkeypatch
):
    """All 5 terminal states (per _TERMINAL_STATES) should be sweepable."""
    jobs_path = tmp_path / "jobs.jsonl"
    monkeypatch.setattr(store, "_JOBS_PATH", jobs_path)

    old_ts = "2026-06-28T00:00:00.000000"
    states = [
        JobState.COMPLETED.value,
        JobState.FAILED.value,
        JobState.CANCELLED.value,
        JobState.HALTED_COST_CAP.value,
        JobState.RECOVERED_UNKNOWN.value,
    ]
    for i, state in enumerate(states):
        jid = f"job_{state}_{i}"
        worker.get_or_create_queue(jid)
        _make_terminal_job_row(jid, state, old_ts, jobs_path)

    n = worker.sweep_stale_queues(grace_seconds=300)
    assert n == len(states)
    assert len(worker.JOB_QUEUES) == 0


def test_sweep_evicts_when_timestamp_unparseable(tmp_path, monkeypatch):
    """Terminal state + garbage timestamp → conservative evict."""
    jobs_path = tmp_path / "jobs.jsonl"
    monkeypatch.setattr(store, "_JOBS_PATH", jobs_path)

    job_id = "job_bad_ts"
    worker.get_or_create_queue(job_id)
    _make_terminal_job_row(job_id, JobState.COMPLETED.value, "", jobs_path)

    n = worker.sweep_stale_queues(grace_seconds=300)
    assert n == 1


def test_sweep_returns_zero_when_dict_empty(tmp_path, monkeypatch):
    jobs_path = tmp_path / "jobs.jsonl"
    monkeypatch.setattr(store, "_JOBS_PATH", jobs_path)
    assert worker.sweep_stale_queues() == 0


# ── periodic_sweep_loop ──────────────────────────────────────────


def test_periodic_sweep_loop_cancels_cleanly():
    """Lifespan shutdown sends CancelledError — loop should swallow
    + re-raise (per asyncio convention) without leaving the worker
    state corrupted. Driven via asyncio.run so we don't need
    pytest-asyncio."""
    async def _drive():
        task = asyncio.create_task(
            worker.periodic_sweep_loop(interval_seconds=1, grace_seconds=300)
        )
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_drive())
