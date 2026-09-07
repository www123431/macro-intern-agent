"""Tests for v30 (2026-07-01) filter-dedup fix in
scripts/papers_curator_daily.py._step_filter.

Locks the invariant that a paper already in summaries.jsonl is NEVER
re-judged, even when its FilterJudgment row is missing from
judgments.jsonl (the "orphan summary" legacy artifact from pre-v30
backfills). Uses tmp paths + a lightweight monkeypatched cache so no
LLM calls happen during tests.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Deferred import so monkeypatching module-level globals works cleanly
import scripts.papers_curator_daily as pcd
from engine.agents.papers_curator import summarizer as _summarizer_module
from engine.agents.papers_curator import judgments_store as _jstore
from engine.agents.papers_curator import summaries_store as _sstore
from engine.agents.papers_curator.crawler import PaperCandidate
from engine.agents.papers_curator.filter import FilterJudgment


# ── Fixture wiring ───────────────────────────────────────────────────


@pytest.fixture
def tmp_stores(tmp_path, monkeypatch):
    """Redirect all three papers-curator jsonl stores to tmp files.
    Also stub load_cache so we control the candidate pool without
    touching the real crawler cache."""
    j_path = tmp_path / "judgments.jsonl"
    s_path = tmp_path / "summaries.jsonl"

    monkeypatch.setattr(_jstore, "JUDGMENTS_PATH", j_path)
    monkeypatch.setattr(_sstore, "SUMMARIES_PATH", s_path)

    return {"judgments": j_path, "summaries": s_path}


def _mk_candidate(source: str, source_id: str,
                   title: str = "test paper") -> PaperCandidate:
    return PaperCandidate(
        source        = source,
        source_id     = source_id,
        title         = title,
        authors       = (),
        abstract      = "abstract",
        abs_url       = "http://example.com/x",
        pdf_url       = "http://example.com/x.pdf",
        published_ts  = "2026-07-01T00:00:00Z",
        categories    = (),
        fetched_ts    = "2026-07-01T00:00:00Z",
    )


def _write_judgment(path: Path, *, source: str, source_id: str,
                      is_yes: bool, judged_ts: str = "2026-06-14T00:00:00Z"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "source":             source,
            "source_id":          source_id,
            "is_tradable_factor": is_yes,
            "confidence":         0.9,
            "one_line_reason":    "test",
            "category_guess":     "new_factor" if is_yes else "off_topic",
            "judged_ts":          judged_ts,
            "model":              "deepseek-v4-pro",
            "raw_response":       "{}",
        }) + "\n")


def _write_summary(path: Path, *, source: str, source_id: str,
                     summarized_ts: str = "2026-06-14T00:00:00Z"):
    """Write a minimum-viable summary row (all fields the loader needs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "source":             source,
            "source_id":          source_id,
            "thesis":             "test thesis",
            "mechanism":          "test mechanism",
            "testable_hypothesis":"test H",
            "why_matters_for_us": "why",
            "risk_flags":         [],
            "recommended_action": "INGEST",
            "triggered_by":       "auto_yes",
            "summarized_ts":      summarized_ts,
            "model":              "deepseek-v4-pro",
            "raw_response":       "{}",
        }) + "\n")


# ── The core v30 fix ────────────────────────────────────────────────


def test_paper_in_summaries_but_missing_judgment_is_skipped(monkeypatch, tmp_stores):
    """The exact 2026-07-01 scenario: 2 papers are in summaries.jsonl
    with triggered_by=auto_yes but have no judgment row. Pre-v30
    would re-judge them wasting DeepSeek cost. Post-v30 they must
    be filtered from `unjudged`."""
    c_orphan = _mk_candidate("ssrn", "10.2139/ssrn.6076487",
                              "orphan summarized paper")
    c_fresh  = _mk_candidate("arxiv", "2026.99999", "truly-new paper")

    # Orphan: in summaries, NOT in judgments
    _write_summary(tmp_stores["summaries"], source="ssrn",
                    source_id="10.2139/ssrn.6076487")
    # (no _write_judgment call for it — that's the whole point)

    monkeypatch.setattr(pcd, "_step_filter", pcd._step_filter)  # ensure fresh
    # Stub load_cache to return our two candidates
    def _fake_load_cache():
        return [c_orphan, c_fresh]
    import engine.agents.papers_curator as pc_mod
    monkeypatch.setattr(pc_mod, "load_cache", _fake_load_cache)

    r = pcd._step_filter(max_filter=0)  # 0 = never actually spend LLM

    assert r["unjudged_total"] == 1, (
        f"expected only c_fresh in unjudged, got total={r['unjudged_total']}"
    )
    assert r["orphan_summaries"] == 1
    assert r["judged_now"] == 0    # max_filter=0 → no LLM calls


def test_paper_in_both_judgments_and_summaries_still_skipped(monkeypatch, tmp_stores):
    """Sanity: a properly-processed paper (both rows present) also stays
    out of `unjudged`. Post-v30 dedup uses UNION, not intersection."""
    c = _mk_candidate("arxiv", "2026.11111", "properly processed paper")
    _write_judgment(tmp_stores["judgments"], source="arxiv",
                     source_id="2026.11111", is_yes=True)
    _write_summary(tmp_stores["summaries"], source="arxiv",
                    source_id="2026.11111")

    def _fake_load_cache():
        return [c]
    import engine.agents.papers_curator as pc_mod
    monkeypatch.setattr(pc_mod, "load_cache", _fake_load_cache)

    r = pcd._step_filter(max_filter=0)
    assert r["unjudged_total"] == 0
    # This paper is NOT orphan (has judgment) — so orphan_summaries=0
    assert r["orphan_summaries"] == 0


def test_paper_in_judgments_only_still_skipped(monkeypatch, tmp_stores):
    """Basic pre-v30 dedup path still works: judgment-only papers stay
    out of the unjudged pool."""
    c = _mk_candidate("arxiv", "2026.22222", "judged-no paper")
    _write_judgment(tmp_stores["judgments"], source="arxiv",
                     source_id="2026.22222", is_yes=False)

    def _fake_load_cache():
        return [c]
    import engine.agents.papers_curator as pc_mod
    monkeypatch.setattr(pc_mod, "load_cache", _fake_load_cache)

    r = pcd._step_filter(max_filter=0)
    assert r["unjudged_total"] == 0
    assert r["orphan_summaries"] == 0


def test_orphan_summaries_count_scales_with_backlog(monkeypatch, tmp_stores):
    """Diagnostic: N orphan summaries → count reports N."""
    # 4 papers all summarized-not-judged (orphans)
    for i in range(4):
        _write_summary(tmp_stores["summaries"], source="arxiv",
                        source_id=f"2026.{i:05d}")

    def _fake_load_cache():
        return []   # empty cache — we only care about the orphan count
    import engine.agents.papers_curator as pc_mod
    monkeypatch.setattr(pc_mod, "load_cache", _fake_load_cache)

    r = pcd._step_filter(max_filter=0)
    assert r["orphan_summaries"] == 4


def test_no_orphans_when_every_summary_has_matching_judgment(
        monkeypatch, tmp_stores):
    """Clean-state case: every summarized paper also has its judgment
    row → orphan count is 0. This is what a healthy pipeline looks like
    after all legacy backfills are cleaned up."""
    for i in range(3):
        sid = f"2026.{i:05d}"
        _write_judgment(tmp_stores["judgments"], source="arxiv",
                         source_id=sid, is_yes=True)
        _write_summary(tmp_stores["summaries"], source="arxiv",
                        source_id=sid)

    def _fake_load_cache():
        return []
    import engine.agents.papers_curator as pc_mod
    monkeypatch.setattr(pc_mod, "load_cache", _fake_load_cache)

    r = pcd._step_filter(max_filter=0)
    assert r["orphan_summaries"] == 0


def test_empty_state_returns_zero_everywhere(monkeypatch, tmp_stores):
    """Cold-start invariant: no cache, no judgments, no summaries → all
    counts zero, no exceptions."""
    def _fake_load_cache():
        return []
    import engine.agents.papers_curator as pc_mod
    monkeypatch.setattr(pc_mod, "load_cache", _fake_load_cache)

    r = pcd._step_filter(max_filter=0)
    assert r["unjudged_total"]    == 0
    assert r["orphan_summaries"] == 0
    assert r["judged_now"]        == 0


def test_return_dict_shape_stable(monkeypatch, tmp_stores):
    """Contract lock: v30 added `orphan_summaries` but must not remove
    any pre-existing key. Downstream (cron log parsers, dashboards)
    depend on the shape."""
    def _fake_load_cache():
        return []
    import engine.agents.papers_curator as pc_mod
    monkeypatch.setattr(pc_mod, "load_cache", _fake_load_cache)

    r = pcd._step_filter(max_filter=0)
    expected_keys = {"unjudged_total", "judged_now", "yes", "no", "errors",
                     "orphan_summaries"}
    assert set(r.keys()) == expected_keys
