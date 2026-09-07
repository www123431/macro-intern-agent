"""Tests for engine.research.dying_families + its burndown_ranker
integration (v22, 2026-06-28).

Covers:
  - load_dying_families thresholds (min_n, green_rate, positive_rate)
  - dying_family_penalty scalar behavior
  - dying_families_report structure
  - Ranker integration: hypotheses in dying families get their rank
    HALVED but stay in the candidate pool (soft, not hard block)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.research.dying_families import (
    MAX_GREEN_RATE,
    MAX_POSITIVE_RATE,
    MIN_AUTOPSIES,
    RANKER_PENALTY,
    dying_families_report,
    dying_family_penalty,
    load_dying_families,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _write_autopsies(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _row(family: str, verdict: str) -> dict:
    return {"strategy_family": family, "actual_verdict": verdict}


# ── load_dying_families ─────────────────────────────────────────────


def test_all_red_family_at_min_n_is_dying(tmp_path):
    """2 RED autopsies at N=2 (exactly the min) → dying."""
    ap = tmp_path / "a.jsonl"
    _write_autopsies(ap, [_row("DEAD_FAM", "RED") for _ in range(2)])
    out = load_dying_families(autopsies_path=ap)
    assert "DEAD_FAM" in out


def test_n_1_family_not_dying_even_if_red(tmp_path):
    """Single-observation family should NOT be flagged — the sample is
    too small to draw a conclusion regardless of the verdict."""
    ap = tmp_path / "a.jsonl"
    _write_autopsies(ap, [_row("BABY_FAM", "RED")])
    out = load_dying_families(autopsies_path=ap)
    assert "BABY_FAM" not in out


def test_all_marginal_family_not_dying(tmp_path):
    """Positive rate = 100% (all MARGINAL counts as positive) → NOT dying."""
    ap = tmp_path / "a.jsonl"
    _write_autopsies(ap, [_row("MARGY_FAM", "MARGINAL") for _ in range(4)])
    out = load_dying_families(autopsies_path=ap)
    assert "MARGY_FAM" not in out


def test_mostly_red_one_marginal_still_dying(tmp_path):
    """4 RED + 0 MARGINAL + 0 GREEN → positive rate = 0. Dying."""
    ap = tmp_path / "a.jsonl"
    _write_autopsies(ap, [_row("MOSTLY_DEAD", "RED") for _ in range(4)])
    out = load_dying_families(autopsies_path=ap)
    assert "MOSTLY_DEAD" in out


def test_mixed_family_with_gt_10pct_green_not_dying(tmp_path):
    """1 GREEN out of 5 = 20% GREEN rate — above 10% cutoff. NOT dying."""
    ap = tmp_path / "a.jsonl"
    rows = [_row("MIXED_FAM", "GREEN")] + [_row("MIXED_FAM", "RED") for _ in range(4)]
    _write_autopsies(ap, rows)
    out = load_dying_families(autopsies_path=ap)
    assert "MIXED_FAM" not in out


def test_marginal_above_positive_threshold_not_dying(tmp_path):
    """40% MARGINAL puts positive rate above 30% → NOT dying even
    though GREEN rate is 0%."""
    ap = tmp_path / "a.jsonl"
    rows = [_row("F", "MARGINAL")] * 2 + [_row("F", "RED")] * 3
    _write_autopsies(ap, rows)   # 40% M / 60% R → positive rate 40%
    out = load_dying_families(autopsies_path=ap)
    assert "F" not in out


def test_superseded_rows_excluded(tmp_path):
    """A RED autopsy tagged superseded_by should not count against
    the family — belief-4 already excludes them, we mirror the rule."""
    ap = tmp_path / "a.jsonl"
    rows = [_row("SUP_FAM", "RED") for _ in range(3)]
    rows[0]["superseded_by"] = "correction_2026_06_15"
    _write_autopsies(ap, rows)
    # 2 non-superseded RED → still meets criteria
    out = load_dying_families(autopsies_path=ap)
    assert "SUP_FAM" in out
    # But if we superseded all of them, family drops back below cutoff
    for r in rows:
        r["superseded_by"] = "correction_2026_06_15"
    _write_autopsies(ap, rows)
    out2 = load_dying_families(autopsies_path=ap)
    assert "SUP_FAM" not in out2


def test_missing_file_returns_empty_set(tmp_path):
    out = load_dying_families(autopsies_path=tmp_path / "missing.jsonl")
    assert out == set()


def test_custom_thresholds_override_defaults(tmp_path):
    """A caller can widen the criteria — useful for future tuning
    experiments without changing the module defaults."""
    ap = tmp_path / "a.jsonl"
    # 1 GREEN + 4 RED = 20% GREEN. Default (max 10%) → not dying.
    rows = [_row("TUNE_FAM", "GREEN")] + [_row("TUNE_FAM", "RED") for _ in range(4)]
    _write_autopsies(ap, rows)
    default_out = load_dying_families(autopsies_path=ap)
    assert "TUNE_FAM" not in default_out
    # With relaxed max_green_rate=0.25 it enters the set
    relaxed_out = load_dying_families(autopsies_path=ap, max_green_rate=0.25)
    assert "TUNE_FAM" in relaxed_out


# ── dying_family_penalty scalar ─────────────────────────────────────


def test_dying_penalty_applies_to_member():
    assert dying_family_penalty("DEAD", {"DEAD"}) == RANKER_PENALTY


def test_dying_penalty_no_effect_for_non_member():
    assert dying_family_penalty("HEALTHY", {"DEAD"}) == 1.0


def test_dying_penalty_case_insensitive():
    """Ranker passes families as uppercase but the invariant is one
    edit away — test both cases work."""
    assert dying_family_penalty("dead", {"DEAD"}) == RANKER_PENALTY


def test_dying_penalty_custom_scalar():
    """Callers can specify a different penalty strength for A/B
    experiments without patching the module constant."""
    assert dying_family_penalty("F", {"F"}, penalty=0.25) == 0.25


def test_dying_penalty_empty_family_string_never_matches():
    assert dying_family_penalty("", {"DEAD"}) == 1.0


# ── dying_families_report ────────────────────────────────────────────


def test_report_contains_thresholds_and_families(tmp_path):
    ap = tmp_path / "a.jsonl"
    _write_autopsies(ap, [
        _row("DEAD", "RED"), _row("DEAD", "RED"),
        _row("ALIVE", "GREEN"), _row("ALIVE", "MARGINAL"),
    ])
    # Redirect the module default so the report uses our tmp file
    import engine.research.dying_families as df
    df._AUTOPSIES_PATH = ap
    rep = dying_families_report()
    assert rep["thresholds"]["min_autopsies"] == MIN_AUTOPSIES
    assert rep["thresholds"]["max_green_rate"] == MAX_GREEN_RATE
    assert rep["n_families_dying"] == 1
    fam_ids = {f["family"]: f["is_dying"] for f in rep["families"]}
    assert fam_ids["DEAD"] is True
    assert fam_ids["ALIVE"] is False


# ── burndown_ranker integration ─────────────────────────────────────


def test_ranker_halves_score_for_dying_family(tmp_path, monkeypatch):
    """The whole point of v22: a hypothesis in a dying family gets its
    rank_score HALVED, but the candidate is not dropped from the pool."""
    from engine.research import burndown_ranker, dying_families

    hyp_path = tmp_path / "hyps.jsonl"
    dispatch_log = tmp_path / "dispatch.jsonl"
    gaps_path = tmp_path / "gaps.jsonl"
    autopsies_path = tmp_path / "autopsies.jsonl"

    # 1 hypothesis in a dying family, 1 in a healthy family. Otherwise
    # identical (same created_ts, same claim length, etc.) so the
    # dying_penalty is the only thing that differs.
    hyps = [
        {"hypothesis_id": "h_dying",   "mechanism_family": "DEAD_FAM",
         "review_state":  "proposed",  "claim": "x", "created_ts": "2026-06-27T00:00:00Z"},
        {"hypothesis_id": "h_healthy", "mechanism_family": "HEALTHY_FAM",
         "review_state":  "proposed",  "claim": "x", "created_ts": "2026-06-27T00:00:00Z"},
    ]
    with hyp_path.open("w", encoding="utf-8") as fh:
        for r in hyps:
            fh.write(json.dumps(r) + "\n")
    dispatch_log.write_text("", encoding="utf-8")
    gaps_path.write_text("", encoding="utf-8")

    # DEAD_FAM has 3 RED autopsies → dying
    _write_autopsies(autopsies_path, [
        _row("DEAD_FAM", "RED") for _ in range(3)
    ])

    # Point dying_families at the tmp autopsies file
    monkeypatch.setattr(dying_families, "_AUTOPSIES_PATH", autopsies_path)
    # Ranker's DISPATCHABLE_FAMILIES gates which families make it into
    # the candidate pool. Whitelist ours so they aren't silently dropped.
    monkeypatch.setattr(
        burndown_ranker, "DISPATCHABLE_FAMILIES",
        {"DEAD_FAM", "HEALTHY_FAM"},
    )
    # Same for the review_state whitelist (ELIGIBLE_REVIEW_STATES).
    monkeypatch.setattr(
        burndown_ranker, "ELIGIBLE_REVIEW_STATES",
        {"proposed", "ready_for_dispatch"},
    )

    ranked = burndown_ranker.rank_candidates(
        hyp_path=hyp_path,
        dispatch_log_path=dispatch_log,
        gaps_path=gaps_path,
        top_k=10,
    )
    # Both candidates must survive
    by_id = {c.hypothesis_id: c for c in ranked}
    assert set(by_id) == {"h_dying", "h_healthy"}
    # Dying gets the penalty; healthy doesn't
    assert by_id["h_dying"].dying_penalty_score == RANKER_PENALTY
    assert by_id["h_healthy"].dying_penalty_score == 1.0
    # And that shows up in rank_score — dying candidate is worth half
    # what the healthy one would be for otherwise-identical data
    assert by_id["h_dying"].rank_score == pytest.approx(
        0.5 * by_id["h_healthy"].rank_score, rel=1e-6,
    )
    # Sort order reflects it — healthy comes first
    assert ranked[0].hypothesis_id == "h_healthy"
    assert ranked[1].hypothesis_id == "h_dying"
