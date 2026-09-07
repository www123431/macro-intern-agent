"""Belief Layer Phase 1 (belief-1, 2026-06-11) tests.

Covers:
  * predict_verdict produces a valid distribution
  * family priors override default for known families
  * n_trials penalty shrinks GREEN
  * post-publication age penalty shifts toward RED
  * load-bearing list surfaces correct assumptions
  * log_prediction writes valid jsonl
  * **structural invariant**: lens / strict_gate / template modules
    MUST NOT import engine.research.belief (air-gap doctrine)
"""
from __future__ import annotations

import ast
import json
import pathlib
import tempfile

import pytest

from engine.research import belief


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


# ── Basic prediction math ───────────────────────────────────────────


def test_predicted_dist_sums_to_one_default():
    pred = belief.predict_verdict(subject_id="t1", family=None)
    s = sum(pred.predicted_verdict_dist.values())
    assert abs(s - 1.0) < 1e-9
    assert set(pred.predicted_verdict_dist) == {"GREEN", "MARGINAL", "RED"}


def test_predicted_dist_uses_default_for_unknown_family():
    pred = belief.predict_verdict(subject_id="t1", family="UNKNOWN_FAM_XYZ")
    # Default has GREEN=0.20
    assert pred.predicted_verdict_dist["GREEN"] == pytest.approx(0.20, abs=0.01)
    assert "default prior" in pred.prediction_basis.lower()


def _pin_out_all_external_calibration(monkeypatch):
    """v19 helper: isolate predict_verdict from real-corpus calibration
    sources so override / default-prior branch tests are deterministic.

    Three input sources need pinning:
      1. _family_observed_dist (Step 1b: observed posterior from events.jsonl)
      2. calibrated_family_prior (Step 1a: belief-4 closed-loop)
      3. _raw_family_empirical_at (Step 3.5: W7 ensemble blend)
    """
    monkeypatch.setattr(
        belief, "_family_observed_dist",
        lambda fam: (dict(belief.DEFAULT_PRIOR), 0),
    )
    import engine.research.belief_prior_calibration as bpc_module
    monkeypatch.setattr(bpc_module, "calibrated_family_prior",
                          lambda fam, **kw: None)
    monkeypatch.setattr(
        belief, "_raw_family_empirical_at",
        lambda fam: ({"GREEN": 0.0, "MARGINAL": 0.0, "RED": 0.0}, 0),
    )


def test_predicted_dist_uses_family_override_when_no_observations(monkeypatch):
    # Force the observed-posterior path to return N=0 so we exercise the
    # override branch deterministically (the real events.jsonl may carry
    # accumulated PROFITABILITY verdicts which would otherwise dominate).
    _pin_out_all_external_calibration(monkeypatch)
    pred = belief.predict_verdict(subject_id="t1", family="PROFITABILITY")
    # PROFITABILITY override has GREEN=0.12 — strictly lower than default 0.20
    assert pred.predicted_verdict_dist["GREEN"] < 0.20
    assert "family prior override" in pred.prediction_basis.lower()
    assert pred.family == "PROFITABILITY"


def test_family_case_insensitive(monkeypatch):
    _pin_out_all_external_calibration(monkeypatch)
    a = belief.predict_verdict(subject_id="t1", family="profitability")
    b = belief.predict_verdict(subject_id="t1", family="PROFITABILITY")
    assert a.family == "PROFITABILITY"
    assert b.family == "PROFITABILITY"
    assert a.predicted_verdict_dist == b.predicted_verdict_dist


# ── Adjustments ─────────────────────────────────────────────────────


def test_n_trials_penalty_shrinks_green():
    dist_no_penalty = {"GREEN": 0.30, "MARGINAL": 0.40, "RED": 0.30}
    out = belief._apply_n_trials_penalty(dist_no_penalty, n_trials=20)
    assert out["GREEN"] < dist_no_penalty["GREEN"]
    assert out["MARGINAL"] > dist_no_penalty["MARGINAL"]
    # RED unchanged (penalty doesn't make real factor fake, just harder to claim)
    assert out["RED"] == pytest.approx(dist_no_penalty["RED"], abs=0.01)
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_n_trials_below_threshold_no_penalty():
    dist = {"GREEN": 0.30, "MARGINAL": 0.40, "RED": 0.30}
    out = belief._apply_n_trials_penalty(dist, n_trials=5)
    assert out == dist


def test_publication_age_penalty_shifts_to_red():
    dist = {"GREEN": 0.30, "MARGINAL": 0.40, "RED": 0.30}
    new_dist, applied = belief._apply_publication_age_penalty(
        dist, paper_year=2000, current_year=2026,
    )
    assert applied is True
    assert new_dist["RED"] > dist["RED"]
    assert new_dist["GREEN"] < dist["GREEN"]
    assert abs(sum(new_dist.values()) - 1.0) < 1e-9


def test_recent_paper_no_age_penalty():
    dist = {"GREEN": 0.30, "MARGINAL": 0.40, "RED": 0.30}
    new_dist, applied = belief._apply_publication_age_penalty(
        dist, paper_year=2020, current_year=2026,
    )
    assert applied is False
    assert new_dist == dist


def test_old_paper_marks_decay_load_bearing():
    pred = belief.predict_verdict(
        subject_id="t1", family="MOMENTUM",
        paper_year=1990, current_year=2026,
    )
    assert "post_publication_decay" in pred.predicted_load_bearing


def test_mature_family_marks_spanning_risk():
    pred = belief.predict_verdict(subject_id="t1", family="PROFITABILITY")
    assert "spanning_risk" in pred.predicted_load_bearing


# ── v35 (2026-07-01): ensemble threshold prevents small-N families ─
# ── from getting 100% RED via w=1.0 empirical replacement ────────────


def test_v35_ensemble_threshold_is_5():
    """Regression: v19 lowered this from 5 to 3, which caused
    VOL_RISK_PREMIUM (3 events all RED) to predict 100% RED —
    structurally blocking any new experimental variation in that
    family from ever being dispatched. v35 raised it back to 5 so
    small samples get shrinkage instead of pure empirical."""
    assert belief._ENSEMBLE_MIN_ELIGIBLE_N == 5


def test_v35_family_with_3_events_falls_back_to_dirichlet(monkeypatch):
    """Concrete regression against the 2026-07-01 VOL_RISK_PREMIUM
    case. Family with N=3 events (all RED) should NOT get 100% RED
    prediction — falls back to Dirichlet-smoothed observed posterior
    which preserves some GREEN mass."""
    # Fake 3 RED events for a synthetic family
    class _E:
        def __init__(self, v): self.verdict = v
    fake_events = [_E("RED"), _E("RED"), _E("RED")]
    class _S:
        @staticmethod
        def filter_events(**kw):
            return fake_events
    monkeypatch.setattr("engine.research_store.store", _S)
    # Also stub belief-4 to None so we exercise the observed-posterior fallback
    import engine.research.belief_prior_calibration as bpc
    monkeypatch.setattr(bpc, "calibrated_family_prior",
                          lambda fam, **kw: None)

    pred = belief.predict_verdict(subject_id="v35_test", family="TEST_SMALL")
    dist = pred.predicted_verdict_dist
    # Dirichlet with alpha=1.0, N=3 all RED, prior GREEN=0.20:
    # posterior_G = (0 + 1.0*0.20) / (3 + 1.0) ≈ 0.05 — small, but > 0
    # Actually the pipeline uses base DEFAULT_PRIOR with 3-way smoothing,
    # so exact number depends on smoothing formula. Assert non-zero + not extreme.
    assert dist["GREEN"] > 0.05, (
        f"v35 must keep at least some GREEN mass on 3-RED families; got "
        f"{dist['GREEN']}. Pre-v35 (threshold=3) would give 0.0."
    )
    assert dist["RED"] < 0.90, (
        f"v35 must temper the extreme empirical; got RED={dist['RED']}. "
        f"Pre-v35 (threshold=3) would give 1.0."
    )
    # And ensemble blend should NOT appear in the basis for N=3
    assert "W7-ensemble blend applied" not in pred.prediction_basis, (
        f"v35: ensemble blend must NOT fire for N<5; basis: {pred.prediction_basis}"
    )


def test_v36_dedups_by_subject_id(monkeypatch):
    """Regression: pre-v36, `_raw_family_empirical_at` counted every
    factor_verdict_filed event as an independent trial. The live audit
    2026-07-01 found CARRY had 32 events but only 17 distinct subjects
    — one subject re-dispatched 15 times (probably cost-stress sweeps),
    all RED. Belief empirical treated this as 32 independent RED trials
    → 100% RED prediction, structurally blocking any new CARRY-family
    hypothesis. Post-v36: dedup by subject, take latest verdict per
    subject. Same subject re-dispatched 15 times → counts as 1 sample."""
    class _E:
        def __init__(self, subj, v):
            self.subject_id = subj
            self.verdict    = v
    # Same subject re-dispatched 5 times, all RED, then 1 GREEN on a
    # different subject. Pre-v36: 5R + 1G = 83% RED. Post-v36: 1R + 1G = 50% RED.
    fake = ([_E("repeat_signal", "RED")] * 5 +
             [_E("distinct_signal", "GREEN")])
    class _S:
        @staticmethod
        def filter_events(**kw):
            return fake
    monkeypatch.setattr("engine.research_store.store", _S)

    emp, n = belief._raw_family_empirical_at("TEST_FAM")
    assert n == 2, f"v36 dedup: expected 2 distinct subjects, got n={n}"
    assert emp["GREEN"] == 0.5, (
        f"v36: two distinct subjects (1G, 1R) → 50% empirical GREEN; "
        f"got {emp}"
    )
    assert emp["RED"] == 0.5


def test_v36_latest_verdict_per_subject_wins(monkeypatch):
    """When the same subject gets multiple verdicts (e.g. re-tested
    after a fix), dedup takes the LATEST — later verdicts overwrite
    earlier ones in the count. Matches belief_prior_calibration
    _autopsies_for_family semantics for supersede."""
    class _E:
        def __init__(self, subj, v):
            self.subject_id = subj
            self.verdict    = v
    # Same subject: first RED (attempt 1), later GREEN (fixed version).
    # Post-v36: takes GREEN (last one in file order).
    fake = [
        _E("evolving_signal", "RED"),
        _E("evolving_signal", "RED"),
        _E("evolving_signal", "GREEN"),
    ]
    class _S:
        @staticmethod
        def filter_events(**kw):
            return fake
    monkeypatch.setattr("engine.research_store.store", _S)

    emp, n = belief._raw_family_empirical_at("TEST_LATEST")
    assert n == 1
    assert emp["GREEN"] == 1.0, (
        f"v36: latest verdict wins; expected 100% GREEN, got {emp}"
    )


def test_v36_skips_events_with_empty_subject(monkeypatch):
    """Defensive: an event with empty/None subject_id is skipped rather
    than being counted as a dedup key of ''. Prevents legacy corruption
    from silently pooling all no-subject events into one 'trial'."""
    class _E:
        def __init__(self, subj, v):
            self.subject_id = subj
            self.verdict    = v
    fake = [
        _E("", "RED"),        # blank
        _E(None, "RED"),      # missing
        _E("real_sub", "GREEN"),
    ]
    class _S:
        @staticmethod
        def filter_events(**kw):
            return fake
    monkeypatch.setattr("engine.research_store.store", _S)

    emp, n = belief._raw_family_empirical_at("TEST_EMPTY")
    assert n == 1
    assert emp["GREEN"] == 1.0


def test_v35_ensemble_still_fires_at_n_equals_5(monkeypatch):
    """The other side of the boundary: at N=5, ensemble SHOULD fire
    (families with 5+ observations are considered well-observed
    enough to trust the empirical distribution).

    v37 (2026-07-01): each mock event needs a distinct subject_id
    because v36 dedup keys off it. Semantic intent unchanged: "5
    distinct trials, mixed 3G/2R" — just made honest to the dedup path."""
    class _E:
        def __init__(self, subj, v):
            self.subject_id = subj
            self.verdict    = v
    fake_events = (
        [_E(f"g_{i}", "GREEN") for i in range(3)]
        + [_E(f"r_{i}", "RED") for i in range(2)]
    )   # 5 distinct subjects, mixed 3G/2R
    class _S:
        @staticmethod
        def filter_events(**kw):
            return fake_events
    monkeypatch.setattr("engine.research_store.store", _S)
    import engine.research.belief_prior_calibration as bpc
    monkeypatch.setattr(bpc, "calibrated_family_prior",
                          lambda fam, **kw: None)

    pred = belief.predict_verdict(subject_id="v35_5", family="TEST_5")
    # At N=5, ensemble should replace with raw empirical (3/5 = 0.6 GREEN)
    assert "W7-ensemble blend applied" in pred.prediction_basis
    assert abs(pred.predicted_verdict_dist["GREEN"] - 0.6) < 0.05, (
        f"At n=5, ensemble replaces with raw empirical (3G/5 = 0.6); "
        f"got {pred.predicted_verdict_dist['GREEN']}"
    )


# ── Logging ─────────────────────────────────────────────────────────


def test_log_prediction_writes_valid_jsonl(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        tmp_path = pathlib.Path(td) / "predictions.jsonl"
        monkeypatch.setattr(belief, "PREDICTIONS_PATH", tmp_path)
        pred = belief.predict_verdict(
            subject_id="t_log", family="QUALITY", paper_year=2015,
        )
        pid = belief.log_prediction(pred)
        assert pid == pred.prediction_id
        assert tmp_path.is_file()
        rows = [
            json.loads(ln) for ln in tmp_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert len(rows) == 1
        assert rows[0]["prediction_id"] == pid
        assert rows[0]["subject_id"] == "t_log"
        assert rows[0]["family"] == "QUALITY"
        assert "GREEN" in rows[0]["predicted_verdict_dist"]


def test_predict_and_log_returns_prediction(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        tmp_path = pathlib.Path(td) / "predictions.jsonl"
        monkeypatch.setattr(belief, "PREDICTIONS_PATH", tmp_path)
        pred = belief.predict_and_log(
            subject_id="t_pl", family="VALUE", signal_kind="cross_sec",
        )
        assert isinstance(pred, belief.Prediction)
        assert pred.subject_id == "t_pl"
        assert tmp_path.is_file()


# ── STRUCTURAL: air-gap invariant ───────────────────────────────────


# Files that legitimately MAY consume belief predictions.
# The air-gap protects VERDICT-COMPUTING code (lens / strict_gate /
# template) from seeing predictions and self-fulfilling them. Modules
# that produce predictions or display them (planners, dashboards) are
# safe by definition — they don't run lens math on factor data.
_BELIEF_CONSUMER_WHITELIST = frozenset({
    # The dispatcher entry hook — the only producer
    "engine/agents/strengthener/factor_dispatcher.py",
    # belief module itself + its tests
    "engine/research/belief.py",
    "tests/test_belief.py",
    # burn-1a planner — shows predictions in dry-run plans for principal
    # review. Does NOT compute verdicts; safe to consume.
    "engine/research/burndown_planner.py",
    # belief-2 autopsy — reads predictions + verdicts AFTER both produced;
    # writes parallel autopsies.jsonl. Doesn't compute verdicts.
    "engine/research/belief_autopsy.py",
    # belief-4 closed-loop prior — reads autopsies, exports calibrated
    # prior consumed BY belief.py itself. No verdict computation.
    "engine/research/belief_prior_calibration.py",
})


def _modules_under(*dirs: str) -> list[pathlib.Path]:
    """Return all .py files under the given repo-relative directories."""
    out: list[pathlib.Path] = []
    for d in dirs:
        root = REPO_ROOT / d
        if not root.is_dir():
            continue
        out.extend(root.rglob("*.py"))
    return out


def _imports_belief(path: pathlib.Path) -> bool:
    """Return True if the file's AST contains any import of engine.research.belief."""
    try:
        src = path.read_text(encoding="utf-8")
    except Exception:
        return False
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "engine.research.belief" or mod.startswith("engine.research.belief."):
                return True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "engine.research.belief" or alias.name.startswith("engine.research.belief."):
                    return True
    return False


def test_air_gap_lens_strict_gate_template_must_not_import_belief():
    """STRUCTURAL INVARIANT (Belief Layer doctrine 2026-06-11):

    No module under engine/research/ (lens / strict_gate / template
    plumbing) or engine/agents/strengthener/templates/ may import
    engine.research.belief. The predict-then-observe contract requires
    that verdict-computing code CANNOT see its own prediction —
    self-fulfilling prophecies would invalidate the entire calibration
    project.

    Whitelist: dispatcher entry hook + belief module + this test file.
    """
    candidate_dirs = (
        "engine/research",
        "engine/agents/strengthener/templates",
        "engine/agents/strengthener",
    )
    offenders: list[str] = []
    for fp in _modules_under(*candidate_dirs):
        try:
            rel = fp.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            continue
        if rel in _BELIEF_CONSUMER_WHITELIST:
            continue
        if _imports_belief(fp):
            offenders.append(rel)
    assert not offenders, (
        f"Air-gap violation: these modules import engine.research.belief "
        f"but are not on the whitelist: {offenders}. "
        f"Move belief consumption OUTSIDE the lens/strict_gate/template "
        f"tree (e.g. into engine.research.belief_autopsy in Phase 2)."
    )
