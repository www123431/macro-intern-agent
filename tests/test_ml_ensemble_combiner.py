"""tests/test_ml_ensemble_combiner.py — v45 template."""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd
import pytest


# ── Verdict + parsing utilities ────────────────────────────────────


def test_verdict_from_t_boundaries():
    from engine.agents.strengthener.templates.ml_ensemble_combiner import (
        _verdict_from_t, _T_GREEN, _T_MARGINAL,
    )
    assert _verdict_from_t(3.0)               == "GREEN"
    assert _verdict_from_t(_T_GREEN)          == "GREEN"
    assert _verdict_from_t(_T_GREEN - 0.01)   == "MARGINAL"
    assert _verdict_from_t(_T_MARGINAL)       == "MARGINAL"
    assert _verdict_from_t(_T_MARGINAL - 0.01) == "RED"
    assert _verdict_from_t(-3.0)              == "GREEN"    # |t| symmetric
    assert _verdict_from_t(float("nan"))      == "RED"


def test_parse_date_range():
    from engine.agents.strengthener.templates.ml_ensemble_combiner import (
        _parse_date_range,
    )
    a, b = _parse_date_range("2015-06:2020-12")
    assert a == _dt.date(2015, 6, 1)
    assert b == _dt.date(2020, 12, 31)


# ── Contract registration ──────────────────────────────────────────


def test_contract_registered_and_fresh():
    from engine.agents.strengthener.templates._template_contract import (
        contract_for_scope,
    )
    c = contract_for_scope("ml_ensemble", "us_equities_top_3000")
    assert c is not None, "ml_ensemble contract must be registered"
    assert c.is_fresh()
    assert c.canonical_paper_id == "gu_kelly_xiu_2020"


def test_contract_lookup_isolated_from_other_signal_kinds():
    """Regression: adding ml_ensemble must not shadow spanning_test /
    cross_sectional_rank / carry lookups."""
    from engine.agents.strengthener.templates._template_contract import (
        contract_for_scope,
    )
    # ml_ensemble on wrong universe → no match
    assert contract_for_scope("ml_ensemble", "fx_g10") is None
    # Other kinds still resolve
    assert contract_for_scope("carry", "fx_g10") is not None


# ── Dispatcher routing ─────────────────────────────────────────────


def test_dispatcher_routes_ml_ensemble(monkeypatch):
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec

    calls: list[str] = []
    def _fake_template(spec):
        calls.append("ml_ensemble")
        return fd.TemplateResult(
            verdict="MARGINAL", summary="stub", metrics={},
            artifacts={}, template_version="stub",
        )
    monkeypatch.setattr(
        "engine.agents.strengthener.templates.ml_ensemble_combiner."
        "template_ml_ensemble_combiner",
        _fake_template,
    )
    spec = FactorSpec(
        hypothesis_id="t", signal_kind="ml_ensemble",
        universe="us_equities_top_3000", date_range="2015-01:2020-12",
        signal_inputs=("crsp.msf.ret",), rebal="monthly",
        weighting="tercile_long_short_dollar_neutral",
        expected_holding_period="monthly", min_obs_months=60,
        pit_audits=("restatement",), cost_model="none",
        rationale="test", extracted_ts="2026-07-02T05:00:00Z",
        model="test",
    )
    r = fd._ml_ensemble_template_lazy(spec)
    assert calls == ["ml_ensemble"]
    assert r.verdict == "MARGINAL"


def test_dispatcher_ml_ensemble_wrong_universe_falls_to_pending(monkeypatch):
    """Only us_equities_top_3000 is wired; other universes → pending_build."""
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec

    spec = FactorSpec(
        hypothesis_id="t", signal_kind="ml_ensemble",
        universe="commodity_futures_24", date_range="2015-01:2020-12",
        signal_inputs=("futures.settle.raw",), rebal="monthly",
        weighting="tercile_long_short_dollar_neutral",
        expected_holding_period="monthly", min_obs_months=60,
        pit_audits=("restatement",), cost_model="none",
        rationale="test", extracted_ts="2026-07-02T05:00:00Z",
        model="test",
    )
    r = fd._ml_ensemble_template_lazy(spec)
    assert r.verdict in ("EXECUTION_ERROR", "PENDING_TEMPLATE_BUILD")


# ── Signal-kind + universe registration ────────────────────────────


def test_ml_ensemble_in_signal_kinds_enum():
    from engine.agents.strengthener.factor_spec_extractor import SIGNAL_KINDS
    assert "ml_ensemble" in SIGNAL_KINDS


def test_template_registry_has_ml_ensemble():
    from engine.agents.strengthener.factor_dispatcher import TEMPLATE_REGISTRY
    assert "ml_ensemble" in TEMPLATE_REGISTRY
    assert TEMPLATE_REGISTRY["ml_ensemble"] is not None


# ── Rank normalization sanity ──────────────────────────────────────


def test_rank_normalization_is_per_month_cross_section(tmp_path, monkeypatch):
    """Load _ml_feature_panel and verify: per-month rank is bounded
    [0, 1] and has median ≈ 0.5. If we accidentally globally-rank,
    values within any month wouldn't span [0, 1]."""
    from engine.agents.strengthener.templates.ml_ensemble_combiner import (
        _load_and_clean, _FEATURES,
    )
    df = _load_and_clean(
        _dt.date(2020, 1, 1), _dt.date(2020, 12, 31),
    )
    if len(df) == 0:
        pytest.skip("_ml_feature_panel.parquet not present or empty for window")
    for f in _FEATURES:
        col = f + "_r"
        assert (df[col] >= 0).all() and (df[col] <= 1).all(), (
            f"{col} rank out of [0, 1]"
        )
        # median close to 0.5 per month
        med_by_month = df.groupby("month")[col].median()
        # allow slack for tiny cross-sections
        assert (med_by_month.between(0.3, 0.7)).mean() > 0.8, (
            f"{col} monthly median not ≈ 0.5"
        )
