"""tests/test_bond_credit_carry.py — v48 template."""
from __future__ import annotations

import datetime as _dt

import pytest


def test_verdict_from_t_boundaries():
    from engine.agents.strengthener.templates.bond_credit_carry import (
        _verdict_from_t, _T_GREEN, _T_MARGINAL,
    )
    assert _verdict_from_t(3.0)               == "GREEN"
    assert _verdict_from_t(_T_GREEN)          == "GREEN"
    assert _verdict_from_t(_T_GREEN - 0.01)   == "MARGINAL"
    assert _verdict_from_t(_T_MARGINAL - 0.01) == "RED"
    assert _verdict_from_t(-3.0)              == "GREEN"    # |t| symmetric
    assert _verdict_from_t(float("nan"))      == "RED"


def test_contract_registered_and_fresh():
    from engine.agents.strengthener.templates._template_contract import (
        contract_for_scope,
    )
    c = contract_for_scope("carry", "corporate_bonds_ig_hy")
    assert c is not None
    assert c.is_fresh()
    assert c.canonical_paper_id == "israel_palhares_richardson_2018"


def test_corporate_bonds_universe_in_enum():
    from engine.agents.strengthener.factor_spec_extractor import UNIVERSES
    assert "corporate_bonds_ig_hy" in UNIVERSES


def test_bond_dispatcher_routes(monkeypatch):
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec

    calls: list[str] = []
    def _fake(spec):
        calls.append("bond_credit_carry")
        return fd.TemplateResult(
            verdict="RED", summary="stub", metrics={}, artifacts={},
            template_version="stub",
        )
    monkeypatch.setattr(
        "engine.agents.strengthener.templates.bond_credit_carry."
        "template_bond_credit_carry",
        _fake,
    )
    spec = FactorSpec(
        hypothesis_id="t", signal_kind="carry",
        universe="corporate_bonds_ig_hy", date_range="2013-06:2024-06",
        signal_inputs=("bond.return.corp",), rebal="monthly",
        weighting="tercile_long_short_dollar_neutral",
        expected_holding_period="monthly", min_obs_months=60,
        pit_audits=("restatement",), cost_model="none",
        rationale="test", extracted_ts="2026-07-02T06:00:00Z",
        model="test",
    )
    r = fd._carry_template_lazy(spec)
    assert calls == ["bond_credit_carry"]
    assert r.verdict == "RED"


def test_fx_g10_and_commodity_routing_unaffected(monkeypatch):
    """Regression: adding corporate_bonds_ig_hy universe must not touch
    fx_g10 or commodity_futures_24 routing."""
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec

    hits: list[str] = []
    monkeypatch.setattr(
        "engine.agents.strengthener.templates.carry_g10_fx.template_carry_g10_fx",
        lambda s: (hits.append("g10"), fd.TemplateResult(
            verdict="RED", summary="stub", metrics={},
            artifacts={}, template_version="stub"))[1],
    )
    monkeypatch.setattr(
        "engine.agents.strengthener.templates.commodity_carry_futures."
        "template_commodity_carry_futures",
        lambda s: (hits.append("cmdty"), fd.TemplateResult(
            verdict="RED", summary="stub", metrics={},
            artifacts={}, template_version="stub"))[1],
    )
    for uni, expected in (
        ("fx_g10",                "g10"),
        ("commodity_futures_24",  "cmdty"),
    ):
        spec = FactorSpec(
            hypothesis_id="t", signal_kind="carry", universe=uni,
            date_range="2015-01:2020-12",
            signal_inputs=("x.y.z",), rebal="monthly",
            weighting="tercile_long_short_dollar_neutral",
            expected_holding_period="monthly", min_obs_months=60,
            pit_audits=("restatement",), cost_model="none",
            rationale="t", extracted_ts="2026-07-02T06:00:00Z", model="t",
        )
        fd._carry_template_lazy(spec)
    assert hits == ["g10", "cmdty"], (
        f"routing regression: expected [g10, cmdty], got {hits}"
    )


def test_amt_weighted_handles_zero_weights():
    """Defensive: if all amount_outstanding are 0 or NaN, return NaN not
    an error."""
    import pandas as pd
    from engine.agents.strengthener.templates.bond_credit_carry import (
        _amt_weighted,
    )
    zero_w = pd.DataFrame({
        "ret_eom": [0.01, -0.02, 0.03],
        "amount_outstanding": [0.0, 0.0, 0.0],
    })
    result = _amt_weighted(zero_w)
    assert not pd.notna(result) or result == 0.0
