"""tests/test_treasury_term_carry.py — v49 template."""
from __future__ import annotations


def test_verdict_from_t_boundaries():
    from engine.agents.strengthener.templates.treasury_term_carry import (
        _verdict_from_t, _T_GREEN, _T_MARGINAL,
    )
    assert _verdict_from_t(3.0)                == "GREEN"
    assert _verdict_from_t(_T_MARGINAL - 0.01) == "RED"
    assert _verdict_from_t(-3.0)               == "GREEN"


def test_contract_registered_and_fresh():
    from engine.agents.strengthener.templates._template_contract import (
        contract_for_scope,
    )
    c = contract_for_scope("carry", "us_treasury_curve")
    assert c is not None
    assert c.is_fresh()
    assert c.canonical_paper_id == "fama_bliss_1987"


def test_universe_in_enum():
    from engine.agents.strengthener.factor_spec_extractor import UNIVERSES
    assert "us_treasury_curve" in UNIVERSES


def test_dispatcher_routes(monkeypatch):
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec

    calls: list[str] = []
    def _fake(spec):
        calls.append("treasury")
        return fd.TemplateResult(
            verdict="RED", summary="stub", metrics={}, artifacts={},
            template_version="stub",
        )
    monkeypatch.setattr(
        "engine.agents.strengthener.templates.treasury_term_carry."
        "template_treasury_term_carry",
        _fake,
    )
    spec = FactorSpec(
        hypothesis_id="t", signal_kind="carry",
        universe="us_treasury_curve", date_range="2005-01:2020-12",
        signal_inputs=("treasury.constant_maturity.dgs10",), rebal="monthly",
        weighting="tercile_long_short_dollar_neutral",
        expected_holding_period="monthly", min_obs_months=60,
        pit_audits=("restatement",), cost_model="none",
        rationale="t", extracted_ts="2026-07-02T06:00:00Z", model="t",
    )
    r = fd._carry_template_lazy(spec)
    assert calls == ["treasury"]


def test_sharpe_and_t_helper_bounds():
    """Regression: _sharpe_and_t must not crash on tiny or degenerate series."""
    import pandas as pd
    from engine.agents.strengthener.templates.treasury_term_carry import (
        _sharpe_and_t,
    )
    # Empty series
    mu, sd, sh, t, n = _sharpe_and_t(pd.Series([], dtype=float))
    assert n == 0
    # Two-element series
    mu, sd, sh, t, n = _sharpe_and_t(pd.Series([0.01, 0.02]))
    assert n == 2
    # Constant series → sd = 0, sh/t = NaN, no exception
    mu, sd, sh, t, n = _sharpe_and_t(pd.Series([0.01, 0.01, 0.01]))
    assert n == 3
    import math
    assert math.isnan(sh) or sh == 0
    assert math.isnan(t) or t == 0


def test_regression_all_four_carry_universes_registered():
    """v49 completes CARRY family across 4 asset classes. Regression:
    all four contracts must be present + fresh + unique template names."""
    from engine.agents.strengthener.templates._template_contract import (
        contract_for_scope,
    )
    seen = set()
    for uni in ("fx_g10", "commodity_futures_24",
                 "corporate_bonds_ig_hy", "us_treasury_curve"):
        c = contract_for_scope("carry", uni)
        assert c is not None, f"CARRY / {uni} contract missing"
        assert c.is_fresh(), f"CARRY / {uni} contract stale"
        seen.add(c.template_name)
    assert len(seen) == 4, f"template names must be unique per universe; got {seen}"
