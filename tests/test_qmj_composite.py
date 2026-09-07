"""tests/test_qmj_composite.py — v50 QMJ template."""
from __future__ import annotations


def test_verdict_from_t_boundaries():
    from engine.agents.strengthener.templates.qmj_composite import (
        _verdict_from_t, _T_GREEN, _T_MARGINAL,
    )
    assert _verdict_from_t(3.0)                == "GREEN"
    assert _verdict_from_t(_T_MARGINAL - 0.01) == "RED"
    assert _verdict_from_t(-3.0)               == "GREEN"
    assert _verdict_from_t(float("nan"))       == "RED"


def test_contract_registered_and_fresh():
    from engine.agents.strengthener.templates._template_contract import (
        contract_for_scope,
    )
    c = contract_for_scope("quality_composite", "us_equities_top_3000")
    assert c is not None
    assert c.is_fresh()
    assert c.canonical_paper_id == "asness_frazzini_pedersen_2013"


def test_quality_composite_in_signal_kinds():
    from engine.agents.strengthener.factor_spec_extractor import SIGNAL_KINDS
    assert "quality_composite" in SIGNAL_KINDS


def test_template_registry():
    from engine.agents.strengthener.factor_dispatcher import TEMPLATE_REGISTRY
    assert TEMPLATE_REGISTRY.get("quality_composite") is not None


def test_dispatcher_routes_quality(monkeypatch):
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec

    calls: list[str] = []
    def _fake(spec):
        calls.append("qmj")
        return fd.TemplateResult(
            verdict="RED", summary="stub", metrics={}, artifacts={},
            template_version="stub",
        )
    monkeypatch.setattr(
        "engine.agents.strengthener.templates.qmj_composite."
        "template_qmj_composite",
        _fake,
    )
    spec = FactorSpec(
        hypothesis_id="t", signal_kind="quality_composite",
        universe="us_equities_top_3000", date_range="2015-01:2020-12",
        signal_inputs=("compustat.funda.gp",), rebal="monthly",
        weighting="tercile_long_short_dollar_neutral",
        expected_holding_period="monthly", min_obs_months=60,
        pit_audits=("restatement",), cost_model="none",
        rationale="t", extracted_ts="2026-07-02T07:00:00Z", model="t",
    )
    r = fd._qmj_composite_template_lazy(spec)
    assert calls == ["qmj"]
    assert r.verdict == "RED"


def test_z_score_helper_is_per_month():
    """Regression: _z_score_per_month must compute WITHIN month, not global."""
    import pandas as pd
    from engine.agents.strengthener.templates.qmj_composite import (
        _z_score_per_month,
    )
    df = pd.DataFrame({
        "month": pd.to_datetime(["2020-01-31"]*3 + ["2020-02-29"]*3),
        "x":     [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
    })
    z = _z_score_per_month(df, "x")
    # Within each month, z should span roughly [-1.22, 1.22] for 3 obs
    assert z.iloc[0] < 0 and z.iloc[2] > 0
    assert z.iloc[3] < 0 and z.iloc[5] > 0
    # And mean ≈ 0 per month
    assert abs(z.iloc[:3].mean()) < 1e-9
    assert abs(z.iloc[3:].mean()) < 1e-9
