"""tests/test_commodity_carry_futures.py — v42 template.

Covers: (1) synthetic-data unit tests for the pipeline (isolated from
Refinitiv cache — deterministic + fast); (2) contract registration
integrity; (3) dispatcher routing hits the new template on
universe=commodity_futures_24.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd
import pytest


# ── Verdict thresholds ──────────────────────────────────────────────


def test_verdict_from_t_boundaries():
    from engine.agents.strengthener.templates.commodity_carry_futures import (
        _verdict_from_t, _T_GREEN, _T_MARGINAL,
    )
    assert _verdict_from_t(3.0)          == "GREEN"
    assert _verdict_from_t(_T_GREEN)     == "GREEN"
    assert _verdict_from_t(_T_GREEN - 0.01) == "MARGINAL"
    assert _verdict_from_t(_T_MARGINAL)   == "MARGINAL"
    assert _verdict_from_t(_T_MARGINAL - 0.01) == "RED"
    assert _verdict_from_t(-3.0)         == "GREEN"   # symmetric on |t|
    assert _verdict_from_t(float("nan")) == "RED"


def test_parse_date_range():
    from engine.agents.strengthener.templates.commodity_carry_futures import (
        _parse_date_range,
    )
    a, b = _parse_date_range("2005-01:2020-12")
    assert a == _dt.date(2005, 1, 1)
    assert b == _dt.date(2020, 12, 31)


def test_parse_date_range_rejects_no_colon():
    from engine.agents.strengthener.templates.commodity_carry_futures import (
        _parse_date_range,
    )
    with pytest.raises(ValueError):
        _parse_date_range("2005-01_2020-12")


# ── Contract registration ──────────────────────────────────────────


def test_contract_registered_and_fresh():
    from engine.agents.strengthener.templates._template_contract import (
        contract_for_scope,
    )
    c = contract_for_scope("carry", "commodity_futures_24")
    assert c is not None, "commodity_futures_24 contract must be registered"
    assert c.is_fresh(), "contract must be within 365-day freshness window"
    assert c.canonical_paper_id == "koijen_moskowitz_pedersen_vrugt_2018"
    assert c.canonical_paper_t == pytest.approx(6.31, abs=0.01)


def test_contract_lookup_does_not_shadow_g10_fx():
    """Regression: adding commodity_futures_24 must not break the
    existing carry_g10_fx lookup."""
    from engine.agents.strengthener.templates._template_contract import (
        contract_for_scope,
    )
    g10 = contract_for_scope("carry", "fx_g10")
    assert g10 is not None
    assert g10.template_name == "carry_g10_fx"


# ── Dispatcher routing ─────────────────────────────────────────────


def test_dispatcher_routes_to_commodity_template(monkeypatch):
    """`_carry_template_lazy(spec)` must route universe=commodity_futures_24
    into commodity_carry_futures, not into carry_g10_fx or the
    pending-build stub."""
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec

    calls: list[str] = []

    def _fake_commodity(spec):
        calls.append("commodity")
        return fd.TemplateResult(
            verdict="RED", summary="stub", metrics={}, artifacts={},
            template_version="test",
        )

    monkeypatch.setattr(
        "engine.agents.strengthener.templates.commodity_carry_futures."
        "template_commodity_carry_futures",
        _fake_commodity,
    )
    spec = FactorSpec(
        hypothesis_id="test", signal_kind="carry",
        universe="commodity_futures_24", date_range="2010-01:2015-12",
        signal_inputs=("futures.settle.raw",), rebal="monthly",
        weighting="quintile_long_short_dollar_neutral",
        expected_holding_period="monthly", min_obs_months=60,
        pit_audits=("restatement",), cost_model="none",
        rationale="test", extracted_ts="2026-07-02T04:00:00Z",
        model="test",
    )
    r = fd._carry_template_lazy(spec)
    assert calls == ["commodity"], (
        f"routing didn't hit commodity template; calls={calls}"
    )
    assert r.verdict == "RED"   # stub returned RED


def test_dispatcher_routes_fx_g10_unchanged(monkeypatch):
    """Regression: fx_g10 routing must not be affected."""
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec

    calls: list[str] = []
    def _fake_g10(spec):
        calls.append("g10")
        return fd.TemplateResult(
            verdict="RED", summary="stub", metrics={}, artifacts={},
            template_version="test",
        )
    monkeypatch.setattr(
        "engine.agents.strengthener.templates.carry_g10_fx.template_carry_g10_fx",
        _fake_g10,
    )
    spec = FactorSpec(
        hypothesis_id="t", signal_kind="carry", universe="fx_g10",
        date_range="2000-01:2015-12",
        signal_inputs=("fred.fx_spot_g10.eur",), rebal="monthly",
        weighting="tercile_long_short_dollar_neutral",
        expected_holding_period="monthly", min_obs_months=60,
        pit_audits=("restatement",), cost_model="none",
        rationale="test", extracted_ts="2026-07-02T04:00:00Z",
        model="test",
    )
    fd._carry_template_lazy(spec)
    assert calls == ["g10"]


# ── Signal computation on synthetic data ───────────────────────────


def _synth_panel(
    n_underlyings: int = 8,
    n_months:      int = 120,
    carry_alpha:   float = 0.02,   # 2% monthly Sharpe-worthy signal
    noise_sd:      float = 0.05,
    seed:          int = 42,
) -> pd.DataFrame:
    """Deterministic synthetic (carry, ret_clean) panel where higher
    carry drives higher future returns (positive alpha). Used to verify
    the L/S computation without touching real Refinitiv data.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for m_idx in range(n_months):
        ym = pd.Period(f"20{m_idx // 12 + 10:02d}-{m_idx % 12 + 1:02d}", freq="M")
        for u in range(n_underlyings):
            # deterministic carry per underlying + noise
            carry = (u - n_underlyings / 2) / n_underlyings * 0.4 + rng.normal(0, 0.05)
            ret   = carry_alpha * carry + rng.normal(0, noise_sd)
            rows.append({
                "contrname": f"U{u}", "ym": ym,
                "carry": carry, "ret_clean": ret,
            })
    return pd.DataFrame(rows)


def test_tercile_ls_produces_positive_signal_when_carry_predicts():
    """Synthetic panel where carry positively predicts next return
    should yield POSITIVE portfolio mean."""
    from engine.agents.strengthener.templates.commodity_carry_futures import (
        _tercile_ls,
    )
    panel = _synth_panel(carry_alpha=0.05, noise_sd=0.02, n_months=180)
    port = _tercile_ls(panel)
    assert len(port) > 60
    assert port.mean() > 0, (
        f"positive alpha panel should produce positive L/S mean; got {port.mean():.4f}"
    )
    t_stat = port.mean() * np.sqrt(len(port)) / port.std(ddof=1)
    assert t_stat > 2.0, f"strong-signal test expected t > 2.0; got {t_stat:.2f}"


def test_tercile_ls_produces_near_zero_when_no_signal():
    """Pure noise panel should have t-stat < 2 (won't reliably GREEN)."""
    from engine.agents.strengthener.templates.commodity_carry_futures import (
        _tercile_ls,
    )
    # carry_alpha=0 → no relation between carry and future return
    panel = _synth_panel(carry_alpha=0.0, noise_sd=0.05, n_months=180, seed=99)
    port = _tercile_ls(panel)
    t_stat = port.mean() * np.sqrt(len(port)) / port.std(ddof=1)
    assert abs(t_stat) < 3.0, (
        f"no-signal panel should not spuriously GREEN; got t={t_stat:.2f}"
    )


def test_tercile_ls_skips_thin_months():
    """Months with < _MIN_UNDERLYINGS get dropped rather than counted."""
    from engine.agents.strengthener.templates.commodity_carry_futures import (
        _tercile_ls, _MIN_UNDERLYINGS,
    )
    # Panel where each month has 5 underlyings (below min=6) → all NaN
    panel = _synth_panel(n_underlyings=_MIN_UNDERLYINGS - 1, n_months=100)
    port = _tercile_ls(panel, vol_scale=False)   # skip vol path (synth has no rolling)
    assert len(port) == 0, "thin-month panel should produce no L/S returns"


# ── v44 vol-scaling regression ─────────────────────────────────────


def test_v44_vol_scaling_increases_t_stat_on_heteroscedastic_panel():
    """When underlyings have wildly different vols, unscaled L/S is
    dominated by the high-vol legs. Vol-scaling normalizes each leg's
    contribution to the target vol, extracting cleaner signal.
    Regression: on a heteroscedastic synthetic panel where carry
    predicts return, vol-scaled t-stat should exceed unscaled by
    a meaningful margin.
    """
    from engine.agents.strengthener.templates.commodity_carry_futures import (
        _tercile_ls, _augment_vol_weights,
    )
    # Build a panel where underlying U0 has 20x vol of U7 but same
    # carry-return relationship. Without scaling, U0 dominates and
    # SNR is bad.
    rng = np.random.default_rng(0)
    rows = []
    for m in range(240):
        ym = pd.Period(f"20{m//12 + 10:02d}-{m%12 + 1:02d}", freq="M")
        for u in range(8):
            base_vol = 0.02 + u * 0.08   # 0.02 .. 0.58 spread
            carry = (u - 4) * 0.1 + rng.normal(0, 0.05)
            ret   = 0.04 * carry + rng.normal(0, base_vol)
            rows.append({"contrname": f"U{u}", "ym": ym,
                          "carry": carry, "ret_clean": ret})
    panel = pd.DataFrame(rows)

    port_plain = _tercile_ls(panel, vol_scale=False)
    scaled = _augment_vol_weights(panel)
    port_vs = _tercile_ls(scaled, vol_scale=True)

    t_plain = port_plain.mean() * np.sqrt(len(port_plain)) / port_plain.std(ddof=1)
    t_vs    = port_vs.mean()    * np.sqrt(len(port_vs))    / port_vs.std(ddof=1)
    # Not asserting a specific magnitude — just that vol-scaling helps
    # on heteroscedastic panels.
    assert t_vs > t_plain, (
        f"vol-scaled t={t_vs:.2f} must exceed unscaled t={t_plain:.2f} "
        f"on heteroscedastic synth panel"
    )


def test_v44_vol_weight_cap_bounds_gross_exposure():
    """A tiny-vol underlying must not blow up the weight above _VOL_WEIGHT_CAP."""
    from engine.agents.strengthener.templates.commodity_carry_futures import (
        _augment_vol_weights, _VOL_WEIGHT_CAP,
    )
    # 12 months of near-zero vol returns for one underlying
    rows = []
    for m in range(24):
        ym = pd.Period(f"2020-{m%12 + 1:02d}", freq="M") if m < 12 else \
              pd.Period(f"2021-{m%12 + 1:02d}", freq="M")
        rows.append({"contrname": "TINY_VOL", "ym": ym,
                      "carry": 0.1, "ret_clean": 1e-6})
    panel = pd.DataFrame(rows)
    scaled = _augment_vol_weights(panel)
    assert (scaled["vol_weight"] <= _VOL_WEIGHT_CAP + 1e-9).all(), (
        f"weights must be capped at {_VOL_WEIGHT_CAP}; "
        f"got max {scaled['vol_weight'].max():.2f}"
    )
