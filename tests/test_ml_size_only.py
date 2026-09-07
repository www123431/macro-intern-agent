"""tests/test_ml_size_only.py — v46 template."""
from __future__ import annotations

import datetime as _dt

import pytest


def test_verdict_from_t_boundaries():
    from engine.agents.strengthener.templates.ml_size_only import (
        _verdict_from_t, _T_GREEN, _T_MARGINAL,
    )
    assert _verdict_from_t(3.0)            == "GREEN"
    assert _verdict_from_t(_T_GREEN)       == "GREEN"
    assert _verdict_from_t(_T_GREEN - 0.01) == "MARGINAL"
    assert _verdict_from_t(_T_MARGINAL - 0.01) == "RED"
    assert _verdict_from_t(float("nan"))   == "RED"


def test_contract_registered_and_fresh():
    from engine.agents.strengthener.templates._template_contract import (
        contract_for_scope,
    )
    c = contract_for_scope("ml_size_only", "us_equities_top_3000")
    assert c is not None
    assert c.is_fresh()
    assert c.canonical_paper_id == "banz_1981"


def test_ml_size_only_in_signal_kinds_enum():
    from engine.agents.strengthener.factor_spec_extractor import SIGNAL_KINDS
    assert "ml_size_only" in SIGNAL_KINDS


def test_template_registry_has_ml_size_only():
    from engine.agents.strengthener.factor_dispatcher import TEMPLATE_REGISTRY
    assert TEMPLATE_REGISTRY.get("ml_size_only") is not None


def test_ml_size_and_ml_ensemble_are_separate_contracts():
    """Regression: v45 and v46 contracts must NOT collide on scope."""
    from engine.agents.strengthener.templates._template_contract import (
        contract_for_scope,
    )
    size = contract_for_scope("ml_size_only", "us_equities_top_3000")
    ens  = contract_for_scope("ml_ensemble",  "us_equities_top_3000")
    assert size is not None and ens is not None
    assert size.template_name != ens.template_name
    assert size.canonical_paper_id != ens.canonical_paper_id


def test_dispatcher_routes_ml_size_only(monkeypatch):
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec

    calls: list[str] = []
    def _fake(spec):
        calls.append("ml_size_only")
        return fd.TemplateResult(
            verdict="GREEN", summary="stub", metrics={},
            artifacts={}, template_version="stub",
        )
    monkeypatch.setattr(
        "engine.agents.strengthener.templates.ml_size_only.template_ml_size_only",
        _fake,
    )
    spec = FactorSpec(
        hypothesis_id="t", signal_kind="ml_size_only",
        universe="us_equities_top_3000", date_range="2015-01:2020-12",
        signal_inputs=("crsp.msf.mktcap",), rebal="monthly",
        weighting="tercile_long_short_dollar_neutral",
        expected_holding_period="monthly", min_obs_months=60,
        pit_audits=("restatement",), cost_model="none",
        rationale="test", extracted_ts="2026-07-02T05:00:00Z",
        model="test",
    )
    r = fd._ml_size_only_template_lazy(spec)
    assert calls == ["ml_size_only"]
    assert r.verdict == "GREEN"


def test_v46_uses_only_log_mcap(tmp_path):
    """Regression: template must reference exactly one feature (log_mcap)."""
    from engine.agents.strengthener.templates.ml_size_only import _FEATURES
    assert _FEATURES == ("log_mcap",), (
        f"v46 must be size-only; got features {_FEATURES}"
    )
