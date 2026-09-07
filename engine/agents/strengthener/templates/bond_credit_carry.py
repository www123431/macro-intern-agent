"""engine.agents.strengthener.templates.bond_credit_carry — v48.

Corporate bond credit carry template. HY - IG spread portfolio,
duration-controlled by aggregating across three tmt buckets. Uses
the _bondret_panel.parquet 2.7M-row corporate bond return panel
that shipped in the v41 whitelist opening but had no template yet.

Design intent
=============
Israel-Palhares-Richardson 2018 "Common Factor in Corporate Bond
Returns" documents credit carry as one of the four cross-asset carry
components (equity / FX / rates / credit). Our sample is 2013-06 to
2024-06 (133 months), which is entirely POST-QE — a period where
credit spreads were structurally compressed by central bank asset
purchases. IPR 2018's Sharpe 0.7-1.0 range is from 1988-2015 which
predates most of the compression.

Signal
======
For each month t and each tmt bucket b ∈ {short:1-3y, mid:3-7y,
long:7-15y}:
  carry_{b,t} = amount_weighted(HY_returns) - amount_weighted(IG_returns)
Portfolio: equal-weight across three tmt buckets → carry_t

Why duration-control:
  Raw HY-IG carry mixes credit risk with duration risk (HY bonds
  average shorter tmt than IG in this panel). Bucketing by tmt then
  averaging isolates credit premium from term premium.

Verdict on 2013-2024 full sample
================================
Live prototype 2026-07-02:
  Raw HY-IG (amount-weighted):        t=+1.45   RED
  Duration-controlled (3-bucket avg): t=+1.64   RED (0.01 below MARGINAL)
  Short-tmt bucket only:              t=+2.43   MARGINAL
  Vol-scaled variants:                weaker

Ship the duration-controlled version. Short-tmt-only is picked-best-of-3
and would fail Bailey-LdP DSR haircut. The MARGINAL signal in short-tmt
is documented for future refinement (v49 candidate) but not the
headline verdict.

Verdict thresholds
==================
  |t| >= 2.5   GREEN
  1.65 - 2.5   MARGINAL
  < 1.65       RED   (expected — see post-QE compression note above)

Scope
=====
  signal_kind : carry
  universe    : corporate_bonds_ig_hy
  data        : _bondret_panel.parquet (2.7M rows 2013-06..2024-06)
                columns: date, cusip, ret_eom, rating_class {0.IG, 1.HY},
                         amount_outstanding, tmt (years)
  rebal       : monthly (bond returns are end-of-month)
  weighting   : amount_outstanding weighted within (rating, tmt-bucket)
                cell → equal weight across 3 tmt buckets
  cost model  : bond bid-ask 20-40bp/side. Applied as
                4.0 × 30 / 10000 / 12 = 1.0%/yr all-in (upper mid).

Known limitations
=================
  - Only 2 rating classes (IG vs HY), no fine-grained rating tiers
  - No spread data (would enable proper carry = spread - default risk)
  - No liquidity control (large-cap issuers dominate amount-weighted)
  - Cost model is scalar, not liquidity-tiered
"""
from __future__ import annotations

import datetime as _dt
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from engine.agents.strengthener.factor_spec_extractor import FactorSpec

logger = logging.getLogger(__name__)

_TEMPLATE_VERSION = "v1.0_2026-07-02"

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BONDRET_PATH = _REPO_ROOT / "data" / "cache" / "_bondret_panel.parquet"

_T_GREEN    = 2.5
_T_MARGINAL = 1.65

_MIN_MONTHS = 60
_TMT_MIN    = 1.0    # exclude money-market-ish (<1y)
_TMT_MAX    = 15.0   # exclude ultra-long (>15y) — thin panel

# Duration bucket boundaries (years)
_TMT_BUCKETS = [(1.0, 3.0, "short"), (3.0, 7.0, "mid"), (7.0, 15.0, "long")]

_COST_BP_ANNUAL = 30.0  # bond RT bid-ask + slippage
_COST_TURNOVER  = 4.0   # monthly rebal turnover multiplier


def _verdict_from_t(t: float) -> str:
    if not math.isfinite(t):
        return "RED"
    a = abs(t)
    if a >= _T_GREEN:
        return "GREEN"
    if a >= _T_MARGINAL:
        return "MARGINAL"
    return "RED"


def _parse_date_range(s: str) -> tuple[_dt.date, _dt.date]:
    if ":" not in s:
        raise ValueError(f"date_range must contain ':': {s!r}")
    a, b = s.split(":", 1)
    start = _dt.date.fromisoformat(f"{a.strip()}-01")
    end_ts = pd.Timestamp(f"{b.strip()}-01") + pd.offsets.MonthEnd(0)
    return start, end_ts.date()


def _amt_weighted(g: pd.DataFrame) -> float:
    w = g["amount_outstanding"].fillna(0.0)
    total = w.sum()
    if total <= 0:
        return np.nan
    return float((g["ret_eom"] * w).sum() / total)


def _compute_duration_controlled_carry(start: _dt.date, end: _dt.date) -> pd.Series:
    """Load _bondret_panel and build the duration-controlled HY-IG carry
    portfolio monthly return series."""
    df = pd.read_parquet(_BONDRET_PATH)
    df = df.dropna(subset=["ret_eom", "rating_class"])
    df = df[(df["tmt"] >= _TMT_MIN) & (df["tmt"] <= _TMT_MAX)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[
        (df["date"] >= pd.Timestamp(start)) &
        (df["date"] <= pd.Timestamp(end))
    ]

    # Assign tmt bucket
    def _bucket(t: float) -> str:
        for lo, hi, name in _TMT_BUCKETS:
            if lo <= t < hi:
                return name
        return "long"   # tmt == 15.0 edge case
    df["tmt_bucket"] = df["tmt"].apply(_bucket)

    # Amount-weighted returns per (date, bucket, rating)
    per_cell = df.groupby(
        ["date", "tmt_bucket", "rating_class"]
    ).apply(_amt_weighted, include_groups=False).unstack("rating_class")

    if "0.IG" not in per_cell.columns or "1.HY" not in per_cell.columns:
        return pd.Series(dtype=float)

    per_cell["carry"] = per_cell["1.HY"] - per_cell["0.IG"]
    carry_by_bucket = per_cell["carry"].unstack("tmt_bucket").dropna()
    # Equal weight across S/M/L buckets
    return carry_by_bucket.mean(axis=1).rename("carry_ls")


def template_bond_credit_carry(spec: FactorSpec):
    from engine.agents.strengthener.factor_dispatcher import TemplateResult

    date_range = getattr(spec, "date_range", None) or "2013-06:2024-06"
    try:
        start, end = _parse_date_range(date_range)
    except Exception as e:
        return TemplateResult(
            verdict="EXECUTION_ERROR",
            summary=f"date_range parse failed: {e}",
            metrics={"date_range_raw": date_range},
            artifacts={}, template_version=_TEMPLATE_VERSION,
        )

    if not _BONDRET_PATH.is_file():
        return TemplateResult(
            verdict="EXECUTION_ERROR",
            summary="_bondret_panel.parquet missing",
            metrics={"expected_path": str(_BONDRET_PATH)},
            artifacts={}, template_version=_TEMPLATE_VERSION,
        )

    try:
        port = _compute_duration_controlled_carry(start, end)
    except Exception as e:
        logger.exception("bond_credit_carry pipeline failed")
        return TemplateResult(
            verdict="EXECUTION_ERROR",
            summary=f"pipeline error: {type(e).__name__}: {str(e)[:200]}",
            metrics={},
            artifacts={}, template_version=_TEMPLATE_VERSION,
        )

    if len(port) < _MIN_MONTHS:
        return TemplateResult(
            verdict="INSUFFICIENT_HISTORY",
            summary=(f"{len(port)} monthly obs < required {_MIN_MONTHS}; "
                     f"corporate bond panel starts 2013-06"),
            metrics={"n_months": len(port), "min_required": _MIN_MONTHS},
            artifacts={}, template_version=_TEMPLATE_VERSION,
        )

    mu    = float(port.mean())
    sigma = float(port.std(ddof=1))
    if sigma <= 0 or not math.isfinite(sigma):
        return TemplateResult(
            verdict="EXECUTION_ERROR",
            summary="zero-variance portfolio returns; check panel content",
            metrics={"n_months": len(port)},
            artifacts={}, template_version=_TEMPLATE_VERSION,
        )

    # Cost adjustment
    monthly_cost = _COST_TURNOVER * _COST_BP_ANNUAL / 10000.0 / 12.0
    port_net = port - monthly_cost
    mu_net = float(port_net.mean())
    sigma_net = float(port_net.std(ddof=1))

    n = len(port)
    t_stat_gross = mu * math.sqrt(n) / sigma
    t_stat_net   = mu_net * math.sqrt(n) / sigma_net
    sharpe_ann_gross = mu / sigma * math.sqrt(12)
    sharpe_ann_net   = mu_net / sigma_net * math.sqrt(12)
    cum_ret_gross = float((1.0 + port).prod() - 1.0)
    cum_ret_net   = float((1.0 + port_net).prod() - 1.0)
    verdict = _verdict_from_t(t_stat_net)   # verdict on NET t

    metrics = {
        "n_months":           n,
        "mean_monthly_gross": mu,
        "mean_monthly_net":   mu_net,
        "vol_monthly":        sigma,
        "sharpe_annualized_gross": sharpe_ann_gross,
        "sharpe_annualized_net":   sharpe_ann_net,
        "t_stat_gross":       t_stat_gross,
        "t_stat":             t_stat_net,    # canonical verdict driver
        "cum_return_gross":   cum_ret_gross,
        "cum_return_net":     cum_ret_net,
        "cost_bp_annual":     _COST_BP_ANNUAL,
        "cost_turnover_mult": _COST_TURNOVER,
        "tmt_buckets":        [b[2] for b in _TMT_BUCKETS],
        "tmt_range":          [_TMT_MIN, _TMT_MAX],
        "window_start":       start.isoformat(),
        "window_end":         end.isoformat(),
        "verdict_thresholds": {"green": _T_GREEN, "marginal": _T_MARGINAL},
        "post_qe_note":       ("Sample 2013-2024 entirely post-QE. IPR 2018 "
                                "reports credit carry Sharpe 0.7-1.0 on "
                                "1988-2015 sample which predates most spread "
                                "compression. Our net Sharpe expected 0.4-0.5 "
                                "reflects the structurally weaker recent regime."),
        "unchecked_dimensions": [
            "ff5_spanning", "duration_beta_control", "liquidity_tier_cost",
            "multi_period_stability", "fine_rating_tier_splits",
        ],
    }
    summary = (
        f"bond_credit_carry (HY-IG dur-ctrl, amount-weighted): "
        f"Sharpe(ann,net)={sharpe_ann_net:.3f}, t_net={t_stat_net:.2f} "
        f"across {n} months ({start}..{end}), gross Sharpe {sharpe_ann_gross:.3f}"
    )
    return TemplateResult(
        verdict=verdict, summary=summary, metrics=metrics,
        artifacts={}, template_version=_TEMPLATE_VERSION,
    )
