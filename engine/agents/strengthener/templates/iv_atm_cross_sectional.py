"""engine.agents.strengthener.templates.iv_atm_cross_sectional — v52.

Cross-sectional options implied-vol level (IV_ATM) factor. First
options-based cross-sectional template (existing vrp_spx / spx_skew /
vrp_treasury are time-series). Opens the OPTIONS_CROSS_SEC family.

Design intent
=============
Xing-Zhang-Zhao 2010 "What Does the Individual Option Volatility
Smirk Tell Us About Future Equity Returns?" documents cross-sectional
alpha from option-implied signals. This template uses the simplest
version: LEVEL of ATM implied vol.

Two directional hypotheses in the literature:
  (a) Ang-Hodrick-Xing-Zhang 2006: high REALIZED-vol stocks earn LOW
      returns (idiosyncratic vol puzzle) → LONG low-iv should win
  (b) Cao-Han 2013: high IMPLIED-vol stocks reflect option-implied
      information → LONG high-iv-anomaly stocks

Live prototype 2026-07-07 07:20 UTC on 2013-2024 sample:
  news_ess tercile L/S:      t=-1.07  (contra-sentiment mild)
  iv_atm tercile L/S:        t=+1.61  ← this template (LONG high-iv)
  iv_skew tercile L/S:       t=+1.16
  sue tercile L/S:           t=+0.12  (dead)
  mom_12_1 tercile L/S:      t=+0.99  (RED — momentum decayed)

iv_atm LONG-high (Cao-Han direction) is the strongest signal at gross
t=+1.61 — on the MARGINAL/RED border. Cost adjustment will drop it
into RED. This is HONEST: option-implied level as a stand-alone
cross-sec factor has decayed post-QE, consistent with the broader
narrative of factor compression in ML-era markets.

Scope
=====
  signal_kind : cross_sectional_rank
  universe    : us_equities_top_3000
  data        : _ml_feature_panel.parquet (iv_atm column)
  signal      : cross-sectional rank by iv_atm
  portfolio   : tercile L/S dollar-neutral, LONG high-iv, SHORT low-iv
                (Cao-Han 2013 direction)
  rebal       : monthly
  cost model  : RT_EQ ≈ 30bp/side × 4× monthly turnover = 1.0%/yr

Verdict thresholds
==================
  |t| >= 2.5   GREEN
  1.65 - 2.5   MARGINAL
  < 1.65       RED   (expected — gross 1.61 net drops below 1.65)

Known limitations
=================
  - Signal direction ambiguous in literature (Ang et al vs Cao-Han).
    We ship LONG-high-iv based on this-window empirical direction.
  - No FF5+MOM spanning residualization
  - No liquidity-tiered cost model (small caps in ivol tercile
    incur higher effective cost)
  - iv_atm coverage 69% of feature panel — smaller-cap thin
"""
from __future__ import annotations

import datetime as _dt
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from engine.agents.strengthener.factor_spec_extractor import FactorSpec

logger = logging.getLogger(__name__)

_TEMPLATE_VERSION = "v1.0_2026-07-02"

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FEATURE_PANEL_PATH = _REPO_ROOT / "data" / "cache" / "_ml_feature_panel.parquet"

_T_GREEN    = 2.5
_T_MARGINAL = 1.65
_MIN_MONTHS = 60
_MIN_XSEC   = 60

_COST_BP_ANNUAL = 30.0
_COST_TURNOVER  = 4.0


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


def _tercile_ls(df: pd.DataFrame, sort_col: str) -> pd.Series:
    def _one_month(g: pd.DataFrame) -> float:
        s = g.dropna(subset=[sort_col])
        if len(s) < _MIN_XSEC:
            return np.nan
        s = s.sort_values(sort_col)
        n = len(s)
        k = max(1, n // 3)
        return float(s["y"].tail(k).mean() - s["y"].head(k).mean())
    return df.groupby("month").apply(_one_month, include_groups=False).dropna()


def template_iv_atm_cross_sectional(spec: FactorSpec):
    from engine.agents.strengthener.factor_dispatcher import TemplateResult

    date_range = getattr(spec, "date_range", None) or "2013-11:2024-06"
    try:
        start, end = _parse_date_range(date_range)
    except Exception as e:
        return TemplateResult(
            verdict="EXECUTION_ERROR",
            summary=f"date_range parse failed: {e}",
            metrics={"date_range_raw": date_range},
            artifacts={}, template_version=_TEMPLATE_VERSION,
        )

    if not _FEATURE_PANEL_PATH.is_file():
        return TemplateResult(
            verdict="EXECUTION_ERROR",
            summary="_ml_feature_panel.parquet missing",
            metrics={},
            artifacts={}, template_version=_TEMPLATE_VERSION,
        )

    try:
        df = pd.read_parquet(_FEATURE_PANEL_PATH)
        df = df.dropna(subset=["y", "iv_atm"])
        df = df[
            (df["month"] >= pd.Timestamp(start)) &
            (df["month"] <= pd.Timestamp(end))
        ].copy()
        port = _tercile_ls(df, "iv_atm")
    except Exception as e:
        logger.exception("iv_atm_cross_sectional pipeline failed")
        return TemplateResult(
            verdict="EXECUTION_ERROR",
            summary=f"pipeline error: {type(e).__name__}: {str(e)[:200]}",
            metrics={},
            artifacts={}, template_version=_TEMPLATE_VERSION,
        )

    if len(port) < _MIN_MONTHS:
        return TemplateResult(
            verdict="INSUFFICIENT_HISTORY",
            summary=f"{len(port)} monthly obs < {_MIN_MONTHS}",
            metrics={"n_months": len(port)},
            artifacts={}, template_version=_TEMPLATE_VERSION,
        )

    monthly_cost = _COST_TURNOVER * _COST_BP_ANNUAL / 10000.0 / 12.0
    port_net = port - monthly_cost
    n = len(port)
    mu_g, sd_g = float(port.mean()), float(port.std(ddof=1))
    mu_n, sd_n = float(port_net.mean()), float(port_net.std(ddof=1))
    if sd_g <= 0:
        return TemplateResult(
            verdict="EXECUTION_ERROR",
            summary="zero-variance portfolio",
            metrics={"n_months": n},
            artifacts={}, template_version=_TEMPLATE_VERSION,
        )
    t_g = mu_g * math.sqrt(n) / sd_g
    t_n = mu_n * math.sqrt(n) / sd_n
    sh_g = mu_g / sd_g * math.sqrt(12)
    sh_n = mu_n / sd_n * math.sqrt(12)
    verdict = _verdict_from_t(t_n)

    metrics = {
        "n_months": n,
        "mean_monthly_gross": mu_g, "mean_monthly_net": mu_n,
        "vol_monthly": sd_g,
        "sharpe_annualized_gross": sh_g, "sharpe_annualized_net": sh_n,
        "t_stat_gross": t_g, "t_stat": t_n,
        "cum_return_gross": float((1 + port).prod() - 1),
        "cum_return_net":   float((1 + port_net).prod() - 1),
        "cost_bp_annual": _COST_BP_ANNUAL, "cost_turnover_mult": _COST_TURNOVER,
        "window_start": start.isoformat(), "window_end": end.isoformat(),
        "verdict_thresholds": {"green": _T_GREEN, "marginal": _T_MARGINAL},
        "signal_direction": "LONG_HIGH_IV (Cao-Han 2013 direction)",
        "unchecked_dimensions": [
            "ff5_spanning", "liquidity_tiered_cost", "small_cap_ivol_bias",
            "signal_direction_stability_pre_2013",
        ],
    }
    summary = (
        f"iv_atm cross-sec tercile L/S: Sharpe(ann,net)={sh_n:.3f}, "
        f"t_net={t_n:.2f} across {n} months ({start}..{end}); "
        f"gross t={t_g:.2f}"
    )
    return TemplateResult(
        verdict=verdict, summary=summary, metrics=metrics,
        artifacts={}, template_version=_TEMPLATE_VERSION,
    )
