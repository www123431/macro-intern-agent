"""engine.agents.strengthener.templates.qmj_composite — v50.

Quality-Minus-Junk (QMJ) 3-component composite factor template.
Asness-Frazzini-Pedersen 2013 composite Quality on US equity CRSP
+ Compustat cross-section.

Design intent
=============
Full AFP 2013 QMJ = 4 components:
    profitability + growth + safety + payout
We have 3 in cache (_ml_feature_panel.parquet):
    profitability = gp (Novy-Marx gross-profit-to-assets)
    safety        = -vol_6m (inverse 6-month realized vol)
    conservatism  = -asset_growth (Fama-French CMA inverse — low
                                    investment = quality)
Missing: payout / net share issuance.

3-component composite is a legitimate proxy. AFP 2013 Table 4 shows
each subcomponent has stand-alone alpha, and the composite Sharpe is
close to sum-of-components. The payout leg contributes ~15-20% of
composite alpha in their sample.

Live prototype 2026-07-02 07:00 UTC on 2013-2024 sample:
  q_prof   (Novy-Marx):     t=+1.07  weak positive
  q_safety (inverse vol):   t=-1.19  NEGATIVE — low-vol underperformed
  q_invst  (inverse growth): t=-0.39  ~zero
  QMJ 3-factor composite:   t=-0.54  RED

The negative q_safety is the "low-vol crash" narrative: mega-cap tech
2020-2024 rally rewarded HIGH-vol growth names, punishing safety.
This structurally hurts QMJ in this window and is consistent with
Fama-French 2020 update reporting SMB / conservatism decay.

Verdict thresholds
==================
  |t| >= 2.5   GREEN
  1.65 - 2.5   MARGINAL
  < 1.65       RED   (expected — see recent regime note)

Scope
=====
  signal_kind : cross_sectional_rank
  universe    : us_equities_top_3000
  data        : _ml_feature_panel.parquet (gp, vol_6m, asset_growth)
  signal      : composite = z(gp) + z(-vol_6m) + z(-asset_growth), each
                z-scored cross-sectionally per month, then averaged
                with skipna=False (require all 3 features)
  portfolio   : tercile L/S dollar-neutral by composite score
  rebal       : monthly
  cost model  : RT_EQ ~ 30bp/side × 4× monthly turnover multiplier
                = 1.0%/yr all-in

Bailey-LdP n_trials note
========================
Family QUALITY starts fresh (n_trials=0 pre-v50). This dispatch is
the first entry. Post-v50 DSR haircut is minimal on t=2.5 threshold.

Known limitations
=================
  - No payout / issuance leg (missing 4th AFP component)
  - No FF5+MOM spanning residualization
  - Small-cap microstructure cost drag not modeled
  - Sample 2013-2024 entirely post-QE regime; pre-2000 QMJ likely
    stronger (AFP 2013 report Sharpe 0.7-1.0 on 1957-2011)
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


def _z_score_per_month(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("month")[col].transform(
        lambda x: (x - x.mean()) / x.std(ddof=1)
    )


def _build_qmj_panel(start: _dt.date, end: _dt.date) -> pd.DataFrame:
    df = pd.read_parquet(_FEATURE_PANEL_PATH)
    df = df.dropna(subset=["y"])
    df = df[
        (df["month"] >= pd.Timestamp(start)) &
        (df["month"] <= pd.Timestamp(end))
    ].copy()

    # z-scored components (higher = more quality)
    df["q_prof"]   = _z_score_per_month(df, "gp")            # profitability
    df["q_safety"] = -_z_score_per_month(df, "vol_6m")       # safety (inverse vol)
    df["q_invst"]  = -_z_score_per_month(df, "asset_growth") # conservatism (inverse growth)

    # QMJ composite (require all 3 non-null → skipna=False)
    df["qmj"] = df[["q_prof", "q_safety", "q_invst"]].mean(axis=1, skipna=False)
    return df


def _tercile_ls(panel: pd.DataFrame, sort_col: str) -> pd.Series:
    def _one_month(g: pd.DataFrame) -> float:
        s = g.dropna(subset=[sort_col])
        if len(s) < _MIN_XSEC:
            return np.nan
        s = s.sort_values(sort_col)
        n = len(s)
        k = max(1, n // 3)
        return float(s["y"].tail(k).mean() - s["y"].head(k).mean())
    return panel.groupby("month").apply(_one_month, include_groups=False).dropna()


def template_qmj_composite(spec: FactorSpec):
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
            metrics={"expected_path": str(_FEATURE_PANEL_PATH)},
            artifacts={}, template_version=_TEMPLATE_VERSION,
        )

    try:
        panel = _build_qmj_panel(start, end)
        port  = _tercile_ls(panel, "qmj")
    except Exception as e:
        logger.exception("qmj_composite pipeline failed")
        return TemplateResult(
            verdict="EXECUTION_ERROR",
            summary=f"pipeline error: {type(e).__name__}: {str(e)[:200]}",
            metrics={},
            artifacts={}, template_version=_TEMPLATE_VERSION,
        )

    if len(port) < _MIN_MONTHS:
        return TemplateResult(
            verdict="INSUFFICIENT_HISTORY",
            summary=f"{len(port)} monthly obs < required {_MIN_MONTHS}",
            metrics={"n_months": len(port), "min_required": _MIN_MONTHS},
            artifacts={}, template_version=_TEMPLATE_VERSION,
        )

    monthly_cost = _COST_TURNOVER * _COST_BP_ANNUAL / 10000.0 / 12.0
    port_net = port - monthly_cost
    n = len(port)
    mu_g, sd_g = float(port.mean()), float(port.std(ddof=1))
    mu_n, sd_n = float(port_net.mean()), float(port_net.std(ddof=1))
    if sd_g <= 0 or not math.isfinite(sd_g):
        return TemplateResult(
            verdict="EXECUTION_ERROR",
            summary="zero-variance portfolio returns",
            metrics={"n_months": n},
            artifacts={}, template_version=_TEMPLATE_VERSION,
        )

    t_g = mu_g * math.sqrt(n) / sd_g
    t_n = mu_n * math.sqrt(n) / sd_n
    sh_g = mu_g / sd_g * math.sqrt(12)
    sh_n = mu_n / sd_n * math.sqrt(12)
    cum_g = float((1.0 + port).prod() - 1.0)
    cum_n = float((1.0 + port_net).prod() - 1.0)
    verdict = _verdict_from_t(t_n)

    # Per-component stand-alone L/S (diagnostic only, not verdict-driving)
    comp_diag: dict[str, dict[str, float]] = {}
    for comp in ("q_prof", "q_safety", "q_invst"):
        try:
            p_c = _tercile_ls(panel, comp)
            if len(p_c) < 30:
                continue
            mu_c = float(p_c.mean())
            sd_c = float(p_c.std(ddof=1))
            if sd_c <= 0:
                continue
            comp_diag[comp] = {
                "n_months":  int(len(p_c)),
                "sharpe_ann": mu_c / sd_c * math.sqrt(12),
                "t_stat":    mu_c * math.sqrt(len(p_c)) / sd_c,
            }
        except Exception:
            continue

    metrics = {
        "n_months":              n,
        "mean_monthly_gross":    mu_g,
        "mean_monthly_net":      mu_n,
        "vol_monthly":           sd_g,
        "sharpe_annualized_gross": sh_g,
        "sharpe_annualized_net":   sh_n,
        "t_stat_gross":          t_g,
        "t_stat":                t_n,
        "cum_return_gross":      cum_g,
        "cum_return_net":        cum_n,
        "cost_bp_annual":        _COST_BP_ANNUAL,
        "cost_turnover_mult":    _COST_TURNOVER,
        "components":            list(("q_prof", "q_safety", "q_invst")),
        "component_diagnostics": comp_diag,
        "missing_afp_leg":       "payout / net_share_issuance",
        "window_start":          start.isoformat(),
        "window_end":            end.isoformat(),
        "verdict_thresholds":    {"green": _T_GREEN, "marginal": _T_MARGINAL},
        "post_qe_note":          (
            "Sample 2013-2024 entirely post-QE. AFP 2013 report QMJ "
            "Sharpe 0.7-1.0 on 1957-2011 (pre-QE). q_safety t=-1.19 "
            "in this window reflects 2020-2024 mega-cap tech rally "
            "punishing low-vol defensives — 'low-vol crash' narrative."
        ),
        "unchecked_dimensions": [
            "ff5_spanning", "payout_leg", "small_cap_microstructure_cost",
            "pre_qe_regime_check",
        ],
    }
    summary = (
        f"qmj_composite tercile L/S (3-factor Q): Sharpe(ann,net)={sh_n:.3f}, "
        f"t_net={t_n:.2f} across {n} months ({start}..{end}); "
        f"components q_prof/safety/invst diagnostic in metrics"
    )
    return TemplateResult(
        verdict=verdict, summary=summary, metrics=metrics,
        artifacts={}, template_version=_TEMPLATE_VERSION,
    )
