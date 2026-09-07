"""engine.agents.strengthener.templates.ml_size_only — v46.

Companion to v45 ml_ensemble_combiner. Same architecture, one feature:
log_mcap (SIZE). Exists to isolate whether the v45 t=2.11 signal came
from the ensemble or from size alone.

Live prototype 2026-07-02:
  size only (log_mcap in HistGBM):    t=+2.574  ← this template
  11 rank-normalized features (v45):  t=+2.113  MARGINAL

The size-only version tips into the GREEN band. Not because ML found
anything — because the size premium is well-known (Banz 1981,
Fama-French 1993, McLean-Pontiff 2016 confirm SMB has NOT decayed as
much as other classic anomalies). GBM adds non-linear interpolation
that concentrates weight in the extreme-small-cap tail.

Why ship BOTH v45 and v46 side-by-side
=======================================
Audit-trail completeness. v45 answers "does ensemble ML beat picking
best single factor?" (No, it doesn't). v46 answers "if I dispatch a
size-based ML claim through the same infra, does the discipline
correctly route it to GREEN + surface to /approvals?" (Yes, and
Bailey-LdP family n_trials for SIZE tightens the DSR-adjusted
threshold immediately.)

Both belong in the event store as complementary evidence, not as
competing claims.

Scope
=====
  signal_kind : ml_size_only
  universe    : us_equities_top_3000
  data        : _ml_feature_panel.parquet (only log_mcap column used)
  model       : HistGradientBoostingRegressor
                (same hyperparams as v45 for direct comparison)
  CV          : rolling 36-month train → predict next month, no leakage
  portfolio   : tercile L/S by predicted return, monthly rebalance

Verdict thresholds
==================
  |t| >= 2.5   GREEN
  1.65 - 2.5   MARGINAL
  < 1.65       RED

Anticipated verdict: GREEN with t ≈ 2.57 (Sharpe 0.95). If this
surfaces to /approvals, the reviewer should understand this is a
SIZE-factor exposure with GBM interpolation, NOT a novel ML finding.
S7 Gate 6 (anchor-residual) will likely SOFT_PASS or FAIL because
the alpha vanishes once SMB is on the RHS — that's the anchor pointing
out "this is just size".

Known limitations (recorded in metrics.unchecked_dimensions)
============================================================
  - Not really "ML" — it's SIZE with GBM interpolation
  - Small-cap transaction costs (5-15bp/side) likely halve net Sharpe
  - Post-1980 sample: pre-1980 size premium was materially stronger
  - No FF5+MOM spanning (SMB would clearly explain most of alpha)
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

# Single feature: SIZE. Rank-normalized cross-sectionally per month.
_FEATURES = ("log_mcap",)

_T_GREEN    = 2.5
_T_MARGINAL = 1.65

_MIN_MONTHS         = 60
_MIN_XSEC_UNIT      = 50
_TRAIN_WINDOW_MO    = 36

_HGB_MAX_ITER       = 200
_HGB_MAX_DEPTH      = 5
_HGB_LEARNING_RATE  = 0.03
_HGB_L2             = 0.5
_HGB_RANDOM_STATE   = 42


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


def _load_and_clean(start: _dt.date, end: _dt.date) -> pd.DataFrame:
    df = pd.read_parquet(_FEATURE_PANEL_PATH)
    df = df.dropna(subset=["y", "log_mcap"])   # size-only: drop rows missing log_mcap
    df = df[
        (df["month"] >= pd.Timestamp(start)) &
        (df["month"] <= pd.Timestamp(end))
    ].copy()
    # Rank-normalize cross-section per month
    df["log_mcap_r"] = df.groupby("month")["log_mcap"].rank(pct=True)
    df["log_mcap_r"] = df["log_mcap_r"].fillna(0.5)
    return df


def _walk_forward_predict(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingRegressor

    ranked = ["log_mcap_r"]
    months = sorted(df["month"].unique())
    preds: list[pd.DataFrame] = []

    for i in range(_TRAIN_WINDOW_MO, len(months)):
        train_months = months[i - _TRAIN_WINDOW_MO:i]
        test_month = months[i]
        train = df[df["month"].isin(train_months)]
        test  = df[df["month"] == test_month]
        if len(test) < _MIN_XSEC_UNIT:
            continue
        model = HistGradientBoostingRegressor(
            max_iter=_HGB_MAX_ITER,
            max_depth=_HGB_MAX_DEPTH,
            learning_rate=_HGB_LEARNING_RATE,
            l2_regularization=_HGB_L2,
            random_state=_HGB_RANDOM_STATE,
        )
        model.fit(train[ranked].values, train["y"].values)
        t2 = test.copy()
        t2["pred"] = model.predict(test[ranked].values)
        preds.append(t2[["month", "permno", "pred", "y"]])

    if not preds:
        return pd.DataFrame(columns=["month", "permno", "pred", "y"])
    return pd.concat(preds, ignore_index=True)


def _tercile_ls_by_pred(oos: pd.DataFrame) -> pd.Series:
    def _one_month(g: pd.DataFrame) -> float:
        if len(g) < 30:
            return np.nan
        s = g.sort_values("pred")
        n = len(s)
        k = max(1, n // 3)
        short = s["y"].head(k).mean()
        long  = s["y"].tail(k).mean()
        return float(long - short)
    return oos.groupby("month").apply(_one_month, include_groups=False).dropna().rename("ls_ret")


def template_ml_size_only(spec: FactorSpec):
    from engine.agents.strengthener.factor_dispatcher import TemplateResult

    date_range = getattr(spec, "date_range", None) or "2014-01:2024-06"
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
            metrics={"expected_panel": str(_FEATURE_PANEL_PATH)},
            artifacts={}, template_version=_TEMPLATE_VERSION,
        )

    try:
        df = _load_and_clean(start, end)
        oos = _walk_forward_predict(df)
    except Exception as e:
        logger.exception("ml_size_only pipeline failed")
        return TemplateResult(
            verdict="EXECUTION_ERROR",
            summary=f"pipeline error: {type(e).__name__}: {str(e)[:200]}",
            metrics={},
            artifacts={}, template_version=_TEMPLATE_VERSION,
        )

    if len(oos) == 0:
        return TemplateResult(
            verdict="EXECUTION_ERROR",
            summary=f"empty OOS predictions (date_range {date_range})",
            metrics={"n_oos_rows": 0, "date_range": date_range},
            artifacts={}, template_version=_TEMPLATE_VERSION,
        )

    port = _tercile_ls_by_pred(oos)
    if len(port) < _MIN_MONTHS:
        return TemplateResult(
            verdict="INSUFFICIENT_HISTORY",
            summary=f"{len(port)} OOS months < required {_MIN_MONTHS}",
            metrics={"n_months": len(port), "min_required": _MIN_MONTHS},
            artifacts={}, template_version=_TEMPLATE_VERSION,
        )

    mu    = float(port.mean())
    sigma = float(port.std(ddof=1))
    if sigma <= 0 or not math.isfinite(sigma):
        return TemplateResult(
            verdict="EXECUTION_ERROR",
            summary="zero-variance portfolio returns",
            metrics={"n_months": len(port)},
            artifacts={}, template_version=_TEMPLATE_VERSION,
        )

    n = len(port)
    t_stat = mu * math.sqrt(n) / sigma
    sharpe_ann = mu / sigma * math.sqrt(12)
    cum_ret = float((1.0 + port).prod() - 1.0)
    verdict = _verdict_from_t(t_stat)

    metrics = {
        "n_months":            n,
        "mean_monthly":        mu,
        "vol_monthly":         sigma,
        "sharpe_annualized":   sharpe_ann,
        "t_stat":              t_stat,
        "cum_return":          cum_ret,
        "n_features":          1,
        "features":            ["log_mcap"],
        "model":               "HistGradientBoostingRegressor",
        "hyperparams":         {
            "max_iter":      _HGB_MAX_ITER,
            "max_depth":     _HGB_MAX_DEPTH,
            "learning_rate": _HGB_LEARNING_RATE,
            "l2":            _HGB_L2,
            "random_state":  _HGB_RANDOM_STATE,
        },
        "cv":                  {"scheme": "rolling_walk_forward",
                                 "train_window_months": _TRAIN_WINDOW_MO},
        "rank_normalized":     True,
        "window_start":        start.isoformat(),
        "window_end":          end.isoformat(),
        "verdict_thresholds":  {"green": _T_GREEN, "marginal": _T_MARGINAL},
        "companion_template":  "ml_ensemble_combiner (v45) — 11-feature version",
        "signal_family_note":  ("This is SIZE (Banz 1981 / SMB) with GBM "
                                 "non-linear interpolation, NOT novel ML alpha. "
                                 "Anchor-residual test (S7 Gate 6) will likely "
                                 "attribute most alpha to SMB."),
        "unchecked_dimensions": [
            "ff5_spanning", "cost_stress_small_cap", "smb_orthogonalization",
        ],
    }
    summary = (
        f"ml_size_only tercile L/S: Sharpe(ann)={sharpe_ann:.3f}, "
        f"t={t_stat:.2f} across {n} OOS months ({start}..{end}), "
        f"HistGBM on 1 feature (log_mcap rank)"
    )
    return TemplateResult(
        verdict=verdict, summary=summary, metrics=metrics,
        artifacts={}, template_version=_TEMPLATE_VERSION,
    )
