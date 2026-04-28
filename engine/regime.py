"""
Regime Detection Layer
======================
Data-driven macro regime classification using Hamilton (1989) Markov
Switching Model (MSM), with a rule-based fallback for small samples or
convergence failures.

Design principles
-----------------
1. Filtered probabilities only — never smoothed.
   Smoothed probabilities (Kalman backward pass) use future data and
   introduce look-ahead bias equivalent to LLM knowledge contamination.
   Only filtered_marginal_probabilities are used.

2. Walk-forward re-estimation.
   Every call to get_regime_on(date, train_end) re-fits the model on
   data up to train_end only. Reusing a model fitted on a later window
   would introduce parameter-estimation look-ahead bias.

3. Honest failure handling.
   MSM can fail to converge (especially on short samples). In that case
   the module falls back to a deterministic rule-based classifier and
   flags the result as approximate.

4. Sample size awareness.
   Monthly data: 60 obs ≈ 5 years (borderline), 120 obs ≈ 10 years (preferred).
   The module reports n_obs and warns when below 60.

Primary variable
----------------
yield_spread (10Y - 2Y Treasury) — monthly mean, sourced from FRED.
Already computed lag-adjusted in history.get_fred_snapshot().
Choice rationale: yield_spread is the single most parsimonious predictor
of US recession regimes (Estrella & Hardouvelis 1991; Ang et al. 2006).
Negative spread = inverted curve → historically precedes risk-off regimes.

Regime label convention
-----------------------
  "risk-on"    : positive yield_spread, low VIX — normal expansion
  "risk-off"   : inverted/flat curve or high VIX — contraction / flight to safety
  "transition" : ambiguous — filtered probability in [0.35, 0.65]

Integration points
------------------
  - Consumes: engine/history.py (_fetch_fred_series, get_vix_on)
  - Consumed by: engine/backtest.py (regime-conditional signal scaling)
  - Consumed by: engine/portfolio.py (regime-conditional position limits)
  - Compared against: memory.py DecisionLog.macro_regime (human label)
"""

from __future__ import annotations

import datetime
import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Minimum monthly observations for MSM to be considered reliable
_MIN_OBS_FOR_MSM = 60

# Probability threshold for regime assignment
_PROB_THRESHOLD = 0.65  # below this → "transition"

# Default training window (months of FRED history to fetch)
_DEFAULT_TRAIN_MONTHS = 120  # 10 years


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class RegimeResult:
    """Output of regime detection for a single date."""
    date:          datetime.date
    regime:        str            # "risk-on" | "risk-off" | "transition"
    p_risk_on:     float          # filtered P(risk-on | data up to date)
    p_risk_off:    float          # 1 - p_risk_on
    method:        str            # "msm" | "rule-based" | "msm-fallback"
    n_obs:         int            # monthly observations used for model fitting
    yield_spread:  float | None   # input value (for audit)
    vix:           float | None   # input value (for audit)
    warning:       str            # non-empty if degraded reliability


# ── FRED monthly data fetcher ──────────────────────────────────────────────────

def _fetch_fred(series_id: str, start: str, end: str) -> pd.Series:
    """
    Fetch a FRED series as a daily pd.Series with DatetimeIndex.
    Handles the current FRED CSV column name (observation_date).
    Publication lag and cutoff enforcement is the caller's responsibility.
    """
    import requests
    from io import StringIO

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        # FRED CSV uses 'observation_date' (not 'DATE')
        date_col = df.columns[0]
        val_col  = df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df = df[(df.index >= start) & (df.index <= end)]
        series = pd.to_numeric(df[val_col], errors="coerce").dropna()
        return series
    except Exception as exc:
        logger.warning("FRED fetch failed for %s: %s", series_id, exc)
        return pd.Series(dtype=float)


def _get_monthly_yield_spread(
    train_end: datetime.date,
    n_months:  int = _DEFAULT_TRAIN_MONTHS,
) -> pd.Series:
    """
    Fetch monthly yield_spread (10Y - 2Y) from FRED up to train_end.
    Resamples daily FRED data to month-end means.
    Enforces 1-day publication lag (yield data released with 1-day lag).

    Returns pd.Series with DatetimeIndex (month-end), values = yield_spread.
    """
    # 1-day publication lag for Treasury yields
    cutoff    = train_end - datetime.timedelta(days=1)
    start_str = str(cutoff - datetime.timedelta(days=n_months * 31 + 30))
    end_str   = str(cutoff)

    t10y = _fetch_fred("DGS10", start_str, end_str)
    t2y  = _fetch_fred("DGS2",  start_str, end_str)

    if t10y.empty or t2y.empty:
        return pd.Series(dtype=float)

    df           = pd.DataFrame({"t10y": t10y, "t2y": t2y}).dropna()
    df["spread"] = df["t10y"] - df["t2y"]

    monthly = df["spread"].resample("ME").mean().dropna()
    return monthly


def _get_monthly_vix(
    train_end: datetime.date,
    n_months:  int = _DEFAULT_TRAIN_MONTHS,
) -> pd.Series:
    """Fetch monthly mean VIX up to train_end."""
    import yfinance as yf

    start = train_end - datetime.timedelta(days=n_months * 31 + 30)
    try:
        raw = yf.download(
            "^VIX",
            start=str(start),
            end=str(train_end + datetime.timedelta(days=1)),
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            return pd.Series(dtype=float)
        close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        return close.resample("ME").mean().dropna()
    except Exception as exc:
        logger.warning("VIX fetch failed: %s", exc)
        return pd.Series(dtype=float)


# ── P2-14 Credit spread fetcher ───────────────────────────────────────────────

def _credit_spread_proxy(start: str, end: str) -> pd.Series:
    """
    Fallback: (LQD - IEF) monthly return spread as credit proxy.
    Units differ from OAS but direction is consistent: widens in stress.
    Only used when FRED BAMLC0A4CBBB is unavailable.
    """
    import yfinance as yf
    try:
        dl = yf.download(["LQD", "IEF"], start=start, end=end,
                         progress=False, auto_adjust=True)
        prices = dl["Close"] if isinstance(dl.columns, pd.MultiIndex) else dl
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = [c[0] for c in prices.columns]
        lqd_ret = prices["LQD"].pct_change() if "LQD" in prices.columns else pd.Series(dtype=float)
        ief_ret = prices["IEF"].pct_change() if "IEF" in prices.columns else pd.Series(dtype=float)
        proxy = (lqd_ret - ief_ret).resample("ME").mean().dropna() * 100
        return proxy
    except Exception:
        return pd.Series(dtype=float)


def _get_monthly_credit_spread(
    train_end: datetime.date,
    n_months:  int = _DEFAULT_TRAIN_MONTHS,
) -> pd.Series:
    """
    Fetch IG credit spread (ICE BofA BBB OAS, FRED: BAMLC0A4CBBB) monthly series.
    Falls back to LQD-IEF proxy on FRED failure.
    """
    cutoff    = train_end - datetime.timedelta(days=1)
    start_str = str(cutoff - datetime.timedelta(days=n_months * 31 + 60))
    end_str   = str(cutoff)
    try:
        series = _fetch_fred("BAMLC0A4CBBB", start_str, end_str)
        if not series.empty:
            monthly = series.resample("ME").last().dropna()
            if len(monthly) >= 12:
                return monthly
    except Exception:
        pass
    logger.info("P2-14: FRED credit spread unavailable, using LQD-IEF proxy")
    return _credit_spread_proxy(start_str, end_str)


# ── Rule-based fallback ────────────────────────────────────────────────────────

def _rule_based_regime(
    yield_spread: float | None,
    vix:          float | None,
) -> tuple[str, float, float]:
    """
    Deterministic regime classification as fallback.
    Returns (regime_label, p_risk_on, p_risk_off).

    Thresholds:
      yield_spread < -0.3 : strongly inverted curve → risk-off
      yield_spread < 0    : inverted curve → risk-off lean
      VIX > 30            : high fear → risk-off
      VIX > 22            : elevated → transition
      otherwise           : risk-on

    These thresholds are approximate and regime-specific to US macro cycles.
    """
    score = 0.0  # positive = more risk-on

    if yield_spread is not None:
        if yield_spread < -0.3:
            score -= 2.0
        elif yield_spread < 0.0:
            score -= 1.0
        elif yield_spread > 1.0:
            score += 1.0

    if vix is not None:
        if vix > 30:
            score -= 2.0
        elif vix > 22:
            score -= 0.8
        elif vix < 15:
            score += 0.5

    # Map score to probability
    p_risk_on = float(1 / (1 + np.exp(-score)))  # sigmoid

    if p_risk_on >= _PROB_THRESHOLD:
        regime = "risk-on"
    elif p_risk_on <= 1 - _PROB_THRESHOLD:
        regime = "risk-off"
    else:
        regime = "transition"

    return regime, p_risk_on, 1 - p_risk_on


# ── MSM core ──────────────────────────────────────────────────────────────────

def _select_k_by_bic(
    spread_series: pd.Series,
    k_range: tuple[int, int] = (2, 3),
) -> int:
    """
    Select number of regimes by BIC (Bayesian Information Criterion).
    Lower BIC = better model accounting for parameter count.

    In practice with 60-120 monthly observations, BIC almost always
    favours k=2 over k=3 due to the penalty term. This function is
    included for methodological rigour rather than expected result change.

    Returns the k with lowest BIC within k_range.
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    best_k, best_bic = k_range[0], float("inf")
    for k in range(k_range[0], k_range[1] + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = MarkovRegression(
                    spread_series, k_regimes=k,
                    trend="c", switching_variance=True,
                ).fit(disp=False, maxiter=200)
            if res.bic < best_bic:
                best_bic, best_k = res.bic, k
                logger.debug("BIC k=%d: %.2f", k, res.bic)
        except Exception as exc:
            logger.debug("BIC selection: k=%d failed — %s", k, exc)
    logger.info("BIC selected k=%d (range %s)", best_k, k_range)
    return best_k


def _fit_and_filter(
    spread_series: pd.Series,
    credit_series: pd.Series | None = None,   # P2-14: optional second variable
) -> tuple[pd.Series, int] | None:
    """
    Fit Markov Switching Model with BIC-selected k on yield_spread series.
    Returns (filtered_prob_risk_on, risk_on_regime_idx) or None on failure.

    k selection: BIC over k ∈ {2, 3} (Bayesian Information Criterion).
    Model: switching mean + switching variance (Hamilton 1989 style).
    Regime identity: regime with HIGHEST mean yield_spread = risk-on.
    This convention is stable: risk-on periods have steeper yield curves.

    Uses filtered_marginal_probabilities — not smoothed — to prevent
    any use of future information.
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    if len(spread_series) < _MIN_OBS_FOR_MSM:
        return None

    # P2-14: Build endog — bivariate when credit_series available and aligned
    bivariate = False
    if credit_series is not None and not credit_series.empty:
        _combined = pd.DataFrame({
            "yield_spread":  spread_series,
            "credit_spread": credit_series,
        }).dropna()
        if len(_combined) >= _MIN_OBS_FOR_MSM:
            endog     = _combined
            bivariate = True
        else:
            endog = spread_series.dropna()
    else:
        endog = spread_series.dropna()

    try:
        # Step 1: BIC k-selection always on yield_spread (stable univariate criterion)
        k = _select_k_by_bic(
            spread_series if not bivariate else endog["yield_spread"],
            k_range=(2, 3),
        )

        # Step 2: Fit final model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model  = MarkovRegression(
                endog,
                k_regimes=k,
                trend="c",
                switching_variance=True,
            )
            result = model.fit(disp=False, maxiter=300)

        # Step 3: Identify risk-on regime via yield_spread component mean
        # Bivariate params naming varies by statsmodels version; try both conventions.
        try:
            means = result.params[[f"yield_spread.const[{i}]" for i in range(k)]]
        except KeyError:
            means = result.params[[f"const[{i}]" for i in range(k)]]
        risk_on_idx = int(np.argmax(means))

        # Step 4: Filtered probabilities
        filtered         = result.filtered_marginal_probabilities
        p_risk_on_series = filtered.iloc[:, risk_on_idx]

        return p_risk_on_series, risk_on_idx

    except Exception as exc:
        logger.warning("MSM fitting failed%s: %s",
                       " (bivariate)" if bivariate else "", exc)
        if bivariate:
            logger.info("P2-14: falling back to univariate MSM")
            return _fit_and_filter(spread_series, credit_series=None)
        return None


# ── Public interface ───────────────────────────────────────────────────────────

def get_regime_on(
    as_of:        datetime.date,
    train_end:    datetime.date | None = None,
    n_train_months: int = _DEFAULT_TRAIN_MONTHS,
) -> RegimeResult:
    """
    Estimate macro regime as of `as_of` using data up to `train_end`.

    For live use: train_end = as_of (use all available data).
    For walk-forward backtest: train_end = last date of training window,
      which must be strictly <= as_of to prevent look-ahead.

    Steps:
      1. Fetch monthly yield_spread and VIX up to train_end
      2. Fit 2-regime MSM on yield_spread (training window only)
      3. Return filtered P(risk-on) at as_of
      4. Fall back to rule-based if MSM fails or sample too small

    Args:
        as_of:          Date for which regime is requested
        train_end:      Last date of data used for model fitting (default: as_of)
        n_train_months: Months of history to fetch for model fitting

    Returns:
        RegimeResult dataclass
    """
    if train_end is None:
        train_end = as_of

    # Cache lookup: only for live use (train_end == as_of) and historical dates
    import datetime as _dt_mod
    _live_use = (train_end == as_of) and (as_of < _dt_mod.date.today())
    if _live_use:
        try:
            from engine.memory import get_regime_snapshot
            cached = get_regime_snapshot(as_of)
            if cached is not None:
                return RegimeResult(**cached)
        except Exception:
            pass

    if train_end > as_of:
        raise ValueError(
            f"train_end ({train_end}) must be <= as_of ({as_of}) "
            "to prevent look-ahead bias."
        )

    # ── Fetch inputs ──────────────────────────────────────────────────────────
    spread_monthly = _get_monthly_yield_spread(train_end, n_train_months)
    vix_monthly    = _get_monthly_vix(train_end, n_train_months)
    credit_monthly = _get_monthly_credit_spread(train_end, n_train_months)  # P2-14

    n_obs          = len(spread_monthly)
    warn_msg       = ""

    # Current point values (for rule-based fallback and audit)
    current_spread = float(spread_monthly.iloc[-1]) if not spread_monthly.empty else None
    if not vix_monthly.empty and not spread_monthly.empty:
        _vix_aligned = vix_monthly.reindex(spread_monthly.index, method="nearest")
        _vix_val = _vix_aligned.iloc[-1]
        current_vix = float(_vix_val.iloc[0]) if hasattr(_vix_val, "iloc") else float(_vix_val)
    else:
        current_vix = None

    # ── MSM path ─────────────────────────────────────────────────────────────
    msm_result = None
    if n_obs >= _MIN_OBS_FOR_MSM:
        msm_result = _fit_and_filter(spread_monthly, credit_series=credit_monthly)
    else:
        warn_msg = f"样本量不足（{n_obs} < {_MIN_OBS_FOR_MSM} 月），使用规则基础分类"
        logger.warning("Regime MSM skipped: n_obs=%d < %d", n_obs, _MIN_OBS_FOR_MSM)

    if msm_result is not None:
        p_risk_on_series, _ = msm_result

        # Find filtered probability at as_of (or nearest prior date)
        as_of_ts  = pd.Timestamp(as_of)
        prior_idx = p_risk_on_series.index[p_risk_on_series.index <= as_of_ts]

        if prior_idx.empty:
            # as_of is before model data — fall back
            msm_result = None
            warn_msg   = f"as_of={as_of} 早于模型数据范围，使用规则基础分类"
        else:
            p_risk_on = float(p_risk_on_series[prior_idx[-1]])
            p_risk_off = 1.0 - p_risk_on

            if p_risk_on >= _PROB_THRESHOLD:
                regime = "risk-on"
            elif p_risk_on <= 1 - _PROB_THRESHOLD:
                regime = "risk-off"
            else:
                regime = "transition"

            if n_obs < 80:
                warn_msg = f"样本量偏低（{n_obs} 月），MSM 参数估计不稳定，结论参考性有限"

            result = RegimeResult(
                date=as_of,
                regime=regime,
                p_risk_on=round(p_risk_on, 4),
                p_risk_off=round(p_risk_off, 4),
                method="msm",
                n_obs=n_obs,
                yield_spread=current_spread,
                vix=current_vix,
                warning=warn_msg,
            )
            if _live_use:
                try:
                    from engine.memory import save_regime_snapshot
                    save_regime_snapshot(result)
                except Exception:
                    pass
            return result

    # ── Fallback: rule-based ──────────────────────────────────────────────────
    regime, p_risk_on, p_risk_off = _rule_based_regime(current_spread, current_vix)
    method = "msm-fallback" if n_obs >= _MIN_OBS_FOR_MSM else "rule-based"

    result = RegimeResult(
        date=as_of,
        regime=regime,
        p_risk_on=round(p_risk_on, 4),
        p_risk_off=round(p_risk_off, 4),
        method=method,
        n_obs=n_obs,
        yield_spread=current_spread,
        vix=current_vix,
        warning=warn_msg or "MSM 拟合失败，使用规则基础分类",
    )
    if _live_use:
        try:
            from engine.memory import save_regime_snapshot
            save_regime_snapshot(result)
        except Exception:
            pass
    return result


def get_regime_series(
    dates:          list[datetime.date],
    n_train_months: int = _DEFAULT_TRAIN_MONTHS,
) -> pd.DataFrame:
    """
    Compute regime for a list of dates (walk-forward compatible).

    For each date, the model is trained on data up to that date only.
    This is the correct walk-forward protocol — each date sees only its
    own past, preventing parameter-estimation look-ahead bias.

    Computationally expensive: O(n_dates) MSM fits.
    For large date lists (>24 months), consider using expanding-window
    batching in backtest.py instead.

    Returns DataFrame with columns matching RegimeResult fields.
    """
    records = []
    for date in dates:
        result = get_regime_on(as_of=date, train_end=date, n_train_months=n_train_months)
        records.append({
            "date":         result.date,
            "regime":       result.regime,
            "p_risk_on":    result.p_risk_on,
            "p_risk_off":   result.p_risk_off,
            "method":       result.method,
            "n_obs":        result.n_obs,
            "yield_spread": result.yield_spread,
            "vix":          result.vix,
            "warning":      result.warning,
        })
        if result.warning:
            logger.info("Regime[%s]: %s — %s", date, result.regime, result.warning)

    return pd.DataFrame(records).set_index("date")


def compare_with_human(
    model_regime:  str,
    human_regime:  str,
) -> dict:
    """
    Compare MSM regime label against human-entered macro_regime from DecisionLog.
    Used for the research question: does human judgment contain incremental
    information beyond statistical models?

    Args:
        model_regime : one of "risk-on" | "risk-off" | "transition"
        human_regime : free-text from DecisionLog.macro_regime field

    Returns dict with:
        agreement  : bool
        divergence : str description if disagreement
    """
    _risk_off_keywords = ("risk-off", "risk off", "收紧", "衰退", "高波动", "避险", "熊")
    _risk_on_keywords  = ("risk-on",  "risk on",  "宽松", "扩张", "复苏", "低波动", "牛")

    human_lower = human_regime.lower()

    if any(k in human_lower for k in _risk_off_keywords):
        human_mapped = "risk-off"
    elif any(k in human_lower for k in _risk_on_keywords):
        human_mapped = "risk-on"
    else:
        human_mapped = "transition"

    agreement = (model_regime == human_mapped) or \
                (model_regime == "transition") or \
                (human_mapped == "transition")

    return {
        "model_regime":  model_regime,
        "human_regime":  human_regime,
        "human_mapped":  human_mapped,
        "agreement":     agreement,
        "divergence":    "" if agreement else
                         f"模型判断 {model_regime} vs 人工判断 {human_mapped} — "
                         "分歧可作为 edit_ratio 研究的辅助变量",
    }
