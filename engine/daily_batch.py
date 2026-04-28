"""
Daily Batch Job
===============
Orchestrates the daily trading agent pipeline. Called once per trading day
when the user opens the app (lazy-load pattern with idempotency guard).

Pipeline order (matches trading_logic_design.md)
-------------------------------------------------
  1. Freshness check   — skip if T-day snapshot already exists
  2. Signal & regime   — run QuantAgent, write SignalSnapshot / RegimeSnapshot
  3. Watchlist patrol  — evaluate invalidation conditions for 'watching' entries
  4. Entry check       — check entry_condition for still-watching entries
  5. Position patrol   — hard stop / signal reversal / regime compression
  6. Month-end check   — generate rebalance orders if last trading day of month

Idempotency
-----------
All write operations check for existing records before inserting.
Running twice on the same T-day is safe and produces no duplicate state changes.

Human approval loop
-------------------
Transitions to 'active' and execution simulation happen ONLY after a
PendingApproval record is resolved (status='approved') by the user in the UI.
DailyBatchJob generates PendingApproval records; it does NOT auto-approve.
"""
from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from engine.memory import (
    SessionFactory,
    SignalSnapshot,
    RegimeSnapshot,
    WatchlistEntry,
    PendingApproval,
    SimulatedPosition,
)
from engine.quant_agent import run_quant_assessment
from engine.signal import get_signal_dataframe
from engine.regime import get_regime_on
from engine.trading_schema import EntryCondition, InvalidationCondition, WEIGHT_LIMITS

logger = logging.getLogger(__name__)

# ── Trading calendar helper ────────────────────────────────────────────────────

def _get_last_nyse_trading_day(ref: Optional[datetime.date] = None) -> datetime.date:
    """
    Return the most recent NYSE trading day on or before ref (default: today).
    Uses pandas_market_calendars when available; falls back to simple weekday rule.
    """
    if ref is None:
        ref = datetime.date.today()
    try:
        import pandas_market_calendars as mcal
        nyse  = mcal.get_calendar("NYSE")
        sched = nyse.schedule(
            start_date=(ref - datetime.timedelta(days=10)).isoformat(),
            end_date=ref.isoformat(),
        )
        if sched.empty:
            return ref
        return sched.index[-1].date()
    except Exception:
        # Fallback: step back from ref until we land on Mon-Fri
        d = ref
        while d.weekday() >= 5:
            d -= datetime.timedelta(days=1)
        return d


def _add_trading_days(start: datetime.date, n: int) -> datetime.date:
    """Approximate n trading days forward (weekday-only, ignores holidays)."""
    d, count = start, 0
    while count < n:
        d += datetime.timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return d


def _is_first_trading_day_of_month(d: datetime.date) -> bool:
    """True if d is the first NYSE trading day of its calendar month."""
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        month_start = d.replace(day=1)
        sched = nyse.schedule(
            start_date=month_start.isoformat(),
            end_date=(month_start + datetime.timedelta(days=7)).isoformat(),
        )
        if sched.empty:
            return False
        return sched.index[0].date() == d
    except Exception:
        first = d.replace(day=1)
        while first.weekday() >= 5:
            first += datetime.timedelta(days=1)
        return d == first


def _is_month_end_trading_day(d: datetime.date) -> bool:
    """True if d is the last NYSE trading day of its calendar month."""
    try:
        import pandas_market_calendars as mcal
        nyse  = mcal.get_calendar("NYSE")
        month_end = (d.replace(day=28) + datetime.timedelta(days=4)).replace(day=1) - datetime.timedelta(days=1)
        sched = nyse.schedule(
            start_date=d.isoformat(),
            end_date=(month_end + datetime.timedelta(days=5)).isoformat(),
        )
        if sched.empty:
            return False
        return sched.index[0].date() > d
    except Exception:
        next_weekday = d + datetime.timedelta(days=1)
        while next_weekday.weekday() >= 5:
            next_weekday += datetime.timedelta(days=1)
        return next_weekday.month != d.month


# ── Price fetch (single ticker, T-day close) ─────────────────────────────────

def _fetch_close(ticker: str, as_of: datetime.date) -> Optional[float]:
    try:
        start = as_of - datetime.timedelta(days=5)
        df = yf.download(ticker, start=start,
                         end=(as_of + datetime.timedelta(days=1)).isoformat(),
                         auto_adjust=True, progress=False, multi_level_index=False)
        if df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


def _fetch_sma(ticker: str, as_of: datetime.date, period: int) -> Optional[float]:
    try:
        start = as_of - datetime.timedelta(days=period * 2)
        df = yf.download(ticker, start=start,
                         end=(as_of + datetime.timedelta(days=1)).isoformat(),
                         auto_adjust=True, progress=False, multi_level_index=False)
        if df.empty or len(df) < period:
            return None
        return float(df["Close"].iloc[-period:].mean())
    except Exception:
        return None


# ── Batch result dataclass ────────────────────────────────────────────────────

@dataclass
class BatchResult:
    as_of_date:          datetime.date
    skipped:             bool = False        # T-day snapshot already existed
    signal_ok:           bool = False
    regime_ok:           bool = False
    invalidations:       list[str] = field(default_factory=list)
    entries_triggered:   list[str] = field(default_factory=list)
    corr_blocked:        list[str] = field(default_factory=list)
    risk_alerts:         list[str] = field(default_factory=list)
    rebalance_orders:    list[str] = field(default_factory=list)
    errors:              list[str] = field(default_factory=list)
    # P4: Tactical patrol results
    tactical_entries:    list[str] = field(default_factory=list)
    tactical_reduces:    list[str] = field(default_factory=list)
    regime_jump:         bool = False
    # P5: Pipeline step status
    cb_level:            str  = "none"        # Circuit Breaker level at run time
    cb_reason:           str  = ""
    debate_sectors:      list[str] = field(default_factory=list)   # sectors where FinDebate ran
    debate_results:      dict = field(default_factory=dict)         # {sector: conclusion}
    rebalance_auto:      bool = False         # True = monthly rebalance was auto-executed
    rebalance_skipped_reason: str = ""        # why auto-rebalance was blocked
    factormad_icir_ran:  bool = False         # True = monthly ICIR update ran today
    factormad_deactivated: list[str] = field(default_factory=list) # factors deactivated this cycle

    @property
    def ok(self) -> bool:
        return not self.errors


# ── Step 1: freshness check ───────────────────────────────────────────────────

def _snapshot_exists(t_day: datetime.date) -> bool:
    with SessionFactory() as s:
        exists = (
            s.query(SignalSnapshot)
             .filter(SignalSnapshot.as_of_date == t_day)
             .first()
        )
        return exists is not None


# ── Step 2: signal & regime ────────────────────────────────────────────────────

def _run_signal_regime(t_day: datetime.date, result: BatchResult) -> None:
    try:
        assessments = run_quant_assessment(as_of=t_day)
        result.signal_ok = bool(assessments)
    except Exception as exc:
        result.errors.append(f"signal: {exc}")
        logger.exception("DailyBatch signal step failed")
        return

    try:
        get_regime_on(t_day)
        result.regime_ok = True
    except Exception as exc:
        result.errors.append(f"regime: {exc}")
        logger.exception("DailyBatch regime step failed")


# ── Step 3: watchlist invalidation patrol ────────────────────────────────────

def _evaluate_quant_invalidation(
    cond: InvalidationCondition,
    current_tsmom: int,
    current_price: Optional[float],
    ticker: str,
    t_day: datetime.date,
) -> bool:
    """Return True if the invalidation condition is triggered."""
    if cond.rule == "tsmom_flipped":
        if cond.entry_value is not None and current_tsmom != 0:
            return int(np.sign(current_tsmom)) != int(np.sign(cond.entry_value))
        return False
    if cond.rule == "price_below_sma":
        period = cond.sma_period or 200
        if current_price is None:
            return False
        sma = _fetch_sma(ticker, t_day, period)
        return sma is not None and current_price < sma
    return False


def _patrol_invalidations(t_day: datetime.date, result: BatchResult) -> None:
    signal_df = get_signal_dataframe(as_of=t_day)

    with SessionFactory() as session:
        watching = (
            session.query(WatchlistEntry)
                   .filter(WatchlistEntry.status == "watching")
                   .all()
        )

        for entry in watching:
            conditions: list[dict] = json.loads(entry.invalidation_json or "[]")
            if not conditions:
                continue

            close = _fetch_close(entry.ticker, t_day)
            current_tsmom = 0
            if not signal_df.empty and entry.sector in signal_df.index:
                current_tsmom = int(signal_df.loc[entry.sector, "tsmom"])

            fired_reason: Optional[str] = None

            for raw in conditions:
                cond = InvalidationCondition(**raw)
                if cond.type == "quant":
                    if _evaluate_quant_invalidation(cond, current_tsmom, close, entry.ticker, t_day):
                        fired_reason = f"{cond.rule} (tsmom={current_tsmom}, entry={cond.entry_value})"
                        break
                # descriptive conditions are surfaced via PendingInvalidationCheck (UI only)

            if fired_reason:
                entry.status            = "invalidated"
                entry.invalidated_date  = t_day
                entry.invalidated_reason = fired_reason
                result.invalidations.append(entry.sector)
                logger.info("WatchlistEntry %s invalidated: %s", entry.sector, fired_reason)

        session.commit()


# ── Step 4: entry condition check ─────────────────────────────────────────────

def _check_entry_condition(
    cond: EntryCondition,
    ticker: str,
    t_day: datetime.date,
    current_price: Optional[float],
) -> bool:
    if cond.type == "immediate":
        return True

    if cond.type == "price_breakout":
        n = cond.n_days or 20
        try:
            start = t_day - datetime.timedelta(days=n * 2)
            df = yf.download(ticker, start=start,
                             end=(t_day + datetime.timedelta(days=1)).isoformat(),
                             auto_adjust=True, progress=False, multi_level_index=False)
            if df.empty or len(df) < n + 1 or current_price is None:
                return False
            # Uses n-day closing-price high, not intraday high — more conservative,
            # triggers later and with larger positive slippage vs. a high-based breakout.
            prior_high = float(df["Close"].iloc[-(n + 1):-1].max())
            return current_price > prior_high
        except Exception:
            return False

    if cond.type == "ma_crossover":
        period = cond.ma_period or 50
        if current_price is None:
            return False
        sma = _fetch_sma(ticker, t_day, period)
        return sma is not None and current_price > sma

    if cond.type == "volume_confirm":
        mult = cond.volume_multiple or 1.5
        try:
            start = t_day - datetime.timedelta(days=60)
            df = yf.download(ticker, start=start,
                             end=(t_day + datetime.timedelta(days=1)).isoformat(),
                             auto_adjust=True, progress=False, multi_level_index=False)
            if df.empty or len(df) < 21 or current_price is None:
                return False
            avg_vol = float(df["Volume"].iloc[-21:-1].mean())
            today_vol = float(df["Volume"].iloc[-1])
            today_up  = float(df["Close"].iloc[-1]) > float(df["Close"].iloc[-2])
            return today_vol > mult * avg_vol and today_up
        except Exception:
            return False

    return False


_CORR_BLOCK_THRESHOLD = 0.75   # corr > this → status = "corr_blocked"
_CORR_WARN_THRESHOLD  = 0.60   # corr > this → proceed with warning note in PendingApproval


def _check_correlation(
    new_ticker: str,
    active_tickers: list[str],
    t_day: datetime.date,
    lookback_days: int = 90,
) -> tuple[float, str]:
    """
    Return (max_pairwise_correlation, most_correlated_active_ticker).
    Computes rolling 90-day return correlation between new_ticker and each
    currently active position.  Returns (0.0, "") on data failure.
    """
    if not active_tickers:
        return 0.0, ""

    all_tickers = list(dict.fromkeys([new_ticker] + active_tickers))  # dedupe, preserve order
    start = t_day - datetime.timedelta(days=lookback_days * 2)
    end   = t_day + datetime.timedelta(days=1)
    try:
        df = yf.download(all_tickers, start=start, end=end,
                         auto_adjust=True, progress=False, multi_level_index=False)
        if df.empty:
            return 0.0, ""

        # yfinance returns MultiIndex when multiple tickers
        if isinstance(df.columns, pd.MultiIndex):
            closes = df["Close"]
        else:
            closes = df[["Close"]].rename(columns={"Close": new_ticker})

        if new_ticker not in closes.columns or len(closes) < 30:
            return 0.0, ""

        rets = closes.pct_change().dropna()
        if len(rets) < 30:
            return 0.0, ""

        max_corr, max_ticker = 0.0, ""
        for tk in active_tickers:
            if tk not in rets.columns or tk == new_ticker:
                continue
            corr = float(rets[new_ticker].corr(rets[tk]))
            if corr > max_corr:
                max_corr, max_ticker = corr, tk
        return max_corr, max_ticker
    except Exception as exc:
        logger.debug("_check_correlation(%s): %s", new_ticker, exc)
        return 0.0, ""


def _patrol_entries(t_day: datetime.date, result: BatchResult) -> None:
    with SessionFactory() as session:
        watching = (
            session.query(WatchlistEntry)
                   .filter(WatchlistEntry.status == "watching")
                   .all()
        )

        # Collect active position tickers for correlation check
        active_tickers = [
            p.ticker for p in
            session.query(SimulatedPosition)
                   .filter(SimulatedPosition.snapshot_date == t_day)
                   .all()
            if p.ticker
        ]

        for entry in watching:
            ec_raw = json.loads(entry.entry_condition_json or '{"type":"immediate"}')
            cond   = EntryCondition(**ec_raw)
            close  = _fetch_close(entry.ticker, t_day)

            if not _check_entry_condition(cond, entry.ticker, t_day, close):
                continue

            # Entry condition met — run correlation check before promoting.
            corr, corr_peer = _check_correlation(entry.ticker, active_tickers, t_day)

            if corr > _CORR_BLOCK_THRESHOLD:
                # Too correlated with existing position — block and record.
                entry.status         = "corr_blocked"
                entry.triggered_date = t_day
                result.corr_blocked.append(entry.sector)
                logger.info(
                    "Entry CORR_BLOCKED: %s (corr=%.2f with %s)",
                    entry.sector, corr, corr_peer,
                )
                session.commit()
                continue

            # Promote to triggered (awaiting human approval).
            entry.status          = "triggered"
            entry.triggered_date  = t_day
            entry.triggered_price = close

            deadline = _add_trading_days(t_day, 3)

            corr_warn = (
                f" | ⚠ Correlation warning: {corr:.0%} with {corr_peer}"
                if corr > _CORR_WARN_THRESHOLD else ""
            )

            existing = (
                session.query(PendingApproval)
                       .filter(
                           PendingApproval.watchlist_entry_id == entry.id,
                           PendingApproval.approval_type == "entry",
                           PendingApproval.status == "pending",
                       )
                       .first()
            )
            if not existing:
                # P3-12: Compute LLM/Quant disagreement flag
                _tsmom_sig   = entry.entry_tsmom_signal  # +1 / -1 / 0
                _llm_dir     = (entry.direction or "").lower()
                _contradicts = (
                    _tsmom_sig is not None and _llm_dir != "" and
                    ((_tsmom_sig > 0 and _llm_dir == "short") or
                     (_tsmom_sig < 0 and _llm_dir == "long"))
                )
                session.add(PendingApproval(
                    approval_type="entry",
                    priority="high" if _contradicts else "normal",
                    watchlist_entry_id=entry.id,
                    sector=entry.sector,
                    ticker=entry.ticker,
                    triggered_condition=(
                        f"{ec_raw.get('type')} triggered at {close:.2f}{corr_warn}"
                        + (" ⚠ LLM与TSMOM方向相反" if _contradicts else "")
                    ),
                    triggered_date=t_day,
                    triggered_price=close,
                    suggested_weight=entry.suggested_weight,
                    position_rank=entry.position_rank,
                    approval_deadline=deadline,
                    contradicts_quant=_contradicts,
                    llm_confidence=entry.confidence,
                ))
                result.entries_triggered.append(entry.sector)
                logger.info("Entry triggered: %s @ %.2f (corr=%.2f)", entry.sector, close or 0, corr)

        session.commit()


# ── Step 5: active position patrol ────────────────────────────────────────────

def _patrol_positions(t_day: datetime.date, result: BatchResult) -> None:
    from engine.config import get_trading_config
    _cfg = get_trading_config()
    _auto_stops = _cfg["auto_execute_stops"]
    _auto_max_w = _cfg["stop_max_weight_auto"]

    signal_df    = get_signal_dataframe(as_of=t_day)
    regime_result = get_regime_on(t_day)
    regime_label  = getattr(regime_result, "regime", "transition")

    with SessionFactory() as session:
        active_positions = (
            session.query(SimulatedPosition)
                   .filter(
                       SimulatedPosition.snapshot_date == (
                           session.query(SimulatedPosition.snapshot_date)
                                  .order_by(SimulatedPosition.snapshot_date.desc())
                                  .limit(1)
                                  .scalar_subquery()
                       ),
                       SimulatedPosition.track == "main",  # quant baseline excluded
                   )
                   .all()
        )

        for pos in active_positions:
            close = _fetch_close(pos.ticker, t_day)
            if close is None:
                continue
            # Skip already-zeroed positions
            if (pos.actual_weight or 0.0) == 0.0:
                continue

            # ── 5.1 Hard stop: trailing ATR stop ──────────────────────────────
            from engine.quant_agent import _fetch_price_context
            atr, _ = _fetch_price_context(pos.ticker, t_day, atr_period=21)

            prev_high = pos.trailing_high or pos.entry_price or close
            new_high  = max(prev_high, close)
            pos.trailing_high = new_high

            if atr > 0:
                stop_price = new_high - 2.0 * atr
                if close < stop_price:
                    _reason = (
                        f"Trailing stop: close {close:.2f} < stop {stop_price:.2f} "
                        f"(trailing_high={new_high:.2f}, ATR={atr:.2f})"
                    )
                    w = abs(pos.actual_weight or 0.0)
                    # Layer 2: auto-execute if enabled and position within size limit
                    if _auto_stops and w <= _auto_max_w:
                        _auto_execute_stop(session, pos, t_day, close, _reason)
                        result.risk_alerts.append(f"{pos.sector}:hard_stop:auto_executed")
                    else:
                        _add_risk_approval(session, pos, t_day, close,
                                           reason=_reason, priority="critical")
                        result.risk_alerts.append(f"{pos.sector}:hard_stop")
                    continue

            # ── 5.2 TSMOM signal reversal (Layer 3 — always needs human) ──────
            if not signal_df.empty and pos.sector in signal_df.index:
                current_tsmom = int(signal_df.loc[pos.sector, "tsmom"])
                entry_tsmom   = pos.signal_tsmom or 0
                if entry_tsmom != 0 and current_tsmom != 0 and current_tsmom != entry_tsmom:
                    _add_risk_approval(
                        session, pos, t_day, close,
                        reason=f"TSMOM flipped: {entry_tsmom:+d} → {current_tsmom:+d}",
                        priority="normal",
                    )
                    result.risk_alerts.append(f"{pos.sector}:tsmom_flip")

            # ── 5.3 Regime compression (Layer 3 — always needs human) ─────────
            if pos.actual_weight:
                rank = "satellite"
                for _regime, _cap in [
                    ("risk-off",   WEIGHT_LIMITS[rank]["risk-off"]),
                    ("transition", WEIGHT_LIMITS[rank]["transition"]),
                ]:
                    if regime_label == _regime and pos.actual_weight > _cap:
                        _add_risk_approval(
                            session, pos, t_day, close,
                            reason=(
                                f"Regime {_regime}: weight {pos.actual_weight:.1%} "
                                f"> cap {_cap:.1%}"
                            ),
                            priority="normal",
                            suggested_weight=_cap,
                        )
                        result.risk_alerts.append(f"{pos.sector}:regime_compress")
                        break

            # ── 5.4 Per-position RiskCondition evaluation (P1-3) ──────────────
            import json as _json
            _rc_raw = pos.risk_conditions_json if hasattr(pos, "risk_conditions_json") else None
            if _rc_raw:
                try:
                    _conditions = _json.loads(_rc_raw)
                except Exception:
                    _conditions = []
                for _rc in _conditions:
                    rc_type = _rc.get("type")
                    if rc_type == "vol_spike":
                        from engine.quant_agent import _fetch_price_context as _fpc
                        _, _ann_vol_rc = _fpc(pos.ticker, t_day, atr_period=21)
                        _threshold = _rc.get("threshold", 0.30)
                        if _ann_vol_rc and _ann_vol_rc > _threshold:
                            _new_cap = _rc.get("vol_spike_cap", 0.05)
                            _add_risk_approval(
                                session, pos, t_day, close,
                                reason=(
                                    f"Vol spike: ann_vol {_ann_vol_rc:.1%} > "
                                    f"threshold {_threshold:.1%}; compress to {_new_cap:.1%}"
                                ),
                                priority="normal",
                                suggested_weight=_new_cap,
                            )
                            result.risk_alerts.append(f"{pos.sector}:vol_spike")
                    elif rc_type == "drawdown":
                        if pos.entry_price and pos.entry_price > 0:
                            _ret = (close - pos.entry_price) / pos.entry_price
                            _threshold = _rc.get("threshold", -0.10)
                            if _ret < _threshold:
                                _reason_dd = (
                                    f"Drawdown: position return {_ret:.1%} < "
                                    f"threshold {_threshold:.1%}"
                                )
                                w = abs(pos.actual_weight or 0.0)
                                # Layer 2: drawdown stop also auto-executes
                                if _auto_stops and w <= _auto_max_w:
                                    _auto_execute_stop(session, pos, t_day, close,
                                                       _reason_dd)
                                    result.risk_alerts.append(
                                        f"{pos.sector}:drawdown_stop:auto_executed")
                                else:
                                    _add_risk_approval(
                                        session, pos, t_day, close,
                                        reason=_reason_dd,
                                        priority="critical",
                                        suggested_weight=0.0,
                                    )
                                    result.risk_alerts.append(
                                        f"{pos.sector}:drawdown_stop")

        session.commit()


def _auto_execute_stop(
    session, pos: SimulatedPosition, t_day: datetime.date,
    close: float, reason: str, target_weight: float = 0.0,
) -> None:
    """
    Layer 2: directly zero (or compress) a position without human approval.
    Writes a SimulatedTrade record for audit trail, then updates actual_weight.
    Called only when auto_execute_stops=True in trading config.
    """
    from engine.memory import SimulatedTrade
    from engine.portfolio_tracker import _get_nav
    w_before = pos.actual_weight or 0.0
    if abs(w_before - target_weight) < 1e-6:
        return  # already at target, skip

    # Compute share-level execution fields
    _nav      = _get_nav()
    _shares   = abs(pos.shares_held) if pos.shares_held else (
        abs(w_before) * _nav / close if close > 0 else None
    )
    _notional = round(_shares * close, 2) if _shares else None

    session.add(SimulatedTrade(
        trade_date    = t_day,
        sector        = pos.sector,
        ticker        = pos.ticker,
        action        = "SELL" if target_weight < w_before else "BUY",
        weight_before = round(w_before, 6),
        weight_after  = round(target_weight, 6),
        weight_delta  = round(target_weight - w_before, 6),
        cost_bps      = round(abs(target_weight - w_before) * 10, 2),
        trigger_reason = f"auto_stop: {reason[:80]}",
        shares        = round(_shares, 4) if _shares else None,
        fill_price    = round(close, 4),
        notional      = _notional,
    ))
    pos.actual_weight  = target_weight
    pos.shares_held    = 0.0 if target_weight == 0.0 else (
        (pos.shares_held or 0) * (target_weight / w_before) if w_before != 0 else None
    )
    pos.cost_basis     = 0.0 if target_weight == 0.0 else pos.cost_basis
    pos.position_value = 0.0 if target_weight == 0.0 else pos.position_value
    pos.notes = (pos.notes or "") + f" | auto_stop {t_day}: {reason[:80]}"


def _add_risk_approval(
    session, pos: SimulatedPosition, t_day: datetime.date,
    close: float, reason: str, priority: str,
    suggested_weight: float = 0.0,
) -> None:
    existing = (
        session.query(PendingApproval)
               .filter(
                   PendingApproval.sector == pos.sector,
                   PendingApproval.approval_type == "risk_control",
                   PendingApproval.status == "pending",
                   PendingApproval.triggered_date == t_day,
               )
               .first()
    )
    if not existing:
        session.add(PendingApproval(
            approval_type="risk_control",
            priority=priority,
            sector=pos.sector,
            ticker=pos.ticker,
            triggered_condition=reason,
            triggered_date=t_day,
            triggered_price=close,
            suggested_weight=suggested_weight,
        ))


# ── Step 5b: Daily tactical patrol (P4-3) ────────────────────────────────────

def _patrol_daily_tactical(t_day: datetime.date, result: BatchResult) -> None:
    """
    P4-3: Daily tactical patrol — 5 event types in priority order.
    Runs after _patrol_positions() so ATR stops are already handled.

    Event 1: Regime Jump       → Layer 2 auto-compress (if configured)
    Event 2: Fast Signal Flip  → Layer 3 approval (reduce 50%)
    Event 3: High-Conf Entry   → Layer 2 or 3 depending on conditions met
    Event 4: Drawdown Alert    → Layer 3 approval (reduce 50%)
    Event 5: Vol Spike Compress→ Layer 3 approval (compress to vol-parity)

    Regime jump skips events 2-5 when triggered (full-portfolio action overrides).
    """
    from engine.config import get_trading_config
    from engine.signal import get_fast_signal_dataframe, compute_composite_scores
    from engine.history import get_active_sector_etf

    _cfg = get_trading_config()
    _auto_regime   = _cfg["auto_execute_regime_compress"]
    _auto_entry    = _cfg["auto_execute_high_conf_entry"]
    _entry_max_w   = _cfg["tactical_entry_max_weight"]
    _entry_limit   = _cfg["tactical_entry_daily_limit"]
    _jump_thr      = _cfg["regime_jump_threshold_ppt"] / 100.0  # convert ppt → fraction
    _score_min     = _cfg["entry_composite_score_min"]
    _zscore_min    = _cfg["entry_momentum_zscore_min"]
    _lb_fast       = _cfg["fast_signal_lookback"]
    _sk_fast       = _cfg["fast_signal_skip"]

    try:
        regime_result = get_regime_on(t_day)
        regime_label  = getattr(regime_result, "regime", "transition")
        p_risk_on_now = getattr(regime_result, "p_risk_on", 0.5)
    except Exception as _re:
        logger.debug("_patrol_daily_tactical: regime fetch failed: %s", _re)
        return

    # ── Event 1: Regime Jump detection ───────────────────────────────────────
    _regime_jump = False
    try:
        from engine.memory import get_daily_brief_snapshot
        yesterday = t_day - datetime.timedelta(days=1)
        _prev_snap = get_daily_brief_snapshot(yesterday)
        p_risk_on_prev = float(_prev_snap.p_risk_on or 0.5) if _prev_snap else 0.5
        _p_change = abs(p_risk_on_now - p_risk_on_prev)

        # Also check VIX spike
        _vix_spike = False
        try:
            from engine.quant import QuantEngine
            _vix_now  = QuantEngine.get_realtime_vix() or 0.0
            _vix_prev = 0.0
            if _prev_snap and hasattr(_prev_snap, "risk_alerts_json"):
                import json as _j
                _alerts = _j.loads(_prev_snap.risk_alerts_json or "[]")
                _vix_refs = [a for a in _alerts if "vix=" in a.lower()]
                if _vix_refs:
                    try:
                        _vix_prev = float(_vix_refs[0].split("vix=")[-1].split(":")[0])
                    except Exception:
                        pass
            if _vix_prev > 0 and _vix_now > _vix_prev * 1.25:
                _vix_spike = True
        except Exception:
            pass

        if _p_change >= _jump_thr or _vix_spike:
            _regime_jump = True
            result.regime_jump = True
            logger.info(
                "_patrol_daily_tactical: regime jump detected "
                "(p_change=%.2f, vix_spike=%s) on %s", _p_change, _vix_spike, t_day
            )

            if _auto_regime:
                # Layer 2: auto-compress all long positions to tactical cap
                _regime_off_max = 0.08   # 8% cap per spec
                with SessionFactory() as session:
                    _active = (
                        session.query(SimulatedPosition)
                               .filter(
                                   SimulatedPosition.snapshot_date == (
                                       session.query(SimulatedPosition.snapshot_date)
                                              .order_by(SimulatedPosition.snapshot_date.desc())
                                              .limit(1).scalar_subquery()
                                   ),
                                   SimulatedPosition.track == "main",
                               ).all()
                    )
                    for _pos in _active:
                        _w = _pos.actual_weight or 0.0
                        if _w > _regime_off_max:
                            _close = _fetch_close(_pos.ticker, t_day)
                            _reason = (
                                f"regime_jump_compress: p_change={_p_change:.2f}, "
                                f"weight {_w:.1%} → {_regime_off_max:.1%}"
                            )
                            _auto_execute_stop(
                                session, _pos, t_day, _close or (_pos.entry_price or 0),
                                reason=_reason, target_weight=_regime_off_max,
                            )
                            result.tactical_reduces.append(_pos.sector)
                            result.risk_alerts.append(f"{_pos.sector}:regime_jump_compress:auto_executed")
                    # Informational PendingApproval (no action required — archive only)
                    session.add(PendingApproval(
                        approval_type="risk_control",
                        priority="normal",
                        sector="ALL",
                        ticker="",
                        triggered_condition=f"regime_jump_compress: 制度跃变记录，已自动压缩 {len(result.tactical_reduces)} 个仓位",
                        triggered_date=t_day,
                        triggered_price=0.0,
                        suggested_weight=0.0,
                        status="auto_approved",
                    ))
                    session.commit()
            else:
                # Auto-compress off: generate Layer 3 approval
                result.risk_alerts.append("ALL:regime_jump:pending_compress")
    except Exception as _je:
        logger.debug("_patrol_daily_tactical event 1 failed: %s", _je)

    # Skip events 2-5 if regime jump was handled
    if _regime_jump:
        return

    # ── Fetch shared data for events 2-5 ─────────────────────────────────────
    try:
        slow_df = get_signal_dataframe(as_of=t_day)
    except Exception:
        slow_df = pd.DataFrame()

    try:
        fast_df = get_fast_signal_dataframe(as_of=t_day, lookback=_lb_fast, skip=_sk_fast)
    except Exception:
        fast_df = pd.DataFrame()

    with SessionFactory() as session:
        _latest_snap = (
            session.query(SimulatedPosition.snapshot_date)
                   .order_by(SimulatedPosition.snapshot_date.desc())
                   .limit(1).scalar_subquery()
        )
        active_positions = (
            session.query(SimulatedPosition)
                   .filter(
                       SimulatedPosition.snapshot_date == _latest_snap,
                       SimulatedPosition.track == "main",
                   ).all()
        )
        active_sectors = {p.sector: p for p in active_positions}

        # ── Event 2: Fast signal flip ─────────────────────────────────────────
        if not fast_df.empty and not slow_df.empty:
            for _pos in active_positions:
                _w = _pos.actual_weight or 0.0
                if abs(_w) < 1e-4:
                    continue
                _sec = _pos.sector
                if _sec not in fast_df.index or _sec not in slow_df.index:
                    continue
                _fast_sig = int(fast_df.loc[_sec, "tsmom"])
                _slow_sig = int(slow_df.loc[_sec, "tsmom"])
                _entry_sig = _pos.signal_tsmom or 0
                # Fast flipped but Slow hasn't → tactical reduce signal
                _holding_long  = _w > 0 and (_entry_sig > 0 or _slow_sig > 0)
                _holding_short = _w < 0 and (_entry_sig < 0 or _slow_sig < 0)
                _fast_flip = (
                    (_holding_long  and _fast_sig < 0 and _slow_sig >= 0) or
                    (_holding_short and _fast_sig > 0 and _slow_sig <= 0)
                )
                if _fast_flip:
                    _close = _fetch_close(_pos.ticker, t_day)
                    _add_risk_approval(
                        session, _pos, t_day, _close or 0,
                        reason=f"tsmom_fast_flip: fast={_fast_sig:+d} vs holding dir, slow={_slow_sig:+d} unchanged",
                        priority="high",
                        suggested_weight=_w * 0.5,
                    )
                    result.tactical_reduces.append(_sec)
                    result.risk_alerts.append(f"{_sec}:tsmom_fast_flip")

        # ── Event 3: High-confidence entry ────────────────────────────────────
        if regime_label == "risk-on" and not fast_df.empty and not slow_df.empty:
            # Get composite scores for filtering
            _composite: dict[str, float] = {}
            try:
                _cs_df = compute_composite_scores(t_day)
                if not _cs_df.empty and "composite_score" in _cs_df.columns:
                    _composite = _cs_df["composite_score"].to_dict()
                elif not _cs_df.empty:
                    _composite = _cs_df.iloc[:, 0].to_dict()
            except Exception:
                pass

            # 5-day momentum z-score across universe
            _5d_zscore: dict[str, float] = {}
            try:
                sector_etf = get_active_sector_etf()
                _tickers   = list(sector_etf.values())
                _5d_end    = t_day + datetime.timedelta(days=1)
                _5d_start  = t_day - datetime.timedelta(days=10)
                import yfinance as _yf
                _5d_raw = _yf.download(
                    _tickers, start=str(_5d_start), end=str(_5d_end),
                    auto_adjust=True, progress=False, multi_level_index=False
                )
                if not _5d_raw.empty:
                    _5d_close = _5d_raw["Close"] if isinstance(_5d_raw.columns, pd.MultiIndex) else _5d_raw
                    _5d_ret = _5d_close.pct_change(5).iloc[-1]
                    _ticker_to_sector = {v: k for k, v in sector_etf.items()}
                    _ret_by_sector = {_ticker_to_sector.get(str(t), ""): float(r)
                                      for t, r in _5d_ret.items() if not pd.isna(r)}
                    _ret_vals = [v for v in _ret_by_sector.values() if v != 0]
                    if len(_ret_vals) >= 3:
                        _mu5  = float(pd.Series(_ret_vals).mean())
                        _std5 = float(pd.Series(_ret_vals).std())
                        if _std5 > 1e-9:
                            _5d_zscore = {s: (r - _mu5) / _std5 for s, r in _ret_by_sector.items() if s}
            except Exception as _5de:
                logger.debug("5d z-score fetch failed: %s", _5de)

            _entries_today = 0
            for _sec, _ticker in get_active_sector_etf().items():
                if _entries_today >= _entry_limit:
                    break
                if _sec not in fast_df.index or _sec not in slow_df.index:
                    continue
                # Must be unpositioned
                _pos = active_sectors.get(_sec)
                if _pos and abs(_pos.actual_weight or 0.0) > 0.005:
                    continue
                _fast = int(fast_df.loc[_sec, "tsmom"])
                _slow = int(slow_df.loc[_sec, "tsmom"])
                if _fast != 1 or _slow != 1:
                    continue

                # Check composite_score and 5d z-score
                _score    = _composite.get(_sec, 0.0)
                _z5       = _5d_zscore.get(_sec, 0.0)
                _score_ok = _score >= _score_min
                _z_ok     = _z5 >= _zscore_min

                # vol-parity weight from slow signal df
                _inv_vol = float(slow_df.loc[_sec, "inv_vol_wt"]) if "inv_vol_wt" in slow_df.columns else 0.0
                _inv_vol_sum = float(slow_df["inv_vol_wt"].sum()) if "inv_vol_wt" in slow_df.columns else 1.0
                _vp_weight = min(_inv_vol / _inv_vol_sum if _inv_vol_sum > 1e-9 else 0.05, _entry_max_w)
                _close = _fetch_close(_ticker, t_day)

                if _score_ok and _z_ok and _auto_entry:
                    # Layer 2: auto-execute — write to SimulatedPosition immediately
                    try:
                        from engine.portfolio_tracker import apply_tactical_weight_update
                        apply_tactical_weight_update(
                            update_date=t_day,
                            new_entries=[{
                                "sector":        _sec,
                                "ticker":        _ticker,
                                "target_weight": _vp_weight,
                                "actual_weight": _vp_weight,
                                "entry_price":   _close or 0.0,
                                "regime_label":  regime_label,
                                "signal_tsmom":  _fast,
                                "track":         "main",
                            }],
                        )
                        result.tactical_entries.append(_sec)
                        result.entries_triggered.append(_sec)
                        _entries_today += 1
                        logger.info(
                            "_patrol_daily_tactical: Layer-2 tactical entry %s "
                            "weight=%.1f%% score=%.0f z5=%.2f",
                            _sec, _vp_weight * 100, _score, _z5,
                        )
                    except Exception as _ue:
                        logger.warning("tactical auto-entry failed: %s", _ue)
                else:
                    # Layer 3 approval
                    _reason_parts = []
                    if not _score_ok:
                        _reason_parts.append(f"composite_score={_score:.0f}<{_score_min}")
                    if not _z_ok:
                        _reason_parts.append(f"5d_z={_z5:.2f}<{_zscore_min}")
                    _reason = "tactical_entry_pending: " + ("; ".join(_reason_parts) or "all conditions met, auto_entry disabled")
                    # Create PendingApproval for entry
                    _existing_pa = (
                        session.query(PendingApproval)
                               .filter(
                                   PendingApproval.sector == _sec,
                                   PendingApproval.approval_type == "entry",
                                   PendingApproval.status == "pending",
                                   PendingApproval.triggered_date == t_day,
                               ).first()
                    )
                    if not _existing_pa:
                        session.add(PendingApproval(
                            approval_type="entry",
                            priority="normal",
                            sector=_sec,
                            ticker=_ticker,
                            triggered_condition=_reason,
                            triggered_date=t_day,
                            triggered_price=_close or 0,
                            suggested_weight=_vp_weight,
                        ))
                        result.entries_triggered.append(_sec)

        # ── Event 4: Drawdown Alert ───────────────────────────────────────────
        for _pos in active_positions:
            _w = _pos.actual_weight or 0.0
            if abs(_w) < 1e-4:
                continue
            _close = _fetch_close(_pos.ticker, t_day)
            if not _close or not _pos.entry_price:
                continue
            _unrealized = _close / _pos.entry_price - 1.0
            if _unrealized < -0.08:   # < -8% unrealized loss
                _existing = (
                    session.query(PendingApproval)
                           .filter(
                               PendingApproval.sector == _pos.sector,
                               PendingApproval.approval_type == "risk_control",
                               PendingApproval.status == "pending",
                               PendingApproval.triggered_condition.like("%drawdown_alert%"),
                           ).first()
                )
                if not _existing:
                    _add_risk_approval(
                        session, _pos, t_day, _close,
                        reason=f"drawdown_alert: unrealized={_unrealized:.1%} < -8% (entry={_pos.entry_price:.2f})",
                        priority="high",
                        suggested_weight=_w * 0.5,
                    )
                    result.tactical_reduces.append(_pos.sector)
                    result.risk_alerts.append(f"{_pos.sector}:drawdown_alert")

        # ── Event 5: Vol Spike Compress ───────────────────────────────────────
        for _pos in active_positions:
            _w = _pos.actual_weight or 0.0
            if abs(_w) < 1e-4:
                continue
            _close = _fetch_close(_pos.ticker, t_day)
            if not _close:
                continue
            try:
                from engine.quant_agent import _fetch_price_context
                _atr21, _ = _fetch_price_context(_pos.ticker, t_day, atr_period=21)
                if _atr21 <= 0:
                    continue
                _atr_pct = _atr21 / _close
                if _atr_pct > 0.03:   # ATR/price > 3%
                    # Check if overweighted vs target
                    _tgt_w = _pos.target_weight or _w
                    if abs(_w) > abs(_tgt_w) * 1.5:
                        _existing = (
                            session.query(PendingApproval)
                                   .filter(
                                       PendingApproval.sector == _pos.sector,
                                       PendingApproval.approval_type == "risk_control",
                                       PendingApproval.status == "pending",
                                       PendingApproval.triggered_condition.like("%vol_spike_compress%"),
                                   ).first()
                        )
                        if not _existing:
                            _add_risk_approval(
                                session, _pos, t_day, _close,
                                reason=(
                                    f"vol_spike_compress: ATR/price={_atr_pct:.1%} > 3%, "
                                    f"w={_w:.1%} > 1.5×target={_tgt_w:.1%}"
                                ),
                                priority="normal",
                                suggested_weight=_tgt_w,
                            )
                            result.tactical_reduces.append(_pos.sector)
                            result.risk_alerts.append(f"{_pos.sector}:vol_spike_compress")
            except Exception:
                continue

        session.commit()


# ── P5: FinDebate auto-trigger ────────────────────────────────────────────────

def _run_auto_debate(t_day: datetime.date, result: BatchResult) -> None:
    """
    Auto-trigger FinDebate for:
      (a) sectors in result.entries_triggered (new entry candidates)
      (b) if regime changed today (re-assess top-weight positions)

    Results are stored in result.debate_results and appended to the
    corresponding PendingApproval notes so the supervisor sees the full
    reasoning chain in the Daily Brief.

    Skipped silently if no LLM model is available (CB=MEDIUM or no key).
    """
    try:
        from engine.key_pool import get_pool as _get_pool
        _pool = _get_pool()
        _pool.check_billing_limits()
        _model = _pool.get_model()
    except Exception:
        logger.debug("_run_auto_debate: no LLM available, skipping")
        return

    # Collect debate targets
    _targets: list[str] = list(result.entries_triggered)

    # On regime change, also re-assess the largest current positions (top 3)
    _regime_changed = False
    try:
        from engine.memory import get_daily_brief_snapshot
        _prev = get_daily_brief_snapshot(t_day - datetime.timedelta(days=1))
        _today_snap = get_daily_brief_snapshot(t_day)
        if _prev and _today_snap:
            _regime_changed = bool(_today_snap.regime_changed)
        if _regime_changed:
            from engine.portfolio_tracker import get_current_positions
            _pos = get_current_positions()
            if not _pos.empty and "actual_weight" in _pos.columns:
                _top = _pos.nlargest(3, "actual_weight").index.tolist()
                for _s in _top:
                    if _s not in _targets:
                        _targets.append(_s)
    except Exception:
        pass

    if not _targets:
        return

    from engine.history import get_active_sector_etf
    from engine.debate import run_sector_debate
    _etf_map = get_active_sector_etf()
    _limit   = 4   # max sectors per day to control LLM cost

    for _sector in _targets[:_limit]:
        try:
            _ticker = _etf_map.get(_sector, "")
            _debate = run_sector_debate(
                sector=_sector,
                tickers=[_ticker] if _ticker else [],
                macro_context=f"T={t_day}, regime_changed={_regime_changed}",
                quant_gate=None,
            )
            _conclusion = _debate.get("final_output", "")
            result.debate_sectors.append(_sector)
            result.debate_results[_sector] = _conclusion[:300]   # truncate for BatchResult

            # Append to PendingApproval notes if entry is pending
            try:
                with SessionFactory() as _ds:
                    _pa = (
                        _ds.query(PendingApproval)
                           .filter(
                               PendingApproval.sector == _sector,
                               PendingApproval.approval_type == "entry",
                               PendingApproval.status == "pending",
                               PendingApproval.triggered_date == t_day,
                           ).first()
                    )
                    if _pa:
                        _existing = _pa.notes or ""
                        _pa.notes = (_existing + f"\n[FinDebate {t_day}] " + _conclusion[:500]).strip()
                        _ds.commit()
            except Exception as _pe:
                logger.debug("_run_auto_debate: PA notes update failed for %s: %s", _sector, _pe)

            logger.info("_run_auto_debate: debate complete for %s", _sector)
        except Exception as _de:
            logger.debug("_run_auto_debate: debate failed for %s: %s", _sector, _de)


# ── Step 6: month-end rebalance check ─────────────────────────────────────────

def _patrol_rebalance(t_day: datetime.date, result: BatchResult) -> None:
    if not _is_month_end_trading_day(t_day):
        return

    try:
        from engine.config import get_trading_config
        from engine.portfolio import construct_portfolio
        from engine.history import get_active_sector_etf
        from engine.portfolio_tracker import execute_rebalance

        _cfg      = get_trading_config()
        _auto     = _cfg.get("monthly_rebalance_auto", False)
        signal_df = get_signal_dataframe(as_of=t_day)
        regime_result = get_regime_on(t_day)
        sector_etf    = get_active_sector_etf()

        if signal_df.empty:
            return

        regime_label = getattr(regime_result, "regime", "transition")

        # ── Auto-execution conditions ─────────────────────────────────────────
        _block_reasons: list[str] = []
        if not _auto:
            _block_reasons.append("monthly_rebalance_auto=false")
        if regime_label == "risk-off":
            _block_reasons.append("制度=risk-off，需人工确认")
        if result.cb_level in ("medium", "severe"):
            _block_reasons.append(f"Circuit Breaker={result.cb_level}")

        # Check ICIR guard: at least one active FactorMAD run this month
        try:
            from engine.memory import get_daily_brief_snapshot
            _month_str = t_day.strftime("%Y-%m")
            _icir_snap = get_daily_brief_snapshot(t_day)
            if not (_icir_snap and getattr(_icir_snap, "icir_month", None) == _month_str):
                _block_reasons.append("本月 FactorMAD ICIR 尚未更新")
        except Exception:
            pass

        if _auto and not _block_reasons:
            # Layer 2: auto-execute rebalance
            try:
                _rb_result = execute_rebalance(t_day, dry_run=False)
                result.rebalance_auto = True
                result.rebalance_orders = [
                    t["sector"] for t in _rb_result.get("trades", [])
                ]
                result.risk_alerts.append(f"rebalance:auto_executed:{t_day}")
                logger.info(
                    "_patrol_rebalance: auto-rebalanced %d sectors, "
                    "turnover=%.1f%%, cost=%.1fbps",
                    len(result.rebalance_orders),
                    _rb_result.get("turnover", 0) * 100,
                    _rb_result.get("total_cost_bps", 0),
                )
            except Exception as _ae:
                _block_reasons.append(f"auto-execute failed: {_ae}")
                result.errors.append(f"auto_rebalance: {_ae}")

        if not result.rebalance_auto:
            # Layer 3: generate pending approvals with block reasons
            result.rebalance_skipped_reason = "；".join(_block_reasons)
            target_weights = construct_portfolio(
                signal_df=signal_df, regime=regime_result
            ).weights

            with SessionFactory() as session:
                latest_date = (
                    session.query(SimulatedPosition.snapshot_date)
                           .order_by(SimulatedPosition.snapshot_date.desc())
                           .scalar()
                )
                positions = {}
                if latest_date:
                    positions = {
                        p.sector: p.actual_weight or 0.0
                        for p in session.query(SimulatedPosition)
                                       .filter(SimulatedPosition.snapshot_date == latest_date)
                    }

                for sector, target_w in target_weights.items():
                    current_w = positions.get(sector, 0.0)
                    abs_diff  = abs(target_w - current_w)
                    rel_diff  = abs_diff / max(current_w, 0.01)
                    if abs_diff > 0.02 or rel_diff > 0.20:
                        ticker = sector_etf.get(sector, sector)
                        close  = _fetch_close(ticker, t_day)
                        _note  = f"阻塞原因：{result.rebalance_skipped_reason}" if result.rebalance_skipped_reason else ""
                        session.add(PendingApproval(
                            approval_type="rebalance",
                            priority="normal",
                            sector=sector,
                            ticker=ticker,
                            triggered_condition=(
                                f"月末再平衡：当前 {current_w:.1%} → "
                                f"目标 {target_w:.1%}（Δ {abs_diff:.1%}）"
                            ),
                            triggered_date=t_day,
                            triggered_price=close,
                            suggested_weight=float(target_w),
                            notes=_note,
                        ))
                        result.rebalance_orders.append(sector)
                session.commit()

    except Exception as exc:
        result.errors.append(f"rebalance: {exc}")
        logger.exception("DailyBatch rebalance step failed")


# ── Public entry point ────────────────────────────────────────────────────────

def run_daily_batch(
    as_of_date: Optional[datetime.date] = None,
    force: bool = False,
) -> BatchResult:
    """
    Execute the full daily trading batch for T-day.

    Args:
        as_of_date : override T-day (default: last NYSE trading day)
        force      : bypass idempotency guard (for testing / manual re-run)

    Returns:
        BatchResult with step outcomes and generated alerts.
    """
    t_day  = as_of_date or _get_last_nyse_trading_day()
    result = BatchResult(as_of_date=t_day)

    if not force and _snapshot_exists(t_day):
        result.skipped = True
        logger.debug("DailyBatch: snapshot already exists for %s, skipping", t_day)
        return result

    logger.info("DailyBatch: starting for %s", t_day)

    _run_signal_regime(t_day, result)
    if result.errors:
        logger.error("DailyBatch: aborting after signal/regime failure: %s", result.errors)
        return result

    _patrol_invalidations(t_day, result)
    _patrol_entries(t_day, result)
    _run_auto_debate(t_day, result)          # P5: FinDebate for entry candidates + regime change
    _patrol_positions(t_day, result)
    _patrol_daily_tactical(t_day, result)
    _patrol_rebalance(t_day, result)

    logger.info(
        "DailyBatch %s complete — invalidations=%d entries=%d corr_blocked=%d risk=%d rebalance=%d errors=%d",
        t_day,
        len(result.invalidations),
        len(result.entries_triggered),
        len(result.corr_blocked),
        len(result.risk_alerts),
        len(result.rebalance_orders),
        len(result.errors),
    )
    return result


def _get_signal_date() -> datetime.date:
    """
    Return the correct as_of date for signal computation.

    If called before NYSE close (4pm ET) today's intraday bar is incomplete —
    use T-1 to avoid feeding partial prices into signal logic.
    """
    try:
        import pytz
        now_et = datetime.datetime.now(pytz.timezone("America/New_York"))
        today  = now_et.date()
        close_et = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        if now_et >= close_et:
            # After 4pm ET: today's close bar is available
            return _get_last_nyse_trading_day(today)
        else:
            # Before 4pm ET: skip today's incomplete bar, use previous trading day
            return _get_last_nyse_trading_day(today - datetime.timedelta(days=1))
    except ImportError:
        # pytz unavailable: conservative fallback — always use previous trading day
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        return _get_last_nyse_trading_day(yesterday)


def _generate_macro_brief_llm(
    model,
    regime: str,
    regime_prev: str,
    regime_changed: bool,
    p_risk_on: float,
    vix: float,
    risk_alerts: list[str],
    entries: list[str],
    rebalance: list[str],
) -> str:
    """
    调用 LLM 生成一段 100-150 字的中文宏观简报。
    仅在 model 可用时调用；失败时静默返回空字符串。
    """
    events_desc = []
    hard_stops  = [a.split(":")[0] for a in risk_alerts if "hard_stop"  in a]
    flips       = [a.split(":")[0] for a in risk_alerts if "tsmom_flip" in a]
    compress    = [a for a in risk_alerts if "regime_compress" in a]
    if hard_stops:
        events_desc.append(f"止损触发：{', '.join(hard_stops)}")
    if flips:
        events_desc.append(f"TSMOM 翻转：{', '.join(flips)}")
    if compress:
        events_desc.append(f"制度压缩 {len(compress)} 笔")
    if entries:
        events_desc.append(f"入场触发：{', '.join(entries)}")
    if rebalance:
        events_desc.append(f"月末再平衡 {len(rebalance)} 笔")
    events_str = "；".join(events_desc) if events_desc else "无风控事件触发"

    regime_change_line = ""
    if regime_changed and regime_prev:
        regime_change_line = f"\n⚠ 制度切换：{regime_prev} → {regime}（重要）"

    prompt = f"""你是宏观量化基金的投资总监助理。根据以下今日系统数据，输出结构化宏观简报。

当前制度：{regime.upper()}（P risk-on = {p_risk_on:.0%}）{regime_change_line}
VIX 水平：{vix:.1f}
今日事件：{events_str}

brief_text 字段要求：
- 专业量化视角，不超过150字
- 点明制度含义及对仓位的影响（如需调整方向）
- 若有制度切换，重点说明切换含义和短期操作建议
- 不用"根据"、"基于"等套话开头，直接陈述判断"""

    try:
        from engine.key_pool import get_pool
        from engine.trading_schema import StructuredMacroBrief, STRUCTURED_MACRO_BRIEF_SCHEMA
        import json as _json
        _pool = get_pool()
        _m = _pool.get_model(response_schema=STRUCTURED_MACRO_BRIEF_SCHEMA)
        raw = _m.generate_content(prompt).text
        _pool.report_success(has_content=bool(raw.strip()))
        data = _json.loads(raw)
        brief = StructuredMacroBrief.from_json(data)
        return brief.brief_text
    except Exception as exc:
        logger.warning("_generate_macro_brief_llm (structured) failed: %s; falling back", exc)
        try:
            resp = model.generate_content(prompt)
            return resp.text.strip() if resp and resp.text else ""
        except Exception as exc2:
            logger.warning("_generate_macro_brief_llm fallback also failed: %s", exc2)
            return ""


def _generate_narrative(
    t_day: datetime.date,
    result: BatchResult,
    regime: str = "unknown",
    p_risk_on: float = 0.5,
    regime_prev: str = "",
    regime_changed: bool = False,
) -> str:
    """
    P3-11: Delegates to NarrativeBuilder for structured 3-section narrative.
    Kept as a thin wrapper so call-sites in ensure_daily_batch_completed() need
    no change beyond passing the extra kwargs.
    """
    try:
        from engine.narrative_builder import build_batch_narrative
        return build_batch_narrative(
            t_day, result,
            regime=regime, p_risk_on=p_risk_on,
            regime_prev=regime_prev, regime_changed=regime_changed,
        )
    except Exception as exc:
        logger.warning("narrative_builder failed, falling back to legacy: %s", exc)
        # Legacy fallback (original logic)
        if result.skipped:
            s1 = f"{t_day} 批次早前已完成，系统维持实时监控。"
        elif result.errors:
            s1 = f"{t_day} 日链完成，但出现 {len(result.errors)} 个步骤异常。"
        elif result.signal_ok and result.regime_ok:
            s1 = f"{t_day} 日链正常完成 — 信号与宏观制度均已更新。"
        else:
            s1 = f"{t_day} 日链完成，但信号或制度数据存在缺失。"
        events = []
        for kw, label in [("hard_stop","止损"),("tsmom_flip","翻转"),("regime_compress","制度压缩")]:
            items = [a for a in result.risk_alerts if kw in a]
            if items:
                events.append(f"{label} {len(items)} 笔")
        return f"{s1} {'；'.join(events) + '。' if events else '今日无触发事件。'}"


# ── P6: QuantOnly shadow NAV ─────────────────────────────────────────────────

def _update_quant_only_snapshot(t_day: datetime.date) -> None:
    """
    Compute and persist a daily NAV row for the QuantOnly shadow portfolio.
    Uses actual_weight from the latest SimulatedPosition (main track) and
    today's close vs. yesterday's close for each holding.
    Idempotent: skips if today's row already exists.
    """
    from engine.memory import QuantOnlySnapshot, SimulatedPosition, SessionFactory as _SF
    try:
        with _SF() as _s:
            if _s.query(QuantOnlySnapshot).filter(QuantOnlySnapshot.date == t_day).first():
                return
            latest = (
                _s.query(SimulatedPosition.snapshot_date)
                  .order_by(SimulatedPosition.snapshot_date.desc())
                  .limit(1).scalar()
            )
            if not latest:
                return
            positions = (
                _s.query(SimulatedPosition)
                  .filter(
                      SimulatedPosition.snapshot_date == latest,
                      SimulatedPosition.track == "main",
                  )
                  .all()
            )
    except Exception as exc:
        logger.debug("_update_quant_only_snapshot: DB read failed: %s", exc)
        return

    if not positions:
        return

    # Fetch daily returns for each holding
    import json as _json
    weights: dict[str, float] = {}
    daily_ret = 0.0
    prev_day  = t_day - datetime.timedelta(days=1)
    for pos in positions:
        w = float(pos.actual_weight or 0.0)
        if w == 0.0:
            continue
        weights[pos.ticker] = w
        try:
            close_t  = _fetch_close(pos.ticker, t_day)
            close_tm1 = _fetch_close(pos.ticker, prev_day)
            if close_t and close_tm1 and close_tm1 > 0:
                daily_ret += w * (close_t / close_tm1 - 1.0)
        except Exception:
            pass

    # Compute NAV: chain-link from most recent row
    try:
        with _SF() as _s:
            prev_snap = (
                _s.query(QuantOnlySnapshot)
                  .order_by(QuantOnlySnapshot.date.desc())
                  .first()
            )
            prev_nav = float(prev_snap.nav) if prev_snap else 1000.0
            new_nav  = round(prev_nav * (1.0 + daily_ret), 4)
            _s.add(QuantOnlySnapshot(
                date=t_day,
                nav=new_nav,
                daily_return=round(daily_ret, 6),
                weights_json=_json.dumps(weights),
            ))
            _s.commit()
    except Exception as exc:
        logger.debug("_update_quant_only_snapshot: write failed: %s", exc)


# ── P6: Calendar helpers ─────────────────────────────────────────────────────

def _is_first_trading_day_of_quarter(d: datetime.date) -> bool:
    """True if d is the first NYSE trading day of its calendar quarter."""
    quarter_months = [1, 4, 7, 10]
    if d.month not in quarter_months:
        return False
    try:
        import pandas_market_calendars as mcal
        nyse  = mcal.get_calendar("NYSE")
        month_start = d.replace(day=1)
        sched = nyse.schedule(
            start_date=month_start.isoformat(),
            end_date=(month_start + datetime.timedelta(days=7)).isoformat(),
        )
        if sched.empty:
            return False
        return sched.index[0].date() == d
    except Exception:
        first = d.replace(day=1)
        while first.weekday() >= 5:
            first += datetime.timedelta(days=1)
        return d == first


# ── P6: Data Quality Gate (Step 1) ───────────────────────────────────────────

def _step1_data_quality(t_day: datetime.date) -> str:
    """
    Check OHLCV and FRED data freshness.
    Returns 'ok' / 'warning' (continue) / 'light' (trigger CB LIGHT).
    Writes one DataQualityLog row per check type.
    """
    from engine.memory import DataQualityLog, SessionFactory as _SF

    def _log(check_type: str, status: str, detail: str = "") -> None:
        try:
            with _SF() as _s:
                _s.add(DataQualityLog(
                    date=t_day, check_type=check_type,
                    status=status, detail=detail[:500] if detail else None,
                    checked_at=datetime.datetime.utcnow(),
                ))
                _s.commit()
        except Exception:
            pass

    overall = "ok"

    # ── Check 1: OHLCV freshness (SPY as proxy) ───────────────────────────────
    try:
        import yfinance as _yf
        _px = _yf.download("SPY", period="5d", progress=False, auto_adjust=True)
        if not _px.empty:
            last_data_date = _px.index[-1].date()
            delay = (t_day - last_data_date).days
            if delay > 1:
                _log("ohlcv_freshness", "warning", f"OHLCV 延迟 {delay} 天")
                overall = "warning"
            else:
                _log("ohlcv_freshness", "ok")
        else:
            _log("ohlcv_freshness", "warning", "yfinance 返回空数据")
            overall = "warning"
    except Exception as exc:
        _log("ohlcv_freshness", "light", str(exc))
        return "light"

    # ── Check 2: FRED DGS10 freshness ─────────────────────────────────────────
    try:
        import pandas as _pd
        _fred_df = _yf.download("^TNX", period="5d", progress=False, auto_adjust=True)
        if not _fred_df.empty:
            fred_last = _fred_df.index[-1].date()
            fred_delay = (t_day - fred_last).days
            if fred_delay > 3:
                _log("fred_delay", "light", f"DGS10代理数据延迟 {fred_delay} 天")
                overall = "light" if overall == "ok" else overall
            else:
                _log("fred_delay", "ok")
        else:
            _log("fred_delay", "warning", "FRED代理数据返回空")
    except Exception as exc:
        _log("fred_delay", "warning", str(exc))

    _log("overall", overall)
    return overall


# ── P6: Per-ticker signal persistence (after Step 2) ─────────────────────────

def _write_signal_record(
    t_day: datetime.date,
    signal_df: "pd.DataFrame",
    regime_label: str,
) -> None:
    """
    Persist per-ticker signals to signal_records table (idempotent on date+ticker).
    Called immediately after signal computation; feeds flip detection and decay patrol.
    """
    from engine.memory import SignalRecord, SessionFactory as _SF
    from engine.history import get_active_sector_etf
    active_etf = get_active_sector_etf()

    with _SF() as _s:
        for sector, row in signal_df.iterrows():
            ticker = active_etf.get(str(sector), str(row.get("ticker", "")))
            # Skip if already written for this date/ticker (idempotent)
            existing = _s.query(SignalRecord).filter(
                SignalRecord.date == t_day,
                SignalRecord.ticker == ticker,
            ).first()
            if existing:
                continue
            _s.add(SignalRecord(
                date=t_day,
                ticker=ticker,
                sector=str(sector),
                tsmom_signal=int(row.get("tsmom", 0) or 0),
                tsmom_raw=float(row.get("raw_return", 0.0) or 0.0),
                csmom_rank=float(row.get("csmom_rank", 0.5) or 0.5),
                carry_norm=float(row.get("carry_norm", 0.0) or 0.0),
                reversal_norm=float(row.get("reversal_norm", 0.0) or 0.0),
                factormad_score=float(row.get("factor_mad_score", 0.0) or 0.0)
                               if not pd.isna(row.get("factor_mad_score", float("nan"))) else None,
                composite_score=float(row.get("composite_score", 50.0) or 50.0),
                gate_status="passed" if float(row.get("composite_score", 50.0) or 50.0) >= 35
                            else "blocked",
                regime_at_calc=regime_label,
            ))
        _s.commit()
    logger.debug("_write_signal_record: wrote %d rows for %s", len(signal_df), t_day)


def _detect_signal_flips(
    t_day: datetime.date,
    signal_df: "pd.DataFrame",
) -> list[str]:
    """
    Compare today's TSMOM signal vs. most recent prior SignalRecord.
    Write SignalFlipLog rows for any direction changes.
    Returns list of flipped sector names (for BatchResult / Daily Brief).
    """
    from engine.memory import SignalRecord, SignalFlipLog, SessionFactory as _SF
    from engine.history import get_active_sector_etf

    active_etf = get_active_sector_etf()
    flipped: list[str] = []

    with _SF() as _s:
        for sector, row in signal_df.iterrows():
            ticker = active_etf.get(str(sector), str(row.get("ticker", "")))
            curr_sig = int(row.get("tsmom", 0) or 0)

            prev_rec = (
                _s.query(SignalRecord)
                  .filter(
                      SignalRecord.ticker == ticker,
                      SignalRecord.date < t_day,
                  )
                  .order_by(SignalRecord.date.desc())
                  .first()
            )
            if prev_rec is None:
                continue
            if prev_rec.tsmom_signal == curr_sig:
                continue

            # Direction changed — write flip log
            _s.add(SignalFlipLog(
                date=t_day,
                ticker=ticker,
                sector=str(sector),
                prev_signal=prev_rec.tsmom_signal,
                new_signal=curr_sig,
                tsmom_raw_prev=prev_rec.tsmom_raw,
                tsmom_raw_new=float(row.get("raw_return", 0.0) or 0.0),
                regime_at_flip=prev_rec.regime_at_calc,
            ))
            flipped.append(str(sector))
        _s.commit()

    if flipped:
        logger.info("_detect_signal_flips: %d flips on %s: %s", len(flipped), t_day, flipped)
    return flipped


# ── P6: Signal decay patrol (5d) ─────────────────────────────────────────────

def _patrol_signal_decay(
    t_day: datetime.date,
    positions: list,
    signal_df: "pd.DataFrame",
    result: BatchResult,
) -> None:
    """
    Check if TSMOM raw_return has decayed >60% from entry peak for active positions.
    Replaces the arbitrary 90-day age rule: signal strength decline is a direct
    measure of thesis validity erosion, not just time elapsed.
    Pushes REVIEW_SIGNAL_DECAY to PendingApproval (Layer 3, human reviews).
    """
    from engine.memory import PendingApproval, SessionFactory as _SF
    from engine.history import get_active_sector_etf

    DECAY_THRESHOLD = 0.60

    with _SF() as _s:
        for pos in positions:
            sector = pos.sector
            if sector not in signal_df.index:
                continue
            curr_raw = abs(float(signal_df.loc[sector, "raw_return"] or 0.0))
            entry_raw = abs(float(pos.entry_price or 0.0))   # proxy: use entry composite_score
            # Better: retrieve entry tsmom_raw from SignalRecord at entry date
            entry_rec = None
            if pos.snapshot_date:
                entry_rec = (
                    _s.query(__import__("engine.memory", fromlist=["SignalRecord"]).SignalRecord)
                    .filter(
                        __import__("engine.memory", fromlist=["SignalRecord"]).SignalRecord.ticker == pos.ticker,
                        __import__("engine.memory", fromlist=["SignalRecord"]).SignalRecord.date <= pos.snapshot_date,
                    )
                    .order_by(__import__("engine.memory", fromlist=["SignalRecord"]).SignalRecord.date.desc())
                    .first()
                )
            peak_raw = abs(entry_rec.tsmom_raw) if entry_rec and entry_rec.tsmom_raw else 0.0
            if peak_raw < 0.01:
                continue
            decay = 1.0 - curr_raw / peak_raw
            if decay > DECAY_THRESHOLD:
                existing = _s.query(PendingApproval).filter(
                    PendingApproval.sector == sector,
                    PendingApproval.approval_type == "risk_control",
                    PendingApproval.status == "pending",
                    PendingApproval.triggered_date == t_day,
                ).first()
                if not existing:
                    close = _fetch_close(pos.ticker, t_day)
                    _s.add(PendingApproval(
                        approval_type="risk_control",
                        priority="normal",
                        sector=sector,
                        ticker=pos.ticker,
                        triggered_condition=(
                            f"Signal decay: TSMOM raw从 {peak_raw:.3f} 衰减至 {curr_raw:.3f}"
                            f" ({decay:.0%} ↓, 超过 {DECAY_THRESHOLD:.0%} 阈值)"
                        ),
                        triggered_date=t_day,
                        triggered_price=close,
                        suggested_weight=0.0,
                    ))
                    result.risk_alerts.append(f"{sector}:signal_decay")
        _s.commit()


# ── P6: Vol spike patrol using ATR(21) (5e) ──────────────────────────────────

def _patrol_vol_spike_p6(
    t_day: datetime.date,
    positions: list,
    result: BatchResult,
) -> None:
    """
    P6 vol spike patrol using ATR(21) — current vol snapshot.
    Distinct from 5a stop-loss which uses ATR(63) for monthly horizon.

    ATR(21)/price > 3% → Layer 3 reduce to 50% suggestion
    ATR(21)/price > 5% → Layer 3 exit suggestion (critical priority)
    """
    from engine.memory import PendingApproval, SessionFactory as _SF

    with _SF() as _s:
        for pos in positions:
            if (pos.actual_weight or 0.0) == 0.0:
                continue
            close = _fetch_close(pos.ticker, t_day)
            if not close or close <= 0:
                continue
            try:
                from engine.quant_agent import _fetch_price_context
                atr21, _ = _fetch_price_context(pos.ticker, t_day, atr_period=21)
            except Exception:
                continue
            if not atr21 or atr21 <= 0:
                continue

            ratio = atr21 / close
            if ratio <= 0.03:
                continue

            priority = "critical" if ratio > 0.05 else "normal"
            action   = "exit" if ratio > 0.05 else "reduce_50pct"
            suggested = 0.0 if ratio > 0.05 else round(abs(pos.actual_weight or 0.0) * 0.5, 4)

            existing = _s.query(PendingApproval).filter(
                PendingApproval.sector == pos.sector,
                PendingApproval.approval_type == "risk_control",
                PendingApproval.status == "pending",
                PendingApproval.triggered_date == t_day,
            ).first()
            if not existing:
                _s.add(PendingApproval(
                    approval_type="risk_control",
                    priority=priority,
                    sector=pos.sector,
                    ticker=pos.ticker,
                    triggered_condition=(
                        f"Vol spike P6: ATR(21)/price={ratio:.1%} "
                        f"({'> 5%，建议清仓' if ratio > 0.05 else '> 3%，建议减仓至50%'})"
                    ),
                    triggered_date=t_day,
                    triggered_price=close,
                    suggested_weight=suggested,
                ))
                result.risk_alerts.append(f"{pos.sector}:vol_spike_p6:{action}")
        _s.commit()


# ── P6: CB event logger ────────────────────────────────────────────────────────

def _log_cb_event(level: str, reason: str, auto_resolved: bool = False) -> None:
    """Write a CircuitBreakerLog row (persistent audit trail, complements JSON state)."""
    from engine.memory import CircuitBreakerLog, SessionFactory as _SF
    try:
        with _SF() as _s:
            _s.add(CircuitBreakerLog(
                triggered_at=datetime.datetime.utcnow(),
                level=level,
                reason=reason,
                auto_resolved=auto_resolved,
            ))
            _s.commit()
    except Exception as exc:
        logger.debug("_log_cb_event failed: %s", exc)


def ensure_daily_batch_completed(model=None) -> BatchResult:
    """
    Idempotent entry point called from app.py on every page navigation.

    Step 0  : Circuit Breaker check — SEVERE aborts, MEDIUM skips LLM
    Fast path: snapshot exists with verify_ran=True → return in <100ms
    Slow path: CB check → signal/regime → patrols → FinDebate → rebalance →
               FactorMAD ICIR → snapshot → LLM brief
    """
    t_day = _get_signal_date()

    # ── Step 0: Circuit Breaker ────────────────────────────────────────────────
    _cb_level  = "none"
    _cb_reason = ""
    try:
        from engine.circuit_breaker import evaluate as _cb_evaluate, LEVEL_SEVERE, LEVEL_MEDIUM
        _cb_state  = _cb_evaluate()
        _cb_level  = _cb_state.level
        _cb_reason = _cb_state.reason
        if _cb_level == LEVEL_SEVERE:
            _log_cb_event(LEVEL_SEVERE, _cb_reason)
            logger.warning("ensure_daily_batch: ABORTED — Circuit Breaker SEVERE: %s", _cb_reason)
            return BatchResult(
                as_of_date=t_day,
                skipped=True,
                cb_level=_cb_level,
                cb_reason=_cb_reason,
                errors=[f"Circuit Breaker SEVERE: {_cb_reason}"],
            )
        if _cb_level == LEVEL_MEDIUM:
            _log_cb_event(LEVEL_MEDIUM, _cb_reason)
            logger.info("ensure_daily_batch: MEDIUM CB — LLM calls will be skipped (%s)", _cb_reason)
            model = None   # suppress all LLM steps
    except Exception as _cbe:
        logger.debug("Circuit Breaker check failed (non-fatal): %s", _cbe)

    # ── Fast path: today's automation already complete ─────────────────────────
    try:
        from engine.memory import get_daily_brief_snapshot
        _snap = get_daily_brief_snapshot(t_day)
        if _snap and _snap.verify_ran and _snap.narrative:
            # Everything done. Only try LLM brief if model now available and brief missing.
            if model is not None and not getattr(_snap, "macro_brief_llm", None):
                try:
                    from engine.memory import upsert_daily_brief_snapshot
                    from engine.quant import QuantEngine
                    vix_now = QuantEngine.get_realtime_vix()
                    _alerts = json.loads(_snap.risk_alerts_json or "[]")
                    llm_brief = _generate_macro_brief_llm(
                        model=model,
                        regime=_snap.regime or "",
                        regime_prev=_snap.regime_prev or "",
                        regime_changed=bool(_snap.regime_changed),
                        p_risk_on=float(_snap.p_risk_on or 0.0),
                        vix=vix_now,
                        risk_alerts=_alerts,
                        entries=[],
                        rebalance=[],
                    )
                    if llm_brief:
                        upsert_daily_brief_snapshot(t_day, macro_brief_llm=llm_brief)
                except Exception as _e:
                    logger.debug("ensure_daily_batch fast-path LLM brief: %s", _e)
            return BatchResult(
                as_of_date=t_day,
                skipped=True,
                signal_ok=bool(_snap.regime),
                regime_ok=bool(_snap.regime),
            )
    except Exception as _e:
        logger.debug("ensure_daily_batch fast-path check failed: %s", _e)

    # ── Slow path: first call today ────────────────────────────────────────────

    # ── P6 Step 1: Data Quality Gate ──────────────────────────────────────────
    try:
        _dq_status = _step1_data_quality(t_day)
        if _dq_status == "light":
            _log_cb_event("LIGHT", "Data quality gate: OHLCV/FRED freshness check failed")
            logger.warning("ensure_daily_batch: data quality LIGHT — continuing with caution")
    except Exception as _dqe:
        logger.debug("ensure_daily_batch: data quality check non-fatal: %s", _dqe)

    result = run_daily_batch(as_of_date=t_day)
    result.cb_level  = _cb_level
    result.cb_reason = _cb_reason

    # ── Post-batch: VERIFY + LEARN ─────────────────────────────────────────────
    # expire_stale_approvals is safe to call repeatedly (idempotent).
    # verify_pending_decisions skips already-verified decisions (verified=True guard).
    n_verified = 0
    try:
        from engine.memory import expire_stale_approvals, verify_pending_decisions
        expire_stale_approvals()
        verify_results = verify_pending_decisions(model=model)
        n_verified = len(verify_results) if verify_results else 0
        if n_verified:
            logger.info("ensure_daily_batch: %d decisions verified (LEARN=%s)",
                        n_verified, model is not None)
    except Exception as exc:
        logger.warning("ensure_daily_batch: VERIFY step failed: %s", exc)

    # ── Post-batch: FactorMAD ICIR monthly update ────────────────────────────
    # Runs only on the first NYSE trading day of each month.
    # DailyBriefSnapshot.icir_month guards against duplicate runs within a month.
    icir_month_ran: Optional[str] = None
    current_month = t_day.strftime("%Y-%m")
    try:
        from engine.memory import get_daily_brief_snapshot as _gds
        snap_today = _gds(t_day)
        already_updated = snap_today and snap_today.icir_month == current_month
        if not already_updated and _is_first_trading_day_of_month(t_day):
            from engine.factor_mad import update_icir
            from engine.memory import SessionFactory as _SF, FactorDefinition as _FD
            # Capture which factors are active before update
            _active_before: set[str] = set()
            try:
                with _SF() as _s:
                    _active_before = {
                        f.factor_id for f in _s.query(_FD)
                        .filter(_FD.status == "active").all()
                    }
            except Exception:
                pass
            update_icir(calc_date=t_day)
            icir_month_ran = current_month
            result.factormad_icir_ran = True
            # Detect newly deactivated factors
            try:
                with _SF() as _s:
                    _active_after = {
                        f.factor_id for f in _s.query(_FD)
                        .filter(_FD.status == "active").all()
                    }
                result.factormad_deactivated = list(_active_before - _active_after)
                if result.factormad_deactivated:
                    logger.info(
                        "ensure_daily_batch: FactorMAD deactivated %s",
                        result.factormad_deactivated,
                    )
            except Exception:
                pass
            logger.info("ensure_daily_batch: FactorMAD ICIR updated for month %s", current_month)
            # Update IC-weighted factor weights
            try:
                from engine.signal import update_factor_ic_weights
                new_w = update_factor_ic_weights(t_day)
                if new_w:
                    logger.info("ensure_daily_batch: factor IC weights updated: %s", new_w)
            except Exception as _e:
                logger.warning("ensure_daily_batch: factor IC weight update failed: %s", _e)
    except Exception as exc:
        logger.warning("ensure_daily_batch: FactorMAD ICIR step failed: %s", exc)

    # ── Post-batch: Regime info for snapshot ─────────────────────────────────
    regime_label, p_risk_on, regime_prev, regime_changed = "", 0.0, "", False
    try:
        from engine.regime import get_regime_on
        regime_result = get_regime_on(as_of=t_day, train_end=t_day)
        regime_label = getattr(regime_result, "regime", "")
        p_risk_on    = float(getattr(regime_result, "p_risk_on", 0.0) or 0.0)
    except Exception:
        pass
    try:
        from engine.memory import get_daily_brief_snapshot
        yesterday = t_day - datetime.timedelta(days=1)
        prev_snap = get_daily_brief_snapshot(yesterday)
        if prev_snap and prev_snap.regime:
            regime_prev    = prev_snap.regime
            regime_changed = (regime_label != "" and regime_label != regime_prev)
    except Exception:
        pass

    # ── P6: Signal persistence + flip detection ──────────────────────────────
    _p6_signal_df = None
    if result.signal_ok:
        try:
            from engine.signal import get_signal_dataframe as _gsd_p6
            _p6_signal_df = _gsd_p6(as_of=t_day)
            _write_signal_record(t_day, _p6_signal_df, regime_label)
            _p6_flips = _detect_signal_flips(t_day, _p6_signal_df)
            if _p6_flips:
                result.invalidations.extend([f"flip:{s}" for s in _p6_flips])
        except Exception as _p6e:
            logger.debug("ensure_daily_batch: P6 signal persistence non-fatal: %s", _p6e)

    # ── P6: Signal decay + vol-spike position patrols ─────────────────────────
    if result.signal_ok:
        try:
            from engine.memory import SimulatedPosition as _SP6
            with SessionFactory() as _sp6_sess:
                _p6_positions = (
                    _sp6_sess.query(_SP6)
                    .filter(
                        _SP6.snapshot_date == (
                            _sp6_sess.query(_SP6.snapshot_date)
                                     .order_by(_SP6.snapshot_date.desc())
                                     .limit(1).scalar_subquery()
                        ),
                        _SP6.track == "main",
                    )
                    .all()
                )
            if _p6_signal_df is not None and not _p6_signal_df.empty:
                _patrol_signal_decay(t_day, _p6_positions, _p6_signal_df, result)
            _patrol_vol_spike_p6(t_day, _p6_positions, result)
        except Exception as _p6pe:
            logger.debug("ensure_daily_batch: P6 position patrols non-fatal: %s", _p6pe)

    # ── P6 Monthly M2: Track B LLM weight overlay ────────────────────────────
    if model is not None and _is_first_trading_day_of_month(t_day):
        try:
            from engine.track_b import run_track_b as _run_track_b
            _tb_sectors = _run_track_b(
                as_of=t_day,
                model=model,
                regime_label=regime_label,
                signal_df=_p6_signal_df,
            )
            if _tb_sectors:
                result.risk_alerts.extend([f"track_b:{s}" for s in _tb_sectors])
                logger.info("ensure_daily_batch: Track B proposed for %d sectors", len(_tb_sectors))
        except Exception as _tbe:
            logger.debug("ensure_daily_batch: Track B non-fatal: %s", _tbe)

    # ── P6 Daily: QuantOnly shadow portfolio NAV ──────────────────────────────
    try:
        _update_quant_only_snapshot(t_day)
    except Exception as _qoe:
        logger.debug("ensure_daily_batch: QuantOnly snapshot non-fatal: %s", _qoe)

    # ── P6 Quarterly: ERA async thread (M4 / first trading day of quarter) ───
    if model is not None and _is_first_trading_day_of_quarter(t_day):
        import threading as _threading
        def _era_thread_fn(_model=model, _day=t_day):
            try:
                from engine.era import run_era as _run_era
                _result = _run_era(as_of=_day, model=_model)
                logger.info("ERA quarterly: %s", _result)
            except Exception as _ee:
                logger.debug("ERA quarterly ERA failed: %s", _ee)
            try:
                from engine.factor_mad import run_quarterly_bh_correction as _bh
                _bh_pass = _bh(as_of=_day)
                if _bh_pass:
                    logger.info("ERA quarterly BH correction: %d pass: %s", len(_bh_pass), _bh_pass)
            except Exception as _bhe:
                logger.debug("ERA quarterly BH correction failed: %s", _bhe)
            try:
                from engine.universe_review import run_universe_review as _ur
                _ur_result = _ur(as_of=_day)
                logger.info("ERA quarterly universe review: proposals=%d", _ur_result.proposals_written)
            except Exception as _ure:
                logger.debug("ERA quarterly universe review failed: %s", _ure)
        _threading.Thread(
            target=_era_thread_fn, daemon=True, name="era_quarterly",
        ).start()
        logger.info("ensure_daily_batch: ERA quarterly audit launched for %s", t_day)

    # ── Post-batch: Persist DailyBriefSnapshot ─────────────────────────────
    try:
        from engine.memory import upsert_daily_brief_snapshot, get_daily_brief_snapshot
        narrative = _generate_narrative(
            t_day, result,
            regime=regime_label, p_risk_on=p_risk_on,
            regime_prev=regime_prev, regime_changed=regime_changed,
        )
        snap_kwargs: dict = dict(
            regime=regime_label,
            regime_prev=regime_prev,
            p_risk_on=p_risk_on,
            regime_changed=regime_changed,
            risk_alerts_json=json.dumps(result.risk_alerts),
            signal_flips_json=json.dumps(result.invalidations),
            n_entries=len(result.entries_triggered),
            n_invalidations=len(result.invalidations),
            n_rebalance=len(result.rebalance_orders),
            n_verified_today=n_verified,
            verify_ran=True,
            narrative=narrative,
            # P4-6: tactical patrol results
            tactical_entries_json=json.dumps(result.tactical_entries),
            tactical_reduces_json=json.dumps(result.tactical_reduces),
            regime_jump_today=result.regime_jump,
        )
        if icir_month_ran:
            snap_kwargs["icir_month"] = icir_month_ran
        upsert_daily_brief_snapshot(t_day, **snap_kwargs)
    except Exception as exc:
        logger.warning("ensure_daily_batch: snapshot save failed: %s", exc)

    # ── Post-batch: LLM 宏观简报（⑥）──────────────────────────────────────────
    # 触发条件：model 可用，且（制度切换 OR 今日尚无 LLM 简报）
    # 失败时静默降级到 rule-based narrative，不影响主流程。
    if model is not None:
        try:
            from engine.memory import get_daily_brief_snapshot, upsert_daily_brief_snapshot
            snap_now = get_daily_brief_snapshot(t_day)
            already_has_llm = snap_now and getattr(snap_now, "macro_brief_llm", None)
            should_generate = regime_changed or not already_has_llm
            if should_generate:
                vix_now = 20.0
                try:
                    from engine.quant import QuantEngine
                    vix_now = QuantEngine.get_realtime_vix()
                except Exception:
                    pass
                llm_brief = _generate_macro_brief_llm(
                    model=model,
                    regime=regime_label,
                    regime_prev=regime_prev,
                    regime_changed=regime_changed,
                    p_risk_on=p_risk_on,
                    vix=vix_now,
                    risk_alerts=result.risk_alerts,
                    entries=result.entries_triggered,
                    rebalance=result.rebalance_orders,
                )
                if llm_brief:
                    upsert_daily_brief_snapshot(t_day, macro_brief_llm=llm_brief)
                    logger.info("ensure_daily_batch: LLM macro brief generated (%d chars)",
                                len(llm_brief))
        except Exception as exc:
            logger.warning("ensure_daily_batch: LLM macro brief step failed: %s", exc)

    return result


# ── P4-7: Tactical patrol backtest validation ─────────────────────────────────

from dataclasses import dataclass as _dc
import math as _math


@_dc
class TacticalValidationResult:
    """Summary of tactical overlay backtest validation."""
    start_date:              datetime.date
    end_date:                datetime.date
    n_months:                int
    # Turnover
    annual_turnover_base:    float   # annualised, 1.0 = 100%
    annual_turnover_tactical: float
    n_fast_flips:            int     # fast-signal flip events
    n_regime_adjustments:    int     # regime-based compress events
    # Sharpe comparison (annualised)
    sharpe_base:             float
    sharpe_tactical:         float
    sharpe_lift:             float
    # Red-line flags
    red_line_sharpe:         bool    # True → Sharpe lift < 0.05
    red_line_turnover:       bool    # True → annual turnover > 200%


def validate_tactical_patrol(
    start_date:     datetime.date | None = None,
    end_date:       datetime.date | None = None,
    slow_lookback:  int = 12,
    slow_skip:      int = 1,
    fast_lookback:  int = 3,
    fast_skip:      int = 1,
    regime_thresh:  float = 0.35,   # p_risk_on below this → regime compress
    compress_ratio: float = 0.50,   # reduce longs to this fraction on flip/compress
    vol_target:     float = 0.10,   # annual vol target for base portfolio scaling
) -> TacticalValidationResult:
    """
    Simulate the tactical overlay (fast-signal flip + regime compress) on
    historical monthly signals. Returns TacticalValidationResult with
    annualised turnover and Sharpe comparison.

    Red lines (per backlog spec):
      - Sharpe lift < 0.05 → insufficient alpha, don't deploy
      - Annual turnover > 2.0 (200%) → tighten thresholds

    Uses:
      - get_signal_dataframe() for slow (12-1M) and fast (3-1M) signals
      - yfinance monthly close for sector ETF returns
      - DailyBriefSnapshot for historical p_risk_on (best-effort)
    """
    import datetime as _dt
    import numpy as _np
    import pandas as _pd
    try:
        import yfinance as _yf
    except ImportError:
        raise RuntimeError("yfinance required for tactical validation")

    from engine.signal import get_signal_dataframe
    from engine.history import get_active_sector_etf

    if end_date is None:
        end_date = _dt.date.today()
    if start_date is None:
        start_date = end_date.replace(year=end_date.year - 3)

    # ── 1. Get sector ETF universe ────────────────────────────────────────────
    _tickers = get_active_sector_etf()
    if not _tickers:
        raise RuntimeError("No active sector ETFs found")

    # ── 2. Fetch monthly price data ───────────────────────────────────────────
    _price_start = start_date - _dt.timedelta(days=400)
    _raw = _yf.download(
        _tickers,
        start=_price_start.isoformat(),
        end=(end_date + _dt.timedelta(days=5)).isoformat(),
        auto_adjust=True,
        progress=False,
    )
    if isinstance(_raw.columns, _pd.MultiIndex):
        _closes = _raw["Close"] if "Close" in _raw.columns.get_level_values(0) else _raw.xs("Close", level=0, axis=1)
    else:
        _closes = _raw[["Close"]] if "Close" in _raw.columns else _raw

    _monthly     = _closes.resample("BME").last().dropna(how="all")
    _monthly_ret = _monthly.pct_change().dropna(how="all")
    _eval_months = _monthly_ret[
        (_monthly_ret.index >= _pd.Timestamp(start_date))
        & (_monthly_ret.index <= _pd.Timestamp(end_date))
    ]
    if len(_eval_months) < 6:
        raise RuntimeError(f"Only {len(_eval_months)} months in range; need at least 6")

    # ── 3. Load historical p_risk_on from DailyBriefSnapshot ─────────────────
    _regime_map: dict[str, float] = {}
    try:
        from engine.memory import SessionFactory, DailyBriefSnapshot as _DBS
        with SessionFactory() as _sess:
            _snaps = _sess.query(_DBS).filter(
                _DBS.as_of_date >= start_date,
                _DBS.as_of_date <= end_date,
                _DBS.p_risk_on.isnot(None),
            ).all()
        for _s in _snaps:
            _regime_map[_s.as_of_date.strftime("%Y-%m")] = float(_s.p_risk_on)
    except Exception:
        pass

    # ── 4. Monthly simulation loop ────────────────────────────────────────────
    _months = list(_eval_months.index)
    _n = len(_months)

    _base_rets:     list[float] = []
    _tactical_rets: list[float] = []
    _base_tv     = 0.0
    _tactical_tv = 0.0
    _n_flips     = 0
    _n_regime    = 0

    _w_base_prev:     dict[str, float] = {}
    _w_tactical_prev: dict[str, float] = {}

    for _i, _ts in enumerate(_months):
        _month_dt = _ts.date()

        try:
            _sig_slow = get_signal_dataframe(
                as_of=_month_dt, lookback_months=slow_lookback,
                skip_months=slow_skip, use_cache=False,
            )
        except Exception:
            continue

        try:
            _sig_fast = get_signal_dataframe(
                as_of=_month_dt, lookback_months=fast_lookback,
                skip_months=fast_skip, use_cache=False,
            )
        except Exception:
            _sig_fast = None

        # Equal-weight long signals, then vol-scale
        try:
            from engine.backtest import _tsmom_weights as _tmw
            _w_slow_raw = _tmw(_sig_slow)
            _ret_sub = _monthly_ret[_monthly_ret.index <= _ts].tail(24)
            if not _w_slow_raw or _ret_sub.empty:
                continue
            _vols = {
                t: float(_ret_sub[t].std(ddof=1) * _math.sqrt(12))
                for t in _w_slow_raw if t in _ret_sub.columns
            }
            _port_vol = float(_np.sqrt(
                sum((w ** 2) * (_vols.get(t, 0.10) ** 2) for t, w in _w_slow_raw.items())
            )) or 0.10
            _scale = min(vol_target / _port_vol, 1.5)
            _w_base: dict[str, float] = {t: w * _scale for t, w in _w_slow_raw.items()}
        except Exception:
            continue

        _w_tactical = dict(_w_base)

        # Fast-flip: slow long but fast short → compress
        if _sig_fast is not None:
            for _t, _w in list(_w_tactical.items()):
                try:
                    _slow_s = float(_sig_slow.loc[_sig_slow.index == _t, "signal"].iloc[0])
                    _fast_s = float(_sig_fast.loc[_sig_fast.index == _t, "signal"].iloc[0])
                except Exception:
                    continue
                if _slow_s > 0 and _fast_s < 0:
                    _w_tactical[_t] = _w * compress_ratio
                    _n_flips += 1

        # Regime compress
        _ym = _month_dt.strftime("%Y-%m")
        _p = _regime_map.get(_ym)
        if _p is not None and _p < regime_thresh:
            for _t in list(_w_tactical.keys()):
                if _w_tactical[_t] > 0:
                    _w_tactical[_t] *= compress_ratio
            _n_regime += 1

        # Next-month return
        if _i + 1 >= _n:
            break
        _next_rets = _eval_months.loc[_months[_i + 1]]

        def _port_ret(wts: dict, rets: "_pd.Series") -> float:
            return sum(
                wts[t] * float(rets[t])
                for t in wts
                if t in rets.index and not _np.isnan(float(rets[t]))
            )

        _base_rets.append(_port_ret(_w_base, _next_rets))
        _tactical_rets.append(_port_ret(_w_tactical, _next_rets))

        def _tv(w_new: dict, w_prev: dict) -> float:
            _all = set(w_new) | set(w_prev)
            return sum(abs(w_new.get(t, 0.0) - w_prev.get(t, 0.0)) for t in _all) / 2.0

        _base_tv     += _tv(_w_base, _w_base_prev)
        _tactical_tv += _tv(_w_tactical, _w_tactical_prev)
        _w_base_prev     = dict(_w_base)
        _w_tactical_prev = dict(_w_tactical)

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    _n_obs = len(_base_rets)
    if _n_obs < 3:
        raise RuntimeError("Too few return observations to compute metrics")

    _ann_sq = _math.sqrt(12)

    def _sharpe(rets: list) -> float:
        _a = _np.array(rets)
        _sd = float(_a.std(ddof=1))
        return round(float(_a.mean()) / _sd * _ann_sq, 4) if _sd > 0 else 0.0

    _sb   = _sharpe(_base_rets)
    _st   = _sharpe(_tactical_rets)
    _lift = round(_st - _sb, 4)
    _ann_factor = 12.0 / _n_obs
    _tv_base     = round(_base_tv * _ann_factor, 4)
    _tv_tactical = round(_tactical_tv * _ann_factor, 4)

    return TacticalValidationResult(
        start_date=start_date,
        end_date=end_date,
        n_months=_n_obs,
        annual_turnover_base=_tv_base,
        annual_turnover_tactical=_tv_tactical,
        n_fast_flips=_n_flips,
        n_regime_adjustments=_n_regime,
        sharpe_base=_sb,
        sharpe_tactical=_st,
        sharpe_lift=_lift,
        red_line_sharpe=_lift < 0.05,
        red_line_turnover=_tv_tactical > 2.0,
    )
