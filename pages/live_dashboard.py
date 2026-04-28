"""
Macro Alpha Pro — Trading Terminal
证券终端：实时报价 · 走势图 · Paper Trading · 组合管理
替代原 portfolio_monitor.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf

import ui.theme as theme
from engine.memory import init_db, get_system_config, set_system_config, get_daily_brief_snapshot

init_db()
theme.init_theme()

today = datetime.date.today()
_is_dark_ld = theme.is_dark()
# Theme-aware muted colors — rgba(255,255,255,X) only works on dark backgrounds
_mu45 = "rgba(255,255,255,0.45)" if _is_dark_ld else "rgba(0,0,0,0.55)"
_mu40 = "rgba(255,255,255,0.40)" if _is_dark_ld else "rgba(0,0,0,0.50)"
_mu35 = "rgba(255,255,255,0.35)" if _is_dark_ld else "rgba(0,0,0,0.45)"
_mu50 = "rgba(255,255,255,0.50)" if _is_dark_ld else "rgba(0,0,0,0.60)"
_txt  = "#f0f6fc"                 if _is_dark_ld else "#0f172a"
_slbl = "#94a3b8"                 if _is_dark_ld else "#4b5563"

# ── Drill-down context: show hard stops / regime compress / tactical from Daily Brief ─
try:
    _snap_ld = get_daily_brief_snapshot(today)
    _alerts_ld: list[str] = []
    if _snap_ld and _snap_ld.risk_alerts_json:
        _alerts_ld = json.loads(_snap_ld.risk_alerts_json)
    # Partition into auto-executed (Layer 2) vs pending manual approval
    _auto_exec_ld = [a for a in _alerts_ld if "auto_executed" in a]
    _pending_ld   = [a for a in _alerts_ld if "auto_executed" not in a]

    _stops_auto   = [a.split(":")[0] for a in _auto_exec_ld
                     if "hard_stop" in a or "drawdown_stop" in a]
    _stops_pend   = [a.split(":")[0] for a in _pending_ld
                     if "hard_stop" in a or "drawdown_stop" in a]
    _compress_ld  = [a.split(":")[0] for a in _alerts_ld if "regime_compress" in a]

    # P4-6: tactical events
    _tact_entries_ld: list[str] = []
    _tact_reduces_ld: list[str] = []
    _regime_jump_ld  = False
    if _snap_ld:
        if getattr(_snap_ld, "tactical_entries_json", None):
            _tact_entries_ld = json.loads(_snap_ld.tactical_entries_json)
        if getattr(_snap_ld, "tactical_reduces_json", None):
            _tact_reduces_ld = json.loads(_snap_ld.tactical_reduces_json)
        _regime_jump_ld = bool(getattr(_snap_ld, "regime_jump_today", False))

    _has_ld = bool(
        _stops_auto or _stops_pend or _compress_ld
        or _tact_entries_ld or _tact_reduces_ld or _regime_jump_ld
    )
    if _has_ld:
        _items_ld = []
        # Regime jump — highest priority, always first
        if _regime_jump_ld:
            _items_ld.append(
                f'<span style="color:#ef4444;font-weight:700;">⚠️ 制度跃变</span>'
                f'<span style="color:{_mu45};"> — Layer 2 已自动压缩多头敞口</span>'
            )
        if _stops_auto:
            _items_ld.append(
                f'<span style="color:#22c55e;font-weight:700;">'
                f'✅ 自动止损已执行：{", ".join(_stops_auto)}</span>'
                f'<span style="color:{_mu45};"> — Layer 2 自动处理，无需操作</span>'
            )
        if _stops_pend:
            _items_ld.append(
                f'<span style="color:#ef4444;font-weight:700;">'
                f'🛑 止损待审批：{", ".join(_stops_pend)}</span>'
                f'<span style="color:{_mu45};"> — 前往 Daily Brief 处理</span>'
            )
        if _compress_ld:
            _items_ld.append(
                f'<span style="color:#f59e0b;">制度压缩：{", ".join(_compress_ld)}</span>'
            )
        if _tact_entries_ld:
            _items_ld.append(
                f'<span style="color:#22c55e;">📍 战术入场触发：{", ".join(_tact_entries_ld)}</span>'
                f'<span style="color:{_mu45};"> — 前往 Daily Brief 查看审批项</span>'
            )
        if _tact_reduces_ld:
            _items_ld.append(
                f'<span style="color:#f59e0b;">📉 战术减仓：{", ".join(_tact_reduces_ld)}</span>'
            )
        _urgent_pend = bool(_stops_pend or _regime_jump_ld)
        _urgent_action = bool(_tact_entries_ld and not _regime_jump_ld)
        if _urgent_pend:
            _bg_ld, _brd_ld = "rgba(239,68,68,0.07)", "rgba(239,68,68,0.4)"
        elif _urgent_action:
            _bg_ld, _brd_ld = "rgba(34,197,94,0.05)", "rgba(34,197,94,0.3)"
        else:
            _bg_ld, _brd_ld = "rgba(245,158,11,0.05)", "rgba(245,158,11,0.3)"
        st.markdown(
            f'<div style="background:{_bg_ld};border:1.5px solid {_brd_ld};'
            f'border-radius:5px;padding:0.6rem 1.1rem;margin-bottom:0.9rem;font-size:0.83rem;">'
            f'<span style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;'
            f'color:{_mu40};margin-right:0.6rem;">Daily Brief →</span>'
            + "  ·  ".join(_items_ld) + '</div>',
            unsafe_allow_html=True,
        )
except Exception:
    pass

# ── Regime / Position Limits Banner ──────────────────────────────────────────
try:
    from engine.regime import get_regime_on as _grld
    _reg_ld = _grld(as_of=today, train_end=today)
    _reg_name = getattr(_reg_ld, "regime", "unknown").upper()
    _reg_p    = float(getattr(_reg_ld, "p_risk_on", 0.0) or 0.0)
    _reg_colors = {"RISK-ON": "#22c55e", "TRANSITION": "#f59e0b", "RISK-OFF": "#ef4444"}
    _reg_col    = _reg_colors.get(_reg_name, "#8b949e")
    _lim_map    = {"RISK-ON": "10多 / 6空", "TRANSITION": "7多 / 7空", "RISK-OFF": "5多 / 8空"}
    _lim_str    = _lim_map.get(_reg_name, "—")
    _cov_str    = "协方差收缩 −30%" if _reg_name == "RISK-OFF" else ""
    _banner_parts = [
        f'<span style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;'
        f'color:{_mu40};">制度</span>  '
        f'<span style="font-family:\'Courier New\',monospace;font-weight:700;font-size:0.9rem;'
        f'color:{_reg_col};">{_reg_name}</span>  '
        f'<span style="color:{_mu35};">P={_reg_p:.0%}</span>',

        f'<span style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;'
        f'color:{_mu40};">仓位上限</span>  '
        f'<span style="font-family:\'Courier New\',monospace;font-weight:700;font-size:0.9rem;'
        f'color:{_reg_col};">{_lim_str}</span>',
    ]
    if _cov_str:
        _banner_parts.append(
            f'<span style="font-size:0.8rem;color:#f59e0b;font-weight:600;">{_cov_str}</span>'
        )
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:2rem;'
        f'background:rgba(0,0,0,0.18);border:1px solid {_reg_col}44;'
        f'border-left:4px solid {_reg_col};border-radius:5px;'
        f'padding:0.5rem 1.2rem;margin-bottom:0.8rem;">'
        + "  &nbsp;·&nbsp;  ".join(_banner_parts)
        + '</div>',
        unsafe_allow_html=True,
    )
except Exception:
    pass

# ── Custom CSS: terminal feel ──────────────────────────────────────────────────
st.markdown(f"""
<style>
.quote-up   {{ color: #22c55e; font-weight: 600; }}
.quote-down {{ color: #ef4444; font-weight: 600; }}
.quote-flat {{ color: #94a3b8; }}
.ticker-label {{ font-size: 1.1rem; font-weight: 700; color: {_txt}; }}
.price-big  {{ font-size: 1.6rem; font-weight: 700; color: {_txt}; }}
.stat-label {{ font-size: 0.72rem; color: {_slbl}; }}
.stat-val   {{ font-size: 0.9rem; font-weight: 600; color: {_txt}; }}
div[data-testid="stMetricValue"] {{ font-size: 1.1rem !important; }}
</style>
""", unsafe_allow_html=True)

# ── NAV config (persistent via SystemConfig) ───────────────────────────────────
_nav_str = get_system_config("paper_trading_nav", "1000000")
try:
    PORTFOLIO_NAV = float(_nav_str)
except ValueError:
    PORTFOLIO_NAV = 1_000_000.0

# ── Load portfolio data ────────────────────────────────────────────────────────
from engine.portfolio_tracker import (
    get_current_positions,
    execute_rebalance,
    record_monthly_return,
    load_all_monthly_returns,
    load_trade_history,
    apply_tactical_weight_update,
)

@st.cache_data(ttl=60, show_spinner=False)
def _cached_positions():
    return get_current_positions()

@st.cache_data(ttl=600, show_spinner=False)
def _cached_monthly():
    return load_all_monthly_returns()

@st.cache_data(ttl=600, show_spinner=False)
def _cached_trades(limit: int = 200):
    return load_trade_history(limit=limit)

@st.cache_data(ttl=300, show_spinner=False)
def _cached_sig_flip(as_of_str: str) -> pd.DataFrame:
    try:
        from engine.signal import get_signal_dataframe as _gsd
        return _gsd(datetime.date.fromisoformat(as_of_str), 12, 1)
    except Exception:
        return pd.DataFrame()

positions  = _cached_positions()
monthly_df = _cached_monthly()

# ── Fetch all live quotes in one batch ────────────────────────────────────────
@st.cache_data(ttl=60)
def _fetch_quotes(tickers: tuple[str, ...]) -> dict:
    """
    Returns dict: ticker → {last, prev_close, chg_pct, day_high, day_low,
                             year_high, year_low, year_chg, bid, ask, volume}
    """
    result: dict = {}
    if not tickers:
        return result
    for t in tickers:
        try:
            fi = yf.Ticker(t).fast_info
            last       = float(fi.last_price or 0)
            prev       = float(fi.regular_market_previous_close or fi.previous_close or last)
            chg_pct    = (last / prev - 1.0) if prev > 0 else 0.0
            day_h   = float(fi.day_high or last)
            day_l   = float(fi.day_low  or last)
            intra   = max(day_h - day_l, last * 0.0001)
            spread  = max(intra * 0.10, last * 0.0002)   # ATR-TC: 10% of daily range, min 0.02%
            result[t]  = {
                "last":      last,
                "prev":      prev,
                "chg_pct":   chg_pct,
                "day_high":  float(fi.day_high or last),
                "day_low":   float(fi.day_low  or last),
                "year_high": float(fi.year_high or last),
                "year_low":  float(fi.year_low  or last),
                "year_chg":  float(fi.year_change or 0),
                "bid":       round(last - spread, 2),
                "ask":       round(last + spread, 2),
                "volume":    int(fi.last_volume or 0),
            }
        except Exception:
            result[t] = {}
    return result


@st.cache_data(ttl=300)
def _fetch_chart(ticker: str, period: str = "6mo") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, auto_adjust=True,
                         progress=False, multi_level_index=False)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def _compute_atr(ticker: str, n: int = 21) -> float:
    """ATR(n) estimate from cached 3-month daily High-Low range."""
    chart = _fetch_chart(ticker, period="3mo")
    if chart.empty:
        return 0.0
    hi_col = next((c for c in chart.columns if str(c).lower() == "high"), None)
    lo_col = next((c for c in chart.columns if str(c).lower() == "low"), None)
    if not hi_col or not lo_col:
        return 0.0
    hl = (chart[hi_col] - chart[lo_col]).dropna().tail(n)
    return float(hl.mean()) if len(hl) > 0 else 0.0


@st.cache_data(ttl=300)
def _load_signal_ctx_ld(tickers: tuple) -> dict:
    """Latest SignalRecord per ticker: composite_score, decay_pct."""
    ctx: dict = {}
    try:
        from engine.memory import SignalRecord, SessionFactory
        with SessionFactory() as _ss:
            for t in tickers:
                row = (
                    _ss.query(SignalRecord)
                       .filter(SignalRecord.ticker == t)
                       .order_by(SignalRecord.date.desc())
                       .first()
                )
                if row:
                    ctx[t] = {
                        "composite": (float(row.composite_score)
                                      if row.composite_score is not None else None),
                        "decay_pct": float(getattr(row, "decay_pct", 0) or 0),
                    }
    except Exception:
        pass
    return ctx


@st.cache_data(ttl=300)
def _load_trackb_ctx_ld(sectors: tuple) -> dict:
    """Latest Track B llm_delta per sector from AlphaMemory."""
    ctx: dict = {}
    try:
        from engine.memory import AlphaMemory, SessionFactory
        with SessionFactory() as _ss:
            for sec in sectors:
                row = (
                    _ss.query(AlphaMemory.llm_delta)
                       .filter(AlphaMemory.source == "track_b",
                               AlphaMemory.sector == sec)
                       .order_by(AlphaMemory.decision_date.desc())
                       .first()
                )
                if row and row[0] is not None:
                    ctx[sec] = float(row[0])
    except Exception:
        pass
    return ctx


# ── Regime ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def _get_regime():
    try:
        from engine.regime import get_regime_on
        return get_regime_on(as_of=today, train_end=today)
    except Exception:
        return None


regime = _get_regime()

# ── Compute portfolio-level P&L ────────────────────────────────────────────────
tickers_in_pos = tuple(positions["ticker"].dropna().unique()) if not positions.empty else ()
sectors_in_pos = tuple(positions.index.tolist()) if not positions.empty else ()
quotes         = _fetch_quotes(tickers_in_pos)
signal_ctx_ld  = _load_signal_ctx_ld(tickers_in_pos)
trackb_ctx_ld  = _load_trackb_ctx_ld(sectors_in_pos)

def _pnl(pos_row: pd.Series, quote: dict) -> tuple[float, float]:
    """Returns (unrealized_pnl_abs, unrealized_pnl_pct) for one position."""
    last      = quote.get("last", 0)
    entry     = float(pos_row.get("entry_price") or 0)
    shares    = float(pos_row.get("shares_held") or 0)
    cost      = float(pos_row.get("cost_basis") or 0)
    if shares > 0 and entry > 0:
        market_val = shares * last
        pnl_abs    = market_val - cost
        pnl_pct    = pnl_abs / cost if cost > 0 else 0.0
        return pnl_abs, pnl_pct
    return 0.0, 0.0

total_market_value = 0.0
total_cost         = 0.0
if not positions.empty:
    for sector, row in positions.iterrows():
        t  = row.get("ticker", "")
        q  = quotes.get(t, {})
        last = q.get("last", 0)
        shares = float(row.get("shares_held") or 0)
        weight = float(row.get("actual_weight") or 0)
        if shares > 0 and last > 0:
            total_market_value += shares * last
            total_cost         += float(row.get("cost_basis") or 0)
        elif last > 0 and weight != 0:
            total_market_value += abs(weight) * PORTFOLIO_NAV
            if row.get("entry_price"):
                total_cost += abs(weight) * PORTFOLIO_NAV / (last / float(row["entry_price"]))

portfolio_pnl     = total_market_value - total_cost if total_cost > 0 else 0.0
portfolio_pnl_pct = portfolio_pnl / total_cost if total_cost > 0 else 0.0

# True NAV = initial capital + unrealized P&L (cash + market value, zero-external-cashflow)
current_nav = PORTFOLIO_NAV + portfolio_pnl if total_cost > 0 else PORTFOLIO_NAV

# Day P&L approximation (weight × daily return × current NAV)
day_pnl = sum(
    (q.get("chg_pct", 0) * float(row.get("actual_weight", 0)) * current_nav)
    for sector, row in (positions.iterrows() if not positions.empty else iter([]))
    for t in [row.get("ticker", "")]
    for q in [quotes.get(t, {})]
)

# ════════════════════════════════════════════════════════════════════════════════
#  HEADER — Portfolio NAV strip
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("### 🖥️  Macro Alpha Pro  ·  Trading Terminal")

h1, h2, h3, h4, h5, h6, h_set = st.columns([2, 2, 2, 2, 2, 2, 1])
h1.metric("模拟总资产", f"¥{current_nav:,.0f}",
          delta=f"{portfolio_pnl:+,.0f}" if portfolio_pnl else None)
h2.metric("持仓市值",   f"¥{total_market_value:,.0f}",
          delta=f"{portfolio_pnl:+,.0f}" if portfolio_pnl else None)
h3.metric("浮动盈亏",
          f"¥{portfolio_pnl:+,.0f}",
          delta=f"{portfolio_pnl_pct:+.2%}" if total_cost > 0 else None)
h4.metric("今日盈亏",   f"¥{day_pnl:+,.0f}")

if regime:
    icons = {"risk-on": "🟢", "risk-off": "🔴", "transition": "🟡"}
    h5.metric("宏观制度", f"{icons.get(regime.regime,'⚪')} {regime.regime}",
              delta=f"P(on)={regime.p_risk_on:.0%}")
h6.metric("持仓数量",  len(positions) if not positions.empty else 0)

with h_set:
    with st.popover("⚙️ 设置"):
        nav_input = st.number_input(
            "模拟总金额 (¥)", min_value=10000, max_value=100_000_000,
            value=int(PORTFOLIO_NAV), step=100000, key="nav_input",
        )
        if st.button("保存", key="save_nav"):
            set_system_config("paper_trading_nav", str(float(nav_input)))
            st.success("已保存")
            st.rerun()

        st.divider()
        st.caption("危险操作")
        if st.button("🔄 重置持仓（以当日信号重新初始化）", key="btn_reset_positions"):
            st.session_state["_reset_confirm"] = True

if st.session_state.get("_reset_confirm"):
    st.warning(
        "确认将以今日信号覆盖全部持仓记录（应用集中度约束，持仓数量将大幅减少）？",
        icon="⚠️",
    )
    _rc1, _rc2, _ = st.columns([1, 1, 6])
    if _rc1.button("确认重置", key="btn_reset_confirm", type="primary"):
        with st.spinner("重新计算并写入持仓 …"):
            try:
                execute_rebalance(today, dry_run=False, nav=PORTFOLIO_NAV)
                st.session_state.pop("_reset_confirm", None)
                st.success("持仓已重置，正在刷新 …")
                st.rerun()
            except Exception as _e:
                st.error(f"重置失败：{_e}")
    if _rc2.button("取消", key="btn_reset_cancel"):
        st.session_state.pop("_reset_confirm", None)
        st.rerun()

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT: Left = position table  |  Right = chart
# ════════════════════════════════════════════════════════════════════════════════
left_col, right_col = st.columns([5, 6], gap="medium")

# ── LEFT: Position book ────────────────────────────────────────────────────────
with left_col:
    st.markdown("#### 持仓账簿")

    if positions.empty:
        st.markdown(
            f'<div style="font-size:0.85rem;color:{_mu50};">'
            '暂无持仓记录。以当前信号初始化模拟组合，或前往 Daily Brief 执行月度再平衡。'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

        # ── Cold-start: preview + one-click initialise ────────────────────────
        _init_key = "cold_start_preview"
        if st.button("📊 预览当前信号目标组合", key="btn_preview_init"):
            with st.spinner("计算目标权重 …"):
                try:
                    _prev = execute_rebalance(today, dry_run=True, nav=PORTFOLIO_NAV)
                    st.session_state[_init_key] = _prev
                except Exception as _e:
                    st.error(f"预览失败：{_e}")

        _preview = st.session_state.get(_init_key)
        if _preview and _preview.get("new_positions"):
            _prev_rows = []
            for _p in _preview["new_positions"]:
                _prev_rows.append({
                    "板块":   _p.get("sector", ""),
                    "ETF":    _p.get("ticker", ""),
                    "目标权重": f"{_p.get('target_weight', 0):.1%}",
                    "信号":   "多" if (_p.get("signal_tsmom") or 0) > 0 else "空",
                })
            if _prev_rows:
                st.dataframe(
                    pd.DataFrame(_prev_rows),
                    use_container_width=True,
                    hide_index=True,
                    height=min(60 + 35 * len(_prev_rows), 320),
                )
                st.caption(
                    f"制度：{_preview.get('regime', '—')}  ·  "
                    f"多头 {_preview.get('n_long', 0)} 个  ·  "
                    f"预估换手成本 {_preview.get('total_cost_bps', 0):.1f} bps"
                )
            if st.button("✅ 确认初始化组合（以今日为基准日）", key="btn_confirm_init",
                         type="primary"):
                with st.spinner("写入持仓 …"):
                    try:
                        execute_rebalance(today, dry_run=False, nav=PORTFOLIO_NAV)
                        del st.session_state[_init_key]
                        st.success("持仓已初始化，正在刷新 …")
                        st.rerun()
                    except Exception as _e:
                        st.error(f"初始化失败：{_e}")
        selected_ticker = None
    else:
        # Build rich position rows — skip zero-weight (closed) positions
        pos_rows = []
        closed_rows = []
        for sector, row in positions.iterrows():
            t    = row.get("ticker", "")
            q    = quotes.get(t, {})
            last = q.get("last", 0)
            prev = q.get("prev", 0)
            chg  = q.get("chg_pct", 0)
            act_w = float(row.get("actual_weight") or 0)
            pnl_abs, pnl_pct = _pnl(row, q)
            shares = float(row.get("shares_held") or 0)
            pos_val = shares * last if shares > 0 and last > 0 else abs(act_w) * current_nav
            # Live market-value weight; preserve sign from act_w (shorts remain negative)
            _sign = np.sign(act_w) if act_w != 0 else 1.0
            live_w = (_sign * shares * last / current_nav
                      if shares > 0 and last > 0 and current_nav > 0 else act_w)
            direction = row.get("direction") or (
                "超配" if act_w > 0.01 else ("低配" if act_w < -0.01 else "标配")
            )
            # ── 建仓日期 / 持有天数 ────────────────────────────────────────────
            snap = row.get("snapshot_date")
            if snap is not None:
                if isinstance(snap, str):
                    try:
                        snap = datetime.date.fromisoformat(snap)
                    except ValueError:
                        snap = None
                elif hasattr(snap, "date"):
                    snap = snap.date()
            entry_date_str = snap.strftime("%Y-%m-%d") if snap else "—"
            days_held      = (today - snap).days if snap else None

            # ── 距止损% = (last - stop) / last, stop = trailing_high - 2×ATR ──
            t_high = float(row.get("trailing_high") or 0)
            if t_high > 0 and last > 0:
                atr = _compute_atr(t) if t else 0.0
                stop = t_high - 2.0 * atr
                dist_stop = (last - stop) / last if last > 0 else None
            else:
                dist_stop = None

            if dist_stop is not None:
                _ds_label = ("⚠ " if dist_stop < 0.05 else "") + f"{dist_stop:.1%}"
            else:
                _ds_label = "—"

            # ── P6 signal context ──────────────────────────────────────────
            _sctx_ld      = signal_ctx_ld.get(t, {})
            _composite_ld = _sctx_ld.get("composite")
            _decay_ld     = float(_sctx_ld.get("decay_pct") or 0)
            _tb_delta_ld  = trackb_ctx_ld.get(str(sector))

            _row_data = {
                "板块":        sector,
                "ETF":         t,
                "方向":        direction,
                "持仓权重":    f"{live_w:+.1%}",
                "目标权重":    f"{act_w:+.1%}",
                "综合评分":    f"{_composite_ld:.0f}" if _composite_ld is not None else "—",
                "TB Δ":        f"{_tb_delta_ld:+.1%}" if _tb_delta_ld is not None else "—",
                "最新价":      f"{last:.2f}" if last else "—",
                "今日涨跌":    f"{chg:+.2%}" if q else "—",
                "盈亏%":       f"{pnl_pct:+.2%}" if pnl_pct != 0 else "—",
                "距止损":      _ds_label,
                "衰减%":       f"{_decay_ld:.0%}" if _decay_ld > 0.05 else "—",
                "建仓日期":    entry_date_str,
                "持有天数":    str(days_held) if days_held is not None else "—",
                "持仓市值":    f"¥{pos_val:,.0f}" if pos_val else "—",
                "_ticker":     t,
                "_chg":        chg,
                "_pnl":        pnl_pct,
                "_act_w":      act_w,
            }
            if abs(act_w) < 1e-6:
                closed_rows.append(_row_data)
            else:
                pos_rows.append(_row_data)

        df_pos = pd.DataFrame(pos_rows) if pos_rows else pd.DataFrame()

        if df_pos.empty:
            st.info("所有持仓已平仓或权重为零。请运行月度再平衡后刷新。")
            selected_ticker = None
        else:
            # ── Regime position-limit gauge ───────────────────────────────
            _n_long_ld  = sum(1 for r in pos_rows if r.get("_act_w", 0) > 0.01)
            _n_short_ld = sum(1 for r in pos_rows if r.get("_act_w", 0) < -0.01)
            _rn_ld = (getattr(regime, "regime", "transition") or "transition").upper()
            _ll = {"RISK-ON": 10, "TRANSITION": 7, "RISK-OFF": 5}.get(_rn_ld, 7)
            _sl = {"RISK-ON": 6,  "TRANSITION": 7, "RISK-OFF": 8}.get(_rn_ld, 7)

            def _bar(n, lim):
                pct = n / lim if lim > 0 else 0
                col = "#22c55e" if pct < 0.8 else ("#f59e0b" if pct < 1.0 else "#ef4444")
                filled = min(int(pct * 10), 10)
                return col, "█" * filled + "░" * (10 - filled)

            _lc, _lb = _bar(_n_long_ld, _ll)
            _sc, _sb = _bar(_n_short_ld, _sl)
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:1.8rem;'
                f'font-family:\'Courier New\',monospace;font-size:0.80rem;'
                f'padding:0.25rem 0 0.45rem;">'
                f'<span style="color:{_mu40};font-size:0.68rem;'
                f'text-transform:uppercase;letter-spacing:0.08em;">多头</span>'
                f'<span style="color:{_lc};letter-spacing:0.02em;">{_lb}</span>'
                f'<b style="color:{_lc};">{_n_long_ld}/{_ll}</b>'
                f'&emsp;'
                f'<span style="color:{_mu40};font-size:0.68rem;'
                f'text-transform:uppercase;letter-spacing:0.08em;">空头</span>'
                f'<span style="color:{_sc};letter-spacing:0.02em;">{_sb}</span>'
                f'<b style="color:{_sc};">{_n_short_ld}/{_sl}</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Interactive table — Streamlit ≥1.35 supports on_select
            display_cols = ["板块", "ETF", "方向", "持仓权重", "目标权重", "综合评分", "TB Δ",
                            "最新价", "今日涨跌", "盈亏%", "距止损", "衰减%",
                            "建仓日期", "持有天数"]
            try:
                event = st.dataframe(
                    df_pos[display_cols],
                    use_container_width=True,
                    hide_index=True,
                    height=420,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="pos_table",
                )
                sel_rows = event.selection.get("rows", []) if hasattr(event, "selection") else []
                if sel_rows:
                    selected_ticker = df_pos.iloc[sel_rows[0]]["_ticker"]
                    st.session_state["selected_ticker"] = selected_ticker
                else:
                    selected_ticker = st.session_state.get("selected_ticker",
                                      df_pos["_ticker"].iloc[0] if len(df_pos) > 0 else None)
            except Exception:
                st.dataframe(df_pos[display_cols], use_container_width=True, hide_index=True)
                selected_ticker = st.session_state.get("selected_ticker",
                                  df_pos["_ticker"].iloc[0] if len(df_pos) > 0 else None)

            # Signal flip alert
            try:
                sig_live = _cached_sig_flip(str(today))
                flips = []
                for sector, row in positions.iterrows():
                    if sector in sig_live.index:
                        cur  = int(sig_live.loc[sector, "tsmom"])
                        prev_tsmom = int(row.get("signal_tsmom") or 0)
                        if cur != prev_tsmom:
                            arrow = "↑" if cur > prev_tsmom else "↓"
                            flips.append(f"{sector}{arrow}")
                if flips:
                    st.warning(f"⚡ 信号翻转：{' | '.join(flips)}")
            except Exception:
                pass

            # Closed positions (actual_weight == 0) — collapsed by default
            if closed_rows:
                with st.expander(f"已平仓 / 权重归零持仓  ({len(closed_rows)} 个)", expanded=False):
                    closed_display = ["板块", "ETF", "持仓权重", "最新价", "今日涨跌"]
                    df_closed = pd.DataFrame(closed_rows)
                    st.dataframe(
                        df_closed[[c for c in closed_display if c in df_closed.columns]],
                        use_container_width=True,
                        hide_index=True,
                    )

# ── RIGHT: Quote card + Chart ──────────────────────────────────────────────────
with right_col:
    sel = selected_ticker if not positions.empty else None

    if sel:
        q = quotes.get(sel, {})
        last     = q.get("last", 0)
        chg      = q.get("chg_pct", 0)
        chg_abs  = last - q.get("prev", last)
        chg_cls  = "quote-up" if chg > 0 else ("quote-down" if chg < 0 else "quote-flat")
        chg_sign = "▲" if chg > 0 else ("▼" if chg < 0 else "—")

        # Quote header
        sector_name = ""
        if not positions.empty and sel in positions["ticker"].values:
            idx = positions.index[positions["ticker"] == sel]
            if len(idx) > 0:
                sector_name = str(idx[0])

        q1, q2, q3, q4, q5 = st.columns(5)
        q1.markdown(
            f'<div class="ticker-label">{sel}</div>'
            f'<div class="stat-label">{sector_name}</div>',
            unsafe_allow_html=True,
        )
        q2.markdown(
            f'<div class="stat-label">最新价</div>'
            f'<div class="price-big {chg_cls}">{last:.2f}</div>',
            unsafe_allow_html=True,
        )
        q3.markdown(
            f'<div class="stat-label">涨跌</div>'
            f'<div class="stat-val {chg_cls}">{chg_sign} {chg_abs:+.2f} ({chg:+.2%})</div>',
            unsafe_allow_html=True,
        )
        q4.markdown(
            f'<div class="stat-label">买入参考 / 卖出参考</div>'
            f'<div class="stat-val">'
            f'<span class="quote-up">{q.get("ask",0):.2f}</span>'
            f' / '
            f'<span class="quote-down">{q.get("bid",0):.2f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        q5.markdown(
            f'<div class="stat-label">今日区间</div>'
            f'<div class="stat-val">{q.get("day_low",0):.2f} – {q.get("day_high",0):.2f}</div>',
            unsafe_allow_html=True,
        )

        # Secondary stats row
        s1, s2, s3, s4 = st.columns(4)
        s1.markdown(
            f'<div class="stat-label">52W 高/低</div>'
            f'<div class="stat-val">{q.get("year_high",0):.2f} / {q.get("year_low",0):.2f}</div>',
            unsafe_allow_html=True,
        )
        s2.markdown(
            f'<div class="stat-label">年涨跌幅</div>'
            f'<div class="stat-val {chg_cls if q.get("year_chg",0)!=0 else ""}">'
            f'{q.get("year_chg",0):+.2%}</div>',
            unsafe_allow_html=True,
        )
        s3.markdown(
            f'<div class="stat-label">成交量</div>'
            f'<div class="stat-val">{q.get("volume",0):,}</div>',
            unsafe_allow_html=True,
        )
        # Entry price & unrealized from position
        if not positions.empty and sel in positions["ticker"].values:
            _r   = positions[positions["ticker"] == sel].iloc[0]
            _ep  = _r.get("entry_price")
            _pct = (_r.get("actual_weight") or 0)
            s4.markdown(
                f'<div class="stat-label">建仓价 / 仓位</div>'
                f'<div class="stat-val">{float(_ep):.2f} / {float(_pct):+.1%}</div>'
                if _ep else '<div class="stat-val">—</div>',
                unsafe_allow_html=True,
            )

        # Chart controls
        period_map = {"1个月": "1mo", "3个月": "3mo", "6个月": "6mo",
                      "1年": "1y", "2年": "2y"}
        chart_type_opt = st.radio("图表类型", ["蜡烛图", "折线图"],
                                  horizontal=True, key="chart_type")
        period_sel = st.select_slider("周期",
                                      options=list(period_map.keys()),
                                      value="6个月", key="chart_period")
        period_yf = period_map[period_sel]

        # Draw chart
        chart_df = _fetch_chart(sel, period_yf)
        if not chart_df.empty:
            colors_up   = "#22c55e"
            colors_down = "#ef4444"

            fig = go.Figure()

            if chart_type_opt == "蜡烛图":
                fig.add_trace(go.Candlestick(
                    x=chart_df.index,
                    open=chart_df["Open"],
                    high=chart_df["High"],
                    low=chart_df["Low"],
                    close=chart_df["Close"],
                    increasing_line_color=colors_up,
                    decreasing_line_color=colors_down,
                    name=sel,
                ))
            else:
                close = chart_df["Close"]
                fig.add_trace(go.Scatter(
                    x=chart_df.index,
                    y=close,
                    mode="lines",
                    line=dict(color="#3b82f6", width=2),
                    name=sel,
                    fill="tozeroy",
                    fillcolor="rgba(59,130,246,0.08)",
                ))

            # Entry price horizontal line
            if not positions.empty and sel in positions["ticker"].values:
                _ep = positions[positions["ticker"] == sel].iloc[0].get("entry_price")
                if _ep:
                    fig.add_hline(
                        y=float(_ep), line_dash="dash",
                        line_color="#f59e0b", line_width=1,
                        annotation_text=f"建仓 {float(_ep):.2f}",
                        annotation_position="right",
                    )

            fig.update_layout(
                height=360,
                xaxis_rangeslider_visible=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(showgrid=True, gridcolor="rgba(100,116,139,0.2)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(100,116,139,0.2)"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"无法获取 {sel} 的历史价格数据。")
    else:
        st.info("点击左侧持仓行查看走势图。")

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
#  BOTTOM SECTION — tabs for analytics + operations
# ════════════════════════════════════════════════════════════════════════════════
t_rebal, t_nav, t_attr, t_trade = st.tabs([
    "⚖️ 再平衡", "📈 净值曲线", "📊 收益归因", "📜 交易记录"
])

# ── Tab: Rebalance ─────────────────────────────────────────────────────────────
with t_rebal:
    # Auto pre-flight dry-run for today (cached 3 min)
    @st.cache_data(ttl=180, show_spinner=False)
    def _auto_preflight(date_str: str) -> dict:
        try:
            return execute_rebalance(datetime.date.fromisoformat(date_str), dry_run=True)
        except Exception as e:
            return {"error": str(e)}

    _pf = _auto_preflight(str(today))
    _pf_err = _pf.get("error") if _pf else "no result"

    if _pf_err:
        st.warning(f"预检计算失败：{_pf_err}")
    else:
        # ── Pre-flight summary strip ──────────────────────────────────────
        _pf_regime = _pf.get("regime", "—")
        _pf_turn   = _pf.get("turnover", 0)
        _pf_cost   = _pf.get("total_cost_bps", 0)
        _pf_nl     = _pf.get("n_long", 0)
        _pf_ns     = _pf.get("n_short", 0)
        _pf_trades = _pf.get("trades", [])
        _pf_warns  = _pf.get("warnings", [])
        _pf_no_op  = not bool(_pf_trades)

        _pc1, _pc2, _pc3, _pc4, _pc5 = st.columns(5)
        _pc1.metric("制度",      _pf_regime)
        _pc2.metric("换手率",    f"{_pf_turn:.1%}")
        _pc3.metric("预估成本",  f"{_pf_cost:.1f} bps")
        _pc4.metric("目标多头",  f"{_pf_nl}")
        _pc5.metric("目标空头",  f"{_pf_ns}")

        for _w in _pf_warns:
            st.warning(_w)

        if _pf_no_op:
            st.success("✅ 今日无须再平衡 — 权重偏差在阈值范围内，组合已是最优")
        else:
            # ── Change manifest ───────────────────────────────────────────
            st.markdown(
                f'<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;'
                f'color:{_mu40};margin:0.5rem 0 0.2rem;">拟变更明细</div>',
                unsafe_allow_html=True,
            )
            _t_df = pd.DataFrame(_pf_trades)
            for _tc in ["weight_before", "weight_after", "weight_delta"]:
                if _tc in _t_df.columns:
                    _t_df[_tc] = _t_df[_tc].apply(lambda x: f"{x:+.2%}")
            if "cost_bps" in _t_df.columns:
                _t_df["cost_bps"] = _t_df["cost_bps"].apply(
                    lambda x: f"{x:.1f}" if x else "—")
            _show_t = [c for c in ["sector", "ticker", "action", "weight_before",
                                   "weight_after", "weight_delta", "cost_bps",
                                   "trigger_reason"] if c in _t_df.columns]
            st.dataframe(
                _t_df[_show_t].rename(columns={
                    "sector": "板块", "ticker": "ETF", "action": "操作",
                    "weight_before": "操作前", "weight_after": "操作后",
                    "weight_delta": "Δ权重", "cost_bps": "成本bps",
                    "trigger_reason": "触发原因",
                }),
                use_container_width=True, hide_index=True,
            )


        # ── Execute button ────────────────────────────────────────────────
        st.markdown('<div style="height:0.3rem;"></div>', unsafe_allow_html=True)
        _ex1, _ex2 = st.columns([1, 2])
        with _ex1:
            if st.button("✅ 执行再平衡", type="primary",
                         use_container_width=True, disabled=_pf_no_op):
                with st.spinner("写入交易记录…"):
                    execute_rebalance(today, dry_run=False)
                st.success("再平衡完成，持仓已更新。")
                st.rerun()
        with _ex2:
            if _pf_no_op:
                st.caption("当前无须执行再平衡。")
            else:
                st.caption(
                    f"将执行 {len(_pf_trades)} 笔调整  ·  "
                    f"预估换手成本 {_pf_cost:.1f} bps  ·  写入 SimulatedTrade 记录"
                )

# ── Tab: NAV chart ─────────────────────────────────────────────────────────────
with t_nav:
    if not monthly_df.empty and "return_month" in monthly_df.columns:
        monthly_df["return_month"] = pd.to_datetime(monthly_df["return_month"])
        port_m = (monthly_df.groupby("return_month")["contribution"]
                  .sum().sort_index())
        cum    = (1 + port_m).cumprod()

        fig_nav = go.Figure()
        fig_nav.add_trace(go.Scatter(
            x=cum.index, y=cum.values,
            mode="lines+markers",
            line=dict(color="#3b82f6", width=2),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.08)",
            name="模拟实盘净值",
        ))
        fig_nav.add_hline(y=1.0, line_dash="dash", line_color="#6b7280",
                          annotation_text="基准 1.0")
        fig_nav.update_layout(
            height=320, xaxis_title="", yaxis_title="累计净值",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"), margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_nav, use_container_width=True)

        cum_ret = float(cum.iloc[-1] - 1) if len(cum) > 0 else 0.0
        ann_ret = float((cum.iloc[-1]) ** (12 / len(port_m)) - 1) if len(port_m) >= 2 else 0.0
        sharpe  = (float(port_m.mean()) / float(port_m.std()) * 12 ** 0.5
                   if port_m.std() > 0 else 0.0)
        max_dd  = float(((cum / cum.cummax()) - 1).min())

        n1, n2, n3, n4 = st.columns(4)
        n1.metric("累计收益", f"{cum_ret:+.2%}")
        n2.metric("年化收益", f"{ann_ret:+.2%}")
        n3.metric("Sharpe", f"{sharpe:.2f}")
        n4.metric("最大回撤", f"{max_dd:.2%}")

        # Record monthly return
        with st.expander("📋 记录上月收益归因"):
            rec_month = st.date_input(
                "记录月份", value=today.replace(day=1) - datetime.timedelta(days=1),
                max_value=today, key="rec_month")
            if st.button("记录", key="btn_record_ret"):
                with st.spinner("计算…"):
                    ret_r = record_monthly_return(rec_month.replace(day=1))
                if "error" in ret_r:
                    st.error(ret_r["error"])
                else:
                    st.success(
                        f"{rec_month.strftime('%Y-%m')} 收益 {ret_r['total_return']:+.2%}  "
                        f"盈利 {ret_r['n_profitable']} / 亏损 {ret_r['n_losing']}")
                    st.rerun()
    else:
        st.info("暂无月度收益数据。执行再平衡并记录月度收益后可见。")

# ── Tab: Monthly attribution ───────────────────────────────────────────────────
with t_attr:
    if not monthly_df.empty and "return_month" in monthly_df.columns:
        monthly_df["return_month"] = pd.to_datetime(monthly_df["return_month"])
        months = sorted(monthly_df["return_month"].unique(), reverse=True)
        for m in months:
            md   = monthly_df[monthly_df["return_month"] == m].copy()
            tot  = float(md["contribution"].sum())
            icon = "🟢" if tot > 0 else "🔴"
            with st.expander(
                f"{icon} {pd.Timestamp(m).strftime('%Y-%m')}  总收益 {tot:+.2%}",
                expanded=False,
            ):
                md = md.sort_values("contribution", ascending=False)
                for col in ["contribution", "sector_return", "weight_held"]:
                    if col in md.columns:
                        md[f"{col}_fmt"] = md[col].apply(
                            lambda x: f"{x:+.2%}" if pd.notna(x) else "—")
                show = ["sector"] + [c+"_fmt" for c in
                        ["weight_held","sector_return","contribution"] if c in md.columns]
                rename = {"sector":"板块","weight_held_fmt":"权重",
                          "sector_return_fmt":"板块收益","contribution_fmt":"贡献"}
                st.dataframe(md[show].rename(columns=rename),
                             use_container_width=True, hide_index=True)
    else:
        st.info("暂无收益归因数据。")

# ── Tab: Trade history ─────────────────────────────────────────────────────────
with t_trade:
    trade_df = _cached_trades(limit=200)
    if not trade_df.empty:
        trade_df = trade_df[
            trade_df["action"].notna() &
            (trade_df["weight_delta"].fillna(0).abs() > 0)
        ].copy()
    if not trade_df.empty:
        for col in ["weight_delta", "weight_before", "weight_after"]:
            if col in trade_df.columns:
                trade_df[col] = trade_df[col].apply(lambda x: f"{x:+.2%}")
        st.dataframe(trade_df.rename(columns={
            "trade_date":"日期","sector":"板块","ticker":"ETF",
            "action":"操作","weight_before":"操作前","weight_after":"操作后",
            "weight_delta":"变化","cost_bps":"成本bps","trigger_reason":"触发",
        }), use_container_width=True, hide_index=True)
    else:
        st.info("暂无交易记录。")

st.caption(
    f"Macro Alpha Pro Trading Terminal  ·  {today}  ·  "
    "买入/卖出参考价为末价 ±0.05% 估算（非实时订单簿）  ·  "
    "所有持仓为模拟Paper Trading，不构成投资建议"
)
