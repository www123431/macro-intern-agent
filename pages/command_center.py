"""
Macro Alpha Pro — Command Center
综合驾驶舱：30 秒掌握全局 — NAV · 制度 · 信号矩阵 · 待审批
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

import ui.theme as theme
from engine.memory import (
    init_db, get_system_config, get_stats,
    get_pending_decisions_for_monitor,
)

init_db()
theme.init_theme()

today = datetime.date.today()
_is_dark = theme.is_dark()
_C = {
    "green":  "#22c55e",
    "red":    "#ef4444",
    "yellow": "#f59e0b",
    "blue":   "#60a5fa",
    "muted":  "#8b949e" if _is_dark else "#64748b",
    "card":   "#1e293b" if _is_dark else "#ffffff",
    "border": "rgba(255,255,255,0.08)" if _is_dark else "#e2e8f0",
    "text":   "#f0f6fc" if _is_dark else "#0f172a",
}

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("### 🖥️ Command Center")
st.caption(f"综合驾驶舱  ·  {today}  ·  30 秒掌握全局")
st.divider()

# ── Load data (cached) ────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def _get_regime():
    try:
        from engine.regime import get_regime_on
        return get_regime_on(as_of=today, train_end=today)
    except Exception:
        return None

@st.cache_data(ttl=60)
def _get_signals(as_of_date, lookback_m, skip_m):
    try:
        from engine.signal import get_signal_dataframe, compute_composite_scores
        sig = get_signal_dataframe(as_of_date, lookback_m, skip_m)
        try:
            comp = compute_composite_scores(as_of_date, lookback_m, skip_m)
        except Exception:
            comp = pd.DataFrame()
        return sig, comp
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=60)
def _get_vix_price():
    try:
        fi = yf.Ticker("^VIX").fast_info
        return float(fi.last_price or 0)
    except Exception:
        return 0.0

@st.cache_data(ttl=60)
def _get_portfolio_stats():
    try:
        from engine.portfolio_tracker import get_current_positions
        nav_str = get_system_config("paper_trading_nav", "1000000")
        nav = float(nav_str)
        positions = get_current_positions()
        return nav, len(positions) if not positions.empty else 0
    except Exception:
        return 1_000_000.0, 0

regime    = _get_regime()
sig_df, comp_df = _get_signals(today, 12, 1)
vix_live  = _get_vix_price()
nav, n_pos = _get_portfolio_stats()
stats = get_stats()

# ── ROW 1: KPI Strip ──────────────────────────────────────────────────────────
col1, col2, col3, col4, col5, col6 = st.columns(6)

# NAV
col1.metric("模拟总资产", f"¥{nav:,.0f}")

# Regime
if regime:
    regime_icons = {"risk-on": "🟢", "risk-off": "🔴", "transition": "🟡"}
    col2.metric(
        "宏观制度",
        f"{regime_icons.get(regime.regime, '⚪')} {regime.regime}",
        delta=f"P(on)={regime.p_risk_on:.0%}",
    )
else:
    col2.metric("宏观制度", "计算中…")

# VIX
vix_delta = None
if vix_live > 0:
    if vix_live < 15:
        vix_label = "😌 平静"
    elif vix_live < 25:
        vix_label = "😐 正常"
    elif vix_live < 35:
        vix_label = "😬 警戒"
    else:
        vix_label = "🚨 危机"
    col3.metric("CBOE VIX", f"{vix_live:.1f}", delta=vix_label)

# Signal counts
if not sig_df.empty:
    n_long  = int((sig_df["tsmom"] == 1).sum())
    n_short = int((sig_df["tsmom"] == -1).sum())
    col4.metric("多头板块", n_long,  delta=f"空头 {n_short}")
else:
    col4.metric("多头板块", "—")

# Portfolio positions
col5.metric("当前持仓", n_pos)

# Pending decisions
pending = get_pending_decisions_for_monitor()
n_overdue = sum(1 for d in pending if d["urgency"] == "overdue")
col6.metric(
    "待验证决策",
    len(pending),
    delta=f"⚠️ {n_overdue} 过期" if n_overdue > 0 else "正常",
    delta_color="inverse" if n_overdue > 0 else "off",
)

st.divider()

# ── ROW 2: Signal Grid + Regime Panel ────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="large")

# ── Signal Grid ───────────────────────────────────────────────────────────────
with left_col:
    st.markdown("#### 📊 信号矩阵")

    if not sig_df.empty:
        _SIG_COLORS = {1: _C["green"], -1: _C["red"], 0: "#4b5563"}

        # Group by asset class if available
        sectors = sig_df.index.tolist()
        rows_html = []
        for sector in sectors:
            row = sig_df.loc[sector]
            ticker  = row.get("ticker", "")
            tsmom   = int(row.get("tsmom", 0))
            csmom   = int(row.get("csmom", 0))
            ret     = float(row.get("raw_return", 0)) * 100
            vol     = float(row.get("ann_vol", 0)) * 100
            tc      = _SIG_COLORS[tsmom]
            cc      = _SIG_COLORS[csmom]
            ret_c   = _C["green"] if ret > 0 else _C["red"]
            ret_str = f"{ret:+.1f}%"
            comp_v  = float(comp_df.loc[sector, "composite_score"]) \
                      if not comp_df.empty and sector in comp_df.index else None
            comp_str = f"{comp_v:.0f}" if comp_v is not None else "—"
            comp_c   = ("#22c55e" if comp_v and comp_v >= 70
                        else "#3b82f6" if comp_v and comp_v >= 50
                        else "#f59e0b" if comp_v and comp_v >= 30
                        else "#ef4444" if comp_v else "#4b5563")
            rows_html.append(
                f'<tr>'
                f'<td style="padding:4px 8px; font-weight:600; color:{_C["text"]}; '
                f'font-size:0.82rem; white-space:nowrap;">{sector}</td>'
                f'<td style="padding:4px 6px; font-size:0.78rem; color:{_C["muted"]};">{ticker}</td>'
                f'<td style="padding:4px 6px; text-align:center; '
                f'background:{tc}22; color:{tc}; font-weight:700; font-size:0.78rem; '
                f'border-radius:3px;">{"▲" if tsmom==1 else "▼" if tsmom==-1 else "—"}</td>'
                f'<td style="padding:4px 6px; text-align:center; '
                f'background:{cc}22; color:{cc}; font-weight:700; font-size:0.78rem; '
                f'border-radius:3px;">{"▲" if csmom==1 else "▼" if csmom==-1 else "—"}</td>'
                f'<td style="padding:4px 6px; text-align:right; color:{ret_c}; '
                f'font-family:monospace; font-size:0.78rem;">{ret_str}</td>'
                f'<td style="padding:4px 6px; text-align:right; color:{comp_c}; '
                f'font-family:monospace; font-size:0.78rem; font-weight:700;">{comp_str}</td>'
                f'</tr>'
            )

        table_html = (
            '<table style="width:100%; border-collapse:collapse;">'
            '<thead><tr>'
            f'<th style="padding:4px 8px; text-align:left; font-size:0.7rem; '
            f'text-transform:uppercase; letter-spacing:0.08em; color:{_C["muted"]}; '
            f'border-bottom:1px solid {_C["border"]};">板块</th>'
            f'<th style="padding:4px 6px; text-align:left; font-size:0.7rem; '
            f'text-transform:uppercase; letter-spacing:0.08em; color:{_C["muted"]}; '
            f'border-bottom:1px solid {_C["border"]};">ETF</th>'
            f'<th style="padding:4px 6px; text-align:center; font-size:0.7rem; '
            f'text-transform:uppercase; letter-spacing:0.08em; color:{_C["muted"]}; '
            f'border-bottom:1px solid {_C["border"]};">TSMOM</th>'
            f'<th style="padding:4px 6px; text-align:center; font-size:0.7rem; '
            f'text-transform:uppercase; letter-spacing:0.08em; color:{_C["muted"]}; '
            f'border-bottom:1px solid {_C["border"]};">CSMOM</th>'
            f'<th style="padding:4px 6px; text-align:right; font-size:0.7rem; '
            f'text-transform:uppercase; letter-spacing:0.08em; color:{_C["muted"]}; '
            f'border-bottom:1px solid {_C["border"]};">12-1M</th>'
            f'<th style="padding:4px 6px; text-align:right; font-size:0.7rem; '
            f'text-transform:uppercase; letter-spacing:0.08em; color:{_C["muted"]}; '
            f'border-bottom:1px solid {_C["border"]};">Score</th>'
            '</tr></thead>'
            '<tbody>' + ''.join(rows_html) + '</tbody>'
            '</table>'
        )
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("信号数据加载中…")

# ── Right Panel: Regime + Alerts ─────────────────────────────────────────────
with right_col:

    # Regime mini-dashboard
    st.markdown("#### 🌊 制度状态")
    if regime:
        regime_color = {
            "risk-on":    _C["green"],
            "risk-off":   _C["red"],
            "transition": _C["yellow"],
        }.get(regime.regime, _C["muted"])

        st.markdown(
            f'<div style="background:{regime_color}15; border:1px solid {regime_color}55; '
            f'border-left:4px solid {regime_color}; border-radius:8px; '
            f'padding:1rem 1.2rem; margin-bottom:1rem;">'
            f'<div style="font-size:1.4rem; font-weight:800; color:{regime_color}; '
            f'text-transform:uppercase; letter-spacing:0.05em;">{regime.regime}</div>'
            f'<div style="font-size:0.85rem; color:{_C["muted"]}; margin-top:0.3rem;">'
            f'P(risk-on) = <strong style="color:{regime_color};">{regime.p_risk_on:.1%}</strong>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Regime indicators
        r1, r2 = st.columns(2)
        r1.metric("P(risk-on)",  f"{regime.p_risk_on:.1%}")
        r2.metric("P(risk-off)", f"{1 - regime.p_risk_on:.1%}")

        if hasattr(regime, "yield_spread") and regime.yield_spread is not None:
            st.metric("10Y-2Y 利差", f"{regime.yield_spread:.2f}%",
                      delta=("曲线倒挂" if regime.yield_spread < 0 else "正常形态"),
                      delta_color="inverse" if regime.yield_spread < 0 else "off")
    else:
        st.info("制度数据加载中…")

    st.divider()

    # Alerts panel
    st.markdown("#### ⚠️ 告警")

    alerts = []

    # VIX alert
    if vix_live >= 35:
        alerts.append(("🔴 CRISIS", f"VIX = {vix_live:.1f} — 进入危机区间", "red"))
    elif vix_live >= 25:
        alerts.append(("🟡 ELEVATED", f"VIX = {vix_live:.1f} — 波动率偏高", "yellow"))

    # Regime alert
    if regime and regime.regime == "risk-off":
        alerts.append(("🔴 RISK-OFF", "宏观制度转为防御性，建议降低风险敞口", "red"))
    elif regime and regime.regime == "transition":
        alerts.append(("🟡 TRANSITION", "制度处于过渡期，信号不稳定，建议谨慎", "yellow"))

    # Signal flip alert
    if not sig_df.empty:
        n_long_now  = int((sig_df["tsmom"] == 1).sum())
        n_short_now = int((sig_df["tsmom"] == -1).sum())
        if n_short_now > n_long_now:
            alerts.append(("🔴 BEAR SKEW", f"空头({n_short_now}) > 多头({n_long_now})，整体偏空", "red"))

    # Overdue decisions
    if n_overdue > 0:
        alerts.append(("⏳ REVIEW", f"{n_overdue} 个决策已过验证期限", "yellow"))

    if alerts:
        for label, msg, color in alerts:
            c = _C[color]
            st.markdown(
                f'<div style="padding:0.5rem 0.8rem; margin-bottom:0.5rem; '
                f'border-left:3px solid {c}; background:{c}11; border-radius:4px; '
                f'font-size:0.82rem;">'
                f'<span style="color:{c}; font-weight:700;">{label}</span>  '
                f'<span style="color:{_C["text"]};">{msg}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f'<div style="padding:0.8rem; text-align:center; color:{_C["muted"]}; '
            f'font-size:0.85rem; border:1px dashed {_C["border"]}; border-radius:6px;">'
            f'✅ 无告警 — 系统运行正常</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ── ROW 3: Pending Approvals ──────────────────────────────────────────────────
st.markdown("#### 📋 近期待验证决策")

if pending:
    _URGENCY_COLOR = {"overdue": _C["red"], "approaching": _C["yellow"], "normal": _C["green"]}
    _DIR_COLOR = {"超配": _C["green"], "低配": _C["red"], "标配": _C["blue"]}

    # Show at most 5 items
    for d in pending[:5]:
        uc = _URGENCY_COLOR.get(d["urgency"], _C["muted"])
        dc = _DIR_COLOR.get(d["direction"], _C["muted"])
        ul = {"overdue":"已过期限", "approaching":"即将到期", "normal":"正常追踪"}.get(d["urgency"], d["urgency"])
        conf = f"{d['confidence_score']}%" if d.get("confidence_score") is not None else "—"
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:1rem; padding:0.5rem 0.8rem; '
            f'margin-bottom:0.4rem; border:1px solid {_C["border"]}; border-radius:4px; '
            f'background:{_C["card"]};">'
            f'<span style="color:{uc}; font-weight:700; font-size:0.75rem; '
            f'min-width:5rem;">{ul}</span>'
            f'<span style="color:{_C["text"]}; font-weight:600; font-size:0.82rem; '
            f'min-width:8rem;">{d.get("sector_name", "—")}</span>'
            f'<span style="color:{dc}; font-weight:700; font-size:0.82rem; '
            f'min-width:4rem;">{d.get("direction", "—")}</span>'
            f'<span style="color:{_C["muted"]}; font-size:0.78rem; font-family:monospace;">'
            f'conf={conf}  ·  到期 {d.get("verify_by", "—")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    if len(pending) > 5:
        st.caption(f"… 还有 {len(pending) - 5} 条，请前往 Decision Journal 查看全部")
else:
    st.markdown(
        f'<div style="padding:0.8rem; text-align:center; color:{_C["muted"]}; '
        f'font-size:0.85rem;">暂无待验证决策</div>',
        unsafe_allow_html=True,
    )

st.divider()
st.caption(
    f"Command Center  ·  {today}  ·  "
    "信号每 60 秒刷新  ·  制度每 1 小时刷新  ·  Macro Alpha Pro"
)
