"""
Macro Alpha Pro — Trading Desk
专业交易台：Watchlist状态树 · 待审批队列 · 价格走势图 · 仓位审批
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

import ui.theme as theme
from engine.memory import (
    init_db, SessionFactory,
    WatchlistEntry, PendingApproval, SimulatedPosition, SimulatedTrade, RegimeSnapshot,
    SignalSnapshot,
)
from engine.daily_batch import ensure_daily_batch_completed

# ── Page config ────────────────────────────────────────────────────────────────
# set_page_config handled by app.py via st.navigation()
# st.set_page_config(page_title="Trading Desk | Macro Alpha Pro", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")
theme.init_theme()
init_db()

_is_dark = theme.is_dark()
_C = {
    "green":  "#22c55e",
    "red":    "#ef4444",
    "yellow": "#fbbf24",
    "blue":   "#60a5fa",
    "muted":  "#64748b" if not _is_dark else "#8b949e",
    "card":   "#1e293b" if _is_dark else "#ffffff",
    "border": "rgba(255,255,255,0.08)" if _is_dark else "#e2e8f0",
    "text":   "#f0f6fc" if _is_dark else "#0f172a",
    "bg":     "#0f172a" if _is_dark else "#f1f5f9",
    "mono":   "'Courier New', 'JetBrains Mono', monospace",
}

# ── Terminal CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* Base terminal feel */
.td-mono   {{ font-family: {_C['mono']}; }}
.td-up     {{ color: {_C['green']}; font-weight:600; }}
.td-down   {{ color: {_C['red']};   font-weight:600; }}
.td-warn   {{ color: {_C['yellow']}; font-weight:600; }}
.td-muted  {{ color: {_C['muted']}; font-size:0.78rem; }}
.td-price  {{ font-family:{_C['mono']}; font-size:1.55rem; font-weight:700; }}
.td-label  {{ font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em;
              color:{_C['muted']}; margin-bottom:2px; }}
.td-val    {{ font-family:{_C['mono']}; font-size:0.92rem; font-weight:600; }}

/* Status bar */
.td-statusbar {{
  display:flex; gap:1.5rem; align-items:center;
  padding:0.5rem 1rem;
  background:{_C['card']};
  border:1px solid {_C['border']};
  border-radius:4px;
  margin-bottom:0.8rem;
  font-family:{_C['mono']};
  font-size:0.82rem;
}}
.td-statusbar .lbl {{ color:{_C['muted']}; margin-right:4px; }}

/* Watchlist row */
.wl-row {{
  display:flex; align-items:center; gap:8px;
  padding:5px 8px; border-radius:3px; cursor:pointer;
  font-family:{_C['mono']}; font-size:0.82rem;
  border-left:3px solid transparent;
  transition: background 0.12s;
}}
.wl-row:hover {{ background:rgba(255,255,255,0.04); }}
.wl-active  {{ border-left-color:{_C['green']}; }}
.wl-triggered {{ border-left-color:{_C['blue']}; }}
.wl-watching {{ border-left-color:{_C['muted']}; }}
.wl-invalid {{ border-left-color:{_C['red']}; opacity:0.6; }}
.wl-corr {{ border-left-color:{_C['yellow']}; opacity:0.85; }}

/* Approval card */
.ap-card {{
  background:{_C['card']};
  border:1px solid {_C['border']};
  border-radius:4px;
  padding:10px 14px;
  margin-bottom:8px;
  font-family:{_C['mono']};
}}
.ap-card-critical {{ border-left:4px solid {_C['red']}; }}
.ap-card-entry    {{ border-left:4px solid {_C['blue']}; }}
.ap-card-rebalance {{ border-left:4px solid {_C['muted']}; }}

/* Section header */
.td-section {{
  font-size:0.68rem; text-transform:uppercase; letter-spacing:0.12em;
  color:{_C['muted']}; padding:4px 0 6px;
  border-bottom:1px solid {_C['border']}; margin-bottom:6px;
}}

/* Quant badge */
.qbadge {{
  display:inline-block; padding:1px 7px; border-radius:2px;
  font-size:0.72rem; font-family:{_C['mono']}; font-weight:600;
}}
.qbadge-on  {{ background:rgba(34,197,94,0.15);  color:{_C['green']}; }}
.qbadge-off {{ background:rgba(239,68,68,0.15);   color:{_C['red']}; }}
.qbadge-tr  {{ background:rgba(251,191,36,0.15);  color:{_C['yellow']}; }}
</style>
""", unsafe_allow_html=True)

# ── Run daily batch (idempotent) ───────────────────────────────────────────────
with st.spinner("Checking market data..."):
    batch = ensure_daily_batch_completed()

today = datetime.date.today()

# ── P1-B: Daily Batch status bar ──────────────────────────────────────────────
with st.container():
    _b_cols = st.columns([1, 1, 1, 1, 2])
    _b_cols[0].metric(
        "信号",
        "✅ OK" if batch.signal_ok else ("⏭ 跳过" if batch.skipped else "❌ 失败"),
    )
    _b_cols[1].metric(
        "制度",
        "✅ OK" if batch.regime_ok else ("⏭ 跳过" if batch.skipped else "❌ 失败"),
    )
    _b_cols[2].metric("止损触发", len(batch.risk_alerts))
    _b_cols[3].metric("入场触发", len(batch.entries_triggered))
    _status_parts = []
    if batch.invalidations:
        _status_parts.append(f"失效: {', '.join(batch.invalidations)}")
    if batch.rebalance_orders:
        _status_parts.append(f"再平衡: {len(batch.rebalance_orders)} 笔")
    if batch.errors:
        st.error(f"Batch 错误: {'; '.join(batch.errors)}")
    elif _status_parts:
        _b_cols[4].caption(" | ".join(_status_parts))
    else:
        _b_cols[4].caption(f"截至 {batch.as_of_date} 无异常")
st.divider()

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _fetch_quote(ticker: str) -> dict:
    try:
        fi   = yf.Ticker(ticker).fast_info
        last = float(fi.last_price or 0)
        prev = float(fi.regular_market_previous_close or fi.previous_close or last)
        chg  = (last / prev - 1.0) if prev > 0 else 0.0
        sprd = last * 0.0005
        return {
            "last":      last, "prev": prev, "chg": chg,
            "day_high":  float(fi.day_high  or last),
            "day_low":   float(fi.day_low   or last),
            "year_high": float(fi.year_high or last),
            "year_low":  float(fi.year_low  or last),
            "volume":    int(fi.last_volume or 0),
            "bid":       round(last - sprd, 2),
            "ask":       round(last + sprd, 2),
        }
    except Exception:
        return {}


@st.cache_data(ttl=300)
def _fetch_ohlcv(ticker: str, period: str = "6mo") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, auto_adjust=True,
                         progress=False, multi_level_index=False)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def _load_latest_tsmom() -> dict[str, int | None]:
    """
    Read the most recent SignalSnapshot and return {sector: tsmom_signal}.
    Used by the approval card freshness strip to detect momentum reversals
    without re-running the full QuantAgent pipeline.
    """
    try:
        with SessionFactory() as s:
            snap = (
                s.query(SignalSnapshot)
                 .order_by(SignalSnapshot.as_of_date.desc())
                 .first()
            )
            if not snap:
                return {}
        df = pd.read_json(snap.signals_json, orient="split")
        if "tsmom" not in df.columns:
            return {}
        return {sector: int(df.loc[sector, "tsmom"]) for sector in df.index}
    except Exception:
        return {}


def _freshness_strip(
    we: WatchlistEntry,
    current_tsmom: dict[str, int | None],
) -> str:
    """
    Return an HTML snippet for the signal freshness strip inside an approval card.
    Shows signal age, TSMOM current vs entry, and entry composite score.
    Color-codes staleness and momentum flip.
    """
    signal_days = (datetime.date.today() - we.created_date).days if we.created_date else 0
    age_color = _C["green"] if signal_days <= 7 else (_C["yellow"] if signal_days <= 21 else _C["red"])
    age_label = f"{signal_days}d"

    entry_tsmom = we.entry_tsmom_signal
    cur_tsmom   = current_tsmom.get(we.sector)
    if entry_tsmom is not None and cur_tsmom is not None:
        tsmom_sign  = lambda v: ("+1" if v > 0 else ("-1" if v < 0 else "0"))
        flipped     = (entry_tsmom > 0) != (cur_tsmom > 0) and cur_tsmom != 0
        tsmom_color = _C["red"] if flipped else _C["green"]
        tsmom_label = (
            f"{tsmom_sign(entry_tsmom)}→{tsmom_sign(cur_tsmom)}"
            f"{' ⚠FLIP' if flipped else ''}"
        )
    else:
        tsmom_color = _C["muted"]
        tsmom_label = "—"

    score_label = f"{we.entry_composite_score}/100" if we.entry_composite_score is not None else "—"

    return (
        f'<div style="border-top:1px solid {_C["border"]}; margin-top:6px; padding-top:5px; '
        f'display:flex; gap:16px; font-size:0.74rem; font-family:{_C["mono"]};">'
        f'<span><span style="color:{_C["muted"]}">AGE </span>'
        f'<b style="color:{age_color}">{age_label}</b></span>'
        f'<span><span style="color:{_C["muted"]}">TSMOM </span>'
        f'<b style="color:{tsmom_color}">{tsmom_label}</b></span>'
        f'<span><span style="color:{_C["muted"]}">SCORE@ENTRY </span>'
        f'<b>{score_label}</b></span>'
        f'</div>'
    )


def _load_watchlist() -> list:
    with SessionFactory() as s:
        return s.query(WatchlistEntry).order_by(WatchlistEntry.created_date.desc()).all()


def _load_pending() -> list:
    with SessionFactory() as s:
        rows = (
            s.query(PendingApproval)
             .filter(PendingApproval.status == "pending")
             .order_by(PendingApproval.triggered_date.desc())
             .all()
        )
        return rows


def _load_active_positions() -> pd.DataFrame:
    with SessionFactory() as s:
        latest = (
            s.query(SimulatedPosition.snapshot_date)
             .order_by(SimulatedPosition.snapshot_date.desc())
             .scalar()
        )
        if not latest:
            return pd.DataFrame()
        rows = s.query(SimulatedPosition).filter(
            SimulatedPosition.snapshot_date == latest
        ).all()
        return pd.DataFrame([{
            "sector": r.sector, "ticker": r.ticker,
            "actual_weight": r.actual_weight, "entry_price": r.entry_price,
            "cost_basis": r.cost_basis, "signal_tsmom": r.signal_tsmom,
            "regime_label": r.regime_label,
        } for r in rows])


def _get_regime() -> dict:
    with SessionFactory() as s:
        snap = (
            s.query(RegimeSnapshot)
             .order_by(RegimeSnapshot.as_of_date.desc())
             .first()
        )
        if snap:
            return {
                "regime":    snap.regime,
                "p_risk_on": snap.p_risk_on,
                "as_of":     snap.as_of_date,
            }
    return {}


# ── Approval actions ──────────────────────────────────────────────────────────

def _approve(approval_id: int) -> None:
    with SessionFactory() as s:
        ap = s.query(PendingApproval).filter(PendingApproval.id == approval_id).first()
        if not ap:
            return
        ap.status      = "approved"
        ap.resolved_at = datetime.datetime.utcnow()
        ap.resolved_by = "human"

        # Transition linked WatchlistEntry to active
        if ap.watchlist_entry_id:
            we = s.query(WatchlistEntry).filter(WatchlistEntry.id == ap.watchlist_entry_id).first()
            if we and we.status == "triggered":
                we.status = "active"

        suggested_weight = ap.suggested_weight or 0.0

        # Use live price; fall back to triggered_price if yfinance unavailable
        q     = _fetch_quote(ap.ticker)
        close = q.get("last") or ap.triggered_price or 0.0

        # Write SimulatedTrade record
        s.add(SimulatedTrade(
            trade_date     = today,
            sector         = ap.sector,
            ticker         = ap.ticker,
            action         = "BUY",
            weight_before  = 0.0,
            weight_after   = suggested_weight,
            weight_delta   = suggested_weight,
            cost_bps       = 10.0,
            trigger_reason = "human_approved",
        ))

        # Upsert SimulatedPosition (unique on snapshot_date + sector)
        pos = s.query(SimulatedPosition).filter(
            SimulatedPosition.snapshot_date == today,
            SimulatedPosition.sector        == ap.sector,
        ).first()
        if pos:
            pos.ticker        = ap.ticker
            pos.actual_weight = suggested_weight
            pos.target_weight = suggested_weight
            pos.entry_price   = close
            pos.trailing_high = close
            pos.regime_label  = regime_info.get("regime", "")
        else:
            s.add(SimulatedPosition(
                snapshot_date = today,
                sector        = ap.sector,
                ticker        = ap.ticker,
                target_weight = suggested_weight,
                actual_weight = suggested_weight,
                entry_price   = close,
                trailing_high = close,
                regime_label  = regime_info.get("regime", ""),
                signal_tsmom  = 1,
            ))

        s.commit()
    st.cache_data.clear()


def _reject(approval_id: int, reason: str = "") -> None:
    with SessionFactory() as s:
        ap = s.query(PendingApproval).filter(PendingApproval.id == approval_id).first()
        if not ap:
            return
        ap.status           = "rejected"
        ap.resolved_at      = datetime.datetime.utcnow()
        ap.resolved_by      = "human"
        ap.rejection_reason = reason
        if ap.watchlist_entry_id and ap.approval_type == "entry":
            we = s.query(WatchlistEntry).filter(WatchlistEntry.id == ap.watchlist_entry_id).first()
            if we:
                we.status = "watching"
        s.commit()
    st.cache_data.clear()


# ── Chart builder ─────────────────────────────────────────────────────────────

def _build_chart(
    ticker: str,
    entry_price: float = 0,
    stop_price: float  = 0,
    trigger_price: float = 0,
    label: str = "",
    period: str = "6mo",
) -> go.Figure:
    df = _fetch_ohlcv(ticker, period=period)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.02,
    )

    bg_color   = "#0f172a" if _is_dark else "#f8fafc"
    grid_color = "rgba(255,255,255,0.06)" if _is_dark else "rgba(0,0,0,0.06)"
    text_color = "#94a3b8" if _is_dark else "#64748b"

    if df.empty:
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(color=text_color, size=14))
        fig.update_layout(height=420, paper_bgcolor=bg_color, plot_bgcolor=bg_color)
        return fig

    # ── Candlesticks ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"],  close=df["Close"],
        increasing_line_color=_C["green"], decreasing_line_color=_C["red"],
        increasing_fillcolor=_C["green"],  decreasing_fillcolor=_C["red"],
        line_width=1, name=ticker,
    ), row=1, col=1)

    # ── Moving averages ───────────────────────────────────────────────────────
    closes = df["Close"]
    for period, color, dash in [(20, "#60a5fa", "dot"), (50, "#a78bfa", "dash"), (200, "#f59e0b", "solid")]:
        if len(closes) >= period:
            sma = closes.rolling(period).mean()
            fig.add_trace(go.Scatter(
                x=df.index, y=sma, mode="lines", name=f"SMA{period}",
                line=dict(color=color, width=1, dash=dash), opacity=0.8,
            ), row=1, col=1)

    # ── Reference price lines ─────────────────────────────────────────────────
    x0, x1 = df.index[0], df.index[-1]

    if entry_price > 0:
        fig.add_hline(y=entry_price, row=1, col=1,
                      line=dict(color=_C["green"], width=1, dash="dash"),
                      annotation_text=f"Entry {entry_price:.2f}",
                      annotation_font_color=_C["green"],
                      annotation_bgcolor=bg_color)

    if stop_price > 0:
        fig.add_hline(y=stop_price, row=1, col=1,
                      line=dict(color=_C["red"], width=1.5, dash="dot"),
                      annotation_text=f"Stop {stop_price:.2f}",
                      annotation_font_color=_C["red"],
                      annotation_bgcolor=bg_color)

    if trigger_price > 0 and trigger_price != entry_price:
        fig.add_hline(y=trigger_price, row=1, col=1,
                      line=dict(color=_C["blue"], width=1, dash="dash"),
                      annotation_text=f"Trigger {trigger_price:.2f}",
                      annotation_font_color=_C["blue"],
                      annotation_bgcolor=bg_color)

    # ── Volume bars ───────────────────────────────────────────────────────────
    colors_vol = [
        _C["green"] if c >= o else _C["red"]
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=colors_vol, marker_opacity=0.6,
        name="Volume", showlegend=False,
    ), row=2, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        height=440,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(family="Courier New, monospace", color=text_color, size=11),
        margin=dict(l=8, r=8, t=28, b=8),
        legend=dict(
            orientation="h", y=1.02, x=0,
            font=dict(size=10), bgcolor="rgba(0,0,0,0)",
        ),
        xaxis_rangeslider_visible=False,
        title=dict(text=f"<b>{ticker}</b> {label}", font=dict(size=13, color=_C["text"]), x=0.01),
        hovermode="x unified",
    )
    for row in [1, 2]:
        fig.update_xaxes(
            row=row, col=1,
            gridcolor=grid_color, zeroline=False,
            showspikes=True, spikecolor=text_color, spikethickness=1,
        )
        fig.update_yaxes(
            row=row, col=1,
            gridcolor=grid_color, zeroline=False,
            tickfont=dict(family="Courier New", size=10),
        )

    return fig


# ── Load all data ──────────────────────────────────────────────────────────────
watchlist   = _load_watchlist()
pending     = _load_pending()
positions   = _load_active_positions()
regime_info = _get_regime()

pending_sorted = sorted(
    pending,
    key=lambda x: (
        0 if x.priority == "critical" else (1 if x.approval_type == "entry" else 2),
        x.triggered_date or datetime.date.min,
    )
)

# Status counts
n_active       = sum(1 for w in watchlist if w.status == "active")
n_approved     = sum(1 for w in watchlist if w.status == "triggered")
n_watching     = sum(1 for w in watchlist if w.status == "watching")
n_corr_blocked = sum(1 for w in watchlist if w.status == "corr_blocked")
n_pending      = len(pending_sorted)

# ── Session state for selection ────────────────────────────────────────────────
if "td_selected_ticker"  not in st.session_state: st.session_state.td_selected_ticker  = None
if "td_selected_entry_p" not in st.session_state: st.session_state.td_selected_entry_p = 0.0
if "td_selected_stop_p"  not in st.session_state: st.session_state.td_selected_stop_p  = 0.0
if "td_selected_trig_p"  not in st.session_state: st.session_state.td_selected_trig_p  = 0.0
if "td_detail_label"     not in st.session_state: st.session_state.td_detail_label      = ""
if "td_reject_id"        not in st.session_state: st.session_state.td_reject_id         = None
if "td_pending_ctx"      not in st.session_state: st.session_state.td_pending_ctx       = {}

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER BAR
# ══════════════════════════════════════════════════════════════════════════════
regime_label = regime_info.get("regime", "—")
regime_color = {"risk-on": _C["green"], "risk-off": _C["red"], "transition": _C["yellow"]}.get(regime_label, _C["muted"])
regime_icon  = {"risk-on": "●", "risk-off": "●", "transition": "◐"}.get(regime_label, "○")
data_date    = regime_info.get("as_of", today - datetime.timedelta(days=1))

n_new_triggers   = len(batch.entries_triggered) if not batch.skipped else 0
n_risk_alerts    = len(batch.risk_alerts)       if not batch.skipped else 0
n_corr_new       = len(batch.corr_blocked)      if not batch.skipped else 0
batch_color    = _C["green"] if (batch.skipped or batch.ok) else _C["red"]
batch_label    = "SKIPPED" if batch.skipped else ("OK" if batch.ok else "ERR")

st.markdown(f"""
<div class="td-statusbar">
  <span><span class="lbl">{today}</span>
    <b style="color:{batch_color}">Batch {batch_label}</b>
  </span>
  <span style="color:{regime_color}">
    {regime_icon} <b>{regime_label.upper()}</b>
    <span style="color:{_C['muted']}; font-size:0.72rem;">
      &nbsp;p={regime_info.get('p_risk_on', 0):.0%}
    </span>
  </span>
  <span><span class="lbl">TRIGGERED</span>
    <b style="color:{'#60a5fa' if n_new_triggers>0 else _C['muted']}">{n_new_triggers}</b>
  </span>
  <span><span class="lbl">CORR BLOCKED</span>
    <b style="color:{'#f59e0b' if n_corr_blocked>0 else _C['muted']}">{n_corr_blocked}</b>
  </span>
  <span><span class="lbl">RISK ALERTS</span>
    <b style="color:{'#ef4444' if n_risk_alerts>0 else _C['muted']}">{n_risk_alerts}</b>
  </span>
  <span><span class="lbl">PENDING</span>
    <b style="color:{'#ef4444' if n_pending>0 else _C['muted']}">{n_pending}</b>
  </span>
  <span><span class="lbl">ACTIVE</span><b style="color:{_C['green']}">{n_active}</b></span>
  <span><span class="lbl">WATCHING</span><b>{n_watching}</b></span>
  <span style="margin-left:auto; font-size:0.72rem; color:{_C['muted']}">
    For Educational Purposes Only · Not Investment Advice
  </span>
</div>
""", unsafe_allow_html=True)

# ── Batch alerts ──────────────────────────────────────────────────────────────
if not batch.skipped and batch.errors:
    st.error(f"Batch errors: {' | '.join(batch.errors)}")
if not batch.skipped and batch.risk_alerts:
    st.warning(f"⚡ Risk alerts triggered: {' · '.join(batch.risk_alerts)}")

# ══════════════════════════════════════════════════════════════════════════════
#  THREE-COLUMN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_center, col_right = st.columns([2, 3, 4], gap="small")

# ─────────────────────────────────────────────────────────────────────────────
#  LEFT: Watchlist State Tree
# ─────────────────────────────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="td-section">POSITIONS &amp; WATCHLIST</div>', unsafe_allow_html=True)

    def _wl_button(entry, css_cls: str, label: str, icon: str, detail: str) -> None:
        q = _fetch_quote(entry.ticker)
        last = q.get("last", 0)
        chg  = q.get("chg", 0)
        chg_color = _C["green"] if chg > 0 else (_C["red"] if chg < 0 else _C["muted"])
        chg_arrow = "▲" if chg > 0 else ("▼" if chg < 0 else "—")

        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.markdown(
                f'<div class="wl-row {css_cls}">'
                f'{icon} <b>{entry.ticker}</b>'
                f'<span class="td-muted"> {entry.sector[:8]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_b:
            btn_key = f"wl_{entry.id}_{entry.status}"
            if st.button(
                f"{last:.1f} {chg_arrow}{abs(chg):.1%}" if last else "select",
                key=btn_key,
                use_container_width=True,
                type="secondary",
            ):
                # Compute stop price for active positions
                stop_p = 0.0
                entry_p = entry.triggered_price or 0.0
                if entry.status == "active":
                    from engine.quant_agent import _fetch_price_context
                    atr, _ = _fetch_price_context(entry.ticker, today, atr_period=21)
                    if entry_p and atr:
                        stop_p = entry_p - 2.0 * atr

                st.session_state.td_selected_ticker  = entry.ticker
                st.session_state.td_selected_entry_p = entry_p
                st.session_state.td_selected_stop_p  = stop_p
                st.session_state.td_selected_trig_p  = entry.triggered_price or 0.0
                st.session_state.td_detail_label     = f"{entry.status.upper()} · {detail}"
                st.session_state.td_pending_ctx      = {}
                st.rerun()

    # ── Active positions ──────────────────────────────────────────────────────
    active_wl = [w for w in watchlist if w.status == "active"]
    if active_wl:
        st.markdown(f'<div class="td-muted">ACTIVE ({len(active_wl)})</div>', unsafe_allow_html=True)
        for e in active_wl:
            _wl_button(e, "wl-active", "active", "◆",
                       f"wt={e.suggested_weight:.1%}")
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Approved (awaiting execution) ─────────────────────────────────────────
    approved_wl = [w for w in watchlist if w.status == "triggered"]
    if approved_wl:
        st.markdown(f'<div class="td-muted">TRIGGERED ({len(approved_wl)})</div>', unsafe_allow_html=True)
        for e in approved_wl:
            _wl_button(e, "wl-triggered", "triggered", "◇",
                       f"trig={e.triggered_price:.2f}" if e.triggered_price else "pending")
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Watching ──────────────────────────────────────────────────────────────
    watching_wl = [w for w in watchlist if w.status == "watching"]
    if watching_wl:
        st.markdown(f'<div class="td-muted">WATCHING ({len(watching_wl)})</div>', unsafe_allow_html=True)
        for e in watching_wl:
            ec = json.loads(e.entry_condition_json or '{"type":"—"}')
            _wl_button(e, "wl-watching", "watching", "○",
                       ec.get("type", "—"))
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Correlation-blocked ───────────────────────────────────────────────────
    corr_blocked_wl = [w for w in watchlist if w.status == "corr_blocked"]
    if corr_blocked_wl:
        st.markdown(
            f'<div class="td-muted" style="color:#f59e0b;">CORR BLOCKED ({len(corr_blocked_wl)})</div>',
            unsafe_allow_html=True,
        )
        for e in corr_blocked_wl:
            col_a, col_b = st.columns([3, 2])
            with col_a:
                st.markdown(
                    f'<div class="wl-row wl-corr">⚡ <b>{e.ticker}</b>'
                    f'<span class="td-muted"> {e.sector[:8]}</span></div>',
                    unsafe_allow_html=True,
                )
            with col_b:
                if st.button("Override", key=f"corr_override_{e.id}",
                             use_container_width=True, type="secondary"):
                    with SessionFactory() as _os:
                        _oe = _os.query(WatchlistEntry).filter(WatchlistEntry.id == e.id).first()
                        if _oe:
                            _oe.status         = "triggered"
                            _oe.triggered_date = today
                            _close = _fetch_quote(_oe.ticker).get("last") or 0.0
                            _oe.triggered_price = _close
                            _deadline = today + datetime.timedelta(days=4)
                            _os.add(PendingApproval(
                                approval_type="entry",
                                priority="normal",
                                watchlist_entry_id=_oe.id,
                                sector=_oe.sector,
                                ticker=_oe.ticker,
                                triggered_condition=(
                                    f"HUMAN OVERRIDE — correlation block bypassed. "
                                    f"Price: {_close:.2f}. Verify portfolio concentration."
                                ),
                                triggered_date=today,
                                triggered_price=_close,
                                suggested_weight=_oe.suggested_weight,
                                position_rank=_oe.position_rank,
                                approval_deadline=_deadline,
                            ))
                            _os.commit()
                    st.cache_data.clear()
                    st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Invalidated (recent 5) ────────────────────────────────────────────────
    invalid_wl = [w for w in watchlist if w.status == "invalidated"][:5]
    if invalid_wl:
        st.markdown(f'<div class="td-muted">INVALIDATED (recent)</div>', unsafe_allow_html=True)
        for e in invalid_wl:
            col_a, col_b = st.columns([3, 2])
            with col_a:
                st.markdown(
                    f'<div class="wl-row wl-invalid">✕ <b>{e.ticker}</b>'
                    f'<span class="td-muted"> {e.sector[:8]}</span></div>',
                    unsafe_allow_html=True,
                )
            with col_b:
                reason = (e.invalidated_reason or "—")[:14]
                st.markdown(
                    f'<div class="td-muted" style="padding-top:4px">{reason}</div>',
                    unsafe_allow_html=True,
                )

    if not watchlist:
        st.markdown(
            '<div class="td-muted" style="padding:12px 0">No watchlist entries.<br>'
            'Run sector analysis to generate recommendations.</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
#  CENTER: Pending Approval Queue
# ─────────────────────────────────────────────────────────────────────────────
with col_center:
    st.markdown('<div class="td-section">PENDING APPROVALS</div>', unsafe_allow_html=True)

    if not pending_sorted:
        st.markdown(
            '<div class="td-muted" style="padding:16px 0">No pending approvals.<br>'
            'System is monitoring positions.</div>',
            unsafe_allow_html=True,
        )
    else:
        _current_tsmom   = _load_latest_tsmom()
        _wl_index        = {w.id: w for w in watchlist}  # id → WatchlistEntry

        for ap in pending_sorted:
            type_color = {
                "risk_control": _C["red"],
                "entry":        _C["blue"],
                "rebalance":    _C["muted"],
            }.get(ap.approval_type, _C["muted"])

            type_icon = {
                "risk_control": "⚠",
                "entry":        "▶",
                "rebalance":    "⇄",
            }.get(ap.approval_type, "·")

            card_css = {
                "risk_control": "ap-card-critical",
                "entry":        "ap-card-entry",
                "rebalance":    "ap-card-rebalance",
            }.get(ap.approval_type, "")

            days_old = (today - ap.triggered_date).days if ap.triggered_date else 0
            deadline_warn = (
                ap.approval_deadline and today > ap.approval_deadline
            )
            expire_note = (
                f' <span style="color:{_C["red"]}">EXPIRED</span>'
                if deadline_warn else ""
            )

            _we = _wl_index.get(ap.watchlist_entry_id) if ap.watchlist_entry_id else None
            _freshness = (
                _freshness_strip(_we, _current_tsmom)
                if (_we and ap.approval_type == "entry")
                else ""
            )

            st.markdown(
                f'<div class="ap-card {card_css}">'
                f'<span style="color:{type_color}; font-weight:700">'
                f'{type_icon} {ap.approval_type.upper().replace("_"," ")}</span>'
                f'&nbsp;&nbsp;<b style="font-size:0.95rem">{ap.ticker}</b>'
                f'&nbsp;<span class="td-muted">{ap.sector}</span>'
                f'<span class="td-muted" style="float:right">T-{days_old}{expire_note}</span>'
                f'<br>'
                f'<span class="td-muted" style="font-size:0.78rem">'
                f'{(ap.triggered_condition or "")[:80]}</span>'
                f'<br>'
                f'<span style="font-size:0.82rem">'
                f'<b>Suggested wt:</b> '
                f'{"—" if not ap.suggested_weight else f"{ap.suggested_weight:.1%}"}'
                f'&nbsp;&nbsp;'
                f'<b>Trig price:</b> '
                f'{"—" if not ap.triggered_price else f"{ap.triggered_price:.2f}"}'
                f'</span>'
                f'{_freshness}'
                f'</div>',
                unsafe_allow_html=True,
            )

            btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 3])

            with btn_col1:
                if st.button("✓ APPROVE", key=f"ap_ok_{ap.id}",
                             type="primary", use_container_width=True):
                    _approve(ap.id)
                    st.success(f"Approved {ap.ticker}")
                    st.rerun()

            with btn_col2:
                if st.button("✕ REJECT", key=f"ap_rej_{ap.id}",
                             type="secondary", use_container_width=True):
                    st.session_state.td_reject_id = ap.id
                    st.rerun()

            with btn_col3:
                if st.button("⤢ VIEW CHART", key=f"ap_chart_{ap.id}",
                             use_container_width=True):
                    st.session_state.td_selected_ticker  = ap.ticker
                    st.session_state.td_selected_trig_p  = ap.triggered_price or 0.0
                    st.session_state.td_selected_entry_p = 0.0
                    st.session_state.td_selected_stop_p  = 0.0
                    st.session_state.td_detail_label     = f"{ap.approval_type.upper()} · {ap.sector}"
                    st.session_state.td_pending_ctx      = {
                        "id":     ap.id,
                        "type":   ap.approval_type,
                        "weight": ap.suggested_weight,
                        "cond":   ap.triggered_condition,
                        "rank":   ap.position_rank,
                    }
                    st.rerun()

            # Rejection reason input
            if st.session_state.td_reject_id == ap.id:
                reason = st.text_input(
                    "Rejection reason", key=f"rej_reason_{ap.id}",
                    placeholder="e.g. Macro thesis changed, revisit next week",
                )
                rc1, rc2 = st.columns(2)
                with rc1:
                    if st.button("Confirm reject", key=f"rej_confirm_{ap.id}", type="primary"):
                        _reject(ap.id, reason)
                        st.session_state.td_reject_id = None
                        st.rerun()
                with rc2:
                    if st.button("Cancel", key=f"rej_cancel_{ap.id}"):
                        st.session_state.td_reject_id = None
                        st.rerun()

            st.markdown('<hr style="margin:4px 0; opacity:0.15">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  RIGHT: Chart + Detail Panel
# ─────────────────────────────────────────────────────────────────────────────
with col_right:
    sel_ticker = st.session_state.td_selected_ticker

    if not sel_ticker:
        st.markdown(
            f'<div style="height:440px; display:flex; align-items:center;'
            f'justify-content:center; color:{_C["muted"]}; '
            f'font-family:{_C["mono"]}; font-size:0.9rem; '
            f'border:1px solid {_C["border"]}; border-radius:4px;">'
            f'← Select a position or approval to view chart'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        q = _fetch_quote(sel_ticker)
        last = q.get("last", 0)
        chg  = q.get("chg",  0)
        chg_color = _C["green"] if chg > 0 else (_C["red"] if chg < 0 else _C["muted"])
        chg_arrow = "▲" if chg > 0 else ("▼" if chg < 0 else "—")

        # ── Quote strip ───────────────────────────────────────────────────────
        qc1, qc2, qc3, qc4, qc5 = st.columns([2, 2, 2, 2, 2])

        qc1.markdown(
            f'<div class="td-label">{st.session_state.td_detail_label}</div>'
            f'<div class="td-price" style="color:{_C["text"]}">{sel_ticker}</div>',
            unsafe_allow_html=True,
        )
        qc2.markdown(
            f'<div class="td-label">LAST</div>'
            f'<div class="td-price" style="color:{chg_color}">{last:.2f}</div>',
            unsafe_allow_html=True,
        )
        qc3.markdown(
            f'<div class="td-label">CHG</div>'
            f'<div class="td-val" style="color:{chg_color}">'
            f'{chg_arrow} {abs(chg):.2%}</div>',
            unsafe_allow_html=True,
        )
        qc4.markdown(
            f'<div class="td-label">DAY RANGE</div>'
            f'<div class="td-val">'
            f'{q.get("day_low",0):.2f} – {q.get("day_high",0):.2f}</div>',
            unsafe_allow_html=True,
        )
        qc5.markdown(
            f'<div class="td-label">52W RANGE</div>'
            f'<div class="td-val">'
            f'{q.get("year_low",0):.2f} – {q.get("year_high",0):.2f}</div>',
            unsafe_allow_html=True,
        )

        # ── Period selector ───────────────────────────────────────────────────
        period_opts = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y"}
        period_sel  = st.segmented_control(
            "Period", list(period_opts.keys()),
            default="6M", key=f"period_{sel_ticker}",
            label_visibility="collapsed",
        )
        period_code = period_opts.get(period_sel or "6M", "6mo")

        # ── Chart ─────────────────────────────────────────────────────────────
        fig = _build_chart(
            sel_ticker,
            entry_price=st.session_state.td_selected_entry_p,
            stop_price=st.session_state.td_selected_stop_p,
            trigger_price=st.session_state.td_selected_trig_p,
            label=st.session_state.td_detail_label,
            period=period_code,
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── Quant metrics strip ────────────────────────────────────────────────
        pending_ctx = st.session_state.td_pending_ctx
        if pending_ctx or st.session_state.td_selected_entry_p:
            st.markdown('<div class="td-section">POSITION DETAILS</div>', unsafe_allow_html=True)

            mc1, mc2, mc3, mc4 = st.columns(4)
            if st.session_state.td_selected_entry_p:
                pnl_pct = (last / st.session_state.td_selected_entry_p - 1.0) if st.session_state.td_selected_entry_p > 0 and last > 0 else 0.0
                pnl_color = _C["green"] if pnl_pct > 0 else _C["red"]
                mc1.markdown(
                    f'<div class="td-label">ENTRY PRICE</div>'
                    f'<div class="td-val">{st.session_state.td_selected_entry_p:.2f}</div>',
                    unsafe_allow_html=True,
                )
                mc2.markdown(
                    f'<div class="td-label">UNREALISED</div>'
                    f'<div class="td-val" style="color:{pnl_color}">{pnl_pct:+.2%}</div>',
                    unsafe_allow_html=True,
                )
            if st.session_state.td_selected_stop_p:
                dist_pct = (last / st.session_state.td_selected_stop_p - 1.0) if st.session_state.td_selected_stop_p > 0 and last > 0 else 0.0
                mc3.markdown(
                    f'<div class="td-label">ATR STOP</div>'
                    f'<div class="td-val" style="color:{_C["red"]}">'
                    f'{st.session_state.td_selected_stop_p:.2f}'
                    f' <span class="td-muted">({dist_pct:+.1%})</span></div>',
                    unsafe_allow_html=True,
                )
            if pending_ctx.get("weight"):
                mc4.markdown(
                    f'<div class="td-label">SUGGESTED WT</div>'
                    f'<div class="td-val">{pending_ctx["weight"]:.1%}</div>',
                    unsafe_allow_html=True,
                )

        # ── Inline approval panel when chart was opened from pending ──────────
        if pending_ctx.get("id"):
            st.markdown('<div class="td-section">APPROVAL</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="td-muted">{pending_ctx.get("cond", "")}</div>',
                unsafe_allow_html=True,
            )
            ia1, ia2 = st.columns(2)
            with ia1:
                if st.button("✓ APPROVE", key=f"inline_ok_{pending_ctx['id']}",
                             type="primary", use_container_width=True):
                    _approve(pending_ctx["id"])
                    st.session_state.td_pending_ctx = {}
                    st.success(f"Approved {sel_ticker}")
                    st.rerun()
            with ia2:
                if st.button("✕ REJECT", key=f"inline_rej_{pending_ctx['id']}",
                             use_container_width=True):
                    st.session_state.td_reject_id = pending_ctx["id"]
                    st.rerun()
