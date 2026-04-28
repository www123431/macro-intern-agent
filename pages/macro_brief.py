"""
Macro Brief — Intelligence Hub
===============================
Three lazy-loaded tabs:
  1. Macro Brief   — news-driven global macro summary  (tabs.render_tab1)
  2. Sector Scan   — full 36-sector alpha scanner       (tabs.render_tab5)
  3. Quant Audit   — agentic deep-dive terminal         (tabs.render_tab4)

Tabs 2 & 3 are gated behind a ▶ Run button inside each renderer,
so page load is always <2 s regardless of data freshness.
"""
import datetime
import json
import streamlit as st
import ui.tabs as tabs
import ui.theme as theme
from engine.memory import init_db, get_daily_brief_snapshot

init_db()
st.markdown(theme.build_css(), unsafe_allow_html=True)

# ── Pull shared state injected by app.py ─────────────────────────────────────
vix_input      = st.session_state.get("_vix_input", 20.0)
agent_executor = st.session_state.get("_agent_executor")

# ── Drill-down context banner ──────────────────────────────────────────────────
try:
    _today = datetime.date.today()
    _snap  = get_daily_brief_snapshot(_today)
    _regime_changed  = getattr(_snap, "regime_changed", False) if _snap else False
    _regime_now      = getattr(_snap, "regime",         "")    if _snap else ""
    _regime_prev_mb  = getattr(_snap, "regime_prev",    "")    if _snap else ""
    _narrative_mb    = getattr(_snap, "narrative",      None)  if _snap else None

    _context_items_mb = []
    if _regime_changed and _regime_prev_mb:
        _context_items_mb.append(
            f'<span style="color:#f59e0b;font-weight:700;">制度切换  '
            f'{_regime_prev_mb} → {_regime_now}</span>'
            f'<span style="color:rgba(255,255,255,0.45);"> — 查看 Macro Brief Tab 了解宏观背景</span>'
        )
    if _narrative_mb and not _regime_changed:
        _context_items_mb.append(
            f'<span style="color:rgba(255,255,255,0.6);">{_narrative_mb[:90]}…</span>'
        )

    if _context_items_mb:
        _is_dark_mb = theme.is_dark()
        _bg_mb = "rgba(96,165,250,0.05)" if _is_dark_mb else "rgba(96,165,250,0.08)"
        _border_mb = "rgba(96,165,250,0.3)"
        _html_items = "  ·  ".join(_context_items_mb)
        st.markdown(
            f'<div style="background:{_bg_mb};border:1px solid {_border_mb};'
            f'border-radius:5px;padding:0.55rem 1rem;margin-bottom:0.9rem;font-size:0.82rem;">'
            f'<span style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;'
            f'color:rgba(255,255,255,0.4);margin-right:0.6rem;">Daily Brief →</span>'
            f'{_html_items}</div>',
            unsafe_allow_html=True,
        )
except Exception:
    pass

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.2rem;">
  <div style="font-size:1.55rem; font-weight:800; letter-spacing:0.02em;">
    Macro Brief
  </div>
  <div style="font-size:0.88rem; color:var(--text-muted); margin-top:0.2rem;
              text-transform:uppercase; letter-spacing:0.1em;">
    Intelligence Hub · News · Sectors · Quant Audit
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_macro, tab_scan, tab_audit = st.tabs([
    "📰  Macro Brief",
    "🔭  Sector Scan",
    "🔬  Quant Audit",
])

with tab_macro:
    tabs.render_tab1(vix_input)

with tab_scan:
    tabs.render_tab5(vix_input)

with tab_audit:
    if agent_executor is None:
        st.warning("Agent executor not initialised — return to main app to reload.")
    else:
        tabs.render_tab4(agent_executor, vix_input)
