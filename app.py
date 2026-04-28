import streamlit as st
import streamlit.components.v1 as components
from engine.agent import build_agent_graph
from engine.key_pool import get_pool, AllKeysExhausted
from engine.memory import init_db
from engine.universe_manager import init_universe_db, seed_batch_a, seed_batch_b, seed_batch_c
from engine.quant import QuantEngine
from engine.daily_batch import ensure_daily_batch_completed
import ui.tabs as tabs
import ui.theme as theme

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first; applies globally via st.navigation()
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Macro Alpha Pro",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global design system
# ─────────────────────────────────────────────────────────────────────────────
theme.init_theme()
st.markdown(theme.build_css(admin=True), unsafe_allow_html=True)

# Navigation typography — section headers vs page links must read differently
st.markdown("""
<style>
/* Page links: monospace, normal weight, slight indent */
[data-testid="stSidebarNavLink"] p {
    font-family: 'Courier New', 'JetBrains Mono', monospace !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.025em !important;
}

/* Section group headers: tiny, uppercase, muted, wide-tracked — clearly NOT clickable */
section[data-testid="stSidebar"]
  [data-testid="stSidebarNav"]
  li:not(:has([data-testid="stSidebarNavLink"])) span {
    font-family: -apple-system, 'Segoe UI', sans-serif !important;
    font-size: 0.60rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: rgba(140, 155, 175, 0.65) !important;
}

/* Extra top padding before each section header for breathing room */
section[data-testid="stSidebar"]
  [data-testid="stSidebarNav"]
  li:not(:has([data-testid="stSidebarNavLink"])) {
    padding-top: 0.7rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Terminal Header Bar  (real-time JS clock via components.html)
# ─────────────────────────────────────────────────────────────────────────────
_hdr_is_dark   = theme.is_dark()
_hdr_border    = "rgba(255,255,255,0.15)" if _hdr_is_dark else "#0B1F3A"
_hdr_logo      = "#F0F6FC"   if _hdr_is_dark else "#0B1F3A"
_hdr_sub       = "#8B949E"   if _hdr_is_dark else "#64748B"
_hdr_clk       = "#8B949E"   if _hdr_is_dark else "#475569"
components.html(f"""
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:transparent;
       font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Helvetica Neue',sans-serif; }}
.hdr {{ display:flex; align-items:center; justify-content:space-between;
       padding:0.55rem 0; border-bottom:2.5px solid {_hdr_border}; }}
</style>
<div class="hdr">
  <div style="display:flex;align-items:baseline;gap:1rem;">
    <span style="font-size:2.1rem;font-weight:900;color:{_hdr_logo};letter-spacing:0.03em;">
      MACRO ALPHA
    </span>
    <span style="font-size:1.05rem;color:{_hdr_sub};text-transform:uppercase;
                 letter-spacing:0.13em;font-weight:600;">
      Pro Terminal&nbsp;&middot;&nbsp;NUS MSBA
    </span>
  </div>
  <span id="clk" style="font-family:'Courier New',monospace;font-size:1.05rem;
                         color:{_hdr_clk};letter-spacing:0.05em;"></span>
</div>
<script>
function updateClock(){{
  var n=new Date(), p=function(x){{return('0'+x).slice(-2);}};
  document.getElementById('clk').textContent=
    n.getFullYear()+'-'+p(n.getMonth()+1)+'-'+p(n.getDate())+
    '  '+p(n.getHours())+':'+p(n.getMinutes())+':'+p(n.getSeconds())+'  SGT';
}}
updateClock();
setInterval(updateClock, 1000);
</script>
""", height=62, scrolling=False)

# Theme toggle
_, _tcol2 = st.columns([7, 1])
with _tcol2:
    theme.render_toggle(key="__theme_toggle_app__")

# ─────────────────────────────────────────────────────────────────────────────
# Init: Secrets → Gemini → Session State → Agent
# ─────────────────────────────────────────────────────────────────────────────
try:
    _pool = get_pool()
    model = _pool.get_model()
except AllKeysExhausted:
    st.error(
        "⛔ 所有 Gemini API Key 已耗尽今日配额。\n\n"
        "请前往 **Key Pool Manager** 页面添加新 Key，或等待明天配额自动重置。"
    )
    st.stop()
except Exception as e:
    st.error(
        f"❌ Gemini Key 池初始化失败：{e}\n\n"
        "请检查 `.streamlit/secrets.toml` 中的 `GEMINI_KEY` 或 `[GEMINI_POOL]` 配置。"
    )
    st.stop()
tabs.set_model(model)
init_db()
init_universe_db()
seed_batch_a()
seed_batch_b()
seed_batch_c()
tabs.restore_today_from_db()
ensure_daily_batch_completed(model=model)

if "dynamic_assets" not in st.session_state:
    st.session_state.dynamic_assets = {
        "新加坡蓝筹": ["DBSDF", "U11.SI", "V03.SI"],
        "科技成长":   ["NVDA", "AAPL", "MSFT"],
    }

agent_executor = build_agent_graph(
    model,
    st.session_state.dynamic_assets,
    av_key=st.secrets.get("AV_KEY", ""),
    gnews_key=st.secrets.get("GNEWS_KEY", ""),
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — live system status + market data
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def _sidebar_system_status():
    """Fetch live system status for sidebar. Cached 60s."""
    import datetime as _dt
    status = {
        "regime": "—", "p_risk_on": None, "regime_color": "#8b949e",
        "limits": "—", "n_pending": 0, "cb_level": "none", "cb_color": "#34D399",
        "dq_status": "ok", "dq_color": "#34D399", "n_universe": 0,
    }
    try:
        from engine.memory import get_daily_brief_snapshot as _gds
        _snap = _gds(_dt.date.today())
        if _snap and _snap.regime:
            status["regime"] = _snap.regime.upper()
            status["p_risk_on"] = float(_snap.p_risk_on or 0.0)
            _regime_colors = {"RISK-ON": "#34D399", "TRANSITION": "#FBBF24", "RISK-OFF": "#F87171"}
            status["regime_color"] = _regime_colors.get(status["regime"], "#8b949e")
            _limits_map = {"RISK-ON": "10L / 6S", "TRANSITION": "7L / 7S", "RISK-OFF": "5L / 8S"}
            status["limits"] = _limits_map.get(status["regime"], "—")
    except Exception:
        pass
    try:
        from engine.memory import get_pending_approvals_by_priority as _gap
        status["n_pending"] = len(_gap())
    except Exception:
        pass
    try:
        from engine.circuit_breaker import get_status as _cbs
        _cb = _cbs()
        status["cb_level"] = str(_cb.level).upper() if _cb else "NONE"
        status["cb_color"] = "#F87171" if status["cb_level"] == "SEVERE" \
            else "#FBBF24" if status["cb_level"] == "MEDIUM" else "#34D399"
    except Exception:
        pass
    try:
        from engine.memory import DataQualityLog, SessionFactory as _SF
        import datetime as _dt2
        with _SF() as _s:
            _dq = (_s.query(DataQualityLog)
                    .filter(DataQualityLog.date == _dt2.date.today(),
                            DataQualityLog.check_type == "overall")
                    .order_by(DataQualityLog.checked_at.desc()).first())
        if _dq:
            _dq_map = {"ok": ("#34D399", "ok"), "warning": ("#FBBF24", "warn"), "light": ("#F87171", "stale")}
            status["dq_color"], status["dq_status"] = _dq_map.get(_dq.status, ("#8b949e", "—"))
    except Exception:
        pass
    try:
        from engine.history import get_active_sector_etf as _gu
        status["n_universe"] = len(_gu())
    except Exception:
        pass
    return status

_sys = _sidebar_system_status()

with st.sidebar:
    # ── System Status (live) ──────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:0.58rem;text-transform:uppercase;letter-spacing:0.15em;'
        'color:rgba(255,255,255,0.35);padding:0.2rem 0 0.5rem;">System Status</div>',
        unsafe_allow_html=True,
    )

    def _sb_row(label: str, value: str, color: str, sub: str = "") -> str:
        return (
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
            f'padding:0.18rem 0;border-bottom:1px solid rgba(255,255,255,0.05);">'
            f'<span style="font-size:0.72rem;color:rgba(255,255,255,0.4);'
            f'text-transform:uppercase;letter-spacing:0.06em;">{label}</span>'
            f'<span style="font-size:0.78rem;font-weight:700;font-family:\'Courier New\',monospace;'
            f'color:{color};">{value}</span></div>'
            + (f'<div style="font-size:0.65rem;color:rgba(255,255,255,0.28);'
               f'padding:0.05rem 0 0.12rem;text-align:right;">{sub}</div>' if sub else "")
        )

    _regime_display = _sys["regime"]
    if _sys["p_risk_on"] is not None:
        _regime_display += f"  {_sys['p_risk_on']:.0%}"
    _pending_color = "#F87171" if _sys["n_pending"] > 0 else "#34D399"
    _pending_val   = f"⚡ {_sys['n_pending']}" if _sys["n_pending"] > 0 else "✓ CLEAR"

    st.markdown(
        _sb_row("Regime",   _regime_display,  _sys["regime_color"], f"Limits: {_sys['limits']}")
        + _sb_row("Pending", _pending_val,     _pending_color)
        + _sb_row("CB",      _sys["cb_level"] or "NONE", _sys["cb_color"])
        + _sb_row("Data",    _sys["dq_status"].upper(), _sys["dq_color"])
        + _sb_row("Universe", f"{_sys['n_universe']} sectors", "#8b949e"),
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:0.6rem;"></div>', unsafe_allow_html=True)
    st.divider()

    # ── Market ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:0.58rem;text-transform:uppercase;letter-spacing:0.15em;'
        'color:rgba(255,255,255,0.35);padding:0.2rem 0 0.5rem;">Market</div>',
        unsafe_allow_html=True,
    )
    @st.cache_data(ttl=300, show_spinner=False)
    def _get_vix_cached() -> float:
        return QuantEngine.get_realtime_vix()
    real_vix = _get_vix_cached()
    if real_vix < 15:
        vix_status, vix_color = "COMPLACENCY", "#34D399"
    elif real_vix < 25:
        vix_status, vix_color = "NORMAL", "#60A5FA"
    elif real_vix < 35:
        vix_status, vix_color = "ELEVATED", "#FBBF24"
    else:
        vix_status, vix_color = "CRISIS", "#F87171"

    st.markdown(
        _sb_row("VIX", f"{real_vix:.1f}  {vix_status}", vix_color),
        unsafe_allow_html=True,
    )
    vix_input = st.number_input(
        "Stress VIX Override",
        value=real_vix,
        help="手动调整 VIX 以模拟压力情景（仅影响 Macro Intel 页面）",
        label_visibility="visible",
    )

    st.divider()

    # ── Data sources ──────────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:0.65rem;color:rgba(255,255,255,0.28);line-height:1.8;">'
        'Data  CBOE · Yahoo Finance · FRED<br>'
        'AI  Gemini 2.5 Flash · LangGraph<br>'
        'Mode  Audit · Human-in-loop'
        '</div>',
        unsafe_allow_html=True,
    )

# Store shared state for pages
st.session_state["_vix_input"]       = vix_input
st.session_state["_agent_executor"]  = agent_executor

# ─────────────────────────────────────────────────────────────────────────────
# Navigation — workflow-first, terminal-style
# ─────────────────────────────────────────────────────────────────────────────
pg = st.navigation(
    {
        "今日": [
            st.Page("pages/orchestrator.py",   title="Operations",  default=True),
            st.Page("pages/live_dashboard.py", title="Positions"),
        ],
        "分析": [
            st.Page("pages/signal_board.py", title="Signals"),
            st.Page("pages/macro_brief.py",  title="Macro Intel"),
        ],
        "研究": [
            st.Page("pages/backtest.py",         title="Backtest"),
            st.Page("pages/factor_dashboard.py", title="Factor Lab"),
        ],
        "绩效": [
            st.Page("pages/clean_zone.py",       title="Performance"),
            st.Page("pages/decision_journal.py", title="Decision Log"),
        ],
        "系统": [
            st.Page("pages/circuit_breaker.py", title="Circuit Breaker"),
            st.Page("pages/key_manager.py",     title="Key Pool"),
        ],
    },
    position="sidebar",
)
pg.run()
