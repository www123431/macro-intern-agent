"""
Macro Alpha Pro — Daily Brief
==============================
Supervisor 每日工作台。

设计原则：编排层是基础设施，不是界面。
用户看到的是决策，而不是管道。

三件事：
  A. 需要我处理什么   (优先级审批队列)
  B. 今天发生了什么   (信号变化、风控告警)
  C. 上一批决策怎样了 (验证摘要)

管道运行细节收折在 Engineering 区域，默认不可见。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime
import streamlit as st
import pandas as pd

import ui.theme as theme
from engine.memory import (
    init_db,
    get_pending_decisions_for_monitor,
    get_pending_approvals_by_priority,
    get_recent_verifications,
    get_daily_brief_snapshot,
    resolve_pending_approval,
)
from engine.daily_batch import ensure_daily_batch_completed
from engine.orchestrator import (
    TradingCycleOrchestrator,
    run_daily_chain,
    GATE_ANALYSIS_DRAFT, GATE_RISK_APPROVAL,
    GATE_MONTHLY_REBALANCE, GATE_COVARIANCE_OVERRIDE,
    ChainResult,
)

init_db()
theme.init_theme()

today  = datetime.date.today()
_is_dark = theme.is_dark()
_C = {
    "green":  "#22c55e",
    "red":    "#ef4444",
    "yellow": "#f59e0b",
    "blue":   "#60a5fa",
    "purple": "#a78bfa",
    "muted":  "#8b949e" if _is_dark else "#64748b",
    "border": "rgba(255,255,255,0.08)" if _is_dark else "#e2e8f0",
    "card":   "#1e293b" if _is_dark else "#f8fafc",
    "text":   "#f0f6fc" if _is_dark else "#0f172a",
    "mono":   "'Courier New','JetBrains Mono',monospace",
}

_GATE_LABELS = {
    GATE_ANALYSIS_DRAFT:      "分析草稿审批",
    GATE_RISK_APPROVAL:       "风控建议审批",
    GATE_MONTHLY_REBALANCE:   "月度再平衡审批",
    GATE_COVARIANCE_OVERRIDE: "协方差覆盖审批",
}
_STATUS_ICONS = {
    "completed":    ("✅", _C["green"]),
    "failed":       ("❌", _C["red"]),
    "running":      ("⏳", _C["yellow"]),
    "pending_gate": ("🔒", _C["blue"]),
    "approved":     ("✅", _C["green"]),
    "rejected":     ("🚫", _C["red"]),
    "skipped":      ("⏭", _C["muted"]),
}
_PRIORITY_STYLE = {
    "urgent":   (_C["red"],    "URGENT"),
    "critical": (_C["red"],    "URGENT"),
    "high":     (_C["yellow"], "HIGH"),
    "normal":   (_C["blue"],   "NORMAL"),
    "low":      (_C["muted"],  "LOW"),
}
_APPROVAL_TYPE_LABEL = {
    "entry":           "入场触发",
    "risk_control":    "风控告警",
    "rebalance":       "再平衡",
    "track_b":         "Track B 叠加",
    "signal_decay":    "信号衰减",
    "vol_spike_p6":    "波动率尖峰",
    "factor_candidate":"因子候选",
    "universe_change": "标的调整",
}
_VERDICT_STYLE = {
    "WIN":    (_C["green"],  "✅"),
    "LOSS":   (_C["red"],    "❌"),
    "REVIEW": (_C["yellow"], "⚠"),
}

tco = TradingCycleOrchestrator()


# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def _load_regime():
    try:
        from engine.regime import get_regime_on
        return get_regime_on(as_of=today, train_end=today)
    except Exception:
        return None

@st.cache_data(ttl=30)
def _load_batch():
    try:
        from ui.tabs import get_model as _get_model
        _m = _get_model()
    except Exception:
        _m = None
    return ensure_daily_batch_completed(model=_m)

@st.cache_data(ttl=60)
def _load_brief_snapshot() -> dict | None:
    try:
        snap = get_daily_brief_snapshot(today)
        if snap is None:
            return None
        return {
            "narrative":             snap.narrative,
            "macro_brief_llm":       getattr(snap, "macro_brief_llm", None),
            "regime":                snap.regime,
            "regime_prev":           snap.regime_prev,
            "regime_changed":        bool(snap.regime_changed),
            "p_risk_on":             float(snap.p_risk_on or 0.0),
            "icir_month":            snap.icir_month,
            "n_verified_today":      int(snap.n_verified_today or 0),
            "verify_ran":            bool(snap.verify_ran),
            "tactical_entries_json": getattr(snap, "tactical_entries_json", None),
            "tactical_reduces_json": getattr(snap, "tactical_reduces_json", None),
            "regime_jump_today":     bool(getattr(snap, "regime_jump_today", False)),
        }
    except Exception:
        return None

@st.cache_data(ttl=30)
def _load_pending_approvals():
    try:
        return get_pending_approvals_by_priority()
    except Exception:
        return []

@st.cache_data(ttl=120)
def _load_recent_verifications():
    try:
        return get_recent_verifications(n=6)
    except Exception:
        return []

@st.cache_data(ttl=300)
def _load_signal_context(ticker: str) -> dict:
    """Load latest SignalRecord for a ticker — composite score, TSMOM signal, flip."""
    ctx: dict = {}
    try:
        from engine.memory import SignalRecord, SessionFactory
        with SessionFactory() as s:
            row = (
                s.query(SignalRecord)
                 .filter(SignalRecord.ticker == ticker)
                 .order_by(SignalRecord.date.desc())
                 .first()
            )
        if row:
            ctx["composite"]  = float(row.composite_score) if row.composite_score is not None else None
            ctx["tsmom"]      = int(row.tsmom_signal)      if row.tsmom_signal is not None else None
            ctx["tsmom_raw"]  = float(row.tsmom_raw)       if getattr(row, "tsmom_raw", None) is not None else None
            ctx["signal_date"]= str(row.date)
            ctx["flipped"]    = bool(getattr(row, "flip_today", False))
            ctx["decay_pct"]  = float(getattr(row, "decay_pct", 0) or 0)
    except Exception:
        pass
    return ctx


def _generate_ops_brief(model, regime_str: str, p_risk_on: float,
                         batch_obj, pending_n: int, events: list[str],
                         vix: float) -> str:
    """Call Gemini to synthesize today's operations context into a human-readable brief."""
    if model is None:
        return ""
    _regime_cn = {"risk-on": "风险偏好", "risk-off": "风险规避", "transition": "过渡"}.get(
        regime_str, regime_str)
    _events_txt = "\n".join(f"- {e}" for e in events[:8]) if events else "- 今日无触发事件"
    _batch_ok = (getattr(batch_obj, "signal_ok", False) and getattr(batch_obj, "regime_ok", False)) \
                if batch_obj else False
    _errs = (getattr(batch_obj, "errors", []) or [])[:3] if batch_obj else []
    _n_flip = len(getattr(batch_obj, "invalidations", []) or []) if batch_obj else 0
    _n_entry = len(getattr(batch_obj, "entries_triggered", []) or []) if batch_obj else 0
    prompt = f"""你是宏观量化策略系统的 Operations 智能助手。请用**100-180字中文**生成今日操盘台备忘录。

## 当日数据（结构化，不要在输出中列举原始数据）
- 宏观制度: {_regime_cn}，P(risk-on)={p_risk_on:.0%}
- VIX: {vix:.1f}
- 今日批次: {"正常" if _batch_ok else "异常 — " + "; ".join(_errs)}
- 待审批: {pending_n} 项
- 信号翻转: {_n_flip} 个，入场候选: {_n_entry} 个
- 今日事件:\n{_events_txt}

## 输出要求
1. 第一句：一句话点明当前制度+风险环境（必须说清楚是否需要收紧敞口）
2. 第二句：今日最值得关注的信号变化（如无则说平稳）
3. 第三句：对 Supervisor 的行动建议（审批优先级、是否需要提前复查持仓）
4. 语气：交易台备忘录风格，专业简洁，不要废话，不要重复列举原始数字

只输出正文，不要标题、不要bullet points。"""
    try:
        return model.generate_content(prompt).text.strip()
    except Exception as e:
        return f"（LLM生成失败: {e}）"


regime            = _load_regime()
batch             = _load_batch()
brief_snap        = _load_brief_snapshot()
pending_approvals = _load_pending_approvals()
@st.cache_data(ttl=30)
def _load_pending_gates():
    return tco.get_pending_gates()

@st.cache_data(ttl=60)
def _load_pending_decisions():
    try:
        return get_pending_decisions_for_monitor()
    except Exception:
        return []

pending_gates     = _load_pending_gates()
recent_verifs     = _load_recent_verifications()
pending_decisions = _load_pending_decisions()
n_overdue         = sum(1 for d in pending_decisions if d["urgency"] == "overdue")
n_approaching     = sum(1 for d in pending_decisions if d["urgency"] == "approaching")

# Total pending actions = CycleState gates + PendingApproval items
n_actions = len(pending_gates) + len(pending_approvals)

# ── Regime fallback chain: live HMM → cached snapshot → VIX rule ─────────────
_live_regime_str  = getattr(regime, "regime", None)
_live_p_risk_on   = getattr(regime, "p_risk_on", None)
_snap_regime_str  = (brief_snap or {}).get("regime", "") or ""
_snap_p_risk_on   = (brief_snap or {}).get("p_risk_on", 0.0)

if _live_regime_str and _live_regime_str not in ("", "unknown"):
    _eff_regime   = _live_regime_str
    _eff_p        = float(_live_p_risk_on or 0.0)
    _regime_src   = ""                       # live — no annotation needed
elif _snap_regime_str and _snap_regime_str not in ("", "unknown"):
    _eff_regime   = _snap_regime_str
    _eff_p        = float(_snap_p_risk_on or 0.0)
    _regime_src   = " (cached)"
else:
    # VIX rule-of-thumb: <15 risk-on, >25 risk-off
    try:
        from engine.quant import QuantEngine as _QE
        _vix_fb = _QE.get_realtime_vix()
        if _vix_fb < 15:
            _eff_regime, _eff_p = "risk-on",    0.80
        elif _vix_fb > 25:
            _eff_regime, _eff_p = "risk-off",   0.20
        else:
            _eff_regime, _eff_p = "transition", 0.50
        _regime_src = f" (VIX={_vix_fb:.0f})"
    except Exception:
        _eff_regime, _eff_p, _regime_src = "transition", 0.50, " (fallback)"

_regime_color = {
    "risk-on":    _C["green"],
    "risk-off":   _C["red"],
    "transition": _C["yellow"],
}.get(_eff_regime, _C["muted"])
_regime_label = f"{_eff_regime.upper()}  P={_eff_p:.0%}{_regime_src}"

try:
    from engine.circuit_breaker import get_status as _cb_get, LEVEL_SEVERE, LEVEL_MEDIUM
    _cb_state = _cb_get()
    _cb_active = _cb_state.level > 0
    _cb_severe = _cb_state.level >= LEVEL_SEVERE
except Exception:
    _cb_state  = None
    _cb_active = False
    _cb_severe = False

# ══════════════════════════════════════════════════════════════════════════════
# STATUS STRIP — compact 5-card bar, always visible, instant
# ══════════════════════════════════════════════════════════════════════════════
_vix_now     = st.session_state.get("_vix_input", 20.0)
_batch_ok    = (batch.signal_ok and batch.regime_ok) if batch else False
_actions_color = _C["red"] if n_actions > 0 else _C["green"]
_cb_label    = "NONE" if not _cb_active else ("SEVERE" if _cb_severe else "MEDIUM")
_cb_color    = _C["muted"] if not _cb_active else (_C["red"] if _cb_severe else _C["yellow"])
_vix_status  = "COMPLACENCY" if _vix_now < 15 else ("NORMAL" if _vix_now < 25 else ("ELEVATED" if _vix_now < 35 else "CRISIS"))
_vix_color   = {"COMPLACENCY": _C["green"], "NORMAL": _C["blue"], "ELEVATED": _C["yellow"], "CRISIS": _C["red"]}.get(_vix_status, _C["muted"])

def _stat_card(label: str, value: str, color: str, sub: str = "") -> str:
    return (
        f'<div style="flex:1;padding:0.55rem 0.9rem;background:{_C["card"]};'
        f'border:1px solid {_C["border"]};border-top:3px solid {color};border-radius:6px;">'
        f'<div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.12em;'
        f'color:{_C["muted"]};margin-bottom:0.18rem;">{label}</div>'
        f'<div style="font-size:0.95rem;font-weight:800;font-family:{_C["mono"]};color:{color};">{value}</div>'
        + (f'<div style="font-size:0.62rem;color:{_C["muted"]};margin-top:0.1rem;">{sub}</div>' if sub else "")
        + '</div>'
    )

st.markdown(
    f'<div style="font-family:{_C["mono"]};font-size:1.4rem;font-weight:900;'
    f'letter-spacing:0.04em;margin-bottom:0.7rem;">'
    f'DAILY BRIEF'
    f'<span style="font-size:0.75rem;font-weight:400;color:{_C["muted"]};'
    f'letter-spacing:0.08em;margin-left:1.2rem;">{today}</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="display:flex;gap:0.6rem;margin-bottom:0.9rem;">'
    + _stat_card("Macro Regime", _regime_label.split("(")[0].strip(), _regime_color,
                  f"P(risk-on)={_eff_p:.0%}")
    + _stat_card("待处理", f"{'⚡ '+str(n_actions) if n_actions else '✅ 无'}", _actions_color,
                  f"Gate {len(pending_gates)} · Alert {len(pending_approvals)}")
    + _stat_card("熔断器", _cb_label, _cb_color,
                  getattr(_cb_state, "reason", "")[:30] if _cb_active and _cb_state else "系统健康")
    + _stat_card("今日批次", "✅ 就绪" if _batch_ok else "⚠ 异常", _C["green"] if _batch_ok else _C["yellow"],
                  f"信号+制度已更新" if _batch_ok else "检查系统日志")
    + _stat_card("VIX", f"{_vix_now:.1f}  {_vix_status}", _vix_color, "实时")
    + '</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div style="border-bottom:2px solid {_C["border"]};margin:0 0 1rem;"></div>',
    unsafe_allow_html=True,
)

# Circuit Breaker alert banner (only visible when active)
if _cb_active and _cb_state:
    _cb_border = _C["red"] if _cb_severe else _C["yellow"]
    _cb_bg     = "rgba(239,68,68,0.08)" if _cb_severe else "rgba(245,158,11,0.08)"
    st.markdown(
        f'<div style="background:{_cb_bg};border:1.5px solid {_cb_border};'
        f'border-radius:6px;padding:0.7rem 1.2rem;margin-bottom:1rem;">'
        f'<b style="color:{_cb_border};">{"🔴 CIRCUIT BREAKER SEVERE" if _cb_severe else "🟡 CIRCUIT BREAKER MEDIUM"}</b>'
        f'  —  {getattr(_cb_state, "reason", "已触发")}</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# LLM BRIEF + NAVIGATION — hero element, placed BEFORE pipeline detail
# ══════════════════════════════════════════════════════════════════════════════

# ── Build event context for the prompt ───────────────────────────────────────
_today_events = []
_entry_sectors = list(getattr(batch, "entries_triggered", []) or []) if batch else []
_stop_sectors  = [a.split(":")[0] for a in (getattr(batch, "risk_alerts", []) or [])
                  if any(k in a for k in ("hard_stop", "drawdown_stop"))] if batch else []
_flip_sectors  = list(getattr(batch, "invalidations", []) or []) if batch else []
_tact_entries  = list(getattr(batch, "tactical_entries", []) or []) if batch else []
for s in _entry_sectors:  _today_events.append(f"入场候选:{s}")
for s in _stop_sectors:   _today_events.append(f"止损告警:{s}")
for s in _flip_sectors:   _today_events.append(f"信号翻转:{s}")
for s in _tact_entries:   _today_events.append(f"战术入场:{s}")

_ops_cache_key = f"ops_brief_{today}_{_eff_regime}_{len(_today_events)}_{len(pending_approvals)}"

_col_brief, _col_btn = st.columns([5, 1])
with _col_btn:
    _gen_btn = st.button(
        "🤖 生成简报", key="gen_ops_brief", use_container_width=True,
        help="调用 Gemini 生成今日智能操盘简报（~5秒）",
    )

_ops_brief_text = st.session_state.get(_ops_cache_key, "")
_llm_brief_stored = (brief_snap or {}).get("macro_brief_llm")

if _gen_btn or (not _ops_brief_text and not _llm_brief_stored):
    _model_obj = None
    try:
        from engine.key_pool import get_pool as _gp
        _model_obj = _gp().get_model()
    except Exception:
        pass
    if _model_obj is not None:
        with st.spinner("Gemini 正在生成今日操盘简报…"):
            _ops_brief_text = _generate_ops_brief(
                _model_obj, _eff_regime, _eff_p, batch,
                len(pending_approvals) + len(pending_gates),
                _today_events, _vix_now,
            )
        if _ops_brief_text and "失败" not in _ops_brief_text:
            st.session_state[_ops_cache_key] = _ops_brief_text

_display_text  = _ops_brief_text or _llm_brief_stored
_bs            = (brief_snap or {})
_regime_changed = _bs.get("regime_changed", False)

if _display_text:
    _narr_border = _C["yellow"] if _regime_changed else _C["purple"]
    _narr_bg     = "rgba(245,158,11,0.04)" if _regime_changed else "rgba(167,139,250,0.04)"
    _src_label   = "Gemini · 实时生成" if _ops_brief_text else "Gemini · 批次生成"
    with _col_brief:
        st.markdown(
            f'<div style="background:{_narr_bg};border:1.5px solid {_narr_border};'
            f'border-radius:6px;padding:0.9rem 1.3rem;margin-bottom:0.7rem;">'
            f'<div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;'
            f'color:{_C["purple"]};margin-bottom:0.4rem;">{_src_label}</div>'
            + (f'<div style="font-size:0.72rem;color:{_C["yellow"]};font-weight:700;'
               f'margin-bottom:0.3rem;">⚠ 制度切换  '
               f'{_bs.get("regime_prev","—")} → {_bs.get("regime","—")}</div>'
               if _regime_changed and _bs.get("regime_prev") else "")
            + f'<div style="font-size:0.97rem;color:{_C["text"]};line-height:1.8;'
            f'letter-spacing:0.01em;">{_display_text}</div>'
            + '</div>',
            unsafe_allow_html=True,
        )
else:
    with _col_brief:
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.02);border:1px solid {_C["border"]};'
            f'border-radius:6px;padding:0.75rem 1.1rem;margin-bottom:0.7rem;">'
            f'<div style="font-size:0.68rem;color:{_C["muted"]};margin-bottom:0.25rem;">'
            f'今日简报  ·  点击右侧「🤖 生成简报」调用 Gemini 生成智能操盘叙述</div>'
            f'<div style="font-size:0.9rem;color:{_C["muted"]};">'
            f'{"信号与制度已就绪  · " if _batch_ok else "批次异常  · "}'
            f'{"无待处理事项" if not n_actions else f"{n_actions} 项待处理（见下方审批队列）"}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

# ── Context-aware navigation links ────────────────────────────────────────────
_nav_cols = st.columns(5)
with _nav_cols[0]:
    if n_actions > 0:
        st.markdown(
            f'<div style="background:rgba(239,68,68,0.08);border:1px solid {_C["red"]}44;'
            f'border-radius:5px;padding:0.45rem 0.7rem;text-align:center;font-size:0.8rem;">'
            f'<span style="color:{_C["red"]};font-weight:700;">⚡ {n_actions} 待审批</span><br>'
            f'<span style="color:{_C["muted"]};font-size:0.7rem;">↓ 见下方待处理事项</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="background:rgba(34,197,94,0.06);border:1px solid {_C["green"]}33;'
            f'border-radius:5px;padding:0.45rem 0.7rem;text-align:center;font-size:0.8rem;">'
            f'<span style="color:{_C["green"]};font-weight:700;">✅ 队列清空</span><br>'
            f'<span style="color:{_C["muted"]};font-size:0.7rem;">无待审批事项</span></div>',
            unsafe_allow_html=True,
        )
with _nav_cols[1]:
    st.page_link("pages/signal_board.py",   label="📊 Signal Board",  use_container_width=True)
with _nav_cols[2]:
    st.page_link("pages/live_dashboard.py", label="📈 Positions",      use_container_width=True)
with _nav_cols[3]:
    st.page_link("pages/decision_journal.py", label="📓 Decision Log", use_container_width=True)
with _nav_cols[4]:
    if _cb_active:
        st.page_link("pages/circuit_breaker.py", label="🔴 Circuit Breaker", use_container_width=True)
    else:
        st.page_link("pages/macro_brief.py", label="🌍 Macro Intel",   use_container_width=True)

st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE TIMELINE — 自动化执行状态（折叠，正常运行无需展开）
# ══════════════════════════════════════════════════════════════════════════════
_pipeline_label = "⚙ 自动化流水线"
if batch and batch.errors:
    _pipeline_label += f"  ⚠ {len(batch.errors)} 个步骤异常"
elif _batch_ok:
    _pipeline_label += "  ✅ 今日执行正常"
with st.expander(_pipeline_label, expanded=bool(batch and batch.errors)):
  st.markdown(
    f'<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;'
    f'color:{_C["muted"]};margin-bottom:0.5rem;">Pipeline  ·  今日自动化执行状态</div>',
    unsafe_allow_html=True,
  )

def _pl_row(step: int, title: str, status: str, summary: str,
            color: str, detail_key: str | None = None, detail_body: str = "") -> None:
    """Render one pipeline step row with optional expandable detail."""
    _icon = {"ok": "✅", "warn": "⚠️", "error": "❌", "skip": "⏭", "block": "🔴"}.get(status, "○")
    _row_color = {
        "ok": _C["green"], "warn": _C["yellow"],
        "error": _C["red"], "skip": _C["muted"], "block": _C["red"],
    }.get(status, _C["muted"])
    st.markdown(
        f'<div style="display:flex;align-items:baseline;padding:0.22rem 0.5rem;'
        f'border-left:3px solid {_row_color};margin-bottom:0.2rem;'
        f'background:rgba(0,0,0,0.04);border-radius:0 4px 4px 0;">'
        f'<span style="font-size:0.78rem;color:{_C["muted"]};width:1.4rem;flex-shrink:0;">{step}</span>'
        f'<span style="font-size:0.82rem;font-weight:700;color:{_C["text"]};width:9rem;flex-shrink:0;">'
        f'{_icon} {title}</span>'
        f'<span style="font-size:0.8rem;color:{_row_color};">{summary}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if detail_key and detail_body:
        with st.expander("展开详情", expanded=False):
            st.markdown(detail_body, unsafe_allow_html=True)

# Step 0: Circuit Breaker
_cb_snap = getattr(batch, "cb_level", "none") if batch else "none"
_cb_rsn  = getattr(batch, "cb_reason", "") if batch else ""
if _cb_state and _cb_state.level != "none":
    _pl_row(0, "Circuit Breaker",
            "block" if _cb_state.level == "severe" else "warn",
            f"{_cb_state.level.upper()} — {_cb_rsn or _cb_state.reason}",
            _C["red"],
            "cb_detail",
            f"VIX今日: {_cb_state.vix_today}  ·  VIX昨日: {_cb_state.vix_prev}"
            f"  ·  API配额: {(_cb_state.quota_frac or 0):.0%}")
else:
    _pl_row(0, "Circuit Breaker", "ok", "NONE — 系统健康", _C["green"])

# Step 1: Signal + Regime
if batch and batch.signal_ok and batch.regime_ok:
    _regime_str  = getattr(regime, "regime", "—").upper()
    _p_str       = f"P={getattr(regime, 'p_risk_on', 0):.0%}"
    _flip_count  = len(batch.invalidations)
    _entry_count = len(batch.entries_triggered)
    _sig_summary = f"{_regime_str} {_p_str}  ·  信号翻转 {_flip_count}  ·  入场候选 {_entry_count}"
    _pl_row(1, "信号 + 制度", "ok", _sig_summary, _C["green"],
            "sig_detail",
            f"制度变化: {'是' if brief_snap and brief_snap.get('regime_changed') else '否'}  "
            f"·  上期制度: {brief_snap.get('regime_prev','—') if brief_snap else '—'}")
elif batch and batch.errors:
    _pl_row(1, "信号 + 制度", "error",
            f"失败 — {batch.errors[0][:60]}", _C["red"])
else:
    _pl_row(1, "信号 + 制度", "skip", "尚未运行", _C["muted"])

# Step 2: FinDebate
_debate_n = len(getattr(batch, "debate_sectors", [])) if batch else 0
if _debate_n:
    _debate_secs = "、".join(getattr(batch, "debate_sectors", []))
    _pl_row(2, "FinDebate", "ok",
            f"{_debate_n} 个板块辩论完成：{_debate_secs}", _C["green"],
            "debate_detail",
            "<br>".join(
                f"<b>{s}</b>: {c[:120]}"
                for s, c in (getattr(batch, "debate_results", {}) or {}).items()
            ))
else:
    _pl_row(2, "FinDebate", "skip",
            "无入场候选 / 无制度变化，未触发", _C["muted"])

# Step 3: Tactical Patrol
_tact_e = len(getattr(batch, "tactical_entries", [])) if batch else 0
_tact_r = len(getattr(batch, "tactical_reduces", [])) if batch else 0
_rj     = getattr(batch, "regime_jump", False) if batch else False
if _rj or _tact_e or _tact_r:
    _tact_parts = []
    if _rj:       _tact_parts.append("⚠️ 制度跃变")
    if _tact_e:   _tact_parts.append(f"战术入场 {_tact_e} 笔")
    if _tact_r:   _tact_parts.append(f"战术减仓 {_tact_r} 笔")
    _pl_row(3, "战术巡逻", "warn" if not _rj else "block",
            " · ".join(_tact_parts), _C["yellow"])
else:
    _pl_row(3, "战术巡逻", "ok", "无触发事件", _C["green"])

# Step 4: Portfolio / Rebalance
_rb_auto    = getattr(batch, "rebalance_auto", False) if batch else False
_rb_orders  = len(getattr(batch, "rebalance_orders", [])) if batch else 0
_rb_skip    = getattr(batch, "rebalance_skipped_reason", "") if batch else ""
if _rb_auto:
    _pl_row(4, "组合 / 再平衡", "ok",
            f"自动执行 ✅  ·  {_rb_orders} 个板块调整", _C["green"])
elif _rb_orders:
    _pl_row(4, "组合 / 再平衡", "warn",
            f"Layer-3 待审批：{_rb_orders} 笔  ·  阻塞：{_rb_skip or '月末未到'}",
            _C["yellow"])
else:
    _pl_row(4, "组合 / 再平衡", "skip",
            "非月末 / 无再平衡需求", _C["muted"])

# Step 5: FactorMAD ICIR
_icir_ran    = getattr(batch, "factormad_icir_ran", False) if batch else False
_icir_deact  = getattr(batch, "factormad_deactivated", []) if batch else []
_icir_month  = brief_snap.get("icir_month") if brief_snap else None
if _icir_ran:
    _icir_msg = f"本月 ICIR 已更新 ({today.strftime('%Y-%m')})"
    if _icir_deact:
        _icir_msg += f"  ·  停用因子：{', '.join(_icir_deact)}"
    _pl_row(5, "FactorMAD", "ok" if not _icir_deact else "warn", _icir_msg, _C["green"])
elif _icir_month:
    _pl_row(5, "FactorMAD", "skip", f"本月已更新 ({_icir_month})，今日跳过", _C["muted"])
else:
    _pl_row(5, "FactorMAD", "skip", "本月 ICIR 待更新（月初首个交易日触发）", _C["muted"])

# Step 6: LLM Macro Brief
_brief_ok = bool(brief_snap and (brief_snap.get("macro_brief_llm") or brief_snap.get("narrative")))
_brief_txt = (brief_snap or {}).get("macro_brief_llm") or (brief_snap or {}).get("narrative") or ""
_pl_row(6, "LLM 宏观简报", "ok" if _brief_ok else "skip",
        "已生成" if _brief_ok else "未生成 / LLM 不可用",
        _C["green"] if _brief_ok else _C["muted"],
        "brief_detail", _brief_txt[:400] + ("…" if len(_brief_txt) > 400 else ""))

# Step 7: Track B (monthly M2 LLM overlay)
_tb_pending = sum(1 for pa in pending_approvals if pa.get("approval_type") == "track_b")
_tb_ran = getattr(batch, "track_b_ran", False) if batch else False
if _tb_ran:
    _tb_adj = getattr(batch, "track_b_adjustments", 0) if batch else 0
    _pl_row(7, "Track B", "ok" if not _tb_pending else "warn",
            f"本月已运行  ·  调整 {_tb_adj} 个板块" + (f"  ·  {_tb_pending} 项待批" if _tb_pending else ""),
            _C["green"] if not _tb_pending else _C["yellow"])
else:
    _pl_row(7, "Track B", "skip",
            f"月度 LLM 叠加 — 本月首个交易日触发" + (f"  ·  {_tb_pending} 项待批" if _tb_pending else ""),
            _C["muted"] if not _tb_pending else _C["yellow"])

# Step 8: ERA / Quarterly Review (quarterly background thread)
_era_pending = sum(1 for pa in pending_approvals
                   if pa.get("approval_type") in ("factor_candidate", "universe_change"))
_is_q_day = False
try:
    from engine.daily_batch import _is_first_trading_day_of_quarter as _iqd
    _is_q_day = _iqd(today)
except Exception:
    pass
if _is_q_day:
    _pl_row(8, "ERA + 季度审计", "ok" if not _era_pending else "warn",
            f"季度首日 — ERA/BH/Universe 已触发" + (f"  ·  {_era_pending} 项待批" if _era_pending else ""),
            _C["green"] if not _era_pending else _C["yellow"])
else:
    _pl_row(8, "ERA + 季度审计", "skip",
            f"季度首日触发（ERA/BH/Universe）" + (f"  ·  {_era_pending} 项待批" if _era_pending else ""),
            _C["muted"] if not _era_pending else _C["yellow"])

st.markdown('<div style="height:0.6rem;"></div>', unsafe_allow_html=True)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — PENDING ACTIONS  (Gates / Alerts / Auto-processed)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f'<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;'
    f'color:{_C["muted"]};margin-bottom:0.6rem;">A · 待处理事项</div>',
    unsafe_allow_html=True,
)

# Partition alerts from today's batch
_all_alerts      = (batch.risk_alerts if batch else [])
_auto_exec_alerts = [a for a in _all_alerts if "auto_executed" in a]
_manual_needed    = [a for a in _all_alerts
                     if "auto_executed" not in a
                     and ("hard_stop" in a or "drawdown_stop" in a)]

_total_actions = len(pending_gates) + len(pending_approvals)

if not pending_gates and not pending_approvals:
    st.markdown(
        f'<div style="background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.2);'
        f'border-radius:6px;padding:0.6rem 1rem;font-size:0.9rem;color:{_C["green"]};">'
        f'✅  今日无待处理事项</div>',
        unsafe_allow_html=True,
    )
else:
    # ── A1: GATES (strategic — CycleState pending_gate) ───────────────────
    if pending_gates:
        st.markdown(
            f'<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;'
            f'color:{_C["blue"]};margin-bottom:0.4rem;margin-top:0.2rem;">🔒  Gates · 战略审批</div>',
            unsafe_allow_html=True,
        )
    for pg in pending_gates:
        gate_label = _GATE_LABELS.get(pg["gate"], pg["gate"])
        st.markdown(
            f'<div style="background:rgba(96,165,250,0.08);border:1.5px solid {_C["blue"]};'
            f'border-left:5px solid {_C["blue"]};border-radius:6px;'
            f'padding:0.9rem 1.2rem;margin-bottom:0.6rem;">'
            f'<span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;'
            f'color:{_C["blue"]};font-weight:800;">GATE</span>'
            f'  <b style="color:{_C["text"]};">{gate_label}</b>'
            f'<span style="font-size:0.78rem;color:{_C["muted"]};margin-left:1rem;">'
            f'Cycle #{pg["id"]}  ·  {pg["cycle_type"]}  ·  {pg["as_of_date"]}</span></div>',
            unsafe_allow_html=True,
        )
        # Show execution feedback from a previous approval if stored
        _exec_fb = st.session_state.get(f"_gate_exec_{pg['id']}")
        if _exec_fb and "error" not in _exec_fb:
            st.success(
                f"✅ 已执行再平衡 — 换手率 {_exec_fb.get('turnover', 0):.1%}  "
                f"成本 {_exec_fb.get('total_cost_bps', 0):.1f} bps  "
                f"{_exec_fb.get('n_trades', 0)} 笔交易"
            )
        elif _exec_fb and "error" in _exec_fb:
            st.error(f"执行失败: {_exec_fb['error']}")

        if pg["gate"] == GATE_MONTHLY_REBALANCE:
            try:
                from engine.portfolio_tracker import execute_rebalance
                dry = execute_rebalance(
                    rebalance_date=datetime.date.fromisoformat(pg["as_of_date"]),
                    dry_run=True,
                )
                if dry:
                    # P3-11: Gate summary narrative line
                    try:
                        from engine.narrative_builder import NarrativeBuilder as _NB2
                        _gate_narr = _NB2().gate_summary_line(
                            gate_label=_GATE_LABELS.get(pg["gate"], pg["gate"]),
                            dry_run_result=dry,
                        )
                        st.caption(_gate_narr)
                    except Exception:
                        pass
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("换手率",     f"{dry.get('turnover', 0):.1%}")
                    mc2.metric("预估成本",   f"{dry.get('total_cost_bps', 0):.1f} bps")
                    mc3.metric("拟执行笔数", len(dry.get("trades", [])))
                    trades = dry.get("trades", [])
                    if trades:
                        tdf = pd.DataFrame(trades)
                        for col in ["weight_before", "weight_after", "weight_delta"]:
                            if col in tdf.columns:
                                tdf[col] = tdf[col].apply(lambda x: f"{x:+.2%}")
                        if "cost_bps" in tdf.columns:
                            tdf["cost_bps"] = tdf["cost_bps"].apply(
                                lambda x: f"{x:.1f}" if x else "—")
                        show = [c for c in ["sector", "ticker", "action", "weight_before",
                                            "weight_after", "weight_delta", "cost_bps",
                                            "trigger_reason"] if c in tdf.columns]
                        st.dataframe(tdf[show], use_container_width=True, hide_index=True)
            except Exception as e:
                st.caption(f"无法预览交易明细: {e}")

        ac1, ac2, ac3 = st.columns([1, 1, 3])
        with ac3:
            gate_note = st.text_input("备注", key=f"gate_note_{pg['id']}",
                                       label_visibility="collapsed",
                                       placeholder="备注（可选）")
        with ac1:
            if st.button("✅ 批准执行", key=f"gate_approve_{pg['id']}",
                          type="primary", use_container_width=True):
                exec_result = tco.approve_gate(
                    pg["id"], approved=True, note=gate_note or "Daily Brief 批准"
                )
                st.session_state[f"_gate_exec_{pg['id']}"] = exec_result
                st.rerun()
        with ac2:
            if st.button("❌ 拒绝", key=f"gate_reject_{pg['id']}",
                          use_container_width=True):
                tco.approve_gate(
                    pg["id"], approved=False, note=gate_note or "Daily Brief 拒绝"
                )
                st.rerun()
        st.markdown('<div style="height:0.3rem;"></div>', unsafe_allow_html=True)

    # ── A2: ALERTS (tactical — PendingApproval awaiting human decision) ───
    if pending_approvals:
        st.markdown(
            f'<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;'
            f'color:{_C["yellow"]};margin-bottom:0.4rem;margin-top:0.6rem;">⚡  Alerts · 战术审批</div>',
            unsafe_allow_html=True,
        )
    for pa in pending_approvals:
        _pri = pa["priority"]
        _pcolor, _plabel = _PRIORITY_STYLE.get(_pri, (_C["muted"], _pri.upper()))
        _atype = pa["approval_type"]
        _type_label = _APPROVAL_TYPE_LABEL.get(_atype, _atype)
        _deadline = pa["approval_deadline"]
        _deadline_str = (
            f'  ·  <span style="color:{_C["yellow"]};">截止 {_deadline}</span>'
            if _deadline != "—" else ""
        )
        _weight_str = (
            f'  ·  目标权重 {pa["suggested_weight"]:.1%}'
            if pa.get("suggested_weight") is not None else ""
        )
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.02);'
            f'border:1px solid {_pcolor}33;border-left:4px solid {_pcolor};'
            f'border-radius:5px;padding:0.55rem 1rem;margin-bottom:0.1rem;">'
            f'<span style="font-size:0.68rem;font-weight:800;letter-spacing:0.08em;'
            f'color:{_pcolor};">[{_plabel}]</span>  '
            f'<b style="color:{_C["text"]};">{_type_label}</b>  '
            f'<span style="color:{_C["muted"]};">·  {pa["sector"]}  ({pa["ticker"]})  '
            f'·  {pa["triggered_condition"][:60]}'
            f'{_weight_str}{_deadline_str}</span></div>',
            unsafe_allow_html=True,
        )
        # P6 signal context row — composite score + TSMOM + decay (numbers only)
        _ticker = pa.get("ticker", "")
        if _ticker and _atype in ("entry", "track_b", "signal_decay", "risk_control", "vol_spike_p6"):
            _sctx = _load_signal_context(_ticker)
            if _sctx:
                _comp   = _sctx.get("composite")
                _tsmom  = _sctx.get("tsmom")
                _decay  = _sctx.get("decay_pct", 0.0)
                _flipped= _sctx.get("flipped", False)
                _sdate  = _sctx.get("signal_date", "")
                _regime_label_short = (getattr(regime, "regime", "—")).upper() if regime else "—"
                _limits_map = {"RISK-ON": "10L/6S", "TRANSITION": "7L/7S", "RISK-OFF": "5L/8S"}
                _limits_str = _limits_map.get(_regime_label_short, "—")
                _parts = []
                if _comp is not None:
                    _bar_w = max(0, min(100, int(_comp)))
                    _bar_col = _C["green"] if _comp >= 60 else (_C["yellow"] if _comp >= 40 else _C["red"])
                    _parts.append(
                        f'<span style="margin-right:0.7rem;">Composite '
                        f'<span style="font-family:{_C["mono"]};font-weight:700;color:{_bar_col};">'
                        f'{_comp:.0f}</span>'
                        f'<span style="display:inline-block;width:{_bar_w//4}rem;height:4px;'
                        f'background:{_bar_col};opacity:0.6;margin-left:0.3rem;vertical-align:middle;">'
                        f'</span></span>'
                    )
                if _tsmom is not None:
                    _ts_col = _C["green"] if _tsmom > 0 else (_C["red"] if _tsmom < 0 else _C["muted"])
                    _ts_sym = "▲" if _tsmom > 0 else ("▼" if _tsmom < 0 else "—")
                    _parts.append(
                        f'<span style="margin-right:0.7rem;">TSMOM '
                        f'<span style="font-family:{_C["mono"]};font-weight:700;color:{_ts_col};">'
                        f'{_ts_sym}{_tsmom:+d}</span></span>'
                    )
                if _decay and _decay > 0.2:
                    _dec_col = _C["red"] if _decay > 0.6 else _C["yellow"]
                    _parts.append(
                        f'<span style="margin-right:0.7rem;">衰减 '
                        f'<span style="font-family:{_C["mono"]};color:{_dec_col};">'
                        f'{_decay:.0%}</span></span>'
                    )
                if _flipped:
                    _parts.append(
                        f'<span style="margin-right:0.7rem;color:{_C["yellow"]};font-weight:700;">'
                        f'⚡ 今日翻转</span>'
                    )
                _parts.append(
                    f'<span style="color:{_C["muted"]};">制度 {_regime_label_short} · 限额 {_limits_str}</span>'
                )
                if _parts:
                    st.markdown(
                        f'<div style="font-size:0.75rem;padding:0.22rem 1rem 0.32rem;'
                        f'margin-bottom:0.1rem;background:rgba(0,0,0,0.06);'
                        f'border-left:4px solid {_pcolor};border-radius:0 0 4px 4px;">'
                        + "  ".join(_parts)
                        + f'<span style="font-size:0.68rem;color:{_C["muted"]};margin-left:0.8rem;">'
                        f'{_sdate}</span></div>',
                        unsafe_allow_html=True,
                    )
        # P3-12: Contrarian override warning banner
        if pa.get("contradicts_quant"):
            _conf_val = pa.get("llm_confidence")
            _conf_str = f"置信度 {_conf_val}" if _conf_val is not None else "置信度未知"
            st.markdown(
                f'<div style="background:rgba(245,158,11,0.08);border:1px solid {_C["yellow"]}55;'
                f'border-radius:4px;padding:0.35rem 0.9rem;margin-top:0.15rem;margin-bottom:0.25rem;'
                f'font-size:0.78rem;color:{_C["yellow"]};">'
                f'⚠ <b>LLM/Quant 分歧</b> — LLM 方向与 TSMOM 信号相反（{_conf_str}）。'
                f'低置信度（&lt;75）批准将被系统自动驳回。</div>',
                unsafe_allow_html=True,
            )
        # Show exec feedback if resolved
        _pa_fb = st.session_state.get(f"_pa_exec_{pa['id']}")
        if _pa_fb:
            if _pa_fb.get("ok"):
                _ed = _pa_fb.get("exec_detail", {})
                _detail = (
                    f"  权重已设为 {_ed['weight_set']:.1%}" if "weight_set" in _ed else
                    f"  已平仓（权重归零，前值 {_ed.get('weight_zeroed', 0):.1%}）" if "weight_zeroed" in _ed else
                    f"  换手率 {_ed.get('turnover', 0):.1%}，{_ed.get('n_trades', 0)} 笔" if "n_trades" in _ed else ""
                )
                st.success(f"✅ {_pa_fb['message']}{_detail}")
            else:
                st.error(f"❌ {_pa_fb.get('message', '执行失败')}")

        pa_ac1, pa_ac2, pa_ac3 = st.columns([1, 1, 3])
        with pa_ac3:
            pa_note = st.text_input(
                "拒绝理由", key=f"pa_note_{pa['id']}",
                label_visibility="collapsed",
                placeholder="拒绝理由（可选）",
            )
        with pa_ac1:
            if st.button("✅ 执行", key=f"pa_approve_{pa['id']}",
                          type="primary", use_container_width=True):
                fb = resolve_pending_approval(pa["id"], approved=True)
                st.session_state[f"_pa_exec_{pa['id']}"] = fb
                st.rerun()
        with pa_ac2:
            if st.button("❌ 跳过", key=f"pa_reject_{pa['id']}",
                          use_container_width=True):
                fb = resolve_pending_approval(
                    pa["id"], approved=False, rejection_reason=pa_note or "Daily Brief 跳过"
                )
                st.session_state[f"_pa_exec_{pa['id']}"] = fb
                st.rerun()
        st.markdown('<div style="height:0.3rem;"></div>', unsafe_allow_html=True)

# ── A3: AUTO-PROCESSED (informational — Layer 2 stops already executed) ───
if _auto_exec_alerts:
    _auto_label = f"今日自动执行止损 · {len(_auto_exec_alerts)} 笔"
    with st.expander(f"✅  {_auto_label}（Layer 2 已自动处理，无需操作）", expanded=False):
        _AUTO_ICONS = {"hard_stop": "🛑", "drawdown_stop": "🔴"}
        for a in _auto_exec_alerts:
            parts = a.split(":")
            _sec  = parts[0] if parts else a
            _typ  = parts[1] if len(parts) > 1 else "stop"
            _icon = _AUTO_ICONS.get(_typ, "⚠")
            st.markdown(
                f'<div style="font-size:0.83rem;padding:0.12rem 0;">'
                f'<span style="color:{_C["green"]};">{_icon}  <b>{_sec}</b>  '
                f'自动平仓 · {_typ.replace("_", " ")}</span>'
                f'<span style="font-size:0.72rem;color:{_C["muted"]};margin-left:0.8rem;">'
                f'已写入 SimulatedTrade 记录</span></div>',
                unsafe_allow_html=True,
            )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — TODAY'S INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f'<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;'
    f'color:{_C["muted"]};margin-bottom:0.6rem;">B · 今日情报</div>',
    unsafe_allow_html=True,
)

# ── Section B: LLM-first intelligence card ───────────────────────────────────
_bs               = brief_snap or {}
_narrative        = _bs.get("narrative")
_llm_brief_stored = _bs.get("macro_brief_llm")
_regime_changed   = _bs.get("regime_changed", False)
_icir_updated     = _bs.get("icir_month")
_n_verified_today = _bs.get("n_verified_today", 0)

# Session-state LLM cache key: regenerate when regime or event fingerprint changes
_today_events = []
if batch:
    _today_events  = list(getattr(batch, "risk_alerts", []) or [])
    _today_events += [f"入场:{s}" for s in (getattr(batch, "entries_triggered", []) or [])]
    _today_events += [f"翻转:{s}" for s in (getattr(batch, "invalidations", []) or [])]
_ops_cache_key = f"ops_brief_{today}_{_eff_regime}_{len(_today_events)}_{len(pending_approvals)}"

# ── Live Ops Brief (LLM-generated, session-cached) ───────────────────────────
_col_brief, _col_btn = st.columns([5, 1])
with _col_btn:
    _gen_btn = st.button(
        "🤖 生成简报",
        key="gen_ops_brief_2",
        use_container_width=True,
        help="调用 Gemini 生成今日智能操盘简报（~5秒）",
    )

_ops_brief_text = st.session_state.get(_ops_cache_key, "")

if _gen_btn or (not _ops_brief_text and not _llm_brief_stored):
    _model_obj = None
    try:
        from engine.key_pool import get_pool as _gp
        _model_obj = _gp().get_model()
    except Exception:
        pass
    if _model_obj is not None:
        _vix_now = st.session_state.get("_vix_input", 20.0)
        with st.spinner("Gemini 正在生成今日操盘简报…"):
            _ops_brief_text = _generate_ops_brief(
                _model_obj, _eff_regime, _eff_p,
                batch, len(pending_approvals) + len(pending_gates),
                _today_events, _vix_now,
            )
        if _ops_brief_text and "失败" not in _ops_brief_text:
            st.session_state[_ops_cache_key] = _ops_brief_text

# ── Display intelligence card ─────────────────────────────────────────────────
_display_text = _ops_brief_text or _llm_brief_stored
_is_live_llm  = bool(_ops_brief_text)
_is_batch_llm = bool(_llm_brief_stored) and not _is_live_llm

if _display_text:
    _narr_border = _C["yellow"] if _regime_changed else _C["purple"]
    _narr_bg     = "rgba(245,158,11,0.04)" if _regime_changed else "rgba(167,139,250,0.04)"
    _src_label   = ("Gemini · 实时生成" if _is_live_llm else "Gemini · 批次生成")
    with _col_brief:
        st.markdown(
            f'<div style="background:{_narr_bg};border:1.5px solid {_narr_border};'
            f'border-radius:6px;padding:0.85rem 1.2rem;margin-bottom:0.9rem;">'
            f'<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;'
            f'color:{_C["purple"]};margin-bottom:0.35rem;">{_src_label}</div>'
            + (f'<div style="font-size:0.72rem;color:{_C["yellow"]};font-weight:700;'
               f'margin-bottom:0.3rem;">⚠ 制度切换  '
               f'{_bs.get("regime_prev","—")} → {_bs.get("regime","—")}</div>'
               if _regime_changed and _bs.get("regime_prev") else "")
            + f'<div style="font-size:0.95rem;color:{_C["text"]};line-height:1.75;">'
            f'{_display_text}</div>'
            + f'<div style="font-size:0.68rem;color:{_C["muted"]};margin-top:0.5rem;">'
            + (f'✅ 今日验证 {_n_verified_today} 条  ·  ' if _n_verified_today else '')
            + (f'ICIR {_icir_updated}  ·  ' if _icir_updated else '')
            + f'<span style="color:{_C["purple"]};">Gemini 2.5 Flash</span></div>'
            + '</div>',
            unsafe_allow_html=True,
        )
else:
    # Pure rule-based fallback when LLM completely unavailable
    _rule_narrative = _narrative
    try:
        from engine.narrative_builder import NarrativeBuilder as _NB
        _rule_narrative = _NB().build_section_b(
            snap_dict=_bs,
            pending_gates=pending_gates,
            pending_approvals=pending_approvals,
        )
    except Exception:
        pass
    _fallback_text = _rule_narrative or (
        f"{today} 信号与制度{'已就绪' if (batch and batch.signal_ok) else '待更新'}。"
        + ("今日无触发事件。" if not _today_events else f"触发事件 {len(_today_events)} 个。")
    )
    with _col_brief:
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.02);border:1px solid {_C["border"]};'
            f'border-radius:6px;padding:0.75rem 1.1rem;margin-bottom:0.9rem;">'
            f'<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;'
            f'color:{_C["muted"]};margin-bottom:0.3rem;">系统叙述 · Rule-Based  —  点击右侧按钮调用 Gemini</div>'
            f'<div style="font-size:0.9rem;color:{_C["text"]};line-height:1.65;">'
            f'{_fallback_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Event details + queue health (two columns) ───────────────────────────────
col_intel_l, col_intel_r = st.columns([3, 2])

# Annotated event list with drill-down hints
_DRILL_HINTS = {
    "hard_stop":            ("🛑", _C["red"],    "硬止损触发",        "→ Positions 查看持仓详情"),
    "tsmom_flip":           ("⚡", _C["yellow"], "TSMOM 翻转",        "→ Signal Board 查看完整信号矩阵"),
    "regime_compress":      ("📉", _C["yellow"], "制度压缩",          "→ Signal Board → Regime Panel"),
    "vol_spike":            ("📈", _C["yellow"], "波动率尖峰",        "→ Positions 查看持仓详情"),
    "drawdown_stop":        ("🔴", _C["red"],    "回撤止损触发",      "→ Positions 查看持仓详情"),
    "tsmom_fast_flip":      ("🔄", _C["green"],  "Fast TSMOM 方向确认", "→ Signal Board 查看快速信号"),
    "regime_jump_compress": ("📉", _C["red"],    "制度跃变自动压缩",  "→ Positions 查看减仓执行"),
}

with col_intel_l:
    _risk_alerts      = batch.risk_alerts if batch else []
    _entries          = batch.entries_triggered if batch else []
    _invalidations    = batch.invalidations if batch else []
    _tactical_entries = getattr(batch, "tactical_entries", []) if batch else []
    _tactical_reduces = getattr(batch, "tactical_reduces", []) if batch else []
    _regime_jump      = getattr(batch, "regime_jump", False) if batch else False

    _has_events = bool(
        _risk_alerts or _entries or _invalidations
        or _tactical_entries or _tactical_reduces or _regime_jump
    )
    if not _has_events:
        st.markdown(
            f'<div style="font-size:0.85rem;color:{_C["muted"]};padding:0.3rem 0;">'
            f'今日无触发事件，组合状态平稳。</div>',
            unsafe_allow_html=True,
        )
    else:
        # ── Regime jump banner (highest priority) ───────────────────────
        if _regime_jump:
            st.markdown(
                f'<div style="font-size:0.85rem;padding:0.25rem 0.6rem;margin-bottom:0.3rem;'
                f'background:rgba(239,68,68,0.12);border-left:3px solid {_C["red"]};'
                f'border-radius:3px;">'
                f'<span style="color:{_C["red"]};font-weight:700;">⚠️  制度跃变</span>'
                f'<span style="font-size:0.78rem;color:{_C["muted"]};margin-left:0.8rem;">'
                f'P(risk-off) 单日跳升超阈值 — Layer 2 已自动压缩多头敞口</span></div>',
                unsafe_allow_html=True,
            )
        # ── Tactical entries (Layer 2 auto or Layer 3 pending) ───────────
        for _te in _tactical_entries:
            st.markdown(
                f'<div style="font-size:0.83rem;padding:0.18rem 0;">'
                f'<span style="color:{_C["green"]};">📍  <b>{_te}</b>  战术入场触发</span>'
                f'<span style="font-size:0.72rem;color:{_C["muted"]};margin-left:0.8rem;">'
                f'→ Pending Actions 查看审批项</span></div>',
                unsafe_allow_html=True,
            )
        # ── Tactical reduces ─────────────────────────────────────────────
        for _tr in _tactical_reduces:
            st.markdown(
                f'<div style="font-size:0.83rem;padding:0.18rem 0;">'
                f'<span style="color:{_C["yellow"]};">📉  <b>{_tr}</b>  战术减仓执行</span>'
                f'<span style="font-size:0.72rem;color:{_C["muted"]};margin-left:0.8rem;">'
                f'→ Positions 查看最新权重</span></div>',
                unsafe_allow_html=True,
            )
        # ── Standard risk alerts ─────────────────────────────────────────
        for alert in _risk_alerts:
            _ak = next((k for k in _DRILL_HINTS if k in alert), None)
            if _ak:
                _icon, _color, _label, _hint = _DRILL_HINTS[_ak]
                _sector = alert.split(":")[0]
                st.markdown(
                    f'<div style="font-size:0.83rem;padding:0.18rem 0;">'
                    f'<span style="color:{_color};">{_icon}  <b>{_sector}</b>  {_label}</span>'
                    f'<span style="font-size:0.72rem;color:{_C["muted"]};margin-left:0.8rem;">'
                    f'{_hint}</span></div>',
                    unsafe_allow_html=True,
                )
        for sector in _entries:
            st.markdown(
                f'<div style="font-size:0.83rem;padding:0.18rem 0;">'
                f'<span style="color:{_C["green"]};">📍  <b>{sector}</b>  入场条件触发</span>'
                f'<span style="font-size:0.72rem;color:{_C["muted"]};margin-left:0.8rem;">'
                f'→ Pending Actions 查看审批项</span></div>',
                unsafe_allow_html=True,
            )
        for sector in _invalidations:
            st.markdown(
                f'<div style="font-size:0.83rem;padding:0.18rem 0;">'
                f'<span style="color:{_C["muted"]};">⊘  <b>{sector}</b>  持仓无效化</span></div>',
                unsafe_allow_html=True,
            )

with col_intel_r:
    # Verification queue health
    st.markdown(
        f'<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;'
        f'color:{_C["muted"]};margin-bottom:0.2rem;">验证队列</div>'
        f'<div style="font-size:0.9rem;">'
        f'<span style="color:{_C["muted"]};">{len(pending_decisions)} 条待验证</span>'
        + (f'  <span style="color:{_C["red"]};font-weight:700;">·  {n_overdue} 已过期</span>'
           if n_overdue else "")
        + (f'  <span style="color:{_C["yellow"]};">·  {n_approaching} 即将到期</span>'
           if n_approaching else "")
        + '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

    # Batch health — show specific failed steps, not just a boolean
    _batch_ok = (batch.signal_ok and batch.regime_ok) if batch else False
    if _batch_ok:
        _batch_html = (
            f'<div style="font-size:0.9rem;color:{_C["green"]};">✅ 信号 + 制度 已就绪</div>'
        )
    else:
        _failed = []
        if batch and not batch.signal_ok:
            _failed.append("信号更新")
        if batch and not batch.regime_ok:
            _failed.append("制度识别")
        _err_lines = ""
        if batch and batch.errors:
            for _e in batch.errors[:3]:
                _err_lines += (
                    f'<div style="font-size:0.72rem;color:{_C["muted"]};'
                    f'font-family:{_C["mono"]};margin-top:0.15rem;'
                    f'word-break:break-all;">{_e[:80]}</div>'
                )
        _batch_html = (
            f'<div style="font-size:0.9rem;color:{_C["yellow"]};">'
            f'⚠ {" · ".join(_failed) if _failed else "步骤"} 异常</div>'
            + _err_lines
        )
    st.markdown(
        f'<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;'
        f'color:{_C["muted"]};margin-bottom:0.2rem;">今日批次</div>'
        + _batch_html,
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

    # Automation health: VERIFY + ICIR
    _verify_done  = _bs.get("verify_ran", False)
    _this_month   = today.strftime("%Y-%m")
    _icir_current = (_bs.get("icir_month") == _this_month)
    st.markdown(
        f'<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;'
        f'color:{_C["muted"]};margin-bottom:0.2rem;">自动化任务</div>'
        f'<div style="font-size:0.82rem;line-height:1.9;">'
        f'<span style="color:{"" + _C["green"] + "" if _verify_done else _C["muted"]};">'
        f'{"✅" if _verify_done else "○"}  VERIFY + LEARN</span><br>'
        f'<span style="color:{"" + _C["green"] + "" if _icir_current else _C["muted"]};">'
        f'{"✅" if _icir_current else "○"}  ICIR 月度更新</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — RECENT VERIFICATIONS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f'<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;'
    f'color:{_C["muted"]};margin-bottom:0.6rem;">C · 最近验证结果</div>',
    unsafe_allow_html=True,
)

if not recent_verifs:
    st.markdown(
        f'<div style="font-size:0.85rem;color:{_C["muted"]};">暂无已验证决策</div>',
        unsafe_allow_html=True,
    )
else:
    v_cols = st.columns(min(len(recent_verifs), 3))
    for i, v in enumerate(recent_verifs):
        col = v_cols[i % 3]
        _acc   = v["accuracy"] or 0.0
        _vdict = v["verdict"] or "—"
        _vcolor, _vicon = _VERDICT_STYLE.get(_vdict, (_C["muted"], "—"))
        _lcs_badge = (
            f'  <span style="font-size:0.65rem;color:{_C["green"]};">LCS✓</span>'
            if v["lcs_passed"] else
            (f'  <span style="font-size:0.65rem;color:{_C["yellow"]};">LCS?</span>'
             if v["lcs_passed"] is None else
             f'  <span style="font-size:0.65rem;color:{_C["muted"]};">LCS✗</span>')
        )
        _review_badge = (
            f'  <span style="font-size:0.65rem;color:{_C["red"]};">⚠待归因</span>'
            if v["needs_review"] else ""
        )
        col.markdown(
            f'<div style="background:{_C["card"]};border:1px solid {_C["border"]};'
            f'border-top:3px solid {_vcolor};border-radius:6px;padding:0.7rem 0.9rem;">'
            f'<div style="font-size:0.78rem;font-weight:700;color:{_C["text"]};">'
            f'{v["sector_name"]}</div>'
            f'<div style="font-size:0.72rem;color:{_C["muted"]};margin:0.15rem 0;">'
            f'{v["direction"]}  ·  {v["horizon"][:2]}  ·  {v["verified_at"]}</div>'
            f'<div style="font-size:1.1rem;font-weight:800;color:{_vcolor};margin-top:0.3rem;">'
            f'{_vicon}  {_acc:.0%}'
            f'<span style="font-size:0.72rem;font-weight:400;color:{_C["muted"]};'
            f'margin-left:0.4rem;">{v["barrier"]}</span>'
            f'{_lcs_badge}{_review_badge}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — ENGINEERING  (默认折叠，不影响日常工作流)
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🔧  系统日志 — 管道历史 · 紧急人工干预（正常运行无需操作）", expanded=False):

    # Emergency manual override
    st.markdown(
        f'<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;'
        f'color:{_C["yellow"]};margin-bottom:0.2rem;">⚠ 紧急人工干预</div>'
        f'<div style="font-size:0.72rem;color:{_C["muted"]};margin-bottom:0.5rem;">'
        f'自动化批次每日 UTC 00:30 由 ensure_daily_batch_completed 执行。'
        f'仅在自动化失败时手动触发。</div>',
        unsafe_allow_html=True,
    )
    ctl1, ctl2, ctl3, ctl4, ctl5 = st.columns([2, 1, 1, 1, 2])
    with ctl1:
        run_type = st.selectbox(
            "类型", ["daily — 信号+制度+组合", "monthly — 含再平衡", "verification — 决策验证"],
            label_visibility="collapsed",
        )
    with ctl2:
        lookback_m = st.number_input("形成期(月)", min_value=6, max_value=24,
                                     value=12, step=1, label_visibility="collapsed")
    with ctl3:
        skip_m = st.number_input("跳过月", min_value=0, max_value=3,
                                  value=1, step=1, label_visibility="collapsed")
    with ctl4:
        dry_run = st.checkbox("Dry Run", value=True)
    with ctl5:
        run_btn = st.button("▶  强制重跑", use_container_width=True,
                            help="绕过自动化缓存，强制重新执行选定批次")

    if run_btn:
        _model = st.session_state.get("_agent_executor")
        if "verification" in run_type:
            with st.spinner("运行验证周期…"):
                try:
                    verify_list = tco.run_verification(model=_model)
                    n_done = len(verify_list) if verify_list else 0
                    st.success(f"验证完成 — {n_done} 条决策已验证")
                    if n_done:
                        _vcols = [c for c in ["sector_name", "direction", "barrier",
                                              "accuracy", "verdict"]
                                  if c in pd.DataFrame(verify_list).columns]
                        st.dataframe(pd.DataFrame(verify_list)[_vcols],
                                     hide_index=True, use_container_width=True)
                except Exception as e:
                    st.error(f"验证失败: {e}")
        elif "monthly" in run_type:
            with st.spinner("运行月度周期…"):
                try:
                    result = tco.run_monthly(as_of=today, lookback_months=lookback_m,
                                             skip_months=skip_m, dry_run=dry_run)
                    st.session_state["_last_chain_result"] = result
                    st.success(
                        f"月度周期完成 — {result.regime}  "
                        f"多/空 {result.n_long}/{result.n_short}  "
                        f"{result.elapsed_s:.1f}s"
                    ) if result.ok else [st.error(e) for e in result.errors]
                except Exception as e:
                    st.error(f"月度周期失败: {e}")
        else:
            with st.spinner("运行今日链式分析…"):
                try:
                    result = run_daily_chain(
                        as_of=today, lookback_months=lookback_m,
                        skip_months=skip_m, dry_run=dry_run, model=_model,
                    )
                    st.session_state["_last_chain_result"] = result
                    if result.ok:
                        st.success(
                            f"完成 — {result.regime}  "
                            f"多/空 {result.n_long}/{result.n_short}  "
                            f"翻转 {len(result.signal_flips)} 个  "
                            f"{result.elapsed_s:.1f}s"
                        )
                        if result.signal_flips:
                            st.warning("⚡ " + " | ".join(result.signal_flips))
                    else:
                        for err in result.errors:
                            st.error(err)
                except Exception as e:
                    st.error(f"链式分析失败: {e}")
        st.rerun()

    # Last run step detail
    _res: ChainResult | None = st.session_state.get(
        "_last_chain_result", st.session_state.get("chain_result")
    )
    if _res:
        st.markdown(
            f'<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;'
            f'color:{_C["muted"]};margin:0.8rem 0 0.3rem;">最近一次运行步骤</div>',
            unsafe_allow_html=True,
        )
        _STEP_ICONS = {"ok": "✅", "failed": "❌", "skipped": "⏭", "running": "⏳"}
        step_rows = [
            {
                "步骤":    s.name,
                "状态":    f'{_STEP_ICONS.get(s.status, "⏳")} {s.status}',
                "耗时(s)": s.elapsed_s,
                "备注":    s.detail[:80] if s.detail else "—",
            }
            for s in _res.steps
        ]
        st.dataframe(pd.DataFrame(step_rows), hide_index=True, use_container_width=True)

    # Recent cycle history
    st.markdown(
        f'<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;'
        f'color:{_C["muted"]};margin:0.8rem 0 0.3rem;">历史周期记录</div>',
        unsafe_allow_html=True,
    )
    recent_cycles = tco.get_recent_cycles(n=10)
    if recent_cycles:
        cyc_rows = []
        for c in recent_cycles:
            si, _ = _STATUS_ICONS.get(c["status"], ("", _C["muted"]))
            cyc_rows.append({
                "ID":    c["id"],
                "类型":  c["cycle_type"],
                "日期":  c["as_of_date"],
                "状态":  f'{si} {c["status"]}',
                "闸门":  _GATE_LABELS.get(c.get("gate"), c.get("gate") or "—"),
                "耗时s": c.get("elapsed_s", "—"),
                "错误":  (c.get("error_log") or "")[:60],
            })
        st.dataframe(pd.DataFrame(cyc_rows), hide_index=True, use_container_width=True)
