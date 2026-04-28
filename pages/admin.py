"""
Macro Alpha Pro — Backend Admin Panel
System learning management, verification, and bias analysis.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime
import re
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from google import genai
import ui.theme as theme

from engine.history import (
    run_sector_backtest, run_walk_forward_backtest, build_snapshot,
    SECTOR_ETF, get_active_sector_etf,
)
from engine.memory import (
    init_db,
    get_stats,
    verify_pending_decisions,
    run_meta_agent_analysis,
    get_learning_patterns,
    get_learning_log_raw,
    get_dormant_pattern_count,
    get_backtest_retry_stubs,
    clear_retry_stub,
    get_backtest_records,
    delete_backtest_records,
    delete_backtest_record_by_ids,
    set_backtest_stop,
    get_backtest_stop,
    get_active_backtest_session,
    mark_pattern_applied,
    get_news_routing_weights,
    update_news_routing_weight,
    get_records_needing_review,
    set_human_label,
    get_training_coverage,
    get_clean_zone_stats,
    BASELINE_HIT_RATE, MIN_ACCEPTABLE, GOOD_THRESHOLD, EXCELLENT,
    _DEFAULT_WEIGHTS,
    TRAIN_TEST_CUTOFF,
    CLEAN_ZONE_START,
    save_stress_test_log,
    get_stress_test_history,
    get_system_config,
    set_system_config,
    get_lasso_lambda,
    _LASSO_GATE_MIN_N,
    _LASSO_GATE_MIN_REGIMES,
    backfill_macro_verified,
    SessionFactory,
    DecisionLog,
    get_unattributed_failures,
    set_failure_attribution,
    get_failure_attribution_stats,
    _FAILURE_TYPE_LABELS,
    get_pending_decisions_for_monitor,
    get_clean_zone_time_series,
    get_failure_mode_stats,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
# set_page_config handled by app.py via st.navigation()
# st.set_page_config(page_title="Admin · Macro Alpha Pro", layout="wide", page_icon="⚙️", initial_sidebar_state="collapsed")

init_db()

# Design system and header rendered by app.py for all pages via st.navigation()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
_ADMIN_MODEL_NAME = "gemini-2.5-flash"


def _make_model(api_key: str | None):
    """
    Return a model-like wrapper using the new google-genai SDK.
    Exposes generate_content(prompt) → object with .text, matching the interface
    expected by memory.py (verify_pending_decisions, run_meta_agent_analysis, etc.).
    """
    if not api_key:
        return None
    _client = genai.Client(api_key=api_key)
    _mname  = _ADMIN_MODEL_NAME

    class _Wrapper:
        def generate_content(self, prompt: str):
            return _client.models.generate_content(model=_mname, contents=prompt)

    return _Wrapper()


def _section(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def _render_analysis(text: str) -> None:
    """
    Render AI analysis text with the [XAI_ATTRIBUTION] block stripped from the
    prose and re-rendered as a structured metrics card below.
    """
    import re as _re

    # ── Split prose from XAI block ────────────────────────────────────────────
    xai_match = _re.search(
        r"###?\s*\[XAI_ATTRIBUTION\](.*?)###?\s*\[/XAI_ATTRIBUTION\]",
        text, _re.DOTALL | _re.IGNORECASE,
    )
    if not xai_match:
        st.markdown(text)
        return

    prose = text[: xai_match.start()].strip()
    block = xai_match.group(1)

    # ── Render clean prose ────────────────────────────────────────────────────
    st.markdown(prose)

    # ── Parse XAI fields ─────────────────────────────────────────────────────
    def _int_field(key: str) -> int | None:
        m = _re.search(rf"{key}\s*:\s*(\d+)", block)
        return min(100, max(0, int(m.group(1)))) if m else None

    def _str_field(key: str) -> str:
        m = _re.search(rf"{key}\s*:\s*(.+)", block)
        return m.group(1).strip() if m else "—"

    overall   = _int_field("overall_confidence")
    macro_c   = _int_field("macro_confidence")
    news_c    = _int_field("news_confidence")
    tech_c    = _int_field("technical_confidence")
    drivers   = _str_field("signal_drivers")
    inval     = _str_field("invalidation_conditions")
    horizon   = _str_field("horizon")

    # ── Confidence bar helper ─────────────────────────────────────────────────
    _is_dark = theme.is_dark()
    def _conf_bar(val: int | None, label: str) -> str:
        if val is None:
            return ""
        if val >= 70:
            color = "#4ade80" if _is_dark else "#16a34a"
        elif val >= 40:
            color = "#fbbf24" if _is_dark else "#d97706"
        else:
            color = "#f87171" if _is_dark else "#dc2626"
        bar_bg = "rgba(255,255,255,0.08)" if _is_dark else "rgba(0,0,0,0.07)"
        text_c = "#e2e8f0" if _is_dark else "#1e293b"
        muted  = "#94a3b8" if _is_dark else "#64748b"
        return (
            f'<div style="margin-bottom:6px;">'
            f'<div style="display:flex; justify-content:space-between; '
            f'font-size:0.76rem; margin-bottom:2px;">'
            f'<span style="color:{muted};">{label}</span>'
            f'<span style="color:{text_c}; font-weight:600;">{val}</span></div>'
            f'<div style="background:{bar_bg}; border-radius:4px; height:6px;">'
            f'<div style="width:{val}%; background:{color}; border-radius:4px; height:6px;"></div>'
            f'</div></div>'
        )

    card_bg     = "rgba(255,255,255,0.04)" if _is_dark else "rgba(0,0,0,0.03)"
    border_c    = "rgba(255,255,255,0.10)" if _is_dark else "rgba(0,0,0,0.08)"
    text_c      = "#e2e8f0" if _is_dark else "#1e293b"
    muted_c     = "#94a3b8" if _is_dark else "#64748b"
    label_c     = "#cbd5e1" if _is_dark else "#475569"

    # Overall badge colour
    if overall is not None and overall >= 70:
        overall_color = "#4ade80" if _is_dark else "#16a34a"
    elif overall is not None and overall >= 40:
        overall_color = "#fbbf24" if _is_dark else "#d97706"
    else:
        overall_color = "#f87171" if _is_dark else "#dc2626"

    # Horizon badge
    horizon_short = horizon.split("(")[0].strip() if "(" in horizon else horizon
    detail_paren  = ("(" + horizon.split("(", 1)[1]) if "(" in horizon else ""

    st.markdown(
        f'<div style="margin-top:1.2rem; padding:1rem 1.2rem; '
        f'background:{card_bg}; border:1px solid {border_c}; border-radius:10px;">'

        # ── Header row ──────────────────────────────────────────────────────
        f'<div style="display:flex; justify-content:space-between; align-items:center; '
        f'margin-bottom:0.9rem;">'
        f'<span style="font-size:0.78rem; font-weight:700; color:{label_c}; '
        f'letter-spacing:0.06em; text-transform:uppercase;">XAI Attribution</span>'
        f'<div style="display:flex; gap:8px; align-items:center;">'
        + (
            f'<span style="font-size:0.72rem; color:{muted_c}; '
            f'background:{card_bg}; border:1px solid {border_c}; '
            f'border-radius:20px; padding:2px 10px;">'
            f'{horizon_short} <span style="color:{muted_c}; font-size:0.68rem;">'
            f'{detail_paren}</span></span>'
            if horizon != "—" else ""
        )
        + (
            f'<span style="font-size:0.82rem; font-weight:700; color:{overall_color}; '
            f'background:{card_bg}; border:1px solid {overall_color}44; '
            f'border-radius:20px; padding:2px 12px;">综合置信度 {overall}</span>'
            if overall is not None else ""
        )
        + f'</div></div>'

        # ── Confidence bars ──────────────────────────────────────────────────
        f'<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:0 1.5rem; '
        f'margin-bottom:0.8rem;">'
        f'<div>{_conf_bar(macro_c, "宏观置信度")}</div>'
        f'<div>{_conf_bar(news_c, "新闻置信度")}</div>'
        f'<div>{_conf_bar(tech_c, "技术置信度")}</div>'
        f'</div>'

        # ── Signal drivers ───────────────────────────────────────────────────
        f'<div style="margin-bottom:0.5rem;">'
        f'<span style="font-size:0.73rem; color:{muted_c}; text-transform:uppercase; '
        f'letter-spacing:0.05em;">信号驱动因素</span><br>'
        f'<span style="font-size:0.83rem; color:{text_c};">{drivers}</span></div>'

        # ── Invalidation conditions ──────────────────────────────────────────
        f'<div>'
        f'<span style="font-size:0.73rem; color:{muted_c}; text-transform:uppercase; '
        f'letter-spacing:0.05em;">失效条件</span><br>'
        f'<span style="font-size:0.83rem; color:{text_c};">{inval}</span></div>'

        f'</div>',
        unsafe_allow_html=True,
    )


def _score_badge(score: float) -> str:
    if score >= EXCELLENT:
        return '<span class="badge badge-pass">HIGH</span>'
    if score >= MIN_ACCEPTABLE:
        return '<span class="badge badge-warn">REVIEW</span>'
    return '<span class="badge badge-block">WEAK</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Risk helpers for Pending Review cards
# ─────────────────────────────────────────────────────────────────────────────
_FOMC_DATES_2026 = [
    datetime.date(2026, 1, 29), datetime.date(2026, 3, 19),
    datetime.date(2026, 5, 7),  datetime.date(2026, 6, 18),
    datetime.date(2026, 7, 30), datetime.date(2026, 9, 17),
    datetime.date(2026, 10, 29),datetime.date(2026, 12, 10),
]


def _upcoming_macro_events(window_days: int = 5) -> list[dict]:
    """Return FOMC / NFP events within window_days of today."""
    today  = datetime.date.today()
    events = []

    for d in _FOMC_DATES_2026:
        delta = (d - today).days
        if -1 <= delta <= window_days:
            events.append({"name": "FOMC", "date": d, "days": delta})

    def _first_friday(year: int, month: int) -> datetime.date:
        d = datetime.date(year, month, 1)
        ahead = (4 - d.weekday()) % 7
        return d + datetime.timedelta(days=ahead)

    for offset in range(3):
        m = today.month + offset
        y = today.year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        nfp   = _first_friday(y, m)
        delta = (nfp - today).days
        if -1 <= delta <= window_days:
            events.append({"name": "NFP", "date": nfp, "days": delta})

    return events


def _etf_spread_bps(ticker: str) -> float:
    """Estimated bid-ask spread (bps) for position-sizing cost display."""
    _MAJOR  = {"SPY","QQQ","IWM","GLD","TLT","VTI","LQD","HYG","AGG","EEM","EFA","DIA","MDY"}
    _SECTOR = {"XLK","XLF","XLV","XLE","XLI","XLY","XLP","XLU","XLB","XLRE","XLC",
               "GDX","SLV","USO","ICLN","ARKK","SMH","IBB","XBI"}
    t = ticker.upper().split(".")[0]
    if t in _MAJOR:
        return 0.5
    if t in _SECTOR:
        return 2.0
    if ".SI" in ticker.upper():
        return 8.0
    return 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Load data  (outside tabs — shared across all tabs)
# ─────────────────────────────────────────────────────────────────────────────
stats          = get_stats()
total_verified = stats.get("total_verified", 0)
total_logged   = stats.get("total_logged", 0)
pending        = stats.get("pending_verification", 0)
unapplied      = stats.get("unapplied_patterns", 0)
hit_rate       = stats.get("overall_hit_rate", 0)
avg_score      = stats.get("overall_avg_score", 0)

# ─────────────────────────────────────────────────────────────────────────────
# Tab layout
# ─────────────────────────────────────────────────────────────────────────────
_tab_pending, _tab_perf, _tab_postmortem, _tab_system = st.tabs([
    "Pending Review",
    "Performance",
    "Post-Mortem",
    "System",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Pending Review  (KPIs · Verification · Anomaly Review)
# ═══════════════════════════════════════════════════════════════════════════════
with _tab_pending:
    # ── Global macro event warning banner ────────────────────────────────────
    _tab_events = _upcoming_macro_events(window_days=5)
    if _tab_events:
        _ev_names = " · ".join(
            f'{ev["name"]} '
            + ("今日" if ev["days"] == 0 else ("明日" if ev["days"] == 1 else f'{ev["days"]}天后'))
            for ev in _tab_events
        )
        st.warning(
            f"⚠️ **高跳空风险窗口** · 近5天内有重大宏观事件：{_ev_names}\n\n"
            "建议：推迟新入场、收紧止损、降低重仓建议的置信阈值。",
            icon=None,
        )

    _section("System Overview · Alpha Memory Health")

    c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
    c1.metric("Total Logged",       total_logged)
    c2.metric("Verified",           total_verified)
    c3.metric("Pending Verify",     pending)
    c4.metric("Overall Hit Rate",   f"{hit_rate:.0%}" if total_verified else "—")
    c5.metric("Unapplied Patterns", unapplied)

    # Per-tab breakdown
    if stats.get("by_tab"):
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        tab_cols = st.columns(len(stats["by_tab"]), gap="medium")
        for col, (tab, data) in zip(tab_cols, stats["by_tab"].items()):
            with col:
                with st.container(border=True):
                    st.markdown(
                        f'<div style="font-size:0.85rem; text-transform:uppercase; '
                        f'letter-spacing:0.08em; color:var(--muted); font-weight:700;">'
                        f'{tab}</div>'
                        f'<div style="font-family:var(--mono); font-size:1.4rem; '
                        f'font-weight:700; color:var(--primary); margin:0.3rem 0;">'
                        f'{data["hit_rate"]:.0%}</div>'
                        f'<div style="font-size:0.92rem; color:var(--muted);">'
                        f'avg {data["avg"]:.2f} · n={data["count"]}</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown(
        f'<div style="font-size:0.92rem; color:var(--muted); margin-top:0.6rem;">'
        f'Thresholds — Baseline: <b>{BASELINE_HIT_RATE:.0%}</b> · '
        f'Acceptable: <b style="color:#D97706;">{MIN_ACCEPTABLE:.0%}</b> · '
        f'Good: <b style="color:#0284C7;">{GOOD_THRESHOLD:.0%}</b> · '
        f'Excellent: <b style="color:#059669;">{EXCELLENT:.0%}</b> · '
        f'Accuracy scored via <b>Triple-Barrier</b> (TP=1σ · SL=0.7σ · time cap)</div>',
        unsafe_allow_html=True,
    )

    # ── Actions ───────────────────────────────────────────────────────────────
    _section("Actions · Verification & Learning")

    col_v, col_m, col_r = st.columns([1, 1, 1], gap="medium")

    with col_v:
        st.markdown(
            '<div style="font-size:1rem; font-weight:700; color:var(--text); margin-bottom:0.5rem;">'
            'Verify Pending Decisions</div>'
            '<div style="font-size:0.95rem; color:var(--muted); margin-bottom:0.8rem;">'
            'Triple-Barrier verification (TP=1σ / SL=0.7σ), accuracy scoring, AI reflections.</div>',
            unsafe_allow_html=True,
        )
        if st.button(
            f"🔍  Verify {pending} Decision{'s' if pending != 1 else ''}" if pending > 0
            else "🔍  No Pending Decisions",
            type="primary", width='stretch', disabled=(pending == 0),
            help=None if pending > 0 else (
                "当前没有已到期的待验证决策。\n\n"
                "等待窗口：季度(3个月) → 45天；半年(6个月) → 90天。\n"
                "决策需等待对应的最短持有期后才进入验证队列。"
            ),
        ):
            with st.spinner("Fetching market data · Scoring accuracy · Generating reflections..."):
                try:
                    from engine.key_pool import get_pool as _get_pool
                    _vmodel = _get_pool().get_model()
                except Exception:
                    _vmodel = None
                results = verify_pending_decisions(model=_vmodel)
            if results:
                # Summarise which barriers were hit
                _b_tp   = sum(1 for r in results if r.get("barrier") == "tp")
                _b_sl   = sum(1 for r in results if r.get("barrier") == "sl")
                _b_time = sum(1 for r in results if r.get("barrier") == "time")
                st.success(
                    f"✅ 已验证 {len(results)} 条决策（三重障碍法打分）\n\n"
                    f"止盈触发 {_b_tp} 条 · 止损触发 {_b_sl} 条 · 时间障碍 {_b_time} 条"
                )
            else:
                st.warning(
                    "⚠️ 当前没有可验证的决策。\n\n"
                    "**等待窗口**：季度(3个月) → 45天 · 半年(6个月) → 90天\n\n"
                    "所有未验证决策均未达到对应持有期，系统将在时间窗口到达后自动纳入队列。"
                )

    with col_m:
        st.markdown(
            '<div style="font-size:1rem; font-weight:700; color:var(--text); margin-bottom:0.5rem;">'
            'Meta-Agent Analysis</div>'
            '<div style="font-size:0.95rem; color:var(--muted); margin-bottom:0.8rem;">'
            'Detect systematic biases and reinforce effective frameworks.</div>',
            unsafe_allow_html=True,
        )
        if st.button(
            f"⚡  Run Analysis  ({unapplied} pattern{'s' if unapplied != 1 else ''} pending)"
            if unapplied > 0 else "⚡  Run Meta-Agent Analysis",
            width='stretch',
        ):
            with st.spinner("Analysing failure cases · Detecting systematic biases..."):
                patterns = run_meta_agent_analysis(min_samples=3)
            if patterns:
                st.success(f"✅ 发现 {len(patterns)} 个系统性模式，已写入 Learning Log，下次分析时自动注入修正。")
            else:
                st.warning(
                    "⚠️ 数据暂不足以运行 Meta-Agent。\n\n"
                    "**原因**：需要每个（板块 × 宏观制度）组合至少 **3 条已验证决策** 才能识别系统性偏差。\n\n"
                    "**下一步**：待 Verify Pending Decisions 积累足够验证数据后重试。"
                )

    with col_r:
        st.markdown(
            '<div style="font-size:1rem; font-weight:700; color:var(--text); margin-bottom:0.5rem;">'
            'Refresh Stats</div>'
            '<div style="font-size:0.95rem; color:var(--muted); margin-bottom:0.8rem;">'
            'Reload all metrics and patterns from database.</div>',
            unsafe_allow_html=True,
        )
        if st.button("↺  Refresh Dashboard", width='stretch'):
            st.success("✅ Dashboard 数据已从数据库重新加载。")

    # ── One-time migration ────────────────────────────────────────────────────
    with st.expander("🔧 数据库迁移工具", expanded=False):
        st.markdown(
            '<div style="font-size:0.9rem; color:var(--muted); margin-bottom:0.8rem;">'
            'Macro 决策无法经过 Triple-Barrier 价格验证。'
            '此操作将所有旧的 <code>verified=False</code> macro 记录标记为 verified，'
            '避免它们堆积在验证队列中。记录本身保留供历史上下文使用。'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("标记旧 Macro 记录为 Verified", type="secondary"):
            _n = backfill_macro_verified()
            if _n:
                st.success(f"✅ 已更新 {_n} 条 macro 记录（verified=True, accuracy_score=NULL）")
            else:
                st.info("✅ 无需更新：未发现遗留的未验证 macro 记录。")

    # ── Anomaly Review ────────────────────────────────────────────────────────
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    _section("异常记录审核 · Human-in-the-Loop Review")
    st.markdown(
        '<div style="font-size:0.95rem; color:var(--muted); margin-bottom:1rem;">'
        '系统自动标记「高置信度 + 严重失准」的记录供人工判定。'
        '请区分外部冲击（黑天鹅）与真实分析失误，以确保 Meta-Agent 从正确信号中学习。'
        '</div>',
        unsafe_allow_html=True,
    )

    review_records = get_records_needing_review()
    if not review_records:
        st.markdown(
            '<div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:6px; '
            'padding:1rem 1.2rem; color:#166534; font-size:0.95rem;">'
            '✓ 暂无需要人工审核的异常记录。</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="background:#FEF9C3; border:1px solid #FDE047; border-radius:6px; '
            f'padding:0.8rem 1.2rem; color:#713F12; font-size:0.95rem; margin-bottom:1rem;">'
            f'⚠️ 共 <b>{len(review_records)}</b> 条记录需要人工审核</div>',
            unsafe_allow_html=True,
        )
        for rec in review_records:
            with st.container(border=True):
                c_info, c_actions = st.columns([3, 1], gap="medium")
                with c_info:
                    st.markdown(
                        f'<div style="font-size:1rem; font-weight:700; margin-bottom:0.3rem;">'
                        f'{rec["sector_name"]}  ·  {rec["direction"]}  ·  {rec["horizon"]}</div>'
                        f'<div style="font-size:0.9rem; color:var(--muted);">'
                        f'日期: {rec["created_at"].strftime("%Y-%m-%d")}  ·  '
                        f'置信度: <b>{rec["confidence_score"]}</b>/100  ·  '
                        f'准确率评分: <b style="color:#EF4444;">{rec["accuracy_score"]:.0%}</b>  ·  '
                        f'VIX: {rec["vix_level"]}  ·  制度: {rec["macro_regime"]}</div>',
                        unsafe_allow_html=True,
                    )
                    if rec["reflection"]:
                        st.caption(f'AI反思: {rec["reflection"][:200]}')
                with c_actions:
                    if st.button(
                        "🌪️ 黑天鹅排除",
                        key=f"bs_{rec['id']}",
                        width='stretch',
                        help="外部不可预见冲击导致失误，排除出学习数据",
                    ):
                        if set_human_label(rec["id"], "black_swan"):
                            st.success("已标记为黑天鹅，排除学习")
                            st.rerun()
                    if st.button(
                        "⚠️ 确认分析失误",
                        key=f"ae_{rec['id']}",
                        width='stretch',
                        help="分析逻辑本身存在问题，纳入 Meta-Agent 重点学习",
                    ):
                        if set_human_label(rec["id"], "analysis_error"):
                            st.success("已确认分析失误，已纳入学习")
                            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Post-Mortem  (Learning + Failure Attribution + Signal Analysis)
# ═══════════════════════════════════════════════════════════════════════════════
with _tab_postmortem:
    _section("Learning Patterns · Strengths & Biases")

    all_patterns   = get_learning_patterns()
    _dormant_count = get_dormant_pattern_count()
    if _dormant_count > 0:
        _dm_bg = "rgba(255,255,255,0.04)" if theme.is_dark() else "#F8FAFC"
        _dm_bd = "rgba(255,255,255,0.10)" if theme.is_dark() else "#CBD5E1"
        _dm_fg = "#7D8590" if theme.is_dark() else "#64748B"
        st.markdown(
            f'<div style="background:{_dm_bg}; border:1px solid {_dm_bd}; border-radius:3px; '
            f'padding:0.55rem 1rem; margin-bottom:0.8rem; color:{_dm_fg}; font-size:0.92rem;">'
            f'💤 <b>{_dormant_count}</b> pattern(s) dormant — '
            f'associated regime unseen for 180+ days. Will auto-revive if regime returns.'
            f'</div>',
            unsafe_allow_html=True,
        )

    if not all_patterns:
        _empty_bg = "rgba(255,255,255,0.04)" if theme.is_dark() else "#F8FAFC"
        _empty_bd = "rgba(255,255,255,0.10)" if theme.is_dark() else "#CBD5E1"
        _empty_fg = "#7D8590" if theme.is_dark() else "#94A3B8"
        st.markdown(
            f'<div style="background:{_empty_bg}; border:1px dashed {_empty_bd}; border-radius:3px; '
            f'padding:2rem; text-align:center; color:{_empty_fg}; font-size:1.05rem;">'
            'No learning patterns yet. Run Meta-Agent Analysis after accumulating verified decisions.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        strengths = [p for p in all_patterns if p["type"] == "strength"]
        biases    = [p for p in all_patterns if p["type"] != "strength"]

        def _render_pattern_cards(pattern_list: list, icon: str, border_color: str) -> None:
            for p in pattern_list:
                tag    = p["sector"] or "All Sectors"
                regime = p["regime"] or "All Regimes"
                with st.container(border=True):
                    c1, c2 = st.columns([6, 1], gap="small")
                    with c1:
                        st.markdown(
                            f'<div style="font-size:1.05rem; color:#1E293B; line-height:1.6;">'
                            f'{icon} <b style="border-left:3px solid {border_color}; '
                            f'padding-left:0.5rem;">[{tag} · {regime}]</b></div>'
                            f'<div style="font-size:1rem; color:#334155; margin-top:0.4rem; '
                            f'line-height:1.7; padding-left:1.1rem;">{p["description"]}</div>'
                            f'<div style="font-size:0.88rem; color:#94A3B8; margin-top:0.3rem; '
                            f'padding-left:1.1rem;">Based on {p["samples"]} verified sample(s)</div>',
                            unsafe_allow_html=True,
                        )
                    with c2:
                        if st.button(
                            "Applied ✓",
                            key=f"apply_{p['id']}",
                            width='stretch',
                        ):
                            mark_pattern_applied(p["id"])
                            st.success("Marked as applied.")
                            st.rerun()

        if strengths:
            st.markdown(
                '<div style="font-size:1rem; font-weight:700; color:#059669; margin:0.5rem 0 0.5rem;">'
                '✅  Effective Frameworks — Reinforce in prompts</div>',
                unsafe_allow_html=True,
            )
            _render_pattern_cards(strengths, "✅", "#059669")

        if biases:
            st.markdown(
                '<div style="font-size:1rem; font-weight:700; color:#D97706; margin:1rem 0 0.5rem;">'
                '⚠  Detected Biases — Correct in prompts</div>',
                unsafe_allow_html=True,
            )
            _render_pattern_cards(biases, "⚠", "#D97706")

    # ── Semantic Drift Monitor ────────────────────────────────────────────────
    _section("Semantic Drift Monitor · 规律原文时序检查")
    st.markdown(
        '<div style="font-size:0.85rem; color:var(--muted); margin-bottom:0.8rem;">'
        '按时间倒序展示所有 Meta-Agent 生成的规律原文（含已归档 / 休眠）。'
        '用于人工检查表述是否随时间变得模糊或相互矛盾。'
        '</div>',
        unsafe_allow_html=True,
    )

    _raw_logs = get_learning_log_raw(limit=200)

    if not _raw_logs:
        st.info("暂无规律记录。Meta-Agent 运行后将在此显示。")
    else:
        # ── Filters ───────────────────────────────────────────────────────────
        _dm1, _dm2, _dm3 = st.columns([2, 2, 2], gap="small")
        _dm_sectors = ["（全部）"] + sorted({r["sector"] for r in _raw_logs})
        _dm_regimes = ["（全部）"] + sorted({r["regime"] for r in _raw_logs})
        _dm_types   = ["（全部）", "strength", "bias"]

        with _dm1:
            _dm_f_sector = st.selectbox("板块", _dm_sectors, key="dm_sector")
        with _dm2:
            _dm_f_regime = st.selectbox("宏观周期", _dm_regimes, key="dm_regime")
        with _dm3:
            _dm_f_type = st.selectbox("类型", _dm_types, key="dm_type")

        _filtered = [
            r for r in _raw_logs
            if (_dm_f_sector == "（全部）" or r["sector"] == _dm_f_sector)
            and (_dm_f_regime == "（全部）" or r["regime"] == _dm_f_regime)
            and (_dm_f_type   == "（全部）" or r["type"]   == _dm_f_type)
        ]

        # ── Contradiction detector ─────────────────────────────────────────
        # Flag: same sector × regime has both a "strength" and a "bias" entry
        _cell_types: dict[str, set] = {}
        for r in _filtered:
            key = f"{r['sector']}|{r['regime']}"
            _cell_types.setdefault(key, set()).add(r["type"])
        _conflict_cells = {k for k, v in _cell_types.items() if "strength" in v and "bias" in v}

        if _conflict_cells:
            st.markdown(
                '<div style="background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.3); '
                'border-radius:3px; padding:0.55rem 1rem; margin-bottom:0.8rem; font-size:0.88rem; '
                'color:var(--danger);">'
                '⚠ 检测到潜在矛盾：以下 板块×周期 同时存在 strength 和 bias 规律，建议人工核查：<br>'
                + "".join(f"<b>{c.replace('|', ' · ')}</b>　" for c in _conflict_cells)
                + '</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div style="font-size:0.82rem; color:var(--muted); margin-bottom:0.5rem;">'
            f'显示 {len(_filtered)} / {len(_raw_logs)} 条</div>',
            unsafe_allow_html=True,
        )

        # ── Entry list ─────────────────────────────────────────────────────
        for _r in _filtered:
            _is_conflict = f"{_r['sector']}|{_r['regime']}" in _conflict_cells
            _tag_color   = (
                "var(--success)" if _r["type"] == "strength"
                else "var(--warn)"
            )
            _status_tags = []
            if _r["applied"]:
                _status_tags.append('<span style="color:var(--muted);font-size:0.75rem;">[已归档]</span>')
            if _r["dormant"]:
                _status_tags.append('<span style="color:var(--muted);font-size:0.75rem;">[休眠]</span>')
            if _is_conflict:
                _status_tags.append('<span style="color:var(--danger);font-size:0.75rem;">[⚠ 矛盾]</span>')

            st.markdown(
                f'<div style="border-left:3px solid {_tag_color}; padding:0.45rem 0.9rem; '
                f'margin-bottom:0.45rem; background:var(--card); border-radius:0 3px 3px 0;">'
                f'<div style="font-size:0.78rem; color:var(--muted); margin-bottom:0.2rem;">'
                f'{_r["created_at"]} · {_r["sector"]} · {_r["regime"]} · '
                f'<b style="color:{_tag_color};">{_r["type"]}</b> · '
                f'n={_r["samples"]} {"".join(_status_tags)}</div>'
                f'<div style="font-size:0.9rem; color:var(--text); line-height:1.6;">'
                f'{_r["description"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── News Routing Weights ──────────────────────────────────────────────────
    _section("News Routing Weights · Learned Relevance by Sector × Regime")

    SECTORS = list(_DEFAULT_WEIGHTS.keys())
    REGIMES = ["高波动/危机", "震荡期", "温和波动", "低波动/牛市"]
    NEWS_CATEGORIES = [
        "央行声明", "OPEC动态", "供应链", "地缘政治", "科技监管",
        "PMI数据", "CPI数据", "就业数据", "零售数据", "信贷数据",
        "监管政策", "FDA动态", "政策医改", "能源政策",
    ]

    col_s, col_r2 = st.columns([1, 1], gap="medium")
    with col_s:
        sel_sector = st.selectbox("Sector", SECTORS, key="admin_sector")
    with col_r2:
        sel_regime = st.selectbox("Macro Regime", REGIMES, key="admin_regime")

    weights = get_news_routing_weights(sel_sector, sel_regime)

    st.markdown(
        f'<div style="font-size:0.95rem; color:var(--muted); margin-bottom:0.8rem;">'
        f'Showing weights for <b>{sel_sector}</b> in <b>{sel_regime}</b> environment. '
        f'Higher weight = higher retrieval priority.</div>',
        unsafe_allow_html=True,
    )

    weight_cols = st.columns(3, gap="medium")
    updated_weights: dict[str, float] = {}
    for i, cat in enumerate(NEWS_CATEGORIES):
        current = weights.get(cat, _DEFAULT_WEIGHTS.get(sel_sector, {}).get(cat, 0.3))
        with weight_cols[i % 3]:
            updated_weights[cat] = st.slider(
                cat, min_value=0.0, max_value=1.0,
                value=round(current, 2), step=0.05,
                key=f"weight_{sel_sector}_{sel_regime}_{cat}",
            )

    if st.button("💾  Save Weight Adjustments", type="primary"):
        saved = []
        for cat, new_w in updated_weights.items():
            old_w = weights.get(cat, 0.5)
            if abs(new_w - old_w) > 0.01:
                update_news_routing_weight(
                    sector_name=sel_sector,
                    macro_regime=sel_regime,
                    news_category=cat,
                    accuracy_when_used=new_w,
                    accuracy_when_not_used=0.5,
                )
                saved.append(f"{cat}: {old_w:.2f} → {new_w:.2f}")
        if saved:
            st.success(
                f"✅ 已保存 {len(saved)} 项权重调整（{sel_sector} · {sel_regime}）：\n\n"
                + "\n".join(f"- {s}" for s in saved)
            )
        else:
            st.info("ℹ️ 所有权重与当前值相同，无需保存。")

    # ── Failure Mode Taxonomy ─────────────────────────────────────────────────
    _section("Failure Mode Taxonomy · 预测失败归因")

    from engine.memory import get_failure_mode_stats
    _fm_stats = get_failure_mode_stats()
    _fm_labels = _fm_stats.get("fm_labels", {})
    _fm_colors = {
        "FM-A": ("#7C3AED", "rgba(124,58,237,0.12)"),   # 逻辑退化 — purple
        "FM-B": ("#DC2626", "rgba(220,38,38,0.12)"),    # 过度自信 — red
        "FM-C": ("#D97706", "rgba(217,119,6,0.12)"),    # 信号污染 — amber
        "FM-D": ("#0284C7", "rgba(2,132,199,0.12)"),    # 制度误判 — blue
    }

    _tv = _fm_stats["total_verified"]
    _tf = _fm_stats["total_failed"]
    _by_mode = _fm_stats["by_mode"]
    _unlabelled = _fm_stats["unlabelled"]

    if _tv == 0:
        st.markdown(
            '<div style="padding:1.5rem; text-align:center; color:var(--muted); '
            'border:1px dashed var(--border); border-radius:4px;">'
            '暂无已验证决策，运行验证后此处将显示故障模式分布。</div>',
            unsafe_allow_html=True,
        )
    else:
        # Summary row
        _fm_cols = st.columns(5)
        for _ci, (_label, _val, _col) in enumerate([
            ("已验证", _tv, "#10B981"),
            ("预测失败", _tf, "#EF4444"),
            *[(f"{fm} {_fm_labels.get(fm,'')}", _by_mode.get(fm, 0), _fm_colors[fm][0])
              for fm in ("FM-A", "FM-B", "FM-C", "FM-D")],
        ]):
            with _fm_cols[_ci]:
                st.markdown(
                    f'<div style="background:var(--card); border:1px solid var(--border); '
                    f'border-radius:4px; padding:0.7rem 0.8rem; text-align:center;">'
                    f'<div style="font-size:1.5rem; font-weight:700; color:{_col};">{_val}</div>'
                    f'<div style="font-size:0.78rem; color:var(--muted); margin-top:0.2rem;">{_label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Per-FM breakdown cards
        if _by_mode:
            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
            _fm_desc = {
                "FM-A": "LCS 镜像测试失败——结论与输入无关，逻辑完全退化",
                "FM-B": "置信度 ≥ 85 但预测明确错误——过度自信未被红队拦截",
                "FM-C": "RSI / Bollinger 等短期技术信号出现在归因驱动因素中",
                "FM-D": "同一板块 × 制度连续 3 次失败——制度标签可能需要重新校准",
            }
            for fm, cnt in sorted(_by_mode.items(), key=lambda x: -x[1]):
                _fg, _bg = _fm_colors.get(fm, ("#6B7280", "rgba(107,114,128,0.1)"))
                pct = cnt / _tf * 100 if _tf else 0
                st.markdown(
                    f'<div style="background:{_bg}; border-left:3px solid {_fg}; '
                    f'border-radius:3px; padding:0.65rem 1rem; margin-bottom:0.5rem; '
                    f'display:flex; justify-content:space-between; align-items:center;">'
                    f'<div>'
                    f'<span style="font-weight:700; color:{_fg}; font-size:0.9rem;">{fm}</span>'
                    f'<span style="color:var(--muted); font-size:0.82rem; margin-left:0.6rem;">'
                    f'{_fm_labels.get(fm,"")} — {_fm_desc.get(fm,"")}</span>'
                    f'</div>'
                    f'<span style="font-weight:700; font-size:1rem; color:{_fg};">'
                    f'{cnt} ({pct:.0f}%)</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        if _unlabelled > 0:
            st.markdown(
                f'<div style="color:var(--muted); font-size:0.82rem; margin-top:0.4rem;">'
                f'另有 {_unlabelled} 条失败记录无法归入上述类别（可能是数据缺失或边界情况）。'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Signal Significance · Block Bootstrap Permutation ────────────────────
    _section("Signal Significance · Block Bootstrap Permutation")

    st.markdown(
        '<div style="font-size:0.85rem; color:var(--muted); margin-bottom:1rem; line-height:1.7;">'
        '对训练集中每个 <b>板块 × 宏观制度</b> 单元格运行移块自举置换检验（Block Bootstrap Permutation），'
        '验证观测准确率是否显著优于随机基准。<br>'
        '多重检验校正：Romano-Wolf（所有板块同时检验，FWER = 0.05）。<br>'
        '<b style="color:var(--warn);">⚠ "数据不足"与"不显著"是完全不同的状态</b> — '
        '前者表示检验未运行（n &lt; 50），后者表示检验运行后未通过。'
        '</div>',
        unsafe_allow_html=True,
    )

    _perm_fast = st.toggle(
        "快速模式（1,000 次置换，仅用于调试）",
        value=False,
        key="perm_fast_mode",
    )

    if st.button("▶  运行置换检验", type="primary", key="run_permutation_btn"):
        _n_perm = 1_000 if _perm_fast else 10_000
        with st.spinner(f"运行 {_n_perm:,} 次块自举置换检验…"):
            try:
                from engine.memory import get_permutation_report
                _perm_results = get_permutation_report(n_permutations=_n_perm)
                st.session_state["perm_results"] = _perm_results
            except Exception as _pe:
                st.error(f"置换检验失败：{_pe}")

    _perm_data = st.session_state.get("perm_results")

    if _perm_data:
        _sig   = [r for r in _perm_data if r["status"] == "significant"]
        _nosig = [r for r in _perm_data if r["status"] == "not_significant"]
        _insuf = [r for r in _perm_data if r["status"] == "insufficient_data"]

        # Summary row
        _sc1, _sc2, _sc3 = st.columns(3, gap="medium")
        with _sc1:
            st.metric("✅ 显著信号", len(_sig), help="p < 校正后阈值")
        with _sc2:
            st.metric("✗ 不显著", len(_nosig), help="n ≥ 50，但 p ≥ 阈值")
        with _sc3:
            st.metric("⏳ 数据不足", len(_insuf), help=f"n < 100，检验未运行")

        # Per-cell cards
        for _status_group, _label, _color in [
            (_sig,   "✅ 显著信号（p < 阈值）",   "#059669"),
            (_nosig, "✗  不显著",                 "#D97706"),
            (_insuf, "⏳ 数据不足（n < 100）",    "#6B7280"),
        ]:
            if not _status_group:
                continue
            st.markdown(
                f'<div style="font-size:0.92rem; font-weight:700; color:{_color}; '
                f'margin: 0.8rem 0 0.4rem;">{_label}</div>',
                unsafe_allow_html=True,
            )
            for _r in _status_group:
                _bg = "rgba(255,255,255,0.03)" if theme.is_dark() else "#F8FAFC"
                _bd = "rgba(255,255,255,0.08)" if theme.is_dark() else "#E2E8F0"

                if _r["status"] == "insufficient_data":
                    _pct = int(_r["progress_pct"] * 100)
                    _bar_w = max(4, _pct)
                    _detail = (
                        f'n = {_r["n"]} / {_r["n_needed"]} 需要'
                        f'<div style="background:{_bd}; border-radius:3px; height:6px; '
                        f'margin-top:0.3rem; overflow:hidden;">'
                        f'<div style="background:{_color}; width:{_bar_w}%; height:100%;"></div></div>'
                        f'<div style="font-size:0.78rem; color:var(--muted); margin-top:0.2rem;">'
                        f'还需 {_r["n_needed"] - _r["n"]} 条已验证训练集决策</div>'
                    )
                elif _r["status"] == "significant":
                    _detail = (
                        f'p = <b>{_r["p_value"]:.4f}</b>  &lt;  阈值 {_r["threshold"]:.4f}  '
                        f'· 准确率 {_r["observed_accuracy"]:.1%}  · n = {_r["n"]}'
                    )
                else:
                    _detail = (
                        f'p = {_r["p_value"]:.4f}  ≥  阈值 {_r["threshold"]:.4f}  '
                        f'· 准确率 {_r["observed_accuracy"]:.1%}  · n = {_r["n"]}'
                    )

                st.markdown(
                    f'<div style="background:{_bg}; border:1px solid {_bd}; border-radius:3px; '
                    f'padding:0.5rem 0.9rem; margin-bottom:0.35rem; '
                    f'border-left:3px solid {_color};">'
                    f'<div style="font-size:0.88rem; font-weight:600; color:var(--text);">'
                    f'{_r["sector"]} · {_r["regime"]}</div>'
                    f'<div style="font-size:0.84rem; color:var(--muted); margin-top:0.2rem;">'
                    f'{_detail}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    elif _perm_data is not None:
        st.info("训练集中暂无已验证决策，置换检验无法运行。")

    # ── Signal Invalidation Risk Calibration ─────────────────────────────────
    st.divider()
    _section("Signal Invalidation Risk · Calibration")
    st.caption(
        "LLM 对信号失效概率的自评（0-100）vs 实际结果校准。"
        "理想状态：高 invalidation_risk → 实际胜率低；低 risk → 胜率高。"
    )

    with SessionFactory() as _sir_s:
        _sir_all = (
            _sir_s.query(DecisionLog)
            .filter(
                DecisionLog.signal_invalidation_risk.isnot(None),
                DecisionLog.tab_type == "sector",
                (DecisionLog.superseded == False) | DecisionLog.superseded.is_(None),
                (DecisionLog.is_backtest == False) | DecisionLog.is_backtest.is_(None),
            )
            .all()
        )
        _sir_verified = [r for r in _sir_all if r.verified and r.accuracy_score is not None]

    _sir_c1, _sir_c2 = st.columns(2, gap="large")

    with _sir_c1:
        st.markdown(
            '<div style="font-size:0.82rem; font-weight:600; color:var(--muted); '
            'text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.5rem;">'
            'Distribution  ·  All Decisions</div>',
            unsafe_allow_html=True,
        )
        if not _sir_all:
            st.caption("暂无含 invalidation_risk 的决策记录（需重新分析后入库）。")
        else:
            _buckets_dist = {"0–24": 0, "25–49": 0, "50–74": 0, "75–100": 0}
            for r in _sir_all:
                v = r.signal_invalidation_risk
                if v < 25:   _buckets_dist["0–24"]   += 1
                elif v < 50: _buckets_dist["25–49"]  += 1
                elif v < 75: _buckets_dist["50–74"]  += 1
                else:        _buckets_dist["75–100"] += 1
            _dist_df = pd.DataFrame(
                {"Risk Bucket": list(_buckets_dist.keys()),
                 "Count":       list(_buckets_dist.values())}
            ).set_index("Risk Bucket")
            st.bar_chart(_dist_df, color=["#6366F1"], height=180)
            st.caption(f"共 {len(_sir_all)} 条决策含此字段（含 {len(_sir_verified)} 条已验证）")

    with _sir_c2:
        st.markdown(
            '<div style="font-size:0.82rem; font-weight:600; color:var(--muted); '
            'text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.5rem;">'
            'Calibration  ·  Verified Only</div>',
            unsafe_allow_html=True,
        )
        if len(_sir_verified) < 5:
            st.caption(f"已验证样本不足（n={len(_sir_verified)} < 5），校准表待积累数据后显示。")
        else:
            _cal_buckets: dict[str, list] = {"0–24": [], "25–49": [], "50–74": [], "75–100": []}
            for r in _sir_verified:
                v = r.signal_invalidation_risk
                key = "0–24" if v < 25 else ("25–49" if v < 50 else ("50–74" if v < 75 else "75–100"))
                _cal_buckets[key].append(r.accuracy_score)

            _cal_rows = []
            for bucket, scores in _cal_buckets.items():
                if not scores:
                    continue
                _n   = len(scores)
                _wr  = sum(1 for s in scores if s >= EXCELLENT) / _n
                _avg = sum(scores) / _n
                _cal_rows.append({
                    "Risk Bucket": bucket,
                    "n": _n,
                    "Win Rate": f"{_wr:.0%}",
                    "Avg Accuracy": f"{_avg:.3f}",
                })
            if _cal_rows:
                st.dataframe(pd.DataFrame(_cal_rows), hide_index=True, use_container_width=True)
                st.caption(
                    "Win Rate = accuracy_score ≥ 0.75 的比例。"
                    "校准成立条件：从 0–24 到 75–100，Win Rate 应单调递减。"
                )

with _tab_postmortem:
    st.divider()
    _section("Failure Attribution · 失败原因分类")

    # ── Summary stats ────────────────────────────────────────────────────────
    _fa_stats = get_failure_attribution_stats()
    _fa_total = _fa_stats["total_failures"]
    _fa_unattr = _fa_stats["unattributed"]

    if _fa_total == 0:
        st.info("暂无已验证的失败记录（accuracy_score < 0.5）。当 Clean Zone 样本积累后将在此显示。")
    else:
        _sc1, _sc2, _sc3 = st.columns(3)
        _sc1.metric("失败记录总数", _fa_total)
        _sc2.metric("已归因", _fa_total - _fa_unattr)
        _sc3.metric("待归因", _fa_unattr,
                    delta=f"-{_fa_unattr}" if _fa_unattr else None,
                    delta_color="inverse")

        if _fa_stats["by_type"]:
            st.markdown("**归因分布**")
            _type_df = pd.DataFrame([
                {
                    "失败类型": _FAILURE_TYPE_LABELS.get(ft, ft),
                    "数量": cnt,
                    "占比": f"{cnt / max(_fa_total - _fa_unattr, 1):.0%}",
                }
                for ft, cnt in sorted(_fa_stats["by_type"].items(),
                                       key=lambda x: -x[1])
            ])
            st.dataframe(_type_df, use_container_width=True, hide_index=True)

        st.divider()

        # ── Unattributed failures — annotation UI ────────────────────────────
        _unattr_records = get_unattributed_failures(min_age_days=20)
        if not _unattr_records:
            st.success("✅ 所有失败记录已完成归因。")
        else:
            st.markdown(
                f"**{len(_unattr_records)} 条失败记录待归因**  "
                f"<span style='color:var(--muted);font-size:0.88em;'>"
                f"（仅显示 accuracy_score < 0.5 且已验证的记录）</span>",
                unsafe_allow_html=True,
            )
            for _fr in _unattr_records:
                with st.container(border=True):
                    _drift_badge = (
                        " 🔄 制度漂移" if _fr["regime_drifted"] else ""
                    )
                    st.markdown(
                        f'<div style="font-weight:700;font-size:1rem;">'
                        f'{_fr["sector_name"] or "全球宏观"}  ·  '
                        f'{_fr["direction"] or "—"}  ·  '
                        f'置信度 {_fr["confidence_score"] or "—"}/100'
                        f'{_drift_badge}</div>'
                        f'<div style="color:var(--muted);font-size:0.88em;">'
                        f'日期: {_fr["created_at"].strftime("%Y-%m-%d")}  ·  '
                        f'制度: {_fr["macro_regime"] or "—"}  ·  '
                        f'accuracy: <b style="color:#EF4444;">'
                        f'{_fr["accuracy_score"]:.0%}</b>  ·  '
                        f'FM: {_fr["failure_mode"] or "—"}</div>',
                        unsafe_allow_html=True,
                    )
                    if _fr["economic_logic"]:
                        st.caption(f'原始逻辑: {_fr["economic_logic"][:200]}')

                    _ac1, _ac2 = st.columns([1, 2])
                    _ft_opts = list(_FAILURE_TYPE_LABELS.keys())
                    _ft_labels = [_FAILURE_TYPE_LABELS[k] for k in _ft_opts]
                    _ft_sel = _ac1.selectbox(
                        "失败类型",
                        options=_ft_opts,
                        format_func=lambda k: _FAILURE_TYPE_LABELS[k],
                        key=f"ft_sel_{_fr['id']}",
                        help=(
                            "hypothesis=假设方向错  data=数据问题  "
                            "regime_drift=制度切换  robustness=样本外失效  "
                            "evaluation=验证参数问题  execution=执行偏差"
                        ),
                    )
                    _ft_note = _ac2.text_input(
                        "备注（可选）",
                        key=f"ft_note_{_fr['id']}",
                        placeholder="简要说明失败原因，将用于 SkillLibrary known_failures",
                    )
                    if st.button(
                        "✅ 确认归因",
                        key=f"ft_confirm_{_fr['id']}",
                        type="primary",
                    ):
                        if set_failure_attribution(_fr["id"], _ft_sel, _ft_note):
                            st.success(
                                f"已将 #{_fr['id']} 归因为「{_FAILURE_TYPE_LABELS[_ft_sel]}」"
                            )
                            # Attempt skill recompression for this cell
                            try:
                                from engine.key_pool import get_pool as _kp_get
                                _skill_model = _kp_get().get_model()
                                _skill_sector = _fr.get("sector_name") or ""
                                _skill_regime = _fr.get("macro_regime") or ""
                                if _skill_sector and _skill_regime:
                                    from engine.memory import maybe_update_skill
                                    _updated = maybe_update_skill(_skill_model, _skill_sector, _skill_regime)
                                    if _updated:
                                        st.caption(f"↺ Skill 已更新：{_skill_sector} × {_skill_regime}")
                            except Exception:
                                pass
                            st.rerun()
                        else:
                            st.error("归因写入失败，请检查 ID")

# ── Post-Mortem: Human vs LCS Pre-scoring Comparison ─────────────────────────
with _tab_postmortem:
    st.divider()
    _section("人工预评 vs AI (LCS) 预测力对比")
    with st.expander("📊 Human vs LCS 信号质量分析", expanded=False):
        st.markdown(
            '<div style="font-size:0.88rem; color:var(--muted); margin-bottom:1rem;">'
            '对比人工预评（<code>human_label: pre_*</code>）与 LCS（AI 逻辑一致性评分）'
            '对 Triple-Barrier 实证结果的预测力。<br>'
            '目标：回答「LCS 是否有真实信息量，还是人工判断更可靠」。<br>'
            '所需样本：同时具备人工预评 + Triple-Barrier 验证结果的 Clean Zone 记录，建议 n ≥ 15。'
            '</div>',
            unsafe_allow_html=True,
        )
        with SessionFactory() as _lcs_s:
            _dual_signal = (
                _lcs_s.query(DecisionLog)
                .filter(
                    DecisionLog.verified == True,
                    DecisionLog.accuracy_score.isnot(None),
                    DecisionLog.decision_date >= CLEAN_ZONE_START,
                    DecisionLog.human_label.in_(["pre_strong", "pre_uncertain", "pre_poor"]),
                    (DecisionLog.is_backtest == False) | DecisionLog.is_backtest.is_(None),
                ).all()
            )
        _MIN_N_LCS = 15
        if len(_dual_signal) < _MIN_N_LCS:
            st.info(
                f"当前双信号样本：**{len(_dual_signal)} / {_MIN_N_LCS}** 条。"
                f"当 Clean Zone 中同时具备人工预评 + Triple-Barrier 验证结果的记录达到 {_MIN_N_LCS} 条后自动展示。"
            )
        else:
            _LCS_LABEL_MAP = {"pre_strong": "逻辑清晰", "pre_uncertain": "有疑虑", "pre_poor": "明显缺陷"}
            _lcs_rows = [{"人工预评": _LCS_LABEL_MAP.get(r.human_label, r.human_label),
                          "LCS通过": "是" if r.lcs_passed else ("否" if r.lcs_passed is False else "未评"),
                          "accuracy": r.accuracy_score}
                         for r in _dual_signal]
            _lcs_df = pd.DataFrame(_lcs_rows)
            st.markdown("**① 人工预评 → 实际结果**")
            _h_tbl = (_lcs_df.groupby("人工预评")["accuracy"]
                      .agg(n="count", avg_acc="mean", win_rate=lambda x: (x >= 0.75).mean())
                      .reset_index()
                      .rename(columns={"n": "样本数", "avg_acc": "平均分", "win_rate": "胜率(≥0.75)"}))
            _h_tbl["平均分"] = _h_tbl["平均分"].round(3)
            _h_tbl["胜率(≥0.75)"] = (_h_tbl["胜率(≥0.75)"] * 100).round(1).astype(str) + "%"
            st.dataframe(_h_tbl, use_container_width=True, hide_index=True)
            st.markdown("**② LCS 评估 → 实际结果**")
            _lcs_grp2 = _lcs_df[_lcs_df["LCS通过"] != "未评"]
            if len(_lcs_grp2) >= 5:
                _l_tbl = (_lcs_grp2.groupby("LCS通过")["accuracy"]
                          .agg(n="count", avg_acc="mean", win_rate=lambda x: (x >= 0.75).mean())
                          .reset_index()
                          .rename(columns={"n": "样本数", "avg_acc": "平均分", "win_rate": "胜率(≥0.75)"}))
                _l_tbl["平均分"] = _l_tbl["平均分"].round(3)
                _l_tbl["胜率(≥0.75)"] = (_l_tbl["胜率(≥0.75)"] * 100).round(1).astype(str) + "%"
                st.dataframe(_l_tbl, use_container_width=True, hide_index=True)
            else:
                st.caption("LCS 已评估样本不足 5 条。")
            st.markdown("**③ 预测方向一致性**")
            _agree2 = sum(1 for r in _dual_signal
                          if (r.human_label == "pre_strong" and r.lcs_passed is True)
                          or (r.human_label == "pre_poor"   and r.lcs_passed is False))
            _disagree2 = sum(1 for r in _dual_signal
                             if (r.human_label == "pre_strong" and r.lcs_passed is False)
                             or (r.human_label == "pre_poor"   and r.lcs_passed is True))
            _n_both2 = len([r for r in _dual_signal if r.lcs_passed is not None])
            if _n_both2 > 0:
                st.markdown(
                    f"人工与 LCS **一致**：{_agree2} 条 &emsp; **相反**：{_disagree2} 条 "
                    f"&emsp; 一致率：**{_agree2/_n_both2:.0%}**\n\n"
                    f"> 若一致率低，说明两个信号捕捉了不同维度的质量——保留两者独立观察价值更大。",
                    unsafe_allow_html=True,
                )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Performance  (Clean Zone Evidence + Decision History + Decision Log)
# ═══════════════════════════════════════════════════════════════════════════════
with _tab_perf:
    _section("Decision History · Verified Records")

    history = stats.get("history", [])
    if not history:
        _empty_bg = "rgba(255,255,255,0.04)" if theme.is_dark() else "#F8FAFC"
        _empty_bd = "rgba(255,255,255,0.10)" if theme.is_dark() else "#CBD5E1"
        _empty_fg = "#7D8590" if theme.is_dark() else "#94A3B8"
        st.markdown(
            f'<div style="background:{_empty_bg}; border:1px dashed {_empty_bd}; border-radius:3px; '
            f'padding:2rem; text-align:center; color:{_empty_fg}; font-size:1.05rem;">'
            'No verified decisions yet.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        def _edit_magnitude(ratio) -> str:
            if ratio is None:
                return "—"
            if ratio < 0.05:
                return "none"
            if ratio < 0.25:
                return "minor"
            if ratio < 0.60:
                return "moderate"
            return "substantial"

        rows = []
        for d in history:
            r5  = f"{d['return_5d']:+.1%}"  if d.get("return_5d")  is not None else "—"
            r20 = f"{d['return_20d']:+.1%}" if d.get("return_20d") is not None else "—"
            sc  = f"{d['score']:.2f}"       if d.get("score")       is not None else "—"
            _src = d.get("source", "ai_drafted")
            _mag = _edit_magnitude(d.get("edit_ratio")) if _src == "human_edited" else "—"
            rows.append({
                "Date":       d["date"],
                "Tab":        d["tab"],
                "Sector":     d["sector"],
                "Direction":  d["direction"],
                "5d Return":  r5,
                "20d Return": r20,
                "Score":      sc,
                "Source":     _src,
                "Edit":       _mag,
                "Verdict":    d.get("verdict", "—"),
            })

        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            width='stretch',
            hide_index=True,
            column_config={
                "Score":   st.column_config.ProgressColumn("Score", min_value=0, max_value=1),
                "Source":  st.column_config.TextColumn("Source", help="ai_drafted | human_edited | human_initiated"),
                "Edit":    st.column_config.TextColumn("Edit Magnitude", help="none / minor / moderate / substantial (Levenshtein-based)"),
                "Verdict": st.column_config.TextColumn("Verdict"),
            },
        )

        if history and history[0].get("reflection"):
            with st.expander("Latest Reflection · AI Self-Analysis", expanded=False):
                st.markdown(
                    f'<div class="decision-card">{history[0]["reflection"]}</div>',
                    unsafe_allow_html=True,
                )

# ── Performance Tab: Clean Zone Evidence ──────────────────────────────────────
with _tab_perf:
    st.divider()
    _section("Clean Zone · Performance Evidence")
    st.markdown(
        '<div style="font-size:0.85rem; color:var(--muted); margin-bottom:1.2rem;">'
        '唯一有效绩效证据 · 决策日期 ≥ 2025-04-01 · LLM 无历史预知</div>',
        unsafe_allow_html=True,
    )

    def _wilson_ci(n: int, k: int, z: float = 1.96) -> tuple:
        if n == 0:
            return 0.0, 1.0
        p = k / n
        denom  = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        margin = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / denom
        return max(0.0, center - margin), min(1.0, center + margin)

    _cz2        = get_clean_zone_stats()
    _clean2     = _cz2.get("clean_b", {})
    _ts2        = get_clean_zone_time_series()
    _cz2_n      = _clean2.get("n", 0)
    _cz2_hr     = _clean2.get("hit_rate")
    _cz2_acc    = _clean2.get("avg_accuracy")
    _cz2_brier  = _clean2.get("brier_score")
    _cz2_bp     = _clean2.get("binom_pvalue")
    _cz2_ci_lo  = _clean2.get("binom_ci_lo")
    _cz2_ci_hi  = _clean2.get("binom_ci_hi")
    _cz2_lcs    = _clean2.get("lcs_pass_rate")

    _wins2       = int(round((_cz2_hr or 0) * _cz2_n))
    _wci2_lo, _wci2_hi = _wilson_ci(_cz2_n, _wins2) if _cz2_n > 0 else (0.0, 1.0)
    _wci2_str    = f"95% CI  [{_wci2_lo:.0%}, {_wci2_hi:.0%}]" if _cz2_n > 0 else "—"
    _wci2_width  = _wci2_hi - _wci2_lo

    _pk1, _pk2, _pk3, _pk4, _pk5 = st.columns(5)
    _pk1.metric("Clean Zone 样本", _cz2_n)
    _pk2.metric("胜率 (≥0.75)",
                f"{_cz2_hr:.0%}" if _cz2_hr is not None else "—",
                help=f"Wilson 95% CI: {_wci2_str}\n区间宽度 {_wci2_width:.0%}")
    _pk3.metric("平均准确率",   f"{_cz2_acc:.3f}" if _cz2_acc is not None else "—")
    _pk4.metric("Brier Score",  f"{_cz2_brier:.3f}" if _cz2_brier is not None else "N/A")
    _pk5.metric("LCS 通过率",   f"{_cz2_lcs:.0%}" if _cz2_lcs is not None else "N/A")

    # Statistical significance
    _CZ_ACCENT = "#10B981"
    if _cz2_n == 0:
        st.info("尚无 Clean Zone 验证数据。")
    elif _cz2_n < 30:
        _hr_disp2 = f"{_cz2_hr:.0%}" if _cz2_hr is not None else "—"
        st.warning(
            f"**样本不足（n={_cz2_n}）** — 当前胜率 {_hr_disp2} 的 Wilson 95% 置信区间为 **{_wci2_str}**，"
            f"区间宽度 **{_wci2_width:.0%}**，无法区分真实能力与随机噪声。\n\n"
            f"首次有意义的统计检验需要再积累 **{30 - _cz2_n}** 个样本。"
        )
    else:
        _sig2_label = "✓ 显著" if _cz2_bp < 0.05 else ("~ 边缘显著" if _cz2_bp < 0.10 else "✗ 不显著")
        _sig2_color = "#10B981" if _cz2_bp < 0.05 else ("#F59E0B" if _cz2_bp < 0.10 else "#EF4444")
        st.markdown(
            f'<div style="border:1px solid {_sig2_color}; border-radius:8px; '
            f'padding:1rem 1.4rem; display:flex; gap:3rem; align-items:center; margin-bottom:1rem;">'
            f'<div><div style="font-size:0.72rem; color:var(--muted); text-transform:uppercase;">结论</div>'
            f'<div style="font-size:1.5rem; font-weight:800; color:{_sig2_color};">{_sig2_label}</div></div>'
            f'<div><div style="font-size:0.72rem; color:var(--muted); text-transform:uppercase;">p-value</div>'
            f'<div style="font-family:var(--mono); font-size:1.3rem; font-weight:700; color:var(--text);">'
            f'{_cz2_bp:.4f}</div></div>'
            f'<div><div style="font-size:0.72rem; color:var(--muted); text-transform:uppercase;">95% CI (胜率)</div>'
            f'<div style="font-family:var(--mono); font-size:1.3rem; font-weight:700; color:var(--text);">'
            f'[{_cz2_ci_lo:.1%}, {_cz2_ci_hi:.1%}]</div></div>'
            f'<div style="font-size:0.82rem; color:var(--muted);">H₀: 胜率=50% · 单侧 Exact Binomial · n={_cz2_n}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Time series + breakdown
    if _ts2:
        _tsc_l, _tsc_r = st.columns([3, 2], gap="large")
        with _tsc_l:
            st.markdown(
                '<div style="font-size:0.82rem; font-weight:700; color:var(--muted); '
                'text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.5rem;">Monthly Win Rate</div>',
                unsafe_allow_html=True,
            )
            _df_ts2 = pd.DataFrame(_ts2).set_index("month")
            st.line_chart(_df_ts2[["win_rate"]].rename(columns={"win_rate": "胜率"}),
                          color=["#10B981"], height=200)
            st.caption("每月样本量")
            st.bar_chart(_df_ts2[["n"]].rename(columns={"n": "样本数"}),
                         color=["#6366F1"], height=80)
        with _tsc_r:
            st.markdown(
                '<div style="font-size:0.82rem; font-weight:700; color:var(--muted); '
                'text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.5rem;">Breakdown</div>',
                unsafe_allow_html=True,
            )
            with SessionFactory() as _ts_s:
                _ts_recs = (
                    _ts_s.query(DecisionLog)
                    .filter(
                        DecisionLog.verified == True,
                        DecisionLog.accuracy_score.isnot(None),
                        DecisionLog.decision_date >= CLEAN_ZONE_START,
                        (DecisionLog.superseded == False) | DecisionLog.superseded.is_(None),
                        (DecisionLog.is_backtest == False) | DecisionLog.is_backtest.is_(None),
                    ).all()
                )
            if _ts_recs:
                from collections import defaultdict as _ddict
                _by_tab2: dict = _ddict(list)
                for _r in _ts_recs:
                    _by_tab2[_r.tab_type or "unknown"].append(_r.accuracy_score)
                _tab2_rows = [{"模块": t, "n": len(s), "胜率": f"{sum(1 for x in s if x >= EXCELLENT)/len(s):.0%}"}
                              for t, s in sorted(_by_tab2.items())]
                st.caption("按分析模块")
                st.dataframe(pd.DataFrame(_tab2_rows), hide_index=True, use_container_width=True)
                _by_reg2: dict = _ddict(list)
                for _r in _ts_recs:
                    _by_reg2[_r.macro_regime or "未知"].append(_r.accuracy_score)
                _reg2_rows = [{"宏观制度": r, "n": len(s), "胜率": f"{sum(1 for x in s if x >= EXCELLENT)/len(s):.0%}"}
                              for r, s in sorted(_by_reg2.items())]
                st.caption("按宏观制度")
                st.dataframe(pd.DataFrame(_reg2_rows), hide_index=True, use_container_width=True)

    # Confidence calibration + Payoff Quality
    with st.expander("Confidence Calibration · 置信度校准", expanded=False):
        st.markdown(
            '<div style="font-size:0.8rem; color:var(--muted); margin-bottom:0.8rem;">'
            '理想状态：置信度=X% 的判断，实际准确率应接近 X%（对角线）。Brier Score 越低越好。</div>',
            unsafe_allow_html=True,
        )
        with SessionFactory() as _cal_s:
            _cal_recs = (
                _cal_s.query(DecisionLog)
                .filter(
                    DecisionLog.verified == True,
                    DecisionLog.accuracy_score.isnot(None),
                    DecisionLog.confidence_score.isnot(None),
                    DecisionLog.decision_date >= CLEAN_ZONE_START,
                    (DecisionLog.superseded == False) | DecisionLog.superseded.is_(None),
                    (DecisionLog.is_backtest == False) | DecisionLog.is_backtest.is_(None),
                ).all()
            )
        if not _cal_recs:
            st.caption("暂无含置信度的记录。")
        else:
            _cbuckets: dict = {}
            for r in _cal_recs:
                _b = min(9, r.confidence_score // 10)
                _cbuckets.setdefault(_b, []).append(r.accuracy_score)
            _cal_rows2 = [{"置信度区间": f"{b*10}–{b*10+9}%", "样本数": len(s),
                           "实际准确率": round(sum(s)/len(s), 3), "理想值": round((b*10+5)/100, 3)}
                          for b, s in sorted(_cbuckets.items())]
            st.dataframe(pd.DataFrame(_cal_rows2), hide_index=True, use_container_width=True)
            st.caption(f"Brier Score = {_cz2_brier:.4f}" if _cz2_brier is not None else "Brier Score: N/A")

    with st.expander("Payoff Quality · 风险调整回报", expanded=False):
        with SessionFactory() as _pq_s:
            _pq_recs2 = (
                _pq_s.query(DecisionLog)
                .filter(
                    DecisionLog.verified == True,
                    DecisionLog.payoff_quality.isnot(None),
                    DecisionLog.barrier_return.isnot(None),
                    DecisionLog.decision_date >= CLEAN_ZONE_START,
                    (DecisionLog.superseded == False) | DecisionLog.superseded.is_(None),
                    (DecisionLog.is_backtest == False) | DecisionLog.is_backtest.is_(None),
                ).all()
            )
        if not _pq_recs2:
            st.caption("暂无含 payoff_quality 的记录（barrier 验证后自动计算）。")
        else:
            _pw = [r.barrier_return for r in _pq_recs2 if r.barrier_return > 0]
            _pl = [r.barrier_return for r in _pq_recs2 if r.barrier_return < 0]
            _apq = sum(r.payoff_quality for r in _pq_recs2) / len(_pq_recs2)
            _cpr2 = abs(sum(_pw)/len(_pw) / (sum(_pl)/len(_pl))) if _pw and _pl else None
            _pqc1, _pqc2, _pqc3, _pqc4 = st.columns(4)
            _pqc1.metric("PQ 样本数", len(_pq_recs2))
            _pqc2.metric("平均 PQ", f"{_apq:.3f}", help="PQ>0 表示正期望风险调整回报")
            _pqc3.metric("PQ>0 比例", f"{sum(1 for r in _pq_recs2 if r.payoff_quality>0)/len(_pq_recs2):.0%}")
            _pqc4.metric("条件盈亏比", f"{_cpr2:.2f}" if _cpr2 else "N/A", help="|E[赢]/E[输]|，理想>1.5")

# ── P2-6 Horizon 分层绩效报告 ──────────────────────────────────────────────────
with _tab_perf:
    st.divider()
    _section("Horizon 分层绩效")
    st.markdown(
        '<div style="font-size:0.85rem; color:var(--muted); margin-bottom:1rem;">'
        '按预测地平线分组：胜率 / 平均准确率 / 平均持仓天数（barrier_days）</div>',
        unsafe_allow_html=True,
    )
    with SessionFactory() as _sh:
        _h_recs = (
            _sh.query(DecisionLog)
            .filter(
                DecisionLog.verified == True,
                DecisionLog.accuracy_score.isnot(None),
                (DecisionLog.superseded == False) | DecisionLog.superseded.is_(None),
                (DecisionLog.is_backtest == False) | DecisionLog.is_backtest.is_(None),
            )
            .all()
        )
    if not _h_recs:
        st.caption("暂无已验证记录。")
    else:
        from collections import defaultdict
        _h_groups: dict[str, list] = defaultdict(list)
        for r in _h_recs:
            _h_label = (r.horizon or "未指定").split("(")[0].strip() or "未指定"
            _h_groups[_h_label].append(r)

        _h_rows = []
        for horizon_label, recs in sorted(_h_groups.items()):
            _n    = len(recs)
            _wr   = sum(1 for r in recs if r.accuracy_score >= EXCELLENT) / _n
            _aa   = sum(r.accuracy_score for r in recs) / _n
            _days = [r.barrier_days for r in recs if r.barrier_days is not None]
            _avd  = sum(_days) / len(_days) if _days else None
            _brier_recs = [r for r in recs if r.confidence_score is not None]
            _bs = (
                sum((r.confidence_score / 100 - r.accuracy_score) ** 2 for r in _brier_recs)
                / len(_brier_recs)
            ) if _brier_recs else None
            _h_rows.append({
                "地平线":       horizon_label,
                "n":            _n,
                "胜率(≥0.75)":  f"{_wr:.0%}",
                "平均准确率":   f"{_aa:.3f}",
                "平均持仓(天)": f"{_avd:.0f}" if _avd is not None else "—",
                "Brier Score":  f"{_bs:.3f}" if _bs is not None else "—",
            })
        import pandas as pd
        st.dataframe(
            pd.DataFrame(_h_rows),
            hide_index=True,
            use_container_width=True,
        )

# ── Performance Tab: Decision Log (agent_decisions content) ────────────────────
with _tab_perf:
    st.divider()
    _section("Decision Log · 全量决策记录")

    _DL_DIR_COLORS = {
        "超配": "#22c55e", "标配": "#f59e0b", "低配": "#ef4444",
        "拦截": "#94a3b8", "通过": "#3b82f6", "中性": "#94a3b8",
    }
    _DL_BARRIER_ICONS = {"tp": "✅", "sl": "❌", "time": "⏱"}
    _DL_FAILURE_SHORT = {
        "hypothesis": "假设失效", "data": "数据问题", "regime_drift": "制度漂移",
        "robustness": "稳健性",   "evaluation": "评估问题", "execution": "执行偏差",
    }

    _dl_f1, _dl_f2, _dl_f3, _dl_f4 = st.columns([2, 2, 2, 1], gap="small")
    _dl_filter_tab = _dl_f1.multiselect(
        "分析类型", ["macro", "sector", "audit", "scanner"],
        default=["sector", "macro"], key="dl_filter_tab",
    )
    _dl_filter_ver = _dl_f2.selectbox(
        "验证状态", ["全部", "已验证", "待验证"], key="dl_filter_ver",
    )
    _dl_filter_result = _dl_f3.selectbox(
        "验证结果", ["全部", "正确(≥0.75)", "部分(0.5)", "失败(<0.5)", "未归因失败"],
        key="dl_filter_result",
    )
    _dl_limit = _dl_f4.selectbox("条数", [30, 50, 100, 200], index=1, key="dl_limit")

    with SessionFactory() as _dl_sess:
        _dl_q = _dl_sess.query(DecisionLog).filter(DecisionLog.superseded == False)
        if _dl_filter_tab:
            _dl_q = _dl_q.filter(DecisionLog.tab_type.in_(_dl_filter_tab))
        if _dl_filter_ver == "已验证":
            _dl_q = _dl_q.filter(DecisionLog.verified == True)
        elif _dl_filter_ver == "待验证":
            _dl_q = _dl_q.filter(DecisionLog.verified == False)
        if _dl_filter_result == "正确(≥0.75)":
            _dl_q = _dl_q.filter(DecisionLog.accuracy_score >= 0.75)
        elif _dl_filter_result == "部分(0.5)":
            _dl_q = _dl_q.filter(DecisionLog.accuracy_score == 0.5)
        elif _dl_filter_result == "失败(<0.5)":
            _dl_q = _dl_q.filter(DecisionLog.accuracy_score < 0.5, DecisionLog.verified == True)
        elif _dl_filter_result == "未归因失败":
            _dl_q = _dl_q.filter(
                DecisionLog.accuracy_score < 0.5, DecisionLog.verified == True,
                DecisionLog.failure_type.is_(None),
            )
        _dl_records = _dl_q.order_by(DecisionLog.created_at.desc()).limit(_dl_limit).all()

    if not _dl_records:
        st.info("暂无符合条件的决策记录。")
    else:
        st.caption(f"共 {len(_dl_records)} 条")
        for _dl_rec in _dl_records:
            if _dl_rec.accuracy_score is not None:
                _dl_border = "#22c55e" if _dl_rec.accuracy_score >= 0.75 else (
                    "#f59e0b" if _dl_rec.accuracy_score == 0.5 else "#ef4444")
            elif not _dl_rec.verified:
                _dl_border = "#3b82f6"
            else:
                _dl_border = "#64748b"

            _dl_dc = _DL_DIR_COLORS.get(_dl_rec.direction or "", "#94a3b8")
            with st.container(border=True):
                _dl_hc, _dl_mc = st.columns([3, 2])
                with _dl_hc:
                    _dl_bi = _DL_BARRIER_ICONS.get(_dl_rec.barrier_hit or "", "")
                    _dl_acc = (f"  {_dl_bi} {_dl_rec.accuracy_score:.0%}"
                               if _dl_rec.accuracy_score is not None else "  ⏳ 待验证")
                    _dl_ft = (f"  🏷 {_DL_FAILURE_SHORT.get(_dl_rec.failure_type, _dl_rec.failure_type)}"
                              if _dl_rec.failure_type else "")
                    st.markdown(
                        f'<span style="font-size:1rem;font-weight:700;">'
                        f'{_dl_rec.sector_name or "全球宏观"}</span>'
                        f'  <span style="color:{_dl_dc};font-weight:600;">{_dl_rec.direction or "—"}</span>'
                        f'<span style="color:#94a3b8;font-size:0.88rem;">{_dl_acc}{_dl_ft}</span>',
                        unsafe_allow_html=True,
                    )
                with _dl_mc:
                    _dl_date = _dl_rec.created_at.strftime("%Y-%m-%d") if _dl_rec.created_at else "—"
                    st.markdown(
                        f'<div style="color:var(--muted);font-size:0.83rem;text-align:right;">'
                        f'{_dl_rec.tab_type}  ·  {_dl_date}<br>'
                        f'置信度 <b>{_dl_rec.confidence_score or "—"}</b>/100  ·  '
                        f'VIX {_dl_rec.vix_level or "—"}  ·  {_dl_rec.macro_regime or "—"}</div>',
                        unsafe_allow_html=True,
                    )
                with st.expander("详情", expanded=False):
                    _dl_d1, _dl_d2 = st.columns(2)
                    with _dl_d1:
                        if _dl_rec.economic_logic:
                            st.markdown("**经济逻辑**")
                            st.caption(_dl_rec.economic_logic[:400])
                        if _dl_rec.invalidation_conditions:
                            st.markdown("**失效条件**")
                            st.caption(_dl_rec.invalidation_conditions[:300])
                    with _dl_d2:
                        if _dl_rec.verified and _dl_rec.barrier_hit:
                            st.markdown("**验证详情**")
                            _dl_b1, _dl_b2 = st.columns(2)
                            _dl_b1.metric("触碰障碍", f"{_DL_BARRIER_ICONS.get(_dl_rec.barrier_hit,'')} {_dl_rec.barrier_hit}")
                            _dl_b2.metric("持仓天数", f"{_dl_rec.barrier_days} 天" if _dl_rec.barrier_days else "—")
                            if _dl_rec.barrier_return is not None:
                                st.metric("到达收益率", f"{_dl_rec.barrier_return:+.2%}")
                    if (_dl_rec.accuracy_score is not None and _dl_rec.accuracy_score < 0.5
                            and _dl_rec.verified and _dl_rec.failure_type is None):
                        st.divider()
                        _dl_fa1, _dl_fa2, _dl_fa3 = st.columns([1, 2, 1])
                        _dl_ft_choice = _dl_fa1.selectbox(
                            "失败类型", options=list(_FAILURE_TYPE_LABELS.keys()),
                            format_func=lambda k: _FAILURE_TYPE_LABELS[k],
                            key=f"dl_ft_{_dl_rec.id}",
                        )
                        _dl_ft_note = _dl_fa2.text_input("备注", key=f"dl_fn_{_dl_rec.id}")
                        if _dl_fa3.button("归因", key=f"dl_fc_{_dl_rec.id}", type="primary"):
                            set_failure_attribution(_dl_rec.id, _dl_ft_choice, _dl_ft_note)
                            st.success("归因已保存")
                            st.rerun()
                    elif _dl_rec.failure_type:
                        st.markdown(
                            f"🏷 **归因**: {_FAILURE_TYPE_LABELS.get(_dl_rec.failure_type, _dl_rec.failure_type)}"
                            + (f"  — {_dl_rec.failure_note}" if _dl_rec.failure_note else "")
                        )

if False:  # BACKTEST HIDDEN — code preserved, tab hidden

    # ═══════════════════════════════════════════════════════════════════════════════
    # TAB 4 — Backtest  (simple batch + walk-forward)
    # ═══════════════════════════════════════════════════════════════════════════════
    with _tab_bt:
        _section("Historical Backtest · Replay Past Environments")

        # ── Persistent resume banner (reads from DB, survives restarts) ────────
        # Uses st.empty() so we can hide it cleanly when a live run starts.
        _banner_placeholder = st.empty()
        _active_session = get_active_backtest_session()
        if _active_session:
            _freq_label  = "季度" if _active_session["freq"] == "QS" else "月度"
            _sectors_str = "、".join(_active_session["sectors"])
            _sess_status = _active_session["status"]   # "quota_hit" | "paused" | "running"
            _done_n  = _active_session["done_pairs"]
            _total_n = _active_session["total_pairs"]
            _rem_n   = _active_session["remaining"]
            _pct_n   = int(_done_n / _total_n * 100) if _total_n else 0

            if _sess_status == "quota_hit":
                _icon        = "⏳"
                _title       = "API 额度已耗尽 · 训练中断"
                _badge_text  = "QUOTA EXCEEDED"
                _action_hint = (
                    "前往 <a href='https://aistudio.google.com/api-keys' target='_blank' "
                    "style='color:#d97706; font-weight:600;'>Key Manager</a> 更新 API Key 后，"
                    "点击 <b>Run Simple Backtest</b> 从断点继续，已完成记录自动跳过。"
                )
                _bg      = "#fffbeb"
                _brd     = "#fde68a"
                _accent  = "#f59e0b"
                _lbl_c   = "#92400e"
                _rem_c   = "#d97706"
            elif _sess_status == "running":
                # Session in "running" state on page load = crashed/interrupted mid-run
                _icon        = "↺"
                _title       = "检测到未完成的训练任务"
                _badge_text  = "RESUMABLE"
                _action_hint = (
                    "上次训练异常中断（已完成 <b>" + str(_done_n) + "</b> 批次）。"
                    "点击 <b>Run Simple Backtest</b> 即可从断点继续，已完成记录自动跳过。"
                )
                _bg      = "#f0f9ff"
                _brd     = "#bae6fd"
                _accent  = "#0284c7"
                _lbl_c   = "#0c4a6e"
                _rem_c   = "#0369a1"
            else:  # "paused" — manual pause
                _icon        = "⏸"
                _title       = "训练已手动暂停 · 断点已保存"
                _badge_text  = "PAUSED"
                _action_hint = (
                    "你主动暂停了本次训练。点击 <b>Run Simple Backtest</b> 即可从断点继续，"
                    "已完成的（板块 × 日期）组合自动跳过，无需任何额外操作。"
                )
                _bg      = "#f5f3ff"
                _brd     = "#ddd6fe"
                _accent  = "#8b5cf6"
                _lbl_c   = "#4c1d95"
                _rem_c   = "#7c3aed"

            _banner_placeholder.markdown(
                f'<div style="background:{_bg}; border:1px solid {_brd}; '
                f'border-left:4px solid {_accent}; border-radius:8px; '
                f'padding:14px 18px; margin-bottom:1rem;">'
                # ── header row ──
                f'<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;">'
                f'<span style="color:#1e293b; font-size:0.95rem; font-weight:700; letter-spacing:0.01em;">'
                f'{_icon}&nbsp;&nbsp;{_title}</span>'
                f'<span style="background:{_accent}22; color:{_accent}; border:1px solid {_accent}66; '
                f'border-radius:4px; padding:2px 10px; font-family:monospace; font-size:0.78rem; font-weight:700;">'
                f'{_badge_text}</span>'
                f'</div>'
                # ── metrics grid ──
                f'<div style="display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-bottom:10px;">'
                f'<div style="background:#ffffff; border:1px solid {_brd}; border-radius:5px; padding:7px 10px;">'
                f'<div style="color:{_lbl_c}; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:2px;">训练窗口</div>'
                f'<div style="color:#1e293b; font-size:0.82rem; font-family:monospace; font-weight:600;">'
                f'{_active_session["start_date"]} → {_active_session["end_date"]}</div>'
                f'<div style="color:#78716c; font-size:0.72rem; margin-top:2px;">{_freq_label}频率</div>'
                f'</div>'
                f'<div style="background:#ffffff; border:1px solid {_brd}; border-radius:5px; padding:7px 10px;">'
                f'<div style="color:{_lbl_c}; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:2px;">训练进度</div>'
                f'<div style="color:#1e293b; font-size:0.82rem; font-family:monospace;">'
                f'<span style="font-size:1rem; font-weight:700;">{_done_n}</span>'
                f'<span style="color:#78716c;"> / {_total_n} 批次</span></div>'
                f'<div style="color:{_rem_c}; font-size:0.72rem; font-weight:600; margin-top:2px;">剩余 {_rem_n} 批次</div>'
                f'</div>'
                f'<div style="background:#ffffff; border:1px solid {_brd}; border-radius:5px; padding:7px 10px;">'
                f'<div style="color:{_lbl_c}; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:2px;">覆盖板块</div>'
                f'<div style="color:#44403c; font-size:0.75rem; line-height:1.5;">{_sectors_str}</div>'
                f'</div>'
                f'</div>'
                # ── action hint ──
                f'<div style="border-top:1px solid {_brd}; padding-top:8px; '
                f'color:#78716c; font-size:0.8rem; line-height:1.6;">'
                f'{_action_hint}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div style="font-size:1rem; color:var(--muted); margin-bottom:0.6rem; line-height:1.7;">'
            'Replay historical dates through the current sector agent. '
            'Each run builds a snapshot from <b>FRED macro data (Path A)</b> + '
            '<b>GDELT news headlines (Path B)</b>, calls the AI, and saves to Alpha Memory '
            'with <code>is_backtest=True</code>. '
            'Walk-Forward mode enforces strict temporal isolation to prevent data leakage.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="background:#fefce8; border:1px solid #fde68a; border-left:3px solid #f59e0b; '
            'border-radius:6px; padding:0.55rem 1rem; font-size:0.82rem; color:#92400e; '
            'margin-bottom:1rem; line-height:1.6;">'
            '⚠️ <b>用途说明</b>：历史回测用于管道验证和系统调试。'
            '产出的决策记录不计入绩效统计，不写回学习表。'
            '系统唯一有效的绩效证据为 Clean Zone（≥ 2025-04-01）live 决策的验证结果。'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── Mode selector ─────────────────────────────────────────────────────────
        bt_mode = st.radio(
            "Backtest Mode",
            ["Simple Batch", "Walk-Forward (防数据泄漏)"],
            horizontal=True,
            key="bt_mode",
            help=(
                "Walk-Forward splits data into a training window and a held-out test window. "
                "Test decisions can only see Alpha Memory records that existed before the split date."
            ),
        )

        _is_wf = (bt_mode == "Walk-Forward (防数据泄漏)")

        # ── Gemini model for backtest ─────────────────────────────────────────────
        def _get_model():
            try:
                key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_KEY", "")
                return _make_model(key)
            except Exception:
                return None

        ALL_SECTORS = list(SECTOR_ETF.keys())

        # ── Sector selector ───────────────────────────────────────────────────────
        bt_sectors = st.multiselect(
            "Sectors to include",
            options=ALL_SECTORS,
            default=["AI算力/半导体", "科技成长(纳指)", "金融", "全球能源", "工业/基建"],
            key="bt_sectors",
        )

        # ── Date controls (differ by mode) ────────────────────────────────────────
        if not _is_wf:
            bt_c1, bt_c2, bt_c3 = st.columns([1, 1, 1], gap="medium")
            with bt_c1:
                bt_start = st.date_input(
                    "Start Date",
                    value=datetime.date(2020, 1, 1),
                    min_value=datetime.date(2015, 3, 1),
                    max_value=datetime.date.today(),
                    key="bt_start",
                )
            with bt_c2:
                bt_end = st.date_input(
                    "End Date",
                    value=datetime.date(2023, 12, 31),
                    min_value=datetime.date(2015, 3, 1),
                    max_value=datetime.date.today(),
                    key="bt_end",
                )
            with bt_c3:
                freq_label = st.selectbox(
                    "Frequency",
                    ["Quarterly", "Monthly"],
                    key="bt_freq",
                )
            freq_code = "QS" if freq_label == "Quarterly" else "MS"

            # ── Train / test split indicator ─────────────────────────────────────
            _in_train = bt_end < TRAIN_TEST_CUTOFF
            _in_test  = bt_start >= TRAIN_TEST_CUTOFF
            _crosses  = bt_start < TRAIN_TEST_CUTOFF <= bt_end

            if _in_train:
                _split_color, _split_icon, _split_msg = (
                    "#10b981", "✅",
                    f"完全在训练集内（截止 {TRAIN_TEST_CUTOFF}）— 验证结果将写入学习表"
                )
            elif _in_test:
                _split_color, _split_icon, _split_msg = (
                    "#6366f1", "🔒",
                    f"完全在测试集内（{TRAIN_TEST_CUTOFF} 之后）— 验证结果仅打分，不写入学习表"
                )
            else:
                _split_color, _split_icon, _split_msg = (
                    "#f59e0b", "⚠️",
                    f"跨越训练/测试边界（{TRAIN_TEST_CUTOFF}）— {TRAIN_TEST_CUTOFF} 前的结果写入学习表，之后的仅打分"
                )

            st.markdown(
                f'<div style="background:{_split_color}18; border:1px solid {_split_color}44; '
                f'border-left:3px solid {_split_color}; border-radius:6px; '
                f'padding:0.5rem 0.9rem; font-size:0.82rem; color:{_split_color}; '
                f'margin-bottom:0.4rem;">'
                f'{_split_icon} <b>训练/测试集</b>：{_split_msg}</div>',
                unsafe_allow_html=True,
            )

            try:
                _total = len(pd.date_range(str(bt_start), str(bt_end), freq=freq_code)) * len(bt_sectors)
            except Exception:
                _total = 0
            _total_wf = 0
            wf_train_start = wf_train_end = wf_test_end = datetime.date.today()

        else:
            wf_c1, wf_c2, wf_c3, wf_c4 = st.columns([1, 1, 1, 1], gap="medium")
            with wf_c1:
                wf_train_start = st.date_input(
                    "Train Start",
                    value=datetime.date(2020, 1, 1),
                    min_value=datetime.date(2015, 3, 1),
                    max_value=datetime.date.today(),
                    key="wf_train_start",
                )
            with wf_c2:
                wf_train_end = st.date_input(
                    "Train End  (split point)",
                    value=datetime.date(2022, 12, 31),
                    min_value=datetime.date(2015, 3, 1),
                    max_value=datetime.date.today(),
                    key="wf_train_end",
                )
            with wf_c3:
                wf_test_end = st.date_input(
                    "Test End",
                    value=datetime.date(2023, 12, 31),
                    min_value=datetime.date(2015, 3, 1),
                    max_value=datetime.date.today(),
                    key="wf_test_end",
                )
            with wf_c4:
                freq_label = st.selectbox(
                    "Frequency",
                    ["Quarterly", "Monthly"],
                    key="wf_freq",
                )
            freq_code = "QS" if freq_label == "Quarterly" else "MS"
            try:
                _total_wf = (
                    len(pd.date_range(str(wf_train_start), str(wf_test_end), freq=freq_code))
                    * len(bt_sectors)
                )
            except Exception:
                _total_wf = 0
            _total = 0
            bt_start = bt_end = datetime.date.today()

            # Walk-forward isolation info box
            st.markdown(
                '<div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:6px; '
                'padding:0.7rem 1rem; margin-top:0.5rem; font-size:0.95rem; color:#1E40AF;">'
                f'🔒 <b>Temporal isolation active</b> · Train: {wf_train_start} → {wf_train_end} '
                f'· Test: {wf_train_end} → {wf_test_end} · '
                f'Each test decision at date T only sees Alpha Memory records with decision_date &lt; T'
                '</div>',
                unsafe_allow_html=True,
            )

        # ── Options ───────────────────────────────────────────────────────────────
        _opt_col1, _opt_col2 = st.columns([1, 3])
        with _opt_col1:
            _sensitivity_on = st.toggle(
                "敏感性测试",
                value=False,
                key="bt_sensitivity",
                help="开启后每条记录额外调用一次 API（VIX ±5 方向变化测试）。quota 紧张时建议关闭。",
            )

        # ── Action buttons ────────────────────────────────────────────────────────
        _btn_col1, _btn_col2, _btn_col3 = st.columns([2, 1, 1], gap="medium")

        with _btn_col1:
            _run_label = (
                f"▶  Run Walk-Forward  ({_total_wf} calls)" if _is_wf
                else f"▶  Run Simple Backtest  ({_total} calls)"
            )
            _run_disabled = (
                (_total_wf == 0 or not bt_sectors) if _is_wf
                else (_total == 0 or not bt_sectors)
            )
            _run_help = None
            if _run_disabled:
                if not bt_sectors:
                    _run_help = "请先在上方「Sectors」选择器中至少选择一个板块。"
                elif (_is_wf and _total_wf == 0) or (not _is_wf and _total == 0):
                    _run_help = (
                        "当前时间窗口内没有可运行的日期点。\n\n"
                        "请检查起止日期设置：结束日期必须晚于开始日期，"
                        "且跨度须能覆盖至少一个采样周期。"
                    )
            run_bt = st.button(
                _run_label,
                type="primary",
                width='stretch',
                disabled=_run_disabled,
                help=_run_help,
            )

        with _btn_col3:
            if st.button(
                "⏸  暂停 / 保存进度",
                width='stretch',
                help="当前 pair 跑完后干净停止，已完成数据全部保留，下次可从断点续训。",
            ):
                set_backtest_stop(True)
                st.toast("暂停指令已发送，将在当前记录完成后停止。")

        with _btn_col2:
            _preview_sector = bt_sectors[0] if bt_sectors else None
            _preview_date   = (wf_train_start if _is_wf else bt_start) if bt_sectors else None
            if st.button(
                "🔍  Preview Snapshot",
                width='stretch',
                disabled=(not bt_sectors),
                help=None if bt_sectors else "请先选择至少一个板块，才能预览快照数据。",
            ):
                with st.spinner(f"Building snapshot for {_preview_date} / {_preview_sector}..."):
                    try:
                        snap = build_snapshot(_preview_date, _preview_sector)
                        with st.expander("Snapshot preview", expanded=True):
                            st.text_area(
                                "Full context",
                                value=snap["full_context"],
                                height=220,
                                key="snap_preview",
                            )
                            if snap.get("data_cutoff_log"):
                                st.markdown("**Data Cutoff Audit**")
                                st.json(snap["data_cutoff_log"])
                    except Exception as e:
                        st.error(f"Snapshot error: {e}")

        # ── Run logic ──────────────────────────────────────────────────────────────
        if run_bt:
            _banner_placeholder.empty()   # hide stale "interrupted" banner while training
            _model = _get_model()
            if not _model:
                st.error("Gemini API key not found in secrets.")
            else:
                progress_bar  = st.progress(0.0)
                status_text   = st.empty()
                results_store = []

                def _render_progress_card(placeholder, current, total, msg, phase_color=None):
                    """Render a styled terminal-style progress card."""
                    _safe_total = max(total, 1)   # guard against ZeroDivisionError
                    pct = min(100, int(current / _safe_total * 100))
                    # Determine accent color and status icon from message content
                    if msg.startswith("✓"):
                        accent = "#10b981"   # emerald — record saved
                        tag    = "SAVED"
                    elif "⊘" in msg or "无效批次" in msg:
                        accent = "#f59e0b"   # amber — skipped no data
                        tag    = "SKIP"
                    elif "↷" in msg:
                        accent = "#475569"   # slate — state unchanged
                        tag    = "PASS"
                    elif "🛑" in msg or "熔断" in msg or "耗尽" in msg:
                        accent = "#ef4444"   # red — halt
                        tag    = "HALT"
                    elif "⚠" in msg or "Error" in msg:
                        accent = "#f97316"   # orange — error
                        tag    = "WARN"
                    elif "🔄" in msg:
                        accent = "#6366f1"   # indigo — key rotation
                        tag    = "SWAP"
                    elif "⏸" in msg:
                        accent = "#8b5cf6"   # purple — paused
                        tag    = "PAUSE"
                    else:
                        accent = phase_color or "#3b82f6"  # blue — processing
                        tag    = "PROC"

                    # Split msg at │ separator into main message and stats line
                    parts     = msg.split("  │  ", 1)
                    main_msg  = parts[0].strip()
                    stats_msg = parts[1].strip() if len(parts) > 1 else ""

                    # Build stats badges from stats_msg (e.g. "有效记录 3 · 无数据跳过 5 · 状态未变跳过 12")
                    stats_html = ""
                    if stats_msg:
                        badge_defs = [
                            ("有效记录",   "#10b981", "#052e16"),
                            ("无数据跳过", "#f59e0b", "#1c1400"),
                            ("状态未变跳过","#64748b", "#0f172a"),
                        ]
                        badges = []
                        for label, fg, bg in badge_defs:
                            m = re.search(rf"{label}\s+(\d+)", stats_msg)
                            val = m.group(1) if m else "0"
                            badges.append(
                                f'<span style="background:{bg}; color:{fg}; border:1px solid {fg}33; '
                                f'border-radius:4px; padding:1px 8px; font-size:0.75rem; '
                                f'font-family:monospace; margin-right:6px; white-space:nowrap;">'
                                f'{label} <b>{val}</b></span>'
                            )
                        stats_html = f'<div style="margin-top:6px">{"".join(badges)}</div>'

                    placeholder.markdown(
                        f'<div style="background:#0d1117; border:1px solid #1e293b; '
                        f'border-left:3px solid {accent}; border-radius:6px; '
                        f'padding:10px 14px; margin-top:4px; font-family:monospace;">'
                        f'<div style="display:flex; align-items:baseline; gap:10px;">'
                        f'<span style="background:{accent}22; color:{accent}; border:1px solid {accent}55; '
                        f'border-radius:3px; padding:1px 6px; font-size:0.7rem; font-weight:700; '
                        f'letter-spacing:0.05em; flex-shrink:0;">{tag}</span>'
                        f'<span style="color:#e2e8f0; font-size:0.88rem; flex:1;">{main_msg}</span>'
                        f'<span style="color:#475569; font-size:0.78rem; white-space:nowrap;">'
                        f'{current} / {total} &nbsp;·&nbsp; {pct}%</span>'
                        f'</div>'
                        f'{stats_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                if not _is_wf:
                    # ── Simple Batch ──────────────────────────────────────────────
                    def _cb(current, total, msg):
                        progress_bar.progress(min(current / max(total, 1), 1.0))
                        _render_progress_card(status_text, current, total, msg)
                    try:
                        results_store, _quota_hit = run_sector_backtest(
                            model=_model,
                            sectors=bt_sectors,
                            start_date=str(bt_start),
                            end_date=str(bt_end),
                            freq=freq_code,
                            progress_cb=_cb,
                            sensitivity_test=_sensitivity_on,
                        )
                        progress_bar.progress(1.0)
                        status_text.empty()
                        if _quota_hit:
                            _saved_n = len(results_store)
                            st.markdown(
                                '<div style="background:#fffbeb; border:1px solid #fde68a; '
                                'border-left:4px solid #f59e0b; border-radius:8px; '
                                'padding:14px 18px; margin-top:0.6rem;">'
                                '<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;">'
                                '<span style="color:#1e293b; font-size:0.95rem; font-weight:700;">⏳&nbsp;&nbsp;API 额度已耗尽 · 训练中断</span>'
                                '<span style="background:#f59e0b22; color:#d97706; border:1px solid #f59e0b66; '
                                'border-radius:4px; padding:2px 10px; font-family:monospace; font-size:0.78rem; font-weight:700;">QUOTA EXCEEDED</span>'
                                '</div>'
                                '<div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:10px;">'
                                '<div style="background:#ffffff; border:1px solid #fde68a; border-radius:5px; padding:7px 10px;">'
                                '<div style="color:#92400e; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:2px;">本次写入记录</div>'
                                f'<div style="color:#059669; font-size:1rem; font-weight:700; font-family:monospace;">{_saved_n} 条</div>'
                                '<div style="color:#78716c; font-size:0.72rem; margin-top:2px;">已存入 Alpha Memory</div>'
                                '</div>'
                                '<div style="background:#ffffff; border:1px solid #fde68a; border-radius:5px; padding:7px 10px;">'
                                '<div style="color:#92400e; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:2px;">续训方式</div>'
                                '<div style="color:#44403c; font-size:0.8rem; line-height:1.5;">更新 Key 后重新运行<br>系统自动从断点继续</div>'
                                '</div>'
                                '</div>'
                                '<div style="border-top:1px solid #fde68a; padding-top:8px; color:#78716c; font-size:0.8rem; line-height:1.6;">'
                                '前往 <a href="https://aistudio.google.com/api-keys" target="_blank" '
                                'style="color:#d97706; text-decoration:none; font-weight:600;">aistudio.google.com/api-keys</a>'
                                ' 获取新 Key，添加至 Key Manager 后点击 <b style="color:#1e293b;">Run Simple Backtest</b> 即可续训，已完成的（板块 × 日期）组合自动跳过。'
                                '</div>'
                                '</div>',
                                unsafe_allow_html=True,
                            )
                        elif len(results_store) == 0:
                            st.info("所有批次均已完成训练，无新记录需要写入。如需重新训练请先删除 Backtest Review 中的对应记录。")
                        else:
                            st.success(f"训练完成 — 本次新增 {len(results_store)} 条记录，已写入 Alpha Memory。")
                    except Exception as e:
                        st.error(f"Backtest error: {e}")

                else:
                    # ── Walk-Forward ──────────────────────────────────────────────
                    _phase_colors = {"[TRAIN]": "#3B82F6", "[TEST]": "#10B981"}

                    def _wf_cb(current, total, phase, msg):
                        progress_bar.progress(min(current / max(total, 1), 1.0))
                        color = _phase_colors.get(f"[{phase.upper()}]", "#64748B")
                        _render_progress_card(status_text, current, total,
                                             f"[{phase.upper()}] {msg}", phase_color=color)

                    try:
                        results_store = run_walk_forward_backtest(
                            model=_model,
                            sectors=bt_sectors,
                            train_start=str(wf_train_start),
                            train_end=str(wf_train_end),
                            test_end=str(wf_test_end),
                            freq=freq_code,
                            progress_cb=_wf_cb,
                            sensitivity_test=_sensitivity_on,
                        )
                        progress_bar.progress(1.0)
                        status_text.empty()
                        train_n = sum(1 for r in results_store if r.get("phase") == "train")
                        test_n  = sum(1 for r in results_store if r.get("phase") == "test")
                        st.success(
                            f"Walk-Forward complete — {len(results_store)} records saved "
                            f"({train_n} train · {test_n} test)"
                        )
                    except Exception as e:
                        st.error(f"Walk-Forward error: {e}")

                # ── Results table ─────────────────────────────────────────────────
                if results_store:
                    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
                    _section("Backtest Results")

                    result_rows = []
                    for r in results_store:
                        phase_label = r.get("phase", "—").upper()
                        result_rows.append({
                            "Phase":    phase_label,
                            "Date":     r.get("date", "—"),
                            "Sector":   r.get("sector", "—"),
                            "Regime":   r.get("regime", "—"),
                            "VIX":      f"{r['vix']:.1f}" if r.get("vix") else "—",
                            "ID":       r.get("saved_id", "—"),
                        })

                    res_df = pd.DataFrame(result_rows)
                    st.dataframe(res_df, width='stretch', hide_index=True)

                    if _is_wf and results_store:
                        train_results = [r for r in results_store if r.get("phase") == "train"]
                        test_results  = [r for r in results_store if r.get("phase") == "test"]
                        col_tr, col_te = st.columns(2, gap="medium")
                        with col_tr:
                            with st.container(border=True):
                                st.markdown(
                                    f'<div style="font-size:0.9rem; font-weight:700; '
                                    f'color:#3B82F6; text-transform:uppercase; '
                                    f'letter-spacing:0.07em;">Training Window</div>'
                                    f'<div style="font-family:var(--mono); font-size:1.5rem; '
                                    f'font-weight:700; color:var(--primary);">{len(train_results)}</div>'
                                    f'<div style="font-size:0.9rem; color:var(--muted);">decisions saved</div>',
                                    unsafe_allow_html=True,
                                )
                        with col_te:
                            with st.container(border=True):
                                st.markdown(
                                    f'<div style="font-size:0.9rem; font-weight:700; '
                                    f'color:#10B981; text-transform:uppercase; '
                                    f'letter-spacing:0.07em;">Test Window</div>'
                                    f'<div style="font-family:var(--mono); font-size:1.5rem; '
                                    f'font-weight:700; color:var(--primary);">{len(test_results)}</div>'
                                    f'<div style="font-size:0.9rem; color:var(--muted);">out-of-sample decisions</div>',
                                    unsafe_allow_html=True,
                                )

                        st.info(
                            "✅ Walk-Forward isolation enforced — test decisions used "
                            "`decision_date < T` filter. "
                            "Run Verify Decisions (Overview tab) after the holding period to score via Triple-Barrier."
                        )

    # ─────────────────────────────────────────────────────────────────────────────
    # Clean Zone Performance Monitor (inside Backtest tab)
    # ─────────────────────────────────────────────────────────────────────────────
        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        _section("Clean Zone · 三区间绩效监控")
        st.markdown(
            '<div style="font-size:0.95rem; color:var(--muted); margin-bottom:1.2rem; line-height:1.7;">'
            '系统性能按时间区间分层展示，区分「学习来源」与「真实评估」。'
            ' · <b>训练集</b>（<2023-01-01）：学习来源，可写回 Alpha Memory '
            ' · <b>测试集A</b>（2023~2025-Q1）：测试但仍受 LLM 历史记忆污染 '
            ' · <b>Clean Zone</b>（≥2025-04-01）：真实 out-of-sample，LLM 无历史预知'
            '</div>',
            unsafe_allow_html=True,
        )

        _cz = get_clean_zone_stats()

        def _zone_card(label: str, accent: str, stats: dict, badge: str = "", show_stats: bool = False) -> None:
            n        = stats.get("n", 0)
            avg_acc  = stats.get("avg_accuracy")
            hit_rate = stats.get("hit_rate")
            lcs_pr   = stats.get("lcs_pass_rate")
            brier    = stats.get("brier_score")
            bp       = stats.get("binom_pvalue")
            ci_lo    = stats.get("binom_ci_lo")
            ci_hi    = stats.get("binom_ci_hi")

            acc_str   = f"{avg_acc:.2f}"  if avg_acc  is not None else "—"
            hit_str   = f"{hit_rate:.0%}" if hit_rate is not None else "—"
            lcs_str   = f"{lcs_pr:.0%}"   if lcs_pr   is not None else "N/A"
            brier_str = f"{brier:.3f}"    if brier    is not None else "N/A"
            n_str     = str(n) if n else "0"

            _badge_html = (
                f'<span style="font-size:0.75rem; background:{accent}22; color:{accent}; '
                f'border-radius:3px; padding:0.1rem 0.5rem; font-weight:700;">{badge}</span>'
                if badge else ""
            )

            # Binomial test row (Clean Zone only, when n >= 30)
            _binom_html = ""
            if show_stats:
                if n < 30:
                    _binom_html = (
                        f'<div style="margin-top:0.7rem; padding:0.5rem 0.7rem; '
                        f'background:var(--surface2); border-radius:6px; '
                        f'font-size:0.78rem; color:var(--muted);">'
                        f'样本不足 (n={n} &lt; 30)，Binomial Test 尚无统计意义</div>'
                    )
                elif bp is not None:
                    _sig  = "✓ 显著" if bp < 0.05 else ("~ 边缘" if bp < 0.10 else "✗ 不显著")
                    _pcol = "#10B981" if bp < 0.05 else ("#F59E0B" if bp < 0.10 else "#EF4444")
                    _binom_html = (
                        f'<div style="margin-top:0.7rem; padding:0.5rem 0.7rem; '
                        f'background:var(--surface2); border-radius:6px; font-size:0.78rem;">'
                        f'<span style="color:var(--muted);">Binomial Test (H₀: 胜率=50%)&emsp;</span>'
                        f'<span style="color:{_pcol}; font-weight:700;">{_sig}&emsp;</span>'
                        f'<span style="color:var(--muted);">p = </span>'
                        f'<span style="font-family:var(--mono); color:var(--text);">{bp:.4f}</span>'
                        f'<span style="color:var(--muted);">&emsp;95% CI: </span>'
                        f'<span style="font-family:var(--mono); color:var(--text);">'
                        f'[{ci_lo:.2%}, {ci_hi:.2%}]</span>'
                        f'</div>'
                    )

            st.markdown(
                f'<div style="border:1px solid {accent}; border-radius:8px; padding:1rem 1.2rem;">'
                f'<div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.7rem;">'
                f'<span style="font-size:1rem; font-weight:700; color:{accent};">{label}</span>'
                f'{_badge_html}'
                f'</div>'
                f'<div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr 1fr; gap:0.5rem;">'
                f'<div><div style="font-size:0.75rem; color:var(--muted); text-transform:uppercase; '
                f'letter-spacing:0.07em;">样本数</div>'
                f'<div style="font-family:var(--mono); font-size:1.3rem; font-weight:700; '
                f'color:var(--text);">{n_str}</div></div>'
                f'<div><div style="font-size:0.75rem; color:var(--muted); text-transform:uppercase; '
                f'letter-spacing:0.07em;">平均准确率</div>'
                f'<div style="font-family:var(--mono); font-size:1.3rem; font-weight:700; '
                f'color:{accent};">{acc_str}</div></div>'
                f'<div><div style="font-size:0.75rem; color:var(--muted); text-transform:uppercase; '
                f'letter-spacing:0.07em;">胜率(≥0.75)</div>'
                f'<div style="font-family:var(--mono); font-size:1.3rem; font-weight:700; '
                f'color:{accent};">{hit_str}</div></div>'
                f'<div><div style="font-size:0.75rem; color:var(--muted); text-transform:uppercase; '
                f'letter-spacing:0.07em;">LCS通过率</div>'
                f'<div style="font-family:var(--mono); font-size:1.3rem; font-weight:700; '
                f'color:var(--muted);">{lcs_str}</div></div>'
                f'<div><div style="font-size:0.75rem; color:var(--muted); text-transform:uppercase; '
                f'letter-spacing:0.07em;">Brier Score</div>'
                f'<div style="font-family:var(--mono); font-size:1.3rem; font-weight:700; '
                f'color:var(--muted);" title="越低越好，0=完美校准">{brier_str}</div></div>'
                f'</div>'
                f'{_binom_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

        _cz_c1, _cz_c2, _cz_c3 = st.columns(3, gap="medium")
        with _cz_c1:
            _zone_card("训练集", "#6366F1", _cz["training"], "< 2023-01-01")
        with _cz_c2:
            _zone_card("测试集 A · 受污染", "#F59E0B", _cz["test_a"], "2023 — 2025 Q1")
        with _cz_c3:
            _zone_card("Clean Zone · 净区间", "#10B981", _cz["clean_b"], "≥ 2025-04-01", show_stats=True)

        _lcs_ov             = _cz.get("lcs_overall", {})
        _lcs_total          = _lcs_ov.get("total_audited", 0)
        _lcs_fail           = _lcs_ov.get("fail_rate")
        _lcs_train_filtered = _cz.get("lcs_filtered_training", {})

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            _q1, _q2, _q3 = st.columns(3, gap="medium")
            _q1.metric(
                "LCS 已审计决策数",
                _lcs_total,
                help="已完成镜像测试 + 噪音测试 + 跨周期锚定测试的决策条数",
            )
            _q2.metric(
                "逻辑退化率",
                f"{_lcs_fail:.0%}" if _lcs_fail is not None else "—",
                help="LCS < 0.70 的决策比例（被质量门拦截，不写入学习表）",
            )
            _q3.metric(
                "LCS过滤后训练集样本",
                _lcs_train_filtered.get("n", 0),
                help="仅统计训练集内 LCS 通过的决策——这是 Alpha Memory 的有效知识基础",
            )

        st.markdown(
            f'<div style="font-size:0.82rem; color:var(--muted); margin-top:0.6rem; line-height:1.8;">'
            f'<b>边界说明</b>：TRAIN_TEST_CUTOFF = <code>{TRAIN_TEST_CUTOFF}</code>（学习回写门控）'
            f'  ·  CLEAN_ZONE_START = <code>{CLEAN_ZONE_START}</code>（前瞻污染隔离）<br>'
            f'<b>LCS 质量门</b>：镜像测试(50%) + 噪音注入(30%) + 跨周期锚定(20%) → 综合得分 &lt; 0.70 时拦截写回。'
            f'注意：LCS 通过不代表无历史记忆污染，仅代表逻辑行为一致性。'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Random Baseline · Null Model Comparison ───────────────────────────
        # Triple-Barrier asymmetry: TP=1σ, SL=0.7σ
        # For a driftless Brownian motion: P(hit TP before SL) = SL/(TP+SL) = 0.7/1.7 ≈ 41.2%
        # i.e. random direction signal has a structural disadvantage vs the asymmetric barrier.
        # Expected accuracy_score_random = 0.412 × 1.0 + 0.588 × 0.0 ≈ 0.41
        # (ignoring time-barrier path for clarity; time-barrier sets score=0.5, partial credit)
        _TP, _SL = 1.0, 0.7
        _p_tp_random      = _SL / (_TP + _SL)          # ≈ 0.412
        _p_sl_random      = _TP / (_TP + _SL)          # ≈ 0.588
        _exp_score_random = _p_tp_random * 1.0 + _p_sl_random * 0.0  # ≈ 0.412
        _rand_hit_rate    = _p_tp_random               # hit rate = P(score≥0.75) ≈ P(TP)

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        _section("随机基准（下限）· Null Model Lower Bound")
        st.markdown(
            '<div style="font-size:0.88rem; color:var(--muted); margin-bottom:0.8rem; line-height:1.7;">'
            '三重障碍法设定 TP=1σ、SL=0.7σ，存在结构性不对称。'
            '对无预测能力的随机信号（布朗运动），'
            '由首中障碍公式 P(先中TP) = SL/(TP+SL) = 0.7/1.7 ≈ <b>41.2%</b>。'
            '此值为<b>下限基准</b>：系统须至少超越此水平方具备信息价值。'
            '</div>',
            unsafe_allow_html=True,
        )
        _rb_c1, _rb_c2, _rb_c3, _rb_c4 = st.columns(4, gap="medium")
        _cz_clean = _cz.get("clean_b", {})
        _sys_hit  = _cz_clean.get("hit_rate")
        _sys_acc  = _cz_clean.get("avg_accuracy")
        _sys_n    = _cz_clean.get("n", 0)

        _rb_c1.metric(
            "随机基准胜率",
            f"{_rand_hit_rate:.1%}",
            help="布朗运动下 TP=1σ 先中概率（理论值）",
        )
        _rb_c2.metric(
            "随机基准期望准确率",
            f"{_exp_score_random:.2f}",
            help="随机信号期望 accuracy_score（忽略时间障碍路径）",
        )
        _rb_c3.metric(
            "系统实测胜率 (Clean Zone)",
            f"{_sys_hit:.1%}" if _sys_hit is not None else "—",
            delta=f"{(_sys_hit - _rand_hit_rate):+.1%} vs 随机" if _sys_hit is not None else None,
            delta_color="normal",
            help="Clean Zone 实测胜率与随机基准差值",
        )
        _rb_c4.metric(
            "系统实测准确率 (Clean Zone)",
            f"{_sys_acc:.2f}" if _sys_acc is not None else "—",
            delta=f"{(_sys_acc - _exp_score_random):+.2f} vs 随机" if _sys_acc is not None else None,
            delta_color="normal",
        )

        st.markdown(
            f'<div style="font-size:0.8rem; color:var(--muted); margin-top:0.3rem;">'
            f'随机基准以理论推导为准，不依赖实际数据。'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Sector-Stratified Win Rate ────────────────────────────────────────
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        _section("板块分层胜率 · Sector-Stratified Win Rate")
        st.markdown(
            '<div style="font-size:0.88rem; color:var(--muted); margin-bottom:0.8rem; line-height:1.7;">'
            'TP=1σ 对低波动板块（如 XLP）与高波动板块（如 SMH）的实际难度不同，'
            '跨板块聚合胜率存在比较偏差。以下按板块分层展示 Clean Zone 实测胜率与样本量，'
            '便于识别哪些板块真正提供了预测信息量。'
            '</div>',
            unsafe_allow_html=True,
        )

        with SessionFactory() as _ss:
            _all_cz_recs = (
                _ss.query(DecisionLog)
                .filter(
                    DecisionLog.verified == True,
                    DecisionLog.accuracy_score.isnot(None),
                    DecisionLog.decision_date >= CLEAN_ZONE_START,
                    (DecisionLog.is_backtest == False) | DecisionLog.is_backtest.is_(None),
                )
                .all()
            )

        if not _all_cz_recs:
            st.info("Clean Zone 暂无已验证记录，积累后将在此显示板块分层胜率。")
        else:
            from collections import defaultdict
            _sector_buckets: dict = defaultdict(list)
            for _rec in _all_cz_recs:
                _sector_buckets[_rec.sector_name or "未知"].append(_rec.accuracy_score)

            _strat_rows = []
            for _sec, _scores in sorted(_sector_buckets.items()):
                _n_sec   = len(_scores)
                _avg_sec = sum(_scores) / _n_sec
                _hit_sec = sum(1 for s in _scores if s >= 0.75) / _n_sec
                _vs_rand = _hit_sec - _rand_hit_rate
                _strat_rows.append({
                    "板块":       _sec,
                    "n":          _n_sec,
                    "胜率":       f"{_hit_sec:.0%}",
                    "平均准确率": f"{_avg_sec:.2f}",
                    "vs随机基准": f"{_vs_rand:+.1%}",
                })

            _strat_df = pd.DataFrame(_strat_rows).sort_values("n", ascending=False)
            st.dataframe(
                _strat_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "n":   st.column_config.NumberColumn("样本数", format="%d"),
                    "胜率": st.column_config.TextColumn("胜率(≥0.75)"),
                },
            )
            st.markdown(
                f'<div style="font-size:0.8rem; color:var(--muted); margin-top:0.2rem;">'
                f'随机基准胜率 = {_rand_hit_rate:.1%}（TP=1σ · SL=0.7σ 理论值）。'
                f'「vs随机基准」正值表示系统相对随机信号的超额胜率。'
                f'</div>',
                unsafe_allow_html=True,
            )
if False:  # BACKTEST REVIEW HIDDEN — code preserved, tab hidden

    # ─────────────────────────────────────────────────────────────────────────────
    # Backtest Review Panel (persistent, reads from DB)
    # ─────────────────────────────────────────────────────────────────────────────
    with _tab_bt_review:
        _section("Backtest Review · Stored Training Records")

        _SECTORS_ALL = ["（全部）"] + list(__import__("engine.history", fromlist=["SECTOR_ETF"]).SECTOR_ETF.keys())
        _REGIMES_ALL = ["（全部）", "高波动/危机", "震荡期", "温和波动", "低波动/牛市"]
        _PHASES_ALL  = ["（全部）", "simple", "train", "test"]

        _rf1, _rf2, _rf3 = st.columns([2, 2, 1], gap="small")
        with _rf1:
            _f_sector = st.selectbox("板块筛选", _SECTORS_ALL, key="bt_review_sector")
        with _rf2:
            _f_regime = st.selectbox("宏观周期筛选", _REGIMES_ALL, key="bt_review_regime")
        with _rf3:
            _f_phase = st.selectbox("Phase", _PHASES_ALL, key="bt_review_phase")

        _bt_rows = get_backtest_records(
            sector=None if _f_sector == "（全部）" else _f_sector,
            regime=None if _f_regime == "（全部）" else _f_regime,
            phase=None if _f_phase == "（全部）" else _f_phase,
        )

        if not _bt_rows:
            st.info("暂无回测记录。运行 Backtest 后数据将在此显示。")
        else:
            _info_col, _selall_col = st.columns([5, 1], gap="small")
            with _info_col:
                st.markdown(
                    f'<div style="font-size:0.88rem; color:var(--muted); padding-top:0.5rem;">'
                    f'共 <b>{len(_bt_rows)}</b> 条记录（最新 500 条）· 勾选行后可删除选中记录</div>',
                    unsafe_allow_html=True,
                )
            with _selall_col:
                _all_selected = st.session_state.get("bt_review_select_all", False)
                if st.button(
                    "☐ 取消全选" if _all_selected else "☑ 全选",
                    width='stretch', key="bt_select_all_btn",
                ):
                    st.session_state["bt_review_select_all"] = not _all_selected
                    st.rerun()

            _review_df = pd.DataFrame([{
                "选择":     _all_selected,
                "运行日期": r["run_date"],
                "场景日期": r["date"],
                "板块":     r["sector"],
                "周期":     r["regime"],
                "方向":     r["direction"],
                "VIX":      f"{r['vix']:.1f}" if r["vix"] else "—",
                "置信度":   r["confidence"] if r["confidence"] else "—",
                "Phase":    r["phase"] or "simple",
                "持仓期":   r["horizon"],
                "LCS":      (f"{r['lcs_score']:.2f} {'✓' if r.get('lcs_passed') else '✗'}"
                             if r.get("lcs_score") is not None else "—"),
                "障碍":     (f"{r['barrier_hit']} {r['barrier_days']}d"
                             if r.get("barrier_hit") else "—"),
            } for r in _bt_rows])

            _edited_df = st.data_editor(
                _review_df,
                width='stretch',
                hide_index=True,
                column_config={
                    "选择": st.column_config.CheckboxColumn("选择", width="small"),
                },
                disabled=["运行日期", "场景日期", "板块", "周期", "方向", "VIX", "置信度", "Phase", "持仓期", "LCS", "障碍"],
                key="bt_review_editor",
            )

            _selected_ids = [
                _bt_rows[i]["id"]
                for i, checked in enumerate(_edited_df["选择"])
                if checked
            ]

            if _selected_ids:
                st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
                _section("危险操作 · 删除回测记录")
                st.markdown(
                    '<div style="font-size:0.9rem; color:var(--muted); margin-bottom:0.8rem;">'
                    f'已选中 <b>{len(_selected_ids)}</b> 条记录。'
                    '删除后无法恢复，Alpha Memory 中对应的训练数据将永久清除。</div>',
                    unsafe_allow_html=True,
                )
                _del_confirm = st.checkbox(
                    "我已了解此操作不可撤销，将永久删除选中的回测记录",
                    key="sel_del_confirm",
                )
                _sel_pwd_col, _sel_btn_col = st.columns([2, 1], gap="small")
                with _sel_pwd_col:
                    _sel_pwd = st.text_input(
                        "管理员密码", type="password", key="sel_del_pwd",
                        placeholder="输入密码后删除选中记录",
                    )
                with _sel_btn_col:
                    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
                    _sel_admin_pwd = st.secrets.get("LOGIN_PWD", "")
                    _sel_ready = _del_confirm and _sel_pwd == _sel_admin_pwd and _sel_admin_pwd != ""
                    if st.button(
                        f"🗑 删除选中 {len(_selected_ids)} 条",
                        type="primary",
                        disabled=not _sel_ready,
                        key="sel_del_btn",
                        width='stretch',
                    ):
                        _deleted = delete_backtest_record_by_ids(_selected_ids)
                        st.success(f"已删除 {_deleted} 条记录。")
                        st.rerun()


        if _bt_rows:
            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
            _section("完整分析记录")
            _bt_options = {
                f"{r['date']} · {r['sector']} · {r['direction']} [{r['phase'] or 'simple'}]": i
                for i, r in enumerate(_bt_rows)
            }
            _sel_label = st.selectbox("查看完整分析", list(_bt_options.keys()), key="bt_review_sel")
            _sel = _bt_rows[_bt_options[_sel_label]]
            with st.expander(f"📋 {_sel['date']} · {_sel['sector']} · {_sel['direction']}", expanded=True):
                st.markdown(
                    f'<div style="font-size:0.85rem; color:var(--muted); margin-bottom:0.6rem;">'
                    f'Regime: {_sel["regime"]} · VIX: {_sel["vix"] or "—"} · '
                    f'Confidence: {_sel["confidence"] or "—"} · Phase: {_sel["phase"] or "simple"}</div>',
                    unsafe_allow_html=True,
                )
                _render_analysis(_sel["ai_conclusion"])


    # ─────────────────────────────────────────────────────────────────────────────
    # Footer
    # ═══════════════════════════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════════════════════════
    # TAB 6 — 训练进度 (Training Coverage)
    # ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — System  (Parameter Status · Risk Parameters · Sector Universe · Stress Test)
# ═══════════════════════════════════════════════════════════════════════════════
with _tab_system:
    # ── P2-9 Learning Stage Progress Bar ──────────────────────────────────────
    from engine.memory import get_learning_stage
    _ls = get_learning_stage()

    _stage_colors = {
        "cold_start":          "#94A3B8",
        "memory_active":       "#3B82F6",
        "parameter_adaptive":  "#8B5CF6",
        "structural_adaptive": "#10B981",
    }
    _bar_color = _stage_colors.get(_ls.stage, "#94A3B8")
    _nxt_str   = f"→ {_ls.next_threshold} 条解锁下一阶段" if _ls.next_threshold else "已达最高阶段"

    st.markdown(
        f'<div style="background:var(--card); border:1px solid var(--border); border-left:4px solid {_bar_color}; '
        f'border-radius:8px; padding:0.8rem 1.2rem; margin-bottom:1.4rem;">'
        f'<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.4rem;">'
        f'<span style="font-weight:700; font-size:0.95rem; color:var(--text);">🧠 学习阶段：{_ls.label}</span>'
        f'<span style="font-size:0.8rem; color:var(--muted);">已验证决策 {_ls.n_verified} 条 &nbsp;·&nbsp; {_nxt_str}</span>'
        f'</div>'
        f'<div style="font-size:0.82rem; color:var(--muted); margin-bottom:0.5rem;">{_ls.description}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.progress(_ls.progress_frac, text=f"{_ls.label}  {_ls.n_verified}/{_ls.next_threshold or _ls.n_verified} 条")

    _section("Parameter Status · Structural Prior Settings")
    st.markdown(
        '<div style="font-size:0.82rem; color:var(--muted); margin-bottom:1.2rem;">'
        '记录各量化参数的当前固定值、来源与更新条件。'
        '参数在满足触发条件前保持固定，避免数据窥探。'
        '</div>',
        unsafe_allow_html=True,
    )

    from engine.memory import get_clean_zone_stats, get_system_config
    _cz_stats = get_clean_zone_stats()
    _cz_n = _cz_stats.get("clean_b", {}).get("n", 0)

    def _param_card(name: str, value: str, source: str, gate: str,
                    gate_met: bool, note: str = "") -> None:
        _gc = "#10B981" if gate_met else "#F59E0B"
        _gl = "✓ 触发条件已满足" if gate_met else "○ 等待触发条件"
        st.markdown(
            f'<div style="border:1px solid var(--border); border-radius:8px; '
            f'padding:0.9rem 1.2rem; margin-bottom:0.6rem;">'
            f'<div style="display:flex; justify-content:space-between; '
            f'align-items:baseline; margin-bottom:0.4rem;">'
            f'<span style="font-size:1rem; font-weight:700; color:var(--text);">{name}</span>'
            f'<span style="font-family:var(--mono); font-size:1.1rem; font-weight:700; '
            f'color:var(--accent);">{value}</span>'
            f'</div>'
            f'<div style="font-size:0.78rem; color:var(--muted); margin-bottom:0.35rem;">'
            f'<b>来源：</b>{source}</div>'
            f'<div style="font-size:0.78rem; color:var(--muted); margin-bottom:0.35rem;">'
            f'<b>更新条件：</b>{gate}</div>'
            f'<div style="font-size:0.75rem; color:{_gc};">{_gl}</div>'
            + (f'<div style="font-size:0.75rem; color:var(--muted); margin-top:0.3rem;">'
               f'{note}</div>' if note else '')
            + '</div>',
            unsafe_allow_html=True,
        )

    _ps_c1, _ps_c2 = st.columns(2, gap="large")

    with _ps_c1:
        st.markdown(
            '<div style="font-size:0.85rem; font-weight:700; color:var(--muted); '
            'text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;">'
            'Risk Parameters</div>',
            unsafe_allow_html=True,
        )
        _sigma = get_system_config("risk.sigma_target", "0.12")
        _param_card(
            name="σ_target（目标波动率）",
            value=f"{float(_sigma):.1%}",
            source="用户手动写入（Risk Parameters Tab）",
            gate="随时可修改，无门控",
            gate_met=True,
        )
        _lambda_val, _lambda_src = get_lasso_lambda()
        _gate_n_ok  = _cz_n >= _LASSO_GATE_MIN_N
        _param_card(
            name="λ（LASSO 正则化强度）",
            value=f"{_lambda_val:.4f}",
            source=f"{'稳定性准则（Clean Zone 数据驱动）' if _lambda_src == 'stability' else '先验值（复合门控未满足）'}",
            gate=f"n≥{_LASSO_GATE_MIN_N} 且 制度多样性≥{_LASSO_GATE_MIN_REGIMES}（两项同时满足）",
            gate_met=(_lambda_src == "stability"),
            note=f"当前 Clean Zone：{_cz_n} 条 / {_LASSO_GATE_MIN_N}（数量）",
        )

        st.markdown(
            '<div style="font-size:0.85rem; font-weight:700; color:var(--muted); '
            'text-transform:uppercase; letter-spacing:0.08em; '
            'margin-top:1rem; margin-bottom:0.6rem;">'
            'Verification Framework</div>',
            unsafe_allow_html=True,
        )
        _param_card(
            name="Triple-Barrier TP 倍数",
            value="1.0σ",
            source="预承诺固定值（Lopez de Prado 2018）",
            gate="不可校准——验证机制超参数须在数据收集前固定",
            gate_met=True,
            note="修改此值将使已有 Clean Zone 验证结果不可比。",
        )
        _param_card(
            name="Triple-Barrier SL 倍数",
            value="0.7σ",
            source="预承诺固定值",
            gate="不可校准——同上",
            gate_met=True,
        )

    with _ps_c2:
        st.markdown(
            '<div style="font-size:0.85rem; font-weight:700; color:var(--muted); '
            'text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;">'
            'Market Regime</div>',
            unsafe_allow_html=True,
        )
        _param_card(
            name="VIX 低波动阈值",
            value="< 15",
            source="业界标准固定值（CBOE 分类惯例）",
            gate="暂不校准——边际提升可忽略，当前值已广泛使用",
            gate_met=True,
        )
        _param_card(
            name="VIX 正常 / 高波动分界",
            value="15 / 25 / 35",
            source="业界标准固定值",
            gate="暂不校准——同上",
            gate_met=True,
        )

        st.markdown(
            '<div style="font-size:0.85rem; font-weight:700; color:var(--muted); '
            'text-transform:uppercase; letter-spacing:0.08em; '
            'margin-top:1rem; margin-bottom:0.6rem;">'
            'Signal Construction</div>',
            unsafe_allow_html=True,
        )
        _param_card(
            name="动量因子窗口",
            value="1M / 3M / 6M",
            source="学术标准（Jegadeesh & Titman 1993）",
            gate="暂不校准——无循证理由替换标准窗口",
            gate_met=True,
        )
        _param_card(
            name="Σ（协方差矩阵）",
            value="Ledoit-Wolf",
            source="分析公式，历史价格收益率估算",
            gate="γ/λ 框架接入实际决策逻辑后启用",
            gate_met=False,
            note="当前 γ/λ 处于 stub 状态，Σ 无操作性连接，暂不需要校准。",
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Stress Test
# ═══════════════════════════════════════════════════════════════════════════════
with _tab_system:
    from engine.scenarios import SCENARIOS, SCENARIO_CATEGORIES, build_stress_context
    from engine.quant import QuantEngine

    _stress_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_KEY", "")

    _section("Stress Test · 压力情景沙盒")

    st.markdown(
        '<div style="font-size:0.88rem; color:var(--muted); margin-bottom:1rem;">'
        '选择一个预设情景或自定义参数，AI 将在假设情景下重新评估各板块配置方向。'
        '分析结果仅存于当前会话，<b>不写入 Alpha Memory 数据库</b>。'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Scenario selector ─────────────────────────────────────────────────────
    _cat_col, _scen_col = st.columns([2, 3], gap="small")

    with _cat_col:
        _st_category = st.selectbox(
            "情景分类",
            SCENARIO_CATEGORIES,
            key="stress_category",
        )

    _scenarios_in_cat = [s for s in SCENARIOS if s["category"] == _st_category]
    _scen_names       = [s["name"] for s in _scenarios_in_cat]

    with _scen_col:
        _st_scen_name = st.selectbox(
            "情景选择",
            _scen_names,
            key="stress_scenario_name",
        )

    _selected_scenario = next(
        (s for s in _scenarios_in_cat if s["name"] == _st_scen_name), None
    )

    if _selected_scenario:
        # ── Scenario description card ─────────────────────────────────────────
        with st.container(border=True):
            _sc1, _sc2 = st.columns([3, 1])
            with _sc1:
                st.markdown(
                    f'<div style="font-size:1.05rem; font-weight:700; '
                    f'color:var(--text); margin-bottom:0.4rem;">'
                    f'{_selected_scenario["name"]}</div>'
                    f'<div style="font-size:0.88rem; color:var(--muted);">'
                    f'{_selected_scenario["description"]}</div>',
                    unsafe_allow_html=True,
                )
            with _sc2:
                st.markdown(
                    f'<div style="font-size:0.8rem; color:var(--muted); line-height:1.8;">'
                    f'预期时长<br><b style="color:var(--text);">'
                    f'{_selected_scenario.get("duration","—")}</b><br>'
                    f'历史发生概率<br><b style="color:var(--text);">'
                    f'{_selected_scenario.get("probability","—")}</b></div>',
                    unsafe_allow_html=True,
                )

        # ── Sandbox param overrides (only for sandbox scenario) ───────────────
        _sandbox_params = {}
        if _selected_scenario["id"] == "sandbox":
            st.markdown("#### 自定义参数")
            _sb1, _sb2, _sb3, _sb4 = st.columns(4)
            _sandbox_params["vix_override"] = _sb1.number_input(
                "VIX（压力值）", value=30.0, min_value=5.0, max_value=150.0,
                key="sb_vix",
            )
            _sandbox_params["fed_funds_delta"] = _sb2.number_input(
                "利率变动 (bps)", value=0, min_value=-500, max_value=500,
                key="sb_rate",
            )
            _sandbox_params["oil_price_delta_pct"] = _sb3.number_input(
                "油价变动 (%)", value=0.0, min_value=-80.0, max_value=200.0,
                key="sb_oil",
            )
            _sandbox_params["usd_delta_pct"] = _sb4.number_input(
                "美元(DXY)变动 (%)", value=0.0, min_value=-30.0, max_value=30.0,
                key="sb_usd",
            )
            _sandbox_params["custom_note"] = st.text_area(
                "补充说明（注入 AI 上下文）", value="", height=80,
                key="sb_note",
            )
            # Apply sandbox overrides to scenario
            _selected_scenario = dict(_selected_scenario)
            _selected_scenario["params"] = {
                **_selected_scenario["params"], **_sandbox_params,
            }

        # ── Sector impact preview ─────────────────────────────────────────────
        _impacts = _selected_scenario.get("sector_impacts", {})
        if _impacts:
            _benefit = [k for k, v in _impacts.items() if v == 1]
            _hurt    = [k for k, v in _impacts.items() if v == -1]
            _neutral = [k for k, v in _impacts.items() if v == 0]

            def _badge_row(sectors: list, bg: str, color: str) -> str:
                if not sectors:
                    return '<span style="font-size:0.8rem; color:var(--muted);">—</span>'
                return "".join(
                    f'<span style="display:inline-block; margin:2px 4px 2px 0; '
                    f'padding:2px 9px; border-radius:12px; font-size:0.78rem; '
                    f'background:{bg}; color:{color}; white-space:nowrap;">{s}</span>'
                    for s in sectors
                )

            _is_dark_mode = theme.is_dark()
            _bg_green  = "rgba(34,197,94,0.18)"  if _is_dark_mode else "rgba(22,163,74,0.12)"
            _bg_red    = "rgba(239,68,68,0.18)"   if _is_dark_mode else "rgba(220,38,38,0.12)"
            _bg_gray   = "rgba(148,163,184,0.15)" if _is_dark_mode else "rgba(100,116,139,0.10)"
            _col_green = "#4ade80" if _is_dark_mode else "#15803d"
            _col_red   = "#f87171" if _is_dark_mode else "#b91c1c"
            _col_gray  = "#94a3b8" if _is_dark_mode else "#475569"

            st.markdown(
                f'<div style="margin:0.6rem 0 0.2rem 0;">'
                f'<span style="font-size:0.78rem; font-weight:700; color:{_col_green}; '
                f'margin-right:0.5rem;">▲ 受益</span>'
                + _badge_row(_benefit, _bg_green, _col_green)
                + '</div>'
                f'<div style="margin:0.3rem 0 0.2rem 0;">'
                f'<span style="font-size:0.78rem; font-weight:700; color:{_col_red}; '
                f'margin-right:0.5rem;">▼ 受损</span>'
                + _badge_row(_hurt, _bg_red, _col_red)
                + '</div>'
                f'<div style="margin:0.3rem 0 0.5rem 0;">'
                f'<span style="font-size:0.78rem; font-weight:700; color:{_col_gray}; '
                f'margin-right:0.5rem;">· 中性</span>'
                + _badge_row(_neutral, _bg_gray, _col_gray)
                + '</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # ── Analysis target ───────────────────────────────────────────────────
        _stress_sector = st.selectbox(
            "选择分析板块",
            list(SECTOR_ETF.keys()),
            key="stress_sector",
        )

        # ── Run button ────────────────────────────────────────────────────────
        if "stress_results" not in st.session_state:
            st.session_state.stress_results = {}

        _run_stress = st.button(
            "运行压力情景分析",
            type="primary",
            key="run_stress_btn",
        )

        if _run_stress:
            _real_vix = QuantEngine.get_realtime_vix()
            _ctx = build_stress_context(_selected_scenario, base_vix=_real_vix)

            _stress_prompt = (
                f"你是一名机构级压力测试分析师。\n\n"
                f"{_ctx['stress_context']}\n\n"
                f"当前真实 VIX（基准）：{_real_vix}\n"
                f"压力情景 VIX：{_ctx['effective_vix']}\n\n"
                f"请对【{_stress_sector}】板块进行压力情景分析：\n\n"
                "### 1. 情景传导路径\n"
                f"[说明此情景如何通过宏观、行业、资金面影响【{_stress_sector}】板块]\n\n"
                "### 2. 潜在受益/受损逻辑\n"
                "[结合情景参数，评估板块的直接暴露与间接传导]\n\n"
                "### 3. 与历史类似情景对比\n"
                "[参考历史上最接近的市场事件，说明本板块的历史表现]\n\n"
                "### 4. 尾部风险与非线性效应\n"
                "[识别此情景中可能被低估的非线性风险（流动性危机、连锁效应等）]\n\n"
                "### 5. 压力情景下的配置建议\n"
                "[给出超配/标配/低配建议，附对冲逻辑]\n\n"
                "→ 综合判断: [一句话总结]\n\n"
                "### [XAI_ATTRIBUTION]\n"
                "overall_confidence: [0-100]\n"
                "macro_confidence: [0-100]\n"
                "news_confidence: [0-100]\n"
                "technical_confidence: [0-100]\n"
                "signal_drivers: [最多3个压力传导因素]\n"
                "invalidation_conditions: [1-2个情景失效条件]\n"
                "horizon: [季度(3个月) / 半年(6个月)]\n"
                "### [/XAI_ATTRIBUTION]\n\n"
                "写作要求：假设情景分析，逻辑严谨，明确区分已知历史数据与情景推演内容。"
            )

            with st.spinner(f"正在分析 {_stress_sector} × {_selected_scenario['name']}..."):
                try:
                    _genai_model   = _make_model(_stress_key)
                    _stress_output = _genai_model.generate_content(_stress_prompt).text
                    st.session_state.stress_results[
                        f"{_selected_scenario['id']}_{_stress_sector}"
                    ] = {
                        "scenario":   _selected_scenario["name"],
                        "sector":     _stress_sector,
                        "output":     _stress_output,
                        "vix_stress": _ctx["effective_vix"],
                        "ran_at":     datetime.datetime.now().strftime("%H:%M:%S"),
                    }
                    # Persist lightweight record for retrospective review
                    _sp = _selected_scenario.get("params", {})
                    save_stress_test_log(
                        scenario_id       = _selected_scenario["id"],
                        scenario_name     = _selected_scenario["name"],
                        scenario_category = _selected_scenario.get("category", ""),
                        sector            = _stress_sector,
                        effective_vix     = _ctx["effective_vix"],
                        ai_output         = _stress_output,
                        fed_funds_delta   = _sp.get("fed_funds_delta"),
                        oil_delta_pct     = _sp.get("oil_price_delta_pct"),
                        usd_delta_pct     = _sp.get("usd_delta_pct"),
                        custom_note       = _sp.get("custom_note", ""),
                    )
                    st.success("分析完成 · 已保存至复盘记录")
                except Exception as _e:
                    st.error(f"分析失败：{_e}")

        # ── Results ───────────────────────────────────────────────────────────
        _res_key = f"{_selected_scenario['id']}_{_stress_sector}"
        _res     = st.session_state.stress_results.get(_res_key)

        if _res:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            _section(
                f"Stress Analysis · {_res['sector']} × {_res['scenario']} "
                f"· VIX {_res['vix_stress']:.0f} · {_res['ran_at']}"
            )
            with st.container(border=True):
                _render_analysis(_res["output"])

        # ── Session history ───────────────────────────────────────────────────
        _all_results = st.session_state.stress_results
        if len(_all_results) > 1:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            _section("本次会话 · 已运行情景")
            _hist_opts = {
                f"{v['sector']} × {v['scenario']} ({v['ran_at']})": k
                for k, v in _all_results.items()
            }
            _picked = st.selectbox(
                "查看历史分析", list(_hist_opts.keys()),
                key="stress_history_picker",
            )
            if _picked and _hist_opts[_picked] != _res_key:
                _prev = _all_results[_hist_opts[_picked]]
                with st.container(border=True):
                    _render_analysis(_prev["output"])

        # ── Retrospective review (DB) ─────────────────────────────────────────
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        _section("历史复盘 · Stress Test Log")
        _st_history = get_stress_test_history(limit=50)
        if not _st_history:
            st.markdown(
                '<div style="font-size:0.88rem; color:var(--muted);">'
                '暂无历史记录。运行第一次压力情景分析后将自动保存。</div>',
                unsafe_allow_html=True,
            )
        else:
            _dir_color = {"超配": "#10b981", "低配": "#ef4444", "标配": "#6366f1"}
            for _h in _st_history:
                _dc = _dir_color.get(_h["ai_direction"], "#94a3b8")
                _params = []
                if _h["fed_funds_delta"] is not None:
                    _sign = "+" if _h["fed_funds_delta"] >= 0 else ""
                    _params.append(f"利率 {_sign}{_h['fed_funds_delta']}bps")
                if _h["oil_delta_pct"] is not None:
                    _sign = "+" if _h["oil_delta_pct"] >= 0 else ""
                    _params.append(f"油价 {_sign}{_h['oil_delta_pct']:.0f}%")
                if _h["usd_delta_pct"] is not None:
                    _sign = "+" if _h["usd_delta_pct"] >= 0 else ""
                    _params.append(f"DXY {_sign}{_h['usd_delta_pct']:.0f}%")
                _param_str = " · ".join(_params) if _params else "默认参数"
                st.markdown(
                    f'<div style="border:1px solid var(--border); border-left:3px solid {_dc}; '
                    f'border-radius:6px; padding:0.6rem 1rem; margin-bottom:0.5rem;">'
                    f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                    f'<span style="font-size:0.9rem; font-weight:700; color:var(--text);">'
                    f'{_h["scenario_name"]} · {_h["sector"]}</span>'
                    f'<span style="font-size:0.78rem; color:var(--muted);">{_h["run_at"]}</span>'
                    f'</div>'
                    f'<div style="font-size:0.8rem; color:var(--muted); margin:0.2rem 0;">'
                    f'VIX {_h["effective_vix"]:.0f} · {_param_str} · '
                    f'<span style="color:{_dc}; font-weight:700;">{_h["ai_direction"]}</span>'
                    f'</div>'
                    f'<div style="font-size:0.82rem; color:var(--text); line-height:1.5;">'
                    f'{_h["ai_summary"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — Risk Parameters
# ═══════════════════════════════════════════════════════════════════════════════
with _tab_system:
    _section("Risk Parameters · Structural Prior Settings")

    st.markdown(
        "参数作为**结构性先验**写入数据库，系统重启后保留。"
        "γ 由目标波动率反推得出；λ 在 Clean Zone 样本量达到门控阈值前使用结构性先验，"
        "达到后自动切换为稳定性准则。",
        unsafe_allow_html=False,
    )

    # ── γ block ──────────────────────────────────────────────────────────────
    _section("γ（风险厌恶系数）— 由目标波动率反推")
    _g_col1, _g_col2 = st.columns([2, 3], gap="large")

    with _g_col1:
        _current_vol = float(get_system_config("risk.vol_target_ann", "0.15"))
        _new_vol = st.number_input(
            "目标年化波动率上限 σ_target",
            min_value=0.01, max_value=1.00,
            value=_current_vol, step=0.01, format="%.2f",
            help="例：0.15 表示 15%。γ 将由此值与信号协方差矩阵反推，不需要手动填写 γ。",
            key="risk_vol_target",
        )
        if st.button("保存 σ_target", key="risk_save_vol_btn", type="primary"):
            set_system_config("risk.vol_target_ann", str(_new_vol))
            st.success(f"已保存：σ_target = {_new_vol:.2%}")

    with _g_col2:
        _saved_vol = float(get_system_config("risk.vol_target_ann", "0.15"))
        st.metric("σ_target（数据库当前值）", f"{_saved_vol:.2%}")
        st.caption("γ 在信号协方差估计完成后由 QuantEngine 调用此值反推，不在此处直接显示。")

    st.divider()

    # ── λ block ──────────────────────────────────────────────────────────────
    _section("λ（LASSO 正则化强度）— 门控稳定性准则")

    _cz_n      = get_clean_zone_stats().get("clean_b", {}).get("n", 0)
    _lambda_val, _lambda_src = get_lasso_lambda()
    _gate_open = (_lambda_src == "stability")

    _status_color = "#22C55E" if _gate_open else "#F59E0B"
    _status_label = (
        f"复合门控已开启 — 稳定性准则（n={_cz_n}，制度多样性已满足）"
        if _gate_open
        else (
            f"复合门控锁定 — 使用先验值 "
            f"（数量：{_cz_n}/{_LASSO_GATE_MIN_N}，"
            f"需同时满足制度多样性≥{_LASSO_GATE_MIN_REGIMES}）"
        )
    )
    st.markdown(
        f'<div style="padding:0.5rem 1rem; border-radius:4px; border-left:4px solid {_status_color}; '
        f'margin-bottom:1rem; font-size:0.92rem;">{_status_label}</div>',
        unsafe_allow_html=True,
    )

    _l_col1, _l_col2 = st.columns([2, 3], gap="large")

    with _l_col1:
        _current_lambda = float(get_system_config("risk.lasso_lambda_prior", "0.10"))
        _new_lambda = st.number_input(
            "λ 人工先验值（基于理论设定，非数据驱动）",
            min_value=0.001, max_value=10.0,
            value=_current_lambda, step=0.01, format="%.3f",
            help=(
                "此值由研究者根据金融理论主动设定，而非系统从数据中学习。"
                "在 Clean Zone 样本量充足（n≥20 且跨制度≥2）之前，"
                "数据驱动的参数估计在统计上不可靠，人工先验优于噪声拟合。"
                "门控开启后此值作为稳定性准则的搜索锚点。"
            ),
            key="risk_lambda_prior",
            disabled=_gate_open,
        )
        if not _gate_open:
            if st.button("保存 λ 先验", key="risk_save_lambda_btn", type="primary"):
                set_system_config("risk.lasso_lambda_prior", str(_new_lambda))
                st.success(f"已保存：λ_prior = {_new_lambda:.3f}")
            st.caption(
                "当前处于人工先验阶段：参数由研究者基于理论设定，"
                "系统不做数据驱动调整，直至复合门控（n≥20 × 制度≥2）满足。"
            )
        else:
            st.caption("门控已开启，先验值不可编辑。λ 由稳定性准则自动计算。")

    with _l_col2:
        st.metric("当前生效 λ", f"{_lambda_val:.4f}", help=f"来源：{_lambda_src}")
        _prog = min(_cz_n / _LASSO_GATE_MIN_N, 1.0)
        st.progress(_prog, text=f"数量进度：{_cz_n} / {_LASSO_GATE_MIN_N}（制度多样性需同步满足≥{_LASSO_GATE_MIN_REGIMES}）")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9 — Sector Universe
# ═══════════════════════════════════════════════════════════════════════════════
with _tab_system:
    _section("Sector Universe · 自定义板块宇宙")

    import json as _json

    st.markdown(
        "覆盖默认的 18 个板块 ETF 映射。修改将持久化到数据库，重启后仍有效。  \n"
        "⚠️ 板块数 < 8 时 MSM 和组合优化结果可能不可靠。"
    )

    _active_etf = get_active_sector_etf()
    _default_etf = dict(SECTOR_ETF)
    _is_custom = _active_etf != _default_etf

    if _is_custom:
        st.info(f"当前使用自定义板块宇宙（{len(_active_etf)} 个板块）。")
    else:
        st.caption(f"当前使用默认板块宇宙（{len(_active_etf)} 个板块）。")

    # ── Current universe table ─────────────────────────────────────────────────
    _sec_col1, _sec_col2 = st.columns([3, 2])

    with _sec_col1:
        st.markdown("**当前板块列表**")
        _etf_rows = [{"板块名称": k, "ETF代码": v} for k, v in _active_etf.items()]
        _etf_df = pd.DataFrame(_etf_rows)
        st.dataframe(_etf_df, use_container_width=True, hide_index=True,
                     height=min(38 * len(_etf_df) + 38, 500))

    with _sec_col2:
        # ── Remove sector ──────────────────────────────────────────────────────
        st.markdown("**移除板块**")
        _remove_choice = st.selectbox(
            "选择要移除的板块",
            options=list(_active_etf.keys()),
            key="sector_remove_choice",
        )
        if st.button("移除板块", key="sector_remove_btn", type="secondary"):
            if len(_active_etf) <= 4:
                st.error("板块数已达最低限制（4个），无法继续移除。")
            else:
                _new_etf = {k: v for k, v in _active_etf.items() if k != _remove_choice}
                set_system_config("sector_etf_overrides", _json.dumps(_new_etf, ensure_ascii=False))
                st.success(f"已移除：{_remove_choice}")
                if len(_new_etf) < 8:
                    st.warning(f"⚠️ 当前板块数为 {len(_new_etf)}，建议保持至少 8 个。")
                st.rerun()

    st.divider()

    # ── ETF Semantic Search ────────────────────────────────────────────────────
    st.markdown("**ETF 语义搜索**")
    st.caption("输入关键词（中英文均可），自动匹配最相关板块及 ETF 代码。")
    _search_col1, _search_col2 = st.columns([3, 1])
    _search_query = _search_col1.text_input(
        "搜索关键词", key="etf_search_query",
        placeholder="例：新能源储能  /  semiconductor AI  /  黄金避险",
        label_visibility="collapsed",
    )
    _search_topk = _search_col2.selectbox("返回数量", [3, 5, 10], key="etf_search_topk",
                                          label_visibility="collapsed")
    if _search_query.strip():
        try:
            from engine.etf_search import get_search_engine
            _se = get_search_engine()
            _sr = _se.search(_search_query.strip(), top_k=_search_topk)
            _sr_rows = []
            for _r in _sr:
                _sr_rows.append({
                    "排名":   _r.rank,
                    "板块":   _r.sector,
                    "ETF":    _r.ticker,
                    "相似度": f"{_r.score:.3f}",
                    "匹配依据": _se.explain(_search_query.strip(), _r),
                    "已在宇宙": "✅" if _r.sector in _active_etf else "—",
                })
            st.dataframe(pd.DataFrame(_sr_rows), use_container_width=True, hide_index=True)
            _fill_opts = [f"{r['板块']}  ({r['ETF']})" for r in _sr_rows if r["已在宇宙"] == "—"]
            if _fill_opts:
                _fill_pick = st.selectbox("一键填入添加表单", ["（不填入）"] + _fill_opts,
                                          key="etf_search_fill")
                if _fill_pick != "（不填入）":
                    _fill_sector = _fill_pick.split("  (")[0]
                    _fill_ticker = _fill_pick.split("(")[1].rstrip(")")
                    st.session_state["new_sector_name"]   = _fill_sector
                    st.session_state["new_sector_ticker"] = _fill_ticker
        except Exception as _se_err:
            st.warning(f"语义搜索暂不可用：{_se_err}")

    st.divider()

    # ── Add sector ─────────────────────────────────────────────────────────────
    st.markdown("**添加新板块**")
    _add_col1, _add_col2, _add_col3 = st.columns([2, 1, 1])
    _new_sector_name   = _add_col1.text_input("板块名称（中文）", key="new_sector_name",
                                               placeholder="例：欧洲蓝筹")
    _new_sector_ticker = _add_col2.text_input("ETF代码（美股）",  key="new_sector_ticker",
                                               placeholder="例：EWG")

    if _add_col3.button("验证并添加", key="sector_add_btn", type="primary"):
        _ns = _new_sector_name.strip()
        _nt = _new_sector_ticker.strip().upper()
        if not _ns or not _nt:
            st.error("板块名称和 ETF 代码不能为空。")
        elif _ns in _active_etf:
            st.error(f"板块「{_ns}」已存在。")
        elif _nt in _active_etf.values():
            _existing = next(k for k, v in _active_etf.items() if v == _nt)
            st.error(f"ETF 代码 {_nt} 已被板块「{_existing}」使用。")
        else:
            # Validate ticker via yfinance
            try:
                import yfinance as _yf
                _info = _yf.Ticker(_nt).fast_info
                _price = getattr(_info, "last_price", None)
                if _price and float(_price) > 0:
                    _new_etf = dict(_active_etf)
                    _new_etf[_ns] = _nt
                    set_system_config("sector_etf_overrides",
                                      _json.dumps(_new_etf, ensure_ascii=False))
                    st.success(f"已添加：{_ns} → {_nt}（最新价 {_price:.2f}）")
                    st.rerun()
                else:
                    st.error(f"无法获取 {_nt} 的价格数据，请检查代码是否正确。")
            except Exception as _e:
                st.error(f"yfinance 验证失败：{_e}")

    st.divider()

    # ── Reset to defaults ──────────────────────────────────────────────────────
    st.markdown("**重置为默认板块宇宙**")
    st.caption(f"默认板块宇宙包含 {len(_default_etf)} 个板块。")
    if st.button("重置为默认", key="sector_reset_btn", type="secondary"):
        set_system_config("sector_etf_overrides", "")
        st.success("已重置为默认板块宇宙。")
        st.rerun()

with _tab_pending:
    st.divider()
    _section("Decision Monitor · Clean Zone 决策追踪")

    _pending_monitor = get_pending_decisions_for_monitor()

    if not _pending_monitor:
        _dm_is_dark = theme.is_dark()
        _dm_bg = "rgba(255,255,255,0.03)" if _dm_is_dark else "#F8FAFC"
        _dm_bd = "rgba(255,255,255,0.08)" if _dm_is_dark else "#CBD5E1"
        _dm_fg = "rgba(255,255,255,0.35)" if _dm_is_dark else "#94A3B8"
        st.markdown(
            f'<div style="background:{_dm_bg}; border:1px dashed {_dm_bd}; border-radius:6px; '
            f'padding:2.5rem; text-align:center; color:{_dm_fg}; font-size:1.05rem;">'
            f'暂无待验证的 Clean Zone 决策。运行各分析模块后，决策将在到达验证窗口前显示于此。'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        _dm_n_overdue    = sum(1 for d in _pending_monitor if d["urgency"] == "overdue")
        _dm_n_approaching = sum(1 for d in _pending_monitor if d["urgency"] == "approaching")
        _dm_n_normal     = sum(1 for d in _pending_monitor if d["urgency"] == "normal")

        _dmc1, _dmc2, _dmc3, _dmc4 = st.columns(4)
        _dmc1.metric("待验证决策", len(_pending_monitor))
        _dmc2.metric("已过期限", _dm_n_overdue,
                     delta="需关注" if _dm_n_overdue > 0 else None,
                     delta_color="inverse" if _dm_n_overdue > 0 else "off")
        _dmc3.metric("即将到期 (≤14天)", _dm_n_approaching,
                     delta="检查失效条件" if _dm_n_approaching > 0 else None,
                     delta_color="inverse" if _dm_n_approaching > 0 else "off")
        _dmc4.metric("正常追踪", _dm_n_normal)

        st.divider()

        st.markdown(
            '<div style="font-size:0.78rem; color:var(--muted); margin-bottom:1rem;">'
            '<span style="color:#EF4444; font-weight:700;">■ 已过期限</span>&emsp;'
            'horizon 已到，等待 Triple-Barrier 触发&emsp;'
            '<span style="color:#F59E0B; font-weight:700;">■ 即将到期</span>&emsp;'
            '≤14 天内到期，建议核查失效条件&emsp;'
            '<span style="color:#10B981; font-weight:700;">■ 正常追踪</span>&emsp;'
            '在有效观察窗口内</div>',
            unsafe_allow_html=True,
        )

        # Pre-compute ES for all unique tickers (cached via QuantEngine)
        from engine.quant import QuantEngine as _QE
        _dm_es_cache: dict[str, float] = {}
        for _t in {d["ticker"] for d in _pending_monitor if d["ticker"] not in ("—", "")}:
            try:
                _t_ret = _QE.get_market_data((_t,))
                if not _t_ret.empty:
                    _s = _t_ret.iloc[:, 0].dropna()
                    _cutoff = float(np.quantile(_s, 0.05))
                    _tail   = _s[_s <= _cutoff]
                    _dm_es_cache[_t] = float(_tail.mean()) if len(_tail) > 0 else _cutoff
            except Exception:
                pass

        _dm_events = _upcoming_macro_events(window_days=5)

        _DM_URGENCY_COLOR = {"overdue": "#EF4444", "approaching": "#F59E0B", "normal": "#10B981"}
        _DM_URGENCY_LABEL = {"overdue": "已过期限", "approaching": "即将到期", "normal": "正常追踪"}
        _DM_DIR_COLOR = {
            "超配": "#10B981", "低配": "#EF4444", "标配": "#60A5FA",
            "拦截": "#F59E0B", "通过": "#A78BFA",
        }
        _DM_TAB_LABEL = {
            "sector": "Sector Risk", "audit": "Quant Audit",
            "scanner": "Alpha Scanner", "macro": "Macro",
        }

        for _dm_d in _pending_monitor:
            _dm_uc   = _DM_URGENCY_COLOR.get(_dm_d["urgency"], "#6B7280")
            _dm_ul   = _DM_URGENCY_LABEL.get(_dm_d["urgency"], _dm_d["urgency"])
            _dm_dc   = _DM_DIR_COLOR.get(_dm_d["direction"], "var(--muted)")
            _dm_tl   = _DM_TAB_LABEL.get(_dm_d["tab_type"], _dm_d["tab_type"])
            _dm_conf = f"{_dm_d['confidence_score']}%" if _dm_d["confidence_score"] is not None else "—"
            _dm_inv    = _dm_d["invalidation_conditions"] or "未记录"
            _dm_ticker = _dm_d.get("ticker", "—")
            _dm_es     = _dm_es_cache.get(_dm_ticker)
            _dm_spread = _etf_spread_bps(_dm_ticker) if _dm_ticker not in ("—", "") else None
            _dm_days_str = (
                f"已逾期 {-_dm_d['days_to_deadline']} 天"
                if _dm_d["days_to_deadline"] < 0
                else f"剩余 {_dm_d['days_to_deadline']} 天"
            )
            _dm_drift_badge = (
                '<span style="font-size:0.7rem; font-weight:700; color:#F59E0B; '
                'background:#F59E0B22; padding:0.1rem 0.5rem; border-radius:3px; '
                'margin-left:0.4rem;">制度已漂移</span>'
                if _dm_d.get("regime_drifted") else ""
            )
            _dm_pre_label = _dm_d.get("human_label", "")
            _dm_pre_label_display = {
                "pre_strong": "逻辑清晰", "pre_uncertain": "有疑虑", "pre_poor": "明显缺陷",
            }.get(_dm_pre_label, "")
            _dm_pre_badge = (
                f'<span style="font-size:0.7rem; font-weight:700; color:#A78BFA; '
                f'background:#A78BFA22; padding:0.1rem 0.5rem; border-radius:3px; '
                f'margin-left:0.4rem;">人工预评: {_dm_pre_label_display}</span>'
                if _dm_pre_label_display else ""
            )

            st.markdown(
                f'<div style="border-left:4px solid {_dm_uc}; border:1px solid {_dm_uc}33; '
                f'border-left:4px solid {_dm_uc}; border-radius:6px; '
                f'padding:0.9rem 1.2rem; margin-bottom:0.7rem;">'
                f'<div style="display:flex; justify-content:space-between; '
                f'align-items:center; margin-bottom:0.5rem;">'
                f'<div style="display:flex; gap:0.8rem; align-items:center; flex-wrap:wrap;">'
                f'<span style="font-size:1rem; font-weight:700; color:var(--text);">'
                f'{_dm_d["sector_name"]}</span>'
                f'<span style="font-size:0.75rem; color:{_dm_dc}; font-weight:700; '
                f'background:{_dm_dc}22; padding:0.1rem 0.45rem; border-radius:3px;">'
                f'{_dm_d["direction"]}</span>'
                f'<span style="font-size:0.72rem; color:var(--muted);">{_dm_tl}</span>'
                f'{_dm_drift_badge}{_dm_pre_badge}</div>'
                f'<span style="font-size:0.8rem; font-weight:700; color:{_dm_uc};">'
                f'{_dm_ul} · {_dm_days_str}</span></div>'
                f'<div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; '
                f'gap:0.4rem; margin-bottom:0.6rem;">'
                f'<div><span style="font-size:0.68rem; color:var(--muted); '
                f'text-transform:uppercase;">决策日期</span><br>'
                f'<span style="font-family:var(--mono); font-size:0.88rem;">'
                f'{_dm_d["decision_date"]}</span></div>'
                f'<div><span style="font-size:0.68rem; color:var(--muted); '
                f'text-transform:uppercase;">期限</span><br>'
                f'<span style="font-family:var(--mono); font-size:0.88rem;">'
                f'{_dm_d["deadline_date"]} ({_dm_d["horizon"]})</span></div>'
                f'<div><span style="font-size:0.68rem; color:var(--muted); '
                f'text-transform:uppercase;">置信度</span><br>'
                f'<span style="font-family:var(--mono); font-size:0.88rem;">'
                f'{_dm_conf}</span></div>'
                f'<div><span style="font-size:0.68rem; color:var(--muted); '
                f'text-transform:uppercase;">宏观制度</span><br>'
                f'<span style="font-size:0.88rem;">{_dm_d["macro_regime"]}</span></div>'
                f'</div>'
                f'<div style="background:var(--surface2); border-radius:4px; '
                f'padding:0.45rem 0.7rem;">'
                f'<span style="font-size:0.68rem; color:var(--muted); '
                f'text-transform:uppercase; letter-spacing:0.06em;">失效条件&nbsp;</span>'
                f'<span style="font-size:0.88rem; color:var(--text);">{_dm_inv}</span>'
                f'</div>'
                + (
                    f'<div style="display:flex; gap:1.2rem; align-items:center; '
                    f'margin-top:0.55rem; padding:0.35rem 0.7rem; '
                    f'background:var(--surface2); border-radius:4px; flex-wrap:wrap;">'
                    + (
                        f'<span style="font-size:0.72rem; color:var(--muted);">尾部风险&nbsp;'
                        f'<b style="color:#F87171; font-family:var(--mono);">'
                        f'ES(5%)={_dm_es*100:.1f}%</b></span>'
                        if _dm_es is not None else ""
                    )
                    + (
                        f'<span style="font-size:0.72rem; color:var(--muted);">成本&nbsp;'
                        f'<b style="font-family:var(--mono);">~{_dm_spread:.0f}bp</b></span>'
                        if _dm_spread is not None else ""
                    )
                    + "".join(
                        '<span style="font-size:0.72rem; font-weight:700; color:#FBBF24; '
                        'background:#FBBF2422; padding:0.1rem 0.5rem; border-radius:3px;">'
                        + f'⚠ {ev["name"]} '
                        + ("今日" if ev["days"] == 0 else ("明日" if ev["days"] == 1 else f'{ev["days"]}天后'))
                        + '</span>'
                        for ev in _dm_events
                    )
                    + f'</div>'
                    if (_dm_es is not None or _dm_spread is not None or _dm_events)
                    else ""
                )
                + f'</div>',
                unsafe_allow_html=True,
            )

            # Pre-verification human annotation
            _dm_label_options = {
                "（未标注）": "", "逻辑清晰": "pre_strong",
                "有疑虑": "pre_uncertain", "明显缺陷": "pre_poor",
            }
            _dm_current_display = {v: k for k, v in _dm_label_options.items()}.get(
                _dm_pre_label, "（未标注）"
            )
            _, _dm_col_widget = st.columns([3, 1])
            with _dm_col_widget:
                _dm_selected = st.selectbox(
                    "人工预评",
                    options=list(_dm_label_options.keys()),
                    index=list(_dm_label_options.keys()).index(_dm_current_display),
                    key=f"dm_prelabel_{_dm_d['id']}",
                    label_visibility="collapsed",
                )
                _dm_new_val = _dm_label_options[_dm_selected]
                if _dm_new_val != _dm_pre_label:
                    if _dm_new_val:
                        set_human_label(_dm_d["id"], _dm_new_val)
                    elif _dm_pre_label.startswith("pre_"):
                        with SessionFactory() as _dm_s:
                            _dm_rec = _dm_s.get(DecisionLog, _dm_d["id"])
                            if _dm_rec:
                                _dm_rec.human_label = None
                                _dm_s.commit()
                    st.rerun()

# ── System Tab: Skill Library Status ─────────────────────────────────────────
with _tab_system:
    st.divider()
    _section("Skill Library · 知识压缩状态")
    st.markdown(
        '<div style="font-size:0.82rem; color:var(--muted); margin-bottom:1rem;">'
        '每个【板块 × 宏观制度】单元格在积累 ≥5 条已验证训练集决策后，系统将通过 LLM 把经验压缩成'
        '行为指令注入下次分析 prompt。此面板显示当前覆盖状态。</div>',
        unsafe_allow_html=True,
    )
    from engine.memory import SkillLibrary as _SkillLib
    with SessionFactory() as _sl_sess:
        _sl_rows = _sl_sess.query(_SkillLib).order_by(_SkillLib.updated_at.desc()).all()

    if not _sl_rows:
        st.info(
            "当前 Skill Library 为空。\n\n"
            "**触发条件**：同一板块 × 宏观制度下积累 ≥5 条训练集已验证决策后，"
            "运行「Verify Pending Decisions」时自动压缩。"
        )
    else:
        _sl_data = []
        for _sl in _sl_rows:
            _sl_data.append({
                "板块":     _sl.sector_name,
                "宏观制度": _sl.macro_regime,
                "版本":     f"v{_sl.version}",
                "样本数":   _sl.sample_count,
                "准确率":   f"{_sl.avg_accuracy:.0%}" if _sl.avg_accuracy else "—",
                "平均PQ":   f"{_sl.avg_payoff_quality:.2f}" if _sl.avg_payoff_quality else "—",
                "更新时间": _sl.updated_at.strftime("%Y-%m-%d") if _sl.updated_at else "—",
            })
        st.dataframe(pd.DataFrame(_sl_data), hide_index=True, use_container_width=True)

        # Show skill text for selected cell
        _sl_opts = [f"{r['板块']} × {r['宏观制度']}" for r in _sl_data]
        _sl_pick = st.selectbox("查看行为指令", _sl_opts, key="sl_inspect_pick")
        if _sl_pick:
            _sl_sel = next((r for r in _sl_rows if f"{r.sector_name} × {r.macro_regime}" == _sl_pick), None)
            if _sl_sel:
                st.markdown(
                    f'<div style="background:var(--card); border:1px solid var(--border); '
                    f'border-left:3px solid #6366F1; border-radius:6px; padding:0.9rem 1.2rem; '
                    f'font-size:0.92rem; color:var(--text); line-height:1.7;">'
                    f'{_sl_sel.skill_text}</div>',
                    unsafe_allow_html=True,
                )

# ── P2-16 熔断机制状态 ────────────────────────────────────────────────────────
with _tab_system:
    st.divider()
    _section("熔断机制 · Circuit Breaker Status")
    from engine.circuit_breaker import (
        get_status as _cb_get_status, manual_reset as _cb_reset,
        LEVEL_NONE, LEVEL_LIGHT, LEVEL_MEDIUM, LEVEL_SEVERE,
    )
    _cb_state = _cb_get_status()
    _cb_colors = {
        LEVEL_NONE:   ("#10B981", "✅ 正常运行"),
        LEVEL_LIGHT:  ("#F59E0B", "⚠️ 轻度异常（数据源降级）"),
        LEVEL_MEDIUM: ("#F97316", "🟡 中度警戒（LLM 配额告急）"),
        LEVEL_SEVERE: ("#EF4444", "🔴 严重熔断（已暂停自动信号生成）"),
    }
    _cb_color, _cb_label = _cb_colors.get(_cb_state.level, ("#94A3B8", "未知"))
    st.markdown(
        f'<div style="background:var(--card); border:1px solid var(--border); '
        f'border-left:4px solid {_cb_color}; border-radius:8px; '
        f'padding:0.8rem 1.2rem; margin-bottom:1rem;">'
        f'<div style="font-weight:700; color:{_cb_color}; margin-bottom:0.3rem;">{_cb_label}</div>'
        f'<div style="font-size:0.82rem; color:var(--muted);">{_cb_state.reason or "无异常"}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if _cb_state.level == LEVEL_SEVERE:
        st.error(
            "⚠️ 当前处于严重熔断状态，所有自动信号生成已暂停。\n\n"
            "请人工确认市场情况后点击下方按钮恢复运行。"
        )
        _reset_reason = st.text_input("恢复理由（必填）", key="cb_reset_reason",
                                      placeholder="例：VIX 已回落至正常区间，确认非系统性风险")
        if st.button("✅ 手动恢复运行", key="cb_manual_reset",
                     type="primary", disabled=not _reset_reason.strip()):
            _cb_reset(reason=_reset_reason.strip())
            st.success("熔断已解除，系统恢复自动运行。")
            st.rerun()
    elif _cb_state.level == LEVEL_MEDIUM and _cb_state.quota_frac:
        st.info(
            f"当日 API 配额已使用 {_cb_state.quota_frac:.0%}，"
            "非核心 LLM 调用已暂停。Quota 次日自动重置后恢复。"
        )

# ── P2-18 TradingCycleOrchestrator 周期历史 ───────────────────────────────────
with _tab_system:
    st.divider()
    _section("交易周期历史 · Cycle Run Log")
    from engine.orchestrator import TradingCycleOrchestrator as _TCO
    _tco = _TCO()

    # Pending gate approvals
    _pending_gates = _tco.get_pending_gates()
    if _pending_gates:
        st.warning(f"⏳ {len(_pending_gates)} 个周期等待人工审批")
        for _pg in _pending_gates:
            _gate_labels = {
                "analysis_draft":      "分析草稿审批",
                "risk_approval":       "风控建议审批",
                "monthly_rebalance":   "月度再平衡审批",
                "covariance_override": "协方差覆盖审批",
            }
            _gl = _gate_labels.get(_pg["gate"], _pg["gate"])
            with st.expander(f"Cycle #{_pg['id']} · {_pg['cycle_type']} · {_pg['as_of_date']} · 等待：{_gl}"):
                _approve_col, _reject_col = st.columns(2)
                with _approve_col:
                    if st.button("✅ 批准", key=f"gate_approve_{_pg['id']}"):
                        _tco.approve_gate(_pg["id"], approved=True, note="Admin UI 批准")
                        st.success("已批准，执行层将继续运行。")
                        st.rerun()
                with _reject_col:
                    if st.button("❌ 拒绝", key=f"gate_reject_{_pg['id']}"):
                        _tco.approve_gate(_pg["id"], approved=False, note="Admin UI 拒绝")
                        st.warning("已拒绝，本次周期终止。")
                        st.rerun()

    # Recent cycle history table
    _recent = _tco.get_recent_cycles(n=15)
    if _recent:
        import pandas as _pd_cycle
        _cyc_df = _pd_cycle.DataFrame(_recent)[
            ["id", "cycle_type", "as_of_date", "status", "gate", "elapsed_s", "started_at"]
        ]
        _cyc_df.columns = ["ID", "类型", "日期", "状态", "闸门", "耗时(s)", "启动时间"]
        _status_icons = {
            "completed": "✅", "failed": "❌", "running": "⏳",
            "pending_gate": "🔒", "approved": "✅✓", "rejected": "🚫",
        }
        _cyc_df["状态"] = _cyc_df["状态"].map(lambda s: f"{_status_icons.get(s, '')} {s}")
        st.dataframe(_cyc_df, use_container_width=True, hide_index=True)
    else:
        st.info("暂无周期运行记录。首次调用 TradingCycleOrchestrator 后此处自动更新。")

# ── P2-10 实验日志哈希链验证 ──────────────────────────────────────────────────
with _tab_system:
    st.divider()
    _section("实验日志哈希链 · Chain Integrity")
    st.markdown(
        '<div style="font-size:0.82rem; color:var(--muted); margin-bottom:1rem;">'
        '每条 DecisionLog 记录的 SHA-256 哈希链接到前一条，形成可验证的完整性链。<br>'
        '<b>诚实声明</b>：整链重算时无法检测串改，是可信度信号而非安全机制。</div>',
        unsafe_allow_html=True,
    )
    if st.button("验证链完整性", key="verify_hash_chain"):
        from engine.memory import verify_chain_integrity
        _chain_ok, _chain_broken, _chain_total = verify_chain_integrity()
        if _chain_total == 0:
            st.info("尚无带 chain_hash 的记录（chain_hash 从下次 save_decision() 起开始写入）。")
        elif _chain_broken == 0:
            st.success(f"✅ 链完整 — 已验证 {_chain_total} 条记录，0 条断链。")
        else:
            st.error(
                f"❌ 发现 {_chain_broken} 条断链（共 {_chain_total} 条）。"
                "可能原因：DB 直接修改、记录删除、迁移前旧记录无 chain_hash。"
            )

with _tab_system:
    st.divider()
    _section("Universe 管理 · ETF 注册表")
    st.markdown(
        '<div style="font-size:0.82rem; color:var(--muted); margin-bottom:1rem;">'
        'P2-11 动态 Universe。批次 0=初始18 / 批次 1=批次A权益因子 / 批次 2=批次B跨资产（需 PRE-8 验证后手动激活）。</div>',
        unsafe_allow_html=True,
    )
    try:
        from engine.universe_manager import get_active_universe, universe_health_check
        from engine.memory import SessionFactory
        from engine.universe_manager import UniverseETF as _UE

        _ucol1, _ucol2, _ucol3 = st.columns(3)
        with SessionFactory() as _us:
            _all_etfs = _us.query(_UE).order_by(_UE.batch, _UE.id).all()
        _active_n  = sum(1 for e in _all_etfs if e.active)
        _batch_a_n = sum(1 for e in _all_etfs if e.batch == 1 and e.active)
        _batch_b_n = sum(1 for e in _all_etfs if e.batch == 2 and e.active)
        _ucol1.metric("活跃 ETF 总数", _active_n)
        _ucol2.metric("批次 A 活跃", _batch_a_n)
        _ucol3.metric("批次 B 活跃", _batch_b_n)

        import pandas as _pd
        _etf_df = _pd.DataFrame([{
            "板块": e.sector,
            "Ticker": e.ticker,
            "资产类别": e.asset_class,
            "批次": e.batch,
            "成立日": str(e.inception_date) if e.inception_date else "—",
            "状态": "✅ 活跃" if e.active else "❌ 停用",
            "停用日期": str(e.removed_at) if e.removed_at else "—",
        } for e in _all_etfs])
        st.dataframe(_etf_df, use_container_width=True, hide_index=True)

        if st.button("运行月度健康检查（ADV 审核）", key="universe_health_check"):
            with st.spinner("正在检查各 ETF 成交量..."):
                _hr = universe_health_check()
            if _hr.inactive_flagged:
                st.warning(f"已标记为 inactive：{', '.join(_hr.inactive_flagged)}")
            else:
                st.success(f"全部 {_active_n} 个 ETF ADV 达标，无需标记。")
            if _hr.warnings:
                with st.expander("警告详情"):
                    for w in _hr.warnings:
                        st.text(w)
    except Exception as _ue:
        st.error(f"Universe Manager 加载失败: {_ue}")

# ─────────────────────────────────────────────────────────────────────────────
_footer_border = "rgba(255,255,255,0.08)" if theme.is_dark() else "#CBD5E1"
_footer_color  = "#7D8590" if theme.is_dark() else "#94A3B8"
st.markdown(f"""
<div style="border-top:1px solid {_footer_border}; margin-top:2rem; padding-top:0.8rem;
            display:flex; justify-content:space-between;">
  <span style="font-size:0.68rem; color:{_footer_color}; text-transform:uppercase;
               letter-spacing:0.08em;">Macro Alpha Pro · Admin Panel</span>
  <span style="font-size:0.68rem; color:{_footer_color}; font-family:'Courier New',monospace;">
    Internal Use Only
  </span>
</div>
""", unsafe_allow_html=True)
