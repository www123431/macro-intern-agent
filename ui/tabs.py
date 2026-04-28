"""
UI Layer — Tab rendering functions.
All Streamlit calls live here; engine/ modules stay UI-free.
"""
import json
import logging
import os
import re
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _edit_ratio(original: str, edited: str) -> float:
    """
    Levenshtein-based edit ratio: 0.0 = identical, 1.0 = complete replacement.
    Uses a fast DP implementation; O(m*n) time, O(min(m,n)) space.
    """
    a, b = (original or "").strip(), (edited or "").strip()
    if a == b:
        return 0.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    # Wagner-Fischer DP
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            )
        prev = curr
    return prev[len(b)] / max_len

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import ui.theme as _theme

from engine.memory import (
    BASELINE_HIT_RATE, EXCELLENT, MIN_ACCEPTABLE,
    get_all_today_sector_reports, get_all_today_sector_full,
    get_all_today_audit_records, get_historical_context, get_stats,
    get_today_report, save_decision, verify_pending_decisions,
    supersede_decision, get_decision_by_id, get_last_revision_time,
    get_revision_chains,
    extract_direction,
    SessionFactory, DecisionLog,
    get_pending_decisions_for_monitor,
    set_human_label,
    save_watch_items, get_pending_watch_items, resolve_watch_item,
    expire_overdue_watch_items, parse_watch_items_from_memo,
)
from engine.debate import run_sector_debate, run_quant_coherence_check
from engine.history import get_active_sector_etf as _get_active_sector_etf
SECTOR_ETF = _get_active_sector_etf()  # live universe, not a stale hardcoded copy
from engine.signal import get_quant_gates, compute_composite_scores
from engine.news import NewsPerceiver
from engine.quant import AnalyticsEngine, fetch_raw_data, generate_pdf_report, get_valuation_snapshot, compute_state_vector, compute_quant_metrics
from engine.scanner import AUDIT_TICKERS, MarketScanner
from engine.quant_agent import run_quant_assessment
from engine.agent import build_position_context

# ─────────────────────────────────────────────────────────────────────────────
# Scanner daily cache (survives page refresh, cleared each new calendar day)
# ─────────────────────────────────────────────────────────────────────────────
_SCANNER_CACHE = os.path.join(os.path.dirname(__file__), "..", "scanner_daily_cache.json")


def _load_scanner_cache() -> dict | None:
    """Return today's scanner cache dict or None if missing/stale."""
    try:
        with open(_SCANNER_CACHE, encoding="utf-8") as f:
            data = json.load(f)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return data if data.get("date") == today else None
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def _save_scanner_cache(scan_results: dict, ai_analysis: str, news_ctx: str) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    payload = {"date": today, "scan_results": scan_results,
               "ai_analysis": ai_analysis, "news_ctx": news_ctx}
    with open(_SCANNER_CACHE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _clear_scanner_cache() -> None:
    try:
        os.remove(_SCANNER_CACHE)
    except FileNotFoundError:
        pass
    for key in ("scan_results", "ai_scan_analysis", "ai_scan_news_ctx", "scan_date"):
        st.session_state.pop(key, None)


# ─────────────────────────────────────────────────────────────────────────────
# Model injection (called once from app.py)
# ─────────────────────────────────────────────────────────────────────────────
_model = None


def _infer_macro_regime(vix: float) -> str:
    """Derive a macro regime label from VIX level for learning log tagging."""
    if vix >= 30:
        return "高波动/危机"
    if vix >= 20:
        return "震荡期"
    if vix >= 15:
        return "温和波动"
    return "低波动/牛市"


def set_model(model) -> None:
    global _model
    _model = model


def get_model():
    return _model


def restore_today_from_db() -> None:
    """Called once per session on startup: pull today's reports from DB into session_state."""
    if st.session_state.get("_db_restored"):
        return
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ── Macro report ──────────────────────────────────────────────────────────
    if not st.session_state.get("macro_memo"):
        report = get_today_report("macro", "全球宏观")
        if report:
            st.session_state.macro_memo = report
            st.session_state.macro_analysis_date = today_str

    # ── Sector reports + debate transcripts + XAI ─────────────────────────────
    sector_full = get_all_today_sector_full()
    for rec in sector_full:
        sname = rec["sector_name"]
        date_str = rec["created_at"].strftime("%Y-%m-%d")

        if not st.session_state.get(f"sector_memo_{sname}"):
            st.session_state[f"sector_memo_{sname}"]          = rec["ai_conclusion"]
            st.session_state[f"sector_analysis_date_{sname}"] = date_str
            st.session_state[f"sector_analysis_time_{sname}"] = rec["created_at"].replace(
                tzinfo=timezone.utc
            )

        # Restore debate transcript (only if not already in session)
        if rec["debate_history"] and not st.session_state.get(f"debate_history_{sname}"):
            st.session_state[f"debate_history_{sname}"] = rec["debate_history"]
        if rec["arb_notes"] and not st.session_state.get(f"debate_arb_{sname}"):
            st.session_state[f"debate_arb_{sname}"]     = rec["arb_notes"]
        if rec["blue_output"] and not st.session_state.get(f"blue_output_{sname}"):
            st.session_state[f"blue_output_{sname}"]    = rec["blue_output"]

        # Restore XAI panel data — only if at least one numeric field has a real value
        if not st.session_state.get(f"sector_xai_{sname}"):
            _xai = rec["xai"]
            _has_data = any(
                _xai.get(f) not in (None, 0, "")
                for f in ("overall_confidence", "macro_confidence", "news_confidence", "technical_confidence")
            )
            if _has_data:
                st.session_state[f"sector_xai_{sname}"] = _xai

    if sector_full and not st.session_state.get("latest_sector_memo"):
        st.session_state["latest_sector_memo"] = sector_full[-1]["ai_conclusion"]

    # ── Tab4 audit cache ──────────────────────────────────────────────────────
    audit_cache = st.session_state.setdefault("_audit_cache", {})
    for rec in get_all_today_audit_records():
        key = rec["target_assets"]
        if key and key not in audit_cache:
            audit_cache[key] = rec

    # ── Alpha Scanner daily cache ─────────────────────────────────────────────
    if not st.session_state.get("scan_results"):
        _sc = _load_scanner_cache()
        if _sc:
            st.session_state.scan_results     = _sc["scan_results"]
            st.session_state.ai_scan_analysis = _sc["ai_analysis"]
            st.session_state.ai_scan_news_ctx = _sc["news_ctx"]
            st.session_state.scan_date        = _sc["date"]

    st.session_state["_db_restored"] = True


def _build_watchlist_context() -> str:
    """
    Load unresolved macro watch items and format them as a prior-context block
    for injection into the next macro analysis prompt.
    Returns empty string if there are no pending items.
    """
    import datetime as _dt
    today = _dt.date.today()
    pending = get_pending_watch_items()
    if not pending:
        return ""

    lines = ["【上期监控清单回顾 — 未解决项目】"]
    lines.append("以下是上次宏观分析生成的监控清单，请在本次分析中评估是否已发生并更新判断：")
    for item in pending[:8]:  # cap at 8 to avoid token bloat
        check_str = item["check_by"].strftime("%m/%d") if item["check_by"] else "—"
        expired   = item["check_by"] < today if item["check_by"] else False
        status    = "⏰ 已到期" if expired else f"截止 {check_str}"
        ev        = f"（预期：{item['expected_value']}）" if item["expected_value"] else ""
        lines.append(f"• [{item['category'] or '—'}] {item['item_text']}{ev} {status}")
    lines.append("请在本次分析的§6监控清单中，对上述到期项目给出实际结果对比。")
    return "\n".join(lines)


def _run_macro_analysis(vix_input: float, overwrite: bool = False) -> None:
    """Fetch news, generate macro brief, persist to DB and session_state."""
    import datetime as _dt

    # ── Expire overdue watch items before new analysis ────────────────────────
    n_expired = expire_overdue_watch_items()
    if n_expired:
        st.toast(f"已自动关闭 {n_expired} 条过期监控项", icon="⏰")

    with st.spinner("Aggregating global macro news · Dispatching analyst network..."):
        perceiver = NewsPerceiver(
            av_key=st.secrets.get("AV_KEY", ""),
            gnews_key=st.secrets.get("GNEWS_KEY", ""),
        )
        news_ctx    = perceiver.build_context("全球宏观", "SPY", n=6)
        _macro_regime = _infer_macro_regime(vix_input)   # compute once, reuse below
        st.session_state.macro_news_ctx = news_ctx
        hist_ctx = get_historical_context("macro", macro_regime=_macro_regime, n=5)

        # ── Inject prior watchlist as residual context ─────────────────────────
        watchlist_ctx = _build_watchlist_context()

        # P2-5: FRED economic data (actual values, MoM trend as surprise proxy)
        try:
            from engine.macro_fetcher import get_economic_surprises as _get_econ
            _econ_ctx = _get_econ()
        except Exception:
            _econ_ctx = ""

        parts = [p for p in [_econ_ctx, watchlist_ctx, hist_ctx, news_ctx] if p]
        augmented_ctx = "\n\n".join(parts)

        macro_res = get_ai_analysis("macro", vix_input, augmented_ctx)
        st.session_state.macro_memo = macro_res
        st.session_state.macro_analysis_time = datetime.now(timezone.utc)
        st.session_state.macro_analysis_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        st.session_state.macro_pdf_bytes = None

        if not _is_ai_error(macro_res):
            # ── Parse §6 and save watch items ─────────────────────────────────
            today = _dt.date.today()
            parsed_items = parse_watch_items_from_memo(macro_res)
            n_saved = save_watch_items(
                items=parsed_items,
                analysis_date=today,
                macro_regime=_macro_regime,
                check_days=7,
            )
            if n_saved:
                st.session_state["watchlist_saved_count"] = n_saved

            if overwrite:
                # Refresh path: user-triggered, save directly.
                save_decision(
                    tab_type="macro",
                    ai_conclusion=macro_res,
                    vix_level=vix_input,
                    sector_name="全球宏观",
                    news_summary=news_ctx[:500],
                    overwrite=overwrite,
                    macro_regime=_macro_regime,
                    horizon="季度(3个月)",
                    decision_source="ai_drafted",
                )
            else:
                # Initial run: park draft for human review before saving.
                st.session_state["macro_draft"] = {
                    "tab_type":    "macro",
                    "ai_conclusion": macro_res,
                    "vix_level":   vix_input,
                    "sector_name": "全球宏观",
                    "news_summary": news_ctx[:500],
                    "macro_regime": _macro_regime,
                    "horizon":     "季度(3个月)",
                }


def _run_sector_analysis(
    selected: str, vix_input: float,
    tickers: list, overwrite: bool = False,
    parent_decision_id: int | None = None,
    revision_reason: str = "",
) -> None:
    """Fetch news, generate sector brief, persist to DB and session_state.

    When parent_decision_id is provided the analysis runs in revision mode:
    the original decision context is injected into the prompt so the LLM
    explicitly compares old vs new judgement, then the parent is superseded.
    """
    memo_key = f"sector_memo_{selected}"
    news_key = f"sector_news_{selected}"
    sector_date_key = f"sector_analysis_date_{selected}"

    _spinner_msg = (
        f"修订分析 · {selected} · 对比原始判断与当前市场环境..."
        if parent_decision_id else
        f"Fetching {selected} news · Scanning spillover signals · Generating sector intelligence..."
    )
    with st.spinner(_spinner_msg):
        ticker_for_news = tickers[0] if tickers else selected
        _perceiver = NewsPerceiver(
            av_key=st.secrets.get("AV_KEY", ""),
            gnews_key=st.secrets.get("GNEWS_KEY", ""),
        )
        _regime = _infer_macro_regime(vix_input)

        # P2-17: 三层新闻数据源（Finnhub → GNews → yfinance）+ 时效性加权
        try:
            from engine.news_fetcher import fetch_sector_news, build_weighted_news_summary
            _news_items = fetch_sector_news(selected, ticker_for_news, days=3, max_total=8)
            _enhanced_news = build_weighted_news_summary(_news_items, max_chars=1200)
            news_ctx = _enhanced_news if _news_items else _perceiver.build_context(
                selected, ticker_for_news, n=6, macro_regime=_regime
            )
        except Exception:
            news_ctx = _perceiver.build_context(selected, ticker_for_news, n=6, macro_regime=_regime)

        spillover_ctx = _perceiver.build_spillover_context(selected, macro_regime=_regime)
        if spillover_ctx:
            news_ctx = news_ctx + "\n\n" + spillover_ctx
        st.session_state[news_key] = news_ctx

        hist_ctx      = get_historical_context(
            "sector", sector_name=selected,
            macro_regime=_infer_macro_regime(vix_input), n=5,
        )
        macro_context = st.session_state.get("macro_memo", "")

        # ── Revision context injection ────────────────────────────────────────
        # When revising an existing decision, prepend the original judgement and
        # the trigger reason so the LLM explicitly diffs old vs new.
        if parent_decision_id:
            _parent = get_decision_by_id(parent_decision_id)
            if _parent:
                _rev_prefix = (
                    f"【修订分析 · 注意】本次分析是对以下历史决策的修订，"
                    f"请在结论中明确说明方向是否改变及改变原因。\n"
                    f"原始决策日期：{_parent['created_at']}\n"
                    f"原始方向：{_parent['direction']}\n"
                    f"原始逻辑摘要：{_parent['economic_logic'][:300] or '未记录'}\n"
                    f"触发修订的条件：{revision_reason}\n"
                    f"请对比原始判断，说明：\n"
                    f"① 方向是否需要调整（维持/上调/下调）\n"
                    f"② 核心假设哪里发生了变化\n"
                    f"③ 修订后的配置建议及新的失效条件\n\n"
                )
                hist_ctx = _rev_prefix + hist_ctx

        # ── Valuation snapshot (yfinance) ─────────────────────────────────────
        etf_ticker   = SECTOR_ETF.get(selected, tickers[0] if tickers else "")
        val_ctx      = get_valuation_snapshot(etf_ticker) if etf_ticker else "估值数据暂不可用"

        # ── Quant metrics (Method 2 + 3 integration) ──────────────────────────
        # Pre-compute quantitative metrics before the debate so they can be:
        #   1. Injected into the Blue node prompt (structured engagement)
        #   2. Stored in DecisionLog for future empirical analysis
        #   3. Used in the post-debate Quant Coherence check
        # Always use AUDIT_TICKERS as the canonical ticker source for quant metrics.
        # This guarantees Tab2 and Tab3 (Quant Audit) compute on the same asset set —
        # Tab3's agent uses AUDIT_TICKERS via dynamic_assets fallback; enforcing the
        # same source here makes the alignment explicit rather than implicit.
        _quant_tickers = tuple(AUDIT_TICKERS.get(selected) or tickers or [])
        quant_ctx = compute_quant_metrics(_quant_tickers, vix_input) if _quant_tickers else {}

        # ── Quant gate (independent decision authority) ───────────────────────
        # Compute per-sector gate BEFORE debate. Gate rules use only market data
        # (TSMOM, CSMOM, composite score, regime) — Goodhart-safe.
        # The gate is injected into both Blue and Arbitration prompts, and
        # enforced post-hoc if the LLM violates it.
        _decision_date_for_gate = datetime.now(timezone.utc).date()
        _regime_for_gate = _infer_macro_regime(vix_input)  # "风险偏好/牛市" → map to risk-on/off
        _regime_map = {
            "低波动/牛市": "risk-on", "温和波动": "risk-on",
            "震荡期": "transition", "高波动/危机": "risk-off",
        }
        _regime_label = _regime_map.get(_regime_for_gate, "transition")
        try:
            _all_gates = get_quant_gates(
                as_of=_decision_date_for_gate,
                regime_label=_regime_label,
            )
            _sector_gate = dict(_all_gates.get(selected, {}))
            # Enrich gate with ann_vol from quant_ctx so the debate prompt shows it
            if quant_ctx and "a_vol" in quant_ctx:
                _sector_gate["ann_vol"] = quant_ctx["a_vol"]
        except Exception as _gate_err:
            logger.warning("Quant gate computation failed for %s: %s", selected, _gate_err)
            _sector_gate = {}

        # ── Multi-round debate (Blue → Red → Blue defense → Arbitration) ──────
        debate = run_sector_debate(
            model              = _model,
            sector_name        = selected,
            vix                = vix_input,
            macro_context      = macro_context,
            news_context       = news_ctx,
            historical_context = hist_ctx,
            valuation_context  = val_ctx,
            quant_context      = quant_ctx or None,
            quant_gate         = _sector_gate or None,
        )

        sector_res = debate["final_output"]   # arbitrated final text
        xai        = debate["final_xai"]      # arbitrated XAI

        # ── Method 3: Quant Coherence check ───────────────────────────────────
        # Deterministic, no LLM call. Flags are stored with the draft for display
        # in the review expander so the human reviewer sees them before confirming.
        # Pass the externally-extracted direction so QC-1 is Goodhart-safe.
        _qc_direction = extract_direction(sector_res)
        qc_flags = run_quant_coherence_check(xai, quant_ctx, direction=_qc_direction)

        # Replace raw quota/rate-limit error strings from debate nodes with a
        # styled card — keeps the UI consistent with Tab1/Tab5 error handling.
        if _debate_output_has_quota_error(sector_res) or not sector_res.strip():
            sector_res = _debate_quota_error_html(selected)

        # Store debate transcript for display
        st.session_state[f"debate_history_{selected}"] = debate["debate_history"]
        st.session_state[f"debate_arb_{selected}"]     = debate["arbitration_notes"]
        st.session_state[f"blue_output_{selected}"]    = debate["blue_output"]

        st.session_state[memo_key] = sector_res
        st.session_state["latest_sector_memo"] = sector_res
        st.session_state[f"sector_xai_{selected}"]          = xai   # for XAI panel
        st.session_state[f"sector_analysis_time_{selected}"] = datetime.now(timezone.utc)
        st.session_state[sector_date_key] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        st.session_state[f"sector_pdf_bytes_{selected}"] = None

        if not _is_ai_error(sector_res):
            # Scale weight_adjustment_pct by confidence (high-conviction calls get
            # bolder sizing; low-conviction calls are attenuated automatically).
            _raw_adj  = debate.get("weight_adjustment_pct") or 0.0
            _conf_val = xai.get("overall_confidence") or 50
            if _conf_val < 40:
                _conf_mult = 0.5
            elif _conf_val > 70:
                _conf_mult = 1.3
            else:
                _conf_mult = 1.0
            _scaled_adj = _raw_adj * _conf_mult

            # Build signal_attribution from debate XAI fields.
            _signal_attribution = {
                "macro":     xai.get("macro_confidence"),
                "news":      xai.get("news_confidence"),
                "technical": xai.get("technical_confidence"),
                "drivers":   xai.get("signal_drivers", ""),
            }
            _decision_date = datetime.now(timezone.utc).date()
            _state_vec = compute_state_vector(ticker_for_news, _decision_date)

            if overwrite or parent_decision_id:
                # Revision path: already user-confirmed via a dedicated button — save directly.
                _saved_id = save_decision(
                    tab_type="sector",
                    ai_conclusion=sector_res,
                    vix_level=vix_input,
                    sector_name=selected,
                    ticker=ticker_for_news,
                    news_summary=news_ctx[:500],
                    overwrite=overwrite,
                    macro_regime=_infer_macro_regime(vix_input),
                    horizon=xai.get("horizon", "季度(3个月)"),
                    confidence_score=xai.get("overall_confidence"),
                    macro_confidence=xai.get("macro_confidence"),
                    news_confidence=xai.get("news_confidence"),
                    technical_confidence=xai.get("technical_confidence"),
                    signal_attribution=_signal_attribution,
                    invalidation_conditions=xai.get("invalidation_conditions", ""),
                    decision_date=_decision_date,
                    debate_transcript={
                        "history":     debate["debate_history"],
                        "arbitration": debate["arbitration_notes"],
                        "blue_output": debate["blue_output"],
                        "state_vector": _state_vec,
                    },
                    parent_decision_id=parent_decision_id,
                    revision_reason=revision_reason,
                    decision_source="ai_drafted",
                    quant_p_noise=quant_ctx.get("p_noise") if quant_ctx else None,
                    quant_val_r2=quant_ctx.get("val_r2") if quant_ctx else None,
                    quant_test_r2=quant_ctx.get("test_r2") if quant_ctx else None,
                    quant_active=quant_ctx.get("active") if quant_ctx else None,
                    weight_adjustment_pct=_scaled_adj,
                    adjustment_reason=debate.get("final_data", {}).get("adjustment_reason"),
                    signal_invalidation_risk=xai.get("signal_invalidation_risk"),
                )
                if parent_decision_id:
                    supersede_decision(parent_decision_id, revision_reason)
                # P1-C: guide user to Watchlist after save
                if _saved_id:
                    st.info(
                        f"✅ 决策已入库（ID={_saved_id}）。"
                        f" 前往 **Trading Desk → Watchlist** 查看仓位状态，"
                        f"或在 **Decision Monitor** 追踪验证进度。",
                        icon="📋",
                    )
            else:
                # Initial generation: park draft in session_state for human review.
                # save_decision is NOT called here — the "确认入库" button in the tab
                # will call it once the user reviews and optionally edits key fields.
                st.session_state[f"sector_draft_{selected}"] = {
                    "tab_type":              "sector",
                    "ai_conclusion":         sector_res,
                    "vix_level":             vix_input,
                    "sector_name":           selected,
                    "ticker":                ticker_for_news,
                    "news_summary":          news_ctx[:500],
                    "macro_regime":          _infer_macro_regime(vix_input),
                    "horizon":               xai.get("horizon", "季度(3个月)"),
                    "confidence_score":      xai.get("overall_confidence"),
                    "macro_confidence":      xai.get("macro_confidence"),
                    "news_confidence":       xai.get("news_confidence"),
                    "technical_confidence":  xai.get("technical_confidence"),
                    "signal_attribution":    _signal_attribution,
                    "invalidation_conditions": xai.get("invalidation_conditions", ""),
                    "decision_date":         _decision_date,
                    "debate_transcript":     {
                        "history":     debate["debate_history"],
                        "arbitration": debate["arbitration_notes"],
                        "blue_output": debate["blue_output"],
                        "state_vector": _state_vec,
                    },
                    # Quant integration fields
                    "quant_p_noise":          quant_ctx.get("p_noise") if quant_ctx else None,
                    "quant_val_r2":           quant_ctx.get("val_r2") if quant_ctx else None,
                    "quant_test_r2":          quant_ctx.get("test_r2") if quant_ctx else None,
                    "quant_active":           quant_ctx.get("active") if quant_ctx else None,
                    "weight_adjustment_pct":  _scaled_adj,
                    "adjustment_reason":      debate.get("final_data", {}).get("adjustment_reason"),
                    "signal_invalidation_risk": xai.get("signal_invalidation_risk"),
                    "qc_flags":               qc_flags,
                }


# ─────────────────────────────────────────────────────────────────────────────
# Shared UI primitives
# ─────────────────────────────────────────────────────────────────────────────

def _section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def _extract_sentiment_score(news_ctx: str) -> float:
    """Parse AV aggregate sentiment score from build_context() output string."""
    m = re.search(r'avg score:\s*([+-]?\d+\.\d+)', news_ctx)
    return float(m.group(1)) if m else 0.0


def _render_divergence_warning(sentiment_score: float, context: dict) -> None:
    """
    Show a concise divergence warning when news sentiment contradicts quant risk signals.

    context keys (all optional):
      vix          float  — current VIX
      confidence   int    — XAI overall_confidence (Tab3)
      d_var        float  — daily VaR (Tab4 audit)
      sharpe       float  — Sharpe ratio
    """
    vix        = context.get("vix", 0)
    confidence = context.get("confidence")
    d_var      = context.get("d_var")
    sharpe     = context.get("sharpe")
    mom_1m     = context.get("mom_1m")
    mom_6m     = context.get("mom_6m")
    direction  = context.get("direction", "")

    # (label, metric_summary)
    signals: list[tuple[str, str]] = []

    # ── Sentiment vs fundamentals ─────────────────────────────────────────────
    if sentiment_score >= 0.20 and confidence is not None and confidence < 55:
        signals.append((
            "叙事层 / 分析层背离",
            f"新闻情绪 {sentiment_score:+.2f} · AI置信度 {confidence}/100 — 多因子信号未收敛，情绪可能超前于基本面",
        ))
    if sentiment_score >= 0.20 and d_var is not None and d_var > 0.025:
        signals.append((
            "情绪层 / 风险层背离",
            f"新闻情绪 {sentiment_score:+.2f} · 日度VaR {d_var:.2%} — 乐观叙事下尾部风险已超机构审慎阈值",
        ))
    if sentiment_score >= 0.20 and sharpe is not None and sharpe < 0.4:
        signals.append((
            "情绪层 / 收益质量背离",
            f"新闻情绪 {sentiment_score:+.2f} · Sharpe {sharpe:.2f} — 风险调整收益不支持当前情绪溢价",
        ))
    if sentiment_score >= 0.35 and vix > 25:
        signals.append((
            "极端情绪 / 市场恐慌背离",
            f"新闻情绪 {sentiment_score:+.2f} · VIX {vix:.0f} — 期权市场在为尾部风险定价，与媒体叙事背道而驰",
        ))

    # ── Multi-period momentum divergence ─────────────────────────────────────
    if mom_1m is not None and mom_6m is not None:
        if mom_1m < -0.02 and mom_6m > 0.03:
            signals.append((
                "短期 / 中期动量背离",
                f"1m动量 {mom_1m:+.1%} · 6m动量 {mom_6m:+.1%} — 短期承压但中期趋势结构完整，可能为阶段性回调",
            ))
        elif mom_1m > 0.03 and mom_6m < -0.03:
            signals.append((
                "反弹陷阱预警",
                f"1m动量 {mom_1m:+.1%} · 6m动量 {mom_6m:+.1%} — 短期技术性反弹叠加中期下行趋势，警惕 Dead Cat Bounce",
            ))

    # ── AI direction vs news sentiment ────────────────────────────────────────
    if direction in ("低配", "拦截") and sentiment_score >= 0.35:
        signals.append((
            "配置方向 / 市场情绪背离",
            f"AI方向: {direction} · 新闻情绪 {sentiment_score:+.2f} — 模型建议减仓/拦截，但媒体情绪极度乐观，需甄别信息质量",
        ))

    if not signals:
        return

    rows = "".join(
        f'<div style="display:flex; gap:0.8rem; padding:0.45rem 0;'
        f'border-bottom:1px solid var(--danger-bd);">'
        f'<span style="font-size:0.82rem; font-weight:700; color:var(--danger);'
        f'white-space:nowrap; padding-top:0.05rem;">[{i}] {label}</span>'
        f'<span style="font-size:0.88rem; color:var(--text);">{summary}</span>'
        f'</div>'
        for i, (label, summary) in enumerate(signals, 1)
    )

    st.markdown(
        f"""
        <div style="background:var(--danger-lt); border:2px solid var(--danger);
                    border-radius:3px; padding:0.9rem 1.1rem; margin:0.8rem 0;">
          <div style="font-size:0.9rem; font-weight:800; color:var(--danger);
                      letter-spacing:0.06em; margin-bottom:0.6rem;">
            ⚠ DIVERGENCE WARNING · 情绪–逻辑背离　({len(signals)} signal{'s' if len(signals)>1 else ''})
          </div>
          {rows}
          <div style="font-size:0.8rem; color:var(--danger); margin-top:0.55rem; opacity:0.85;">
            建议在最终配置前独立核查基本面锚点，并评估是否需要通过降仓或对冲工具管理尾部风险。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _info_panel(label: str, items: list[str]) -> None:
    """Small metadata panel used in Tab 1 & 3."""
    rows = "".join(
        f'<div style="padding:0.3rem 0; border-bottom:1px solid var(--border); '
        f'font-size:0.95rem; color:var(--text);">{item}</div>'
        for item in items
    )
    st.markdown(f"""
    <div style="background:var(--card); border:1px solid var(--border); border-radius:3px;
                padding:1rem 1.1rem;">
      <div style="font-size:0.82rem; text-transform:uppercase; letter-spacing:0.1em;
                  color:var(--muted); font-weight:700; margin-bottom:0.6rem;">{label}</div>
      {rows}
    </div>""", unsafe_allow_html=True)


def render_tv_chart(symbol: str, title: str, theme: str = "light") -> None:
    code = f"""
    <div style="height:340px;">
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script>new TradingView.MediumWidget({{
        "symbols": [["{title}", "{symbol}|12M"]],
        "width": "100%", "height": 340,
        "locale": "zh_CN", "colorTheme": "{theme}"
    }});</script></div>"""
    components.html(code, height=345)


def _render_memo_sections(text: str, accent: str = "#1A56DB") -> None:
    """
    Parse AI memo text that uses ### headers and render each section as a
    styled block. Body text is rendered via st.markdown so **bold** and
    bullet lists display correctly rather than showing raw symbols.
    """
    import re

    # Strip HTML tags (e.g. error messages that slipped through)
    text = re.sub(r"<[^>]+>", "", text).strip()
    if not text:
        return

    def _clean(t: str) -> str:
        # Remove stray lone/double # markers not part of ### section headers
        t = re.sub(r"(?m)^#{1,2}(?!#)\s*", "", t)
        return t.strip()

    # Split on ### headers
    parts = re.split(r"(?m)^###\s*", text)

    if len(parts) <= 1:
        # No ### structure — render via st.markdown so formatting works
        st.markdown(
            f'<div style="border-left:3px solid {accent}; padding-left:0.8rem; '
            f'margin-bottom:0.2rem;"></div>',
            unsafe_allow_html=True,
        )
        st.markdown(_clean(text))
        return

    # Preamble before the first ###
    preamble = _clean(parts[0])
    if preamble:
        st.markdown(preamble)

    for section in parts[1:]:
        lines = section.strip().split("\n", 1)
        # Strip any leading # the AI may have echoed back into the header line
        header = re.sub(r"^[#\s]+", "", lines[0]).rstrip("#").strip()
        body = _clean(lines[1]) if len(lines) > 1 else ""

        # Section header — styled HTML label, larger than body text
        st.markdown(
            f'<div style="font-size:1.1rem; font-weight:700; '
            f'color:{accent}; margin:1.1rem 0 0.35rem; '
            f'padding-bottom:0.25rem; border-bottom:1.5px solid var(--border);">'
            f'{header}</div>',
            unsafe_allow_html=True,
        )
        # Body — via st.markdown so **bold**, bullet lists, etc. render properly
        if body:
            st.markdown(body)


def _render_news_feed(news_text: str) -> None:
    """Render build_context() output with visual hierarchy and badge coloring."""
    _BADGE_STYLES = {
        "🔴 LIVE":       "border-left:3px solid #EF4444; background:rgba(239,68,68,0.08);",
        "🟡 ACTIVE":     "border-left:3px solid #F59E0B; background:rgba(245,158,11,0.08);",
        "🟠 COOLING":    "border-left:3px solid #F97316; background:rgba(249,115,22,0.08);",
        "📁 BACKGROUND": "border-left:3px solid var(--border); background:var(--card); opacity:0.75;",
    }
    for line in news_text.strip().split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("【"):
            st.markdown(
                f'<div style="font-size:0.82rem; font-weight:700; text-transform:uppercase; '
                f'letter-spacing:0.09em; color:var(--muted); margin:0.6rem 0 0.3rem;">{stripped}</div>',
                unsafe_allow_html=True,
            )
        elif stripped.startswith("Alpha Vantage"):
            st.markdown(
                f'<div style="font-size:0.92rem; color:var(--accent); background:var(--accent-lt); '
                f'border-radius:3px; padding:0.3rem 0.6rem; margin-bottom:0.4rem;">{stripped}</div>',
                unsafe_allow_html=True,
            )
        elif "活跃信号区" in stripped:
            st.markdown(
                f'<div style="font-size:0.88rem; font-weight:700; color:var(--success); '
                f'text-transform:uppercase; letter-spacing:0.07em; '
                f'margin:0.8rem 0 0.3rem; border-top:1px solid var(--success-bd); padding-top:0.4rem;">'
                f'{stripped}</div>',
                unsafe_allow_html=True,
            )
        elif "历史背景区" in stripped:
            st.markdown(
                f'<div style="font-size:0.88rem; font-weight:700; color:var(--muted); '
                f'text-transform:uppercase; letter-spacing:0.07em; '
                f'margin:0.8rem 0 0.3rem; border-top:1px solid var(--border); padding-top:0.4rem;">'
                f'{stripped}</div>',
                unsafe_allow_html=True,
            )
        else:
            badge_style = ""
            for badge, style in _BADGE_STYLES.items():
                if badge in stripped:
                    badge_style = style
                    break
            st.markdown(
                f'<div style="font-size:0.95rem; color:var(--text); padding:0.35rem 0.6rem; '
                f'margin-bottom:0.2rem; border-radius:3px; {badge_style}">{stripped}</div>',
                unsafe_allow_html=True,
            )


def _generate_pdf_content(report_type: str, brief: str, news_ctx: str, extra: dict) -> str:
    """
    Generate extended, data-rich content specifically for PDF export.
    The web brief is passed as context so the PDF deepens rather than repeats it.
    """
    vix = extra.get("vix", "N/A")
    sector = extra.get("sector", "")
    macro_ctx = extra.get("macro_ctx", "")

    if report_type == "macro":
        prompt = (
            f"你是一名顶级宏观对冲基金首席分析师，正在撰写机构级《全球宏观策略深度报告》。\n"
            f"当前 VIX：{vix}\n\n"
            f"今日宏观简报（作为背景参考）：\n{brief}\n\n"
            f"近期宏观新闻情报（近48小时）：\n{news_ctx}\n\n"
            "请撰写一份完整的深度研究报告，结构如下：\n\n"
            "### 执行摘要\n"
            "[核心结论 2-3 句，直接点明当前宏观政策信号与市场风险判断]\n\n"
            "### 1. 央行政策信号精读\n"
            "[美联储/ECB/BOJ 最新立场，CME FedWatch 隐含利率路径，与上季度的偏移幅度，"
            "对国债收益率曲线的具体影响]\n\n"
            "### 2. 风险资产定价环境\n"
            f"[VIX {vix} 处于历史什么分位，与 2020/2022 波动率峰值的对比，"
            "股债相关性是否已由负转正，大宗商品 CRB 指数信号]\n\n"
            "### 3. 关键新闻事件逐条深度解读\n"
            "[对每条新闻：① 事件背景与核心信息 ② 对哪类资产最直接冲击 "
            "③ 传导机制与时间窗口 ④ 专业分析员独到判断，必须引用新闻原标题]\n\n"
            "### 4. 宏观风险情景矩阵\n"
            "[列出 3 个情景（基准/悲观/乐观），每个包含：触发条件、概率估计、"
            "对股债商品的具体影响路径]\n\n"
            "### 5. 跨资产配置信号\n"
            "[股票/债券/大宗商品/黄金/外汇的相对强弱排序，基于宏观周期定位的战术权重调整建议]\n\n"
            "### 6. 未来 5 个交易日监控清单\n"
            "[具体数据发布（时间+预期值+前值）、关键价位（支撑/阻力）、需重点观察的市场信号]\n\n"
            "写作要求：引用具体新闻标题与数据点、每节不少于 4 句、"
            "从专业分析员视角给出独到判断、严格机构语气、严禁情绪化或投资劝说表达。"
        )
    else:
        intro = (
            f"你是一名顶级行业研究分析师，正在撰写机构级《{sector}板块深度策略报告》。\n"
            f"当前 VIX：{vix}\n\n"
            f"今日板块简报（作为背景参考）：\n{brief}\n\n"
            f"板块近期新闻情报（近48小时）：\n{news_ctx}\n\n"
        )
        if macro_ctx:
            intro += f"宏观背景：\n{macro_ctx}\n\n"
        prompt = intro + (
            "请撰写一份完整的板块深度研究报告，结构如下：\n\n"
            "### 执行摘要\n"
            "[配置方向一句话结论 + 核心逻辑]\n\n"
            f"### 1. {sector} 板块结构与周期定位\n"
            "[当前板块所处行业周期阶段（扩张/顶部/收缩/底部），与大盘的 beta 特征，"
            "行业集中度与主力资金行为]\n\n"
            "### 2. 宏观-板块联动分析\n"
            f"[当前宏观环境（VIX {vix}、利率、通胀）对 {sector} 的具体传导机制，"
            "历史上相似宏观环境下该板块的表现数据]\n\n"
            "### 3. 近期新闻催化剂逐条深度解读\n"
            "[对每条新闻：① 事件核心信息 ② 对板块内龙头标的的直接影响 "
            "③ 利多/利空信号强度评级（强/中/弱）④ 分析员独到判断，必须引用新闻原标题]\n\n"
            "### 4. 量化风险敞口评估\n"
            f"[结合 VIX {vix}，估算板块潜在回撤区间，列出 2-3 个关键下行风险因子，"
            "给出每个风险因子的当前信号状态（已触发/警戒/未激活）]\n\n"
            "### 5. 相对价值与竞争板块比较\n"
            "[与 2-3 个竞争板块的风险调整收益对比，说明当前为何优选/回避该板块]\n\n"
            "### 6. 战术配置建议\n"
            "[超配/标配/低配结论，目标权重区间，建仓/减仓条件，止损信号]\n\n"
            "写作要求：引用具体新闻标题、每节不少于 4 句、给出专业分析员独到判断、"
            "严格机构语气、禁止情绪化表达。"
        )
    try:
        return _model.generate_content(prompt).text
    except Exception:
        return brief  # fallback to web brief on error


def _is_ai_error(text: str) -> bool:
    """Returns True if the text is an _ai_error HTML block, not real analysis."""
    return text.lstrip().startswith("<p")


def _ai_error(e: Exception) -> str:
    """Unified AI error message — no markdown asterisks, consistent format."""
    msg = str(e)
    if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
        return (
            '<p style="font-size:1.15rem; margin:0 0 0.6rem;">⏳ 首席分析师正在"闭关"</p>'
            '<p style="margin:0 0 0.4rem; color:var(--text);">当前状态：AI 研报额度已耗尽（API Quota Exceeded）。</p>'
            '<p style="margin:0 0 0.4rem; color:var(--muted); font-size:1rem;">'
            "建议操作：稍后 1 分钟后重新点击分析；量化指标数据仍实时有效，可直接参考。"
            "</p>"
            '<p style="margin:0; color:var(--muted); font-size:1rem;">'
            '如需更新配额，请前往 <a href="https://aistudio.google.com/api-keys" '
            'target="_blank" style="color:var(--accent);">aistudio.google.com/api-keys</a>'
            "</p>"
        )
    return (
        '<p style="font-size:1.15rem; margin:0 0 0.6rem;">⚠ AI 投研引擎暂时离线</p>'
        f'<p style="margin:0; color:var(--muted); font-size:1rem;">错误信息：{msg[:80]}</p>'
    )


_DEBATE_QUOTA_MARKERS = ("生成失败:", "429", "RESOURCE_EXHAUSTED", "quota", "Quota")
_AUDIT_ERROR_MARKERS  = ("生成失败:", "⛔", "⚠️ AI 分析引擎暂时离线",
                         "所有 Gemini API Key 已耗尽", "超过最大重试次数")

_KEY_MGR_LINK = (
    '<a href="/Key_Manager" target="_self" style="color:var(--accent);">Key Pool Manager</a>'
)


def _debate_output_has_quota_error(text: str) -> bool:
    return any(m in text for m in _DEBATE_QUOTA_MARKERS)


def _audit_output_has_error(text: str) -> bool:
    return any(m in text for m in _AUDIT_ERROR_MARKERS)


def _debate_quota_error_html(sector: str = "") -> str:
    label = f"【{sector}】" if sector else ""
    return (
        f'<p style="font-size:1.15rem; margin:0 0 0.6rem;">⛔ {label}所有 Key 额度耗尽，辩论中止</p>'
        '<p style="margin:0 0 0.4rem; color:var(--text);">'
        "所有 Gemini API Key 的今日配额已全部耗尽，分析流程无法继续。</p>"
        '<p style="margin:0 0 0.4rem; color:var(--muted); font-size:1rem;">'
        "历史量化数据仍有效，可先切换至其他板块或等待明天配额重置。</p>"
        f'<p style="margin:0; color:var(--muted); font-size:1rem;">'
        f"如需立即恢复，请前往 {_KEY_MGR_LINK} 添加新 Key。</p>"
    )


def _audit_quota_error_html(sector: str = "") -> str:
    label = f"【{sector}】" if sector else ""
    return (
        f'<p style="font-size:1.15rem; margin:0 0 0.6rem;">⛔ {label}所有 Key 额度耗尽，审计中止</p>'
        '<p style="margin:0 0 0.4rem; color:var(--text);">'
        "所有 Gemini API Key 的今日配额已全部耗尽，AI 解读无法生成。</p>"
        '<p style="margin:0 0 0.4rem; color:var(--muted); font-size:1rem;">'
        "量化指标（VaR、Sharpe、动量）仍正常显示，AI 解读部分将在额度重置后补全。</p>"
        f'<p style="margin:0; color:var(--muted); font-size:1rem;">'
        f"如需立即恢复，请前往 {_KEY_MGR_LINK} 添加新 Key。</p>"
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _get_ai_analysis_cached(prompt_type: str, vix_val: float, news_context: str = "") -> str:
    """Raises on error so errors are never cached."""
    news_section = f"\n\n【宏观新闻情报（近48小时）】\n{news_context}" if news_context else ""
    prompts = {
        "macro": (
            f"当前市场 VIX 风险指数为 {vix_val}。你是一名全球宏观策略首席分析师，"
            f"负责向机构投资委员会汇报每日宏观态势。{news_section}\n\n"
            "请基于以上新闻情报，按以下框架撰写【每日宏观简报】：\n\n"
            "### 1. 宏观政策信号\n"
            "[美联储/主要央行最新动向，利率路径预期，政策立场变化]\n\n"
            "### 2. 风险资产定价环境\n"
            "[当前 VIX 水平解读，股债商品的相对强弱，市场情绪判断]\n\n"
            "### 3. 地缘与宏观事件风险\n"
            "[近期新闻中需重点关注的宏观尾部风险，明确引用新闻标题]\n\n"
            "### 4. 本周关键信号\n"
            "[未来 5 个交易日需重点监控的指标或事件]\n\n"
            "### 今日信号摘要\n"
            "请在报告最后输出以下固定格式（不可省略）：\n"
            "⚡ 即时扰动 (24小时内): [列举近期具体事件，如：原油跳升、科技股承压]\n"
            "📅 近期催化剂 (1-3个月): [列举中期关键驱动，如：FOMC议息、财报季、政策窗口]\n"
            "🏗 长期结构逻辑 (1年以上): [列举结构性趋势，如：AI算力需求持续扩张、美债供需失衡]\n"
            "→ 综合判断: [一句话——当前哪个层面主导市场，短期扰动是否动摇长期逻辑]\n\n"
            "写作要求：机构级语气、每节 2-3 句、严禁投资劝说性表达、总字数控制在 300 字以内。"
        ),
    }
    return _model.generate_content(prompts[prompt_type]).text


def get_ai_analysis(prompt_type: str, vix_val: float, news_context: str = "") -> str:
    try:
        return _get_ai_analysis_cached(prompt_type, vix_val, news_context)
    except Exception as e:
        return _ai_error(e)


@st.cache_data(ttl=3600, show_spinner=False)
def _generate_sector_analysis_cached(
    sector_name: str, vix_val: float,
    macro_context: str = "", news_context: str = "",
) -> str:
    """Raises on error so errors are never cached."""
    macro_section = (
        f"\n【宏观背景（来自宏观分析师）】\n{macro_context}\n"
        if macro_context else ""
    )
    news_section = (
        f"\n【近期板块新闻（近48小时）】\n{news_context}\n"
        if news_context else ""
    )
    prompt = (
        f"你是一名机构级板块研究分析师，正在向投资委员会汇报【{sector_name}】板块。\n"
        f"当前市场 VIX 指数为 {vix_val}。"
        f"{macro_section}{news_section}\n"
        "请按以下框架撰写专业板块分析报告：\n\n"
        "### 1. 板块驱动逻辑\n"
        f"[结合当前宏观环境，分析【{sector_name}】板块的核心催化剂与主要驱动因子]\n\n"
        "### 2. 新闻事件解读\n"
        "[逐一分析上方近期新闻的潜在影响，必须明确引用具体新闻标题，"
        "说明其对板块的短期利多/利空含义]\n\n"
        "### 3. 风险敞口评估\n"
        f"[结合 VIX {vix_val} 环境与宏观背景，量化评估该板块面临的主要下行风险]\n\n"
        "### 4. 战术配置方向\n"
        "[基于综合分析，给出中性、客观的配置方向（超配/标配/低配）与一句话理由]\n\n"
        "### 今日信号摘要\n"
        "请在报告最后输出以下固定格式（不可省略）：\n"
        "⚡ 即时扰动 (24小时内): [列举近期具体事件，如板块内龙头公司财报、行业监管动态]\n"
        "📅 近期催化剂 (1-3个月): [列举中期关键驱动，如政策决议、行业数据发布、产业链变化]\n"
        f"🏗 长期结构逻辑 (1年以上): [列举对【{sector_name}】的结构性趋势判断]\n"
        f"→ 综合判断: [一句话——当前哪个层面主导对【{sector_name}】的配置决策，短期扰动是否改变长期逻辑]\n\n"
        "### [XAI_ATTRIBUTION]\n"
        "请在报告最末输出以下机器可读块（格式严格，不可更改字段名）：\n"
        "overall_confidence: [0-100，综合置信度]\n"
        "macro_confidence: [0-100，宏观信号对本次结论的支撑程度]\n"
        "news_confidence: [0-100，新闻信号对本次结论的支撑程度]\n"
        "technical_confidence: [0-100，动量/量价信号对本次结论的支撑程度]\n"
        "signal_drivers: [最多3个关键驱动因素，格式: 因素描述(信号类型,权重高/中/低)，用·分隔]\n"
        "invalidation_conditions: [1-2个明确条件，满足时本建议自动失效，如：CPI连续两月回落/VIX突破30]\n"
        "horizon: [二选一 — 季度(3个月,基本面/政策传导,1个财报季) / 半年(6个月,结构性趋势,2个财报季)]\n"
        "  季度(3个月) = 基本面/政策传导逻辑，90天内趋势方向确认（默认选项，三重障碍时间障碍=90天）\n"
        "  半年(6个月) = 结构性趋势，180天内兑现，适用于产业周期、估值修复（三重障碍时间障碍=180天）\n"
        "  注意：本系统基于低频信息源（宏观新闻+量化月度指标），不具备预测短期价格的信息优势，禁止输出短期\n"
        "### [/XAI_ATTRIBUTION]\n\n"
        "写作要求：机构级语气、逻辑严谨、每节 3-4 句、严禁情绪化或劝说性表达。"
    )
    return _model.generate_content(prompt).text


def _render_xai_panel(xai: dict, sector: str = "") -> None:
    """Render the XAI attribution block as a structured UI panel."""
    if not xai:
        return

    overall   = xai.get("overall_confidence")
    macro_c   = xai.get("macro_confidence")
    news_c    = xai.get("news_confidence")
    tech_c    = xai.get("technical_confidence")
    drivers   = xai.get("signal_drivers", "")
    inv_cond  = xai.get("invalidation_conditions", "")
    horizon   = xai.get("horizon", "—")

    # Skip panel if no meaningful data
    if not any([overall, macro_c, news_c, tech_c]):
        return

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    _section_label("Signal Attribution  ·  XAI Panel")

    with st.container(border=True):
        # ── Row 1: confidence scores ──────────────────────────────────────────
        def _conf_color(v):
            if v is None: return "var(--muted)"
            return "var(--success)" if v >= 70 else ("var(--warn)" if v >= 50 else "var(--danger)")

        def _conf_bar(v):
            if v is None: return ""
            color = _conf_color(v)
            return (
                f'<div style="height:4px; border-radius:2px; background:var(--border); margin-top:4px;">'
                f'<div style="width:{v}%; height:100%; border-radius:2px; background:{color};"></div>'
                f'</div>'
            )

        c1, c2, c3, c4, c5 = st.columns(5)
        for col, label, val in [
            (c1, "综合置信度",  overall),
            (c2, "宏观信号",   macro_c),
            (c3, "新闻信号",   news_c),
            (c4, "技术/动量",  tech_c),
        ]:
            display = f"{val}/100" if val is not None else "—"
            color   = _conf_color(val)
            col.markdown(
                f'<div style="text-align:center; padding:0.5rem 0;">'
                f'<div style="font-size:0.78rem; text-transform:uppercase; '
                f'letter-spacing:0.07em; color:var(--muted); font-weight:600; margin-bottom:0.25rem;">'
                f'{label}</div>'
                f'<div style="font-size:1.35rem; font-weight:800; color:{color};">{display}</div>'
                f'{_conf_bar(val)}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Horizon badge in last column
        _hz_color = {"季度(3个月)": "var(--accent)", "半年(6个月)": "var(--success)"}
        hz_bg = _hz_color.get(horizon, "var(--muted)")
        c5.markdown(
            f'<div style="text-align:center; padding:0.5rem 0;">'
            f'<div style="font-size:0.78rem; text-transform:uppercase; '
            f'letter-spacing:0.07em; color:var(--muted); font-weight:600; margin-bottom:0.35rem;">'
            f'持仓周期</div>'
            f'<span style="background:{hz_bg}; color:white; font-size:0.92rem; '
            f'font-weight:700; padding:0.3rem 0.7rem; border-radius:3px; '
            f'letter-spacing:0.04em;">{horizon}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Row 2: signal drivers ─────────────────────────────────────────────
        if drivers:
            st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:0.8rem; text-transform:uppercase; '
                'letter-spacing:0.07em; color:var(--muted); font-weight:600; '
                'margin-bottom:0.4rem;">核心驱动因素</div>',
                unsafe_allow_html=True,
            )
            _weight_colors = {"高": "var(--danger)", "中": "var(--warn)", "低": "var(--success)"}
            driver_parts = [d.strip() for d in drivers.split("·") if d.strip()]
            chips = []
            for dp in driver_parts:
                import re as _re
                m = _re.match(r"^(.+?)\((.+?),\s*(高|中|低)\)$", dp)
                if m:
                    desc, sig_type, weight = m.group(1), m.group(2), m.group(3)
                    wc = _weight_colors.get(weight, "var(--muted)")
                    chips.append(
                        f'<span style="display:inline-flex; align-items:center; gap:0.3rem; '
                        f'background:var(--card); border:1px solid var(--border); border-radius:3px; '
                        f'padding:0.25rem 0.6rem; margin:0.15rem; font-size:0.88rem; color:var(--text);">'
                        f'{desc}'
                        f'<span style="font-size:0.78rem; color:var(--muted);">· {sig_type}</span>'
                        f'<span style="font-size:0.78rem; font-weight:700; color:{wc};">· {weight}</span>'
                        f'</span>'
                    )
                else:
                    chips.append(
                        f'<span style="background:var(--card); border:1px solid var(--border); '
                        f'border-radius:3px; padding:0.25rem 0.6rem; margin:0.15rem; '
                        f'font-size:0.88rem; color:var(--text);">{dp}</span>'
                    )
            st.markdown(
                '<div style="display:flex; flex-wrap:wrap; gap:0.2rem;">'
                + "".join(chips) + "</div>",
                unsafe_allow_html=True,
            )

        # ── Row 3: invalidation conditions ───────────────────────────────────
        if inv_cond:
            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:0.8rem; text-transform:uppercase; '
                'letter-spacing:0.07em; color:var(--muted); font-weight:600; '
                'margin-bottom:0.3rem;">失效条件 · 触发时建议重新评估</div>',
                unsafe_allow_html=True,
            )
            import re as _re
            conditions = _re.split(r"\s*[1-9]\.\s*", inv_cond)
            conditions = [c.strip() for c in conditions if c.strip()]
            for idx, cond in enumerate(conditions, 1):
                st.markdown(
                    f'<div style="display:flex; gap:0.5rem; align-items:flex-start; '
                    f'padding:0.3rem 0; border-bottom:1px solid var(--border);">'
                    f'<span style="min-width:1.4rem; height:1.4rem; border-radius:2px; '
                    f'background:var(--warn-lt); color:var(--warn); font-size:0.8rem; font-weight:700; '
                    f'display:flex; align-items:center; justify-content:center;">{idx}</span>'
                    f'<span style="font-size:0.95rem; color:var(--text); line-height:1.6;">{cond}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )



def run_sensitivity_test(sector_name: str, vix_val: float,
                         macro_context: str = "", news_context: str = "") -> str:
    """
    Run a lightweight sensitivity check: vary VIX by ±5 and ask if direction changes.
    Returns "LOW", "MEDIUM", or "HIGH" sensitivity flag.
    """
    if _model is None:
        return ""
    prompt = (
        f"你是一名板块研究分析师。当前分析对象是【{sector_name}】板块，VIX={vix_val}。\n"
        f"宏观背景摘要：{macro_context[:200] if macro_context else '无'}\n\n"
        f"问题：如果 VIX 从 {vix_val} 上升到 {vix_val + 5:.1f}，你对该板块的"
        f"配置方向（超配/标配/低配）是否会改变？\n"
        f"同样，如果 VIX 下降到 {max(10, vix_val - 5):.1f}，方向是否改变？\n\n"
        "请只回答：\n"
        "VIX+5方向: [超配/标配/低配]\n"
        "VIX-5方向: [超配/标配/低配]\n"
        "原始方向: [超配/标配/低配]"
    )
    try:
        resp = _model.generate_content(prompt).text
        import re
        directions = re.findall(r"(超配|标配|低配)", resp)
        if len(directions) >= 3:
            orig = directions[2]
            up   = directions[0]
            down = directions[1]
            changes = sum([up != orig, down != orig])
            if changes == 0:
                return "LOW"
            if changes == 1:
                return "MEDIUM"
            return "HIGH"
    except Exception:
        pass
    return ""


def generate_sector_analysis(
    sector_name: str, vix_val: float,
    macro_context: str = "", news_context: str = "",
) -> str:
    try:
        return _generate_sector_analysis_cached(sector_name, vix_val, macro_context, news_context)
    except Exception as e:
        return _ai_error(e)


def generate_ai_reasons(asset_name: str, scan_data: dict, news_context: str = "", vix_val: float = 0.0) -> str:
    sharpe     = scan_data.get("sharpe", 0)
    market_fit = scan_data.get("market_fit", "N/A")
    fund_flow  = scan_data.get("fund_flow", 1.0)
    mom_1m     = scan_data.get("mom_1m")
    mom_3m     = scan_data.get("mom_3m")
    mom_6m     = scan_data.get("mom_6m")
    ann_return = scan_data.get("ann_return")

    # Multi-period momentum block — only include if data is available
    if mom_1m is not None and mom_3m is not None and mom_6m is not None:
        # Infer trend direction for the model
        if mom_1m > mom_3m > 0:
            trend_note = "近期动能加速（1m > 3m），短期势头强劲"
        elif mom_1m < mom_3m and mom_3m > 0:
            trend_note = "动能高位回落（1m < 3m），需警惕动量衰减"
        elif mom_1m < 0 < mom_6m:
            trend_note = "中期结构向上但近期承压，可能处于回调阶段"
        elif mom_6m < 0:
            trend_note = "多周期动量全面走弱，趋势背景偏空"
        else:
            trend_note = "动量结构中性"

        momentum_section = (
            f"- 多周期动量: 1m={mom_1m:+.1%}  3m={mom_3m:+.1%}  6m={mom_6m:+.1%}\n"
            f"  动量趋势判断: {trend_note}"
        )
    else:
        momentum_section = ""

    ann_return_line = f"- 年化收益率: {ann_return:.1%}" if ann_return is not None else ""
    news_section = f"\n    近期相关新闻（请在风险提示中引用）：\n    {news_context}" if news_context else ""

    prompt = f"""
    你是一名持牌投资顾问，正在向基金经理提交量化扫描结论摘要。
    本次扫描冠军为【{asset_name}】，原始指标如下：
    - 年化夏普比率: {sharpe:.2f}
    {ann_return_line}
    - 市场契合度百分位: {market_fit}%
    - 近期资金活跃度: {fund_flow:.2f}x（近10日均量 / 全期均量）
    {momentum_section}
    - 当前 VIX: {vix_val if vix_val else '未知'}
    {news_section}

    请撰写一份简短的【量化选股逻辑摘要】，包含以下四点：
    1. 【风险调整收益】：基于夏普比率 {sharpe:.2f} 说明其相对于市场的收益质量。
    2. 【动量结构判断】：结合1m/3m/6m多周期动量，判断当前动能是加速、衰减还是反转，并说明配置意义。
    3. 【资金面信号】：解读资金活跃度 {fund_flow:.2f}x 背后的市场含义。
    4. 【潜在风险提示】：结合 VIX 环境与近期新闻，指出一个需要关注的下行风险。

    写作要求：
    - 语气：客观、克制、机构级别，避免任何情绪化或劝说性表达
    - 严禁使用"白送"、"必涨"、"不冲等啥"等非专业措辞
    - 每点一句话，共4句，总字数不超过200字

    在正文结束后，必须输出以下归因块（供系统解析，格式严格固定）：

    ### [XAI_ATTRIBUTION]
    overall_confidence: [0-100，综合判断置信度]
    direction: [三选一 — 超配 / 标配 / 低配]
    invalidation_conditions: [1条失效条件]
    ### [/XAI_ATTRIBUTION]
    """
    try:
        return _model.generate_content(prompt).text
    except Exception as e:
        return _ai_error(e)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Macro Intelligence
# ─────────────────────────────────────────────────────────────────────────────

def render_tab1(vix_input: float) -> None:
    _section_label("Macro Intelligence  ·  Global News-Driven Brief")

    left, right = st.columns([3, 1], gap="large")

    with right:
        _info_panel("News Sources", [
            "Alpha Vantage Sentiment API",
            "GNews API (global macro)",
            "Yahoo Finance RSS",
            "Google News RSS",
        ])
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        _info_panel("Coverage Topics", [
            "Federal Reserve / Central Banks",
            "Inflation / CPI / GDP",
            "Geopolitical Risk Events",
            "Global Risk Sentiment (VIX)",
        ])

    with left:
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        already_today = st.session_state.get("macro_analysis_date") == today_str

        run = st.button("▶  Run Macro Analysis", type="primary", width='stretch')

        if run and already_today:
            st.info("今日宏观简报已生成，直接展示缓存结果。如需强制刷新请点击 ↺ Refresh。")
        elif run:
            _run_macro_analysis(vix_input, overwrite=False)

        if st.session_state.get("macro_memo"):
            memo = st.session_state.macro_memo
            # Timestamp + force-refresh row
            ts = st.session_state.get("macro_analysis_time")
            ts_str = ts.strftime("%Y-%m-%d  %H:%M UTC") if ts else "—"
            col_ts, col_refresh = st.columns([3, 1], gap="small")
            with col_ts:
                st.markdown(
                    f'<div style="font-size:0.88rem; color:var(--muted); margin-bottom:0.4rem;">'
                    f'Last updated: {ts_str}</div>',
                    unsafe_allow_html=True,
                )
            with col_refresh:
                if st.button("↺  Refresh", key="refresh_macro", width='stretch'):
                    _get_ai_analysis_cached.clear()
                    st.session_state.pop("macro_pdf_bytes", None)
                    _run_macro_analysis(vix_input, overwrite=True)
                    st.rerun()
            with st.container(border=True):
                _render_memo_sections(memo, accent="#1A56DB")
            if not _is_ai_error(memo):
                if st.button("📄  Prepare Full PDF Report", key="prepare_macro_pdf",
                             width='stretch'):
                    with st.spinner("Generating extended research report for PDF..."):
                        extended = _generate_pdf_content(
                            "macro", memo,
                            st.session_state.get("macro_news_ctx", ""),
                            {"vix": vix_input},
                        )
                        st.session_state.macro_pdf_bytes = generate_pdf_report(
                            extended, "Macro Intelligence Brief", {"VIX": vix_input},
                        )
                if st.session_state.get("macro_pdf_bytes"):
                    st.download_button(
                        "↓  Download PDF Report",
                        st.session_state.macro_pdf_bytes,
                        "Macro_Intelligence_Brief.pdf",
                        mime="application/pdf",
                        width='stretch',
                    )
            # ── Human review gate ──────────────────────────────────────────────
            _macro_draft = st.session_state.get("macro_draft")
            if _macro_draft:
                with st.expander("📋  审核草稿 · 确认入库", expanded=True):
                    st.markdown(
                        '<div style="font-size:0.82rem; color:var(--muted); margin-bottom:0.8rem;">'
                        'Macro 决策自动标记为已验证（无价格验证路径）。'
                        '确认前请检查宏观制度判断是否与当前 VIX 及市场状态一致。'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    _regime_edit = st.selectbox(
                        "宏观制度（可修改）",
                        options=["低波动/牛市", "温和波动", "震荡期", "高波动/危机"],
                        index=["低波动/牛市", "温和波动", "震荡期", "高波动/危机"].index(
                            _macro_draft.get("macro_regime", "温和波动")
                            if _macro_draft.get("macro_regime") in
                               ["低波动/牛市", "温和波动", "震荡期", "高波动/危机"]
                            else "温和波动"
                        ),
                        key="macro_regime_edit",
                    )
                    _mc1, _mc2 = st.columns(2)
                    with _mc1:
                        if st.button("✅  确认入库", key="confirm_macro_draft",
                                     type="primary", width='stretch'):
                            _edited = _regime_edit != _macro_draft.get("macro_regime")
                            # Macro only editable field is regime (categorical → binary ratio)
                            _m_ratio = 1.0 if _edited else None
                            save_decision(
                                **{k: v for k, v in _macro_draft.items()
                                   if k != "macro_regime"},
                                macro_regime=_regime_edit,
                                decision_source="human_edited" if _edited else "ai_drafted",
                                edit_ratio=_m_ratio,
                            )
                            st.session_state.pop("macro_draft", None)
                            st.success("宏观决策已入库。")
                            st.rerun()
                    with _mc2:
                        if st.button("🗑  放弃草稿", key="discard_macro_draft",
                                     width='stretch'):
                            st.session_state.pop("macro_draft", None)
                            st.rerun()

            # ── Watchlist panel ────────────────────────────────────────────────
            _wl_saved = st.session_state.pop("watchlist_saved_count", None)
            if _wl_saved:
                st.success(f"✅ 已从本次分析提取 {_wl_saved} 条监控项并持久化")

            _wl_pending = get_pending_watch_items()
            if _wl_pending:
                import datetime as _dt
                _today = _dt.date.today()
                _overdue = [i for i in _wl_pending if i["check_by"] and i["check_by"] < _today]
                _upcoming = [i for i in _wl_pending if not (i["check_by"] and i["check_by"] < _today)]
                _expander_label = (
                    f"📋 监控清单  ·  {len(_wl_pending)} 项未解决"
                    + (f"  ⚠️ {len(_overdue)} 项已到期" if _overdue else "")
                )
                with st.expander(_expander_label, expanded=bool(_overdue)):
                    _cat_icons = {
                        "data_release": "📊",
                        "key_level":    "📍",
                        "market_signal": "📡",
                    }
                    if _overdue:
                        st.markdown("**已到期 — 请记录实际结果**")
                        for _item in _overdue:
                            with st.container(border=True):
                                _ic = _cat_icons.get(_item["category"], "•")
                                st.markdown(
                                    f"{_ic} **{_item['item_text']}**"
                                    + (f"  *(预期: {_item['expected_value']})*" if _item["expected_value"] else "")
                                )
                                _col_a, _col_b, _col_c = st.columns([2, 1, 1])
                                _actual = _col_a.text_input(
                                    "实际结果", key=f"wl_actual_{_item['id']}",
                                    placeholder="例：CPI 3.4%，超预期",
                                )
                                _outcome = _col_b.selectbox(
                                    "结果",
                                    ["matched", "surprised", "n/a"],
                                    key=f"wl_outcome_{_item['id']}",
                                )
                                if _col_c.button("确认", key=f"wl_resolve_{_item['id']}"):
                                    resolve_watch_item(
                                        _item["id"],
                                        actual_value=_actual,
                                        outcome=_outcome,
                                    )
                                    st.rerun()

                    if _upcoming:
                        st.markdown("**待跟踪项目**")
                        for _item in _upcoming:
                            _ic = _cat_icons.get(_item["category"], "•")
                            _check = _item["check_by"].strftime("%m/%d") if _item["check_by"] else "—"
                            st.markdown(
                                f"{_ic} {_item['item_text']}"
                                + (f"  *(预期: {_item['expected_value']})*" if _item["expected_value"] else "")
                                + f"  <span style='color:var(--muted);font-size:0.85em;'>截止 {_check}</span>",
                                unsafe_allow_html=True,
                            )

            # Show the raw news that fed the analysis
            news_ctx_display = st.session_state.get("macro_news_ctx", "")
            if news_ctx_display and news_ctx_display != "暂无近48小时相关新闻。":
                with st.expander("Macro News Feed  ·  Source Data", expanded=False):
                    _render_news_feed(news_ctx_display)
        else:
            _e_bg = "rgba(255,255,255,0.04)" if _theme.is_dark() else "#F8FAFC"
            _e_bd = "rgba(255,255,255,0.10)" if _theme.is_dark() else "#CBD5E1"
            st.markdown(f"""
            <div style="background:{_e_bg}; border:1px dashed {_e_bd}; border-radius:3px;
                        padding:2rem; text-align:center; color:var(--muted); font-size:1.05rem;">
              Click "Run Macro Analysis" to aggregate global macro news and generate the intelligence brief.
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Signal Station (daily cache)
# ─────────────────────────────────────────────────────────────────────────────

# Derive signal ETF map from the live universe — no hardcoded copy.
_SIGNAL_ETF_MAP: dict[str, str] = _get_active_sector_etf()

_MOVE_THRESHOLD = 0.015   # 1.5% daily move triggers a signal


def _fetch_market_snapshot() -> dict:
    """
    Fetch today's returns for sector ETFs + VIX.
    Returns {ticker: {"return": float, "sector": str}} and vix float.
    Single yfinance batch download — one request for all tickers.
    """
    import yfinance as yf

    tickers = list(_SIGNAL_ETF_MAP.values()) + ["^VIX", "SPY"]
    try:
        data = yf.download(tickers, period="2d", progress=False, auto_adjust=True)
        closes = data["Close"]
        if len(closes) < 2:
            return {"etfs": {}, "vix": None, "spy_return": None}

        returns: dict[str, dict] = {}
        for sector, ticker in _SIGNAL_ETF_MAP.items():
            if ticker in closes.columns:
                p0 = float(closes[ticker].iloc[-2])
                p1 = float(closes[ticker].iloc[-1])
                if p0 > 0:
                    returns[ticker] = {
                        "return": (p1 - p0) / p0,
                        "sector": sector,
                        "price":  p1,
                    }

        vix = None
        if "^VIX" in closes.columns:
            vix = float(closes["^VIX"].iloc[-1])

        spy_ret = None
        if "SPY" in closes.columns:
            p0 = float(closes["SPY"].iloc[-2])
            p1 = float(closes["SPY"].iloc[-1])
            if p0 > 0:
                spy_ret = (p1 - p0) / p0

        return {"etfs": returns, "vix": vix, "spy_return": spy_ret}
    except Exception as e:
        return {"etfs": {}, "vix": None, "spy_return": None, "error": str(e)}


def _check_invalidation_alerts(
    current_vix: float,
    etf_returns: dict | None = None,
    current_regime: str = "",
) -> list[dict]:
    """
    Query Alpha Memory for unverified, non-superseded decisions that have
    invalidation_conditions, then check three trigger dimensions:

    1. VIX threshold  — "VIX突破30" / "VIX>30"
    2. Regime change  — "宏观制度切换" / "制度改变" keywords + compare stored regime
    3. ETF drawdown   — "跌幅超过X%" / "下跌X%" patterns matched against live return

    Returns list of alert dicts (triggered ones sorted first).
    """
    import re as _re
    alerts = []
    # Build sector → 1-day return lookup from live ETF data
    _sector_ret: dict[str, float] = {}
    if etf_returns:
        for _, _info in etf_returns.items():
            _s = _info.get("sector", "")
            if _s:
                _sector_ret[_s] = _info.get("return", 0.0)

    _REGIME_KEYWORDS = ("宏观制度切换", "制度改变", "制度切换", "宏观制度变化", "regime change")
    _DRAWDOWN_PAT    = _re.compile(r"(?:跌幅?超过?|下跌超过?|下跌)\s*(\d+(?:\.\d+)?)\s*%")

    try:
        with SessionFactory() as session:
            pending = (
                session.query(DecisionLog)
                .filter(
                    DecisionLog.verified == False,
                    DecisionLog.invalidation_conditions.isnot(None),
                    (DecisionLog.superseded == False) | (DecisionLog.superseded == None),
                )
                .order_by(DecisionLog.created_at.desc())
                .limit(20)
                .all()
            )
            for dec in pending:
                cond     = dec.invalidation_conditions or ""
                age_days = (datetime.now(timezone.utc).replace(tzinfo=None) - dec.created_at).days
                sector   = dec.sector_name or dec.tab_type or ""
                reasons: list[str] = []

                # ── 1. VIX threshold ─────────────────────────────────────────
                vix_match = _re.search(r"VIX\s*[>＞突破]\s*(\d+)", cond)
                if vix_match and current_vix is not None:
                    threshold = float(vix_match.group(1))
                    if current_vix >= threshold * 0.9:
                        tag = "≥" if current_vix >= threshold else "≈"
                        reasons.append(f"VIX={current_vix:.1f} {tag} 阈值{threshold:.0f}")

                # ── 2. Regime change ─────────────────────────────────────────
                if any(kw in cond for kw in _REGIME_KEYWORDS):
                    stored_regime  = dec.macro_regime or ""
                    if current_regime and stored_regime and current_regime != stored_regime:
                        reasons.append(
                            f"宏观制度切换：{stored_regime} → {current_regime}"
                        )

                # ── 3. ETF drawdown ──────────────────────────────────────────
                dd_match = _DRAWDOWN_PAT.search(cond)
                if dd_match and sector in _sector_ret:
                    dd_threshold = float(dd_match.group(1)) / 100
                    live_ret     = _sector_ret[sector]
                    if live_ret <= -dd_threshold * 0.9:
                        tag = "≤" if live_ret <= -dd_threshold else "≈"
                        reasons.append(
                            f"{sector} 1D={live_ret:+.2%} {tag} 跌幅阈值-{dd_threshold:.0%}"
                        )

                triggered = bool(reasons)
                reason    = "  ·  ".join(reasons)
                if triggered:
                    status = "🚨 已触发" if any(
                        (current_vix or 0) >= float(m.group(1))
                        for m in [_re.search(r"VIX\s*[>＞突破]\s*(\d+)", cond)] if m
                    ) or any(kw in cond for kw in _REGIME_KEYWORDS) else "⚠ 接近触发"
                else:
                    status = "✅ 未触发"

                alerts.append({
                    "decision_id": dec.id,
                    "sector":    sector,
                    "direction": dec.direction or "—",
                    "date":      dec.created_at.strftime("%Y-%m-%d"),
                    "age_days":  age_days,
                    "condition": cond[:80],
                    "triggered": triggered,
                    "status":    status,
                    "reason":    reason,
                    "horizon":   dec.horizon or "季度(3个月)",
                })
    except Exception:
        pass
    return alerts


def _detect_regime_change(current_vix: float) -> dict:
    """
    Compare current VIX-inferred regime against the last recorded macro_regime
    in Alpha Memory. Returns change info dict.
    """
    current_regime = _infer_macro_regime(current_vix) if current_vix else "未知"
    try:
        with SessionFactory() as session:
            last = (
                session.query(DecisionLog.macro_regime, DecisionLog.created_at)
                .filter(DecisionLog.macro_regime.isnot(None))
                .order_by(DecisionLog.created_at.desc())
                .first()
            )
            if not last:
                return {"changed": False, "current": current_regime, "previous": None}

            previous_regime = last[0]
            last_date       = last[1].strftime("%Y-%m-%d")

            if previous_regime != current_regime:
                # Count how many unverified decisions used the old regime
                affected = (
                    session.query(DecisionLog)
                    .filter(
                        DecisionLog.macro_regime == previous_regime,
                        DecisionLog.verified == False,
                    )
                    .count()
                )
                return {
                    "changed":   True,
                    "current":   current_regime,
                    "previous":  previous_regime,
                    "last_date": last_date,
                    "affected":  affected,
                }
    except Exception:
        pass
    return {"changed": False, "current": current_regime, "previous": None}


def _compute_tab2_signals() -> dict:
    """Compute all three signal directions. Called once per day."""
    snapshot       = _fetch_market_snapshot()
    vix            = snapshot.get("vix")
    current_regime = _infer_macro_regime(vix or 20.0)
    alerts         = _check_invalidation_alerts(
        vix or 20.0,
        etf_returns=snapshot.get("etfs", {}),
        current_regime=current_regime,
    )
    regime    = _detect_regime_change(vix or 20.0)

    # Direction 1: significant movers
    movers = [
        v for v in snapshot.get("etfs", {}).values()
        if abs(v["return"]) >= _MOVE_THRESHOLD
    ]
    movers.sort(key=lambda x: abs(x["return"]), reverse=True)

    return {
        "snapshot":     snapshot,
        "vix":          vix,
        "spy_return":   snapshot.get("spy_return"),
        "movers":       movers,
        "inv_alerts":   alerts,
        "regime":       regime,
        "computed_at":  datetime.now(timezone.utc).strftime("%H:%M UTC"),
    }


def render_tab2() -> None:
    _section_label("Signal Station  ·  Daily Market Intelligence")

    today_str = datetime.now(timezone.utc).date().isoformat()
    if st.session_state.get("tab2_signal_date") != today_str:
        with st.spinner("Computing daily signals..."):
            st.session_state["tab2_signals"]     = _compute_tab2_signals()
            st.session_state["tab2_signal_date"] = today_str

    sig = st.session_state.get("tab2_signals", {})

    # Manual refresh button
    col_title, col_btn = st.columns([4, 1])
    with col_btn:
        if st.button("↺ Refresh", key="tab2_refresh", width='stretch'):
            with st.spinner("Refreshing signals..."):
                st.session_state["tab2_signals"]     = _compute_tab2_signals()
                st.session_state["tab2_signal_date"] = today_str
            st.rerun()
    with col_title:
        st.markdown(
            f'<div style="font-size:1.05rem; color:var(--muted); padding-top:0.4rem;">'
            f'Last computed: {sig.get("computed_at", "—")}  ·  '
            f'VIX: <b style="color:var(--text)">{sig.get("vix", "—"):.1f}</b>'
            f'  ·  SPY: <b style="color:var(--text)">'
            f'{sig.get("spy_return", 0):+.2%}</b></div>'
            if sig.get("vix") else
            '<div style="font-size:1.05rem; color:var(--muted);">Loading...</div>',
            unsafe_allow_html=True,
        )

    # ── Direction 3: Regime change alert ──────────────────────────────────────
    regime = sig.get("regime", {})
    if regime.get("changed"):
        st.markdown(f"""
        <div style="background:var(--warn-lt); border:1.5px solid var(--warn-bd);
                    border-radius:3px; padding:0.8rem 1.2rem; margin-bottom:0.8rem;">
          <span style="font-size:1.1rem; font-weight:800; color:var(--warn);">
            🔔 宏观制度切换检测
          </span><br>
          <span style="font-size:1.0rem; color:var(--text);">
            <b>{regime['previous']}</b> → <b>{regime['current']}</b>
            &nbsp;·&nbsp; 上次记录: {regime.get('last_date','—')}
            &nbsp;·&nbsp; 受影响未到期决策: <b>{regime.get('affected', 0)}</b> 条
          </span><br>
          <span style="font-size:0.95rem; color:var(--warn);">
            → 建议前往 Tab 3 重新评估相关板块配置
          </span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:var(--success-lt); border:1px solid var(--success-bd);
                    border-radius:3px; padding:0.6rem 1.2rem; margin-bottom:0.8rem;">
          <span style="font-size:1.0rem; color:var(--success);">
            ✅ 宏观制度稳定 · 当前制度: <b>{regime.get('current', '—')}</b>
          </span>
        </div>""", unsafe_allow_html=True)

    # ── Direction 1: Significant movers ───────────────────────────────────────
    movers = sig.get("movers", [])
    _section_label("板块 ETF 显著波动")
    if movers:
        cols = st.columns(min(len(movers), 4), gap="small")
        for idx, m in enumerate(movers[:8]):
            col = cols[idx % min(len(movers), 4)]
            ret     = m["return"]
            color   = "var(--success)" if ret > 0 else "var(--danger)"
            bg      = "var(--success-lt)" if ret > 0 else "var(--danger-lt)"
            border  = "var(--success-bd)" if ret > 0 else "var(--danger-bd)"
            arrow   = "▲" if ret > 0 else "▼"
            sector  = m["sector"]
            # Check if we have a recent decision for this sector
            today_reports = get_all_today_sector_reports()
            has_analysis  = sector in today_reports
            tag = "✓ 已分析" if has_analysis else "→ 建议分析"
            tag_color = "var(--muted)" if has_analysis else "var(--warn)"
            col.markdown(f"""
            <div style="background:{bg}; border:1.5px solid {border}; border-radius:3px;
                        padding:0.6rem 0.8rem; text-align:center;">
              <div style="font-size:0.95rem; font-weight:700; color:var(--text);">{sector}</div>
              <div style="font-size:1.3rem; font-weight:800; color:{color};">
                {arrow} {abs(ret):.2%}
              </div>
              <div style="font-size:0.88rem; color:{tag_color}; font-weight:600;">{tag}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="color:var(--muted); font-size:1.05rem; padding:0.5rem 0;">'
            '今日无板块 ETF 涨跌幅超过 1.5%，市场波动平稳。</div>',
            unsafe_allow_html=True,
        )

    # ── Direction 2: Invalidation condition alerts ────────────────────────────
    inv_alerts = sig.get("inv_alerts", [])
    _section_label("失效条件监控")

    # ── Auto-trigger toggle ───────────────────────────────────────────────────
    if "auto_revision_enabled" not in st.session_state:
        st.session_state["auto_revision_enabled"] = False

    _auto_on = st.session_state["auto_revision_enabled"]
    _tog_col, _status_col = st.columns([2, 5], gap="small")
    with _tog_col:
        if _auto_on:
            if st.button("⏹ 关闭自动修订", key="auto_revision_off", width='stretch'):
                st.session_state["auto_revision_enabled"] = False
                st.rerun()
        else:
            if st.button("▶ 开启自动修订", key="auto_revision_on", width='stretch'):
                st.session_state["_auto_revision_arming"] = True
                st.rerun()
    with _status_col:
        if _auto_on:
            st.markdown(
                '<div style="font-size:0.88rem; color:var(--success); padding-top:0.55rem; font-weight:600;">'
                '● 自动修订已启用 · 失效条件触发时将自动发起修订分析（24h/板块冷却）</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="font-size:0.88rem; color:var(--muted); padding-top:0.55rem;">'
                '○ 自动修订未启用 · 触发条件须手动发起修订</div>',
                unsafe_allow_html=True,
            )

    # ── 10-second arm countdown ───────────────────────────────────────────────
    if st.session_state.get("_auto_revision_arming"):
        st.session_state.pop("_auto_revision_arming")
        _warn = st.empty()
        for _i in range(10, 0, -1):
            _warn.warning(
                f"⚠ 自动修订即将启用（{_i}秒后生效）· 启用后每次失效条件触发将自动消耗 API 配额并写入 Alpha Memory。"
                f"如需取消请立即刷新页面。"
            )
            time.sleep(1)
        _warn.empty()
        st.session_state["auto_revision_enabled"] = True
        st.rerun()

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    if inv_alerts:
        triggered = [a for a in inv_alerts if a["triggered"]]
        normal    = [a for a in inv_alerts if not a["triggered"]]

        if triggered:
            for a in triggered:
                _sector   = a["sector"]
                _dec_id   = a.get("decision_id")
                _tickers  = AUDIT_TICKERS.get(_sector, [])
                _rev_reason = a["reason"]

                # ── Auto-trigger execution (once per sector per 24h) ──────────
                if _auto_on and _dec_id:
                    _last_rev = get_last_revision_time(_sector)
                    _now      = datetime.now(timezone.utc)
                    _elapsed  = (
                        (_now - _last_rev.replace(tzinfo=timezone.utc)).total_seconds() / 3600
                        if _last_rev else 999
                    )
                    if _elapsed >= 24:
                        _run_sector_analysis(
                            _sector, sig.get("vix", 20.0), _tickers,
                            overwrite=True,
                            parent_decision_id=_dec_id,
                            revision_reason=_rev_reason,
                        )
                        st.success(f"✅ 自动修订完成 · {_sector} · {_rev_reason}")
                        st.rerun()

                # ── Alert card + manual revision button ───────────────────────
                _confirm_key = f"revise_confirm_{_dec_id}"
                _card_col, _btn_col = st.columns([5, 1], gap="small")
                with _card_col:
                    st.markdown(f"""
                    <div style="background:var(--danger-lt); border:1.5px solid var(--danger-bd);
                                border-radius:3px; padding:0.7rem 1rem; margin-bottom:0.3rem;">
                      <span style="font-size:1.05rem; font-weight:700; color:var(--danger);">
                        {a['status']}  ·  {_sector}
                      </span>
                      <span style="font-size:0.95rem; color:var(--muted); margin-left:1rem;">
                        {a['direction']} · {a['date']} · {a['age_days']}天前
                      </span><br>
                      <span style="font-size:0.95rem; color:var(--text);">
                        条件: {a['condition']}
                      </span><br>
                      <span style="font-size:0.92rem; color:var(--danger);">
                        {_rev_reason}
                      </span>
                    </div>""", unsafe_allow_html=True)
                with _btn_col:
                    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                    if _dec_id:
                        if not st.session_state.get(_confirm_key):
                            if st.button(
                                "修订", key=f"revise_{_dec_id}",
                                width='stretch',
                                help="点击后可补充修订原因，再确认发起分析",
                            ):
                                st.session_state[_confirm_key] = True
                                st.rerun()
                        else:
                            if st.button(
                                "取消", key=f"revise_cancel_{_dec_id}",
                                width='stretch',
                            ):
                                st.session_state.pop(_confirm_key, None)
                                st.rerun()

                # ── Confirm panel (shown after clicking 修订) ─────────────────
                if _dec_id and st.session_state.get(_confirm_key):
                    with st.container(border=True):
                        st.markdown(
                            '<div style="font-size:0.88rem; color:var(--muted); margin-bottom:0.3rem;">'
                            '补充修订原因（系统已预填触发信号，可直接使用或详细说明背景）</div>',
                            unsafe_allow_html=True,
                        )
                        _edited_reason = st.text_area(
                            "修订原因",
                            value=_rev_reason,
                            height=80,
                            key=f"revise_reason_{_dec_id}",
                            label_visibility="collapsed",
                            placeholder="例：Fed 意外加息 50bp，流动性收缩超预期，原超配假设不成立",
                        )
                        if st.button(
                            "确认修订  ▶", key=f"revise_submit_{_dec_id}",
                            type="primary", width='stretch',
                        ):
                            st.session_state.pop(_confirm_key, None)
                            _run_sector_analysis(
                                _sector, sig.get("vix", 20.0), _tickers,
                                overwrite=True,
                                parent_decision_id=_dec_id,
                                revision_reason=_edited_reason.strip() or _rev_reason,
                            )
                            st.success(f"✅ 修订完成 · {_sector}")
                            st.rerun()

        if normal:
            _brd = "var(--border)"
            _txt = "var(--text)"
            _mut = "var(--muted)"
            _rows_html = "".join(
                f'<div style="font-size:0.95rem; color:{_txt}; padding:0.35rem 0; '
                f'border-bottom:1px solid {_brd};">'
                f'✅ <b>{a["sector"]}</b>'
                f'<span style="color:{_mut}; margin-left:0.6rem;">'
                f'{a["direction"]} · {a["date"]} · 条件: {a["condition"]}</span>'
                f'</div>'
                for a in normal
            )
            st.markdown(
                f'<div style="font-size:0.88rem; color:{_mut}; margin-bottom:0.3rem;">'
                f'监控中的决策（{len(normal)} 条）· 全部正常</div>'
                f'<div style="overflow-y:auto; max-height:420px; padding-right:4px;">'
                f'{_rows_html}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div style="color:var(--muted); font-size:1.05rem; padding:0.5rem 0;">'
            'Alpha Memory 暂无待监控的失效条件。</div>',
            unsafe_allow_html=True,
        )

    # ── TradingView charts (collapsed) ────────────────────────────────────────
    CHARTS = [
        ("NASDAQ:SMH",   "SMH  ·  AI算力/半导体"),
        ("NASDAQ:QQQ",   "QQQ  ·  科技成长(纳指)"),
        ("AMEX:XBI",     "XBI  ·  生物科技"),
        ("AMEX:XLF",     "XLF  ·  金融"),
        ("AMEX:XLE",     "XLE  ·  全球能源"),
        ("AMEX:XLI",     "XLI  ·  工业/基建"),
        ("AMEX:XLV",     "XLV  ·  医疗健康"),
        ("AMEX:XLP",     "XLP  ·  防御消费"),
        ("AMEX:XLY",     "XLY  ·  消费科技"),
        ("AMEX:VNQ",     "VNQ  ·  美国REITs"),
        ("AMEX:GLD",     "GLD  ·  黄金"),
        ("NASDAQ:TLT",   "TLT  ·  美国长债"),
        ("NASDAQ:ICLN",  "ICLN  ·  清洁能源"),
        ("AMEX:ASHR",    "ASHR  ·  沪深300"),
        ("AMEX:KWEB",    "KWEB  ·  中国科技"),
        ("AMEX:EWS",     "EWS  ·  新加坡蓝筹"),
        ("AMEX:XLC",     "XLC  ·  通讯传媒"),
        ("AMEX:HYG",     "HYG  ·  高收益债"),
    ]
    with st.expander("市场走势图表  ·  TradingView Live Charts", expanded=False):
        for i in range(0, len(CHARTS), 2):
            c1, c2 = st.columns(2, gap="small")
            sym1, ttl1 = CHARTS[i]
            sym2, ttl2 = CHARTS[i + 1]
            with c1:
                st.markdown(
                    f'<div style="font-size:0.88rem; font-weight:700; color:var(--muted); '
                    f'text-transform:uppercase; letter-spacing:0.08em; '
                    f'margin-bottom:0.3rem;">{ttl1}</div>',
                    unsafe_allow_html=True,
                )
                render_tv_chart(sym1, ttl1.split("·")[0].strip())
            with c2:
                st.markdown(
                    f'<div style="font-size:0.88rem; font-weight:700; color:var(--muted); '
                    f'text-transform:uppercase; letter-spacing:0.08em; '
                    f'margin-bottom:0.3rem;">{ttl2}</div>',
                    unsafe_allow_html=True,
                )
                render_tv_chart(sym2, ttl2.split("·")[0].strip())


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Sector Risk
# ─────────────────────────────────────────────────────────────────────────────

def render_tab3(vix_input: float) -> None:
    _section_label("Sector Risk  ·  Dynamic Sector Intelligence")

    # ── Sector selector ───────────────────────────────────────────────────────
    sector_names = list(_get_active_sector_etf().keys())
    st.markdown("""
    <div style="background:var(--accent-lt); border:1.5px solid var(--border);
                border-radius:3px; padding:0.7rem 1rem 0.3rem; margin-bottom:0.8rem;">
      <div style="font-size:1rem; font-weight:800; text-transform:uppercase;
                  letter-spacing:0.1em; color:var(--accent); margin-bottom:0.15rem;">
        选择分析板块
      </div>
      <div style="font-size:0.92rem; color:var(--muted); margin-bottom:0.4rem;">
        覆盖全球 18 个核心板块 · 结合实时新闻与宏观背景生成专属研究报告
      </div>
    </div>""", unsafe_allow_html=True)
    col_sel, col_btn = st.columns([3, 1], gap="medium")
    with col_sel:
        selected = st.selectbox(
            "选择板块",
            sector_names,
            key="sector_risk_selector",
        )
    with col_btn:
        st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
        run = st.button("▶  开始分析", type="primary", width='stretch')

    tickers = AUDIT_TICKERS.get(selected, [])
    memo_key = f"sector_memo_{selected}"
    news_key = f"sector_news_{selected}"

    left, right = st.columns([3, 1], gap="large")

    with right:
        _info_panel("Selected Sector", [selected] + [f"  {t}" for t in tickers[:4]])
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        _info_panel("Methodology", [
            "Live News Sentiment",
            "Macro Context Overlay",
            "VIX-adjusted Risk Bands",
        ])
        if st.session_state.get("macro_memo"):
            st.markdown("""
            <div style="margin-top:0.5rem; padding:0.45rem 0.7rem; background:var(--success-lt);
                        border:1px solid var(--success-bd); border-radius:3px;
                        font-size:0.92rem; color:var(--success); font-weight:600;">
              ✓  Macro context loaded from Tab 1
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="margin-top:0.5rem; padding:0.45rem 0.7rem; background:var(--warn-lt);
                        border:1px solid var(--warn-bd); border-radius:3px;
                        font-size:0.92rem; color:var(--warn); font-weight:600;">
              ⚠  Run Macro Analysis first for richer context
            </div>""", unsafe_allow_html=True)

    with left:
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        sector_date_key = f"sector_analysis_date_{selected}"
        already_today = st.session_state.get(sector_date_key) == today_str

        if run and already_today:
            st.info(f"今日【{selected}】板块分析已生成，直接展示缓存结果。如需强制刷新请点击 ↺ Refresh。")
        elif run:
            _run_sector_analysis(selected, vix_input, tickers, overwrite=False)

        if st.session_state.get(memo_key):
            memo = st.session_state[memo_key]
            # Timestamp + force-refresh row
            ts = st.session_state.get(f"sector_analysis_time_{selected}")
            ts_str = ts.strftime("%Y-%m-%d  %H:%M UTC") if ts else "—"
            col_ts, col_refresh = st.columns([3, 1], gap="small")
            with col_ts:
                st.markdown(
                    f'<div style="font-size:0.88rem; color:var(--muted); margin-bottom:0.4rem;">'
                    f'Last updated: {ts_str}</div>',
                    unsafe_allow_html=True,
                )
            with col_refresh:
                if st.button("↺  Refresh", key=f"refresh_sector_{selected}", width='stretch'):
                    _generate_sector_analysis_cached.clear()
                    st.session_state.pop(f"sector_pdf_bytes_{selected}", None)
                    _run_sector_analysis(selected, vix_input, tickers, overwrite=True)
                    st.rerun()
            with st.container(border=True):
                _render_memo_sections(memo, accent="#1A56DB")
            _xai_data = st.session_state.get(f"sector_xai_{selected}", {})
            if _xai_data:
                _render_xai_panel(_xai_data, sector=selected)
            # Divergence check: bullish news vs AI confidence / direction / VIX
            _sent_score = _extract_sentiment_score(
                st.session_state.get(f"sector_news_{selected}", "")
            )
            _render_divergence_warning(_sent_score, {
                "vix":        vix_input,
                "confidence": (_xai_data.get("overall_confidence") if _xai_data else None),
                "direction":  extract_direction(memo),
            })
            if not _is_ai_error(memo):
                pdf_btn_key = f"prepare_sector_pdf_{selected}"
                pdf_bytes_key = f"sector_pdf_bytes_{selected}"
                if st.button("📄  Prepare Full PDF Report", key=pdf_btn_key,
                             width='stretch'):
                    with st.spinner(f"Generating extended {selected} report for PDF..."):
                        extended = _generate_pdf_content(
                            "sector", memo,
                            st.session_state.get(news_key, ""),
                            {
                                "vix": vix_input,
                                "sector": selected,
                                "macro_ctx": st.session_state.get("macro_memo", ""),
                            },
                        )
                        st.session_state[pdf_bytes_key] = generate_pdf_report(
                            extended,
                            f"Sector Intelligence — {selected}",
                            {"Sector": selected, "VIX": vix_input},
                        )
                if st.session_state.get(pdf_bytes_key):
                    st.download_button(
                        "↓  Download PDF Report",
                        st.session_state[pdf_bytes_key],
                        f"Sector_{selected}.pdf",
                        mime="application/pdf",
                        width='stretch',
                    )
            # ── Human review gate ──────────────────────────────────────────────
            _draft_key = f"sector_draft_{selected}"
            _draft = st.session_state.get(_draft_key)
            if _draft:
                with st.expander("📋  审核草稿 · 确认入库", expanded=True):
                    st.markdown(
                        '<div style="font-size:0.82rem; color:var(--muted); margin-bottom:0.8rem;">'
                        '以下为 AI 草稿关键字段。请在确认前核查并按需修改失效条件，'
                        '修改后入库的记录将被标记为 <b>human_edited</b>。'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    # ── Quant Coherence flags (Method 3) ─────────────────────
                    # QC-1 (direction vs momentum) and QC-2 (overconfidence vs p_noise)
                    # are deterministic gate checks — shown as warnings.
                    # QC-3/4 are informational hints shown separately (Goodhart-safe).
                    _qc_flags = _draft.get("qc_flags") or []
                    if _qc_flags:
                        st.markdown(
                            '<div style="background:#3d2000;border-left:3px solid #f0a500;'
                            'padding:0.6rem 0.8rem;border-radius:4px;margin-bottom:0.8rem;">'
                            '<b style="color:#f0a500;">⚠️ 量化一致性检验（QC）发现以下问题，请在确认前核查：</b>'
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        for _flag in _qc_flags:
                            st.warning(_flag)
                    elif _draft.get("quant_p_noise") is not None:
                        st.success("✅ QC-1/2 通过：方向与动量一致，置信度与噪音水平匹配。")
                    _inv_edit = st.text_area(
                        "失效条件（可修改）",
                        value=_draft.get("invalidation_conditions", ""),
                        height=100,
                        key=f"inv_edit_{selected}",
                    )
                    _conf_edit = st.number_input(
                        "置信度 % （可修改）",
                        min_value=0, max_value=100,
                        value=int(_draft.get("confidence_score") or 60),
                        step=5,
                        key=f"conf_edit_{selected}",
                    )
                    _c1, _c2 = st.columns(2)
                    with _c1:
                        if st.button("✅  确认入库", key=f"confirm_draft_{selected}",
                                     type="primary", width='stretch'):
                            _orig_inv  = (_draft.get("invalidation_conditions") or "").strip()
                            _orig_conf = _draft.get("confidence_score") or 60
                            _inv_changed  = _inv_edit.strip() != _orig_inv
                            _conf_changed = _conf_edit != _orig_conf
                            _edited = _inv_changed or _conf_changed
                            # Compute Levenshtein-based edit magnitude
                            _text_ratio = _edit_ratio(_orig_inv, _inv_edit.strip()) if _inv_changed else 0.0
                            _conf_ratio = abs(_conf_edit - _orig_conf) / 100.0
                            _e_ratio    = max(_text_ratio, _conf_ratio) if _edited else None
                            _draft_save_keys = {
                                "qc_flags", "quant_p_noise", "quant_val_r2",
                                "quant_test_r2", "quant_active",
                                "invalidation_conditions", "confidence_score",
                            }
                            save_decision(
                                **{k: v for k, v in _draft.items()
                                   if k not in _draft_save_keys},
                                invalidation_conditions=_inv_edit.strip(),
                                confidence_score=_conf_edit,
                                decision_source="human_edited" if _edited else "ai_drafted",
                                quant_p_noise=_draft.get("quant_p_noise"),
                                quant_val_r2=_draft.get("quant_val_r2"),
                                quant_test_r2=_draft.get("quant_test_r2"),
                                quant_active=_draft.get("quant_active"),
                                edit_ratio=_e_ratio,
                            )
                            st.session_state.pop(_draft_key, None)
                            st.success("决策已入库。")
                            # P1-C: show Watchlist flow hint when auto-link conditions are met
                            _dir_hint = extract_direction(_draft.get("ai_conclusion", ""))
                            _inv_risk_hint = _draft.get("signal_invalidation_risk") or 50
                            _eff_conv = _conf_edit * (1.0 - 0.6 * _inv_risk_hint / 100.0)
                            if _dir_hint == "超配" and _eff_conv >= 55:
                                st.info("✓ 已加入 Watchlist — 前往 **Trading Desk** 查看待触发信号")
                            st.rerun()
                    with _c2:
                        if st.button("🗑  放弃草稿", key=f"discard_draft_{selected}",
                                     width='stretch'):
                            st.session_state.pop(_draft_key, None)
                            st.rerun()

            # Show news feed that drove the analysis
            news_ctx_display = st.session_state.get(news_key, "")
            if news_ctx_display and news_ctx_display != "暂无近48小时相关新闻。":
                with st.expander(f"News Feed  ·  {selected}  ·  Source Data", expanded=False):
                    _render_news_feed(news_ctx_display)

            # ── Debate transcript ─────────────────────────────────────────────
            debate_history = st.session_state.get(f"debate_history_{selected}", [])
            arb_notes      = st.session_state.get(f"debate_arb_{selected}", "")
            blue_output    = st.session_state.get(f"blue_output_{selected}", "")
            if debate_history:
                with st.expander("辩论记录  ·  Blue vs Red · 仲裁过程", expanded=False):
                    # Blue team original
                    if blue_output:
                        st.markdown(
                            '<div style="font-size:0.88rem; font-weight:700; '
                            'color:var(--accent); margin-bottom:0.3rem;">🔵 蓝队初始分析</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div style="background:var(--accent-lt); border-left:3px solid var(--accent); '
                            f'padding:0.7rem 1rem; border-radius:3px; font-size:0.95rem; '
                            f'color:var(--text); white-space:pre-wrap;">{blue_output[:600]}…</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

                    # Debate rounds
                    for entry in debate_history:
                        is_red   = entry["role"] == "red"
                        color    = "var(--danger)"  if is_red else "var(--success)"
                        bg       = "var(--danger-lt)" if is_red else "var(--success-lt)"
                        border   = "var(--danger-bd)" if is_red else "var(--success-bd)"
                        icon     = "🔴 红队挑战" if is_red else "🔵 蓝队防御"
                        label    = f"{icon}  ·  第 {entry['round']} 轮"
                        st.markdown(
                            f'<div style="font-size:0.88rem; font-weight:700; '
                            f'color:{color}; margin:0.5rem 0 0.2rem;">{label}</div>'
                            f'<div style="background:{bg}; border-left:3px solid {border}; '
                            f'padding:0.7rem 1rem; border-radius:3px; font-size:0.95rem; '
                            f'color:var(--text); white-space:pre-wrap;">{entry["content"]}</div>',
                            unsafe_allow_html=True,
                        )

                    # Arbitration notes
                    if arb_notes:
                        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                        st.markdown(
                            '<div style="font-size:0.88rem; font-weight:700; '
                            'color:#7C3AED; margin-bottom:0.3rem;">⚖️ 仲裁摘要</div>',
                            unsafe_allow_html=True,
                        )
                        with st.container():
                            st.markdown(
                                '<div style="background:rgba(124,58,237,0.07); border-left:3px solid #A78BFA; '
                                'border-radius:3px; padding:0.2rem 0.8rem 0.4rem;">',
                                unsafe_allow_html=True,
                            )
                            _render_memo_sections(arb_notes, accent="#7C3AED")
                            st.markdown("</div>", unsafe_allow_html=True)
        else:
            _e_bg = "rgba(255,255,255,0.04)" if _theme.is_dark() else "#F8FAFC"
            _e_bd = "rgba(255,255,255,0.10)" if _theme.is_dark() else "#CBD5E1"
            st.markdown(f"""
            <div style="background:{_e_bg}; border:1px dashed {_e_bd}; border-radius:3px;
                        padding:2rem; text-align:center; color:var(--muted); font-size:1.05rem;">
              Select a sector and click "Analyse Sector" to generate the intelligence report.<br>
              <span style="font-size:0.92rem; color:var(--muted); opacity:0.7;">
                Each sector fetches its own live news and combines it with macro context.
              </span>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Quant Audit
# ─────────────────────────────────────────────────────────────────────────────

def render_tab4(agent_executor, vix_input: float) -> None:
    _section_label("Quant Audit  ·  Agentic Deep-Dive Terminal")

    # All sectors always available; dynamic_assets may add extras (e.g. custom bundles)
    _all_sector_names = list(AUDIT_TICKERS.keys())
    _extra_assets = [k for k in st.session_state.dynamic_assets if k not in AUDIT_TICKERS]
    current_assets = _all_sector_names + _extra_assets
    sync_asset = st.session_state.get("audit_target_sync")
    default_idx = current_assets.index(sync_asset) if sync_asset in current_assets else 0

    col_sel, col_btn = st.columns([3, 1], gap="medium")
    with col_sel:
        choice = st.selectbox(
            "Target Portfolio",
            current_assets,
            index=default_idx,
            key="manual_audit_selector",
            label_visibility="collapsed",
        )
    with col_btn:
        start_audit = st.button("▶  Launch Audit", type="primary", width='stretch')

    if st.session_state.get("auto_trigger"):
        st.markdown(
            f'<span class="badge badge-info">AUTO-SYNC: {choice}</span>',
            unsafe_allow_html=True,
        )
        st.session_state.auto_trigger = False

    # ── Quick Quant Snapshot (no LLM required) ────────────────────────────────
    with st.expander("📊 Quant Signal Snapshot — 全市场因子概览（无需运行 LLM）", expanded=False):
        try:
            import datetime as _dt
            _snap_date = _dt.date.today()
            _scores_df = compute_composite_scores(_snap_date)
            if _scores_df.empty:
                st.info("暂无信号数据，请等待每日批次完成。")
            else:
                _display_cols = [c for c in
                    ["composite_score", "tsmom", "raw_return", "ann_vol", "sharpe", "carry"]
                    if c in _scores_df.columns]
                _snap = _scores_df[_display_cols].copy()
                # Friendly column labels
                _col_labels = {
                    "composite_score": "综合得分",
                    "tsmom": "TSMOM",
                    "raw_return": "原始收益",
                    "ann_vol": "年化波动",
                    "sharpe": "Sharpe",
                    "carry": "Carry",
                }
                _snap = _snap.rename(columns={c: _col_labels.get(c, c) for c in _display_cols})
                _snap = _snap.sort_values("综合得分", ascending=False)
                # Format percentages
                for _col in ["原始收益", "年化波动", "Carry"]:
                    if _col in _snap.columns:
                        _snap[_col] = _snap[_col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
                for _col in ["Sharpe"]:
                    if _col in _snap.columns:
                        _snap[_col] = _snap[_col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
                _snap["综合得分"] = _snap["综合得分"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "—")
                if "TSMOM" in _snap.columns:
                    _snap["TSMOM"] = _snap["TSMOM"].map(
                        lambda x: "▲ +1" if x == 1 else ("▼ -1" if x == -1 else "— 0")
                    )
                st.dataframe(_snap, use_container_width=True)
                st.caption(f"数据日期: {_snap_date}  ·  综合得分 = TSMOM 40% + CSMOM百分位 30% + Sharpe 20% + Carry 10%")
        except Exception as _e:
            st.caption(f"信号快照加载失败: {_e}")

    audit_cache = st.session_state.setdefault("_audit_cache", {})

    # Cache hit indicator + timestamp + Refresh
    if choice in audit_cache:
        _rec_ts = audit_cache[choice].get("created_at")
        _ts_str = _rec_ts.strftime("%Y-%m-%d  %H:%M UTC") if _rec_ts else "—"
        c_info, c_ref = st.columns([4, 1], gap="small")
        with c_info:
            st.markdown(
                f'<div style="font-size:0.88rem; color:var(--muted); padding:0.3rem 0;">'
                f'Last updated: {_ts_str}  ·  今日缓存有效，刷新页面不会重新运行</div>',
                unsafe_allow_html=True,
            )
        with c_ref:
            if st.button("↺  Refresh", key=f"refresh_audit_{choice}", width='stretch'):
                audit_cache.pop(choice, None)
                st.session_state.pop(f"_audit_saved_{choice}", None)
                st.session_state.pop(f"audit_pdf_bytes_{choice}", None)
                st.rerun()

    # ── Execution ────────────────────────────────────────────────────────────
    if start_audit and choice in audit_cache:
        st.info(f"今日【{choice}】审计已完成，直接展示缓存结果。如需重新运行请点击 ↺ Refresh。")

    if start_audit and choice not in audit_cache:
        with st.status(f"Agent auditing  {choice}...", expanded=True) as status:
            try:
                st.write("Extracting standardised factors...")
                tickers = st.session_state.dynamic_assets.get(choice) or AUDIT_TICKERS.get(choice, [])
                df = fetch_raw_data(tickers, period="6mo")
                standard_metrics = AnalyticsEngine.calculate_metrics(df)
                if not standard_metrics:
                    raise ValueError("Cannot extract valid quant factors. Check data source.")

                # Multi-period momentum from price series
                def _audit_mom(close_series, n: int) -> float | None:
                    try:
                        n = min(n, len(close_series) - 1)
                        p0, p1 = float(close_series.iloc[-n - 1]), float(close_series.iloc[-1])
                        return round((p1 - p0) / p0, 4) if p0 > 0 else None
                    except Exception:
                        return None

                _close = df.mean(axis=1) if len(df.columns) > 1 else df.iloc[:, 0]
                audit_mom_1m = _audit_mom(_close, 21)
                audit_mom_3m = _audit_mom(_close, 63)
                audit_mom_6m = _audit_mom(_close, 126)

                # Reuse Sector Risk news if already fetched today for this sector
                _audit_regime = _infer_macro_regime(vix_input)
                _cached_sector_news = st.session_state.get(f"sector_news_{choice}", "")
                if _cached_sector_news:
                    st.write("Reusing news context from Sector Risk (same sector, same session)...")
                    news_ctx = _cached_sector_news
                else:
                    st.write("Fetching real-time news context + spillover signals...")
                    ticker_for_news = tickers[0] if tickers else choice
                    _audit_perceiver = NewsPerceiver(
                        av_key=st.secrets.get("AV_KEY", ""),
                        gnews_key=st.secrets.get("GNEWS_KEY", ""),
                    )
                    news_ctx = _audit_perceiver.build_context(
                        choice, ticker_for_news, macro_regime=_audit_regime,
                    )
                    _audit_spillover = _audit_perceiver.build_spillover_context(
                        choice, macro_regime=_audit_regime,
                    )
                    if _audit_spillover:
                        news_ctx = news_ctx + "\n\n" + _audit_spillover

                scan = st.session_state.get("scan_results") or {}

                # P0-4: compute raw QuantAssessment for red_team injection
                import datetime as _dt
                _qa_raw = ""
                try:
                    _qa_list = run_quant_assessment(
                        as_of=_dt.date.today(), sectors=[choice]
                    )
                    if _qa_list:
                        _qa_raw = _qa_list[0].to_prompt_context_raw()
                except Exception as _qa_err:
                    logger.debug("QuantAssessment fetch skipped: %s", _qa_err)

                current_state = {
                    "target_assets": choice,
                    "vix_level": vix_input,
                    "macro_context": st.session_state.get("macro_memo", "N/A"),
                    "sector_risks": st.session_state.get("latest_sector_memo", "N/A"),
                    "sector_rankings": scan.get("rankings", []),
                    "news_context": news_ctx,
                    "macro_regime": _audit_regime,
                    "quant_context_raw": _qa_raw,
                    "position_context": build_position_context(choice),
                    "quant_results": {
                        "d_var":       standard_metrics["var"],
                        "var_cf":      standard_metrics.get("var_cf"),
                        "es_5pct":     standard_metrics.get("es_5pct"),
                        "var_ci":      standard_metrics.get("var_ci"),
                        "sharpe":      standard_metrics["sharpe"],
                        "sharpe_ci":   standard_metrics.get("sharpe_ci"),
                        "skewness":    standard_metrics.get("skewness"),
                        "excess_kurt": standard_metrics.get("excess_kurt"),
                        "vol":         standard_metrics["volatility"],
                        "mom_1m":      audit_mom_1m,
                        "mom_3m":      audit_mom_3m,
                        "mom_6m":      audit_mom_6m,
                    },
                    "is_robust": True,
                    "alternative_suggestion": "",
                }

                # Ensure agent's preset_assets dict has this sector's tickers
                if choice not in st.session_state.dynamic_assets:
                    st.session_state.dynamic_assets[choice] = AUDIT_TICKERS.get(choice, tickers)

                for event in agent_executor.stream(current_state):
                    if not event:
                        continue
                    for node_name, node_output in event.items():
                        icon = "⬡" if "red_team" in node_name else "◈" if "reflect" in node_name else "◇"
                        st.write(f"{icon}  `{node_name}` completed")
                        if isinstance(node_output, dict):
                            current_state.update({k: v for k, v in node_output.items() if v is not None})
                        else:
                            st.warning(f"Node {node_name} returned unexpected format.")

                # compute_quant_metrics() now includes sharpe, sharpe_ci, skewness,
                # excess_kurt, var_cf, var_ci directly — no separate merge needed.
                # Only merge `vol` (volatility label) which uses a different key name.
                _vol = standard_metrics.get("volatility")
                if _vol is not None:
                    current_state.setdefault("quant_results", {})["vol"] = _vol

                audit_cache[choice] = current_state
                st.session_state.pop(f"_audit_saved_{choice}", None)
                status.update(label="Audit pipeline complete.", state="complete")
            except Exception as e:
                st.error(f"Audit interrupted: {e}")
                status.update(label="Audit terminated with error.", state="error")

    # ── Results ───────────────────────────────────────────────────────────────
    s = audit_cache.get(choice)
    if not s:
        return
    q = s.get("quant_results", {})
    # Treat an all-None dict (DB-restored empty record) the same as missing data
    if not q or not any(v is not None for v in q.values()):
        return

    # Save audit decision exactly once per bundle per session
    saved_key = f"_audit_saved_{choice}"
    if not st.session_state.get(saved_key, False):
        _save_target = s.get("target_assets", "")
        tickers_for_save = st.session_state.dynamic_assets.get(_save_target) or AUDIT_TICKERS.get(_save_target, [])
        save_decision(
            tab_type="audit",
            ai_conclusion=(s.get("red_team_critique", "") + "\n" + s.get("audit_memo", "")),
            vix_level=s.get("vix_level", 0),
            sector_name=s.get("target_assets", ""),
            ticker=tickers_for_save[0] if tickers_for_save else "",
            news_summary=s.get("news_context", ""),   # full text, no truncation
            quant_metrics={
                "d_var":          q.get("d_var"),
                "var_ci":         q.get("var_ci"),
                "var_cf":         q.get("var_cf"),
                "es_5pct":        q.get("es_5pct"),
                "is_robust":      s.get("is_robust"),
                "confidence":     q.get("confidence_score"),
                "sharpe":         q.get("sharpe"),
                "sharpe_ci":      q.get("sharpe_ci"),
                "skewness":       q.get("skewness"),
                "excess_kurt":    q.get("excess_kurt"),
                "vol":            q.get("vol"),
                "a_ret":          q.get("a_ret"),
                "a_vol":          q.get("a_vol"),
                "p_noise":        q.get("p_noise"),
                "active":         q.get("active"),
                "sparsity":       q.get("sparsity"),
                "mom_1m":         q.get("mom_1m"),
                "mom_3m":         q.get("mom_3m"),
                "mom_6m":         q.get("mom_6m"),
                "market_fit":     q.get("market_fit"),
                "fund_flow":      q.get("fund_flow"),
                "momentum":       q.get("momentum"),
                "technical_report": s.get("technical_report", ""),
                "red_team_critique": s.get("red_team_critique", ""),
                "audit_memo":     s.get("audit_memo", ""),
            },
            confidence_score=(int(q["confidence_score"]) if q.get("confidence_score") is not None else None),
            macro_regime=_infer_macro_regime(s.get("vix_level", 0)),
            horizon="季度(3个月)",
            economic_logic=s.get("audit_memo", "")[:300],
            reflection_chain=s.get("reflection_chain", ""),
            quant_p_noise=q.get("p_noise"),
            quant_val_r2=q.get("val_r2"),
            quant_test_r2=q.get("test_r2"),
            quant_active=q.get("active"),
        )
        st.session_state[saved_key] = True

    score = q.get("confidence_score") or 0
    is_robust = s.get("is_robust", False)

    # Divergence check: bullish news vs quant risk metrics + momentum structure
    _audit_sent = _extract_sentiment_score(s.get("news_context", ""))
    _render_divergence_warning(_audit_sent, {
        "vix":    s.get("vix_level", 0),
        "d_var":  q.get("d_var"),
        "sharpe": q.get("sharpe"),
        "mom_1m": q.get("mom_1m"),
        "mom_6m": q.get("mom_6m"),
    })

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    _section_label("Audit Results")

    # ── Entire results section wrapped in a bordered container ────────────────
    with st.container(border=True):
        # Status banner
        if score >= 85 and is_robust:
            badge_cls, verdict = "badge-pass", "PASS — High Confidence"
            card_cls = "card-pass"
        elif score >= 60 and is_robust:
            badge_cls, verdict = "badge-warn", "PASS — Moderate Confidence"
            card_cls = "card-pass"
        else:
            badge_cls, verdict = "badge-block", "BLOCKED — Critical Issues Detected"
            card_cls = "card-block"

        st.markdown(f"""
        <div class="{card_cls}" style="display:flex; align-items:center;
                    justify-content:space-between; padding:0.8rem 1.2rem; margin-bottom:1rem;">
          <span style="font-size:1.15rem; font-weight:700; color:inherit;">{verdict}</span>
          <span class="badge {badge_cls}" style="font-size:1.05rem;">{int(score)} / 100</span>
        </div>""", unsafe_allow_html=True)

        # ── Row 1: risk & model quality ───────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Confidence Score", f"{int(score)}/100")

        _var_ci   = q.get("var_ci")
        _var_cf   = q.get("var_cf")
        _var_help = None
        if _var_ci and all(map(lambda x: x == x, _var_ci)):   # not NaN
            _var_help = (
                f"Bootstrap 95% CI: [{_var_ci[0]:.2%}, {_var_ci[1]:.2%}]\n"
                f"（左尾比右尾更宽 → 下行风险非对称）\n"
                + (f"Cornish-Fisher 调整值: {_var_cf:.2%}" if _var_cf is not None else "")
            )
        _es = q.get("es_5pct")
        _es_help = (
            "Expected Shortfall (CVaR) at 5% — 最坏5%情形下的平均日损失。\n"
            "比 VaR 更完整地刻画尾部风险：VaR 告诉你亏损上限，ES 告诉你突破上限后平均有多惨。\n"
            "Basel III 以 ES 取代 VaR 作为风险资本标准。"
        ) if _es is not None else None
        m2.metric("Dynamic VaR", f"{q.get('d_var', 0):.2%}", help=_var_help)
        m2.metric("ES (CVaR 5%)", f"{_es:.2%}" if _es is not None else "—", help=_es_help)

        p_noise = q.get("p_noise", 0)
        active  = q.get("active", 0)
        m3.metric("P-hacking Risk", f"{p_noise:.1%}",
                  delta="↑ Elevated" if p_noise > 0.3 else "✓ Normal", delta_color="inverse")
        m4.metric("Active Factors", str(active),
                  delta="↓ Sparse" if active < 2 else "✓ Sufficient", delta_color="inverse")

        # ── Row 2: return & volatility ────────────────────────────────────────
        st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        a_ret = q.get("a_ret")
        a_vol = q.get("a_vol")
        r1.metric("Ann. Return",
                  f"{a_ret:+.2%}" if a_ret is not None else "—",
                  delta=("↑ Positive" if a_ret > 0 else "↓ Negative") if a_ret is not None else None,
                  delta_color="normal")
        r2.metric("Ann. Volatility", f"{a_vol:.2%}" if a_vol is not None else "—")

        _sr_ci   = q.get("sharpe_ci")
        _skew    = q.get("skewness")
        _kurt    = q.get("excess_kurt")
        _sr_help = None
        if _sr_ci and all(map(lambda x: x == x, _sr_ci)):
            _skew_note = (
                f"收益左偏 → 下行尾部风险较大" if _skew is not None and _skew < -0.1
                else ("收益右偏 → 正向肥尾" if _skew is not None and _skew > 0.1
                      else "收益近似对称")
            )
            _sr_help = (
                f"Bootstrap 95% CI: [{_sr_ci[0]:.2f}, {_sr_ci[1]:.2f}]\n"
                f"（SR > 0 时上界比下界离中心更远 → 右偏抽样分布）\n"
                f"偏度: {_skew:.3f}  超额峰度: {_kurt:.3f}\n{_skew_note}"
            ) if _skew is not None else (
                f"Bootstrap 95% CI: [{_sr_ci[0]:.2f}, {_sr_ci[1]:.2f}]"
            )
        _sharpe_val = q.get("sharpe")
        r3.metric(
            "Sharpe Ratio",
            f"{_sharpe_val:.2f}" if _sharpe_val is not None else "—",
            help=_sr_help,
        )

        # ── Row 3: Lasso model quality (temporal split) ───────────────────────
        val_r2  = q.get("val_r2")
        test_r2 = q.get("test_r2")
        if val_r2 is not None:
            st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)
            lq1, lq2, lq3 = st.columns(3)
            lq1.metric(
                "Lasso In-Sample R²",
                f"{val_r2:.4f}",
                help="训练集 R²（前80%时间序列数据），反映模型拟合质量",
            )
            lq2.metric(
                "Lasso Out-of-Sample R²",
                f"{test_r2:.4f}",
                delta="⚠ 泛化弱于均值" if test_r2 < 0 else "✓ 正向泛化",
                delta_color="inverse",
                help="留出集 R²（后20%时间序列数据）。负值表示模型泛化能力弱于无条件均值，在金融数据中常见，应解读为信号微弱而非模型崩溃",
            )
            lq3.metric(
                "Overfit Gap",
                f"{val_r2 - test_r2:+.4f}",
                delta="↑ 过拟合明显" if (val_r2 - test_r2) > 0.05 else "✓ 过拟合可控",
                delta_color="inverse",
                help="样本内 - 样本外 R² 差值。差值越大表示过拟合越严重",
            )

        # ── Quota / engine error detection ───────────────────────────────────
        _ai_texts = [s.get("audit_memo", ""), s.get("red_team_critique", ""),
                     s.get("technical_report", ""), s.get("alternative_suggestion", "")]
        _has_ai_error = any(_audit_output_has_error(t) for t in _ai_texts if t)

        if _has_ai_error:
            st.markdown(
                f'<div class="decision-card">{_audit_quota_error_html(s.get("target_assets",""))}</div>',
                unsafe_allow_html=True,
            )
        elif not is_robust:
            # ── BLOCKED ───────────────────────────────────────────────────────
            _section_label("Red Team Findings")
            with st.container(border=True):
                _render_memo_sections(
                    s.get("red_team_critique", "No detail available."),
                    accent="#DC2626",
                )
            st.info("⚠ 建议：检查数据是否存在前瞻性偏差，或降低模型复杂度后重新提交审计。")

            if s.get("alternative_suggestion"):
                _section_label("Reflection Engine  ·  Alternative Recommendation")
                with st.container(border=True):
                    _render_memo_sections(s["alternative_suggestion"], accent="#7C3AED")

        else:
            # ── PASSED ────────────────────────────────────────────────────────
            _section_label("Executive Decision Memo")
            with st.container(border=True):
                _render_memo_sections(s.get("audit_memo", ""), accent="#059669")

            _section_label("Red Team Sign-off")
            with st.container(border=True):
                _render_memo_sections(
                    s.get("red_team_critique", "No material issues identified."),
                    accent="#059669",
                )

            _section_label("Technical Audit Detail")
            with st.container(border=True):
                _render_memo_sections(
                    s.get("technical_report", "Technical audit data unavailable."),
                    accent="#475569",
                )

        # News context
        news_ctx = s.get("news_context", "")
        if news_ctx and news_ctx != "暂无近48小时相关新闻。":
            with st.expander("News Context  ·  Injected into Red Team Audit", expanded=False):
                _render_news_feed(news_ctx)

        # ── Lasso Factor Attribution ──────────────────────────────────────────
        with st.expander("Quantitative Backing  ·  Lasso Factor Attribution", expanded=False):
            if q.get("X") is not None and q.get("coefs") is not None:
                import numpy as _np
                _feat_names = list(q["X"].columns)
                _coefs      = list(q["coefs"])
                _rows = sorted(
                    [{"Factor": f, "Coefficient": round(c, 4)} for f, c in zip(_feat_names, _coefs)],
                    key=lambda x: abs(x["Coefficient"]), reverse=True,
                )
                _active   = [r for r in _rows if r["Coefficient"] != 0]
                _inactive = [r for r in _rows if r["Coefficient"] == 0]

                _is_dark = _theme.is_dark()
                _pos_col = "#34D399" if _is_dark else "#059669"
                _neg_col = "#F87171" if _is_dark else "#DC2626"
                _zero_col = "#6B7280"
                _bg      = "var(--card)"
                _bd      = "var(--border)"
                _txt     = "var(--text)"
                _mut     = "var(--muted)"

                st.markdown(
                    f'<div style="font-size:0.82rem;color:{_mut};margin-bottom:0.6rem;">'
                    f'Active features: <strong>{len(_active)}</strong> / {len(_rows)} total  ·  '
                    f'Sparsity: <strong>{q.get("sparsity", 0):.0%}</strong>  ·  '
                    f'P-hacking risk: <strong>{q.get("p_noise", 0):.1%}</strong>'
                    f'</div>', unsafe_allow_html=True,
                )

                if _active:
                    _max_abs = max(abs(r["Coefficient"]) for r in _active) or 1
                    for r in _active:
                        _c   = r["Coefficient"]
                        _col = _pos_col if _c > 0 else _neg_col
                        _bar_w = int(abs(_c) / _max_abs * 120)
                        _dir   = "▲" if _c > 0 else "▼"
                        # Clean up feature name for display
                        _label = r["Factor"].replace("_lag1", " (lag1)").replace("_", " ")
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:0.7rem;'
                            f'padding:0.28rem 0;border-bottom:1px solid {_bd};">'
                            f'<span style="width:220px;font-size:0.85rem;color:{_txt};'
                            f'font-family:monospace;flex-shrink:0;">{_label}</span>'
                            f'<div style="width:{_bar_w}px;height:10px;background:{_col};'
                            f'border-radius:2px;flex-shrink:0;"></div>'
                            f'<span style="font-size:0.85rem;color:{_col};font-weight:600;">'
                            f'{_dir} {abs(_c):.4f}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No active features — model is fully sparse.")

                if _inactive:
                    with st.expander(f"Zeroed-out features ({len(_inactive)})", expanded=False):
                        st.caption("  ·  ".join(
                            r["Factor"].replace("_lag1", "(lag1)") for r in _inactive
                        ))
            else:
                st.caption("Factor data unavailable — research node may not have run.")

        # ── Reflection Chain (blocked audits only) ────────────────────────────
        _chain = s.get("reflection_chain", "")
        if _chain:
            with st.expander("Reflection Chain  ·  Full Reasoning Audit Trail", expanded=False):
                _render_memo_sections(_chain, accent="#7C3AED")

        # ── PDF Export ────────────────────────────────────────────────────────
        pdf_bytes_key = f"audit_pdf_bytes_{choice}"
        if st.button("📄  Prepare Full PDF Report", key=f"prepare_audit_pdf_{choice}",
                     width='stretch'):
            with st.spinner(f"Compiling audit report for {choice}..."):
                # Assemble all audit sections into one document (no extra AI call needed)
                if is_robust:
                    pdf_text = "\n\n".join(filter(None, [
                        s.get("audit_memo", ""),
                        s.get("red_team_critique", ""),
                        s.get("technical_report", ""),
                    ]))
                else:
                    pdf_text = "\n\n".join(filter(None, [
                        s.get("red_team_critique", ""),
                        s.get("alternative_suggestion", ""),
                    ]))
                st.session_state[pdf_bytes_key] = generate_pdf_report(
                    pdf_text,
                    f"Quant Audit Report — {choice}",
                    {
                        "Asset Bundle": choice,
                        "VIX": vix_input,
                        "Verdict": verdict,
                        "Confidence Score": f"{int(score)}/100",
                    },
                )
        if st.session_state.get(pdf_bytes_key):
            st.download_button(
                "↓  Download PDF Report",
                st.session_state[pdf_bytes_key],
                f"Quant_Audit_{choice}.pdf",
                mime="application/pdf",
                width='stretch',
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Alpha Scanner
# ─────────────────────────────────────────────────────────────────────────────

def render_tab5(vix_input: float) -> None:
    _section_label("Alpha Scanner  ·  18-Sector Global Coverage")

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    already_today = st.session_state.get("scan_date") == today_str

    top_bar_l, top_bar_r = st.columns([2, 1], gap="medium")
    with top_bar_l:
        run = st.button("▶  Run Full Market Scan  (16 sectors)", type="primary",
                        width='stretch', key="scan_btn_tab5")
    with top_bar_r:
        scan = st.session_state.get("scan_results")
        if scan:
            n = len(scan.get("rankings", []))
            st.markdown(f"""
            <div style="background:var(--success-lt); border:1px solid var(--success-bd);
                        border-radius:3px; padding:0.45rem 0.8rem; text-align:center;">
              <span style="font-size:1.05rem; font-weight:700; color:var(--success);
                           text-transform:uppercase; letter-spacing:0.08em;">
                ✓  {n} sectors scanned
              </span>
            </div>""", unsafe_allow_html=True)

    if run and already_today:
        st.info("今日扫描已完成，直接展示缓存结果。如需重新扫描请点击 ↺ Refresh。")
    elif run:
        with st.spinner("Scanning 16 global sectors · Computing Sharpe, momentum, fund flow..."):
            try:
                result = MarketScanner().run_daily_scan()
                st.session_state.scan_results = result
                st.session_state.scan_date    = today_str
                st.session_state.pop("ai_scan_analysis", None)
                st.session_state.pop("ai_scan_news_ctx", None)
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"Scanner failed: {e}")

    scan = st.session_state.get("scan_results")
    if not scan:
        _s_bg = "rgba(255,255,255,0.04)" if _theme.is_dark() else "#F8FAFC"
        _s_bd = "rgba(255,255,255,0.10)" if _theme.is_dark() else "#CBD5E1"
        st.markdown(f"""
        <div style="background:{_s_bg}; border:1px dashed {_s_bd}; border-radius:3px;
                    padding:3rem; text-align:center; color:var(--muted); font-size:1.05rem;
                    margin-top:1rem;">
          Run the scan to discover the highest-quality sector opportunity across 16 global ETFs.
        </div>""", unsafe_allow_html=True)
        return

    # ── Timestamp + Refresh row ───────────────────────────────────────────────
    _scan_date = st.session_state.get("scan_date", "—")
    _col_ts, _col_ref = st.columns([4, 1], gap="small")
    with _col_ts:
        st.markdown(
            f'<div style="font-size:0.88rem; color:var(--muted); padding:0.3rem 0;">'
            f'Last scan: {_scan_date}  ·  今日缓存有效，刷新页面不会重新运行</div>',
            unsafe_allow_html=True,
        )
    with _col_ref:
        if st.button("↺  Refresh", key="refresh_scanner", width='stretch'):
            _clear_scanner_cache()
            st.rerun()

    best = scan["best"]
    rankings = scan["rankings"]

    # ── Champion Card (dark HTML) ─────────────────────────────────────────────
    metrics_html = "".join(f"""
      <div>
        <div class="label">{lbl}</div>
        <div class="value">{val}</div>
      </div>""" for lbl, val in [
        ("Sharpe Ratio",    f"{best['sharpe']:.2f}"),
        ("Momentum (60d)",  f"{best['momentum']:+.1%}"),
        ("Market Fit",      f"{best['market_fit']:.0f}%"),
        ("Fund Flow",       f"{best['fund_flow']:.2f}x"),
    ])

    st.markdown(f"""
    <div class="champion-card">
      <div style="font-size:1.05rem; text-transform:uppercase; letter-spacing:0.12em;
                  opacity:0.45; margin-bottom:0.3rem;">▲  Top-Ranked Sector</div>
      <div class="name">{best['name']}</div>
      <div class="ticker">{best['ticker']}  ·  Last Price  ${best.get('last_price', 0):.2f}</div>
      <div style="display:flex; gap:2.5rem; margin-top:1.2rem;">{metrics_html}</div>
    </div>""", unsafe_allow_html=True)

    # ── Rankings Table + Chart ────────────────────────────────────────────────
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        _section_label("Full Rankings  ·  Sorted by Sharpe Ratio")

        rows = []
        for r in rankings:
            sharpe_val = r["sharpe"]
            if sharpe_val >= 1.5:
                indicator = "🟢"
            elif sharpe_val >= 0.5:
                indicator = "🟡"
            else:
                indicator = "🔴"
            rows.append({
                "#":      r["rank"],
                "板块":   r["name"],
                "ETF":    r["ticker"],
                "Sharpe": sharpe_val,
                "动能":   f"{r['momentum']:+.1%}",
                "资金":   f"{r['fund_flow']:.2f}x",
                "契合度": f"{r['market_fit']:.0f}%",
                "波动率": f"{r['volatility']:.1%}",
                " ":      indicator,
            })
        df_rank = pd.DataFrame(rows)
        st.dataframe(
            df_rank,
            width='stretch',
            hide_index=True,
            column_config={
                "#":      st.column_config.NumberColumn(width="small"),
                "Sharpe": st.column_config.NumberColumn(format="%.2f", width="small"),
                " ":      st.column_config.TextColumn(width="small"),
            },
        )

    with right_col:
        _section_label("Sharpe Ratio  ·  Cross-sector Comparison")
        chart_df = (
            pd.DataFrame({"板块": [r["name"] for r in rankings],
                          "Sharpe": [r["sharpe"] for r in rankings]})
            .sort_values("Sharpe")
        )
        st.bar_chart(chart_df, x="板块", y="Sharpe", height=420)

    # ── Fetch news once → feed AI commentary + headlines expander ─────────────
    if "ai_scan_analysis" not in st.session_state:
        with st.spinner("Fetching headlines and generating quantitative commentary..."):
            perceiver = NewsPerceiver(
                av_key=st.secrets.get("AV_KEY", ""),
                gnews_key=st.secrets.get("GNEWS_KEY", ""),
            )
            news_ctx = perceiver.build_context(best["name"], best["ticker"], n=6,
                                               macro_regime=_infer_macro_regime(vix_input))
            st.session_state.ai_scan_news_ctx = news_ctx
            # Inject scanner history for this sector
            hist_ctx = get_historical_context("scanner", sector_name=best["name"], n=5)
            augmented_ctx = (hist_ctx + "\n\n" + news_ctx) if hist_ctx else news_ctx
            analysis = generate_ai_reasons(best["name"], best, augmented_ctx, vix_val=vix_input)
            st.session_state.ai_scan_analysis = analysis
            if not _is_ai_error(analysis):
                # Parse XAI block for confidence_score
                _scanner_conf = None
                _xai_conf = re.search(r"overall_confidence:\s*(\d+)", analysis)
                if _xai_conf:
                    _scanner_conf = int(_xai_conf.group(1))
                save_decision(
                    tab_type="scanner",
                    ai_conclusion=analysis,
                    vix_level=vix_input,
                    sector_name=best["name"],
                    ticker=best["ticker"],
                    news_summary=news_ctx[:500],
                    quant_metrics={
                        "sharpe":      best.get("sharpe"),
                        "ann_return":  best.get("ann_return"),
                        "momentum":    best.get("momentum"),
                        "mom_1m":      best.get("mom_1m"),
                        "mom_3m":      best.get("mom_3m"),
                        "mom_6m":      best.get("mom_6m"),
                        "market_fit":  best.get("market_fit"),
                        "fund_flow":   best.get("fund_flow"),
                    },
                    confidence_score=_scanner_conf,
                    macro_regime=_infer_macro_regime(vix_input),
                    horizon="季度(3个月)",
                )
            # Persist to daily cache so the result survives page refresh
            _save_scanner_cache(scan, analysis, news_ctx)
            st.session_state.scan_date = today_str

    # ── AI Commentary ─────────────────────────────────────────────────────────
    with st.expander("Quantitative Commentary  ·  AI Analyst", expanded=True):
        st.markdown(
            f'<div class="decision-card" style="font-size:1.2rem; line-height:1.75;">'
            f'{st.session_state.get("ai_scan_analysis", "")}</div>',
            unsafe_allow_html=True,
        )

    # ── Latest Headlines for Champion ────────────────────────────────────────
    with st.expander(f"Latest Headlines  ·  {best['name']}  ({best['ticker']})", expanded=False):
        news_ctx_display = st.session_state.get("ai_scan_news_ctx", "")
        if news_ctx_display and news_ctx_display != "暂无近48小时相关新闻。":
            _render_news_feed(news_ctx_display)
        else:
            st.caption("No recent headlines found in the last 48 hours.")



# ─────────────────────────────────────────────────────────────────────────────
# Tab 6 — Alpha Memory  ·  Decision Performance & Self-Reflection
# ─────────────────────────────────────────────────────────────────────────────

def render_tab6() -> None:
    _section_label("Alpha Memory  ·  Decision Performance & Self-Reflection")

    stats = get_stats()
    total_logged   = stats.get("total_logged", 0)
    total_verified = stats.get("total_verified", 0)

    # ── Top metrics ───────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Decisions Logged", total_logged)
    c2.metric("Verified",         total_verified)
    if total_verified > 0:
        hit_str = f"{stats['overall_hit_rate']:.0%}"
        baseline_delta = stats['overall_hit_rate'] - BASELINE_HIT_RATE
        c3.metric("Overall Hit Rate", hit_str,
                  delta=f"{baseline_delta:+.0%} vs random baseline",
                  delta_color="normal")
    else:
        c3.metric("Overall Hit Rate", "—", delta="No verified decisions yet")

    # ── Baseline reference ────────────────────────────────────────────────────
    _bm_bg  = "rgba(255,255,255,0.04)" if _theme.is_dark() else "#F8FAFC"
    _bm_bd  = "rgba(255,255,255,0.08)" if _theme.is_dark() else "#E2E8F0"
    _bm_txt = "var(--muted)"
    _bm_val = "var(--text)"
    st.markdown(f"""
    <div style="background:{_bm_bg}; border:1px solid {_bm_bd}; border-radius:3px;
                padding:0.6rem 1rem; font-size:0.92rem; color:{_bm_txt}; margin:0.5rem 0 1rem;">
      Performance Benchmarks:
      <span style="margin:0 1rem;">
        Random baseline <strong style="color:{_bm_val};">{BASELINE_HIT_RATE:.0%}</strong>
      </span>
      <span style="margin:0 1rem;">
        Min. acceptable <strong style="color:var(--warn);">{MIN_ACCEPTABLE:.0%}</strong>
      </span>
      <span style="margin:0 1rem;">
        Excellent <strong style="color:var(--success);">{EXCELLENT:.0%}</strong>
      </span>
      <span style="color:var(--muted);">
        · Accuracy scored against 20-day ETF price return (medium-term horizon)
      </span>
    </div>""", unsafe_allow_html=True)

    # ── Admin shortcut ────────────────────────────────────────────────────────
    unapplied = stats.get("unapplied_patterns", 0)
    _adm_bg = "rgba(76,142,247,0.08)" if _theme.is_dark() else "#F0F9FF"
    _adm_bd = "rgba(76,142,247,0.25)" if _theme.is_dark() else "#BAE6FD"
    _adm_fg = "var(--accent)"
    st.markdown(
        f'<div style="background:{_adm_bg}; border:1px solid {_adm_bd}; border-radius:3px; '
        f'padding:0.8rem 1.2rem; font-size:1rem; color:{_adm_fg}; margin-bottom:0.5rem;">'
        f'Verification, Meta-Agent analysis, and learning pattern management have moved to '
        f'<b>Admin Panel</b> (left sidebar navigation).'
        f'{f" &nbsp;·&nbsp; <b>{unapplied} pattern(s)</b> pending review." if unapplied else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if total_verified == 0:
        _e_bg = "rgba(255,255,255,0.04)" if _theme.is_dark() else "#F8FAFC"
        _e_bd = "rgba(255,255,255,0.10)" if _theme.is_dark() else "#CBD5E1"
        _e_fg = "var(--muted)"
        st.markdown(f"""
        <div style="background:{_e_bg}; border:1px dashed {_e_bd}; border-radius:3px;
                    padding:2.5rem; text-align:center; color:{_e_fg}; font-size:1.05rem;">
          No verified decisions yet. Run analyses across the tabs — results will be
          automatically verified after 5 market days.
        </div>""", unsafe_allow_html=True)
        return

    # ── By-tab breakdown ──────────────────────────────────────────────────────
    _section_label("Performance by Analysis Module")
    tab_labels = {"macro": "Macro", "sector": "Sector Risk",
                  "audit": "Quant Audit", "scanner": "Alpha Scanner"}
    by_tab = stats.get("by_tab", {})
    if by_tab:
        def _rating(hr: float) -> str:
            if hr >= EXCELLENT:      return "EXCELLENT"
            if hr >= MIN_ACCEPTABLE: return "ACCEPTABLE"
            return "WEAK"

        mod_rows = []
        for tab_key, data in by_tab.items():
            hr = data["hit_rate"]
            mod_rows.append({
                "Module":       tab_labels.get(tab_key, tab_key),
                "Hit Rate":     f"{hr:.0%}",
                "Verified":     data["count"],
                "Avg Score":    f"{data['avg']:.2f}",
                "Rating":       _rating(hr),
            })
        st.dataframe(
            pd.DataFrame(mod_rows),
            width='stretch',
            hide_index=True,
            column_config={
                "Hit Rate":  st.column_config.TextColumn(width="small"),
                "Verified":  st.column_config.NumberColumn(width="small"),
                "Avg Score": st.column_config.TextColumn(width="small"),
                "Rating":    st.column_config.TextColumn(width="medium"),
            },
        )

    # ── Decision history table ────────────────────────────────────────────────
    history = stats.get("history", [])
    if history:
        _section_label("Verified Decision Log  ·  Sector & Scanner")
        rows = []
        for d in history:
            score = d["score"] or 0
            flag  = "✅" if score >= EXCELLENT else ("⚠️" if score >= 0.5 else "❌")
            ret   = f"{d['return_5d']:+.1%}" if d["return_5d"] is not None else "—"
            rows.append({
                "Date":      d["date"],
                "Module":    tab_labels.get(d["tab"], d["tab"]),
                "Sector":    d["sector"],
                "Direction": d["direction"],
                "5D Return": ret,
                "Score":     f"{score:.2f}",
                " ":         flag,
            })
        st.dataframe(
            pd.DataFrame(rows),
            width='stretch', hide_index=True,
            column_config={
                "Score": st.column_config.NumberColumn(format="%.2f", width="small"),
                " ":     st.column_config.TextColumn(width="small"),
            },
        )

        # ── Revision chain ────────────────────────────────────────────────────
        chains = get_revision_chains()
        if chains:
            _section_label("修订历史  ·  Decision Revision Log")
            _is_dk = _theme.is_dark()
            _ch_bg   = "rgba(255,255,255,0.03)" if _is_dk else "#F8FAFC"
            _ch_bd   = "rgba(255,255,255,0.10)" if _is_dk else "#E2E8F0"
            _arr_col = "#60A5FA" if _is_dk else "#2563EB"
            _chg_col = "#FBBF24" if _is_dk else "#D97706"
            _same_col= "#34D399" if _is_dk else "#059669"
            _mut     = "var(--muted)"
            _txt     = "var(--text)"

            for c in chains:
                _dir_changed = c["direction_changed"]
                _arrow_html  = (
                    f'<span style="color:{_chg_col}; font-weight:800;">⟳ 方向改变</span>'
                    if _dir_changed else
                    f'<span style="color:{_same_col}; font-weight:600;">✓ 方向维持</span>'
                )
                st.markdown(
                    f'<div style="background:{_ch_bg}; border:1px solid {_ch_bd}; '
                    f'border-radius:6px; padding:0.7rem 1rem; margin-bottom:0.5rem;">'
                    f'<div style="display:flex; align-items:center; gap:0.8rem; flex-wrap:wrap;">'

                    # 原始决策
                    f'<div style="font-size:0.9rem;">'
                    f'<span style="color:{_mut}; font-size:0.78rem;">原始 #{c["parent_id"]} · {c["parent_date"]}</span><br>'
                    f'<span style="color:{_txt}; font-weight:600;">{c["sector"]}</span>'
                    f'<span style="color:{_mut}; margin-left:0.4rem;">{c["parent_direction"]}</span>'
                    f'</div>'

                    # 箭头
                    f'<span style="color:{_arr_col}; font-size:1.2rem;">→</span>'

                    # 修订决策
                    f'<div style="font-size:0.9rem;">'
                    f'<span style="color:{_mut}; font-size:0.78rem;">修订 #{c["revision_id"]} · {c["revision_date"]}</span><br>'
                    f'<span style="color:{_txt}; font-weight:700;">{c["revision_direction"]}</span>'
                    f'<span style="margin-left:0.6rem;">{_arrow_html}</span>'
                    f'</div>'

                    f'</div>'
                    f'<div style="font-size:0.82rem; color:{_mut}; margin-top:0.35rem;">'
                    f'触发原因：{c["revision_reason"]}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Self-reflections
        reflections = [d for d in history if d.get("reflection")]
        if reflections:
            with st.expander("AI Self-Reflections  ·  Lessons Learned", expanded=False):
                for d in reflections:
                    score = d["score"] or 0
                    flag  = "✅" if score >= EXCELLENT else ("⚠️" if score >= 0.5 else "❌")
                    ret   = f"{d['return_5d']:+.1%}" if d["return_5d"] is not None else "—"
                    st.markdown(
                        f'<div style="padding:0.6rem 0; border-bottom:1px solid var(--border);">'
                        f'<span style="font-size:0.88rem; color:var(--muted);">'
                        f'{d["date"]} · {d["sector"]} · {d["direction"]} · 5D={ret} {flag}</span><br>'
                        f'<span style="font-size:0.95rem; color:var(--text); line-height:1.6;">'
                        f'{d["reflection"]}</span></div>',
                        unsafe_allow_html=True,
                    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Invalidation Monitor
# ══════════════════════════════════════════════════════════════════════════════

def render_tab7() -> None:
    _section_label("Invalidation Monitor  ·  Live Decision Watchlist")

    pending = get_pending_decisions_for_monitor()

    if not pending:
        _is_dark = _theme.is_dark()
        _e_bg = "rgba(255,255,255,0.03)" if _is_dark else "#F8FAFC"
        _e_bd = "rgba(255,255,255,0.08)" if _is_dark else "#CBD5E1"
        _e_fg = "rgba(255,255,255,0.35)" if _is_dark else "#94A3B8"
        st.markdown(
            f'<div style="background:{_e_bg}; border:1px dashed {_e_bd}; border-radius:6px; '
            f'padding:2.5rem; text-align:center; color:{_e_fg}; font-size:1.05rem;">'
            f'暂无待验证的 Clean Zone 决策。运行各分析模块后，决策将在到达验证窗口前显示于此。'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Summary strip ─────────────────────────────────────────────────────────
    n_overdue    = sum(1 for d in pending if d["urgency"] == "overdue")
    n_approaching = sum(1 for d in pending if d["urgency"] == "approaching")
    n_normal     = sum(1 for d in pending if d["urgency"] == "normal")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("待验证决策", len(pending))
    c2.metric("已过期限", n_overdue,
              delta="需关注" if n_overdue > 0 else None,
              delta_color="inverse" if n_overdue > 0 else "off")
    c3.metric("即将到期 (≤14天)", n_approaching,
              delta="检查失效条件" if n_approaching > 0 else None,
              delta_color="inverse" if n_approaching > 0 else "off")
    c4.metric("正常追踪", n_normal)

    st.divider()

    # ── Legend ────────────────────────────────────────────────────────────────
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

    # ── Decision cards ────────────────────────────────────────────────────────
    _URGENCY_COLOR = {
        "overdue":    "#EF4444",
        "approaching": "#F59E0B",
        "normal":     "#10B981",
    }
    _URGENCY_LABEL = {
        "overdue":    "已过期限",
        "approaching": "即将到期",
        "normal":     "正常追踪",
    }
    _DIR_COLOR = {
        "超配": "#10B981",
        "低配": "#EF4444",
        "标配": "#60A5FA",
        "拦截": "#F59E0B",
        "通过": "#A78BFA",
    }
    _TAB_LABEL = {
        "sector":  "Sector Risk",
        "audit":   "Quant Audit",
        "scanner": "Alpha Scanner",
        "macro":   "Macro",
    }

    for d in pending:
        _uc   = _URGENCY_COLOR.get(d["urgency"], "#6B7280")
        _ul   = _URGENCY_LABEL.get(d["urgency"], d["urgency"])
        _dc   = _DIR_COLOR.get(d["direction"], "var(--muted)")
        _tl   = _TAB_LABEL.get(d["tab_type"], d["tab_type"])
        _conf = f"{d['confidence_score']}%" if d["confidence_score"] is not None else "—"
        _inv  = d["invalidation_conditions"] or "未记录"
        _days_str = (
            f"已逾期 {-d['days_to_deadline']} 天"
            if d["days_to_deadline"] < 0
            else f"剩余 {d['days_to_deadline']} 天"
        )
        _drift_badge = (
            '<span style="font-size:0.7rem; font-weight:700; color:#F59E0B; '
            'background:#F59E0B22; padding:0.1rem 0.5rem; border-radius:3px; '
            'margin-left:0.4rem;">制度已漂移</span>'
            if d.get("regime_drifted") else ""
        )

        _pre_label = d.get("human_label", "")
        _pre_label_display = {
            "pre_strong":    "逻辑清晰",
            "pre_uncertain": "有疑虑",
            "pre_poor":      "明显缺陷",
        }.get(_pre_label, "")
        _pre_badge = (
            f'<span style="font-size:0.7rem; font-weight:700; color:#A78BFA; '
            f'background:#A78BFA22; padding:0.1rem 0.5rem; border-radius:3px; '
            f'margin-left:0.4rem;">人工预评: {_pre_label_display}</span>'
            if _pre_label_display else ""
        )

        st.markdown(
            f'<div style="border-left:3px solid {_uc}; border:1px solid {_uc}33; '
            f'border-left:4px solid {_uc}; border-radius:6px; '
            f'padding:0.9rem 1.2rem; margin-bottom:0.7rem;">'

            # Header row
            f'<div style="display:flex; justify-content:space-between; '
            f'align-items:center; margin-bottom:0.5rem;">'
            f'<div style="display:flex; gap:0.8rem; align-items:center; flex-wrap:wrap;">'
            f'<span style="font-size:1rem; font-weight:700; color:var(--text);">'
            f'{d["sector_name"]}</span>'
            f'<span style="font-size:0.75rem; color:{_dc}; font-weight:700; '
            f'background:{_dc}22; padding:0.1rem 0.45rem; border-radius:3px;">'
            f'{d["direction"]}</span>'
            f'<span style="font-size:0.72rem; color:var(--muted); '
            f'background:var(--surface2); padding:0.1rem 0.45rem; border-radius:3px;">'
            f'{_tl}</span>'
            f'{_drift_badge}'
            f'{_pre_badge}'
            f'</div>'
            f'<span style="font-size:0.8rem; font-weight:700; color:{_uc};">'
            f'{_ul} · {_days_str}</span>'
            f'</div>'

            # Meta row
            f'<div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; '
            f'gap:0.4rem; margin-bottom:0.6rem;">'
            f'<div><span style="font-size:0.68rem; color:var(--muted); '
            f'text-transform:uppercase;">决策日期</span><br>'
            f'<span style="font-family:var(--mono); font-size:0.88rem;">'
            f'{d["decision_date"]}</span></div>'
            f'<div><span style="font-size:0.68rem; color:var(--muted); '
            f'text-transform:uppercase;">期限</span><br>'
            f'<span style="font-family:var(--mono); font-size:0.88rem;">'
            f'{d["deadline_date"]} ({d["horizon"]})</span></div>'
            f'<div><span style="font-size:0.68rem; color:var(--muted); '
            f'text-transform:uppercase;">置信度</span><br>'
            f'<span style="font-family:var(--mono); font-size:0.88rem;">'
            f'{_conf}</span></div>'
            f'<div><span style="font-size:0.68rem; color:var(--muted); '
            f'text-transform:uppercase;">宏观制度</span><br>'
            f'<span style="font-size:0.88rem;">{d["macro_regime"]}</span></div>'
            f'</div>'

            # Invalidation conditions
            f'<div style="background:var(--surface2); border-radius:4px; '
            f'padding:0.45rem 0.7rem;">'
            f'<span style="font-size:0.68rem; color:var(--muted); '
            f'text-transform:uppercase; letter-spacing:0.06em;">失效条件&nbsp;</span>'
            f'<span style="font-size:0.88rem; color:var(--text);">{_inv}</span>'
            f'</div>'

            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Pre-verification human annotation ─────────────────────────────────
        # User scores the decision thesis BEFORE the outcome is known.
        # Creates an independent signal correlatable against Triple-Barrier results
        # to assess whether human pre-scoring outperforms LCS — breaking the AI loop.
        _label_options = {
            "（未标注）":  "",
            "逻辑清晰":   "pre_strong",
            "有疑虑":     "pre_uncertain",
            "明显缺陷":   "pre_poor",
        }
        _current_display = {v: k for k, v in _label_options.items()}.get(
            _pre_label, "（未标注）"
        )
        _, _col_widget = st.columns([3, 1])
        with _col_widget:
            _selected = st.selectbox(
                "人工预评",
                options=list(_label_options.keys()),
                index=list(_label_options.keys()).index(_current_display),
                key=f"prelabel_{d['id']}",
                label_visibility="collapsed",
            )
            _new_val = _label_options[_selected]
            if _new_val != _pre_label:
                if _new_val:
                    set_human_label(d["id"], _new_val)
                elif _pre_label.startswith("pre_"):
                    # Clear pre-label back to NULL
                    with SessionFactory() as _s:
                        _rec = _s.get(DecisionLog, d["id"])
                        if _rec:
                            _rec.human_label = None
                            _s.commit()
                st.rerun()

