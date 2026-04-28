"""
Macro Alpha Pro — Circuit Breaker
熔断机制状态 · 交易周期历史 · 哈希链完整性 · Universe 管理
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import ui.theme as theme
from engine.memory import init_db

init_db()
theme.init_theme()

st.title("🔒 Circuit Breaker")
st.caption("系统安全机制 · 熔断状态 · 周期审批 · Universe 健康检查")

t_cb, t_cycle, t_chain, t_universe = st.tabs([
    "⚡ 熔断状态",
    "🔄 周期历史",
    "🔗 链完整性",
    "🌐 Universe",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Circuit Breaker Status
# ══════════════════════════════════════════════════════════════════════════════
with t_cb:
    st.subheader("熔断机制 · Circuit Breaker Status")

    try:
        from engine.circuit_breaker import (
            get_status as _cb_get_status, manual_reset as _cb_reset,
            LEVEL_NONE, LEVEL_LIGHT, LEVEL_MEDIUM, LEVEL_SEVERE,
        )
        cb_state = _cb_get_status()
        cb_colors = {
            LEVEL_NONE:   ("#10B981", "✅ 正常运行"),
            LEVEL_LIGHT:  ("#F59E0B", "⚠️ 轻度异常（数据源降级）"),
            LEVEL_MEDIUM: ("#F97316", "🟡 中度警戒（LLM 配额告急）"),
            LEVEL_SEVERE: ("#EF4444", "🔴 严重熔断（已暂停自动信号生成）"),
        }
        cb_color, cb_label = cb_colors.get(cb_state.level, ("#94A3B8", "未知"))

        st.markdown(
            f'<div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1); '
            f'border-left:4px solid {cb_color}; border-radius:8px; '
            f'padding:0.8rem 1.2rem; margin-bottom:1rem;">'
            f'<div style="font-weight:700; color:{cb_color}; margin-bottom:0.3rem;">{cb_label}</div>'
            f'<div style="font-size:0.82rem; color:rgba(255,255,255,0.5);">{cb_state.reason or "无异常"}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("熔断级别", str(cb_state.level))
        c2.metric("触发时间", str(cb_state.triggered_at) if hasattr(cb_state, "triggered_at") and cb_state.triggered_at else "—")
        c3.metric("配额使用", f"{cb_state.quota_frac:.0%}" if hasattr(cb_state, "quota_frac") and cb_state.quota_frac else "—")

        if cb_state.level == LEVEL_SEVERE:
            st.error(
                "⚠️ 当前处于严重熔断状态，所有自动信号生成已暂停。\n\n"
                "请人工确认市场情况后点击下方按钮恢复运行。"
            )
            reset_reason = st.text_input(
                "恢复理由（必填）", key="cb_reset_reason",
                placeholder="例：VIX 已回落至正常区间，确认非系统性风险",
            )
            if st.button("✅ 手动恢复运行", key="cb_manual_reset",
                         type="primary", disabled=not reset_reason.strip()):
                _cb_reset(reason=reset_reason.strip())
                st.success("熔断已解除，系统恢复自动运行。")
                st.rerun()
        elif cb_state.level == LEVEL_MEDIUM and hasattr(cb_state, "quota_frac") and cb_state.quota_frac:
            st.info(
                f"当日 API 配额已使用 {cb_state.quota_frac:.0%}，"
                "非核心 LLM 调用已暂停。配额次日自动重置后恢复。"
            )
        else:
            st.success("系统运行正常。所有自动化流程均处于激活状态。")

        # History
        try:
            from engine.memory import get_stress_test_history
            history = get_stress_test_history(limit=10)
            if history:
                st.divider()
                st.subheader("熔断历史记录")
                hist_df = pd.DataFrame(history)
                st.dataframe(hist_df, use_container_width=True, hide_index=True)
        except Exception:
            pass

    except ImportError:
        st.error("circuit_breaker 模块不可用。")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Trading Cycle History
# ══════════════════════════════════════════════════════════════════════════════
with t_cycle:
    st.subheader("交易周期历史 · Cycle Run Log")

    try:
        from engine.orchestrator import TradingCycleOrchestrator as _TCO
        tco = _TCO()

        # Pending gate approvals
        pending_gates = tco.get_pending_gates()
        if pending_gates:
            st.warning(f"⏳ {len(pending_gates)} 个周期等待人工审批")
            for pg in pending_gates:
                gate_labels = {
                    "analysis_draft":      "分析草稿审批",
                    "risk_approval":       "风控建议审批",
                    "monthly_rebalance":   "月度再平衡审批",
                    "covariance_override": "协方差覆盖审批",
                }
                gl = gate_labels.get(pg["gate"], pg["gate"])
                with st.expander(
                    f"Cycle #{pg['id']} · {pg['cycle_type']} · {pg['as_of_date']} · 等待：{gl}"
                ):
                    col_a, col_r = st.columns(2)
                    with col_a:
                        if st.button("✅ 批准", key=f"gate_approve_{pg['id']}"):
                            tco.approve_gate(pg["id"], approved=True, note="Circuit Breaker UI 批准")
                            st.success("已批准，执行层将继续运行。")
                            st.rerun()
                    with col_r:
                        if st.button("❌ 拒绝", key=f"gate_reject_{pg['id']}"):
                            tco.approve_gate(pg["id"], approved=False, note="Circuit Breaker UI 拒绝")
                            st.warning("已拒绝，本次周期终止。")
                            st.rerun()
        else:
            st.success("✅ 无等待审批的周期。")

        st.divider()

        # Recent cycle history
        recent = tco.get_recent_cycles(n=20)
        if recent:
            cyc_df = pd.DataFrame(recent)[[
                "id", "cycle_type", "as_of_date", "status", "gate", "elapsed_s", "started_at"
            ]]
            cyc_df.columns = ["ID", "类型", "日期", "状态", "闸门", "耗时(s)", "启动时间"]
            status_icons = {
                "completed":    "✅",
                "failed":       "❌",
                "running":      "⏳",
                "pending_gate": "🔒",
                "approved":     "✅✓",
                "rejected":     "🚫",
            }
            cyc_df["状态"] = cyc_df["状态"].map(lambda s: f"{status_icons.get(s, '')} {s}")
            st.dataframe(cyc_df, use_container_width=True, hide_index=True)
        else:
            st.info("暂无周期运行记录。首次调用 TradingCycleOrchestrator 后此处自动更新。")

    except ImportError:
        st.info("TradingCycleOrchestrator 模块不可用。")
    except Exception as e:
        st.error(f"周期历史加载失败: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Chain Integrity
# ══════════════════════════════════════════════════════════════════════════════
with t_chain:
    st.subheader("实验日志哈希链 · Chain Integrity")
    st.markdown(
        '<div style="font-size:0.82rem; color:rgba(255,255,255,0.5); margin-bottom:1rem;">'
        '每条 DecisionLog 记录的 SHA-256 哈希链接到前一条，形成可验证的完整性链。<br>'
        '<b>诚实声明</b>：整链重算时无法检测串改，是可信度信号而非安全机制。</div>',
        unsafe_allow_html=True,
    )

    if st.button("验证链完整性", key="verify_hash_chain", type="primary"):
        try:
            from engine.memory import verify_chain_integrity
            chain_ok, chain_broken, chain_total = verify_chain_integrity()
            if chain_total == 0:
                st.info("尚无带 chain_hash 的记录（chain_hash 从下次 save_decision() 起开始写入）。")
            elif chain_broken == 0:
                st.success(f"✅ 链完整 — 已验证 {chain_total} 条记录，0 条断链。")
            else:
                st.error(
                    f"❌ 发现 {chain_broken} 条断链（共 {chain_total} 条）。"
                    "可能原因：DB 直接修改、记录删除、迁移前旧记录无 chain_hash。"
                )
        except Exception as e:
            st.error(f"链验证失败: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Universe Management
# ══════════════════════════════════════════════════════════════════════════════
with t_universe:
    st.subheader("Universe 管理 · ETF 注册表")
    st.markdown(
        '<div style="font-size:0.82rem; color:rgba(255,255,255,0.5); margin-bottom:1rem;">'
        'P2-11 动态 Universe。批次 0=初始18 / 批次 1=批次A权益因子 / 批次 2=批次B跨资产。</div>',
        unsafe_allow_html=True,
    )

    try:
        from engine.universe_manager import get_active_universe, universe_health_check
        from engine.memory import SessionFactory
        from engine.universe_manager import UniverseETF as _UE

        with SessionFactory() as us:
            all_etfs = us.query(_UE).order_by(_UE.batch, _UE.id).all()

        active_n  = sum(1 for e in all_etfs if e.active)
        batch_a_n = sum(1 for e in all_etfs if e.batch == 1 and e.active)
        batch_b_n = sum(1 for e in all_etfs if e.batch == 2 and e.active)

        u1, u2, u3 = st.columns(3)
        u1.metric("活跃 ETF 总数", active_n)
        u2.metric("批次 A 活跃",   batch_a_n)
        u3.metric("批次 B 活跃",   batch_b_n)

        etf_df = pd.DataFrame([{
            "板块":     e.sector,
            "Ticker":   e.ticker,
            "资产类别": e.asset_class,
            "批次":     e.batch,
            "成立日":   str(e.inception_date) if e.inception_date else "—",
            "状态":     "✅ 活跃" if e.active else "❌ 停用",
            "停用日期": str(e.removed_at) if e.removed_at else "—",
        } for e in all_etfs])
        st.dataframe(etf_df, use_container_width=True, hide_index=True)

        if st.button("运行月度健康检查（ADV 审核）", key="universe_health_check"):
            with st.spinner("正在检查各 ETF 成交量…"):
                hr = universe_health_check()
            if hr.inactive_flagged:
                st.warning(f"已标记为 inactive：{', '.join(hr.inactive_flagged)}")
            else:
                st.success(f"全部 {active_n} 个 ETF ADV 达标，无需标记。")
            if hr.warnings:
                with st.expander("警告详情"):
                    for w in hr.warnings:
                        st.text(w)

    except Exception as ue:
        st.error(f"Universe Manager 加载失败: {ue}")
