"""
Macro Alpha Pro — Agent Decisions
增强版决策日志：完整决策生命周期追踪 + 失败归因 + Clean Zone 胜率
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime
import json
import streamlit as st
import pandas as pd

import ui.theme as theme
from engine.memory import (
    init_db, get_stats, get_clean_zone_stats, SessionFactory, DecisionLog,
    set_failure_attribution, get_failure_attribution_stats,
    _FAILURE_TYPE_LABELS, BASELINE_HIT_RATE, TRAIN_TEST_CUTOFF,
)

init_db()

# set_page_config handled by app.py via st.navigation()
# st.set_page_config(page_title="Agent Decisions | Macro Alpha Pro", page_icon="🧠", layout="wide")
theme.init_theme()

st.title("🧠 Agent Decisions")
st.caption("决策全生命周期追踪 · Triple-Barrier 验证 · 失败归因")

today = datetime.date.today()

_DIRECTION_COLORS = {
    "超配": "#22c55e",
    "标配": "#f59e0b",
    "低配": "#ef4444",
    "拦截": "#94a3b8",
    "通过": "#3b82f6",
    "中性": "#94a3b8",
}
_BARRIER_ICONS = {"tp": "✅", "sl": "❌", "time": "⏱"}
_FAILURE_TYPE_SHORT = {
    "hypothesis":   "假设失效",
    "data":         "数据问题",
    "regime_drift": "制度漂移",
    "robustness":   "稳健性",
    "evaluation":   "评估问题",
    "execution":    "执行偏差",
}

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("筛选")
    filter_tab = st.multiselect(
        "分析类型",
        ["macro", "sector", "audit", "scanner"],
        default=["sector", "macro"],
    )
    filter_verified = st.selectbox(
        "验证状态",
        ["全部", "已验证", "待验证"],
        index=0,
    )
    filter_direction = st.multiselect(
        "方向",
        ["超配", "标配", "低配", "拦截", "通过", "中性"],
        default=[],
    )
    filter_result = st.selectbox(
        "验证结果",
        ["全部", "正确(≥0.75)", "部分(0.5)", "失败(<0.5)", "未归因失败"],
        index=0,
    )
    limit = st.slider("显示条数", 20, 200, 50)

# ── Summary KPIs ───────────────────────────────────────────────────────────────
stats = get_stats()
cz    = get_clean_zone_stats()
fa    = get_failure_attribution_stats()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("总决策数",   stats.get("total_logged", 0))
k2.metric("已验证",     stats.get("total_verified", 0))
k3.metric("待验证",     stats.get("pending_verification", 0))
k4.metric("Clean Zone 胜率",
          f"{cz.get('accuracy_rate', 0):.0%}" if cz.get("n_verified", 0) > 0 else "—",
          f"n={cz.get('n_verified', 0)}")
k5.metric("随机基准",   f"{BASELINE_HIT_RATE:.0%}",
          delta_color="off")
k6.metric("待归因失败", fa.get("unattributed", 0),
          delta_color="inverse")

st.divider()

# ── Load decisions ─────────────────────────────────────────────────────────────
with SessionFactory() as sess:
    q = (
        sess.query(DecisionLog)
        .filter(DecisionLog.superseded == False)
    )
    if filter_tab:
        q = q.filter(DecisionLog.tab_type.in_(filter_tab))
    if filter_verified == "已验证":
        q = q.filter(DecisionLog.verified == True)
    elif filter_verified == "待验证":
        q = q.filter(DecisionLog.verified == False)
    if filter_direction:
        q = q.filter(DecisionLog.direction.in_(filter_direction))
    if filter_result == "正确(≥0.75)":
        q = q.filter(DecisionLog.accuracy_score >= 0.75)
    elif filter_result == "部分(0.5)":
        q = q.filter(DecisionLog.accuracy_score == 0.5)
    elif filter_result == "失败(<0.5)":
        q = q.filter(DecisionLog.accuracy_score < 0.5, DecisionLog.verified == True)
    elif filter_result == "未归因失败":
        q = q.filter(
            DecisionLog.accuracy_score < 0.5,
            DecisionLog.verified == True,
            DecisionLog.failure_type.is_(None),
        )
    records = (
        q.order_by(DecisionLog.created_at.desc())
        .limit(limit)
        .all()
    )

if not records:
    st.info("暂无符合条件的决策记录。")
    st.stop()

# ── Decision Cards ─────────────────────────────────────────────────────────────
st.subheader(f"决策记录（{len(records)} 条）")

for rec in records:
    # Determine card border color by result
    if rec.accuracy_score is not None:
        if rec.accuracy_score >= 0.75:
            border = "#22c55e"
        elif rec.accuracy_score == 0.5:
            border = "#f59e0b"
        else:
            border = "#ef4444"
    elif not rec.verified:
        border = "#3b82f6"
    else:
        border = "#64748b"

    direction_color = _DIRECTION_COLORS.get(rec.direction or "", "#94a3b8")

    with st.container(border=True):
        # Header row
        head_col, meta_col = st.columns([3, 2])

        with head_col:
            barrier_icon = _BARRIER_ICONS.get(rec.barrier_hit or "", "")
            acc_str = (
                f"  {barrier_icon} {rec.accuracy_score:.0%}"
                if rec.accuracy_score is not None else
                "  ⏳ 待验证"
            )
            drift_badge = " 🔄" if rec.regime_drifted else ""
            ft_badge = (
                f"  🏷️ {_FAILURE_TYPE_SHORT.get(rec.failure_type, rec.failure_type)}"
                if rec.failure_type else ""
            )
            title_col, jump_col = st.columns([5, 1])
            with title_col:
                st.markdown(
                    f'<span style="font-size:1.05rem;font-weight:700;">'
                    f'{rec.sector_name or "全球宏观"}</span>'
                    f'  <span style="color:{direction_color};font-weight:600;">'
                    f'{rec.direction or "—"}</span>'
                    f'<span style="color:#94a3b8;font-size:0.9rem;">'
                    f'{acc_str}{drift_badge}{ft_badge}</span>',
                    unsafe_allow_html=True,
                )
            with jump_col:
                if rec.tab_type == "sector" and rec.sector_name:
                    if st.button(
                        "重新分析",
                        key=f"reanalyze_{rec.id}",
                        help=f"跳转至 Trading Desk 重新分析 {rec.sector_name}",
                        use_container_width=True,
                    ):
                        st.session_state["audit_target_sync"] = rec.sector_name
                        st.switch_page("pages/trading_desk.py")

        with meta_col:
            date_str = rec.created_at.strftime("%Y-%m-%d") if rec.created_at else "—"
            st.markdown(
                f'<div style="color:var(--muted);font-size:0.85rem;text-align:right;">'
                f'{rec.tab_type}  ·  {date_str}<br>'
                f'置信度 <b>{rec.confidence_score or "—"}</b>/100  ·  '
                f'VIX {rec.vix_level or "—"}  ·  '
                f'{rec.macro_regime or "—"}</div>',
                unsafe_allow_html=True,
            )

        # Expandable details
        with st.expander("展开详情", expanded=False):
            d1, d2 = st.columns(2)

            with d1:
                if rec.economic_logic:
                    st.markdown("**经济逻辑**")
                    st.caption(rec.economic_logic[:400])
                if rec.invalidation_conditions:
                    st.markdown("**失效条件**")
                    st.caption(rec.invalidation_conditions[:300])

            with d2:
                # Quant metrics snapshot
                if rec.quant_p_noise is not None:
                    st.markdown("**量化指标快照**")
                    qc1, qc2, qc3 = st.columns(3)
                    qc1.metric("p_noise", f"{rec.quant_p_noise:.0%}")
                    qc2.metric("val_R²",  f"{rec.quant_val_r2:.2f}" if rec.quant_val_r2 else "—")
                    qc3.metric("test_R²", f"{rec.quant_test_r2:.2f}" if rec.quant_test_r2 else "—")

                # Triple-Barrier details
                if rec.verified and rec.barrier_hit:
                    st.markdown("**验证详情**")
                    b1, b2 = st.columns(2)
                    b1.metric(
                        "触碰障碍",
                        f"{_BARRIER_ICONS.get(rec.barrier_hit, '')} {rec.barrier_hit}",
                    )
                    b2.metric(
                        "持仓天数",
                        f"{rec.barrier_days} 天" if rec.barrier_days else "—",
                    )
                    if rec.barrier_return is not None:
                        st.metric("到达收益率", f"{rec.barrier_return:+.2%}")
                    if rec.llm_weight_alpha is not None:
                        _alpha_label = (
                            "LLM α 贡献"
                        )
                        _alpha_delta = (
                            "正贡献" if rec.llm_weight_alpha > 0
                            else ("负贡献" if rec.llm_weight_alpha < 0 else "持平")
                        )
                        st.metric(
                            _alpha_label,
                            f"{rec.llm_weight_alpha:+.4f}",
                            delta=_alpha_delta,
                            delta_color="normal",
                            help="llm_weight_alpha = 20日实际收益 × (主轨道权重 − 量化基准权重)",
                        )

            # Failure attribution inline editor (only for unattributed failures)
            if (rec.accuracy_score is not None and rec.accuracy_score < 0.5
                    and rec.verified and rec.failure_type is None):
                st.divider()
                st.markdown(
                    "⚠️ **失败未归因** — 请选择失败类型",
                )
                fa_col1, fa_col2, fa_col3 = st.columns([1, 2, 1])
                ft_choice = fa_col1.selectbox(
                    "失败类型",
                    options=list(_FAILURE_TYPE_LABELS.keys()),
                    format_func=lambda k: _FAILURE_TYPE_LABELS[k],
                    key=f"ft_{rec.id}",
                )
                ft_note = fa_col2.text_input(
                    "备注",
                    key=f"fn_{rec.id}",
                    placeholder="简要说明失败原因",
                )
                if fa_col3.button("确认", key=f"fc_{rec.id}", type="primary"):
                    set_failure_attribution(rec.id, ft_choice, ft_note)
                    st.success("归因已保存")
                    st.rerun()

            elif rec.failure_type:
                st.markdown(
                    f"🏷️ **归因**: {_FAILURE_TYPE_LABELS.get(rec.failure_type, rec.failure_type)}"
                    + (f"  — {rec.failure_note}" if rec.failure_note else "")
                )

            # Decision source badge
            src_colors = {
                "ai_drafted":    "#3b82f6",
                "human_edited":  "#f59e0b",
                "human_initiated": "#22c55e",
            }
            if rec.decision_source:
                src = rec.decision_source
                sc = src_colors.get(src, "#64748b")
                ratio_str = (
                    f"  edit_ratio={rec.edit_ratio:.2f}"
                    if rec.edit_ratio and src == "human_edited" else ""
                )
                st.markdown(
                    f'<span style="background:{sc};color:white;padding:2px 8px;'
                    f'border-radius:3px;font-size:0.8rem;">{src}{ratio_str}</span>',
                    unsafe_allow_html=True,
                )

st.divider()

# ── Failure Attribution Summary ────────────────────────────────────────────────
st.subheader("失败归因分布")

if fa["total_failures"] == 0:
    st.info("暂无已验证的失败记录。")
else:
    fa_col1, fa_col2 = st.columns(2)
    with fa_col1:
        if fa["by_type"]:
            import plotly.graph_objects as go
            labels = [_FAILURE_TYPE_SHORT.get(k, k) for k in fa["by_type"].keys()]
            values = list(fa["by_type"].values())
            if fa["unattributed"]:
                labels.append("未归因")
                values.append(fa["unattributed"])

            fig_pie = go.Figure(go.Pie(
                labels=labels, values=values,
                hole=0.4,
                marker_colors=[
                    "#ef4444", "#f59e0b", "#f97316",
                    "#a855f7", "#3b82f6", "#22c55e", "#64748b",
                ][:len(labels)],
                textinfo="label+percent",
            ))
            fig_pie.update_layout(
                height=280,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("所有失败记录尚未归因。")

    with fa_col2:
        st.markdown("**归因含义**")
        for k, v in _FAILURE_TYPE_LABELS.items():
            cnt = fa["by_type"].get(k, 0)
            st.markdown(
                f'`{k}`  {v}  '
                f'<span style="color:#94a3b8;">({cnt} 条)</span>',
                unsafe_allow_html=True,
            )
        if fa["unattributed"]:
            st.warning(f"⚠️ {fa['unattributed']} 条失败未归因 — 请在上方展开对应决策完成标注")

st.divider()

# ── Clean Zone Accuracy over time ─────────────────────────────────────────────
st.subheader("Clean Zone 胜率趋势")

with SessionFactory() as sess:
    cz_records = (
        sess.query(
            DecisionLog.verified_at,
            DecisionLog.accuracy_score,
            DecisionLog.sector_name,
            DecisionLog.direction,
        )
        .filter(
            DecisionLog.verified == True,
            DecisionLog.accuracy_score.isnot(None),
            DecisionLog.decision_date >= TRAIN_TEST_CUTOFF,
            DecisionLog.is_backtest == False,
        )
        .order_by(DecisionLog.verified_at)
        .all()
    )

if len(cz_records) < 3:
    st.info(
        f"Clean Zone 样本 n={len(cz_records)}，不足以绘制趋势图（需要至少 3 条）。"
        "继续积累 Clean Zone 数据后自动激活。"
    )
else:
    import plotly.graph_objects as go
    dates  = [r.verified_at for r in cz_records]
    scores = [r.accuracy_score for r in cz_records]
    running_avg = [
        sum(scores[:i+1]) / (i+1) for i in range(len(scores))
    ]

    fig_cz = go.Figure()
    fig_cz.add_trace(go.Scatter(
        x=dates, y=scores,
        mode="markers",
        marker=dict(
            color=["#22c55e" if s >= 0.75 else ("#f59e0b" if s == 0.5 else "#ef4444")
                   for s in scores],
            size=8,
        ),
        name="单次验证",
    ))
    fig_cz.add_trace(go.Scatter(
        x=dates, y=running_avg,
        mode="lines",
        line=dict(color="#3b82f6", width=2),
        name="累计平均胜率",
    ))
    fig_cz.add_hline(
        y=BASELINE_HIT_RATE,
        line_dash="dash", line_color="#94a3b8",
        annotation_text=f"随机基准 {BASELINE_HIT_RATE:.0%}",
    )
    fig_cz.update_layout(
        height=300,
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_cz, use_container_width=True)
    st.caption(
        f"n={len(cz_records)}  ·  "
        f"当前累计平均：{running_avg[-1]:.0%}  ·  "
        f"随机基准：{BASELINE_HIT_RATE:.0%}  ·  "
        f"统计显著性检验需 n≥200"
    )
