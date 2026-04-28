"""
Macro Alpha Pro — Regime Analysis
MSM 制度历史序列 · 转移矩阵 · 与人工判断对比
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import ui.theme as theme
from engine.regime import get_regime_on, get_regime_series, RegimeResult
from engine.memory import (
    init_db, load_structured_backtest, list_structured_backtests, SessionFactory, DecisionLog,
)

init_db()

# set_page_config handled by app.py via st.navigation()
# st.set_page_config(page_title="Regime Analysis | Macro Alpha Pro", page_icon="🌊", layout="wide")
theme.init_theme()

st.title("🌊 Regime Analysis")
st.caption("Hamilton MSM 制度检测 · Filtered Probability · 与人工判断对比")

today = datetime.date.today()

_REGIME_COLORS = {
    "risk-on":    "#22c55e",
    "risk-off":   "#ef4444",
    "transition": "#f59e0b",
    "unknown":    "#64748b",
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("参数")
    n_months = st.slider("历史月数", 12, 120, 36, step=6,
                         help="计算最近 N 个月的制度序列（月频，每个点独立估计）")
    use_backtest_data = st.checkbox(
        "从已保存回测加载",
        value=True,
        help="直接使用回测结果中的制度序列，速度更快",
    )

# ── Load regime data ───────────────────────────────────────────────────────────
regime_df = None

if use_backtest_data:
    saved = list_structured_backtests()
    if saved:
        _r = saved[0]  # most recent
        bt = load_structured_backtest(_r["run_id"])
        if bt and "returns" in bt:
            returns_df = bt["returns"]
            if "regime_label" in returns_df.columns:
                regime_df = returns_df[["regime_label", "p_risk_on", "yield_spread"]].copy()
                regime_df.index = pd.to_datetime(regime_df.index)
                st.sidebar.caption(
                    f"已加载回测数据：{_r['start_date']} → {_r['end_date']}  "
                    f"（{_r['n_months']} 个月）"
                )

if regime_df is None:
    with st.spinner(f"计算最近 {n_months} 个月制度序列（每个月独立估计，耗时约 {n_months * 2}s）…"):
        end   = today
        start = today.replace(
            month=((today.month - n_months - 1) % 12) + 1,
            year=today.year + ((today.month - n_months - 1) // 12),
        )
        # Generate monthly dates
        dates = []
        d = start
        while d <= end:
            dates.append(d)
            m = d.month + 1
            y = d.year + (m - 1) // 12
            m = (m - 1) % 12 + 1
            d = d.replace(year=y, month=m, day=min(d.day, 28))

        regime_series = get_regime_series(dates)
        regime_df = regime_series[["regime", "p_risk_on", "yield_spread"]].rename(
            columns={"regime": "regime_label"}
        )
        regime_df.index = pd.to_datetime(regime_df.index)

if regime_df is None or regime_df.empty:
    st.error("无法获取制度数据。")
    st.stop()

# ── KPI Strip ─────────────────────────────────────────────────────────────────
counts = regime_df["regime_label"].value_counts()
total  = len(regime_df)

k1, k2, k3, k4 = st.columns(4)
k1.metric("risk-on 期间",
          f"{counts.get('risk-on', 0)} 月",
          f"{counts.get('risk-on', 0)/total:.0%}")
k2.metric("risk-off 期间",
          f"{counts.get('risk-off', 0)} 月",
          f"{counts.get('risk-off', 0)/total:.0%}")
k3.metric("transition 期间",
          f"{counts.get('transition', 0)} 月",
          f"{counts.get('transition', 0)/total:.0%}")

# Current regime
try:
    current = get_regime_on(today, train_end=today)
    ri = {"risk-on": "🟢", "risk-off": "🔴", "transition": "🟡"}
    k4.metric("当前制度",
              f"{ri.get(current.regime, '⚪')} {current.regime}",
              f"P(risk-on)={current.p_risk_on:.1%}")
except Exception:
    k4.metric("当前制度", "计算中…")

st.divider()

# ── Regime Timeline ────────────────────────────────────────────────────────────
st.subheader("制度历史时间轴")

fig = go.Figure()

# P(risk-on) line
if "p_risk_on" in regime_df.columns:
    fig.add_trace(go.Scatter(
        x=regime_df.index,
        y=regime_df["p_risk_on"],
        mode="lines",
        name="P(risk-on)",
        line=dict(color="#3b82f6", width=2),
        yaxis="y",
    ))
    fig.add_hline(y=0.65, line_dash="dash", line_color="#22c55e",
                  annotation_text="risk-on 阈值 0.65", line_width=1)
    fig.add_hline(y=0.35, line_dash="dash", line_color="#ef4444",
                  annotation_text="risk-off 阈值 0.35", line_width=1)

# Yield spread line (secondary axis)
if "yield_spread" in regime_df.columns and regime_df["yield_spread"].notna().any():
    fig.add_trace(go.Scatter(
        x=regime_df.index,
        y=regime_df["yield_spread"],
        mode="lines",
        name="10Y-2Y 利差",
        line=dict(color="#f59e0b", width=1.5, dash="dot"),
        yaxis="y2",
        opacity=0.8,
    ))

# Regime background shading
prev_regime = None
start_idx   = regime_df.index[0]
for idx, row in regime_df.iterrows():
    r = row["regime_label"]
    if r != prev_regime:
        if prev_regime is not None:
            fig.add_vrect(
                x0=start_idx, x1=idx,
                fillcolor=_REGIME_COLORS.get(prev_regime, "#64748b"),
                opacity=0.08,
                layer="below",
                line_width=0,
            )
        start_idx   = idx
        prev_regime = r
# Last segment
fig.add_vrect(
    x0=start_idx, x1=regime_df.index[-1],
    fillcolor=_REGIME_COLORS.get(prev_regime, "#64748b"),
    opacity=0.08, layer="below", line_width=0,
)

fig.update_layout(
    height=380,
    yaxis=dict(title="P(risk-on)", range=[0, 1], tickformat=".0%"),
    yaxis2=dict(title="利差 (%)", overlaying="y", side="right",
                showgrid=False, tickformat=".2f"),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=0, r=60, t=30, b=0),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# Regime segments bar (bottom band)
regime_seq = regime_df["regime_label"].tolist()
dates_seq  = regime_df.index.tolist()
bar_colors = [_REGIME_COLORS.get(r, "#64748b") for r in regime_seq]

fig2 = go.Figure(go.Bar(
    x=dates_seq,
    y=[1] * len(dates_seq),
    marker_color=bar_colors,
    showlegend=False,
))
fig2.update_layout(
    height=50,
    margin=dict(l=0, r=60, t=0, b=0),
    xaxis=dict(showticklabels=False),
    yaxis=dict(showticklabels=False, showgrid=False),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    bargap=0,
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Transition Matrix ─────────────────────────────────────────────────────────
st.subheader("制度转移矩阵")
st.caption("每个格子 = 上一个月制度 → 当月制度 的历史频率")

regimes = ["risk-on", "transition", "risk-off"]
trans   = pd.DataFrame(0, index=regimes, columns=regimes)

labels = regime_df["regime_label"].tolist()
for i in range(1, len(labels)):
    fr, to = labels[i-1], labels[i]
    if fr in regimes and to in regimes:
        trans.loc[fr, to] += 1

# Normalize to probabilities
trans_pct = trans.div(trans.sum(axis=1).replace(0, 1), axis=0)

col_m, col_c = st.columns([1, 1])
with col_m:
    st.markdown("**次数**")
    st.dataframe(trans.rename(
        index={"risk-on": "🟢 risk-on", "transition": "🟡 transition", "risk-off": "🔴 risk-off"},
        columns={"risk-on": "→ risk-on", "transition": "→ transition", "risk-off": "→ risk-off"},
    ), use_container_width=True)

with col_c:
    st.markdown("**概率**")
    st.dataframe(
        trans_pct.map(lambda x: f"{x:.0%}").rename(
            index={"risk-on": "🟢 risk-on", "transition": "🟡 transition", "risk-off": "🔴 risk-off"},
            columns={"risk-on": "→ risk-on", "transition": "→ transition", "risk-off": "→ risk-off"},
        ),
        use_container_width=True,
    )

if trans_pct.loc["risk-off", "risk-off"] > 0.5:
    st.info(
        f"📌 risk-off 制度自持续性：{trans_pct.loc['risk-off','risk-off']:.0%}  "
        "— 一旦进入 risk-off，下月仍在 risk-off 的概率超过 50%，支持制度叠加缩仓的设计。"
    )

st.divider()

# ── Human vs Model Comparison ─────────────────────────────────────────────────
st.subheader("人工制度判断 vs MSM 模型对比")
st.caption("来源：DecisionLog.macro_regime（人工标注）vs 同日期 MSM 输出")

with SessionFactory() as sess:
    human_records = (
        sess.query(
            DecisionLog.decision_date,
            DecisionLog.macro_regime,
            DecisionLog.sector_name,
        )
        .filter(
            DecisionLog.macro_regime.isnot(None),
            DecisionLog.is_backtest == False,
        )
        .order_by(DecisionLog.decision_date.desc())
        .limit(60)
        .all()
    )

if not human_records:
    st.info("暂无人工制度标注记录。运行 Sector Analysis 并确认入库后将在此显示对比。")
else:
    _HUMAN_TO_MODEL = {
        "低波动/牛市": "risk-on",
        "温和波动":   "risk-on",
        "震荡期":     "transition",
        "高波动/危机": "risk-off",
    }

    comparison = []
    for date, human_regime, sector in human_records:
        if date is None:
            continue
        model_regime = regime_df.loc[
            regime_df.index <= pd.Timestamp(date)
        ]["regime_label"].iloc[-1] if len(regime_df.loc[regime_df.index <= pd.Timestamp(date)]) > 0 else None

        human_mapped = _HUMAN_TO_MODEL.get(human_regime, "unknown")
        match = (human_mapped == model_regime) if model_regime else None

        comparison.append({
            "日期":       date,
            "板块":       sector,
            "人工制度":   human_regime,
            "人工(mapped)": human_mapped,
            "MSM制度":    model_regime or "—",
            "一致":       "✅" if match else ("❌" if match is False else "—"),
        })

    comp_df = pd.DataFrame(comparison)
    n_match = (comp_df["一致"] == "✅").sum()
    n_total = (comp_df["一致"] != "—").sum()

    if n_total > 0:
        m1, m2 = st.columns(2)
        m1.metric("人工 vs 模型一致率", f"{n_match/n_total:.0%}",
                  f"{n_match}/{n_total} 条有效对比")
        m2.caption(
            "注：一致率 < 100% 不代表哪方更准确，"
            "差异点才是最值得研究的。"
        )

    st.dataframe(
        comp_df[["日期", "板块", "人工制度", "MSM制度", "一致"]],
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.caption(
    "方法论：Hamilton (1989) Markov Switching Model  ·  "
    "BIC 选 k∈{2,3}  ·  Filtered probability（无前视偏差）  ·  "
    "信号变量：10Y-2Y 国债利差（FRED）"
)
