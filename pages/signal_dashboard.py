"""
Macro Alpha Pro — Signal Dashboard
18 板块 TSMOM / CSMOM 信号热力图 + 历史信号变化追踪
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import ui.theme as theme
from engine.signal import get_signal_dataframe
from engine.regime import get_regime_on
from engine.memory import init_db
from engine.signal import compute_composite_scores, get_quant_gates

init_db()

# set_page_config handled by app.py via st.navigation()
# st.set_page_config(page_title="Signal Dashboard | Macro Alpha Pro", page_icon="📡", layout="wide")
theme.init_theme()

st.title("📡 Signal Dashboard")
st.caption("TSMOM / CSMOM 信号 · 18 板块 · 实时计算")

today = datetime.date.today()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("参数")
    as_of_date = st.date_input(
        "信号计算日期",
        value=today,
        max_value=today,
        help="信号仅使用该日期之前可用的数据（无前视偏差）",
    )
    lookback_m = st.slider("TSMOM 形成期（月）", 6, 24, 12,
                           help="标准：12月（Moskowitz et al. 2012）")
    skip_m     = st.slider("跳过最近（月）", 0, 3, 1,
                           help="跳过 1 月避免微观结构偏差（Jegadeesh & Titman 1993）")
    compare_months = st.slider("与 N 个月前对比", 1, 6, 1)
    show_vol = st.checkbox("显示波动率权重列", value=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("计算信号…"):
    sig_now = get_signal_dataframe(as_of_date, lookback_m, skip_m)
    try:
        composite_df = compute_composite_scores(as_of_date, lookback_m, skip_m)
    except Exception:
        composite_df = pd.DataFrame()

    try:
        regime_now = get_regime_on(as_of=as_of_date, train_end=as_of_date)
    except Exception:
        regime_now = None

    try:
        _rl = regime_now.regime if regime_now else "transition"
        gates_now = get_quant_gates(as_of_date, regime_label=_rl, lookback_months=lookback_m, skip_months=skip_m)
    except Exception:
        gates_now = {}

    prev_date = as_of_date.replace(
        month=((as_of_date.month - compare_months - 1) % 12) + 1,
        year=as_of_date.year + ((as_of_date.month - compare_months - 1) // 12),
    )
    sig_prev = get_signal_dataframe(prev_date, lookback_m, skip_m)

if sig_now.empty:
    st.error("无法获取信号数据，请检查网络连接或日期范围。")
    st.stop()

# ── KPI Strip ─────────────────────────────────────────────────────────────────
n_long  = int((sig_now["tsmom"] == 1).sum())
n_short = int((sig_now["tsmom"] == -1).sum())
n_neut  = int((sig_now["tsmom"] == 0).sum())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("多头信号", n_long,  delta=None)
k2.metric("空头信号", n_short, delta=None)
k3.metric("中性信号", n_neut,  delta=None)

if regime_now:
    regime_icons = {"risk-on": "🟢", "risk-off": "🔴", "transition": "🟡"}
    k4.metric("当前制度",
              f"{regime_icons.get(regime_now.regime, '⚪')} {regime_now.regime}")
    k5.metric("P(risk-on)", f"{regime_now.p_risk_on:.1%}")

st.divider()

# ── Signal Heatmap ────────────────────────────────────────────────────────────
st.subheader("信号热力图")

# Build display table
rows = []
for sector in sig_now.index:
    cur  = sig_now.loc[sector]
    prev = sig_prev.loc[sector] if not sig_prev.empty and sector in sig_prev.index else None

    tsmom_now  = int(cur["tsmom"])
    csmom_now  = int(cur["csmom"])
    tsmom_prev = int(prev["tsmom"]) if prev is not None else None
    csmom_prev = int(prev["csmom"]) if prev is not None else None

    # Signal change detection
    tsmom_changed = (tsmom_prev is not None and tsmom_now != tsmom_prev)
    csmom_changed = (csmom_prev is not None and csmom_now != csmom_prev)

    sig_map = {1: "▲ 多头", -1: "▼ 空头", 0: "— 中性"}

    comp  = float(composite_df.loc[sector, "composite_score"]) \
            if not composite_df.empty and sector in composite_df.index else None
    gate  = gates_now.get(sector, {})
    blocked = gate.get("blocked", [])
    gate_label = ""
    if "超配" in blocked and "标配" in blocked:
        gate_label = "仅低配"
    elif "超配" in blocked and "低配" in blocked:
        gate_label = "仅标配"
    elif "超配" in blocked:
        gate_label = "禁超配"
    elif "低配" in blocked:
        gate_label = "禁低配"

    rows.append({
        "板块":        sector,
        "ETF":         cur["ticker"],
        "12-1M收益%":  round(cur["raw_return"] * 100, 2),
        "年化波动%":   round(cur["ann_vol"] * 100, 1),
        "TSMOM":       tsmom_now,
        "TSMOM_label": sig_map[tsmom_now],
        "CSMOM":       csmom_now,
        "CSMOM_label": sig_map[csmom_now],
        "TSMOM_flip":  tsmom_changed,
        "CSMOM_flip":  csmom_changed,
        "inv_vol_wt":  round(cur.get("inv_vol_wt", 0), 4),
        "composite":   comp,
        "gate_label":  gate_label,
        "gate_sev":    gate.get("severity", "clear"),
    })

df_display = pd.DataFrame(rows)

# Color-coded signal values for heatmap
_SIGNAL_COLORS = {1: "#22c55e", -1: "#ef4444", 0: "#94a3b8"}
_GATE_SEV_COLORS = {"hard": "#ef4444", "soft": "#f59e0b", "clear": "rgba(0,0,0,0)"}

# Composite score color gradient (red=0 → gray=50 → green=100)
def _comp_color(v):
    if v is None: return "rgba(0,0,0,0)"
    if v >= 70:   return "#22c55e"
    if v >= 50:   return "#3b82f6"
    if v >= 30:   return "#f59e0b"
    return "#ef4444"

# Plotly table for color coding
header_vals = ["板块", "ETF", "12-1M收益%", "年化波动%", "TSMOM", "CSMOM", "合成分", "门控"]
if show_vol:
    header_vals.append("反向波动权重")

tsmom_colors = [_SIGNAL_COLORS[v] for v in df_display["TSMOM"]]
csmom_colors = [_SIGNAL_COLORS[v] for v in df_display["CSMOM"]]
ret_colors   = ["#22c55e" if v > 0 else "#ef4444" for v in df_display["12-1M收益%"]]
comp_colors  = [_comp_color(v) for v in df_display["composite"]]
gate_colors  = [_GATE_SEV_COLORS.get(v, "rgba(0,0,0,0)") for v in df_display["gate_sev"]]

cell_vals = [
    df_display["板块"].tolist(),
    df_display["ETF"].tolist(),
    [f"{v:+.2f}%" for v in df_display["12-1M收益%"]],
    [f"{v:.1f}%" for v in df_display["年化波动%"]],
    df_display["TSMOM_label"].tolist(),
    df_display["CSMOM_label"].tolist(),
    [f"{v:.0f}" if v is not None else "—" for v in df_display["composite"]],
    [v if v else "—" for v in df_display["gate_label"]],
]
cell_colors = [
    ["rgba(0,0,0,0)"] * len(df_display),
    ["rgba(0,0,0,0)"] * len(df_display),
    ret_colors,
    ["rgba(0,0,0,0)"] * len(df_display),
    tsmom_colors,
    csmom_colors,
    comp_colors,
    gate_colors,
]

if show_vol:
    cell_vals.append([f"{v:.4f}" for v in df_display["inv_vol_wt"]])
    cell_colors.append(["rgba(0,0,0,0)"] * len(df_display))

# Mark signal flips with asterisk
flip_markers = []
for i, row in df_display.iterrows():
    markers = []
    if row["TSMOM_flip"]: markers.append("TSMOM翻转")
    if row["CSMOM_flip"]: markers.append("CSMOM翻转")
    flip_markers.append(" ⚡" + "/".join(markers) if markers else "")

cell_vals[0] = [s + flip_markers[i] for i, s in enumerate(df_display["板块"].tolist())]

fig_table = go.Figure(data=[go.Table(
    header=dict(
        values=[f"<b>{h}</b>" for h in header_vals],
        fill_color="#1e293b",
        font=dict(color="white", size=13),
        align="left",
        height=35,
    ),
    cells=dict(
        values=cell_vals,
        fill_color=cell_colors,
        font=dict(color="white", size=12),
        align="left",
        height=30,
    ),
)])
fig_table.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    height=35 + 30 * len(df_display) + 20,
)
st.plotly_chart(fig_table, use_container_width=True)

# Signal flip callouts
flipped = df_display[df_display["TSMOM_flip"] | df_display["CSMOM_flip"]]
if not flipped.empty:
    st.warning(
        f"⚡ **{len(flipped)} 个板块信号自 {compare_months} 个月前发生翻转**：  "
        + "  |  ".join(
            f"{r['板块']} TSMOM:{r['TSMOM_label']}"
            for _, r in flipped.iterrows()
        )
    )

st.divider()

# ── Return Bar Chart ──────────────────────────────────────────────────────────
st.subheader("12-1M 形成期收益率排序")

df_sorted = df_display.sort_values("12-1M收益%")
bar_colors = ["#22c55e" if v > 0 else "#ef4444" for v in df_sorted["12-1M收益%"]]

fig_bar = go.Figure(go.Bar(
    x=df_sorted["12-1M收益%"],
    y=df_sorted["板块"],
    orientation="h",
    marker_color=bar_colors,
    text=[f"{v:+.2f}%" for v in df_sorted["12-1M收益%"]],
    textposition="outside",
))
fig_bar.update_layout(
    height=max(350, len(df_sorted) * 26),
    xaxis_title="收益率",
    xaxis=dict(ticksuffix="%"),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0"),
    margin=dict(l=0, r=80, t=10, b=10),
)
fig_bar.add_vline(x=0, line_color="#64748b", line_width=1)
st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── TSMOM vs CSMOM agreement matrix ──────────────────────────────────────────
st.subheader("TSMOM × CSMOM 一致性矩阵")

agree = df_display[df_display["TSMOM"] == df_display["CSMOM"]]
disagree = df_display[df_display["TSMOM"] != df_display["CSMOM"]]

col_a, col_d = st.columns(2)
with col_a:
    st.markdown(f"**✅ 信号一致 ({len(agree)} 个)**")
    for _, r in agree.iterrows():
        color = "#22c55e" if r["TSMOM"] == 1 else ("#ef4444" if r["TSMOM"] == -1 else "#94a3b8")
        st.markdown(
            f'<span style="color:{color}">● {r["板块"]}</span> {r["TSMOM_label"]}',
            unsafe_allow_html=True,
        )

with col_d:
    st.markdown(f"**⚠️ 信号分歧 ({len(disagree)} 个)**")
    for _, r in disagree.iterrows():
        st.markdown(
            f'● {r["板块"]}  TSMOM:{r["TSMOM_label"]} / CSMOM:{r["CSMOM_label"]}',
        )

st.divider()
st.caption(
    f"数据截止：{as_of_date}  ·  形成期：{lookback_m}-1 月  ·  "
    f"参考：Moskowitz, Ooi & Pedersen (2012)  ·  "
    f"信号翻转对比日期：{prev_date}"
)
