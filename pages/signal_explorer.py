"""
Macro Alpha Pro — Signal Explorer
Full quantitative layer: TSMOM/CSMOM heatmap, regime time series, gate debug,
and GARCH volatility comparison across the 32-ETF universe.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import ui.theme as theme
from engine.signal import get_signal_dataframe, compute_composite_scores, get_quant_gates
from engine.regime import get_regime_on, get_regime_series
from engine.memory import init_db

init_db()
theme.init_theme()

today = datetime.date.today()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("参数")
    as_of_date = st.date_input(
        "信号计算日期", value=today, max_value=today,
        help="信号仅使用该日期之前可用的数据（无前视偏差）",
    )
    lookback_m = st.slider("TSMOM 形成期（月）", 6, 24, 12,
                           help="标准：12月（Moskowitz et al. 2012）")
    skip_m     = st.slider("跳过最近（月）", 0, 3, 1,
                           help="跳过 1 月避免微观结构偏差（Jegadeesh & Titman 1993）")
    compare_months = st.slider("与 N 个月前对比", 1, 6, 1)
    show_vol = st.checkbox("显示波动率权重列", value=True)

# ── Load signal data ───────────────────────────────────────────────────────────
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
        gates_now = get_quant_gates(as_of_date, regime_label=_rl,
                                    lookback_months=lookback_m, skip_months=skip_m)
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

# ── Main tabs ─────────────────────────────────────────────────────────────────
t_heatmap, t_regime, t_gate, t_vol = st.tabs([
    "📊 Signal Heatmap",
    "🌊 Regime Analysis",
    "🔒 Gate Debug",
    "📈 Volatility",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Signal Heatmap
# ══════════════════════════════════════════════════════════════════════════════
with t_heatmap:
    st.caption(f"TSMOM / CSMOM 信号 · {len(sig_now)} 板块 · 截止 {as_of_date}")

    n_long  = int((sig_now["tsmom"] == 1).sum())
    n_short = int((sig_now["tsmom"] == -1).sum())
    n_neut  = int((sig_now["tsmom"] == 0).sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("多头信号", n_long)
    k2.metric("空头信号", n_short)
    k3.metric("中性信号", n_neut)
    if regime_now:
        icons = {"risk-on": "🟢", "risk-off": "🔴", "transition": "🟡"}
        k4.metric("当前制度",
                  f"{icons.get(regime_now.regime, '⚪')} {regime_now.regime}")
        k5.metric("P(risk-on)", f"{regime_now.p_risk_on:.1%}")

    st.divider()

    # Build display table
    rows = []
    for sector in sig_now.index:
        cur  = sig_now.loc[sector]
        prev = sig_prev.loc[sector] if not sig_prev.empty and sector in sig_prev.index else None

        tsmom_now  = int(cur["tsmom"])
        csmom_now  = int(cur["csmom"])
        tsmom_prev = int(prev["tsmom"]) if prev is not None else None
        csmom_prev = int(prev["csmom"]) if prev is not None else None
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
            "资产类别":    cur.get("asset_class", "—"),
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

    _SIGNAL_COLORS  = {1: "#22c55e", -1: "#ef4444", 0: "#94a3b8"}
    _GATE_SEV_COLORS = {"hard": "#ef4444", "soft": "#f59e0b", "clear": "rgba(0,0,0,0)"}

    def _comp_color(v):
        if v is None:  return "rgba(0,0,0,0)"
        if v >= 70:    return "#22c55e"
        if v >= 50:    return "#3b82f6"
        if v >= 30:    return "#f59e0b"
        return "#ef4444"

    header_vals = ["板块", "ETF", "资产类别", "12-1M收益%", "年化波动%", "TSMOM", "CSMOM", "合成分", "门控"]
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
        df_display["资产类别"].tolist(),
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

    # Within-class CSMOM grouping
    st.subheader("跨资产类别 CSMOM 排序")
    if "资产类别" in df_display.columns:
        for cls in sorted(df_display["资产类别"].dropna().unique()):
            cls_df = df_display[df_display["资产类别"] == cls].sort_values("12-1M收益%", ascending=False)
            if cls_df.empty:
                continue
            with st.expander(f"{cls} ({len(cls_df)} ETF)", expanded=True):
                cls_sorted = cls_df.sort_values("12-1M收益%", ascending=False)
                cols = st.columns(len(cls_sorted))
                for ci, (_, row) in enumerate(cls_sorted.iterrows()):
                    clr = _SIGNAL_COLORS.get(row["CSMOM"], "#94a3b8")
                    with cols[ci]:
                        st.markdown(
                            f'<div style="text-align:center; padding:0.5rem; '
                            f'border:1px solid {clr}; border-radius:4px;">'
                            f'<div style="font-weight:700; color:{clr}; font-size:1.1rem;">{row["ETF"]}</div>'
                            f'<div style="font-size:0.72rem; color:#94a3b8;">{row["板块"][:10]}</div>'
                            f'<div style="font-size:0.82rem; font-weight:600; color:{("#22c55e" if row["12-1M收益%"]>0 else "#ef4444")};">'
                            f'{row["12-1M收益%"]:+.1f}%</div>'
                            f'<div style="font-size:0.7rem; color:#64748b;">{row["CSMOM_label"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    st.divider()

    # Return bar chart
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
        height=max(400, len(df_sorted) * 26),
        xaxis_title="收益率",
        xaxis=dict(ticksuffix="%"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        margin=dict(l=0, r=80, t=10, b=10),
    )
    fig_bar.add_vline(x=0, line_color="#64748b", line_width=1)
    st.plotly_chart(fig_bar, use_container_width=True)

    # TSMOM vs CSMOM agreement
    st.divider()
    st.subheader("TSMOM × CSMOM 一致性矩阵")
    agree    = df_display[df_display["TSMOM"] == df_display["CSMOM"]]
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
            st.markdown(f'● {r["板块"]}  TSMOM:{r["TSMOM_label"]} / CSMOM:{r["CSMOM_label"]}')

    st.caption(
        f"数据截止：{as_of_date}  ·  形成期：{lookback_m}-1 月  ·  "
        f"参考：Moskowitz, Ooi & Pedersen (2012)  ·  对比日期：{prev_date}"
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Regime Analysis
# ══════════════════════════════════════════════════════════════════════════════
with t_regime:
    _REGIME_COLORS = {
        "risk-on":    "#22c55e",
        "risk-off":   "#ef4444",
        "transition": "#f59e0b",
        "unknown":    "#64748b",
    }

    n_months_regime = st.slider("历史月数", 12, 120, 36, step=6, key="regime_n_months",
                                help="计算最近 N 个月的制度序列")

    regime_df = None

    # Try loading from saved backtest first
    try:
        from engine.memory import load_structured_backtest, list_structured_backtests
        saved = list_structured_backtests()
        if saved:
            bt = load_structured_backtest(saved[0]["run_id"])
            if bt and "returns" in bt:
                returns_df = bt["returns"]
                if "regime_label" in returns_df.columns:
                    regime_df = returns_df[["regime_label", "p_risk_on", "yield_spread"]].copy()
                    regime_df.index = pd.to_datetime(regime_df.index)
                    st.caption(
                        f"已加载回测数据：{saved[0]['start_date']} → {saved[0]['end_date']} "
                        f"（{saved[0]['n_months']} 个月）"
                    )
    except Exception:
        pass

    if regime_df is None:
        with st.spinner(f"计算最近 {n_months_regime} 个月制度序列…"):
            end   = today
            start = today.replace(
                month=((today.month - n_months_regime - 1) % 12) + 1,
                year=today.year + ((today.month - n_months_regime - 1) // 12),
            )
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
    else:
        counts = regime_df["regime_label"].value_counts()
        total  = len(regime_df)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("risk-on 期间",   f"{counts.get('risk-on', 0)} 月",
                  f"{counts.get('risk-on', 0)/total:.0%}")
        k2.metric("risk-off 期间",  f"{counts.get('risk-off', 0)} 月",
                  f"{counts.get('risk-off', 0)/total:.0%}")
        k3.metric("transition 期间",f"{counts.get('transition', 0)} 月",
                  f"{counts.get('transition', 0)/total:.0%}")
        if regime_now:
            ri = {"risk-on": "🟢", "risk-off": "🔴", "transition": "🟡"}
            k4.metric("当前制度",
                      f"{ri.get(regime_now.regime,'⚪')} {regime_now.regime}",
                      f"P(risk-on)={regime_now.p_risk_on:.1%}")

        st.divider()
        st.subheader("制度历史时间轴")

        fig = go.Figure()

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

        prev_regime = None
        start_idx   = regime_df.index[0]
        for idx, row in regime_df.iterrows():
            r = row["regime_label"]
            if r != prev_regime:
                if prev_regime is not None:
                    fig.add_vrect(
                        x0=start_idx, x1=idx,
                        fillcolor=_REGIME_COLORS.get(prev_regime, "#64748b"),
                        opacity=0.08, layer="below", line_width=0,
                    )
                start_idx   = idx
                prev_regime = r
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

        regime_seq = regime_df["regime_label"].tolist()
        dates_seq  = regime_df.index.tolist()
        bar_colors = [_REGIME_COLORS.get(r, "#64748b") for r in regime_seq]
        fig2 = go.Figure(go.Bar(
            x=dates_seq, y=[1] * len(dates_seq),
            marker_color=bar_colors, showlegend=False,
        ))
        fig2.update_layout(
            height=50, margin=dict(l=0, r=60, t=0, b=0),
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False, showgrid=False),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", bargap=0,
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("制度转移矩阵")
        regimes = ["risk-on", "transition", "risk-off"]
        trans = pd.DataFrame(0, index=regimes, columns=regimes)
        labels = regime_df["regime_label"].tolist()
        for i in range(1, len(labels)):
            fr, to = labels[i-1], labels[i]
            if fr in regimes and to in regimes:
                trans.loc[fr, to] += 1

        trans_pct = trans.div(trans.sum(axis=1).replace(0, 1), axis=0)
        col_m, col_c = st.columns(2)
        with col_m:
            st.markdown("**次数**")
            st.dataframe(trans.rename(
                index={"risk-on":"🟢 risk-on","transition":"🟡 transition","risk-off":"🔴 risk-off"},
                columns={"risk-on":"→ risk-on","transition":"→ transition","risk-off":"→ risk-off"},
            ), use_container_width=True)
        with col_c:
            st.markdown("**概率**")
            st.dataframe(trans_pct.map(lambda x: f"{x:.0%}").rename(
                index={"risk-on":"🟢 risk-on","transition":"🟡 transition","risk-off":"🔴 risk-off"},
                columns={"risk-on":"→ risk-on","transition":"→ transition","risk-off":"→ risk-off"},
            ), use_container_width=True)

        st.caption(
            "方法论：Hamilton (1989) Markov Switching Model  ·  "
            "BIC 选 k∈{2,3}  ·  Filtered probability（无前视偏差）"
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Gate Debug
# ══════════════════════════════════════════════════════════════════════════════
with t_gate:
    st.subheader("Quant Gate 触发详情")
    st.caption("R1–R5 规则触发频率及各板块门控状态详情")

    if not gates_now:
        st.info("暂无门控数据。")
    else:
        gate_rows = []
        for sector, gate in gates_now.items():
            blocked  = gate.get("blocked", [])
            reasons  = gate.get("reasons", [])
            severity = gate.get("severity", "clear")
            gate_rows.append({
                "板块":     sector,
                "严重度":   severity,
                "封锁操作": " / ".join(blocked) if blocked else "—",
                "触发规则": " | ".join(reasons[:3]) if reasons else "—",
            })

        gdf = pd.DataFrame(gate_rows)
        _sev_map = {"hard": "🔴 Hard", "soft": "🟡 Soft", "clear": "✅ Clear"}
        gdf["严重度"] = gdf["严重度"].map(_sev_map)

        n_hard = sum(1 for r in gate_rows if r["严重度"] == "hard")
        n_soft = sum(1 for r in gate_rows if r["严重度"] == "soft")
        g1, g2, g3 = st.columns(3)
        g1.metric("Hard 门控", n_hard)
        g2.metric("Soft 门控", n_soft)
        g3.metric("正常通过", len(gate_rows) - n_hard - n_soft)
        st.divider()
        st.dataframe(gdf, use_container_width=True, hide_index=True)

        # Rule frequency
        all_reasons = [r for gate in gates_now.values() for r in gate.get("reasons", [])]
        if all_reasons:
            from collections import Counter
            counts = Counter(all_reasons)
            rf_df = pd.DataFrame(
                [{"规则": k, "触发次数": v} for k, v in counts.most_common()],
            )
            st.divider()
            st.subheader("规则触发频率")
            st.dataframe(rf_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Volatility
# ══════════════════════════════════════════════════════════════════════════════
with t_vol:
    st.subheader("波动率对比")
    st.caption("GARCH(1,1) 预测 vs 21日已实现 vs 12个月历史波动率")

    vol_data = []
    for sector in sig_now.index:
        row = sig_now.loc[sector]
        vol_data.append({
            "板块":          sector,
            "ETF":           row["ticker"],
            "年化波动% (信号)": round(row["ann_vol"] * 100, 1),
            "GARCH预测%":    round(row.get("garch_vol", row["ann_vol"]) * 100, 1),
            "反向波动权重":   round(row.get("inv_vol_wt", 0), 4),
        })

    vdf = pd.DataFrame(vol_data).sort_values("年化波动% (信号)", ascending=False)

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=vdf["板块"], y=vdf["年化波动% (信号)"],
        name="年化波动率", marker_color="#3b82f6",
    ))
    if vdf["GARCH预测%"].notna().any():
        fig_vol.add_trace(go.Scatter(
            x=vdf["板块"], y=vdf["GARCH预测%"],
            mode="markers+lines",
            name="GARCH(1,1) 预测",
            marker=dict(color="#f59e0b", size=8),
            line=dict(color="#f59e0b", dash="dot"),
        ))
    fig_vol.update_layout(
        height=380,
        xaxis=dict(tickangle=-45),
        yaxis_title="年化波动率 (%)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=10, b=80),
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    st.divider()
    st.dataframe(vdf, use_container_width=True, hide_index=True)
    st.caption(
        "GARCH(1,1) 模型使用过去 252 交易日训练，arch 库一步预测；"
        "反向波动权重 = 1/σ 归一化，用于 Carry/组合构建加权。"
    )
