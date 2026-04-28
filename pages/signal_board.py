"""
Macro Alpha Pro — Signal Board
================================
量化信号层 · TSMOM/CSMOM 热力矩阵 · Hamilton MSM 制度时序 · Quant Gate 状态

这是 Intelligence 层的定量视角：
  Tab 1  Signal Matrix — 36 ETF 信号一览，含跨资产类别 CSMOM 分组
  Tab 2  Regime — P(risk-on) 时序，转移矩阵，人工 vs 模型对比
  Tab 3  Gates — Quant Gate 触发规则频率，板块门控状态
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import ui.theme as theme
from engine.memory import init_db, get_daily_brief_snapshot
from engine.signal import get_signal_dataframe, compute_composite_scores, get_quant_gates
from engine.regime import get_regime_on

init_db()
theme.init_theme()

today = datetime.date.today()
_is_dark_sig = theme.is_dark()
_chart_font  = "#e2e8f0" if _is_dark_sig else "#1e293b"

# ── Drill-down context banner ──────────────────────────────────────────────────
# Shows what surfaced in today's Daily Brief that's relevant to this page.
try:
    _snap = get_daily_brief_snapshot(today)
    _alerts_raw = []
    if _snap and _snap.risk_alerts_json:
        import json as _json
        _alerts_raw = _json.loads(_snap.risk_alerts_json)
    _flips_here   = [a for a in _alerts_raw if "tsmom_flip"      in a]
    _compress_here = [a for a in _alerts_raw if "regime_compress" in a]
    _regime_changed = getattr(_snap, "regime_changed", False) if _snap else False
    _snap_regime    = getattr(_snap, "regime",         "")    if _snap else ""
    _snap_regime_prev = getattr(_snap, "regime_prev",  "")    if _snap else ""

    _context_items = []
    if _regime_changed and _snap_regime_prev:
        _context_items.append(
            f'<span style="color:#f59e0b;font-weight:700;">制度切换  '
            f'{_snap_regime_prev} → {_snap_regime}</span>'
            f'<span style="color:rgba(255,255,255,0.45);"> — Regime Tab 查看 P(risk-on) 时序</span>'
        )
    if _flips_here:
        _sectors = ", ".join(a.split(":")[0] for a in _flips_here)
        _context_items.append(
            f'<span style="color:#f59e0b;font-weight:700;">TSMOM 翻转：{_sectors}</span>'
            f'<span style="color:rgba(255,255,255,0.45);"> — Signal Matrix Tab 定位相关行</span>'
        )
    if _compress_here:
        _sectors = ", ".join(a.split(":")[0] for a in _compress_here)
        _context_items.append(
            f'<span style="color:#f59e0b;">制度压缩：{_sectors}</span>'
        )

    if _context_items:
        _is_dark_sb    = theme.is_dark()
        _banner_bg     = "rgba(245,158,11,0.06)" if _is_dark_sb else "rgba(245,158,11,0.08)"
        _banner_border = "rgba(245,158,11,0.35)"
        _sb_label_c    = "rgba(255,255,255,0.4)"  if _is_dark_sb else "rgba(0,0,0,0.5)"
        _sb_hint_c     = "rgba(255,255,255,0.45)" if _is_dark_sb else "rgba(0,0,0,0.55)"
        # Fix hint-text colors in already-built items (replace dark-only rgba)
        _items_html = (
            "  ·  ".join(_context_items)
            .replace("rgba(255,255,255,0.45)", _sb_hint_c)
            .replace("rgba(255,255,255,0.4)",  _sb_label_c)
        )
        st.markdown(
            f'<div style="background:{_banner_bg};border:1px solid {_banner_border};'
            f'border-radius:5px;padding:0.55rem 1rem;margin-bottom:0.9rem;font-size:0.82rem;">'
            f'<span style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;'
            f'color:{_sb_label_c};margin-right:0.6rem;">Daily Brief →</span>'
            f'{_items_html}</div>',
            unsafe_allow_html=True,
        )
except Exception:
    pass

# ── Sidebar params ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("参数")
    as_of   = st.date_input("截止日期", value=today, max_value=today)
    lk_m    = st.slider("TSMOM 形成期（月）", 6, 24, 12)
    sk_m    = st.slider("跳过最近（月）",      0,  3,  1)
    cmp_m   = st.slider("对比 N 个月前",       1,  6,  1)
    show_wt = st.checkbox("显示反向波动权重", value=False)

# ── Cached data loaders (persist across page navigations, TTL in seconds) ─────
@st.cache_data(ttl=300, show_spinner=False)
def _cached_sig(as_of_str: str, lk: int, sk: int) -> pd.DataFrame:
    return get_signal_dataframe(datetime.date.fromisoformat(as_of_str), lk, sk)

@st.cache_data(ttl=300, show_spinner=False)
def _cached_comp(as_of_str: str, lk: int, sk: int) -> pd.DataFrame:
    try:
        return compute_composite_scores(datetime.date.fromisoformat(as_of_str), lk, sk)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_regime(as_of_str: str):
    try:
        d = datetime.date.fromisoformat(as_of_str)
        return get_regime_on(as_of=d, train_end=d)
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def _cached_gates(as_of_str: str, regime_label: str, lk: int, sk: int) -> dict:
    try:
        return get_quant_gates(
            datetime.date.fromisoformat(as_of_str),
            regime_label=regime_label, lookback_months=lk, skip_months=sk,
        )
    except Exception:
        return {}

# ── Data ──────────────────────────────────────────────────────────────────────
_as_str = str(as_of)
prev_date = as_of.replace(
    month=((as_of.month - cmp_m - 1) % 12) + 1,
    year=as_of.year + ((as_of.month - cmp_m - 1) // 12),
)

_is_cached = st.session_state.get("_sig_cache_key") == (_as_str, lk_m, sk_m)
with st.spinner("计算信号…" if not _is_cached else ""):
    sig      = _cached_sig(_as_str, lk_m, sk_m)
    comp     = _cached_comp(_as_str, lk_m, sk_m)
    regime   = _cached_regime(_as_str)
    _rl      = regime.regime if regime else "transition"
    gates    = _cached_gates(_as_str, _rl, lk_m, sk_m)
    sig_prev = _cached_sig(str(prev_date), lk_m, sk_m)
    st.session_state["_sig_cache_key"] = (_as_str, lk_m, sk_m)

if sig.empty:
    st.error("信号数据加载失败，请检查网络或日期范围。")
    st.stop()

t_matrix, t_regime, t_gates = st.tabs([
    "📊 Signal Matrix",
    "🌊 Regime",
    "🔒 Gates",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Signal Matrix
# ════════════════════════════════════════════════════════════════════════════════
with t_matrix:
    n_long  = int((sig["tsmom"] ==  1).sum())
    n_short = int((sig["tsmom"] == -1).sum())
    n_neut  = int((sig["tsmom"] ==  0).sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("多头", n_long)
    k2.metric("空头", n_short)
    k3.metric("中性", n_neut)
    if regime:
        ri = {"risk-on":"🟢","risk-off":"🔴","transition":"🟡"}
        k4.metric("制度", f"{ri.get(regime.regime,'⚪')} {regime.regime}")
        k5.metric("P(risk-on)", f"{regime.p_risk_on:.1%}")

    st.divider()

    # ── Build display table ────────────────────────────────────────────────────
    _SC  = {1:"#22c55e", -1:"#ef4444", 0:"#94a3b8"}
    _GC  = {"hard":"#ef4444","soft":"#f59e0b","clear":"rgba(0,0,0,0)"}
    def _cc(v):
        if v is None: return "rgba(0,0,0,0)"
        return "#22c55e" if v>=70 else "#3b82f6" if v>=50 else "#f59e0b" if v>=30 else "#ef4444"

    rows = []
    for sector in sig.index:
        cur  = sig.loc[sector]
        prev = sig_prev.loc[sector] if not sig_prev.empty and sector in sig_prev.index else None
        tm, cm = int(cur["tsmom"]), int(cur["csmom"])
        tp = int(prev["tsmom"]) if prev is not None else None
        cp = int(prev["csmom"]) if prev is not None else None
        sm = {1:"▲ 多头",-1:"▼ 空头",0:"— 中性"}
        cv = float(comp.loc[sector,"composite_score"]) \
             if not comp.empty and sector in comp.index else None
        g  = gates.get(sector,{})
        bk = g.get("blocked",[])
        gl = ""
        if "超配" in bk and "标配" in bk: gl="仅低配"
        elif "超配" in bk and "低配" in bk: gl="仅标配"
        elif "超配" in bk: gl="禁超配"
        elif "低配" in bk:  gl="禁低配"
        rows.append({
            "板块":        sector,
            "ETF":         cur["ticker"],
            "资产类别":    cur.get("asset_class","—"),
            "12-1M%":      round(cur["raw_return"]*100,2),
            "波动%":       round(cur["ann_vol"]*100,1),
            "TSMOM":       tm, "TM_lbl": sm[tm],
            "CSMOM":       cm, "CM_lbl": sm[cm],
            "TM_flip":     tp is not None and tm!=tp,
            "CM_flip":     cp is not None and cm!=cp,
            "inv_wt":      round(cur.get("inv_vol_wt",0),4),
            "score":       cv,
            "gate_lbl":    gl,
            "gate_sev":    g.get("severity","clear"),
        })
    df = pd.DataFrame(rows)

    headers = ["板块","ETF","资产类别","12-1M%","波动%","TSMOM","CSMOM","合成分","门控"]
    if show_wt: headers.append("反向波动权重")

    flip_pfx = []
    for _, r in df.iterrows():
        m = []
        if r["TM_flip"]: m.append("T")
        if r["CM_flip"]: m.append("C")
        flip_pfx.append("⚡" if m else "")

    cell_vals = [
        [flip_pfx[i]+df.at[i,"板块"] for i in range(len(df))],
        df["ETF"].tolist(),
        df["资产类别"].tolist(),
        [f"{v:+.2f}%" for v in df["12-1M%"]],
        [f"{v:.1f}%"  for v in df["波动%"]],
        df["TM_lbl"].tolist(),
        df["CM_lbl"].tolist(),
        [f"{v:.0f}" if v is not None else "—" for v in df["score"]],
        [v if v else "—" for v in df["gate_lbl"]],
    ]
    cell_colors = [
        ["rgba(0,0,0,0)"]*len(df),
        ["rgba(0,0,0,0)"]*len(df),
        ["rgba(0,0,0,0)"]*len(df),
        ["#22c55e" if v>0 else "#ef4444" for v in df["12-1M%"]],
        ["rgba(0,0,0,0)"]*len(df),
        [_SC[v] for v in df["TSMOM"]],
        [_SC[v] for v in df["CSMOM"]],
        [_cc(v) for v in df["score"]],
        [_GC.get(v,"rgba(0,0,0,0)") for v in df["gate_sev"]],
    ]
    if show_wt:
        cell_vals.append([f"{v:.4f}" for v in df["inv_wt"]])
        cell_colors.append(["rgba(0,0,0,0)"]*len(df))

    # Per-cell font color: white on colored backgrounds, theme-text on transparent
    _transparent = {"rgba(0,0,0,0)", "", None}
    _hdr_bg  = "#1e293b" if _is_dark_sig else "#1e293b"  # header stays dark always
    def _fc(bg: str) -> str:
        return "white" if bg not in _transparent else _chart_font
    font_colors = [[_fc(bg) for bg in col] for col in cell_colors]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in headers],
            fill_color=_hdr_bg, font=dict(color="white", size=12),
            align="left", height=32,
        ),
        cells=dict(
            values=cell_vals, fill_color=cell_colors,
            font=dict(color=font_colors, size=11), align="left", height=28,
        ),
    )])
    fig.update_layout(
        margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        height=32+28*len(df)+16,
    )
    st.plotly_chart(fig, use_container_width=True)

    flipped = df[df["TM_flip"]|df["CM_flip"]]
    if not flipped.empty:
        st.warning(
            f"⚡ {len(flipped)} 个板块信号翻转（vs {cmp_m}个月前）：  "
            + "  |  ".join(f'{r["板块"]} {r["TM_lbl"]}' for _,r in flipped.iterrows())
        )

    # ── Within-class CSMOM grouping ───────────────────────────────────────────
    st.divider()
    st.subheader("类内 CSMOM 排序")
    for cls in sorted(df["资产类别"].dropna().unique()):
        cdf = df[df["资产类别"]==cls].sort_values("12-1M%",ascending=False)
        if cdf.empty: continue
        with st.expander(f"{cls}  ({len(cdf)} ETF)", expanded=True):
            cols = st.columns(min(len(cdf), 8))
            for ci,(_, r) in enumerate(cdf.iterrows()):
                if ci >= len(cols): break
                clr = _SC.get(r["CSMOM"],"#94a3b8")
                rc  = "#22c55e" if r["12-1M%"]>0 else "#ef4444"
                with cols[ci]:
                    _sec_c = "#64748b" if _is_dark_sig else "#475569"
                    st.markdown(
                        f'<div style="text-align:center;padding:0.45rem 0.3rem;'
                        f'border:1px solid {clr};border-radius:4px;">'
                        f'<div style="font-weight:700;color:{clr};font-size:0.95rem;">'
                        f'{r["ETF"]}</div>'
                        f'<div style="font-size:0.68rem;color:{_sec_c};'
                        f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'
                        f'{r["板块"][:8]}</div>'
                        f'<div style="font-size:0.78rem;font-weight:600;color:{rc};">'
                        f'{r["12-1M%"]:+.1f}%</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ── Return bar chart ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("12-1M 收益率排序")
    ds = df.sort_values("12-1M%")
    fig_b = go.Figure(go.Bar(
        x=ds["12-1M%"], y=ds["板块"], orientation="h",
        marker_color=["#22c55e" if v>0 else "#ef4444" for v in ds["12-1M%"]],
        text=[f"{v:+.2f}%" for v in ds["12-1M%"]], textposition="outside",
    ))
    fig_b.add_vline(x=0, line_color="#64748b", line_width=1)
    fig_b.update_layout(
        height=max(360,len(ds)*22), xaxis=dict(ticksuffix="%"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_chart_font), margin=dict(l=0,r=60,t=6,b=6),
    )
    st.plotly_chart(fig_b, use_container_width=True)

    st.caption(
        f"截止 {as_of}  ·  形成期 {lk_m}-1 月  ·  "
        f"Moskowitz, Ooi & Pedersen (2012)  ·  对比日期 {prev_date}"
    )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Regime
# ════════════════════════════════════════════════════════════════════════════════
with t_regime:
    _RC = {"risk-on":"#22c55e","risk-off":"#ef4444","transition":"#f59e0b","unknown":"#64748b"}

    n_months_r = st.slider("历史月数", 12, 120, 36, step=6, key="regime_nm")

    regime_df = None
    try:
        from engine.memory import list_structured_backtests, load_structured_backtest
        saved = list_structured_backtests()
        if saved:
            bt = load_structured_backtest(saved[0]["run_id"])
            if bt and "returns" in bt:
                rdf = bt["returns"]
                if "regime_label" in rdf.columns:
                    regime_df = rdf[["regime_label","p_risk_on","yield_spread"]].copy()
                    regime_df.index = pd.to_datetime(regime_df.index)
                    st.caption(
                        f"已加载回测数据：{saved[0]['start_date']} → {saved[0]['end_date']}")
    except Exception:
        pass

    if regime_df is None:
        with st.spinner(f"计算最近 {n_months_r} 个月制度序列…"):
            from engine.regime import get_regime_series
            end   = today
            start = today.replace(
                month=((today.month-n_months_r-1)%12)+1,
                year=today.year+((today.month-n_months_r-1)//12),
            )
            dates = []
            d = start
            while d <= end:
                dates.append(d)
                m = d.month+1; y = d.year+(m-1)//12; m=(m-1)%12+1
                d = d.replace(year=y,month=m,day=min(d.day,28))
            rs = get_regime_series(dates)
            regime_df = rs[["regime","p_risk_on","yield_spread"]].rename(
                columns={"regime":"regime_label"})
            regime_df.index = pd.to_datetime(regime_df.index)

    if regime_df is not None and not regime_df.empty:
        cnts  = regime_df["regime_label"].value_counts()
        total = len(regime_df)
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("risk-on",    f"{cnts.get('risk-on',0)} 月", f"{cnts.get('risk-on',0)/total:.0%}")
        k2.metric("risk-off",   f"{cnts.get('risk-off',0)} 月", f"{cnts.get('risk-off',0)/total:.0%}")
        k3.metric("transition", f"{cnts.get('transition',0)} 月", f"{cnts.get('transition',0)/total:.0%}")
        if regime:
            k4.metric("当前",
                      f"{'🟢' if regime.regime=='risk-on' else '🔴' if regime.regime=='risk-off' else '🟡'} "
                      f"{regime.regime}",
                      f"P={regime.p_risk_on:.1%}")

        fig_r = go.Figure()
        if "p_risk_on" in regime_df.columns:
            fig_r.add_trace(go.Scatter(
                x=regime_df.index, y=regime_df["p_risk_on"],
                mode="lines", name="P(risk-on)",
                line=dict(color="#3b82f6",width=2), yaxis="y",
            ))
            fig_r.add_hline(y=0.65, line_dash="dash", line_color="#22c55e",
                            annotation_text="0.65 risk-on", line_width=1)
            fig_r.add_hline(y=0.35, line_dash="dash", line_color="#ef4444",
                            annotation_text="0.35 risk-off", line_width=1)
        if "yield_spread" in regime_df.columns and regime_df["yield_spread"].notna().any():
            fig_r.add_trace(go.Scatter(
                x=regime_df.index, y=regime_df["yield_spread"],
                mode="lines", name="10Y-2Y 利差",
                line=dict(color="#f59e0b",width=1.5,dash="dot"),
                yaxis="y2", opacity=0.8,
            ))
        prev_r = None; sx = regime_df.index[0]
        for idx, row in regime_df.iterrows():
            r = row["regime_label"]
            if r != prev_r:
                if prev_r is not None:
                    fig_r.add_vrect(x0=sx,x1=idx,fillcolor=_RC.get(prev_r,"#64748b"),
                                    opacity=0.08,layer="below",line_width=0)
                sx=idx; prev_r=r
        fig_r.add_vrect(x0=sx,x1=regime_df.index[-1],
                        fillcolor=_RC.get(prev_r,"#64748b"),
                        opacity=0.08,layer="below",line_width=0)
        fig_r.update_layout(
            height=340,
            yaxis=dict(title="P(risk-on)",range=[0,1],tickformat=".0%"),
            yaxis2=dict(title="利差(%)",overlaying="y",side="right",
                        showgrid=False,tickformat=".2f"),
            plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=_chart_font),
            legend=dict(orientation="h",yanchor="bottom",y=1.02),
            margin=dict(l=0,r=60,t=24,b=0), hovermode="x unified",
        )
        st.plotly_chart(fig_r, use_container_width=True)

        # Band bar
        fig_band = go.Figure(go.Bar(
            x=regime_df.index,
            y=[1]*len(regime_df),
            marker_color=[_RC.get(r,"#64748b") for r in regime_df["regime_label"]],
            showlegend=False,
        ))
        fig_band.update_layout(
            height=40, margin=dict(l=0,r=60,t=0,b=0),
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False,showgrid=False),
            plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",bargap=0,
        )
        st.plotly_chart(fig_band, use_container_width=True)

        st.divider()
        st.subheader("转移矩阵")
        regimes = ["risk-on","transition","risk-off"]
        trans = pd.DataFrame(0,index=regimes,columns=regimes)
        labels = regime_df["regime_label"].tolist()
        for i in range(1,len(labels)):
            fr,to = labels[i-1],labels[i]
            if fr in regimes and to in regimes:
                trans.loc[fr,to]+=1
        tp = trans.div(trans.sum(axis=1).replace(0,1),axis=0)
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**次数**")
            st.dataframe(trans.rename(
                index={"risk-on":"🟢 risk-on","transition":"🟡 transition","risk-off":"🔴 risk-off"},
                columns={"risk-on":"→ on","transition":"→ tr","risk-off":"→ off"},
            ), use_container_width=True)
        with c2:
            st.markdown("**概率**")
            st.dataframe(tp.map(lambda x: f"{x:.0%}").rename(
                index={"risk-on":"🟢 risk-on","transition":"🟡 transition","risk-off":"🔴 risk-off"},
                columns={"risk-on":"→ on","transition":"→ tr","risk-off":"→ off"},
            ), use_container_width=True)

        st.caption(
            "Hamilton (1989) MSM  ·  BIC k∈{2,3}  ·  Filtered probability（无前视偏差）"
        )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Gates
# ════════════════════════════════════════════════════════════════════════════════
with t_gates:
    st.subheader("Quant Gate 状态  ·  R1–R5 规则")

    if not gates:
        st.info("暂无门控数据。")
    else:
        from collections import Counter
        gate_rows = []
        all_reasons = []
        for sector, g in gates.items():
            bk  = g.get("blocked",[])
            rs  = g.get("reasons",[])
            sev = g.get("severity","clear")
            all_reasons.extend(rs)
            gate_rows.append({
                "板块":     sector,
                "严重度":   {"hard":"🔴 Hard","soft":"🟡 Soft","clear":"✅ Clear"}.get(sev,sev),
                "封锁操作": " / ".join(bk) if bk else "—",
                "触发规则": " | ".join(rs[:2]) if rs else "—",
            })

        gdf = pd.DataFrame(gate_rows)
        n_hard = sum(1 for r in gate_rows if "Hard" in r["严重度"])
        n_soft = sum(1 for r in gate_rows if "Soft" in r["严重度"])
        g1,g2,g3 = st.columns(3)
        g1.metric("Hard 门控", n_hard)
        g2.metric("Soft 门控", n_soft)
        g3.metric("正常通过",  len(gate_rows)-n_hard-n_soft)

        st.divider()
        st.dataframe(gdf, use_container_width=True, hide_index=True)

        if all_reasons:
            st.divider()
            st.subheader("规则触发频率")
            rf = Counter(all_reasons)
            st.dataframe(
                pd.DataFrame([{"规则":k,"触发次数":v}
                              for k,v in rf.most_common()]),
                use_container_width=True, hide_index=True,
            )

    st.caption(
        f"截止 {as_of}  ·  Quant Gate R1–R5  ·  "
        "Hard = 绝对封锁，Soft = 降权限制"
    )
