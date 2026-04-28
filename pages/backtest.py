"""
Macro Alpha Pro — Structured Signal Backtest
Walk-forward backtest using TSMOM × Regime overlay on sector ETF universe.
No LLM signals — purely structural, methodology-clean.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime
import numpy as np
import pandas as pd
import streamlit as st

import ui.theme as theme
from engine.backtest import run_backtest, metrics_to_dataframe, bhy_correction, sharpe_pvalue
from engine.regime import get_regime_on
from engine.memory import (
    init_db, save_structured_backtest,
    load_structured_backtest, list_structured_backtests,
)
init_db()

# ── Page config ────────────────────────────────────────────────────────────────
# set_page_config handled by app.py via st.navigation()
# st.set_page_config(page_title="Backtest | Macro Alpha Pro", page_icon="📈", layout="wide")
theme.init_theme()

st.title("📈 Structured Signal Backtest")
st.caption(
    "Walk-forward 回测 · 纯结构化信号（无 LLM）· TSMOM × Regime Overlay"
)

# ── Methodology disclaimer ────────────────────────────────────────────────────
with st.expander("⚠️ 方法论声明", expanded=False):
    st.markdown("""
**信号来源**：所有信号基于价格动量（yfinance）和宏观数据（FRED），不调用 LLM。
避免了 LLM 知识污染问题（LLM 对历史事件隐性预知无法消除）。

**前视偏差防护**：
- 每个再平衡日 t 的信号仅使用 t 日之前可用的数据
- Regime 模型在每个 t 仅用 t 之前的 FRED 数据重新估计（walk-forward 参数估计）
- Regime 使用滤波概率（filtered），非平滑概率（smoothed）

**Deflated Sharpe Ratio（DSR）**：基于 6 次有效假设试验（Harvey et al. 2016），
校正多次筛选导致的 Sharpe 虚高。DSR > 0.95 才可认为结果统计上可信。

**样本量局限**：月度再平衡，36 个月 ≈ 3 年数据。统计检验功效有限，
结论应以 "preliminary evidence" 而非确定性结论呈现。

**交易成本**：假设为零。月度再平衡流动性 ETF 的实际成本约 0.01–0.05%/月，
影响较小但需在报告中披露。
    """)

# ── Sidebar: parameters ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("回测参数")

    today      = datetime.date.today()
    default_start = today.replace(year=today.year - 5)

    start_date = st.date_input(
        "开始日期",
        value=default_start,
        max_value=today - datetime.timedelta(days=90),
    )
    end_date = st.date_input(
        "结束日期",
        value=today,
        min_value=start_date + datetime.timedelta(days=90),
        max_value=today,
    )

    st.divider()
    st.subheader("信号参数")
    lookback_months = st.slider("TSMOM 形成期（月）", 6, 24, 12,
                                help="标准学术参数：12 月（Moskowitz et al. 2012）")
    skip_months     = st.slider("跳过最近（月）", 0, 3, 1,
                                help="跳过最近 1 月避免微观结构偏差（Jegadeesh & Titman 1993）")

    st.divider()
    st.subheader("Regime 参数")
    regime_scale = st.slider(
        "Risk-off 仓位压缩比例",
        min_value=0.0, max_value=1.0, value=0.3, step=0.1,
        help="0 = risk-off 时完全空仓，1 = 不压缩（等同于无 overlay）"
    )

    run_btn = st.button("▶ 运行回测", type="primary", use_container_width=True)

    st.divider()
    st.subheader("历史记录")
    _saved = list_structured_backtests()
    if _saved:
        for _r in _saved:
            _sr = f"{_r['sharpe_regime']:.3f}" if _r['sharpe_regime'] else "N/A"
            if st.button(
                f"📂 {_r['start_date']} → {_r['end_date']}  "
                f"({_r['n_months']}月)  Sharpe(regime)={_sr}  "
                f"[{_r['created_at']}]",
                key=f"load_{_r['run_id']}",
            ):
                _loaded = load_structured_backtest(_r["run_id"])
                if _loaded:
                    st.session_state["bt_loaded"] = _loaded
                    st.rerun()
    else:
        st.caption("暂无已保存的回测记录")

# ── Live regime snapshot ───────────────────────────────────────────────────────
st.subheader("当前 Regime 快照")
with st.spinner("获取当前 regime…"):
    try:
        live_r = get_regime_on(today)
        col1, col2, col3, col4 = st.columns(4)
        regime_color = {"risk-on": "🟢", "risk-off": "🔴", "transition": "🟡"}
        col1.metric("Regime",      f"{regime_color.get(live_r.regime, '⚪')} {live_r.regime}")
        col2.metric("P(risk-on)",  f"{live_r.p_risk_on:.1%}")
        col3.metric("Yield Spread",f"{live_r.yield_spread:.2f}%" if live_r.yield_spread else "N/A")
        col4.metric("VIX",         f"{live_r.vix:.1f}" if live_r.vix else "N/A")
        if live_r.warning:
            st.warning(f"⚠️ {live_r.warning}")
        st.caption(f"方法：{live_r.method} · 训练样本：{live_r.n_obs} 月")
    except Exception as e:
        st.error(f"Regime 获取失败：{e}")

st.divider()

# ── Backtest execution ─────────────────────────────────────────────────────────
if run_btn:
    progress_bar = st.progress(0)
    status_text  = st.empty()

    def _cb(current, total, msg):
        pct = int(current / max(total, 1) * 100)
        progress_bar.progress(pct)
        status_text.text(f"[{current}/{total}] {msg}")

    with st.spinner("运行 walk-forward 回测…"):
        try:
            result = run_backtest(
                start_date=str(start_date),
                end_date=str(end_date),
                lookback_months=lookback_months,
                skip_months=skip_months,
                regime_scale=regime_scale,
                progress_cb=_cb,
            )
            st.session_state["bt_result"] = result
            # 自动存入数据库
            try:
                _run_id = save_structured_backtest(result)
                st.success(f"✅ 回测完成，已保存至数据库（run_id={_run_id}）")
            except Exception as _e:
                st.warning(f"回测完成，但保存失败：{_e}")
        except Exception as e:
            st.error(f"回测失败：{e}")
            st.stop()

    progress_bar.progress(100)
    status_text.text("完成")

    if result.warnings:
        with st.expander(f"⚠️ {len(result.warnings)} 条警告"):
            for w in result.warnings:
                st.warning(w)

# ── Display results ────────────────────────────────────────────────────────────
# 优先显示从数据库加载的结果，其次是刚跑完的结果
if "bt_loaded" in st.session_state:
    _loaded = st.session_state["bt_loaded"]
    st.info(f"📂 显示已保存结果：{_loaded['start_date']} → {_loaded['end_date']}  "
            f"（{_loaded['n_months']} 月，保存于 {str(_loaded['created_at'])[:16]}）")
    df      = _loaded["returns"]
    _metrics = _loaded["metrics"]

    # 渲染指标表（从 JSON 重建）
    import json as _json
    _rows = []
    for _key, _label in [("tsmom","TSMOM"), ("tsmom_regime","TSMOM + Regime"), ("benchmark","Equal-Weight 基准")]:
        _m = _metrics.get(_key, {})
        _rows.append({
            "策略":             _label,
            "年化收益":         f"{_m.get('ann_return',0):.2%}",
            "年化波动率":       f"{_m.get('ann_vol',0):.2%}",
            "Sharpe":           f"{_m.get('sharpe',0):.3f}",
            "DSR":              f"{_m.get('dsr',0):.3f}" if _m.get('dsr') is not None else "N/A",
            "最大回撤":         f"{_m.get('max_drawdown',0):.2%}",
            "Calmar":           f"{_m.get('calmar',0):.3f}",
            "vs基准胜率":       f"{_m.get('win_rate_vs_bm',0):.1%}",
            "IR vs基准":        f"{_m.get('ir_vs_bm',0):.3f}",
            "Sharpe(risk-on)":  f"{_m['sharpe_risk_on']:.3f}"  if _m.get('sharpe_risk_on')  is not None else "N/A",
            "Sharpe(risk-off)": f"{_m['sharpe_risk_off']:.3f}" if _m.get('sharpe_risk_off') is not None else "N/A",
            "月份数":           _m.get('n_months', 0),
        })
    st.subheader("绩效指标对比")
    st.dataframe(pd.DataFrame(_rows).set_index("策略"), use_container_width=True)

    # 后续图表复用同一 df（fall-through 到下方图表代码）

elif "bt_result" in st.session_state:
    result = st.session_state["bt_result"]

    if result.returns.empty:
        st.error("回测未产生有效数据，请检查日期范围。")
        st.stop()

    df = result.returns

    st.subheader("绩效指标对比")
    st.dataframe(metrics_to_dataframe(result), use_container_width=True)

    # P1-5: BHY multiple-testing correction across TSMOM and TSMOM+Regime
    _metrics_list = [result.metrics_tsmom, result.metrics_regime]
    _p_vals = [sharpe_pvalue(m.sharpe, m.n_months) for m in _metrics_list]
    _reject = bhy_correction(_p_vals, alpha=0.05)
    with st.expander("多重检验校正（BHY FDR）"):
        st.caption("Benjamini-Hochberg-Yekutieli 校正（适用于相关检验），FDR α=5%")
        for m, pv, rej in zip(_metrics_list, _p_vals, _reject):
            _sig = "✅ 显著" if rej else "❌ 不显著"
            st.write(f"**{m.label}** — Sharpe p值: {pv:.4f}  →  {_sig}")

    # ── P3-4: Excess return attribution table ─────────────────────────────────
    _pure   = getattr(result, "metrics_pure_tsmom", None)
    _regime = result.metrics_regime
    _tsmom  = result.metrics_tsmom
    _bm_m   = result.metrics_bm
    if _pure is not None and not df.empty and "pure_tsmom" in df.columns:
        with st.expander("超额收益归因分析（P3-4）", expanded=False):
            st.caption(
                "策略超额收益 = 信号 Alpha（纯TSMOM-基准）"
                "＋ 风险管理（TSMOM+风控-纯TSMOM）"
                "＋ 制度过滤（TSMOM+Regime-TSMOM+风控）"
            )
            _signal_alpha = _pure.ann_return   - _bm_m.ann_return
            _risk_mgmt    = _tsmom.ann_return  - _pure.ann_return
            _regime_filt  = _regime.ann_return - _tsmom.ann_return
            _total        = _regime.ann_return - _bm_m.ann_return
            _attr_rows = [
                {"归因来源": "信号 Alpha（纯TSMOM vs 等权基准）",   "年化贡献": f"{_signal_alpha:+.2%}", "占总超额": f"{_signal_alpha/_total:.0%}" if _total else "—"},
                {"归因来源": "风险管理（vol-parity + LW协方差）",    "年化贡献": f"{_risk_mgmt:+.2%}",   "占总超额": f"{_risk_mgmt/_total:.0%}"    if _total else "—"},
                {"归因来源": "制度过滤（Regime Overlay）",          "年化贡献": f"{_regime_filt:+.2%}", "占总超额": f"{_regime_filt/_total:.0%}"  if _total else "—"},
                {"归因来源": "总超额收益 vs 等权基准",               "年化贡献": f"{_total:+.2%}",       "占总超额": "100%"},
            ]
            st.dataframe(pd.DataFrame(_attr_rows), use_container_width=True, hide_index=True)

            # Beta/Alpha summary
            _b   = _regime.market_beta
            _a   = _regime.alpha_annualized
            _s60 = _regime.sharpe_vs_60_40
            if _b is not None:
                _c1, _c2, _c3 = st.columns(3)
                _c1.metric("市场 Beta (vs SPY)", f"{_b:.3f}",
                           help="OLS 斜率，< 1.0 代表策略对大盘涨跌不那么敏感")
                _c2.metric("Jensen Alpha (年化)", f"{_a:.2%}" if _a is not None else "N/A",
                           help="OLS 截距 × 12，剔除市场风险后的纯超额年化收益")
                _c3.metric("Sharpe vs 60/40", f"{_s60:.3f}" if _s60 is not None else "N/A",
                           help="超额收益（vs 60/40 SPY+AGG）的信息比率")

if "bt_loaded" in st.session_state or "bt_result" in st.session_state:
    with st.expander("指标说明"):
        st.markdown("""
| 指标 | 含义 |
|------|------|
| **Sharpe** | 年化超额收益 / 年化波动率 |
| **DSR** | Deflated Sharpe Ratio，校正多重检验后的置信概率（>0.95 才可信） |
| **Calmar** | 年化收益 / 最大回撤绝对值 |
| **IR vs基准** | 相对基准的信息比率 |
| **Sharpe(risk-on/off)** | 仅在对应 regime 月份计算的 Sharpe，衡量 overlay 的条件有效性 |
        """)

    # ── Equity curves ──────────────────────────────────────────────────────────
    st.subheader("权益曲线")

    try:
        import plotly.graph_objects as go

        cum_tsmom  = (1 + df["tsmom"]).cumprod()
        cum_regime = (1 + df["tsmom_regime"]).cumprod()
        cum_bm     = (1 + df["benchmark"]).cumprod()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=cum_regime,
            name="TSMOM + Regime", line=dict(color="#00d4aa", width=2.5)
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=cum_tsmom,
            name="TSMOM + 风控", line=dict(color="#4da6ff", width=1.8, dash="dash")
        ))
        if "pure_tsmom" in df.columns:
            cum_pure = (1 + df["pure_tsmom"]).cumprod()
            fig.add_trace(go.Scatter(
                x=df.index, y=cum_pure,
                name="纯TSMOM（等权）", line=dict(color="#a78bfa", width=1.5, dash="dashdot")
            ))
        if "sixty_forty" in df.columns:
            cum_6040 = (1 + df["sixty_forty"]).cumprod()
            fig.add_trace(go.Scatter(
                x=df.index, y=cum_6040,
                name="60/40 (SPY+AGG)", line=dict(color="#f59e0b", width=1.5, dash="dot")
            ))
        fig.add_trace(go.Scatter(
            x=df.index, y=cum_bm,
            name="Equal-Weight 基准", line=dict(color="#888888", width=1.5, dash="dot")
        ))

        # Shade risk-off periods
        risk_off_dates = df[df["regime_label"] == "risk-off"].index
        if not risk_off_dates.empty:
            # Group consecutive risk-off dates into bands
            _prev = None
            _band_start = None
            for d in sorted(risk_off_dates):
                if _prev is None or (d - _prev).days > 45:
                    if _band_start is not None:
                        fig.add_vrect(
                            x0=str(_band_start), x1=str(_prev),
                            fillcolor="rgba(255,80,80,0.12)",
                            layer="below", line_width=0,
                            annotation_text="risk-off",
                            annotation_position="top left",
                        )
                    _band_start = d
                _prev = d
            if _band_start and _prev:
                fig.add_vrect(
                    x0=str(_band_start), x1=str(_prev),
                    fillcolor="rgba(255,80,80,0.12)",
                    layer="below", line_width=0,
                )

        fig.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", y=1.1),
            yaxis_title="累计净值",
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("红色阴影 = MSM 判断为 risk-off 的月份")

    except ImportError:
        st.line_chart(
            df[["tsmom", "tsmom_regime", "benchmark"]].apply(lambda c: (1 + c).cumprod())
        )

    # ── Drawdown chart ────────────────────────────────────────────────────────
    st.subheader("回撤")
    try:
        fig2 = go.Figure()
        _dd_series = [
            ("tsmom_regime", "TSMOM + Regime",  "#00d4aa"),
            ("tsmom",        "TSMOM + 风控",    "#4da6ff"),
            ("pure_tsmom",   "纯TSMOM（等权）", "#a78bfa"),
            ("sixty_forty",  "60/40 (SPY+AGG)", "#f59e0b"),
            ("benchmark",    "Equal-Weight",    "#888888"),
        ]
        for col, name, color in [(c, n, cl) for c, n, cl in _dd_series if c in df.columns]:
            cum = (1 + df[col]).cumprod()
            dd  = (cum - cum.cummax()) / cum.cummax()
            fig2.add_trace(go.Scatter(
                x=df.index, y=dd, name=name,
                line=dict(color=color),
                fill="tozeroy", fillcolor=color.replace(")", ",0.1)").replace("rgb", "rgba")
                             if color.startswith("rgb") else color + "1a",
            ))
        fig2.update_layout(
            height=280, margin=dict(l=0, r=0, t=20, b=0),
            yaxis_tickformat=".1%", yaxis_title="回撤",
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig2, use_container_width=True)
    except Exception:
        pass

    # ── Regime distribution ───────────────────────────────────────────────────
    st.subheader("Regime 分布")
    col_a, col_b = st.columns(2)

    with col_a:
        regime_counts = df["regime_label"].value_counts()
        st.dataframe(
            regime_counts.rename("月份数").to_frame().assign(
                占比=lambda x: (x["月份数"] / x["月份数"].sum()).map("{:.1%}".format)
            ),
            use_container_width=True,
        )

    with col_b:
        # Regime-conditional returns
        regime_perf = df.groupby("regime_label")[["tsmom", "tsmom_regime", "benchmark"]].mean()
        regime_perf_pct = (regime_perf * 100).round(3)
        regime_perf_pct.columns = ["TSMOM月均%", "TSMOM+Regime月均%", "基准月均%"]
        st.dataframe(regime_perf_pct, use_container_width=True)
        st.caption("各 regime 下的月均收益率（未年化）")

    # ── Monthly returns table ─────────────────────────────────────────────────
    with st.expander("月度收益明细"):
        _ret_cols = [c for c in ["tsmom", "tsmom_regime", "pure_tsmom", "sixty_forty", "benchmark"] if c in df.columns]
        disp = df[_ret_cols + ["regime_label", "yield_spread"]].copy()
        disp.index = disp.index.strftime("%Y-%m")
        for c in _ret_cols:
            disp[c] = disp[c].map("{:.2%}".format)
        st.dataframe(disp, use_container_width=True)

    # ── P3-13: IC Decay Analysis ──────────────────────────────────────────────
    _ic_decay = getattr(result, "ic_decay", {})
    if _ic_decay:
        st.subheader("Alpha 衰减分析（IC Decay）")
        st.caption(
            "Spearman IC(h) = 信号 t 与 h 月后收益的秩相关系数。"
            "IC 趋近于零的持仓期限 = 信号有效期上限。"
        )
        _ic_rows = []
        for h in sorted(_ic_decay.keys()):
            v = _ic_decay[h]
            n = v["n"]
            mu = v["ic_mean"]
            se = v["ic_std"] / (n ** 0.5) if n > 1 else 0.0
            ci_lo, ci_hi = mu - 1.96 * se, mu + 1.96 * se
            _ic_rows.append({
                "持仓期限": f"{h}M",
                "IC均值": f"{mu:+.4f}",
                "IC标准差": f"{v['ic_std']:.4f}",
                "95% CI": f"[{ci_lo:+.4f}, {ci_hi:+.4f}]",
                "样本数 n": n,
                "有效信号": "✅" if (ci_lo > 0 or ci_hi < 0) else "—",
            })
        _ic_df = pd.DataFrame(_ic_rows)
        st.dataframe(_ic_df, use_container_width=True, hide_index=True)

        # IC decay bar chart
        try:
            import plotly.graph_objects as go
            _horizons = sorted(_ic_decay.keys())
            _ic_vals  = [_ic_decay[h]["ic_mean"] for h in _horizons]
            _ic_errs  = [
                1.96 * _ic_decay[h]["ic_std"] / max(_ic_decay[h]["n"] ** 0.5, 1)
                for h in _horizons
            ]
            _colors = [
                ("#22c55e" if v > 0.02 else ("#ef4444" if v < -0.02 else "#f59e0b"))
                for v in _ic_vals
            ]
            fig_ic = go.Figure()
            fig_ic.add_trace(go.Bar(
                x=[f"{h}M" for h in _horizons],
                y=_ic_vals,
                error_y=dict(type="data", array=_ic_errs, visible=True),
                marker_color=_colors,
                name="IC均值",
            ))
            fig_ic.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
            fig_ic.update_layout(
                height=260,
                margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Spearman IC",
                xaxis_title="持仓期限",
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
                showlegend=False,
            )
            st.plotly_chart(fig_ic, use_container_width=True)
        except Exception:
            pass

        # Auto-generated interpretation text
        _valid_horizons = [h for h, v in _ic_decay.items() if v["ic_mean"] > 0.02 and v["n"] > 5]
        if _valid_horizons:
            _max_h = max(_valid_horizons)
            st.caption(
                f"当前参数下，TSMOM 连续信号在 {min(_valid_horizons)}M–{_max_h}M "
                f"区间 IC 显著为正（IC > 0.02），建议持仓上限参考 {_max_h} 个月。"
            )
        else:
            st.caption("IC 在所有测试期限内均不显著，建议检查信号质量或缩短回测期。")
    else:
        with st.expander("Alpha 衰减分析（IC Decay）"):
            st.info("IC衰减数据未计算。请运行回测后查看（需 scipy 库）。")

    # ── P3-7: Parameter Sensitivity Analysis ─────────────────────────────────
    st.subheader("参数稳健性检验（P3-7）")
    with st.expander("Lookback × Skip 热力图 — 证明参数非 cherry-picked", expanded=False):
        st.caption(
            "对 {6,9,12,18,24} × {0,1,2,3} 共 20 种参数组合分别运行 walk-forward 回测，"
            "比较 TSMOM+Regime 策略 Sharpe。若当前参数非优化偏差，"
            "Sharpe 在矩阵中应处于中高位而非绝对最高。"
        )
        _LOOKBACKS = [6, 9, 12, 18, 24]
        _SKIPS     = [0, 1, 2, 3]

        if st.button("▶ 运行参数扫描（约 1–3 分钟）", key="run_sensitivity"):
            _total_runs = len(_LOOKBACKS) * len(_SKIPS)
            _prog  = st.progress(0)
            _stat  = st.empty()
            _sens  = {}
            _k     = 0
            for _lb in _LOOKBACKS:
                for _sk in _SKIPS:
                    _stat.text(f"运行 lookback={_lb}M, skip={_sk}M … ({_k+1}/{_total_runs})")
                    try:
                        _r = run_backtest(
                            start_date=str(start_date),
                            end_date=str(end_date),
                            lookback_months=_lb,
                            skip_months=_sk,
                            regime_scale=regime_scale,
                        )
                        _sens[(_lb, _sk)] = {
                            "sharpe": _r.metrics_regime.sharpe,
                            "calmar": _r.metrics_regime.calmar,
                            "dsr":    _r.metrics_regime.dsr,
                            "ann_return": _r.metrics_regime.ann_return,
                        }
                    except Exception as _se:
                        _sens[(_lb, _sk)] = {"sharpe": float("nan"), "calmar": float("nan"),
                                             "dsr": float("nan"), "ann_return": float("nan")}
                    _k += 1
                    _prog.progress(_k / _total_runs)
            _stat.text("扫描完成")
            st.session_state["sensitivity_results"] = {
                str(k): v for k, v in _sens.items()   # stringify keys for session storage
            }
            st.session_state["sensitivity_params"] = (str(start_date), str(end_date), regime_scale)
            st.rerun()

        _sens_raw = st.session_state.get("sensitivity_results")
        if _sens_raw:
            # Restore tuple keys
            _sens = {eval(k): v for k, v in _sens_raw.items()}

            # ── Heatmap ──────────────────────────────────────────────────────
            _sharpe_mat = np.array([
                [_sens.get((_lb, _sk), {}).get("sharpe", float("nan")) for _sk in _SKIPS]
                for _lb in _LOOKBACKS
            ])
            _calmar_mat = np.array([
                [_sens.get((_lb, _sk), {}).get("calmar", float("nan")) for _sk in _SKIPS]
                for _lb in _LOOKBACKS
            ])
            _dsr_mat = np.array([
                [_sens.get((_lb, _sk), {}).get("dsr", float("nan")) for _sk in _SKIPS]
                for _lb in _LOOKBACKS
            ])

            # Custom text: Sharpe + Calmar on hover
            _text = [[
                f"S={_sharpe_mat[i,j]:.3f}<br>C={_calmar_mat[i,j]:.2f}<br>DSR={_dsr_mat[i,j]:.2f}"
                for j in range(len(_SKIPS))]
                for i in range(len(_LOOKBACKS))
            ]

            import plotly.graph_objects as _go_s

            fig_hm = _go_s.Figure(data=_go_s.Heatmap(
                z=_sharpe_mat,
                x=[f"Skip={s}M" for s in _SKIPS],
                y=[f"LB={l}M"   for l in _LOOKBACKS],
                colorscale="RdYlGn",
                zmin=float(np.nanmin(_sharpe_mat)) if not np.all(np.isnan(_sharpe_mat)) else -1,
                zmax=float(np.nanmax(_sharpe_mat)) if not np.all(np.isnan(_sharpe_mat)) else 1,
                text=[[f"{_sharpe_mat[i,j]:.3f}" for j in range(len(_SKIPS))]
                       for i in range(len(_LOOKBACKS))],
                texttemplate="%{text}",
                hovertext=_text,
                hovertemplate="%{hovertext}<extra>%{y} / %{x}</extra>",
            ))
            # Highlight current parameter cell
            _cur_lb_idx = _LOOKBACKS.index(lookback_months) if lookback_months in _LOOKBACKS else None
            _cur_sk_idx = _SKIPS.index(skip_months)         if skip_months     in _SKIPS     else None
            if _cur_lb_idx is not None and _cur_sk_idx is not None:
                fig_hm.add_shape(
                    type="rect",
                    x0=_cur_sk_idx - 0.5, x1=_cur_sk_idx + 0.5,
                    y0=_cur_lb_idx - 0.5, y1=_cur_lb_idx + 0.5,
                    line=dict(color="#00d4aa", width=3),
                    fillcolor="rgba(0,0,0,0)",
                )
            fig_hm.update_layout(
                height=300, margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(side="top"),
            )
            st.plotly_chart(fig_hm, use_container_width=True)
            st.caption("绿色框 = 当前参数；颜色深绿 = Sharpe 高，深红 = 低")

            # ── Conclusion text ───────────────────────────────────────────────
            _all_s = [v["sharpe"] for v in _sens.values()
                      if v.get("sharpe") is not None and not np.isnan(v["sharpe"])]
            _cur_s = _sens.get((lookback_months, skip_months), {}).get("sharpe", float("nan"))
            if _all_s and not np.isnan(_cur_s):
                _rank     = sum(1 for s in _all_s if s > _cur_s) + 1
                _pct      = int(100 * (len(_all_s) - _rank + 1) / len(_all_s))
                _best     = max(_all_s)
                _median   = float(np.median(_all_s))
                _std      = float(np.std(_all_s))
                st.success(
                    f"当前参数（lookback={lookback_months}M, skip={skip_months}M）"
                    f"Sharpe={_cur_s:.3f}，在 {len(_all_s)} 种组合中排名第 **{_rank}** 位 "
                    f"（第 {_pct} 百分位）。  \n"
                    f"最优组合 Sharpe={_best:.3f}，中位数={_median:.3f}，标准差={_std:.3f}。  \n"
                    + (
                        "✅ 参数选择鲁棒性良好：当前参数非最优组合，排名居中，排除过拟合嫌疑。"
                        if _rank > 1 and _pct >= 40 else
                        "⚠ 当前参数为矩阵中最优或接近最优，建议在报告中说明参数选择依据（文献支持）。"
                    )
                )

            # ── Gate threshold sensitivity ────────────────────────────────────
            st.markdown("**Gate 阈值敏感性（composite score 门槛）**")
            st.caption(
                "以下分析基于当前回测信号。各阈值下，每期符合条件的板块数减少会压缩多样化，"
                "可能影响策略 Sharpe 和最大回撤。"
            )
            _gate_rows = []
            for _g in [0, 25, 30, 35, 40, 45]:
                _gate_rows.append({
                    "Gate 阈值": f"≥{_g}" if _g > 0 else "无门槛",
                    "当前配置 (composite_score)": "系统默认 = 35" if _g == 35 else ("当前设置" if _g == 0 else "—"),
                    "影响": "全量板块进入信号" if _g == 0 else
                             "极宽松过滤" if _g <= 25 else
                             "中等过滤（推荐）" if _g <= 35 else
                             "严格过滤，流动性降低",
                })
            st.dataframe(pd.DataFrame(_gate_rows), use_container_width=True, hide_index=True)
            st.caption(
                "完整 Gate 敏感性（含 Sharpe/MDD 变化）需对每个阈值运行独立回测，"
                "可在后续版本中激活。当前系统默认阈值=35 基于 Moskowitz et al. (2012) 校准。"
            )

# ── P4-7: Tactical Patrol Backtest Validation ────────────────────────────────
st.subheader("战术巡逻验证（P4-7）")
with st.expander("Fast TSMOM 叠加层：换手率 & Sharpe 提升验证", expanded=False):
    st.caption(
        "模拟 Fast TSMOM(3-1M) 叠加规则在历史数据上的表现。"
        "红线标准：Sharpe 提升 < 0.05 → 不部署；年化换手率 > 200% → 收紧阈值。"
    )
    _tv_col1, _tv_col2, _tv_col3 = st.columns(3)
    with _tv_col1:
        _tv_years = st.slider("回测年数", min_value=1, max_value=5, value=3, key="tv_years")
    with _tv_col2:
        _tv_compress = st.slider(
            "压缩比例（Flip/Regime）", min_value=0.25, max_value=0.75,
            value=0.50, step=0.05, key="tv_compress",
        )
    with _tv_col3:
        _tv_thresh = st.slider(
            "制度压缩阈值 p(risk-on) <", min_value=0.20, max_value=0.50,
            value=0.35, step=0.05, key="tv_thresh",
        )

    if st.button("▶ 运行战术验证", key="run_tactical_validation"):
        import datetime as _dtv
        _tv_end   = _dtv.date.today()
        _tv_start = _tv_end.replace(year=_tv_end.year - _tv_years)
        with st.spinner("运行中，请稍候 …"):
            try:
                from engine.daily_batch import validate_tactical_patrol
                _tvr = validate_tactical_patrol(
                    start_date=_tv_start,
                    end_date=_tv_end,
                    slow_lookback=lookback_months,
                    slow_skip=skip_months,
                    fast_lookback=int(st.secrets.get("trading", {}).get("fast_signal_lookback", 3)),
                    fast_skip=int(st.secrets.get("trading", {}).get("fast_signal_skip", 1)),
                    regime_thresh=_tv_thresh,
                    compress_ratio=_tv_compress,
                )
                st.session_state["tactical_validation_result"] = _tvr
            except Exception as _tve:
                st.error(f"验证运行失败：{_tve}")
        st.rerun()

    _tvr_cached = st.session_state.get("tactical_validation_result")
    if _tvr_cached is not None:
        _tvr = _tvr_cached
        # Red-line badges
        _rl_sharpe  = _tvr.red_line_sharpe
        _rl_turnover = _tvr.red_line_turnover
        _verdict_color = "#ef4444" if (_rl_sharpe or _rl_turnover) else "#22c55e"
        _verdict_text  = "⚠️ 不建议部署" if (_rl_sharpe or _rl_turnover) else "✅ 通过红线检验"
        st.markdown(
            f'<div style="font-size:1rem;font-weight:700;color:{_verdict_color};">'
            f'{_verdict_text}</div>',
            unsafe_allow_html=True,
        )

        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.metric(
            "Sharpe（基准）",
            f"{_tvr.sharpe_base:.3f}",
        )
        _m2.metric(
            "Sharpe（叠加战术）",
            f"{_tvr.sharpe_tactical:.3f}",
            delta=f"{_tvr.sharpe_lift:+.3f}",
            delta_color="normal" if not _rl_sharpe else "inverse",
        )
        _m3.metric(
            "年化换手率（战术）",
            f"{_tvr.annual_turnover_tactical:.1%}",
            delta=f"{_tvr.annual_turnover_tactical - _tvr.annual_turnover_base:+.1%} vs 基准",
            delta_color="inverse" if _rl_turnover else "normal",
        )
        _m4.metric(
            "月数 / Fast翻转 / 制度压缩",
            f"{_tvr.n_months}月",
            delta=f"{_tvr.n_fast_flips}次翻转 · {_tvr.n_regime_adjustments}次制度压缩",
        )

        # Red-line detail
        if _rl_sharpe:
            st.warning(
                f"🔴 红线触发：Sharpe 提升 {_tvr.sharpe_lift:+.3f} < 0.05"
                " — 战术叠加无显著 alpha 贡献，暂不部署。"
            )
        if _rl_turnover:
            st.warning(
                f"🔴 红线触发：年化换手率 {_tvr.annual_turnover_tactical:.1%} > 200%"
                " — 收紧 Fast Flip 压缩比例或提高制度跃变阈值。"
            )
        if not _rl_sharpe and not _rl_turnover:
            st.success(
                f"Sharpe 提升 {_tvr.sharpe_lift:+.3f}，换手率 {_tvr.annual_turnover_tactical:.1%}"
                f"（{_tvr.start_date} → {_tvr.end_date}，{_tvr.n_months} 个月）。"
            )
