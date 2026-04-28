"""
Factor Mining — FactorMAD 自动挖掘中心
========================================
页面主线：FactorMAD 四层防御流水线
  Layer 1 → MI 污染扫描
  Layer 2 → Proposer-Critic 辩论 + 制度条件 ICIR 评估
  Layer 3 → 符号回归结构审计
  Layer 4 → Supervisor 人工裁决（Human Gate）

页面结构（Workflow-first）：
  Header   — 流水线状态仪表盘（各阶段因子数量、上次更新时间）
  Tab 1    — 待审批候选因子（Layer 4 Human Gate，最紧迫，放最前）
  Tab 2    — 生产因子库（活跃因子 ICIR + 制度条件 ICIR）
  Tab 3    — 提交候选因子（Layer 1 MI 扫描入口）
  Tab 4    — 因子诊断（IC 时序、参数敏感性，二线研究）
"""
import datetime
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

import ui.theme as theme
from engine.memory import init_db, get_daily_brief_snapshot

init_db()
theme.init_theme()
st.markdown(theme.build_css(), unsafe_allow_html=True)

# ── Shared state ──────────────────────────────────────────────────────────────
today     = datetime.date.today()
vix_input = st.session_state.get("_vix_input", 20.0)

# ── Drill-down context banner ──────────────────────────────────────────────────
try:
    _snap_fd = get_daily_brief_snapshot(today)
    _icir_month_fd = getattr(_snap_fd, "icir_month", None) if _snap_fd else None
    _this_month_fd = today.strftime("%Y-%m")
    _is_icir_day = (_icir_month_fd == _this_month_fd)
    _is_dark_fd = theme.is_dark()
    if _is_icir_day:
        _bg_fd = "rgba(34,197,94,0.06)" if _is_dark_fd else "rgba(34,197,94,0.08)"
        st.markdown(
            f'<div style="background:{_bg_fd};border:1px solid rgba(34,197,94,0.3);'
            f'border-radius:5px;padding:0.55rem 1rem;margin-bottom:0.9rem;font-size:0.82rem;">'
            f'<span style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;'
            f'color:rgba(255,255,255,0.4);margin-right:0.6rem;">自动任务 →</span>'
            f'<span style="color:#22c55e;font-weight:700;">ICIR 月度更新已完成（{_icir_month_fd}）</span>'
            f'<span style="color:rgba(255,255,255,0.45);"> — 查看 生产因子库 Tab 审核更新结果</span></div>',
            unsafe_allow_html=True,
        )
    else:
        _bg_fd2 = "rgba(255,255,255,0.02)" if _is_dark_fd else "rgba(0,0,0,0.02)"
        st.markdown(
            f'<div style="background:{_bg_fd2};border:1px solid rgba(255,255,255,0.08);'
            f'border-radius:5px;padding:0.5rem 1rem;margin-bottom:0.9rem;font-size:0.78rem;'
            f'color:#000000;">'
            f'当有 <b>因子审批</b> 事项时 Daily Brief A 区会出现待处理项 — '
            f'来此处理 Layer 4 人工裁决。ICIR 月度更新每月第一个交易日自动运行。</div>',
            unsafe_allow_html=True,
        )
except Exception:
    pass

with st.sidebar:
    st.markdown("**Factor Mining 参数**")
    as_of      = st.date_input("截止日期", value=today, max_value=today)
    asset_cls  = st.selectbox(
        "资产类别", ["equity_sector", "equity_factor", "fixed_income", "commodity", "all"],
        index=0,
    )
    _asset_cls_arg = None if asset_cls == "all" else [asset_cls]
    lookback_m = st.slider("TSMOM 形成期（月）", 3, 24, 12)
    skip_m     = st.slider("跳过最近月", 0, 3, 1)

@st.cache_data(ttl=60, show_spinner=False)
def _cached_pipeline_status():
    from engine.memory import SessionFactory, FactorDefinition, FactorICIR, DiscoveredFactor
    with SessionFactory() as s:
        active_n   = s.query(FactorDefinition).filter(FactorDefinition.active == True).count()
        pending_n  = s.query(DiscoveredFactor).filter(
            DiscoveredFactor.status.in_(["pending", "pending_further_review"])).count()
        rejected_n = s.query(DiscoveredFactor).filter(DiscoveredFactor.status == "rejected").count()
        approved_n = s.query(DiscoveredFactor).filter(DiscoveredFactor.status == "active").count()
        last_icir  = s.query(FactorICIR.calc_date).order_by(FactorICIR.calc_date.desc()).first()
    return {
        "active": active_n, "pending": pending_n,
        "rejected": rejected_n, "approved": approved_n,
        "last_icir_date": str(last_icir[0]) if last_icir else "从未运行",
    }

@st.cache_data(ttl=300, show_spinner="加载信号数据…")
def _load_factor_signals(as_of_d, lk, sk):
    from engine.signal import get_signal_dataframe, compute_composite_scores
    sig = get_signal_dataframe(as_of_d, lk, sk)
    cmp = compute_composite_scores(as_of_d, lk, sk)
    return sig.join(cmp, how="left") if not sig.empty else pd.DataFrame()

# ── FactorMAD imports ─────────────────────────────────────────────────────────
try:
    from engine.factor_mad import (
        FACTOR_REGISTRY, get_factor_mad_scores, update_icir,
        approve_factor, reject_factor, defer_factor,
        scan_mi_contamination, compute_regime_conditional_icir,
        get_all_regime_icirs, RegimeICIR,
        compute_harvey_liu_t,
    )
    from engine.memory import SessionFactory, FactorDefinition, FactorICIR, DiscoveredFactor
    _fmad_ok = True
except Exception as _fmade:
    _fmad_ok = False
    _fmade_msg = str(_fmade)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE HEADER — Pipeline status dashboard
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-bottom:1rem;">
  <div style="font-size:1.55rem;font-weight:800;letter-spacing:0.02em;">Factor Mining</div>
  <div style="font-size:0.88rem;color:var(--text-muted);margin-top:0.2rem;
              text-transform:uppercase;letter-spacing:0.1em;">
    FactorMAD · 自动因子挖掘 · 四层防御流水线
  </div>
</div>
""", unsafe_allow_html=True)

if not _fmad_ok:
    st.error(f"FactorMAD 引擎加载失败：{_fmade_msg}")
    st.stop()

# ── Pipeline status metrics ───────────────────────────────────────────────────
_ps = _cached_pipeline_status()
_active_prod        = _ps["active"]
_pending_candidates = _ps["pending"]
_rejected_cnt       = _ps["rejected"]
_approved_cnt       = _ps["approved"]
_last_icir_date     = _ps["last_icir_date"]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("生产因子（活跃）", _active_prod,
          help="Layer 4 批准并注册到 FACTOR_REGISTRY 的因子")
m2.metric("待审批候选",
          _pending_candidates,
          delta="需要处理" if _pending_candidates > 0 else None,
          delta_color="inverse" if _pending_candidates > 0 else "off")
m3.metric("历史批准",   _approved_cnt)
m4.metric("历史驳回",   _rejected_cnt)
m5.metric("ICIR 最后更新", _last_icir_date)

if _pending_candidates > 0:
    st.warning(f"⚠️  有 {_pending_candidates} 个候选因子等待 Layer 4 人工裁决，请前往「候选因子审批」Tab 处理。",
               icon="🔔")

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════════
tab_gate, tab_pool, tab_submit, tab_diag = st.tabs([
    f"⚖️  候选因子审批{'  🔴' if _pending_candidates else ''}",
    "🏭  生产因子库",
    "🧬  提交候选因子",
    "🔬  因子诊断",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Layer 4 Human Gate（候选因子审批）
# ════════════════════════════════════════════════════════════════════════════════
with tab_gate:
    st.markdown("""
    <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;
                color:var(--text-muted);margin-bottom:0.8rem;">
    Layer 4 · Supervisor 人工裁决 — 通过 Layer 1-3 的候选因子在此等待最终批准
    </div>
    """, unsafe_allow_html=True)

    with SessionFactory() as _s4:
        _pending = (
            _s4.query(DiscoveredFactor)
            .filter(DiscoveredFactor.status.in_(["pending", "pending_further_review"]))
            .order_by(DiscoveredFactor.discovered_at.desc())
            .all()
        )

    if not _pending:
        st.success("✅  当前无待审批候选因子。流水线畅通。")
    else:
        st.markdown(f"**{len(_pending)} 个候选因子等待裁决**")

        for _df_row in _pending:
            _status_icon = "🟡" if _df_row.status == "pending" else "🔵"
            _icir_t  = f"{_df_row.icir_test:.3f}"  if _df_row.icir_test  else "—"
            _icir_tr = f"{_df_row.icir_train:.3f}" if _df_row.icir_train else "—"
            _mi_r    = f"{_df_row.mi_ratio:.2f}"   if _df_row.mi_ratio   else "—"
            _audit_icon = {"positive": "✅", "neutral": "⚪", "danger": "⚠️"}.get(
                _df_row.audit_signal_type or "neutral", "⚪"
            )

            with st.expander(
                f"{_status_icon}  {_df_row.name}"
                f"  ·  ICIR_test={_icir_t}"
                f"  ·  MI_ratio={_mi_r}"
                f"  ·  Layer3={_audit_icon}",
                expanded=True,
            ):
                # ── Four-layer summary ─────────────────────────────────────────
                l1c, l2c, l3c, l4c = st.columns(4)

                with l1c:
                    mi_color = "#ef4444" if (_df_row.mi_ratio or 0) > 2.0 else "#22c55e"
                    st.markdown(f"""
                    <div style="padding:0.6rem;border-radius:4px;background:rgba(255,255,255,0.04);
                                border:1px solid rgba(255,255,255,0.08);">
                      <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;
                                  color:var(--text-muted);">Layer 1 · MI 污染</div>
                      <div style="font-size:1.3rem;font-weight:700;color:{mi_color};">
                        {_mi_r}×
                      </div>
                      <div style="font-size:0.72rem;color:var(--text-muted);">
                        {'⚠️ 疑似前视' if (_df_row.mi_ratio or 0)>2 else '✅ 通过'}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                with l2c:
                    st.markdown(f"""
                    <div style="padding:0.6rem;border-radius:4px;background:rgba(255,255,255,0.04);
                                border:1px solid rgba(255,255,255,0.08);">
                      <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;
                                  color:var(--text-muted);">Layer 2 · ICIR</div>
                      <div style="font-size:1.3rem;font-weight:700;color:var(--text);">
                        {_icir_tr} / {_icir_t}
                      </div>
                      <div style="font-size:0.72rem;color:var(--text-muted);">训练 / 测试</div>
                    </div>
                    """, unsafe_allow_html=True)

                with l3c:
                    _a3_color = {"positive": "#22c55e", "neutral": "#94a3b8",
                                 "danger": "#ef4444"}.get(_df_row.audit_signal_type or "neutral", "#94a3b8")
                    st.markdown(f"""
                    <div style="padding:0.6rem;border-radius:4px;background:rgba(255,255,255,0.04);
                                border:1px solid rgba(255,255,255,0.08);">
                      <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;
                                  color:var(--text-muted);">Layer 3 · 符号回归</div>
                      <div style="font-size:1.3rem;font-weight:700;color:{_a3_color};">
                        {_audit_icon} {(_df_row.audit_signal_type or '—').upper()}
                      </div>
                      <div style="font-size:0.72rem;color:var(--text-muted);">结构一致性</div>
                    </div>
                    """, unsafe_allow_html=True)

                with l4c:
                    _corr_val = f"{_df_row.correlation_with_existing:.2f}" \
                                if _df_row.correlation_with_existing else "—"
                    st.markdown(f"""
                    <div style="padding:0.6rem;border-radius:4px;background:rgba(255,255,255,0.04);
                                border:1px solid rgba(255,255,255,0.08);">
                      <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;
                                  color:var(--text-muted);">现有因子相关性</div>
                      <div style="font-size:1.3rem;font-weight:700;color:var(--text);">
                        {_corr_val}
                      </div>
                      <div style="font-size:0.72rem;color:var(--text-muted);">冗余度指标</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"**描述**: {_df_row.description or '—'}")

                _det1, _det2 = st.columns(2)
                if _df_row.audit_report:
                    with _det1.expander("符号回归审计报告（Layer 3）"):
                        st.text(_df_row.audit_report)
                if _df_row.debate_log:
                    with _det2.expander("Critic 辩论记录（Layer 2）"):
                        st.text(
                            _df_row.debate_log[:3000]
                            + ("..." if len(_df_row.debate_log) > 3000 else "")
                        )
                if _df_row.code_snippet:
                    with st.expander("因子代码 ⚠️ 须人工审查前视偏差后方可批准"):
                        st.code(_df_row.code_snippet, language="python")

                st.markdown("---")
                _ac1, _ac2, _ac3 = st.columns([2, 2, 2])

                with _ac1:
                    _factor_code_id = st.text_input(
                        "factor_id（注册名）",
                        key=f"fmid_{_df_row.id}",
                        placeholder="如 mom_6m_vol_adj",
                    )
                    if st.button("✅ 批准激活", key=f"approve_{_df_row.id}", type="primary"):
                        if _factor_code_id:
                            approve_factor(_df_row.id, _factor_code_id)
                            st.success(f"{_df_row.name} 已批准，factor_id={_factor_code_id}")
                            st.rerun()
                        else:
                            st.warning("请先填写 factor_id")

                with _ac2:
                    _reject_reason = st.text_input("驳回原因", key=f"rr_{_df_row.id}")
                    if st.button("❌ 驳回", key=f"reject_{_df_row.id}"):
                        reject_factor(_df_row.id, _reject_reason or "Supervisor 驳回")
                        st.info(f"{_df_row.name} 已驳回。")
                        st.rerun()

                with _ac3:
                    _defer_note = st.text_input("补充要求", key=f"dr_{_df_row.id}")
                    if st.button("🔄 要求补充验证", key=f"defer_{_df_row.id}"):
                        defer_factor(_df_row.id, _defer_note or "需补充验证")
                        st.info(f"{_df_row.name} 已标记 pending_further_review。")
                        st.rerun()

    # ── Historical record ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 历史审批记录")
    with SessionFactory() as _sh:
        _hist_rows = (
            _sh.query(DiscoveredFactor)
            .filter(DiscoveredFactor.status.in_(["active", "rejected"]))
            .order_by(DiscoveredFactor.discovered_at.desc())
            .limit(30)
            .all()
        )
    if _hist_rows:
        st.dataframe(pd.DataFrame([{
            "名称":     r.name,
            "状态":     "✅ 激活" if r.status == "active" else "❌ 驳回",
            "ICIR测试": f"{r.icir_test:.3f}" if r.icir_test else "—",
            "MI ratio": f"{r.mi_ratio:.2f}"  if r.mi_ratio  else "—",
            "Layer3":   r.audit_signal_type or "—",
            "发现日":   str(r.discovered_at.date()) if r.discovered_at else "—",
        } for r in _hist_rows]), use_container_width=True, hide_index=True)
    else:
        st.caption("暂无已完成审核记录。")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — 生产因子库
# ════════════════════════════════════════════════════════════════════════════════
with tab_pool:
    st.markdown("""
    <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;
                color:var(--text-muted);margin-bottom:0.8rem;">
    生产因子库 · 活跃因子 ICIR 监控 · 制度条件分析
    </div>
    """, unsafe_allow_html=True)

    with SessionFactory() as _sp:
        _prod_factors = _sp.query(FactorDefinition).order_by(FactorDefinition.id).all()
        _icir_latest: dict = {}
        for fdef in _prod_factors:
            row = (
                _sp.query(FactorICIR.icir_12m, FactorICIR.ic_value, FactorICIR.calc_date)
                .filter(FactorICIR.factor_id == fdef.factor_id,
                        FactorICIR.icir_12m.isnot(None))
                .order_by(FactorICIR.calc_date.desc())
                .first()
            )
            _icir_latest[fdef.factor_id] = row

    if not _prod_factors:
        st.info("尚无注册因子。提交并批准候选因子后将在此展示。")
    else:
        # ── ICIR overview bar chart ───────────────────────────────────────────
        _pool_rows = []
        for fdef in _prod_factors:
            r = _icir_latest.get(fdef.factor_id)
            _hl_t = None
            if _fmad_ok:
                try:
                    _hl_t = compute_harvey_liu_t(
                        fdef.factor_id, as_of,
                        asset_class=asset_cls if asset_cls != "all" else "equity_sector",
                    )
                except Exception:
                    pass
            _pool_rows.append({
                "factor_id":  fdef.factor_id,
                "描述":       fdef.description or fdef.factor_id,
                "状态":       "活跃" if fdef.active else "停用",
                "ICIR_12m":   r[0] if r else None,
                "Harvey-Liu t": round(_hl_t, 2) if _hl_t is not None else None,
                "最新IC":     r[1] if r else None,
                "更新日":     str(r[2]) if r else "—",
            })
        pool_df = pd.DataFrame(_pool_rows)
        active_pool = pool_df[pool_df["状态"] == "活跃"].copy()

        if not active_pool.empty and active_pool["ICIR_12m"].notna().any():
            _sorted_pool = active_pool.sort_values("ICIR_12m", ascending=True, na_position="last")
            _colors_pool = [
                "#22c55e" if (v or 0) >= 0.3 else
                "#60a5fa" if (v or 0) >= 0.15 else
                "#f59e0b" if (v or 0) >= 0   else "#ef4444"
                for v in _sorted_pool["ICIR_12m"]
            ]
            fig_pool = go.Figure(go.Bar(
                x=_sorted_pool["ICIR_12m"],
                y=_sorted_pool["factor_id"],
                orientation="h",
                marker_color=_colors_pool,
                text=[f"{v:.3f}" if v is not None else "—" for v in _sorted_pool["ICIR_12m"]],
                textposition="outside",
            ))
            fig_pool.add_vline(x=0.15, line_dash="dash", line_color="#f59e0b",
                               annotation_text="生存线 0.15")
            fig_pool.add_vline(x=0.30, line_dash="dot", line_color="#22c55e",
                               annotation_text="优质 0.30")
            fig_pool.update_layout(
                height=max(200, len(_sorted_pool) * 45 + 80),
                xaxis_title="滚动12月 ICIR", yaxis_title="",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                margin=dict(l=0, r=60, t=20, b=0),
            )
            st.plotly_chart(fig_pool, use_container_width=True)

        _pool_display = pool_df.rename(columns={
            "factor_id": "Factor ID",
            "ICIR_12m": "ICIR 12M",
            "最新IC": "最新 IC",
        })
        _hl_caption = (
            "Harvey-Liu t = ICIR × √n  ·  t > 3.0 建议纳入  ·  展示用，生存线仍以 ICIR≥0.3 为准"
            if "Harvey-Liu t" in _pool_display.columns else ""
        )
        st.dataframe(_pool_display, use_container_width=True, hide_index=True)
        if _hl_caption:
            st.caption(_hl_caption)

        if st.button("🔄  运行月度 ICIR 更新", key="pool_icir_update"):
            with st.spinner("计算中…"):
                update_icir(as_of, asset_class=asset_cls if asset_cls != "all" else "equity_sector")
            st.success(f"ICIR 已更新（截止 {as_of}）。")
            st.rerun()

    # ── Regime-Conditional ICIR ───────────────────────────────────────────────
    st.divider()
    st.markdown("#### 制度条件 ICIR — 因子制度敏感性")
    st.caption(
        "VIX < 18 = risk-on  ·  VIX > 25 = risk-off  ·  "
        "delta = ICIR_on − ICIR_off  ·  |delta| ≥ 0.20 为制度依赖型因子"
    )

    if st.button("计算制度条件 ICIR", key="regime_icir_btn"):
        with st.spinner("分析历史 IC 记录…"):
            _regime_icirs = get_all_regime_icirs(
                as_of=as_of,
                n_months=36,
                asset_class=asset_cls if asset_cls != "all" else "equity_sector",
            )
        if not _regime_icirs:
            st.info("历史 IC 记录不足（需先运行月度 ICIR 更新 ≥ 6 次）。")
        else:
            _ri_rows = [{
                "Factor":       r.factor_id,
                "ICIR (risk-on)":  f"{r.icir_risk_on:.3f}"  if r.icir_risk_on  is not None else "—",
                "ICIR (risk-off)": f"{r.icir_risk_off:.3f}" if r.icir_risk_off is not None else "—",
                "ICIR (全样本)":   f"{r.icir_full:.3f}"     if r.icir_full     is not None else "—",
                "Delta":           f"{r.delta:+.3f}"         if r.delta         is not None else "—",
                "制度依赖":        "✅ 是" if r.is_regime_dependent else "—",
                "解读":            r.interpretation,
            } for r in _regime_icirs]
            st.dataframe(pd.DataFrame(_ri_rows), use_container_width=True, hide_index=True)

            # Delta bar chart
            _ri_with_delta = [r for r in _regime_icirs if r.delta is not None]
            if _ri_with_delta:
                _fi = [r.factor_id for r in _ri_with_delta]
                _dv = [r.delta for r in _ri_with_delta]
                fig_delta = go.Figure(go.Bar(
                    x=_fi, y=_dv,
                    marker_color=["#22c55e" if d >= 0 else "#ef4444" for d in _dv],
                    text=[f"{d:+.3f}" for d in _dv],
                    textposition="outside",
                ))
                fig_delta.add_hline(y=0.20,  line_dash="dot", line_color="#22c55e",
                                    annotation_text="+0.20 (risk-on型阈值)")
                fig_delta.add_hline(y=-0.20, line_dash="dot", line_color="#ef4444",
                                    annotation_text="-0.20 (risk-off型阈值)")
                fig_delta.add_hline(y=0, line_color="#475569", line_width=1)
                fig_delta.update_layout(
                    height=280, yaxis_title="ICIR delta (on − off)",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"),
                    margin=dict(l=0, r=0, t=20, b=40),
                )
                st.plotly_chart(fig_delta, use_container_width=True)
                st.caption(
                    "正值 → 该因子在低 VIX 环境（risk-on）更有效，适合配置到 risk-on 因子池；"
                    "负值 → 在高 VIX 环境（risk-off）更有效，适合配置到 risk-off 因子池。"
                )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — 提交候选因子（Layer 1 MI 扫描入口）
# ════════════════════════════════════════════════════════════════════════════════
with tab_submit:
    st.markdown("""
    <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;
                color:var(--text-muted);margin-bottom:0.8rem;">
    提交新候选因子 → Layer 1 MI 污染扫描 → 进入审批队列
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **提交流程：**
    1. 在下方输入因子逻辑（Python 函数，签名必须为 `fn(prices: pd.DataFrame) -> pd.Series`）
    2. 点击「运行 Layer 1 扫描」— 系统自动检测前视偏差（MI ratio）
    3. 若通过 MI 扫描，因子进入待审批队列（Layer 2-4 由 Supervisor 手动触发）

    **防前视规则：**  因子函数只能使用 `prices` 中的历史价格，禁止访问外部数据。
    `prices` 的最后一行是评估日 T，函数不得使用 T+1 之后的数据。
    """)

    _factor_name = st.text_input("候选因子名称", placeholder="如 Vol-Adjusted 6M Momentum")
    _factor_desc = st.text_area("因子描述（理论依据）",
                                placeholder="基于 Moskowitz et al. (2012)…", height=80)
    _factor_code = st.text_area(
        "因子代码（Python）",
        height=200,
        placeholder=(
            "def candidate_factor(prices: pd.DataFrame) -> pd.Series:\n"
            "    # prices: 行=日期, 列=ticker\n"
            "    # 返回: pd.Series, index=ticker, values=因子值（越大越看多）\n"
            "    if len(prices) < 130:\n"
            "        return pd.Series(dtype=float)\n"
            "    ret = prices.iloc[-130] / prices.iloc[-22] - 1\n"
            "    vol = prices.pct_change().iloc[-22:].std()\n"
            "    return ret / vol.replace(0, float('nan'))\n"
        ),
        key="submit_code",
    )

    if st.button("▶  运行 Layer 1 MI 污染扫描", type="primary", key="run_layer1"):
        if not _factor_name or not _factor_code:
            st.warning("请填写因子名称和代码。")
        else:
            with st.spinner("下载价格数据 + 运行 MI 扫描（约 20-30 秒）…"):
                try:
                    import io, contextlib
                    _ns: dict = {}
                    exec(_factor_code, {"pd": pd, "np": np}, _ns)  # noqa: S102
                    _candidate_fn = next(
                        (v for v in _ns.values() if callable(v)), None
                    )
                    if _candidate_fn is None:
                        st.error("代码中未找到可调用函数。")
                    else:
                        from engine.universe_manager import get_active_universe
                        import yfinance as yf
                        _tickers = list(get_active_universe(
                            asset_classes=_asset_cls_arg
                        ).values())
                        _start = as_of - datetime.timedelta(days=500)
                        _dl = yf.download(
                            _tickers, start=str(_start), end=str(as_of),
                            progress=False, auto_adjust=True,
                        )
                        _prices = (_dl["Close"] if "Close" in _dl else _dl).dropna(how="all")
                        if isinstance(_prices.columns, pd.MultiIndex):
                            _prices.columns = [c[0] for c in _prices.columns]

                        _scan_result = scan_mi_contamination(
                            _candidate_fn, _prices, train_end=as_of
                        )

                        _flagged = _scan_result["flagged"]
                        _ratio   = _scan_result["ratio"]

                        if _flagged:
                            st.error(
                                f"🚨  Layer 1 未通过：MI ratio={_ratio:.2f}（阈值 2.0×）\n\n"
                                f"{_scan_result['reason']}\n\n"
                                "该因子疑似包含统计前视偏差，不进入审批队列。"
                            )
                        else:
                            st.success(
                                f"✅  Layer 1 通过：MI ratio={_ratio:.2f}，"
                                f"候选 MI={_scan_result['candidate_mi']:.4f}"
                            )
                            from engine.memory import DiscoveredFactor
                            with SessionFactory() as _sw:
                                _new = DiscoveredFactor(
                                    name=_factor_name,
                                    description=_factor_desc,
                                    status="pending",
                                    mi_ratio=_ratio,
                                    code_snippet=_factor_code,
                                )
                                _sw.add(_new)
                                _sw.commit()
                            st.info("✅  候选因子已加入待审批队列，请前往「候选因子审批」处理 Layer 4。")
                except SyntaxError as _syn:
                    st.error(f"代码语法错误：{_syn}")
                except Exception as _e:
                    st.error(f"扫描失败：{_e}")

    st.divider()
    st.markdown("#### 内置基准因子（FACTOR_REGISTRY）")
    st.caption("这些因子已通过四层审计并注册到生产环境，作为 MI 扫描的校准白名单。")
    for _fid, _fn in FACTOR_REGISTRY.items():
        st.markdown(f"- `{_fid}` — {getattr(_fn, '__doc__', '') or '—'}")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — 因子诊断（参数敏感性 + 截面分析）
# ════════════════════════════════════════════════════════════════════════════════
with tab_diag:
    st.markdown("""
    <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;
                color:var(--text-muted);margin-bottom:0.8rem;">
    因子诊断 · 截面快照 · 参数敏感性 · 辅助研究
    </div>
    """, unsafe_allow_html=True)

    try:
        factor_df   = _load_factor_signals(as_of, lookback_m, skip_m)
        _signals_ok = not factor_df.empty
    except Exception:
        factor_df   = pd.DataFrame()
        _signals_ok = False

    if not _signals_ok:
        st.error("信号数据加载失败，请检查网络连接。")
    else:
        # ── Current cross-section composite scores ────────────────────────────
        if "composite_score" in factor_df.columns:
            st.markdown("#### 合成分截面（当前参数）")
            _cs = factor_df["composite_score"].sort_values(ascending=False)
            _bar_colors = [
                "#22c55e" if v >= 70 else
                "#60a5fa" if v >= 50 else
                "#f59e0b" if v >= 30 else "#ef4444"
                for v in _cs
            ]
            fig_cs = go.Figure(go.Bar(
                x=_cs.index, y=_cs.values,
                marker_color=_bar_colors,
                text=[f"{v:.0f}" for v in _cs.values],
                textposition="outside",
            ))
            fig_cs.add_hline(y=50, line_dash="dot", line_color="#64748b",
                             annotation_text="中性 50")
            fig_cs.update_layout(
                height=280, yaxis_title="合成分 (0-100)", yaxis_range=[0, 115],
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                margin=dict(l=0, r=0, t=10, b=40),
                xaxis_tickangle=-35,
            )
            st.plotly_chart(fig_cs, use_container_width=True)

        # ── Factor correlation heatmap ────────────────────────────────────────
        st.markdown("#### 因子截面相关性")
        _num_cols = [c for c in
                     ["tsmom", "csmom", "ann_vol", "sharpe_raw", "carry_raw", "composite_score"]
                     if c in factor_df.columns]
        if len(_num_cols) >= 3:
            _corr = factor_df[_num_cols].corr()
            fig_corr = px.imshow(
                _corr, color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, text_auto=".2f",
            )
            fig_corr.update_layout(
                height=380,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        # ── Parameter sensitivity scan ────────────────────────────────────────
        st.divider()
        st.markdown("#### 参数敏感性扫描")
        _p_looks = st.multiselect("形成期（月）", [3, 6, 9, 12, 18, 24], default=[6, 12, 18])
        _p_skips = st.multiselect("跳过月",       [0, 1, 2, 3],          default=[0, 1])

        if st.button("运行敏感性扫描", key="diag_sens"):
            if not _p_looks or not _p_skips:
                st.warning("请至少选择一个形成期和跳过月。")
            else:
                _sens_rows: list[dict] = []
                _total = len(_p_looks) * len(_p_skips)
                _prog  = st.progress(0)
                _i = 0
                for _lk in _p_looks:
                    for _sk in _p_skips:
                        try:
                            _c = compute_composite_scores(as_of, _lk, _sk)
                            if not _c.empty and "composite_score" in _c.columns:
                                for _sec in _c.index:
                                    _sens_rows.append({
                                        "形成期": _lk, "跳过月": _sk,
                                        "板块": _sec,
                                        "合成分": float(_c.loc[_sec, "composite_score"]),
                                        "参数":   f"L{_lk}-S{_sk}",
                                    })
                        except Exception:
                            pass
                        _i += 1
                        _prog.progress(_i / _total)

                if _sens_rows:
                    _res = pd.DataFrame(_sens_rows)
                    st.session_state["diag_sensitivity"] = _res
                    st.success(f"扫描完成：{len(_p_looks)*len(_p_skips)} 组参数")

        if "diag_sensitivity" in st.session_state:
            _res = st.session_state["diag_sensitivity"]
            _pivot = _res.pivot_table(index="板块", columns="参数",
                                      values="合成分", aggfunc="mean")
            fig_heat = px.imshow(
                _pivot, color_continuous_scale="RdYlGn",
                zmin=0, zmax=100, text_auto=".0f", aspect="auto",
            )
            fig_heat.update_layout(
                height=max(350, len(_pivot) * 22 + 80),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            _stab = (
                _res.groupby("板块")["合成分"]
                .agg(均值="mean", 标准差="std", 最小="min", 最大="max")
                .sort_values("均值", ascending=False)
                .round(1)
            )
            _stab["稳定性"] = _stab["标准差"].apply(
                lambda x: "🟢 稳定" if x < 10 else ("🟡 中等" if x < 20 else "🔴 不稳定")
            )
            st.dataframe(_stab, use_container_width=True)

st.divider()
st.caption(
    f"Factor Mining  ·  FactorMAD P2-13  ·  截止 {as_of}  ·  "
    "IC/ICIR 使用 Spearman 秩相关  ·  Macro Alpha Pro"
)
