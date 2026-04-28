"""
Macro Alpha Pro — Decision Journal
Pending Review · Performance Analysis · Post-Mortem · Alpha Memory
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime
import streamlit as st
import pandas as pd
import ui.theme as theme
import ui.tabs as tabs

from engine.memory import (
    init_db, get_stats, get_pending_decisions_for_monitor,
    get_records_needing_review, get_clean_zone_stats,
    get_learning_patterns, get_dormant_pattern_count,
    BASELINE_HIT_RATE, MIN_ACCEPTABLE, GOOD_THRESHOLD, EXCELLENT,
    SessionFactory, DecisionLog,
    get_failure_mode_stats,
    get_daily_brief_snapshot,
)

init_db()
theme.init_theme()

today = datetime.date.today()

# ── Cached DB loaders (TTL prevents re-fetching on every page navigation) ──────
@st.cache_data(ttl=120)
def _cached_stats():
    return get_stats()

@st.cache_data(ttl=120)
def _cached_needs_review(limit: int = 20):
    rows = get_records_needing_review()
    return rows[:limit] if rows else rows

@st.cache_data(ttl=180)
def _cached_failure_modes():
    return get_failure_mode_stats()

@st.cache_data(ttl=300)
def _cached_dormant_count():
    return get_dormant_pattern_count()

@st.cache_data(ttl=300)
def _cached_lcs_records():
    with SessionFactory() as sess:
        rows = (
            sess.query(
                DecisionLog.id, DecisionLog.decision_date,
                DecisionLog.sector_name, DecisionLog.direction,
                DecisionLog.confidence_score, DecisionLog.lcs_score,
                DecisionLog.ai_label, DecisionLog.human_label,
            )
            .filter(DecisionLog.lcs_score.isnot(None), DecisionLog.human_label.isnot(None))
            .order_by(DecisionLog.decision_date.desc())
            .limit(30)
            .all()
        )
        return [{"id": r.id, "decision_date": r.decision_date, "sector_name": r.sector_name,
                 "direction": r.direction, "confidence_score": r.confidence_score,
                 "lcs_score": r.lcs_score, "ai_label": r.ai_label, "human_label": r.human_label}
                for r in rows]

@st.cache_data(ttl=300)
def _cached_era_rows(cutoff_str: str):
    from engine.memory import AlphaMemory
    cutoff = datetime.date.fromisoformat(cutoff_str)
    with SessionFactory() as se:
        rows = (
            se.query(AlphaMemory)
              .filter(AlphaMemory.source == "track_b", AlphaMemory.decision_date >= cutoff)
              .order_by(AlphaMemory.decision_date.desc())
              .limit(200)
              .all()
        )
        return [{"decision_date": r.decision_date, "sector": r.sector,
                 "llm_delta": r.llm_delta, "era_verdict": r.era_verdict,
                 "era_score": r.era_score, "era_reasoning": r.era_reasoning}
                for r in rows]

st.title("📓 Decision Journal")
st.caption("决策验证 · 绩效追踪 · 事后分析 · Alpha Memory")

# ── Drill-down context banner ──────────────────────────────────────────────────
try:
    _snap_dj = get_daily_brief_snapshot(today)
    _verified_today = getattr(_snap_dj, "n_verified_today", 0) if _snap_dj else 0
    _is_dark_dj = theme.is_dark()
    _needs_review = _cached_needs_review()
    _n_review = len(_needs_review) if _needs_review else 0
    _items_dj = []
    if _verified_today:
        _items_dj.append(
            f'<span style="color:#22c55e;font-weight:700;">今日新增验证 {_verified_today} 条</span>'
            f'<span style="color:rgba(255,255,255,0.45);"> — 查看 Performance 标签页</span>'
        )
    if _n_review:
        _items_dj.append(
            f'<span style="color:#f59e0b;font-weight:700;">{_n_review} 条待人工归因</span>'
            f'<span style="color:rgba(255,255,255,0.45);"> — 查看 Needs Review 标签页补充失败原因</span>'
        )
    if _items_dj:
        _bg_dj = "rgba(255,255,255,0.02)" if _is_dark_dj else "rgba(0,0,0,0.02)"
        st.markdown(
            f'<div style="background:{_bg_dj};border:1px solid rgba(255,255,255,0.1);'
            f'border-radius:5px;padding:0.55rem 1rem;margin-bottom:0.9rem;font-size:0.82rem;">'
            f'<span style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;'
            f'color:rgba(255,255,255,0.4);margin-right:0.6rem;">Daily Brief →</span>'
            + "  ·  ".join(_items_dj) + '</div>',
            unsafe_allow_html=True,
        )
except Exception:
    pass

t_pending, t_perf, t_postmortem, t_memory, t_era = st.tabs([
    "⏳ Pending Review",
    "📊 Performance",
    "🔍 Post-Mortem",
    "💡 Alpha Memory",
    "🔎 ERA 审计",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Pending Review
# ══════════════════════════════════════════════════════════════════════════════
with t_pending:
    st.subheader("待验证决策监控")
    tabs.render_tab7()

    st.divider()

    # Quick stats
    stats = _cached_stats()
    pending_n  = stats.get("pending", 0)
    total_n    = stats.get("total_logged", 0)
    verified_n = stats.get("total_verified", 0)
    hit_rate   = stats.get("overall_hit_rate", 0.0)
    unapplied  = stats.get("unapplied_patterns", 0)

    st.subheader("系统总览 · Alpha Memory 健康度")
    c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
    c1.metric("Total Logged",       total_n)
    c2.metric("Verified",           verified_n)
    c3.metric("Pending Verify",     pending_n)
    c4.metric("Overall Hit Rate",   f"{hit_rate:.0%}" if verified_n else "—")
    c5.metric("Unapplied Patterns", unapplied)

    # Records needing review
    try:
        review_records = _cached_needs_review(limit=20)
        if review_records:
            st.divider()
            st.subheader("需人工审核的记录")
            st.caption(f"共 {len(review_records)} 条需要标注 human_label")
            rrdf = pd.DataFrame(review_records)
            display_cols = [c for c in ["id","decision_date","sector_name","tab_type",
                                        "direction","confidence_score","ai_label"]
                            if c in rrdf.columns]
            st.dataframe(rrdf[display_cols], use_container_width=True, hide_index=True,
                         height=300)
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Performance
# ══════════════════════════════════════════════════════════════════════════════
with t_perf:
    stats = _cached_stats()
    total_verified = stats.get("total_verified", 0)
    hit_rate       = stats.get("overall_hit_rate", 0.0)
    by_tab         = stats.get("by_tab", {})
    history        = stats.get("history", [])

    p1, p2, p3 = st.columns(3)
    p1.metric("Verified Decisions", total_verified)
    p2.metric("Overall Hit Rate",
              f"{hit_rate:.0%}" if total_verified else "—",
              delta=f"{hit_rate - BASELINE_HIT_RATE:+.0%} vs baseline" if total_verified else None)
    p3.metric("Baseline (random)", f"{BASELINE_HIT_RATE:.0%}")

    st.markdown(
        f'<div style="font-size:0.88rem; color:rgba(255,255,255,0.5); margin-top:0.5rem;">'
        f'Thresholds — Baseline: <b>{BASELINE_HIT_RATE:.0%}</b> · '
        f'Acceptable: <b>{MIN_ACCEPTABLE:.0%}</b> · '
        f'Good: <b>{GOOD_THRESHOLD:.0%}</b> · '
        f'Excellent: <b>{EXCELLENT:.0%}</b> · '
        f'Triple-Barrier method (TP=1σ / SL=0.7σ / time cap)</div>',
        unsafe_allow_html=True,
    )

    if by_tab:
        st.divider()
        st.subheader("模块绩效分解")
        tab_labels = {"macro":"Macro","sector":"Sector Risk",
                      "audit":"Quant Audit","scanner":"Alpha Scanner"}
        def _rating(hr: float) -> str:
            if hr >= EXCELLENT:      return "✅ EXCELLENT"
            if hr >= GOOD_THRESHOLD: return "🔵 GOOD"
            if hr >= MIN_ACCEPTABLE: return "🟡 ACCEPTABLE"
            return "🔴 WEAK"

        mod_rows = []
        for tab_key, data in by_tab.items():
            hr = data["hit_rate"]
            mod_rows.append({
                "Module":    tab_labels.get(tab_key, tab_key),
                "Hit Rate":  f"{hr:.0%}",
                "Verified":  data["count"],
                "Avg Score": f"{data['avg']:.2f}",
                "Rating":    _rating(hr),
            })
        st.dataframe(pd.DataFrame(mod_rows), use_container_width=True, hide_index=True)

    if history:
        st.divider()
        st.subheader("已验证决策记录")
        rows = []
        for d in history:
            score = d.get("score") or 0
            flag  = "✅" if score >= EXCELLENT else ("⚠️" if score >= 0.5 else "❌")
            ret   = f"{d['return_5d']:+.1%}" if d.get("return_5d") is not None else "—"
            rows.append({
                "Date":      d["date"],
                "Module":    tab_labels.get(d["tab"], d["tab"]) if "tab" in d else "—",
                "Sector":    d.get("sector", "—"),
                "Direction": d.get("direction", "—"),
                "5D Return": ret,
                "Score":     f"{score:.2f}",
                " ":         flag,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Clean Zone stats
    try:
        cz = get_clean_zone_stats()
        if cz:
            st.divider()
            st.subheader("Clean Zone 绩效 (Post-Training)")
            cz_cols = st.columns(4)
            cz_cols[0].metric("Clean Zone 决策", cz.get("n_total", 0))
            cz_cols[1].metric("已验证", cz.get("n_verified", 0))
            cz_cols[2].metric("Hit Rate", f"{cz.get('hit_rate', 0):.0%}" if cz.get("n_verified") else "—")
            cz_cols[3].metric("vs Baseline", f"{(cz.get('hit_rate',0) - BASELINE_HIT_RATE):+.0%}"
                              if cz.get("n_verified") else "—")
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Post-Mortem
# ══════════════════════════════════════════════════════════════════════════════
with t_postmortem:
    st.subheader("失败归因分析")
    st.caption("系统性失误模式识别 · 决策质量诊断")

    try:
        failure_stats = _cached_failure_modes()
        if failure_stats:
            st.markdown("**失败模式分布**")
            fs_df = pd.DataFrame([
                {"失败类型": k, "次数": v}
                for k, v in failure_stats.items() if v > 0
            ]).sort_values("次数", ascending=False)
            if not fs_df.empty:
                st.dataframe(fs_df, use_container_width=True, hide_index=True)
        else:
            st.info("暂无失败归因数据。验证决策后将在此显示失败模式分析。")
    except Exception:
        st.info("暂无失败归因数据。")

    # Learning patterns
    st.divider()
    st.subheader("学习模式库")
    try:
        patterns = get_learning_patterns(limit=20, include_dormant=False)
        dormant_n = _cached_dormant_count()
        p1, p2 = st.columns(2)
        p1.metric("活跃模式", len(patterns))
        p2.metric("休眠模式", dormant_n)

        if patterns:
            for p in patterns[:10]:
                applied = p.get("times_applied", 0)
                last_ok = p.get("last_outcome", "—")
                with st.expander(
                    f"[{p.get('tab_type','?')}] {p.get('sector_name','?')} · "
                    f"应用 {applied} 次 · 结果 {last_ok}"
                ):
                    st.markdown(f"**模式内容**: {p.get('pattern_text', '—')}")
                    st.markdown(f"**置信度**: {p.get('confidence', 0):.0%}")
                    st.caption(f"生成于 {p.get('created_at', '—')}")
    except Exception:
        st.info("暂无学习模式。")

    # Human vs LCS comparison
    st.divider()
    st.subheader("人工标注 vs LCS 预评分对比")
    try:
        lcs_records = _cached_lcs_records()

        if lcs_records:
            lcs_df = pd.DataFrame([{
                "ID": r["id"],
                "日期": r["decision_date"],
                "板块": r["sector_name"],
                "方向": r["direction"],
                "置信度": f"{r['confidence_score']}%" if r["confidence_score"] else "—",
                "LCS分": f"{r['lcs_score']:.2f}" if r["lcs_score"] else "—",
                "AI标注": r["ai_label"],
                "人工标注": r["human_label"],
                "一致": "✅" if r["ai_label"] == r["human_label"] else "❌",
            } for r in lcs_records])
            n_match = (lcs_df["一致"] == "✅").sum()
            m1, m2 = st.columns(2)
            m1.metric("LCS vs 人工一致率", f"{n_match/len(lcs_df):.0%}",
                      f"{n_match}/{len(lcs_df)} 条")
            m2.caption("一致率 < 100% 不代表哪方更准确，差异点是研究重点。")
            st.dataframe(lcs_df, use_container_width=True, hide_index=True)
        else:
            st.info("暂无同时具备 LCS 分和人工标注的记录。")
    except Exception:
        st.info("暂无 LCS 对比数据。")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Alpha Memory
# ══════════════════════════════════════════════════════════════════════════════
with t_memory:
    tabs.render_tab6()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ERA 审计 (External Reality Audit)
# ══════════════════════════════════════════════════════════════════════════════
with t_era:
    _is_dark_era = theme.is_dark()
    _C_era = {
        "green":  "#22c55e",
        "red":    "#ef4444",
        "yellow": "#f59e0b",
        "blue":   "#60a5fa",
        "muted":  "#8b949e" if _is_dark_era else "#64748b",
        "text":   "#f0f6fc" if _is_dark_era else "#0f172a",
        "mono":   "'Courier New','JetBrains Mono',monospace",
    }
    st.markdown(
        f'<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;'
        f'color:{_C_era["muted"]};margin-bottom:0.6rem;">'
        f'ERA · External Reality Audit · 季度宏观逻辑回测</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "ERA 每季度首个交易日在后台自动运行。对 Track B 历史决策进行宏观现实核查，"
        "判断决策逻辑是否与随后宏观走势吻合。结果仅供参考，不进入下一期决策模型。"
    )

    try:
        _lookback_era = st.slider("回看天数", 30, 365, 90, step=30, key="era_lookback")
        _cutoff_era   = today - datetime.timedelta(days=_lookback_era)
        _era_rows = _cached_era_rows(str(_cutoff_era))

        if not _era_rows:
            st.info(f"过去 {_lookback_era} 天内暂无 Track B 决策记录。")
        else:
            # ── Verdict distribution summary ──────────────────────────────────
            _audited   = [r for r in _era_rows if r["era_verdict"] is not None]
            _unaudited = [r for r in _era_rows if r["era_verdict"] is None]
            _verdict_counts = {"logic_correct": 0, "lucky_guess": 0, "logic_wrong": 0}
            for r in _audited:
                _verdict_counts[r["era_verdict"]] = _verdict_counts.get(r["era_verdict"], 0) + 1

            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
            _mc1.metric("总决策数", len(_era_rows))
            _mc2.metric("已审计", len(_audited))
            _mc3.metric("待审计", len(_unaudited))
            _correct_rate = (
                _verdict_counts["logic_correct"] / len(_audited)
                if _audited else 0.0
            )
            _mc4.metric("逻辑正确率", f"{_correct_rate:.0%}")

            # Verdict badges
            if _audited:
                _badge_html = ""
                _badge_map = {
                    "logic_correct": (_C_era["green"],  "✅ 逻辑正确"),
                    "lucky_guess":   (_C_era["yellow"], "🎲 运气成分"),
                    "logic_wrong":   (_C_era["red"],    "❌ 逻辑有误"),
                }
                for _v, (_vc, _vl) in _badge_map.items():
                    _n = _verdict_counts.get(_v, 0)
                    if _n:
                        _badge_html += (
                            f'<span style="display:inline-block;padding:0.2rem 0.7rem;'
                            f'margin-right:0.5rem;background:{_vc}22;border:1px solid {_vc}55;'
                            f'border-radius:4px;font-size:0.82rem;color:{_vc};font-weight:700;">'
                            f'{_vl}  {_n}</span>'
                        )
                st.markdown(
                    f'<div style="margin:0.6rem 0 0.8rem;">{_badge_html}</div>',
                    unsafe_allow_html=True,
                )

            # ── Decision table ────────────────────────────────────────────────
            _era_df_rows = []
            for r in _era_rows:
                _era_df_rows.append({
                    "日期":    str(r["decision_date"]),
                    "板块":    r["sector"] or "—",
                    "Δ权重":   f"{float(r['llm_delta'] or 0):+.1%}",
                    "ERA裁决": r["era_verdict"] or "待审计",
                    "置信度":  f"{float(r['era_score']):.2f}" if r["era_score"] is not None else "—",
                    "理由":    (r["era_reasoning"] or "")[:60] or "—",
                })
            _era_df = pd.DataFrame(_era_df_rows)

            def _era_color_verdict(val):
                if val == "logic_correct":
                    return f"color: {_C_era['green']}; font-weight: 700"
                if val == "logic_wrong":
                    return f"color: {_C_era['red']}; font-weight: 700"
                if val == "lucky_guess":
                    return f"color: {_C_era['yellow']}"
                return f"color: {_C_era['muted']}"

            st.dataframe(
                _era_df.style.applymap(_era_color_verdict, subset=["ERA裁决"]),
                use_container_width=True,
                hide_index=True,
            )

            # ── ERA score distribution (only audited) ─────────────────────────
            if len(_audited) >= 3:
                import plotly.express as _px_era
                _score_data = [
                    {"verdict": r["era_verdict"], "score": float(r["era_score"] or 0)}
                    for r in _audited if r["era_score"] is not None
                ]
                if _score_data:
                    _sdf = pd.DataFrame(_score_data)
                    _color_disc = {
                        "logic_correct": _C_era["green"],
                        "lucky_guess":   _C_era["yellow"],
                        "logic_wrong":   _C_era["red"],
                    }
                    _fig_era = _px_era.histogram(
                        _sdf, x="score", color="verdict",
                        nbins=10, barmode="overlay",
                        color_discrete_map=_color_disc,
                        labels={"score": "ERA 置信度", "verdict": "裁决"},
                        title="ERA 置信度分布",
                    )
                    _fig_era.update_layout(
                        height=260,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color=_C_era["text"]),
                        margin=dict(l=0, r=0, t=30, b=0),
                    )
                    st.plotly_chart(_fig_era, use_container_width=True)

    except Exception as _era_exc:
        st.warning(f"ERA 数据加载失败：{_era_exc}")
