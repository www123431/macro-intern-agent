"""
Macro Alpha Pro — Clean Zone Performance Dashboard
Primary evidence dashboard for the system's foreknowledge-free live decisions.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import ui.theme as theme
from engine.memory import get_clean_zone_stats, get_clean_zone_time_series, EXCELLENT

# set_page_config handled by app.py via st.navigation()
# Design system and header rendered by app.py for all pages via st.navigation()

@st.cache_data(ttl=300, show_spinner=False)
def _cached_cz_stats():
    return get_clean_zone_stats()

@st.cache_data(ttl=300, show_spinner=False)
def _cached_cz_ts():
    return get_clean_zone_time_series()

@st.cache_data(ttl=300, show_spinner=False)
def _cached_cz_breakdown():
    from engine.memory import SessionFactory, DecisionLog, CLEAN_ZONE_START
    with SessionFactory() as s:
        rows = (
            s.query(DecisionLog)
            .filter(
                DecisionLog.verified == True,
                DecisionLog.accuracy_score.isnot(None),
                DecisionLog.decision_date >= CLEAN_ZONE_START,
                (DecisionLog.superseded == False) | DecisionLog.superseded.is_(None),
                (DecisionLog.is_backtest == False) | DecisionLog.is_backtest.is_(None),
            )
            .all()
        )
        return [{"tab_type": r.tab_type, "accuracy_score": r.accuracy_score,
                 "macro_regime": r.macro_regime} for r in rows]

@st.cache_data(ttl=300, show_spinner=False)
def _cached_cz_calibration():
    from engine.memory import SessionFactory, DecisionLog, CLEAN_ZONE_START
    with SessionFactory() as s:
        rows = (
            s.query(DecisionLog)
            .filter(
                DecisionLog.verified == True,
                DecisionLog.accuracy_score.isnot(None),
                DecisionLog.confidence_score.isnot(None),
                DecisionLog.decision_date >= CLEAN_ZONE_START,
                (DecisionLog.superseded == False) | DecisionLog.superseded.is_(None),
                (DecisionLog.is_backtest == False) | DecisionLog.is_backtest.is_(None),
            )
            .all()
        )
        return [{"confidence_score": r.confidence_score, "accuracy_score": r.accuracy_score}
                for r in rows]

@st.cache_data(ttl=300, show_spinner=False)
def _cached_cz_payoff():
    from engine.memory import SessionFactory, DecisionLog, CLEAN_ZONE_START
    with SessionFactory() as s:
        rows = (
            s.query(DecisionLog)
            .filter(
                DecisionLog.verified == True,
                DecisionLog.payoff_quality.isnot(None),
                DecisionLog.barrier_return.isnot(None),
                DecisionLog.decision_date >= CLEAN_ZONE_START,
                (DecisionLog.superseded == False) | DecisionLog.superseded.is_(None),
                (DecisionLog.is_backtest == False) | DecisionLog.is_backtest.is_(None),
            )
            .all()
        )
        return [{"payoff_quality": r.payoff_quality, "barrier_return": r.barrier_return,
                 "confidence_score": r.confidence_score}
                for r in rows]

st.markdown(
    '<div style="font-size:1.6rem; font-weight:800; letter-spacing:0.04em; '
    'margin-bottom:0.2rem;">Clean Zone · Performance Evidence</div>'
    '<div style="font-size:0.85rem; color:var(--muted); margin-bottom:1.6rem;">'
    '唯一有效绩效证据 · 决策日期 ≥ 2025-04-01 · LLM 无历史预知</div>',
    unsafe_allow_html=True,
)

# ── Load data ─────────────────────────────────────────────────────────────────
cz    = _cached_cz_stats()
clean = cz.get("clean_b", {})
ts    = _cached_cz_ts()

n         = clean.get("n", 0)
hit_rate  = clean.get("hit_rate")
avg_acc   = clean.get("avg_accuracy")
brier     = clean.get("brier_score")
bp        = clean.get("binom_pvalue")
ci_lo     = clean.get("binom_ci_lo")
ci_hi     = clean.get("binom_ci_hi")
lcs_pr    = clean.get("lcs_pass_rate")


def _wilson_ci(n: int, k: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — more accurate than normal approximation for small n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom  = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / denom
    return max(0.0, center - margin), min(1.0, center + margin)


# Always compute Wilson CI (works at any n, unlike normal approximation)
_wins        = int(round((hit_rate or 0) * n))
_wci_lo, _wci_hi = _wilson_ci(n, _wins) if n > 0 else (0.0, 1.0)
_wci_str     = f"95% CI  [{_wci_lo:.0%}, {_wci_hi:.0%}]" if n > 0 else "—"
_wci_width   = _wci_hi - _wci_lo  # interval width: diagnostic for how informative the estimate is

# ── Top KPI strip ─────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Clean Zone 样本", n, help="verified=True 的 Clean Zone 决策数")
k2.metric("胜率 (≥0.75)",
          f"{hit_rate:.0%}" if hit_rate is not None else "—",
          help=(
              f"accuracy_score ≥ 0.75（Triple-Barrier TP）的比例\n\n"
              f"Wilson 95% CI: {_wci_str}\n"
              f"区间宽度 {_wci_width:.0%}——{'⚠ 样本不足，估计不可靠' if _wci_width > 0.5 else '估计有参考价值'}"
          ))
k3.metric("平均准确率",
          f"{avg_acc:.3f}" if avg_acc is not None else "—",
          help="accuracy_score 均值，0–1 scale")
k4.metric("Brier Score",
          f"{brier:.3f}" if brier is not None else "N/A",
          help="置信度校准误差，越低越好，0=完美校准。仅含 confidence_score 非空记录。")
k5.metric("LCS 通过率",
          f"{lcs_pr:.0%}" if lcs_pr is not None else "N/A",
          help="逻辑一致性审计通过比例")

st.divider()

# ── Statistical significance ──────────────────────────────────────────────────
_ACCENT = "#10B981"
st.markdown(
    f'<div style="font-size:1rem; font-weight:700; color:{_ACCENT}; '
    f'text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.8rem;">'
    f'Statistical Significance</div>',
    unsafe_allow_html=True,
)

if n == 0:
    st.info("尚无 Clean Zone 验证数据。")
elif n < 30:
    _hr_disp = f"{hit_rate:.0%}" if hit_rate is not None else "—"
    st.warning(
        f"**样本不足（n={n}）** — 当前胜率 {_hr_disp} 的 Wilson 95% 置信区间为 **{_wci_str}**，"
        f"区间宽度 **{_wci_width:.0%}**，无法区分真实能力与随机噪声。\n\n"
        f"以 2–4 次/年的决策频率，首次有意义的统计检验需要再积累 **{30 - n}** 个样本"
        f"（约 {max(1, (30 - n + 1) // 3)}-{max(1, 30 - n)} 年）。"
        f"在此之前，胜率数字仅供参考，不构成有效绩效证据。"
    )
else:
    _sig_label = "✓ 显著" if bp < 0.05 else ("~ 边缘显著" if bp < 0.10 else "✗ 不显著")
    _sig_color = "#10B981" if bp < 0.05 else ("#F59E0B" if bp < 0.10 else "#EF4444")
    _pval_str  = f"{bp:.4f}" if bp is not None else "—"
    _ci_str    = f"[{ci_lo:.1%}, {ci_hi:.1%}]" if ci_lo is not None else "—"

    st.markdown(
        f'<div style="border:1px solid {_sig_color}; border-radius:8px; '
        f'padding:1rem 1.4rem; display:flex; gap:3rem; align-items:center;">'
        f'<div><div style="font-size:0.72rem; color:var(--muted); '
        f'text-transform:uppercase; letter-spacing:0.07em;">结论</div>'
        f'<div style="font-size:1.5rem; font-weight:800; color:{_sig_color};">'
        f'{_sig_label}</div></div>'
        f'<div><div style="font-size:0.72rem; color:var(--muted); '
        f'text-transform:uppercase; letter-spacing:0.07em;">p-value</div>'
        f'<div style="font-family:var(--mono); font-size:1.3rem; font-weight:700; '
        f'color:var(--text);">{_pval_str}</div></div>'
        f'<div><div style="font-size:0.72rem; color:var(--muted); '
        f'text-transform:uppercase; letter-spacing:0.07em;">95% CI (胜率)</div>'
        f'<div style="font-family:var(--mono); font-size:1.3rem; font-weight:700; '
        f'color:var(--text);">{_ci_str}</div></div>'
        f'<div style="font-size:0.82rem; color:var(--muted); max-width:280px;">'
        f'H₀: 胜率 = 50%（随机基线）<br>单侧检验（alternative: greater）<br>'
        f'Exact Binomial · n={n} · wins={int((hit_rate or 0)*n)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Time series chart ─────────────────────────────────────────────────────────
_col_l, _col_r = st.columns([3, 2], gap="large")

with _col_l:
    st.markdown(
        '<div style="font-size:1rem; font-weight:700; color:var(--text); '
        'text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.8rem;">'
        'Monthly Win Rate</div>',
        unsafe_allow_html=True,
    )
    if not ts:
        st.caption("暂无时序数据。")
    else:
        import pandas as pd
        df_ts = pd.DataFrame(ts)
        df_ts = df_ts.set_index("month")

        # Win rate line
        import streamlit as _st
        st.line_chart(
            df_ts[["win_rate"]].rename(columns={"win_rate": "胜率"}),
            color=["#10B981"],
            height=240,
        )

        # Sample count bar
        st.caption("每月样本量")
        st.bar_chart(
            df_ts[["n"]].rename(columns={"n": "样本数"}),
            color=["#6366F1"],
            height=100,
        )

with _col_r:
    st.markdown(
        '<div style="font-size:1rem; font-weight:700; color:var(--text); '
        'text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.8rem;">'
        'Breakdown</div>',
        unsafe_allow_html=True,
    )

    # By tab_type
    _tab_data = cz.get("clean_b", {})
    if ts:
        import pandas as pd
        _recs = _cached_cz_breakdown()

        if _recs:
            # By tab_type
            from collections import defaultdict
            _by_tab: dict[str, list] = defaultdict(list)
            for r in _recs:
                _by_tab[r["tab_type"] or "unknown"].append(r["accuracy_score"])

            _tab_rows = []
            for tab_name, scores in sorted(_by_tab.items()):
                _n   = len(scores)
                _wr  = sum(1 for s in scores if s >= EXCELLENT) / _n
                _tab_rows.append({"模块": tab_name, "n": _n, "胜率": f"{_wr:.0%}"})

            st.caption("按分析模块")
            st.dataframe(
                pd.DataFrame(_tab_rows),
                hide_index=True,
                use_container_width=True,
            )

            # By macro_regime
            _by_regime: dict[str, list] = defaultdict(list)
            for r in _recs:  # noqa — same list as breakdown
                _by_regime[r["macro_regime"] or "未知"].append(r["accuracy_score"])

            _reg_rows = []
            for regime, scores in sorted(_by_regime.items()):
                _n  = len(scores)
                _wr = sum(1 for s in scores if s >= EXCELLENT) / _n
                _reg_rows.append({"宏观制度": regime, "n": _n, "胜率": f"{_wr:.0%}"})

            st.caption("按宏观制度")
            st.dataframe(
                pd.DataFrame(_reg_rows),
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.caption("暂无数据。")

st.divider()

# ── Confidence calibration ────────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:1rem; font-weight:700; color:var(--text); '
    'text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;">'
    'Confidence Calibration</div>'
    '<div style="font-size:0.8rem; color:var(--muted); margin-bottom:0.8rem;">'
    '理想状态：置信度=X% 的判断，实际准确率应接近 X%（对角线）。'
    'Brier Score 越低，校准越好。</div>',
    unsafe_allow_html=True,
)

if not ts:
    st.caption("暂无数据。")
else:
    import pandas as pd
    _cal_recs = _cached_cz_calibration()

    if not _cal_recs:
        st.caption("暂无含置信度的记录（confidence_score 需 XAI block 解析）。")
    else:
        # Bucket into deciles
        import math
        _buckets: dict[int, list] = {}
        for r in _cal_recs:
            _bucket = min(9, r["confidence_score"] // 10)
            _buckets.setdefault(_bucket, []).append(r["accuracy_score"])

        _cal_rows = []
        for b in sorted(_buckets):
            scores = _buckets[b]
            _cal_rows.append({
                "置信度区间": f"{b*10}–{b*10+9}%",
                "样本数": len(scores),
                "实际准确率": round(sum(scores) / len(scores), 3),
                "理想值": round((b * 10 + 5) / 100, 3),
            })

        df_cal = pd.DataFrame(_cal_rows)
        st.dataframe(df_cal, hide_index=True, use_container_width=True)
        st.caption(
            f"Brier Score = {brier:.4f}" if brier is not None else "Brier Score: N/A"
        )

st.divider()

# ── Payoff Quality ────────────────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:1rem; font-weight:700; color:var(--text); '
    'text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;">'
    'Payoff Quality</div>'
    '<div style="font-size:0.8rem; color:var(--muted); margin-bottom:0.8rem;">'
    '风险调整回报质量：PQ = barrier_return / (hist_vol × √T)。'
    'PQ&gt;1 = 强赢（≥1期σ方向正确）；0–1 = 弱赢；&lt;0 = 亏损。'
    '条件盈亏比 = |E[赢时回报] / E[输时回报]|，衡量赢大输小能力。</div>',
    unsafe_allow_html=True,
)

import pandas as pd

_pq_recs = _cached_cz_payoff()

if not _pq_recs:
    st.caption("暂无含 payoff_quality 的记录（需完成 barrier 验证后自动计算）。")
else:
    _wins_ret  = [r["barrier_return"] for r in _pq_recs if r["barrier_return"] > 0]
    _loss_ret  = [r["barrier_return"] for r in _pq_recs if r["barrier_return"] < 0]
    _all_pq    = [r["payoff_quality"] for r in _pq_recs]
    _pos_pq    = [r["payoff_quality"] for r in _pq_recs if r["payoff_quality"] > 0]
    _strong_pq = [r["payoff_quality"] for r in _pq_recs if r["payoff_quality"] >= 1.0]

    _e_win  = sum(_wins_ret) / len(_wins_ret) if _wins_ret else None
    _e_loss = sum(_loss_ret) / len(_loss_ret) if _loss_ret else None
    _cpr    = abs(_e_win / _e_loss) if (_e_win is not None and _e_loss is not None and _e_loss != 0) else None
    _avg_pq = sum(_all_pq) / len(_all_pq)
    _pq_pos_rate  = len(_pos_pq)    / len(_all_pq)
    _pq_strong_rate = len(_strong_pq) / len(_all_pq)

    _pq1, _pq2, _pq3, _pq4, _pq5 = st.columns(5)
    _pq1.metric("PQ 样本数",   len(_all_pq))
    _pq2.metric("平均 PQ",     f"{_avg_pq:.3f}",
                help="PQ>0 表示系统整体产生正期望风险调整回报")
    _pq3.metric("PQ>0 比例",   f"{_pq_pos_rate:.0%}",
                help="盈利交易占比（含弱赢）")
    _pq4.metric("PQ≥1 比例",   f"{_pq_strong_rate:.0%}",
                help="强赢比例：回报 ≥ 1期标准差的正确方向")
    _pq5.metric("条件盈亏比",
                f"{_cpr:.2f}" if _cpr is not None else "N/A",
                help="| E[赢时 barrier_return] / E[输时 barrier_return] |，理想值 > 1.5")

    # PQ distribution
    st.markdown(
        '<div style="font-size:0.8rem; color:var(--muted); '
        'margin-top:0.8rem; margin-bottom:0.3rem;">PQ 分布</div>',
        unsafe_allow_html=True,
    )
    import math
    _pq_buckets = {"<-1": 0, "-1–0": 0, "0–1": 0, "1–2": 0, ">2": 0}
    for pq in _all_pq:
        if pq < -1:
            _pq_buckets["<-1"] += 1
        elif pq < 0:
            _pq_buckets["-1–0"] += 1
        elif pq < 1:
            _pq_buckets["0–1"] += 1
        elif pq < 2:
            _pq_buckets["1–2"] += 1
        else:
            _pq_buckets[">2"] += 1

    df_pq_dist = pd.DataFrame(
        {"PQ 区间": list(_pq_buckets.keys()), "count": list(_pq_buckets.values())}
    ).set_index("PQ 区间")
    st.bar_chart(df_pq_dist, color=["#10B981"], height=140)

    # Conditional payoff by confidence bucket
    if any(r["confidence_score"] is not None for r in _pq_recs):
        st.markdown(
            '<div style="font-size:0.8rem; color:var(--muted); '
            'margin-top:0.8rem; margin-bottom:0.3rem;">按置信度分组的条件盈亏比</div>',
            unsafe_allow_html=True,
        )
        _cb: dict[str, list] = {}
        for r in _pq_recs:
            if r["confidence_score"] is None:
                continue
            _b = f"{(r['confidence_score'] // 20) * 20}–{(r['confidence_score'] // 20) * 20 + 19}%"
            _cb.setdefault(_b, []).append(r["barrier_return"])

        # Build a parallel pq-by-bucket map for avg PQ per bucket
        _cb_pq: dict[str, list] = {}
        for r in _pq_recs:
            if r["confidence_score"] is None:
                continue
            _b2 = f"{(r['confidence_score'] // 20) * 20}–{(r['confidence_score'] // 20) * 20 + 19}%"
            _cb_pq.setdefault(_b2, []).append(r["payoff_quality"])

        _cb_rows = []
        for bucket, rets in sorted(_cb.items()):
            _w = [x for x in rets if x > 0]
            _l = [x for x in rets if x < 0]
            _ew = sum(_w) / len(_w) if _w else None
            _el = sum(_l) / len(_l) if _l else None
            _cr = abs(_ew / _el) if (_ew is not None and _el is not None and _el != 0) else None
            _pq_vals = _cb_pq.get(bucket, [])
            _apq = sum(_pq_vals) / len(_pq_vals) if _pq_vals else None
            _cb_rows.append({
                "置信度区间": bucket,
                "n": len(rets),
                "E[赢]": f"{_ew:.3f}" if _ew is not None else "—",
                "E[输]": f"{_el:.3f}" if _el is not None else "—",
                "条件盈亏比": f"{_cr:.2f}" if _cr is not None else "—",
                "平均 PQ": f"{_apq:.3f}" if _apq is not None else "—",
            })

        st.dataframe(pd.DataFrame(_cb_rows), hide_index=True, use_container_width=True)
