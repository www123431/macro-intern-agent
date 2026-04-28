"""
Macro Alpha Pro — Gemini Key Pool Manager
Real-time monitoring and management of the Gemini API key pool.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import datetime
import streamlit as st
import ui.theme as theme
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
# set_page_config handled by app.py via st.navigation()
# st.set_page_config(page_title="Key Pool Manager · Macro Alpha Pro", page_icon="🔑", layout="wide")
theme.init_theme()  # idempotent — ensures theme session state is set

from engine.key_pool import (
    KeyPoolManager, get_pool, reset_pool,
    STATS_FILE, QUOTA_FAILS_BEFORE_SWITCH, EMPTY_OUTPUT_LIMIT,
)

# ── Helpers ───────────────────────────────────────────────────────────────────
_is_dark = theme.is_dark()

def _section(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)

def _status_badge(status: str) -> str:
    cfg = {
        "active":    ("#4ade80", "#052e16", "#4ade80", "#bbf7d0") if _is_dark else ("#16a34a", "#f0fdf4", "#16a34a", "#dcfce7"),
        "exhausted": ("#f87171", "#450a0a", "#f87171", "#fee2e2") if _is_dark else ("#dc2626", "#fef2f2", "#dc2626", "#fee2e2"),
        "halted":    ("#fbbf24", "#451a03", "#fbbf24", "#fef9c3") if _is_dark else ("#d97706", "#fffbeb", "#d97706", "#fef3c7"),
    }
    color, bg, border, _ = cfg.get(status, cfg["active"])
    label = {"active": "● 运行中", "exhausted": "✕ 已耗尽", "halted": "⚠ 熔断"}.get(status, status)
    return (
        f'<span style="font-size:0.75rem; font-weight:600; padding:2px 10px; '
        f'border-radius:12px; background:{bg}; color:{color}; '
        f'border:1px solid {border};">{label}</span>'
    )

def _bar(val: int, color: str, max_val: int = 1500) -> str:
    pct = min(100, int(val / max_val * 100)) if max_val else 0
    bg  = "rgba(255,255,255,0.08)" if _is_dark else "rgba(0,0,0,0.07)"
    return (
        f'<div style="background:{bg}; border-radius:3px; height:5px; margin-top:3px;">'
        f'<div style="width:{pct}%; background:{color}; border-radius:3px; height:5px;"></div>'
        f'</div>'
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<h2 style="font-size:1.4rem; font-weight:700; margin-bottom:0.2rem;">'
    '🔑 Gemini Key Pool Manager</h2>'
    '<p style="font-size:0.85rem; color:var(--muted); margin-bottom:1.5rem;">'
    '管理 API Key 池 · 实时监控用量 · 查看熔断异常日志</p>',
    unsafe_allow_html=True,
)

# ── Load pool ─────────────────────────────────────────────────────────────────
try:
    pool = get_pool()
    summary = pool.pool_summary()
    all_stats = pool.get_all_stats()
    pool_ok = True
except Exception as e:
    st.error(f"Key 池初始化失败：{e}")
    pool_ok = False
    summary = {}
    all_stats = []

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Pool summary cards
# ═══════════════════════════════════════════════════════════════════════════════
if pool_ok:
    _section("Pool Overview · 今日汇总")

    _c1, _c2, _c3, _c4, _c5 = st.columns(5)
    card_bg = "rgba(255,255,255,0.04)" if _is_dark else "rgba(0,0,0,0.03)"
    brd     = "rgba(255,255,255,0.10)" if _is_dark else "rgba(0,0,0,0.08)"
    txt     = "#e2e8f0" if _is_dark else "#1e293b"
    muted   = "#94a3b8" if _is_dark else "#64748b"

    def _summary_card(col, label, value, sub="", color=None):
        color = color or txt
        col.markdown(
            f'<div style="padding:0.8rem 1rem; background:{card_bg}; '
            f'border:1px solid {brd}; border-radius:10px; text-align:center;">'
            f'<div style="font-size:1.6rem; font-weight:700; color:{color};">{value}</div>'
            f'<div style="font-size:0.75rem; color:{muted}; margin-top:2px;">{label}</div>'
            + (f'<div style="font-size:0.7rem; color:{muted};">{sub}</div>' if sub else "")
            + '</div>',
            unsafe_allow_html=True,
        )

    _summary_card(_c1, "Key 总数",   summary.get("total", 0))
    _summary_card(_c2, "运行中",     summary.get("active", 0),    color="#4ade80" if _is_dark else "#16a34a")
    _summary_card(_c3, "已耗尽",     summary.get("exhausted", 0), color="#f87171" if _is_dark else "#dc2626")
    _summary_card(_c4, "今日调用",   summary.get("today_calls", 0),  sub=f"跳过 {summary.get('today_skips',0)} 次")
    _summary_card(_c5, "今日错误",   summary.get("today_errors", 0), color="#fbbf24" if _is_dark else "#d97706")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Key health diagnostics ─────────────────────────────────────────────────
    _unused_today   = [s for s in all_stats if s["today_calls"] == 0]
    _cold_exhausted = [s for s in _unused_today
                       if s["status"] in ("exhausted", "halted") or s["today_errors"] > 0]
    _cold_idle      = [s for s in _unused_today
                       if s["status"] == "active" and s["today_errors"] == 0]

    if _cold_exhausted:
        _warn_color  = "#fbbf24" if _is_dark else "#d97706"
        _warn_bg     = "rgba(251,191,36,0.08)" if _is_dark else "#fffbeb"
        _warn_border = "rgba(251,191,36,0.35)" if _is_dark else "#fde68a"
        _rows = "　".join(
            f'<b>{s["label"]}</b>（{s["status"]}，今日错误 {s["today_errors"]} 次）'
            for s in _cold_exhausted
        )
        st.markdown(
            f'<div style="background:{_warn_bg}; border:1.5px solid {_warn_border}; '
            f'border-radius:6px; padding:0.6rem 1rem; margin-bottom:0.5rem;">'
            f'<span style="font-weight:700; color:{_warn_color};">⚠ 冷启动配额异常</span>'
            f'<span style="font-size:0.88rem; color:{txt}; margin-left:0.8rem;">'
            f'以下 Key 今日尚未成功调用即已出现错误或耗尽：{_rows}</span>'
            f'<br><span style="font-size:0.8rem; color:{muted};">→ 建议检查 Key 有效性，或手动切换至备用 Key</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if _cold_idle:
        _idle_color  = "#60a5fa" if _is_dark else "#2563eb"
        _idle_bg     = "rgba(96,165,250,0.06)" if _is_dark else "#eff6ff"
        _idle_border = "rgba(96,165,250,0.25)" if _is_dark else "#bfdbfe"
        _idle_rows = "、".join(s["label"] for s in _cold_idle)
        st.markdown(
            f'<div style="background:{_idle_bg}; border:1px solid {_idle_border}; '
            f'border-radius:6px; padding:0.5rem 1rem; margin-bottom:0.5rem;">'
            f'<span style="font-size:0.88rem; color:{_idle_color};">ℹ 今日未使用 Key：{_idle_rows}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Per-key status table (collapsible, scrollable)
# ═══════════════════════════════════════════════════════════════════════════════
if pool_ok and all_stats:
    green  = "#4ade80" if _is_dark else "#16a34a"
    red    = "#f87171" if _is_dark else "#dc2626"
    yellow = "#fbbf24" if _is_dark else "#d97706"

    _section("Key Status · 各 Key 明细")

    # Build all rows as a single HTML block inside a scrollable container.
    # Each row is ~76px; showing 6 rows = 456px max-height.
    _ROW_HEIGHT = 76
    _MAX_VISIBLE = 6
    _scroll_h = _ROW_HEIGHT * _MAX_VISIBLE

    rows_html = ""
    for s in all_stats:
        _row_bg  = ("rgba(99,102,241,0.08)" if _is_dark else "rgba(99,102,241,0.05)") \
                   if s["is_current"] else card_bg
        _row_brd = ("rgba(99,102,241,0.4)" if _is_dark else "rgba(99,102,241,0.3)") \
                   if s["is_current"] else brd

        _current_tag = (
            '<span style="font-size:0.68rem; color:#818cf8; margin-left:6px;">▶ 当前</span>'
        ) if s["is_current"] else ""

        _last = s.get("last_used") or "—"
        if _last != "—":
            _last = _last[11:19]

        _cf = s["consecutive_quota"]
        _cf_color = red if _cf >= QUOTA_FAILS_BEFORE_SWITCH - 1 else \
                    yellow if _cf > 0 else txt

        def _mini_bar(val, color, max_val):
            pct = min(100, int(val / max_val * 100)) if max_val else 0
            bg  = "rgba(255,255,255,0.08)" if _is_dark else "rgba(0,0,0,0.07)"
            return (
                f'<div style="background:{bg}; border-radius:3px; '
                f'height:4px; margin-top:3px; width:100%;">'
                f'<div style="width:{pct}%; background:{color}; '
                f'border-radius:3px; height:4px;"></div></div>'
            )

        rows_html += (
            f'<div style="display:grid; grid-template-columns:3fr 2fr 1.5fr 1.5fr 1.5fr 2fr; '
            f'gap:6px; margin-bottom:6px;">'

            # col A — label + masked key
            f'<div style="padding:0.5rem 0.7rem; background:{_row_bg}; '
            f'border:1px solid {_row_brd}; border-radius:8px;">'
            f'<div style="font-size:0.85rem; font-weight:600; color:{txt};">'
            f'{s["label"]}{_current_tag}</div>'
            f'<div style="font-size:0.7rem; color:{muted}; font-family:monospace;">'
            f'{s["key_masked"]}</div></div>'

            # col B — status badge
            f'<div style="padding:0.5rem 0.7rem; background:{_row_bg}; '
            f'border:1px solid {_row_brd}; border-radius:8px; '
            f'display:flex; align-items:center; justify-content:center;">'
            + _status_badge(s["status"]) +
            f'</div>'

            # col C — today calls
            f'<div style="padding:0.5rem 0.7rem; background:{_row_bg}; '
            f'border:1px solid {_row_brd}; border-radius:8px;">'
            f'<div style="font-size:0.75rem; color:{muted};">今日调用</div>'
            f'<div style="font-size:0.95rem; font-weight:700; color:{txt};">'
            f'{s["today_calls"]}</div>'
            + _mini_bar(s["today_calls"], green, 1500) +
            f'</div>'

            # col D — errors
            f'<div style="padding:0.5rem 0.7rem; background:{_row_bg}; '
            f'border:1px solid {_row_brd}; border-radius:8px;">'
            f'<div style="font-size:0.75rem; color:{muted};">错误次数</div>'
            f'<div style="font-size:0.95rem; font-weight:700; '
            f'color:{red if s["today_errors"] > 0 else txt};">'
            f'{s["today_errors"]}</div>'
            + _mini_bar(s["today_errors"], red, 10) +
            f'</div>'

            # col E — consecutive quota fails
            f'<div style="padding:0.5rem 0.7rem; background:{_row_bg}; '
            f'border:1px solid {_row_brd}; border-radius:8px;">'
            f'<div style="font-size:0.75rem; color:{muted};">连续失败</div>'
            f'<div style="font-size:0.95rem; font-weight:700; color:{_cf_color};">'
            f'{_cf}/{QUOTA_FAILS_BEFORE_SWITCH}</div>'
            + _mini_bar(_cf, _cf_color, QUOTA_FAILS_BEFORE_SWITCH) +
            f'</div>'

            # col F — last used
            f'<div style="padding:0.5rem 0.7rem; background:{_row_bg}; '
            f'border:1px solid {_row_brd}; border-radius:8px;">'
            f'<div style="font-size:0.75rem; color:{muted};">最近使用</div>'
            f'<div style="font-size:0.85rem; color:{txt}; font-family:monospace;">'
            f'{_last}</div></div>'

            f'</div>'
        )

    st.markdown(
        f'<div style="overflow-y:auto; max-height:{_scroll_h}px; '
        f'padding-right:4px;">{rows_html}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Manual key switch
# ═══════════════════════════════════════════════════════════════════════════════
if pool_ok and len(all_stats) > 1:
    _section("手动切换 Key · 冗余备份")
    _current_label = next((s["label"] for s in all_stats if s["is_current"]), None)
    _status_icons  = {"active": "●", "exhausted": "✕", "halted": "⚠"}
    _key_options   = [
        f'{s["label"]}  {_status_icons.get(s["status"], "?")} {s["status"]}'
        for s in all_stats
    ]
    _default_idx = next((i for i, s in enumerate(all_stats) if s["is_current"]), 0)

    _sw_col, _sw_btn = st.columns([4, 1], gap="small")
    with _sw_col:
        _selected_option = st.selectbox(
            "强制切换当前 Key",
            _key_options,
            index=_default_idx,
            key="manual_key_switch_select",
            help="选择目标 Key 后点击切换，下次 API 调用即生效。可切换至任意状态的 Key（含已耗尽）。",
        )
    with _sw_btn:
        st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
        if st.button("切换", key="manual_key_switch_btn", width='stretch'):
            _target_label = _selected_option.split("  ")[0]
            if _target_label == _current_label:
                _notice = st.empty()
                _notice.info(f"{_target_label} 已是当前 Key，无需切换")
                time.sleep(3)
                _notice.empty()
            else:
                try:
                    pool.force_switch_to(_target_label)
                    st.success(f"已切换：{_current_label}  →  {_target_label}（下次 API 调用即生效）")
                    st.rerun()
                except Exception as e:
                    st.error(f"切换失败：{e}")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Anomaly log
# ═══════════════════════════════════════════════════════════════════════════════
if pool_ok:
    _section("Anomaly Log · 异常事件")

    _all_anomalies = []
    for s in all_stats:
        for entry in s.get("anomaly_log", []):
            _all_anomalies.append({**entry, "label": s["label"]})

    _all_anomalies.sort(key=lambda x: x["ts"], reverse=True)

    if not _all_anomalies:
        st.markdown(
            f'<div style="color:{muted}; font-size:0.85rem; padding:0.5rem 0;">暂无异常记录</div>',
            unsafe_allow_html=True,
        )
    else:
        _event_colors = {
            "quota_error":       yellow,
            "key_exhausted":     red,
            "circuit_breaker":   red,
            "empty_output":      yellow,
            "daily_reset":       green,
        }
        _anomaly_rows = ""
        for entry in _all_anomalies[:40]:
            _ec = _event_colors.get(entry["event"], muted)
            _ts = entry["ts"][11:19]
            _anomaly_rows += (
                f'<div style="display:flex; align-items:baseline; gap:0.8rem; '
                f'padding:0.3rem 0; border-bottom:1px solid {brd}; font-size:0.82rem;">'
                f'<span style="color:{muted}; font-family:monospace; min-width:60px;">{_ts}</span>'
                f'<span style="color:{muted}; min-width:140px;">{entry["label"]}</span>'
                f'<span style="color:{_ec}; font-weight:600; min-width:120px;">'
                f'{entry["event"]}</span>'
                f'<span style="color:{txt};">{entry["msg"]}</span>'
                f'</div>'
            )
        st.markdown(
            f'<div style="overflow-y:auto; max-height:252px; padding-right:4px;">'
            f'{_anomaly_rows}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Interactive key management
# ═══════════════════════════════════════════════════════════════════════════════
_section("Key 配置管理")

SECRETS_PATH = Path(__file__).parent.parent / ".streamlit" / "secrets.toml"

# ── Read / write helpers ──────────────────────────────────────────────────────
def _read_pool_from_toml() -> dict[str, str]:
    """Return {label: key} from [GEMINI_POOL] section of secrets.toml."""
    if not SECRETS_PATH.exists():
        return {}
    import re
    text = SECRETS_PATH.read_text(encoding="utf-8")
    m = re.search(r"\[GEMINI_POOL\](.*?)(?=\n\[|\Z)", text, re.DOTALL)
    if not m:
        return {}
    pool: dict[str, str] = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        kv = re.match(r'^"?([^"=]+)"?\s*=\s*"([^"]+)"', line)
        if kv:
            pool[kv.group(1).strip()] = kv.group(2).strip()
    return pool

def _write_pool_to_toml(pool: dict[str, str]) -> None:
    """Rewrite [GEMINI_POOL] section in secrets.toml."""
    import re
    text = SECRETS_PATH.read_text(encoding="utf-8")
    # Remove existing [GEMINI_POOL] block (including commented-out template)
    text = re.sub(
        r"\n*#?\s*\[GEMINI_POOL\].*?(?=\n\[|\Z)",
        "", text, flags=re.DOTALL,
    ).rstrip()
    if pool:
        lines = ["\n\n[GEMINI_POOL]"]
        for label, key in pool.items():
            lines.append(f'"{label}" = "{key}"')
        text += "\n".join(lines)
    SECRETS_PATH.write_text(text.strip() + "\n", encoding="utf-8")

# ── Load current pool from file ───────────────────────────────────────────────
_file_pool = _read_pool_from_toml()
_pool_labels = list(_file_pool.keys())

# ── Layout: existing keys (left) + add form (right) ──────────────────────────
_left, _right = st.columns([3, 2], gap="large")

with _left:
    if not _file_pool:
        st.markdown(
            f'<div style="font-size:0.83rem; color:{muted}; padding:0.6rem 0;">'
            f'尚未配置任何 Key。使用右侧表单添加，或直接编辑 '
            f'<code>.streamlit/secrets.toml</code>。</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="font-size:0.82rem; font-weight:600; color:{txt}; '
            f'margin-bottom:0.6rem;">已配置的 Key（{len(_file_pool)} 个）</div>',
            unsafe_allow_html=True,
        )
        # Each row ~52px; 6 rows + account headers ≈ 340px
        with st.container(height=340):
                # Group by account (part before "/")
                _groups: dict[str, list[str]] = {}
                for lbl in _pool_labels:
                    acct = lbl.split("/")[0] if "/" in lbl else "默认"
                    _groups.setdefault(acct, []).append(lbl)

                for acct, labels in _groups.items():
                    st.markdown(
                        f'<div style="font-size:0.75rem; color:{muted}; '
                        f'text-transform:uppercase; letter-spacing:0.05em; '
                        f'margin:0.5rem 0 0.2rem 0;">{acct}</div>',
                        unsafe_allow_html=True,
                    )
                    for lbl in labels:
                        key_val = _file_pool[lbl]
                        masked  = key_val[:8] + "·····" + key_val[-4:]
                        proj    = lbl.split("/", 1)[1] if "/" in lbl else lbl

                        # Find runtime status for this label
                        _runtime = next(
                            (s for s in all_stats if s["label"] == lbl), None
                        ) if pool_ok else None
                        _badge_html = _status_badge(
                            _runtime["status"] if _runtime else "active"
                        )

                        _row, _del = st.columns([5, 1])
                        with _row:
                            st.markdown(
                                f'<div style="display:flex; align-items:center; gap:0.6rem; '
                                f'padding:0.45rem 0.7rem; background:{card_bg}; '
                                f'border:1px solid {brd}; border-radius:7px; margin-bottom:4px;">'
                                f'<span style="font-size:0.83rem; color:{txt}; '
                                f'font-weight:500; flex:1;">{proj}</span>'
                                f'<span style="font-size:0.72rem; color:{muted}; '
                                f'font-family:monospace;">{masked}</span>'
                                + _badge_html
                                + '</div>',
                                unsafe_allow_html=True,
                            )
                        with _del:
                            if st.button("✕", key=f"del_{lbl}",
                                         help=f"删除 {lbl}",
                                         width='stretch'):
                                _new_pool = {k: v for k, v in _file_pool.items() if k != lbl}
                                _write_pool_to_toml(_new_pool)
                                reset_pool()
                                st.success(f"已删除 {lbl}")
                                st.rerun()

with _right:
    st.markdown(
        f'<div style="font-size:0.82rem; font-weight:600; color:{txt}; '
        f'margin-bottom:0.6rem;">添加新 Key</div>',
        unsafe_allow_html=True,
    )
    with st.form("add_key_form", clear_on_submit=True):
        _new_acct  = st.text_input(
            "账号名", placeholder="Account1",
            help="同账号下的多个 Key 会自动分组显示",
        )
        _new_key   = st.text_input(
            "API Key", placeholder="AIza...",
            type="password",
        )
        _submitted = st.form_submit_button("添加 Key", type="primary", width='stretch')

        if _submitted:
            _errs = []
            _acct        = _new_acct.strip()
            _key_stripped = _new_key.strip()

            if not _acct:
                _errs.append("账号名不能为空")
            if not _key_stripped:
                _errs.append("API Key 不能为空")

            # Check key format: Gemini keys always start with "AIza"
            if _key_stripped and not _key_stripped.startswith("AIza"):
                _errs.append("API Key 格式有误，Gemini Key 应以 AIza 开头")

            # Check for duplicate account name (same account already exists)
            _existing_accts = {lbl.split("/")[0] for lbl in _file_pool}
            if _acct and _acct in _existing_accts:
                _next_idx = len([l for l in _file_pool if l.split("/")[0] == _acct]) + 1
                st.warning(
                    f'账号名 "{_acct}" 已存在，新 Key 将追加为 {_acct}/{_next_idx}'
                )

            # Check for duplicate key value
            _dup_label = next(
                (lbl for lbl, val in _file_pool.items() if val == _key_stripped),
                None,
            )
            if _dup_label:
                _errs.append(f"该 Key 已存在（标签：{_dup_label}），请勿重复添加")

            # Auto-number label
            _existing = [l for l in _file_pool if l.split("/")[0] == _acct]
            _label = f"{_acct}/{len(_existing) + 1}"
            if _label in _file_pool:
                _errs.append(f"标签 {_label} 已存在")

            if _errs:
                for e in _errs:
                    st.error(e)
            else:
                _updated = {**_file_pool, _label: _key_stripped}
                _write_pool_to_toml(_updated)
                reset_pool()
                st.success(f"已添加 {_label}，Key 池已重新加载")
                st.rerun()

# ── Behaviour summary ─────────────────────────────────────────────────────────
st.markdown(
    f'<div style="font-size:0.8rem; color:{muted}; line-height:1.9; '
    f'padding:0.7rem 1rem; background:{card_bg}; border:1px solid {brd}; '
    f'border-radius:8px; margin-top:0.8rem;">'
    f'<b style="color:{txt};">自动切换逻辑</b>　'
    f'连续 <b style="color:{txt};">{QUOTA_FAILS_BEFORE_SWITCH}</b> 次 quota 错误 → 切换下一个 Key　｜　'
    f'连续 <b style="color:{txt};">{EMPTY_OUTPUT_LIMIT}</b> 次空输出 → 熔断停止回测　｜　'
    f'额度重置跟随 Gemini 官方节奏（即 SGT 16:00）自动恢复 exhausted 状态'
    f'</div>',
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Manual controls
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
_section("手动控制")

_ctrl1, _ctrl2, _ctrl3 = st.columns(3)

with _ctrl1:
    if st.button("🔄 重新加载 Key 池", help="从 secrets.toml 重新读取所有 Key"):
        reset_pool()
        st.success("Key 池已重新初始化")
        st.rerun()

with _ctrl2:
    if st.button("🗑 清空统计数据", help="清除 key_pool_stats.json，重置所有计数器"):
        try:
            if STATS_FILE.exists():
                STATS_FILE.unlink()
            reset_pool()
            st.success("统计数据已清空")
            st.rerun()
        except Exception as e:
            st.error(f"清空失败：{e}")

with _ctrl3:
    if pool_ok:
        _exhausted_labels = [
            s["label"] for s in all_stats if s["status"] == "exhausted"
        ]
        if _exhausted_labels:
            if st.button(
                f"♻ 手动恢复 {len(_exhausted_labels)} 个已耗尽 Key",
                help="将标记为 exhausted 的 Key 强制恢复为 active（适用于额度已刷新的情况）"
            ):
                stats_data = json.loads(STATS_FILE.read_text()) if STATS_FILE.exists() else {}
                for lbl in _exhausted_labels:
                    if lbl in stats_data:
                        stats_data[lbl]["status"]            = "active"
                        stats_data[lbl]["consecutive_quota"] = 0
                        stats_data[lbl]["exhausted_at"]      = None
                STATS_FILE.write_text(json.dumps(stats_data, ensure_ascii=False, indent=2))
                reset_pool()
                st.success(f"已恢复 {len(_exhausted_labels)} 个 Key")
                st.rerun()
        else:
            st.button("♻ 无需恢复（无耗尽 Key）", disabled=True)

# ── Auto-refresh hint ─────────────────────────────────────────────────────────
st.markdown(
    f'<div style="font-size:0.72rem; color:{muted}; margin-top:1.5rem; text-align:right;">'
    f'页面数据反映最近一次写入磁盘的状态 · 刷新页面获取最新数据</div>',
    unsafe_allow_html=True,
)
