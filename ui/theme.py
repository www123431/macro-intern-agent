"""
Shared theme / design-system CSS for Macro Alpha Pro.

Usage
-----
    from ui.theme import init_theme, render_toggle, build_css

    init_theme()                      # call once near top of each page
    st.markdown(build_css(), unsafe_allow_html=True)

    with st.sidebar:
        render_toggle()               # sun / moon button
"""

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Token sets
# ─────────────────────────────────────────────────────────────────────────────
_DARK = {
    "accent":           "#4C8EF7",
    "accent_lt":        "rgba(76,142,247,0.12)",
    "accent_bd":        "rgba(76,142,247,0.25)",
    "success":          "#23D333",
    "success_lt":       "rgba(35,211,51,0.10)",
    "success_bd":       "#23D333",
    "warn":             "#FFD33D",
    "warn_lt":          "rgba(255,211,61,0.10)",
    "warn_bd":          "#FFD33D",
    "danger":           "#F85149",
    "danger_lt":        "rgba(248,81,73,0.10)",
    "danger_bd":        "#F85149",
    "bg":               "#111827",
    "card":             "#1E293B",
    "border":           "rgba(255,255,255,0.12)",
    "text":             "#F0F6FC",
    "muted":            "#8B949E",
    "heading_color":    "#F0F6FC",
    "metric_val_color": "#F0F6FC",
    "shadow":           "rgba(0,0,0,0.30)",
    "badge_pass_bg":    "rgba(35,211,51,0.15)",
    "badge_pass_fg":    "#23D333",
    "badge_warn_bg":    "rgba(255,211,61,0.15)",
    "badge_warn_fg":    "#FFD33D",
    "badge_block_bg":   "rgba(248,81,73,0.15)",
    "badge_block_fg":   "#F85149",
    "badge_info_bg":    "rgba(76,142,247,0.15)",
    "badge_info_fg":    "#4C8EF7",
    "reflect_bd":       "rgba(76,142,247,0.25)",
    "body_bg_extra":    "background: #111827; color: #F0F6FC;",
}

_LIGHT = {
    "accent":           "#1A56DB",
    "accent_lt":        "#EBF5FF",
    "accent_bd":        "#BFDBFE",
    "success":          "#065F46",
    "success_lt":       "#ECFDF5",
    "success_bd":       "#6EE7B7",
    "warn":             "#92400E",
    "warn_lt":          "#FFFBEB",
    "warn_bd":          "#FCD34D",
    "danger":           "#991B1B",
    "danger_lt":        "#FEF2F2",
    "danger_bd":        "#FCA5A5",
    "bg":               "#EEF2F7",
    "card":             "#FFFFFF",
    "border":           "#CBD5E1",
    "text":             "#0F172A",
    "muted":            "#64748B",
    "heading_color":    "#0B1F3A",
    "metric_val_color": "#0B1F3A",
    "shadow":           "rgba(0,0,0,0.06)",
    "badge_pass_bg":    "#D1FAE5",
    "badge_pass_fg":    "#065F46",
    "badge_warn_bg":    "#FEF3C7",
    "badge_warn_fg":    "#92400E",
    "badge_block_bg":   "#FEE2E2",
    "badge_block_fg":   "#991B1B",
    "badge_info_bg":    "#DBEAFE",
    "badge_info_fg":    "#1E40AF",
    "reflect_bd":       "#BFDBFE",
    "body_bg_extra":    "",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────
def init_theme() -> None:
    """Initialize dark_mode in session_state (default: light)."""
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False


def is_dark() -> bool:
    return st.session_state.get("dark_mode", True)


def render_toggle(key: str = "__theme_toggle__") -> None:
    """Render the light/dark toggle button."""
    dark = is_dark()
    label = "🌙  夜间模式" if dark else "☀️  日间模式"
    if st.button(label, key=key, width='stretch'):
        st.session_state.dark_mode = not dark
        st.rerun()


def build_css(admin: bool = False) -> str:
    """Return the full <style> block for the current theme.

    admin=True uses slightly larger base font sizes (Admin Panel layout).
    """
    t = _DARK if is_dark() else _LIGHT
    container_pad = "1.4rem 2.2rem 2.5rem" if admin else "1.2rem 2rem 2rem"
    base_font    = "1.05rem !important" if admin else "1rem"
    metric_val   = "2rem !important"    if admin else "1.7rem"
    metric_lbl   = "0.95rem !important" if admin else "0.88rem"
    btn_size     = "1.05rem !important" if admin else "1rem"
    btn_h        = "2.8em" if admin else "2.6em"
    sect_lbl_sz  = "1.1rem !important"  if admin else "1rem"
    h1_sz        = "1.7rem !important"  if admin else "1.5rem !important"
    h2_sz        = "1.4rem !important"  if admin else "1.25rem !important"
    h3_sz        = "1.2rem !important"  if admin else "1.1rem !important"
    h1_w         = "900 !important"     if admin else "800 !important"
    h2_w         = "800 !important"     if admin else "700 !important"
    # In dark mode, --primary (#0B1F3A) is near-invisible on dark bg → use accent instead
    dark = is_dark()
    sect_lbl_color  = ("var(--accent)" if dark else "var(--primary)") if admin else "var(--muted)"
    sect_lbl_border = "2px solid var(--accent)" if admin else "1px solid var(--border)"
    _alert_color_override = (
        '[data-testid="stAlert"] * { color: #F0F6FC !important; }'
        '[data-testid="stTooltipContent"], [data-testid="stTooltipContent"] *,'
        '[role="tooltip"], [role="tooltip"] * { color: #1E293B !important; }'
    ) if dark else ""

    return f"""
<style>
/* ── Tokens ─────────────────────────────────────── */
:root {{
    --primary:    #0B1F3A;
    --accent:     {t['accent']};
    --accent-lt:  {t['accent_lt']};
    --success:    {t['success']};
    --success-lt: {t['success_lt']};
    --success-bd: {t['success_bd']};
    --warn:       {t['warn']};
    --warn-lt:    {t['warn_lt']};
    --warn-bd:    {t['warn_bd']};
    --danger:     {t['danger']};
    --danger-lt:  {t['danger_lt']};
    --danger-bd:  {t['danger_bd']};
    --bg:         {t['bg']};
    --card:       {t['card']};
    --border:     {t['border']};
    --text:       {t['text']};
    --muted:      {t['muted']};
    --mono:       'JetBrains Mono', 'Courier New', 'SF Mono', monospace;
}}

/* ── Layout ─────────────────────────────────────── */
.main .block-container {{ padding: {container_pad}; max-width: 1440px; }}
.main {{ background: var(--bg); color: var(--text); }}
[data-testid="stAppViewContainer"] {{ background: var(--bg); {t['body_bg_extra']} }}
body, p, span, label, div {{ color: var(--text); }}
.stMarkdown, .stText {{ color: var(--text); }}

/* ── Base typography ─────────────────────────────── */
body, p, div, span, li {{ font-size: {base_font}; }}

/* ── Sidebar (always dark terminal) ─────────────── */
section[data-testid="stSidebar"] > div:first-child {{
    background: var(--primary);
    padding-top: 1.2rem;
}}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] small {{ color: rgba(255,255,255,0.75) !important; }}
section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ color: rgba(255,255,255,0.75) !important; }}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{ color: #FFFFFF !important; }}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {{
    color: #FFFFFF !important; font-family: var(--mono);
}}
section[data-testid="stSidebar"] div[data-testid="metric-container"] {{
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-top: 2px solid rgba(255,255,255,0.4) !important;
    border-radius: 3px;
}}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] input[type="number"],
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] [data-baseweb="input"] input,
section[data-testid="stSidebar"] [data-baseweb="base-input"] input {{
    background: #1B2B45 !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: white !important;
    -webkit-text-fill-color: white !important;
    border-radius: 3px;
}}
section[data-testid="stSidebar"] [data-baseweb="input"],
section[data-testid="stSidebar"] [data-baseweb="base-input"] {{
    background: #1B2B45 !important;
}}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.12) !important; }}
section[data-testid="stSidebar"] .stAlert {{
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: rgba(255,255,255,0.8) !important;
}}
section[data-testid="stSidebar"] .stButton > button {{
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    color: rgba(255,255,255,0.85) !important;
    font-size: 0.95rem !important;
    height: 2.4em;
    letter-spacing: 0.04em;
    font-weight: 600;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(255,255,255,0.14) !important;
    border-color: rgba(255,255,255,0.3) !important;
}}

/* ── Tab Bar ─────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    background: transparent;
    border-bottom: 2px solid var(--border);
}}
.stTabs [data-baseweb="tab"] {{
    padding: 0.65rem 1.4rem;
    font-size: {btn_size};
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    background: transparent;
}}
.stTabs [aria-selected="true"] {{
    color: var(--accent);
    border-bottom-color: var(--accent);
    background: transparent;
}}

/* ── Metric Cards ────────────────────────────────── */
div[data-testid="metric-container"] {{
    background: var(--card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    padding: 0.9rem 1rem;
    border-radius: 3px;
    box-shadow: 0 1px 3px {t['shadow']};
}}
[data-testid="stMetricValue"] {{
    font-family: var(--mono);
    font-size: {metric_val};
    color: {t['metric_val_color']};
    font-weight: 700;
    letter-spacing: -0.01em;
}}
[data-testid="stMetricLabel"] {{
    font-size: {metric_lbl};
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: var(--muted);
    font-weight: 600;
}}
[data-testid="stMetricDelta"] {{ font-size: 1rem !important; }}

/* ── Buttons ─────────────────────────────────────── */
.stButton > button {{
    border-radius: 3px;
    font-size: {btn_size};
    font-weight: 600;
    letter-spacing: 0.04em;
    height: {btn_h};
    transition: all 0.15s ease;
}}
.stButton > button[kind="primary"] {{
    background: var(--accent);
    color: white !important;
    border: none;
}}
.stButton > button[kind="primary"] * {{
    color: white !important;
}}
.stButton > button[kind="primary"]:hover {{ background: #1D4ED8; }}
.stButton > button[kind="primary"]:disabled,
.stButton > button[kind="primary"][disabled] {{
    background: var(--accent) !important;
    color: white !important;
    opacity: 0.55;
    cursor: not-allowed;
}}
.stButton > button[kind="primary"]:disabled * {{
    color: white !important;
}}
.stButton > button:not([kind="primary"]) {{
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--text);
}}

/* ── Content Cards ───────────────────────────────── */
.decision-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    padding: 1.2rem 1.4rem;
    border-radius: 3px;
    font-size: 0.95rem;
    line-height: 1.75;
    color: var(--text);
}}
.card-pass {{
    background: var(--success-lt);
    border: 1px solid var(--success-bd);
    border-left: 4px solid var(--success);
    padding: 1.2rem 1.4rem;
    border-radius: 3px;
}}
.card-block {{
    background: var(--danger-lt);
    border: 1px solid var(--danger-bd);
    border-left: 4px solid var(--danger);
    padding: 1.2rem 1.4rem;
    border-radius: 3px;
}}
.card-reflect {{
    background: var(--accent-lt);
    border: 1px solid {t['reflect_bd']};
    border-left: 4px solid var(--accent);
    padding: 1.2rem 1.4rem;
    border-radius: 3px;
}}

/* ── Section Label ───────────────────────────────── */
.section-label {{
    font-size: {sect_lbl_sz};
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {sect_lbl_color};
    border-bottom: {sect_lbl_border};
    padding-bottom: 0.5rem;
    margin: 1.4rem 0 1rem;
}}

/* ── Status Badge ────────────────────────────────── */
.badge {{
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 2px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}}
.badge-pass  {{ background: {t['badge_pass_bg']};  color: {t['badge_pass_fg']}; }}
.badge-warn  {{ background: {t['badge_warn_bg']};  color: {t['badge_warn_fg']}; }}
.badge-block {{ background: {t['badge_block_bg']}; color: {t['badge_block_fg']}; }}
.badge-info  {{ background: {t['badge_info_bg']};  color: {t['badge_info_fg']}; }}

/* ── Champion Card ───────────────────────────────── */
.champion-card {{
    background: linear-gradient(135deg, #0B1F3A 0%, #163566 100%);
    color: white !important;
    padding: 1.4rem 1.8rem;
    border-radius: 4px;
    margin-bottom: 1rem;
}}
.champion-card div, .champion-card span, .champion-card p {{
    color: white !important;
}}
.champion-card .label {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    opacity: 0.55;
    color: white !important;
}}
.champion-card .value {{
    font-family: var(--mono);
    font-size: 1.2rem;
    font-weight: 700;
    margin-top: 0.15rem;
    color: white !important;
}}
.champion-card .name  {{ font-size: 1.5rem; font-weight: 800; letter-spacing: 0.01em; color: white !important; }}
.champion-card .ticker {{ font-family: var(--mono); font-size: 0.85rem; opacity: 0.55; margin-top: 0.15rem; color: white !important; }}

/* ── Expander ────────────────────────────────────── */
.streamlit-expanderHeader {{
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}}

/* ── Dataframe ───────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid var(--border);
    border-radius: 3px;
    overflow: hidden;
    font-size: 1.02rem !important;
}}

/* ── Input / select labels ───────────────────────── */
label[data-testid="stWidgetLabel"] p {{
    font-size: {base_font};
    font-weight: 600 !important;
    color: var(--text) !important;
}}
.stSlider label {{ font-size: {base_font}; font-weight: 600 !important; }}
.stSlider [data-testid="stMarkdownContainer"] p {{ font-size: 1rem !important; }}
.stRadio label {{ font-size: {base_font}; }}
.stRadio [data-testid="stMarkdownContainer"] p {{ font-size: {base_font}; }}

/* ── Alerts ──────────────────────────────────────── */
[data-testid="stAlert"] p {{ font-size: {base_font}; line-height: 1.7 !important; }}
{_alert_color_override}

/* ── Headings ────────────────────────────────────── */
h1 {{ font-size: {h1_sz};  font-weight: {h1_w}; color: {t['heading_color']} !important; }}
h2 {{ font-size: {h2_sz};  font-weight: {h2_w}; color: {t['heading_color']} !important; }}
h3 {{ font-size: {h3_sz};  font-weight: 600 !important; color: var(--text) !important; }}
</style>
"""
