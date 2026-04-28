"""
Macro Alpha Pro — Macro Intelligence
Global macro news brief, sentiment analysis, and AI-driven market commentary.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import ui.theme as theme
import ui.tabs as tabs

theme.init_theme()

vix = st.session_state.get("_vix_input", 20.0)
tabs.render_tab1(vix)
