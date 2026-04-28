"""
Deployment-level trading configuration.
========================================
Reads the [trading] section from .streamlit/secrets.toml.
Engine code imports from here — never imports streamlit directly.

Priority: secrets.toml [trading] > hardcoded defaults
Changes require app restart (deployment-level config, not runtime preference).
"""
from __future__ import annotations

_DEFAULTS: dict = {
    # Existing
    "auto_execute_stops":     True,   # ATR/drawdown stops auto-execute (Layer 2)
    "auto_execute_entries":   False,  # entry triggers require human approval (Layer 3)
    "monthly_rebalance_auto": False,  # monthly rebalance requires human approval (Layer 3)
    "stop_max_weight_auto":   0.25,   # positions above this weight still need human confirmation
    # P4-2: Tactical patrol config
    "auto_execute_regime_compress": True,   # regime jump → auto-compress all longs (Layer 2)
    "auto_execute_high_conf_entry": False,  # high-confidence entry auto-execute (Layer 2; off during debug)
    "tactical_entry_max_weight":    0.05,   # max weight per tactical entry
    "tactical_entry_daily_limit":   2,      # max Layer-2 entries per day
    "regime_jump_threshold_ppt":    30.0,   # P(risk-off) single-day change threshold (ppt)
    "fast_signal_lookback":         3,      # TSMOM-Fast formation window (months)
    "fast_signal_skip":             1,      # TSMOM-Fast skip period (months)
    "entry_composite_score_min":    60,     # minimum composite score for high-conf entry
    "entry_momentum_zscore_min":    1.5,    # minimum 5-day momentum z-score for high-conf entry
}


def get_trading_config() -> dict:
    """
    Return merged trading config dict.
    Safe to call from engine code — catches all streamlit import errors.
    """
    try:
        import streamlit as st
        cfg = dict(st.secrets.get("trading", {}))
        merged = {**_DEFAULTS, **cfg}
        # Coerce types in case secrets.toml returns strings
        merged["auto_execute_stops"]     = _bool(merged["auto_execute_stops"])
        merged["auto_execute_entries"]   = _bool(merged["auto_execute_entries"])
        merged["monthly_rebalance_auto"] = _bool(merged["monthly_rebalance_auto"])
        merged["stop_max_weight_auto"]   = float(merged["stop_max_weight_auto"])
        # P4-2 tactical
        merged["auto_execute_regime_compress"] = _bool(merged["auto_execute_regime_compress"])
        merged["auto_execute_high_conf_entry"] = _bool(merged["auto_execute_high_conf_entry"])
        merged["tactical_entry_max_weight"]    = float(merged["tactical_entry_max_weight"])
        merged["tactical_entry_daily_limit"]   = int(merged["tactical_entry_daily_limit"])
        merged["regime_jump_threshold_ppt"]    = float(merged["regime_jump_threshold_ppt"])
        merged["fast_signal_lookback"]         = int(merged["fast_signal_lookback"])
        merged["fast_signal_skip"]             = int(merged["fast_signal_skip"])
        merged["entry_composite_score_min"]    = int(merged["entry_composite_score_min"])
        merged["entry_momentum_zscore_min"]    = float(merged["entry_momentum_zscore_min"])
        return merged
    except Exception:
        return dict(_DEFAULTS)


def _bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    return bool(val)
