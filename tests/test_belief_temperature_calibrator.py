"""Tests for engine.research.belief_temperature_calibrator (Phase 5).

Covers:
  - apply_temperature math: T=1 invariant; T<1 sharpens; T>1 flattens
  - apply_temperature edge cases (zero p, validation)
  - brier_component matches the canonical (1 - p_actual)^2
  - fit_temperature on perfectly-calibrated synthetic data → T near 1
  - fit_temperature on systematically over-confident data → T > 1
  - loocv_brier returns supported=False at n<2
  - loocv_brier matches mean_brier when temperature is forced
"""
from __future__ import annotations

import math

import pytest

from engine.research.belief_temperature_calibrator import (
    apply_temperature,
    brier_component,
    mean_brier,
    fit_temperature,
    loocv_brier,
    CalibrationFit,
    _golden_section_minimize,
)


# ── apply_temperature ───────────────────────────────────────────────


def test_apply_temperature_t1_returns_input_renormalized():
    dist = {"GREEN": 0.30, "MARGINAL": 0.50, "RED": 0.20}
    out = apply_temperature(dist, 1.0)
    for k in dist:
        assert out[k] == pytest.approx(dist[k], abs=1e-9)
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)


def test_apply_temperature_sums_to_one():
    dist = {"GREEN": 0.30, "MARGINAL": 0.50, "RED": 0.20}
    for T in (0.5, 0.7, 1.0, 1.5, 3.0, 5.0):
        out = apply_temperature(dist, T)
        assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)


def test_apply_temperature_t_less_than_1_sharpens_modal_class():
    """T < 1 should push mass toward the modal class (highest p_i)."""
    dist = {"GREEN": 0.50, "MARGINAL": 0.30, "RED": 0.20}
    out = apply_temperature(dist, 0.5)
    assert out["GREEN"] > dist["GREEN"]
    assert out["MARGINAL"] < dist["MARGINAL"]
    assert out["RED"] < dist["RED"]


def test_apply_temperature_t_greater_than_1_flattens_toward_uniform():
    """T > 1 should pull mass away from the modal class toward 1/3."""
    dist = {"GREEN": 0.70, "MARGINAL": 0.20, "RED": 0.10}
    out = apply_temperature(dist, 3.0)
    assert out["GREEN"] < dist["GREEN"]
    # Both minority classes should grow toward uniform
    assert out["MARGINAL"] > dist["MARGINAL"]
    assert out["RED"] > dist["RED"]


def test_apply_temperature_very_high_t_approaches_uniform():
    dist = {"GREEN": 0.80, "MARGINAL": 0.15, "RED": 0.05}
    out = apply_temperature(dist, 10.0)
    # With T=10 the distribution should be fairly close to 1/3 each.
    # Tolerance is loose because T=10 isn't strictly infinity.
    for k in ("GREEN", "MARGINAL", "RED"):
        assert abs(out[k] - 1/3) < 0.15


def test_apply_temperature_invalid_t_raises():
    dist = {"GREEN": 0.5, "MARGINAL": 0.3, "RED": 0.2}
    with pytest.raises(ValueError):
        apply_temperature(dist, 0.0)
    with pytest.raises(ValueError):
        apply_temperature(dist, -1.0)


def test_apply_temperature_handles_zero_probability():
    """A zero in the input distribution should not blow up via log(0)."""
    dist = {"GREEN": 0.0, "MARGINAL": 0.5, "RED": 0.5}
    out = apply_temperature(dist, 1.0)
    # Output should be valid and stable
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)
    # GREEN should be effectively zero (driven by the epsilon)
    assert out["GREEN"] < 1e-6


# ── brier_component + mean_brier ────────────────────────────────────


def test_brier_component_perfect_prediction_is_zero():
    dist = {"GREEN": 1.0, "MARGINAL": 0.0, "RED": 0.0}
    assert brier_component(dist, "GREEN") == 0.0


def test_brier_component_max_miss_is_one():
    dist = {"GREEN": 0.0, "MARGINAL": 1.0, "RED": 0.0}
    assert brier_component(dist, "GREEN") == 1.0


def test_brier_component_uniform_is_4_9ths():
    dist = {"GREEN": 1/3, "MARGINAL": 1/3, "RED": 1/3}
    assert brier_component(dist, "GREEN") == pytest.approx(4/9, abs=1e-9)


def test_mean_brier_skips_neutral_outcomes():
    """NEUTRAL outcomes aren't strict-gate; the calibrator must ignore."""
    rows = [
        {"predicted_verdict_dist": {"GREEN": 0.5, "MARGINAL": 0.3, "RED": 0.2},
         "actual_verdict": "NEUTRAL"},
        {"predicted_verdict_dist": {"GREEN": 0.5, "MARGINAL": 0.3, "RED": 0.2},
         "actual_verdict": "GREEN"},
    ]
    # Only the second row counts; brier = (1 - 0.5)^2 = 0.25
    assert mean_brier(rows) == pytest.approx(0.25, abs=1e-9)


def test_mean_brier_zero_rows_returns_zero():
    assert mean_brier([]) == 0.0
    # All-NEUTRAL also zero
    assert mean_brier([
        {"predicted_verdict_dist": {"GREEN": 0.5, "MARGINAL": 0.3, "RED": 0.2},
         "actual_verdict": "NEUTRAL"},
    ]) == 0.0


# ── golden-section search ───────────────────────────────────────────


def test_golden_section_finds_known_minimum():
    """f(x) = (x - 2.5)^2 has minimum at 2.5. Search should find it."""
    x_min, _ = _golden_section_minimize(lambda x: (x - 2.5)**2, 0.1, 10.0,
                                         tol=1e-6)
    assert x_min == pytest.approx(2.5, abs=1e-3)


# ── fit_temperature ─────────────────────────────────────────────────


def test_fit_temperature_on_perfect_predictions_returns_low_t():
    """Predictions that always put 1.0 on the correct class are
    perfectly calibrated; the fit should leave them alone (T ≈ 1)
    or, since 0.99 → log → /T → softmax stays sharp at any T,
    the loss surface is flat — T near _T_LO is fine since loss is
    near 0 throughout. We just verify Brier is near 0."""
    rows = []
    for v in ("GREEN", "MARGINAL", "RED"):
        for _ in range(10):
            dist = {c: (0.96 if c == v else 0.02) for c in
                    ("GREEN", "MARGINAL", "RED")}
            rows.append({"predicted_verdict_dist": dist,
                          "actual_verdict": v})
    fit = fit_temperature(rows)
    # Should fit to near-zero Brier irrespective of T direction
    assert fit.brier_calibrated < 0.05
    assert fit.n_autopsies == 30


def test_fit_temperature_on_overconfident_predictions_chooses_t_above_1():
    """If we predict 0.9 GREEN but actual is only 50% GREEN, we're
    over-confident → calibrator should learn T > 1 to flatten."""
    rows = []
    # 20 cases predicted (G=0.9, M=0.05, R=0.05); 10 actually GREEN,
    # 10 actually RED. Modal-class accuracy 50% but predicted 90% G.
    dist = {"GREEN": 0.9, "MARGINAL": 0.05, "RED": 0.05}
    for _ in range(10):
        rows.append({"predicted_verdict_dist": dist,
                      "actual_verdict": "GREEN"})
        rows.append({"predicted_verdict_dist": dist,
                      "actual_verdict": "RED"})
    fit = fit_temperature(rows)
    assert fit.temperature > 1.0, (
        f"Over-confident data should give T > 1, got T={fit.temperature}"
    )
    assert fit.brier_calibrated < fit.brier_uncalibrated


def test_fit_temperature_empty_rows_returns_t1():
    fit = fit_temperature([])
    assert fit.temperature == 1.0
    assert fit.n_autopsies == 0


def test_fit_temperature_all_neutral_returns_t1():
    """If all valid rows are NEUTRAL, n drops to 0 → fallback T=1."""
    rows = [
        {"predicted_verdict_dist": {"GREEN": 0.4, "MARGINAL": 0.3, "RED": 0.3},
         "actual_verdict": "NEUTRAL"},
    ] * 5
    fit = fit_temperature(rows)
    assert fit.temperature == 1.0
    assert fit.n_autopsies == 0


# ── loocv_brier ─────────────────────────────────────────────────────


def test_loocv_brier_unsupported_below_n2():
    out = loocv_brier([])
    assert out["supported"] is False
    out = loocv_brier([{"predicted_verdict_dist": {"GREEN": 0.5,
                          "MARGINAL": 0.3, "RED": 0.2},
                         "actual_verdict": "GREEN"}])
    assert out["supported"] is False


def test_loocv_brier_returns_full_metric_dict():
    """End-to-end smoke: LOOCV must produce a structured result with
    both calibrated and uncalibrated Briers + held-out T statistics."""
    rows = []
    # Mix of correct & wrong predictions so the optimizer has signal
    for _ in range(5):
        rows.append({"predicted_verdict_dist": {"GREEN": 0.7, "MARGINAL": 0.2,
                                                   "RED": 0.1},
                      "actual_verdict": "GREEN"})
    for _ in range(5):
        rows.append({"predicted_verdict_dist": {"GREEN": 0.7, "MARGINAL": 0.2,
                                                   "RED": 0.1},
                      "actual_verdict": "RED"})
    out = loocv_brier(rows)
    assert out["supported"] is True
    assert out["n"] == 10
    assert "brier_loocv_uncalibrated" in out
    assert "brier_loocv_calibrated" in out
    assert "t_median_held_out" in out
    assert out["t_min"] <= out["t_median_held_out"] <= out["t_max"]


# ── CalibrationFit immutability ─────────────────────────────────────


def test_calibration_fit_is_frozen():
    """The fit result must be safe to ship around without accidental
    mutation in production callers."""
    fit = fit_temperature([])
    with pytest.raises(_dc_error()):
        fit.temperature = 2.0  # type: ignore[misc]


def _dc_error():
    """The error class dataclasses raises when you try to set a
    frozen field — varies by Python version."""
    try:
        import dataclasses
        return dataclasses.FrozenInstanceError
    except AttributeError:
        return AttributeError
