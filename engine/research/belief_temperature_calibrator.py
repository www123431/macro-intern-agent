"""engine.research.belief_temperature_calibrator — Belief Layer Phase 5 (offline).

Post-hoc temperature-scaling calibration for predict_verdict output.
Closes the gap between W6-rigor T6 (Hosmer-Lemeshow GoF REJECTED at
p=0.0469) and a well-calibrated predictor: the predicted distributions
have correct ordering but wrong absolute confidence levels.

This module ships the OFFLINE calibrator + LOOCV measurement only —
deliberately separate from belief.predict_verdict so we can measure
the calibrator's effect with rigor BEFORE wiring it into production.
Wire-in is a later phase (call it Phase 5.1) gated on demonstrated
Brier improvement under LOOCV.

Doctrine
========
  - Air-gap preserved: reads autopsies (verdict OUTCOMES), produces a
    scalar T parameter. The hot prediction path doesn't see autopsies
    directly; T is the only signal that crosses the air-gap.

  - 1-parameter (single scalar T) is deliberate: at n≈100 autopsies,
    multi-parameter post-hoc calibration (e.g. per-class isotonic) over-
    fits. Guo et al. 2017 "On Calibration of Modern Neural Networks"
    shows temperature scaling matches per-class methods on calibration
    error at fraction of the complexity, especially for small N.

Math
====

Given a predicted distribution dist = {GREEN: p_g, MARGINAL: p_m,
RED: p_r} (sums to 1), temperature scaling with T > 0 produces:

    calibrated_p_i = exp(log(p_i) / T) / sum_j exp(log(p_j) / T)

T = 1.0 leaves the distribution unchanged.
T > 1.0 flattens (less confident, mass moves toward uniform).
T < 1.0 sharpens (more confident, modal class gets more mass).

Brier loss being minimized:
    loss(T) = (1/N) * sum_a (1 - p_actual_a(T))^2

We fit T via 1-D gold-section / scalar minimization over [0.1, 10.0]
— a wide interval that comfortably covers under- and over-confident
regimes. No gradient, no autograd needed (1 parameter, smooth, convex).

Caveats
=======
  - Calibration improves PROBABILITY ESTIMATES, not modal-class accuracy.
    A well-calibrated predictor with the SAME modal-class rate as an
    overconfident one is still strictly preferable (Brier strict decomp:
    calibration + refinement; we improve calibration without touching
    refinement).
  - p_i must be > 0 for log() to work. Predictions where any p_i = 0
    have log(0) = -inf; we add a tiny epsilon to guarantee finite logs.
    The downstream consumer must accept this nuance.
"""
from __future__ import annotations

import dataclasses as _dc
import json
import logging
import math
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AUTOPSIES_PATH = _REPO_ROOT / "data" / "research" / "autopsies.jsonl"

# Verdict classes the calibrator operates on. NEUTRAL is intentionally
# excluded — autopsies with actual_verdict=NEUTRAL aren't strict-gate
# outcomes and shouldn't enter the calibration loss.
_CLASSES: tuple[str, ...] = ("GREEN", "MARGINAL", "RED")

# Numerical guard so log(0) doesn't appear; tiny enough not to bias
# the optimizer at realistic distributions.
_LOG_EPSILON = 1e-9

# Search bounds for T. Wide enough to handle catastrophic miscalibration
# in either direction (very over-confident → T → 10; very under-
# confident → T → 0.1).
_T_LO, _T_HI = 0.1, 10.0

# Gold-section search tolerance — converge within 0.001 in T-space
# is more than enough; Brier changes by < 1e-5 in that window.
_T_TOL = 1e-3


@_dc.dataclass(frozen=True)
class CalibrationFit:
    """The output of fitting temperature scaling on a training set.

    Immutable + safe to serialize for production hand-off."""
    temperature:           float
    n_autopsies:           int
    brier_uncalibrated:    float
    brier_calibrated:      float
    relative_improvement:  float       # (uncal - cal) / uncal
    iterations:            int
    classes:               tuple[str, ...] = _CLASSES


def apply_temperature(
    dist: dict[str, float],
    temperature: float,
    *,
    classes: tuple[str, ...] = _CLASSES,
) -> dict[str, float]:
    """Apply temperature T to a verdict distribution.

    T must be > 0. Returns a new dict; input is not mutated.

    Edge cases:
      - T = 1.0 → returns a (re-normalized) copy of the input
      - p_i = 0 for some i → uses epsilon to keep log finite, so
        the corresponding output mass is tiny but nonzero
    """
    if temperature <= 0:
        raise ValueError(
            f"temperature must be > 0, got {temperature}"
        )
    log_probs = [
        math.log(max(float(dist.get(c, 0.0)), _LOG_EPSILON))
        for c in classes
    ]
    scaled = [lp / temperature for lp in log_probs]
    # Numerically-stable softmax
    m = max(scaled)
    exp_s = [math.exp(s - m) for s in scaled]
    denom = sum(exp_s)
    if denom <= 0:
        # Degenerate — return uniform
        n = len(classes)
        return {c: 1.0 / n for c in classes}
    return {c: e / denom for c, e in zip(classes, exp_s)}


def brier_component(dist: dict[str, float], actual_verdict: str) -> float:
    """Standard (1 - p_actual)^2 component. Mirrors
    engine.research.belief_autopsy._brier_component for consistency."""
    p_actual = float(dist.get(actual_verdict, 0.0))
    return (1.0 - p_actual) ** 2


def mean_brier(
    rows: list[dict],
    *,
    temperature: float = 1.0,
) -> float:
    """Mean Brier component across rows after applying temperature T.

    `rows` items must have keys 'predicted_verdict_dist' and
    'actual_verdict'. Rows whose actual is not in _CLASSES are skipped
    (NEUTRAL outcomes are not strict-gate). Returns 0.0 when no valid
    rows."""
    total = 0.0
    n = 0
    for r in rows:
        dist = r.get("predicted_verdict_dist") or {}
        actual = r.get("actual_verdict")
        if actual not in _CLASSES or not dist:
            continue
        calibrated = (apply_temperature(dist, temperature)
                       if temperature != 1.0 else dist)
        total += brier_component(calibrated, actual)
        n += 1
    return total / n if n else 0.0


def _golden_section_minimize(
    f, lo: float, hi: float, tol: float = _T_TOL, max_iter: int = 100,
) -> tuple[float, int]:
    """1-D minimization via golden-section search. Pure: no scipy dep.

    Returns (x_min, n_iter). The function f is assumed unimodal on
    [lo, hi] (Brier-vs-T is convex for temperature scaling — standard
    result, see Guo et al. 2017 §3).
    """
    phi = (math.sqrt(5.0) - 1.0) / 2.0   # ≈ 0.618
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = f(c), f(d)
    n = 0
    while abs(b - a) > tol and n < max_iter:
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = f(d)
        n += 1
    return (a + b) / 2.0, n


def fit_temperature(rows: list[dict]) -> CalibrationFit:
    """Fit T on the supplied training rows.

    Rows whose actual_verdict is not in _CLASSES are dropped silently.
    Returns CalibrationFit with diagnostic fields.
    """
    valid = [
        r for r in rows
        if r.get("actual_verdict") in _CLASSES
        and r.get("predicted_verdict_dist")
    ]
    n = len(valid)
    if n == 0:
        return CalibrationFit(
            temperature=1.0, n_autopsies=0,
            brier_uncalibrated=0.0, brier_calibrated=0.0,
            relative_improvement=0.0, iterations=0,
        )

    uncal_brier = mean_brier(valid, temperature=1.0)
    f = lambda T: mean_brier(valid, temperature=T)
    t_star, iters = _golden_section_minimize(f, _T_LO, _T_HI)
    cal_brier = mean_brier(valid, temperature=t_star)

    rel_imp = (
        (uncal_brier - cal_brier) / uncal_brier
        if uncal_brier > 0 else 0.0
    )
    return CalibrationFit(
        temperature=round(t_star, 4),
        n_autopsies=n,
        brier_uncalibrated=round(uncal_brier, 6),
        brier_calibrated=round(cal_brier, 6),
        relative_improvement=round(rel_imp, 6),
        iterations=iters,
    )


def _load_autopsies(path: Optional[Path] = None) -> list[dict]:
    """Load non-superseded autopsies with both predicted_verdict_dist
    AND actual_verdict populated."""
    p = path or _AUTOPSIES_PATH
    if not p.is_file():
        return []
    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("superseded_by"):
                continue
            if not r.get("predicted_verdict_dist"):
                continue
            if not r.get("actual_verdict"):
                continue
            rows.append(r)
    return rows


def fit_from_autopsies(
    autopsies_path: Optional[Path] = None,
) -> CalibrationFit:
    """Convenience: load autopsies from disk + fit T. Used by the
    LOOCV report + any future offline tooling."""
    rows = _load_autopsies(autopsies_path)
    return fit_temperature(rows)


def loocv_brier(rows: list[dict]) -> dict:
    """Leave-one-out cross-validated mean Brier under temperature
    calibration. For each row R: fit T on all rows except R, apply
    that T to R, accumulate brier_component(R).

    Compared against the SAME LOOCV procedure with T forced to 1.0
    (no calibration) so the delta is the calibrator's honest gain
    after held-out validation.

    Returns a dict with both LOOCV Briers + T statistics.
    """
    valid = [
        r for r in rows
        if r.get("actual_verdict") in _CLASSES
        and r.get("predicted_verdict_dist")
    ]
    n = len(valid)
    if n < 2:
        return {
            "n":               n,
            "supported":       False,
            "reason":          "n<2 — LOOCV undefined",
        }

    sum_cal = 0.0
    sum_uncal = 0.0
    t_values: list[float] = []
    for i, held_out in enumerate(valid):
        train = valid[:i] + valid[i+1:]
        fit = fit_temperature(train)
        t_values.append(fit.temperature)
        dist = held_out["predicted_verdict_dist"]
        actual = held_out["actual_verdict"]
        cal_dist = apply_temperature(dist, fit.temperature)
        sum_cal   += brier_component(cal_dist, actual)
        sum_uncal += brier_component(dist, actual)

    cal_mean = sum_cal / n
    uncal_mean = sum_uncal / n
    rel_imp = (uncal_mean - cal_mean) / uncal_mean if uncal_mean > 0 else 0.0

    t_sorted = sorted(t_values)
    t_median = t_sorted[n // 2]
    t_mean = sum(t_values) / n
    return {
        "n":                       n,
        "supported":               True,
        "brier_loocv_uncalibrated": round(uncal_mean, 6),
        "brier_loocv_calibrated":   round(cal_mean, 6),
        "relative_improvement":    round(rel_imp, 6),
        "t_median_held_out":       round(t_median, 4),
        "t_mean_held_out":         round(t_mean, 4),
        "t_min":                   round(min(t_values), 4),
        "t_max":                   round(max(t_values), 4),
    }
