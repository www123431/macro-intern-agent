"""Gate 2 — Cost-robust (Almgren-Chriss stress survival).

Reads the verdict's `cost_stress` and `cost_robust_verdict` from
metrics (produced by FORWARD dispatcher's cost-stress step). The
dispatcher already tests the strategy at the realistic round-trip
cost `tc_bp_per_rt` (typically 13bp for US equity); the `cost_stress`
dict additionally tests at a grid of stressed cost levels (e.g.
0bp / 30bp / 60bp / 80bp for monthly cross-sec, or 0bp / 8bp /
16bp / 24bp for higher-turnover strategies).

Gate 2's job is NOT to re-run cost stress — that's the dispatcher's
job. Gate 2 audits whether the recorded stress results actually
demonstrate cost-robustness, not just baseline survival. A strategy
that's GREEN at the realistic baseline but dies at 2-6x stress is
fragile, not robust — institutional capital should not deploy on it.

## Tier logic (v1 — empirical study on 63 verdicts with cost_stress)

  - **PASS**:      `cost_robust_verdict == 'GREEN'` AND highest stress
                   level still has verdict in (GREEN, MARGINAL).
                   Real cost-robustness — survives baseline + 2-6x
                   stress without collapsing.
                   (5/63 in current corpus.)

  - **SOFT_PASS**: `cost_robust_verdict == 'GREEN'` BUT highest
                   stress level verdict == 'RED'. Survives realistic
                   baseline cost but breaks under stress — fragile.
                   Human reviewer should examine the cost grid before
                   approving at Gate 9.
                   (4/63 in current corpus — 2 GREEN→MARGINAL counted
                   as PASS, plus 2 GREEN→RED counted as SOFT.)

  - **FAIL**:      `cost_robust_verdict in ('RED', 'FAIL')`. Does
                   not survive even realistic baseline cost — promote
                   would deploy into a strategy whose paper edge
                   evaporates under realistic execution.

  - **SKIPPED**:   No `cost_stress` block, OR `cost_robust_verdict`
                   missing. Pre-Phase-2 verdict that legitimately
                   lacks the stress test. Human reviewer must verify
                   cost-robustness manually.
                   (~55/305 GREEN events in current corpus lack the
                   block — pre-dispatcher-v0.1.0 verdicts.)

## Reference / anchors

  - Almgren-Chriss 2000 — sqrt-impact + half-spread cost model
    (institutional canonical, captures size-dependent slippage)
  - Frazzini-Israel-Moskowitz 2018 — measured implementation cost
    drag of 20-50bps for systematic strategies at AUM > $1B
  - engine/validation/cost_stress.py — the stress runner the
    dispatcher invokes; the breakeven_cost result there could be
    plumbed through to enrich this gate later (currently we only
    read the per-level grid verdicts)
"""
from __future__ import annotations

from typing import Any

from engine.operator_console.gates import GateResult, GateStatus


GATE_ID = "gate2_cost_robust"
GATE_TITLE = "Cost-robust (Almgren-Chriss)"


def _parse_bp(key: str) -> int | None:
    """Cost grid keys are like '0bp' / '30bp'. Returns int bp or None."""
    if not isinstance(key, str) or not key.endswith("bp"):
        return None
    try:
        return int(key[:-2])
    except ValueError:
        return None


def _grid_ordered(cost_stress: dict[str, Any]) -> list[tuple[int, dict]]:
    """Return [(bp, level_dict), ...] sorted by bp ascending. Skips
    keys we can't parse as '<int>bp'."""
    out: list[tuple[int, dict]] = []
    for k, v in cost_stress.items():
        bp = _parse_bp(k)
        if bp is None or not isinstance(v, dict):
            continue
        out.append((bp, v))
    out.sort(key=lambda x: x[0])
    return out


def check(verdict_event: dict[str, Any], config: dict[str, Any]) -> GateResult:
    metrics = verdict_event.get("metrics") or {}
    cost_stress = metrics.get("cost_stress")
    cost_robust_verdict = metrics.get("cost_robust_verdict")
    tc_bp_per_rt = metrics.get("tc_bp_per_rt")
    avg_turnover = metrics.get("avg_turnover")

    # ── Tier 4: pre-Phase-2 verdict, no stress data ─────────────
    if not isinstance(cost_stress, dict) or not cost_stress:
        return GateResult(
            gate_id = GATE_ID,
            title   = GATE_TITLE,
            status  = GateStatus.SKIPPED,
            summary = ("Verdict has no cost_stress block. Pre-dispatcher-"
                       "v0.1.0 verdict — human reviewer must verify "
                       "cost-robustness manually (re-run via "
                       "engine.validation.cost_stress)."),
            detail  = {"metric_keys": list(metrics.keys())[:20]},
        )

    if cost_robust_verdict is None:
        return GateResult(
            gate_id = GATE_ID,
            title   = GATE_TITLE,
            status  = GateStatus.SKIPPED,
            summary = ("Verdict has cost_stress block but no "
                       "cost_robust_verdict field — dispatcher version "
                       "mismatch. Reviewer must read cost_stress grid "
                       "directly."),
            detail  = {"cost_stress_keys": list(cost_stress.keys())},
        )

    grid = _grid_ordered(cost_stress)
    detail: dict[str, Any] = {
        "cost_robust_verdict":  cost_robust_verdict,
        "baseline_tc_bp_per_rt": tc_bp_per_rt,
        "avg_turnover":         avg_turnover,
        "stress_grid":          [
            {
                "bp":      bp,
                "verdict": lvl.get("verdict"),
                "sharpe":  lvl.get("sharpe"),
                "nw_t":    lvl.get("nw_t_stat"),
            }
            for bp, lvl in grid
        ],
    }

    # ── Tier 3: dispatcher said cost-robust verdict is not GREEN ──
    if str(cost_robust_verdict).upper() not in ("GREEN",):
        return GateResult(
            gate_id = GATE_ID,
            title   = GATE_TITLE,
            status  = GateStatus.FAIL,
            summary = (f"cost_robust_verdict = {cost_robust_verdict!r} "
                       f"at baseline tc={tc_bp_per_rt}bp. Strategy "
                       f"does not survive realistic execution cost; "
                       f"promote would deploy capital on an edge that "
                       f"evaporates under cost."),
            detail  = detail,
        )

    # ── From here cost_robust_verdict == 'GREEN'. Check stress tail.
    if not grid:
        # cost_robust_verdict GREEN but grid empty / unparseable keys —
        # treat as soft because we can't verify stress survival
        return GateResult(
            gate_id = GATE_ID,
            title   = GATE_TITLE,
            status  = GateStatus.SOFT_PASS,
            summary = ("cost_robust_verdict=GREEN but cost_stress grid "
                       "keys unparseable (expected '<int>bp' format). "
                       "Reviewer should re-run cost stress."),
            detail  = detail,
        )

    highest_bp, highest_level = grid[-1]
    highest_verdict = str(highest_level.get("verdict") or "").upper()

    # ── Tier 2: GREEN at baseline, breaks at highest stress ────────
    if highest_verdict == "RED":
        return GateResult(
            gate_id = GATE_ID,
            title   = GATE_TITLE,
            status  = GateStatus.SOFT_PASS,
            summary = (f"cost_robust_verdict=GREEN at baseline tc="
                       f"{tc_bp_per_rt}bp, but stress at {highest_bp}bp "
                       f"(~{highest_bp / (tc_bp_per_rt or 1):.1f}x baseline) "
                       f"flips to RED. Survives realistic cost but "
                       f"fragile under stress — human reviewer should "
                       f"verify the deployment AUM won't push effective "
                       f"cost into the breakdown zone."),
            detail  = detail,
        )

    # ── Tier 1: GREEN at baseline AND stress tail ≠ RED ────────────
    if highest_verdict not in ("GREEN", "MARGINAL"):
        # Unknown verdict label — degrade to SOFT_PASS rather than PASS
        return GateResult(
            gate_id = GATE_ID,
            title   = GATE_TITLE,
            status  = GateStatus.SOFT_PASS,
            summary = (f"cost_robust_verdict=GREEN but stress at "
                       f"{highest_bp}bp returned unrecognized verdict "
                       f"{highest_verdict!r}. Reviewer should examine "
                       f"the cost grid directly."),
            detail  = detail,
        )

    return GateResult(
        gate_id = GATE_ID,
        title   = GATE_TITLE,
        status  = GateStatus.PASS,
        summary = (f"cost_robust_verdict=GREEN at baseline tc="
                   f"{tc_bp_per_rt}bp; stress survives through "
                   f"{highest_bp}bp (~{highest_bp / (tc_bp_per_rt or 1):.1f}x "
                   f"baseline) with verdict {highest_verdict}. Cost-"
                   f"robust under realistic + stressed execution."),
        detail  = detail,
    )
