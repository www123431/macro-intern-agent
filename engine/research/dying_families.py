"""engine.research.dying_families — burndown_ranker input signal.

Detects mechanism families where the empirical autopsy history says
"nothing survives here" — high probability RED verdicts with no green
signal, even at small sample sizes. Used by burndown_ranker to
DE-prioritize new hypotheses in these families.

Why this exists
===============
v19's belief_coverage.md (2026-06-28) identified a real gap: several
mechanism families (VALUE / INVESTMENT / SIZE / LOW_VOL /
VOL_RISK_PREMIUM) are 100% RED across all autopsies but with N=1-2
each — too small for belief-4's Bayesian shrinkage to activate
(MIN_AUTOPSIES_FOR_OVERRIDE=3 since v19).

That means belief-1 keeps handing these families the hand-calibrated
FAMILY_PRIOR_OVERRIDES value of GREEN ≈ 0.15-0.20, but the empirical
GREEN rate is 0.00. The predictor knows too little; the SELECTOR
(burndown_ranker) still ranks new hypotheses in those families
normally. Result: research capacity spent on families where the
prior N observations were all RED.

This module bridges the gap OUTSIDE the belief-1 → belief-4 loop:
it feeds a `dying_family_penalty` factor into the ranker so we don't
prioritize proposals in families with no observed successes.

Doctrine boundary
=================
- Air-gap preserved: the ranker signal is a SELECTION heuristic. It
  never enters belief-1's PREDICTION path (which would corrupt the
  calibration measurement). Ranker + predictor are decoupled by design.
- Conservative: penalty is a MULTIPLICATIVE factor, not a hard block.
  Principal / autopilot / /approvals can still override — this only
  affects rank_score ordering.
- Reversible: the penalty is derived at query-time from autopsies,
  so as soon as a dying family accumulates a GREEN verdict it exits
  the set automatically. No manual list to maintain.

Detection rule (v22 initial)
============================
A family is "dying" if:
  - N >= MIN_AUTOPSIES     (need SOME data — avoids all N=0 families
                             falling into the set spuriously)
  - GREEN_rate <= MAX_GREEN_RATE  (0.10 = allow one lucky GREEN in
                                    ten-plus autopsies)
  - MARGINAL_rate + GREEN_rate <= MAX_POSITIVE_RATE (0.30 = at least
                                                     70% RED)

The thresholds are deliberate:
  - Sitting purely at 100%-RED N>=2 misses "hopeless with a coin-flip"
    families (e.g. 3 RED + 1 MARGINAL in 4 autopsies — still very
    unattractive)
  - Requires N >= 2 so we don't flag brand-new families (N=1) that
    just happened to draw a RED on their first observation
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AUTOPSIES_PATH = _REPO_ROOT / "data" / "research" / "autopsies.jsonl"

MIN_AUTOPSIES:     int   = 2
MAX_GREEN_RATE:    float = 0.10
MAX_POSITIVE_RATE: float = 0.30    # (GREEN + MARGINAL) / N

# Ranker uses this to scale down candidates in dying families.
# 0.5 halves rank_score; not a hard block. Human review at /approvals
# can still promote a specific hypothesis regardless of the penalty.
RANKER_PENALTY: float = 0.5


def _iter_autopsies(path: Path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as fh:
        for ln_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("dying_families: %s line %d malformed",
                                 path.name, ln_no)


def _family_stats(autopsies_path: Optional[Path] = None) -> dict[str, dict]:
    """Group autopsies by strategy_family. Excludes rows with
    superseded_by. Returns dict: {FAMILY: {n, GREEN, MARGINAL, RED,
    NEUTRAL}}.
    """
    stats: dict[str, dict] = {}
    p = autopsies_path or _AUTOPSIES_PATH
    for row in _iter_autopsies(p):
        if row.get("superseded_by"):
            continue
        fam = (row.get("strategy_family") or "").upper()
        if not fam:
            continue
        actual = row.get("actual_verdict") or ""
        slot = stats.setdefault(fam, {
            "n": 0, "GREEN": 0, "MARGINAL": 0, "RED": 0, "NEUTRAL": 0,
        })
        slot["n"] += 1
        if actual in slot:
            slot[actual] += 1
    return stats


def load_dying_families(
    *,
    autopsies_path: Optional[Path] = None,
    min_autopsies:    int   = MIN_AUTOPSIES,
    max_green_rate:   float = MAX_GREEN_RATE,
    max_positive_rate: float = MAX_POSITIVE_RATE,
) -> set[str]:
    """Return the set of family names that meet the dying criteria.

    The rate thresholds are exposed as kwargs for tests + future
    tuning — production callers should use the defaults.
    """
    stats = _family_stats(autopsies_path)
    out: set[str] = set()
    for fam, s in stats.items():
        n = s["n"]
        if n < min_autopsies:
            continue
        green_rate = s["GREEN"] / n
        positive_rate = (s["GREEN"] + s["MARGINAL"]) / n
        if green_rate <= max_green_rate and positive_rate <= max_positive_rate:
            out.add(fam)
    return out


def dying_family_penalty(
    family: str,
    dying_set: set[str],
    *,
    penalty: float = RANKER_PENALTY,
) -> float:
    """Returns `penalty` if family is in the dying set, else 1.0.

    Extracted as its own tiny function so the ranker composition
    remains a clean `novelty * demand * recency * dying_penalty`
    read at the call site."""
    return penalty if (family or "").upper() in dying_set else 1.0


def dying_families_report(
    autopsies_path: Optional[Path] = None,
) -> dict:
    """Diagnostic report — used by scripts + tests to inspect why a
    family did / didn't land in the set. Returns per-family stats
    plus the current dying-set membership."""
    stats = _family_stats(autopsies_path)
    families = []
    for fam, s in sorted(stats.items(), key=lambda x: -x[1]["n"]):
        n = s["n"]
        green_rate = s["GREEN"] / n if n else 0.0
        positive_rate = (s["GREEN"] + s["MARGINAL"]) / n if n else 0.0
        families.append({
            "family":         fam,
            "n":              n,
            "green":          s["GREEN"],
            "marginal":       s["MARGINAL"],
            "red":            s["RED"],
            "neutral":        s["NEUTRAL"],
            "green_rate":     round(green_rate, 4),
            "positive_rate":  round(positive_rate, 4),
            "is_dying":       (
                n >= MIN_AUTOPSIES
                and green_rate <= MAX_GREEN_RATE
                and positive_rate <= MAX_POSITIVE_RATE
            ),
        })
    return {
        "thresholds": {
            "min_autopsies":     MIN_AUTOPSIES,
            "max_green_rate":    MAX_GREEN_RATE,
            "max_positive_rate": MAX_POSITIVE_RATE,
            "ranker_penalty":    RANKER_PENALTY,
        },
        "n_families_dying": sum(1 for f in families if f["is_dying"]),
        "families":         families,
    }
