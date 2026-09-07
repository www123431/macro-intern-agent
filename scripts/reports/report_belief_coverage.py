"""Belief Layer Phase 4 coverage report.

Surfaces three things the principal needs to know about the
closed-loop calibration (belief-4):

  1. PER-FAMILY ELIGIBILITY — how many autopsies each family has,
     whether it clears MIN_AUTOPSIES_FOR_OVERRIDE, and what the
     calibrated prior looks like vs the hand-calibrated override.

  2. OVERALL COVERAGE — what fraction of autopsies are in families
     that get the calibrated prior vs fall through to overrides /
     default. Higher = belief-4 is doing more work.

  3. PRIOR DRIFT — for eligible families, how far the calibrated
     posterior has moved from the original FAMILY_PRIOR_OVERRIDES.
     Large drift means the hand-calibrated prior was significantly
     off; tiny drift means the prior is well-anchored.

Output: writes Markdown to data/research/belief_coverage.md so the
report can be linked from Cockpit / pulled into commit notes. Pure
function over autopsies — no LLM, no network, no live verdict.

Usage:
    python scripts/reports/report_belief_coverage.py
    python scripts/reports/report_belief_coverage.py --json    # JSON only
    python scripts/reports/report_belief_coverage.py --out PATH
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_AUTOPSIES_PATH = _REPO_ROOT / "data" / "research" / "autopsies.jsonl"
_DEFAULT_OUT    = _REPO_ROOT / "data" / "research" / "belief_coverage.md"
_DEFAULT_JSON   = _REPO_ROOT / "data" / "research" / "belief_coverage.json"


def _load_autopsies(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for ln_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Quiet skip — main belief module logs malformed rows
                continue
    return out


def _family_breakdown(autopsies: list[dict]) -> dict[str, dict]:
    """Per-family verdict counts (excluding superseded rows)."""
    out: dict[str, dict] = {}
    for a in autopsies:
        if a.get("superseded_by"):
            continue
        fam = (a.get("strategy_family") or "UNKNOWN").upper()
        slot = out.setdefault(fam, {"GREEN": 0, "MARGINAL": 0, "RED": 0,
                                       "NEUTRAL": 0, "n": 0})
        v = a.get("actual_verdict") or ""
        if v in slot:
            slot[v] += 1
        slot["n"] += 1
    return out


def _kl_div(p: dict[str, float], q: dict[str, float]) -> float:
    """KL(p || q) for verdict distributions. Returns 0 if either degenerate."""
    import math
    out = 0.0
    for k in ("GREEN", "MARGINAL", "RED"):
        pi = float(p.get(k, 0.0))
        qi = float(q.get(k, 0.0))
        if pi <= 0 or qi <= 0:
            continue
        out += pi * math.log(pi / qi)
    return out


def compute_report() -> dict:
    """Pure function. Returns a dict ready for rendering."""
    from engine.research.belief_prior_calibration import (
        MIN_AUTOPSIES_FOR_OVERRIDE,
        calibrated_family_prior,
    )
    from engine.research.belief import (
        DEFAULT_PRIOR,
        FAMILY_PRIOR_OVERRIDES,
    )

    autopsies = _load_autopsies(_AUTOPSIES_PATH)
    fam_data  = _family_breakdown(autopsies)

    families = []
    for fam, vc in sorted(fam_data.items(), key=lambda x: -x[1]["n"]):
        n = vc["n"]
        eligible = n >= MIN_AUTOPSIES_FOR_OVERRIDE
        calibrated = calibrated_family_prior(fam) if eligible else None
        override = (FAMILY_PRIOR_OVERRIDES.get(fam) or
                    DEFAULT_PRIOR)
        kl = (_kl_div(calibrated, override)
              if calibrated is not None else None)
        empirical_dist = (
            {k: vc[k] / n for k in ("GREEN", "MARGINAL", "RED")}
            if n > 0 else None
        )
        families.append({
            "family":          fam,
            "n":               n,
            "verdict_counts":  {k: vc[k] for k in ("GREEN", "MARGINAL",
                                                      "RED", "NEUTRAL")},
            "empirical_dist":  empirical_dist,
            "eligible":        eligible,
            "calibrated_prior": calibrated,
            "override_prior":  override,
            "kl_from_override": kl,
        })

    n_total = sum(f["n"] for f in families)
    n_eligible_families = sum(1 for f in families if f["eligible"])
    n_eligible_autopsies = sum(f["n"] for f in families if f["eligible"])

    summary = {
        "n_autopsies":               n_total,
        "n_families":                len(families),
        "min_autopsies_for_override": MIN_AUTOPSIES_FOR_OVERRIDE,
        "eligible_families":         n_eligible_families,
        "eligible_autopsies":        n_eligible_autopsies,
        "coverage_pct":              (n_eligible_autopsies / n_total * 100.0
                                       if n_total else 0.0),
    }

    return {
        "reported_ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "summary":     summary,
        "families":    families,
    }


def render_markdown(report: dict) -> str:
    s = report["summary"]
    families = report["families"]

    lines = []
    lines.append("# Belief Layer Phase 4 — Coverage Report")
    lines.append("")
    lines.append(f"_Generated: {report['reported_ts']}_")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"- **{s['eligible_autopsies']}/{s['n_autopsies']} autopsies "
        f"({s['coverage_pct']:.1f}%)** in families with the "
        f"closed-loop calibrated prior active "
        f"(threshold N >= {s['min_autopsies_for_override']}).")
    lines.append(
        f"- **{s['eligible_families']}/{s['n_families']} distinct "
        f"families** clear the calibration cutoff.")
    lines.append("")

    lines.append("## Per-family breakdown")
    lines.append("")
    lines.append("Legend: `n` = autopsy count. `Empirical G/M/R` = raw "
                 "frequencies. `Override G` = the hand-calibrated "
                 "GREEN prior (FAMILY_PRIOR_OVERRIDES). `Calibrated "
                 "G` = belief-4 Dirichlet-posterior GREEN (None when "
                 "below threshold). `KL drift` = how far the "
                 "calibrated posterior moved from the override.")
    lines.append("")
    lines.append("| family | n | G | M | R | Override G | Calibrated G | KL drift |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for f in families:
        n = f["n"]
        emp = f["empirical_dist"] or {}
        override_g = f["override_prior"].get("GREEN", 0.0)
        cal_g = (f["calibrated_prior"].get("GREEN")
                 if f["calibrated_prior"] is not None else None)
        cal_g_s = f"{cal_g:.2f}" if cal_g is not None else "—"
        kl = f["kl_from_override"]
        kl_s = f"{kl:.3f}" if kl is not None else "—"
        flag = "" if f["eligible"] else " *(below cutoff)*"
        lines.append(
            f"| `{f['family']}`{flag} | {n} | "
            f"{emp.get('GREEN',0):.2f} | "
            f"{emp.get('MARGINAL',0):.2f} | "
            f"{emp.get('RED',0):.2f} | "
            f"{override_g:.2f} | {cal_g_s} | {kl_s} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- **Large KL drift** (> 0.5) means the hand-calibrated "
                 "override was significantly off from the data. These "
                 "families benefited most from the closed-loop update.")
    lines.append("- **Families below cutoff** with empirical G = 0% but "
                 "override G > 0.10 are systematically over-optimistic "
                 "until they accumulate more autopsies. Manual review "
                 "of FAMILY_PRIOR_OVERRIDES recommended for families "
                 "stuck at N=1-2.")
    lines.append("- A family showing **100% RED at N < cutoff** is a "
                 "candidate for either (a) further data collection, "
                 "(b) being merged into a parent family for "
                 "calibration purposes, or (c) hard prior override.")

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",  action="store_true",
                     help="Print JSON to stdout instead of writing the "
                          "Markdown report.")
    ap.add_argument("--out",   type=Path, default=_DEFAULT_OUT,
                     help="Markdown output path.")
    ap.add_argument("--json-out", type=Path, default=_DEFAULT_JSON,
                     help="JSON output path (for downstream consumers).")
    args = ap.parse_args()

    report = compute_report()

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2,
                                          ensure_ascii=False),
                              encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    md = render_markdown(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")

    s = report["summary"]
    print(f"Wrote {args.out}")
    print(f"  coverage: {s['eligible_autopsies']}/{s['n_autopsies']} "
          f"autopsies ({s['coverage_pct']:.1f}%) in "
          f"{s['eligible_families']}/{s['n_families']} families")
    return 0


if __name__ == "__main__":
    sys.exit(main())
