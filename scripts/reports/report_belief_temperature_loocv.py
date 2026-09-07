"""LOOCV Brier measurement for temperature-scaling calibration.

Phase 5 (offline). Evaluates whether single-scalar temperature scaling
post-hoc calibration improves predictor Brier under leave-one-out
cross-validation. Honest measurement — does NOT wire the calibrator
into production. The decision to ship Phase 5.1 (production wire-in)
is gated on this report.

Output: writes Markdown + JSON to data/research/belief_temperature.md
and .json for Cockpit / commit-note linkage.

Headline question: does fitting T improve held-out Brier?
  Yes (relative_improvement > 0)  → consider Phase 5.1 wire-in
  No / negative                    → publish as negative finding,
                                     try per-class methods at higher N

Usage:
  python scripts/reports/report_belief_temperature_loocv.py
  python scripts/reports/report_belief_temperature_loocv.py --json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from engine.research.belief_temperature_calibrator import (
    fit_from_autopsies,
    loocv_brier,
    _load_autopsies,
)

_DEFAULT_OUT  = _REPO_ROOT / "data" / "research" / "belief_temperature.md"
_DEFAULT_JSON = _REPO_ROOT / "data" / "research" / "belief_temperature.json"


def compute_report() -> dict:
    rows = _load_autopsies()
    in_sample = fit_from_autopsies()
    loocv = loocv_brier(rows)
    return {
        "reported_ts":  _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "n_autopsies":  in_sample.n_autopsies,
        "in_sample": {
            "temperature":           in_sample.temperature,
            "brier_uncalibrated":    in_sample.brier_uncalibrated,
            "brier_calibrated":      in_sample.brier_calibrated,
            "relative_improvement":  in_sample.relative_improvement,
            "iterations":            in_sample.iterations,
        },
        "loocv": loocv,
    }


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Belief Layer Phase 5 — Temperature Scaling LOOCV Report")
    lines.append("")
    lines.append(f"_Generated: {report['reported_ts']}_")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    n = report["n_autopsies"]
    loocv = report["loocv"]
    in_sample = report["in_sample"]
    rel_loocv = loocv.get("relative_improvement", 0.0)
    verdict = (
        "**POSITIVE** — held-out Brier improves; consider Phase 5.1 wire-in"
        if rel_loocv > 0.01
        else "**NEGATIVE** — held-out Brier does NOT improve; single-scalar "
             "temperature scaling is insufficient. Per-class calibration "
             "(isotonic / Platt) would need more data than n=" + str(n)
             + " to validate."
    )
    lines.append(f"- Sample size: **n = {n} autopsies**")
    lines.append(f"- In-sample T fit: **{in_sample['temperature']}** "
                  f"({'sharpens' if in_sample['temperature'] < 1 else 'flattens'} "
                  f"the predicted distribution)")
    lines.append(f"- In-sample Brier: "
                  f"{in_sample['brier_uncalibrated']:.4f} → "
                  f"{in_sample['brier_calibrated']:.4f} "
                  f"({in_sample['relative_improvement']:.1%} improvement)")
    if loocv.get("supported"):
        lines.append(f"- LOOCV Brier (honest): "
                      f"{loocv['brier_loocv_uncalibrated']:.4f} → "
                      f"{loocv['brier_loocv_calibrated']:.4f} "
                      f"({rel_loocv:.1%} improvement)")
        lines.append(f"- Held-out T median: {loocv['t_median_held_out']}; "
                      f"range [{loocv['t_min']}, {loocv['t_max']}]")
    lines.append("")
    lines.append(f"**Conclusion**: {verdict}")
    lines.append("")
    lines.append("## What this means")
    lines.append("")
    lines.append("Temperature scaling applies a single scalar transformation "
                 "to every predicted distribution: `p_calibrated = softmax("
                 "log(p) / T)`. T<1 sharpens (more confident); T>1 flattens "
                 "(less confident). It's the simplest post-hoc calibration "
                 "method that exists (Guo et al. 2017).")
    lines.append("")
    if rel_loocv <= 0.01:
        lines.append("The fact that LOOCV doesn't improve means our system's "
                      "miscalibration is NOT a uniform over- or under-confidence — "
                      "it's bin-specific (the Hosmer-Lemeshow test, which rejected "
                      "at p=0.0469 in W6, is sensitive to bin-level errors that a "
                      "single T can't address).")
        lines.append("")
        lines.append("Plausible next steps when more data is available:")
        lines.append("- **Per-class isotonic regression** (one-vs-rest for "
                     "GREEN / MARGINAL / RED)")
        lines.append("- **Per-family temperature** (different T per "
                     "strategy_family with sufficient sample)")
        lines.append("- **Calibration error stratified by source tier** "
                     "(belief-4 vs override vs default) — see "
                     "data/research/belief_coverage.md from v19")
    else:
        lines.append("LOOCV improvement justifies cautious Phase 5.1 wire-in "
                     "of `apply_temperature` at the tail of "
                     "`belief.predict_verdict`. Recommended: ship behind a "
                     "feature flag (e.g. `BELIEF_TEMPERATURE_CALIBRATION_"
                     "ENABLED`) with the T value frozen from this run, then "
                     "re-measure on the next ~25 autopsies before promoting "
                     "to default-on.")
    lines.append("")
    lines.append("## Doctrine alignment")
    lines.append("")
    lines.append("This report is an instance of the CLAUDE.md \"honest "
                 "negative finding publication\" doctrine. A 0% / negative "
                 "Brier delta IS a result — it tells future-us not to spend "
                 "Phase 5.1 effort on this approach yet. The calibrator code "
                 "stays shipped (it's a small, well-tested module) so the "
                 "LOOCV report can be re-run when n grows and the answer "
                 "may flip.")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true",
                     help="Print JSON to stdout instead of writing Markdown.")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path, default=_DEFAULT_JSON)
    args = ap.parse_args()

    report = compute_report()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    md = render_markdown(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    rel_loocv = report["loocv"].get("relative_improvement", 0.0)
    print(f"Wrote {args.out}")
    print(f"  n={report['n_autopsies']}, in-sample T={report['in_sample']['temperature']}, "
          f"LOOCV rel_improvement={rel_loocv:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
