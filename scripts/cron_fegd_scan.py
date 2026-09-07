"""scripts/cron_fegd_scan.py — weekly Factor Exposure Gap Detector scan.

Wraps `engine.research.factor_exposure_gap_detector.emit_fegd_demand`
for all deployed alpha sleeves so factor-exposure gaps surface in the
demand ledger automatically — closing the YAML-curated-knowledge
blindspot identified 2026-06-17.

What this does
==============
For each deployed sleeve (equity_book / cross_asset_carry /
cross_asset_tsmom):

  1. Build monthly PnL via combined_book builder
  2. Regress on canonical factor matrix
     (MKT_RF SMB HML RMW CMA MOM BAB VRP XA_CARRY XA_TSMOM)
  3. Identify factors where |t-stat| < GAP_T_THRESHOLD (= 1.65)
  4. Emit one capability_gaps row per (sleeve, gap_factor) tagged
     `source:fegd_factor_gap`. burndown_ranker reads these and
     applies a ×1.5 demand multiplier to the matching family.

Idempotent: re-emissions of the same (sleeve, family, gap_factor)
signature are skipped (`emit_fegd_demand` checks before write).
Safe to schedule weekly (it's almost a no-op when nothing changed).

Why a separate cron from deployment_demand_emitter
==================================================
`deployment_demand_emitter` reads the human-curated
`improvement_directions` in active_deployment.yaml. FEGD detects
gaps the human DIDN'T list — a fundamentally different signal
source. Both write to the same `data/research/capability_gaps.jsonl`
with distinguishing `source:` tags; the ranker doesn't care which
source surfaced a family, but the audit trail does.

Health
======
Writes a status row to `data/agents/_health/fegd_scan.jsonl` so the
AgentHealth tile can render "FEGD last ran X ago — N gaps detected"
alongside DailyMemo / DirectionProposer / DecayAudit.

Cadence
=======
Weekly Mondays 06:50 SGT — between WorkflowExecutor (06:40) and the
weekly chief_of_staff substrate (Sun 10:00). Slow-changing signal,
weekly is plenty.

Invocation
==========
    python scripts/cron_fegd_scan.py            # write (production)
    python scripts/cron_fegd_scan.py --dry-run  # preview only

Scheduled via Windows Task Scheduler — see scripts/install_agentic_cron.py
entry MacroAlphaPro_FEGDScan.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HEALTH_PATH = REPO_ROOT / "data" / "agents" / "_health" / "fegd_scan.jsonl"


def _record(status: str, *, elapsed_s: float, parsed: int = 0,
             present: int = 0, written: int = 0,
             error: str | None = None) -> None:
    HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "agent_id":  "fegd_scan",
        "ts":        _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "status":    status,
        "elapsed_s": round(elapsed_s, 2),
        "date_key":  _dt.date.today().isoformat(),
        "parsed":    parsed,
        "already_present": present,
        "written":   written,
    }
    if error:
        row["error"] = error[:500]
    with HEALTH_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def run_once(*, dry_run: bool) -> dict:
    """Pure callable for tests + the script entry-point. Returns a
    summary dict with per-sleeve + aggregate counts."""
    from engine.portfolio.combined_book import (
        build_equity_book, build_carry_book, build_tsmom_book,
    )
    from engine.research.factor_exposure_gap_detector import (
        build_canonical_factor_matrix, emit_fegd_demand,
    )

    # Exclude the sleeve's own factor proxy when regressing (avoid
    # cross_asset_carry → XA_CARRY self-regression circular loading).
    sleeves = [
        ("equity_book",       build_equity_book, ()),
        ("cross_asset_carry", build_carry_book,  ("XA_CARRY",)),
        ("cross_asset_tsmom", build_tsmom_book,  ("XA_TSMOM",)),
    ]

    fm = build_canonical_factor_matrix()

    total_parsed = 0
    total_present = 0
    total_written = 0
    per_sleeve: list[dict] = []
    for sleeve_id, builder, excl in sleeves:
        s = builder().dropna()
        result = emit_fegd_demand(
            sleeve_id=sleeve_id, sleeve_pnl=s,
            factor_matrix=fm, exclude_factors=excl, dry_run=dry_run,
        )
        per_sleeve.append({
            "sleeve_id":       sleeve_id,
            "parsed":          result["parsed"],
            "already_present": result["already_present"],
            "written":         result["written"],
            "new_rows":        result["new_rows"],
        })
        total_parsed  += result["parsed"]
        total_present += result["already_present"]
        total_written += result["written"]

    return {
        "dry_run":         dry_run,
        "factor_matrix_cols": list(fm.columns),
        "total_parsed":    total_parsed,
        "total_present":   total_present,
        "total_written":   total_written,
        "per_sleeve":      per_sleeve,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                     help="preview only — do not commit rows to "
                          "capability_gaps.jsonl")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = _dt.datetime.utcnow()
    try:
        summary = run_once(dry_run=args.dry_run)
        elapsed = (_dt.datetime.utcnow() - t0).total_seconds()
        _record(
            "ok", elapsed_s=elapsed,
            parsed=summary["total_parsed"],
            present=summary["total_present"],
            written=summary["total_written"],
        )
        mode = "DRY-RUN" if args.dry_run else "WRITE"
        print(f"[cron_fegd_scan] {mode} ok — parsed={summary['total_parsed']} "
              f"already_present={summary['total_present']} "
              f"written={summary['total_written']} in {elapsed:.1f}s")
        for ps in summary["per_sleeve"]:
            print(f"  {ps['sleeve_id']:<22} parsed={ps['parsed']:>2} "
                  f"present={ps['already_present']:>2} written={ps['written']:>2}")
            for row in ps["new_rows"]:
                label = "WOULD write" if args.dry_run else "wrote"
                print(f"    {label} [{row['family']:<22}] {row['gap_factor']:<10} "
                      f"β={row['beta']:+.3f} t={row['t_stat']:+.3f}")
        return 0
    except Exception as exc:
        elapsed = (_dt.datetime.utcnow() - t0).total_seconds()
        _record("error", elapsed_s=elapsed,
                 error=f"{type(exc).__name__}: {exc}")
        logger.exception("cron_fegd_scan failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
