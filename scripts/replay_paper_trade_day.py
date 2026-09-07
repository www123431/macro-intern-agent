"""scripts/replay_paper_trade_day.py — v51.

Replay a historical paper-trade day WITHOUT mutating live state
(position book / NAV history / production attribution log / RM ledger).

Usage:
    python scripts/replay_paper_trade_day.py --as-of 2026-05-27
    python scripts/replay_paper_trade_day.py --backfill 2026-05-27 2026-06-09

Design intent
=============
Live daily runner (`run_paper_trade_daily.py`) is forward-mode: reads
current config, mutates position book, triggers RM checks, writes to
prod tables. A replay of a historical date on that path corrupts the
current state because:
  - position book grows with historical trades → drifts vs current config
  - RM pre-trade sees "37.5% ss_sp500 vs 29.5% expected" alarm and halts
  - even --ignore-circuit-breaker doesn't bypass the RM inner check

This script bypasses that by calling ONLY the orchestrator
(pure computation `run_paper_trade_day(as_of)`) and serializing the
result to a SHADOW log at
    data/paper_trade/replay_attribution_log.jsonl
The live prod attribution_log / nav_history / position book are
NEVER touched.

Limitations (intentional MVP scope)
===================================
  - No RM / DQ checks (their whole reason for existing is to guard
    the live path we're not on)
  - Uses CURRENT config, not historical config snapshot. If the user
    wants to replay a day under a different config, they'd need to
    check out the git commit for that config first. Historical
    active_deployment.yaml has a `history:` section but the runner
    doesn't parameterize on it.
  - No NAV / UI artifact generation. Shadow file has trade-level detail
    only. Aggregate stats can be computed by the caller off the shadow.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_SHADOW_ATTR = _REPO_ROOT / "data" / "paper_trade" / "replay_attribution_log.jsonl"


def _serialize_result_to_shadow(as_of: _dt.date, result, ts_utc: str) -> int:
    """Append per-position rows to shadow attribution log. Returns rows written."""
    _SHADOW_ATTR.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with _SHADOW_ATTR.open("a", encoding="utf-8") as fh:
        for sig in result.signals:
            weights = sig.weights
            # weights may be pd.Series or dict; normalize to items iterator
            if weights is None:
                continue
            if hasattr(weights, "items"):
                items = weights.items()
            else:
                items = ((k, weights[k]) for k in getattr(weights, "index", []))
            for ticker, weight in items:
                row = {
                    "date":             as_of.isoformat(),
                    "as_of":            as_of.isoformat(),
                    "replay_ts":        ts_utc,
                    "strategy_name":    sig.strategy_name,
                    "sleeve_id":        sig.sleeve_id,
                    "status":           sig.status,
                    "ticker":           ticker,
                    "weight":           float(weight),
                    "side":             "long" if weight > 0 else "short" if weight < 0 else "flat",
                    "intra_sleeve_weight": float(sig.intra_sleeve_weight),
                    "replay_mode":      True,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
    return n_written


def _replay_one_day(as_of: _dt.date, logger: logging.Logger) -> int:
    """Compute what paper trade WOULD have been on as_of + write to shadow.
    Returns rows written."""
    from engine.portfolio.paper_trade_combined import run_paper_trade_day
    ts_utc = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        result = run_paper_trade_day(as_of)
    except Exception as e:
        logger.error("orchestrator raised: %s: %s", type(e).__name__, str(e)[:200])
        return 0

    n_sigs = len(result.signals)
    def _wlen(w):
        if w is None:  return 0
        try: return len(w)
        except TypeError:
            return sum(1 for _ in w)
    total_positions = sum(_wlen(sig.weights) for sig in result.signals)
    gross = float(result.combined_portfolio.abs().sum()) if len(result.combined_portfolio) else 0.0
    logger.info(
        "%s replay: %d strategies %d positions gross=%.4f",
        as_of, n_sigs, total_positions, gross,
    )
    n = _serialize_result_to_shadow(as_of, result, ts_utc)
    logger.info("  shadow rows written: %d", n)
    return n


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("replay_paper_trade_day")

    ap = argparse.ArgumentParser(
        description="v51 (2026-07-02) — replay a historical paper-trade day "
                    "to shadow log without mutating live prod state.",
    )
    ap.add_argument("--as-of", type=str,
                     help="Single date to replay YYYY-MM-DD")
    ap.add_argument("--backfill", nargs=2, metavar=("START", "END"),
                     help="Backfill business days from START to END inclusive")
    args = ap.parse_args()

    if args.as_of and args.backfill:
        logger.error("--as-of and --backfill are mutually exclusive")
        return 2
    if not args.as_of and not args.backfill:
        logger.error("supply either --as-of YYYY-MM-DD or --backfill START END")
        return 2

    dates: list[_dt.date] = []
    if args.as_of:
        dates.append(_dt.date.fromisoformat(args.as_of))
    else:
        import pandas as pd
        start = _dt.date.fromisoformat(args.backfill[0])
        end   = _dt.date.fromisoformat(args.backfill[1])
        bdays = pd.bdate_range(start, end)
        dates.extend([d.date() for d in bdays])

    logger.info("replaying %d business day(s) → %s", len(dates), _SHADOW_ATTR)

    total_rows = 0
    failed: list[_dt.date] = []
    for d in dates:
        try:
            n = _replay_one_day(d, logger)
            total_rows += n
            if n == 0:
                failed.append(d)
        except Exception as e:
            logger.error("day %s hard-failed: %s", d, str(e)[:200])
            failed.append(d)

    logger.info("=== Replay summary ===")
    logger.info("  days requested: %d", len(dates))
    logger.info("  days succeeded: %d", len(dates) - len(failed))
    logger.info("  days failed:    %d %s", len(failed), failed[:8])
    logger.info("  total shadow rows: %d", total_rows)
    logger.info("  shadow log:  %s", _SHADOW_ATTR)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
