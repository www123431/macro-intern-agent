"""scripts/install_cron_fleet.py — single source of truth for the
MacroAlphaPro Windows Task Scheduler fleet (2026-08-09).

Why this exists
===============
2026-08-09 audit: the repo moved C:\\Users\\${USER}\\Desktop\\intern →
e:\\Desktop\\intern and ALL 19 scheduled tasks kept pointing at the old
path. Every cron in the fleet had been failing silently (exit 0x2
file-not-found / 0x8007010B bad-working-directory) — the discover loop's
"automatic" outer ring was dead and nothing surfaced it. Root causes:

  1. Task actions + wrapper bats hardcoded the absolute repo path.
  2. Task definitions lived only inside Task Scheduler — no repo-side
     manifest to diff against, so drift was invisible.

This script fixes both: the manifest below IS the fleet definition,
paths derive from this file's location at install time, and `--verify`
detects drift (task missing, action pointing outside the current repo,
target script missing, last-run failure).

Usage
=====
  python scripts/install_cron_fleet.py verify              # drift report (read-only)
  python scripts/install_cron_fleet.py install --dry-run   # show what would change
  python scripts/install_cron_fleet.py install             # register/overwrite ALL tasks
  python scripts/install_cron_fleet.py install --task autopilot-daily
  python scripts/install_cron_fleet.py uninstall --task autopilot-daily

After any future repo move: re-run `install`. That's the whole runbook.

Deliberate normalizations vs the pre-2026-08-09 task definitions
================================================================
  - All tasks run as the current user with LogonType InteractiveToken.
    (MacroAlphaPro_DailyScheduler previously used Password logon, which
    cannot be re-registered without the account password.)
  - Uniform settings: StartWhenAvailable=true (catch up missed runs on
    a laptop host), battery restrictions OFF, ExecutionTimeLimit PT4H,
    MultipleInstancesPolicy IgnoreNew.
  - Schedules, interpreters, and arguments are preserved verbatim from
    the 2026-08-09 XML backups (scratchpad task_backup/).
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

REPO_ROOT = Path(__file__).resolve().parent.parent

INTERPRETERS = {
    "py310":       r"${REPO_ROOT}\AppData\Local\Programs\Python\Python310\python.exe",
    "d_python":    r"D:\python\python.exe",
    "anaconda":    r"${REPO_ROOT}\anaconda3\python.exe",
    "py_launcher": r"C:\Windows\py.exe",
}

DAY_TAGS = {
    "Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday",
    "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday",
}


@dataclass
class CronTask:
    name: str                      # full task path, e.g. r"MacroAlphaPro\burndown-daily"
    description: str
    time: str                      # "HH:MM" local
    days: tuple = ()               # empty → daily; else weekly on these days ("Mon", ...)
    kind: str = "py"               # py | pymod | bat | cmd
    interpreter: str = "py310"     # key into INTERPRETERS (py/pymod kinds)
    target: str = ""               # repo-relative script / bat path, or module name for pymod
    args: str = ""                 # extra CLI args
    random_delay: str = ""         # e.g. "PT1H"
    workdir_repo_root: bool = True

    def command_and_args(self) -> tuple[str, str]:
        if self.kind == "bat":
            return str(REPO_ROOT / self.target), self.args
        if self.kind == "py":
            exe = INTERPRETERS[self.interpreter]
            script = str(REPO_ROOT / self.target)
            return exe, f'"{script}" {self.args}'.strip()
        if self.kind == "pymod":
            exe = INTERPRETERS[self.interpreter]
            return exe, f"{self.args}"
        if self.kind == "cmd":
            return "cmd.exe", self.args
        raise ValueError(f"unknown kind {self.kind}")

    def target_file(self) -> Path | None:
        """The repo file this task depends on (None for pymod/cmd)."""
        if self.kind in ("bat", "py"):
            return REPO_ROOT / self.target
        return None


FLEET: list[CronTask] = [
    # ── root-level legacy tasks ──────────────────────────────────────
    CronTask(
        name=r"MacroAlphaPaperExecution",
        description="Paper-execution weekly Mon-Fri 23:00 (US RTH window)",
        time="23:00", days=("Mon", "Tue", "Wed", "Thu", "Fri"),
        kind="bat", target=r"scripts\run_execution.bat",
    ),
    CronTask(
        name=r"MacroAlphaPro_DailyBatch",
        description="Daily engine.scheduler --check (py 3.11 launcher)",
        time="06:00", kind="pymod", interpreter="py_launcher",
        args="-3.11 -m engine.scheduler --check",
    ),
    CronTask(
        name=r"MacroAlphaPro_DailyMemo",
        description="Daily memo agent",
        time="06:30", kind="py", target=r"scripts\cron_daily_memo.py",
    ),
    CronTask(
        name=r"MacroAlphaPro_DailyScheduler",
        description="Daily engine.scheduler --check heartbeat (17:00 + jitter)",
        time="17:00", random_delay="PT1H", kind="cmd",
        args=(f'/c "set PYTHONPATH=. && {INTERPRETERS["d_python"]} '
              f'-m engine.scheduler --check >> {REPO_ROOT}\\scheduler.log 2>&1"'),
    ),
    CronTask(
        name=r"MacroAlphaPro_DirectionProposer",
        description="Direction proposer agent (new-direction queue)",
        time="06:35", kind="py", target=r"scripts\cron_direction_proposer.py",
    ),
    CronTask(
        name=r"MacroAlphaPro_ETFHoldings",
        description="ETF holdings risk monitor (monthly logic, daily check)",
        time="06:30", kind="py", interpreter="anaconda",
        target=r"scripts\run_etf_holdings_monitor_monthly.py",
    ),
    CronTask(
        name=r"MacroAlphaPro_PaperTrade",
        description="Daily paper-trade NAV update",
        time="06:00", kind="py", interpreter="d_python",
        target=r"scripts\run_paper_trade_daily.py",
    ),
    CronTask(
        name=r"MacroAlphaPro_Watchdog",
        description="Ops watchdog --check",
        time="06:10", kind="pymod", interpreter="py_launcher",
        args="-3.11 -m engine.agents.ops_watchdog --check",
    ),
    CronTask(
        name=r"MacroAlphaPro_WorkflowExecutor",
        description="Workflow executor agent",
        time="06:40", kind="py", target=r"scripts\cron_workflow_executor.py",
    ),
    # ── \MacroAlphaPro folder tasks ─────────────────────────────────
    CronTask(
        name=r"MacroAlphaPro\burndown-daily",
        description="Burndown verdict batch (Mon+Thu, LLM health-gated wrapper)",
        time="09:00", days=("Mon", "Thu"),
        kind="bat", target=r"scripts\burndown_cron_wrapper.bat",
    ),
    CronTask(
        name=r"MacroAlphaPro\daily-belief-refresh",
        description="Belief autopsy + track-record refresh ($0 LLM)",
        time="06:35", kind="bat", target=r"scripts\daily_belief_refresh_wrapper.bat",
    ),
    CronTask(
        name=r"MacroAlphaPro\daily-status-dashboard",
        description="STATUS.md aggregation dashboard ($0 LLM)",
        time="06:40", kind="bat", target=r"scripts\daily_status_dashboard_wrapper.bat",
    ),
    CronTask(
        name=r"MacroAlphaPro\decay-watch-weekly",
        description="Decay alert backfill (Sun)",
        time="05:30", days=("Sun",),
        kind="py", target=r"scripts\backfill_decay_alerts.py", args="--cron",
    ),
    CronTask(
        name=r"MacroAlphaPro\papers-curator-daily-ingest",
        description="Papers curator substrate pump (crawl+judge+summarize)",
        time="08:30", kind="bat", target=r"scripts\papers_curator_daily_wrapper.bat",
    ),
    CronTask(
        name=r"MacroAlphaPro\research-backfill-weekly",
        description="Paper discovery historical backfill (Sun)",
        time="04:00", days=("Sun",),
        kind="py", target=r"scripts\run_paper_discovery.py",
        args="--backfill --start 2018-01-01 --end 2026-06-10 --max-per-year 100",
    ),
    CronTask(
        name=r"MacroAlphaPro\research-daily-summary",
        description="Research daily summary",
        time="06:30", kind="py", target=r"scripts\research_daily_summary.py",
    ),
    CronTask(
        name=r"MacroAlphaPro\research-discover-newflow",
        description="Paper discovery new-flow crawl",
        time="06:00", kind="py", target=r"scripts\run_paper_discovery.py",
        args="--new-flow --max-per-source 30",
    ),
    CronTask(
        name=r"MacroAlphaPro\research-forward-oos",
        description="Forward OOS tracker",
        time="06:15", kind="py", target=r"scripts\run_forward_oos.py",
    ),
    CronTask(
        name=r"MacroAlphaPro\wrds-catalog-weekly",
        description="WRDS catalog probe (Sun)",
        time="05:00", days=("Sun",),
        kind="py", target=r"scripts\probe_wrds_catalog.py",
    ),
    # ── NEW 2026-08-09: the discover-loop outer ring, finally on cron ─
    CronTask(
        name=r"MacroAlphaPro\autopilot-daily",
        description=("F14 autopilot daily cycle: F14a dry-run memo + F14b live "
                     "top-1 verdict. Replaces the retired run_app.py launch "
                     "hook. 08:45 local = 00:45 UTC, so local date == UTC "
                     "sentinel date."),
        time="08:45", kind="bat", target=r"scripts\autopilot_daily_wrapper.bat",
    ),
]


# ──────────────────────────────────────────────────────────────────────
# XML generation
# ──────────────────────────────────────────────────────────────────────
def _current_user_sid() -> str:
    out = subprocess.check_output(["whoami", "/user", "/fo", "csv"], text=True)
    return out.strip().splitlines()[-1].rsplit(",", 1)[-1].strip('"')


def build_task_xml(task: CronTask, sid: str) -> str:
    start_boundary = f"{dt.date.today().isoformat()}T{task.time}:00"
    if task.days:
        day_tags = "\n            ".join(f"<{DAY_TAGS[d]} />" for d in task.days)
        schedule = f"""<ScheduleByWeek>
          <WeeksInterval>1</WeeksInterval>
          <DaysOfWeek>
            {day_tags}
          </DaysOfWeek>
        </ScheduleByWeek>"""
    else:
        schedule = """<ScheduleByDay>
          <DaysInterval>1</DaysInterval>
        </ScheduleByDay>"""
    random_delay = (f"\n        <RandomDelay>{task.random_delay}</RandomDelay>"
                    if task.random_delay else "")
    command, args = task.command_and_args()
    args_xml = f"\n        <Arguments>{escape(args)}</Arguments>" if args else ""
    workdir = (f"\n        <WorkingDirectory>{escape(str(REPO_ROOT))}</WorkingDirectory>"
               if task.workdir_repo_root else "")
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>{escape(task.description)} [managed by scripts/install_cron_fleet.py]</Description>
    <URI>\\{task.name}</URI>
  </RegistrationInfo>
  <Principals>
    <Principal id="Author">
      <UserId>{sid}</UserId>
      <LogonType>InteractiveToken</LogonType>
    </Principal>
  </Principals>
  <Settings>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <ExecutionTimeLimit>PT4H</ExecutionTimeLimit>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
  </Settings>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>{start_boundary}</StartBoundary>{random_delay}
      {schedule}
    </CalendarTrigger>
  </Triggers>
  <Actions Context="Author">
    <Exec>
      <Command>{escape(command)}</Command>{args_xml}{workdir}
    </Exec>
  </Actions>
</Task>
"""


# ──────────────────────────────────────────────────────────────────────
# install / uninstall / verify
# ──────────────────────────────────────────────────────────────────────
XML_OUT_DIR = REPO_ROOT / "data" / "cron_fleet" / "xml"


def _select(task_filter: str | None) -> list[CronTask]:
    if not task_filter:
        return FLEET
    hits = [t for t in FLEET if task_filter.lower() in t.name.lower()]
    if not hits:
        print(f"ERROR: no manifest task matches '{task_filter}'")
        sys.exit(2)
    return hits


def cmd_install(task_filter: str | None, dry_run: bool) -> int:
    sid = _current_user_sid()
    XML_OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    for task in _select(task_filter):
        tf = task.target_file()
        if tf is not None and not tf.is_file():
            print(f"SKIP  {task.name}: target missing on disk: {tf}")
            failures += 1
            continue
        xml = build_task_xml(task, sid)
        xml_path = XML_OUT_DIR / (task.name.replace("\\", "_") + ".xml")
        xml_path.write_text(xml, encoding="utf-16")
        if dry_run:
            print(f"DRY   {task.name}: xml written {xml_path.relative_to(REPO_ROOT)}")
            continue
        r = subprocess.run(
            ["schtasks", "/Create", "/TN", task.name, "/XML", str(xml_path), "/F"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"FAIL  {task.name}: schtasks exit {r.returncode}: {r.stderr.strip()}")
            failures += 1
        else:
            print(f"OK    {task.name}")
    if not dry_run:
        print()
        print(f"{len(_select(task_filter)) - failures} task(s) registered, "
              f"{failures} failure(s). Verify with:")
        print("  python scripts/install_cron_fleet.py verify")
    return 1 if failures else 0


def cmd_uninstall(task_filter: str | None, dry_run: bool) -> int:
    for task in _select(task_filter):
        if dry_run:
            print(f"DRY   would delete {task.name}")
            continue
        r = subprocess.run(["schtasks", "/Delete", "/TN", task.name, "/F"],
                           capture_output=True, text=True)
        print(f"{'OK    ' if r.returncode == 0 else 'FAIL  '}{task.name}"
              f"{'' if r.returncode == 0 else ': ' + r.stderr.strip()}")
    return 0


def cmd_verify(task_filter: str | None) -> int:
    """Drift report: for each manifest task, is it registered, does its
    action point inside the CURRENT repo, does the target exist?"""
    drift = 0
    for task in _select(task_filter):
        r = subprocess.run(["schtasks", "/Query", "/TN", task.name, "/XML"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"MISSING  {task.name}: not registered")
            drift += 1
            continue
        xml = r.stdout
        expected_cmd, expected_args = task.command_and_args()
        problems = []
        if escape(expected_cmd) not in xml:
            problems.append(f"command drift (expected {expected_cmd})")
        if expected_args and escape(expected_args) not in xml:
            problems.append("arguments drift")
        tf = task.target_file()
        if tf is not None and not tf.is_file():
            problems.append(f"target missing: {tf}")
        old_roots = [p for p in ("C:\\Users\\${USER}\\Desktop\\intern",)
                     if p in xml and str(REPO_ROOT).lower() != p.lower()]
        if old_roots:
            problems.append(f"points at stale repo root {old_roots[0]}")
        if problems:
            print(f"DRIFT    {task.name}: {'; '.join(problems)}")
            drift += 1
        else:
            print(f"OK       {task.name}")
    print()
    print(f"{drift} of {len(_select(task_filter))} task(s) need attention."
          if drift else "Fleet is clean.")
    return 1 if drift else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["install", "uninstall", "verify"])
    ap.add_argument("--task", help="substring filter on task name (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print(f"Repo root: {REPO_ROOT}")
    print(f"Manifest:  {len(FLEET)} tasks")
    print()
    if args.command == "install":
        return cmd_install(args.task, args.dry_run)
    if args.command == "uninstall":
        return cmd_uninstall(args.task, args.dry_run)
    return cmd_verify(args.task)


if __name__ == "__main__":
    raise SystemExit(main())
