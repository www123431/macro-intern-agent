"""tests/test_month_end_trading_day.py — month-end rebalance trigger.

Pins engine.daily_batch._is_month_end_trading_day, the sole gate on
`_patrol_rebalance`. It regressed on 2026-05-30 (when
pandas_market_calendars was installed) and silently disabled every
month-end rebalance for two months: the mcal branch built its schedule
starting at `d`, so `sched.index[0] == d` for any trading day and the
`> d` comparison was False by construction. Watchdog mode 11
(rule_rebalance_frequency_audit) caught the symptom as SEVERE, which
latched the circuit breaker and halted the daily chain.

Both branches are covered — the mcal path AND the except-fallback — because
the scheduled tasks run three different interpreters (D:\\python,
py -3.11, Python310) and mcal is not installed in all of them, so which
branch executes in production depends on which task invoked the batch.
"""
from __future__ import annotations

import builtins
import datetime

import pytest

from engine.daily_batch import _is_month_end_trading_day

# Last NYSE trading day of each month vs. an ordinary trading day in the
# same month. 2026-05-29 is the Friday before Memorial Day (05-30/31 are a
# weekend, 06-01 is the next trading day), which is exactly the kind of
# non-last-calendar-day month end the naive weekday fallback must also get
# right.
MONTH_END_TRADING_DAYS = [
    datetime.date(2026, 5, 29),   # Fri; 05-30/31 weekend
    datetime.date(2026, 6, 30),   # Tue
    datetime.date(2026, 7, 31),   # Fri
    datetime.date(2026, 8, 31),   # Mon
    datetime.date(2026, 9, 30),   # Wed
]

NON_MONTH_END_TRADING_DAYS = [
    datetime.date(2026, 7, 1),
    datetime.date(2026, 7, 15),
    datetime.date(2026, 7, 30),   # the day BEFORE month end — the off-by-one trap
    datetime.date(2026, 9, 29),
]


@pytest.mark.parametrize("d", MONTH_END_TRADING_DAYS)
def test_month_end_trading_day_is_detected(d):
    assert _is_month_end_trading_day(d) is True, f"{d} is the last trading day of its month"


@pytest.mark.parametrize("d", NON_MONTH_END_TRADING_DAYS)
def test_ordinary_trading_day_is_not_month_end(d):
    assert _is_month_end_trading_day(d) is False, f"{d} is not the last trading day of its month"


def test_regression_never_returns_false_for_every_trading_day():
    """The exact shape of the 2026-05-30 regression.

    The bug did not make one date wrong — it made the mcal branch return
    False for ALL trading days, so `_patrol_rebalance` never fired. A test
    pinning a single date could pass while the gate stays dead, so assert
    that at least one month end is detected across a multi-month span.
    """
    assert any(_is_month_end_trading_day(d) for d in MONTH_END_TRADING_DAYS)


@pytest.fixture
def without_mcal(monkeypatch):
    """Force the except-branch by making the mcal import fail."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas_market_calendars":
            raise ImportError("forced for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    return None


def test_fallback_branch_is_actually_exercised(without_mcal):
    with pytest.raises(ImportError):
        import pandas_market_calendars  # noqa: F401


@pytest.mark.parametrize("d", MONTH_END_TRADING_DAYS)
def test_fallback_detects_month_end(without_mcal, d):
    assert _is_month_end_trading_day(d) is True


@pytest.mark.parametrize("d", NON_MONTH_END_TRADING_DAYS)
def test_fallback_rejects_ordinary_days(without_mcal, d):
    assert _is_month_end_trading_day(d) is False


def test_both_branches_agree_across_a_year():
    """mcal and fallback may legitimately differ on exchange holidays, but
    they must agree on the overwhelming majority of days — a wholesale
    disagreement means one branch is inverted again."""
    real_import = builtins.__import__
    day = datetime.date(2026, 1, 1)
    disagreements = []
    while day < datetime.date(2027, 1, 1):
        with_mcal = _is_month_end_trading_day(day)

        def fake_import(name, *args, **kwargs):
            if name == "pandas_market_calendars":
                raise ImportError("forced")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = fake_import
        try:
            without = _is_month_end_trading_day(day)
        finally:
            builtins.__import__ = real_import

        if with_mcal != without:
            disagreements.append(day)
        day += datetime.timedelta(days=1)

    # A handful of holiday-driven differences is expected; dozens is a bug.
    assert len(disagreements) <= 6, f"branches disagree on {len(disagreements)} days: {disagreements}"
