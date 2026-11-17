"""
The far wings' calendar is vol-arb's: the first session after each monthly expiry, held to the next monthly expiry.
From today (2026-11-17) the next entry is Monday 2026-11-23 (the Nov 20 expiry's next session), trading the Dec 18
expiry; every other night up to the next expiry is no entry, and its plan with an empty book is []. A run's dates: D
the last NYSE session closed, its orders for the session after D whatever the clock says (run_dates), and whether that
session has opened (session_open), across delayed runs, weekends, holidays and early closes.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pytest

import fw_live_synth as S
from trading_app.lib import far_wings_live as FW
from trading_app.lib import vol_arb_live as VL
from trading_app.lib import wheel_live as L


def test_the_next_entry_is_monday_the_23rd():
    assert FW.next_entry(dt.date(2026, 11, 17)) == (dt.date(2026, 11, 23), dt.date(2026, 12, 18))
    assert FW.next_entry(dt.date(2026, 11, 23)) == (dt.date(2026, 11, 23), dt.date(2026, 12, 18))
    assert FW.next_entry(dt.date(2026, 11, 24)) == (dt.date(2026, 12, 21), dt.date(2027, 1, 15))
    assert FW.night(dt.date(2026, 11, 23)) == {"entry": True, "expiry": dt.date(2026, 12, 18), "forced": False,
                                               "late": False}
    assert FW.next_session(dt.date(2026, 11, 23)) == dt.date(2026, 11, 24)


def test_every_other_night_to_the_expiry_is_no_entry_and_writes_nothing():
    days = [d.date() for d in L.sessions("2026-11-17", "2026-12-18")]
    entries = [d for d in days if FW.night(d)["entry"]]
    assert entries == [dt.date(2026, 11, 23)]
    cfg = S.config()
    for d in days:
        if d in entries:
            continue
        assert FW.entry_expiry(d) is None
        got = FW.plan_night(cfg, d, FW.next_session(d), FW.Book(), name=S.NVDA_NAME)
        assert got["orders"] == [] and got["entry"] is None


def test_the_calendar_is_vol_arbs_over_three_years():
    idx = L.sessions("2026-11-17", "2029-12-31")
    want = {e.date(): x.date() for e, x in VL.entry_sessions(idx, "2026-11-17", "2029-12-31").items()}
    got = {d.date(): FW.entry_expiry(d) for d in idx if FW.entry_expiry(d) is not None}
    got = {e: x for e, x in got.items() if x <= dt.date(2029, 12, 31)}      # the window's last cycle ends in 2029
    assert got == want and len(got) >= 36
    for e, x in got.items():
        assert x.weekday() == 4 and 15 <= x.day <= 21 or FW.is_session(x)     # the third Friday, or a Thursday holiday
        assert e == FW.next_session(FW.previous_expiry(x))


def test_force_entry_trades_the_next_expiry():
    forced = FW.night(dt.date(2026, 11, 16), force_entry=True)
    assert forced == {"entry": True, "expiry": dt.date(2026, 12, 18), "forced": True, "late": False}


# --------------------------------------------------------------------------- the run's dates
# order.txt: 'for' is "The session the order belongs to. Not 'next'. A late run must not fill Monday's order on
# Tuesday". On 2026-11-17 GitHub started the vol-arb's 03:00 UTC run at 11:23 UTC and its 06:00 retry at 15:29 UTC; the
# retry wrote "Quotes from the close of 2026-11-16, orders for 2026-11-18" (wheel_live.decision_dates takes 'for' from
# the clock). The far wings' 'for' is the session after D, always, and a run once it has opened is refused.
NY = ZoneInfo("America/New_York")


def utc(y, m, d, hh, mm):
    """A UTC clock time as far_wings_orders.now_ny() reads it: New York time, naive."""
    return dt.datetime(y, m, d, hh, mm, tzinfo=dt.timezone.utc).astimezone(NY).replace(tzinfo=None)


MON, TUE, WED = dt.date(2026, 11, 16), dt.date(2026, 11, 17), dt.date(2026, 11, 18)


@pytest.mark.parametrize("now, open_", [
    (dt.datetime(2026, 11, 16, 22, 0), False),         # the 22:00 run
    (dt.datetime(2026, 11, 17, 1, 0), False),          # the 01:00 retry
    (utc(2026, 11, 17, 11, 23), False),                # the 22:00 run as GitHub started it: 06:23, before the open
    (dt.datetime(2026, 11, 17, 9, 29, 59), False),
    (dt.datetime(2026, 11, 17, 9, 30), True),          # the open itself: too late
    (utc(2026, 11, 17, 15, 29), True),                 # the retry as GitHub started it: 10:29, during Tuesday
    (dt.datetime(2026, 11, 17, 15, 59, 59), True),     # Tuesday's close not yet in: still Monday's D
])
def test_d_s_orders_are_for_the_next_session_whatever_the_clock(now, open_):
    assert FW.run_dates(now) == (MON, TUE)
    assert FW.session_open(TUE, now) is open_
    # where the clock-based dates the vol-arb used would have labelled Monday's decisions for Wednesday
    assert L.decision_dates(now)[1] == (WED if open_ else TUE)


def test_the_live_actions_clock_times_are_the_new_york_times_the_tests_use():
    assert utc(2026, 11, 17, 3, 0) == dt.datetime(2026, 11, 16, 22, 0)      # winter: UTC-5
    assert utc(2026, 11, 17, 11, 23) == dt.datetime(2026, 11, 17, 6, 23)
    assert utc(2026, 11, 17, 15, 29) == dt.datetime(2026, 11, 17, 10, 29)
    assert utc(2026, 7, 15, 3, 0) == dt.datetime(2026, 7, 14, 23, 0)        # summer: UTC-4


@pytest.mark.parametrize("now, d, for_day", [
    (dt.datetime(2026, 11, 17, 16, 0), TUE, WED),                                       # D's close at 16:00 sharp
    (dt.datetime(2026, 11, 20, 22, 0), dt.date(2026, 11, 20), dt.date(2026, 11, 23)),   # Friday night: Monday
    (dt.datetime(2026, 11, 21, 12, 0), dt.date(2026, 11, 20), dt.date(2026, 11, 23)),   # Saturday
    (dt.datetime(2026, 11, 22, 23, 0), dt.date(2026, 11, 20), dt.date(2026, 11, 23)),   # Sunday night
    (dt.datetime(2026, 11, 23, 9, 0), dt.date(2026, 11, 20), dt.date(2026, 11, 23)),    # Monday before the open
    (dt.datetime(2027, 1, 15, 22, 0), dt.date(2027, 1, 15), dt.date(2027, 1, 19)),      # Friday before MLK: Tuesday
    (dt.datetime(2027, 1, 18, 12, 0), dt.date(2027, 1, 15), dt.date(2027, 1, 19)),      # on the Monday holiday
    (dt.datetime(2027, 1, 19, 8, 0), dt.date(2027, 1, 15), dt.date(2027, 1, 19)),       # Tuesday before the open
    (dt.datetime(2026, 11, 25, 22, 0), dt.date(2026, 11, 25), dt.date(2026, 11, 27)),   # before Thanksgiving
    (dt.datetime(2026, 11, 27, 12, 59), dt.date(2026, 11, 25), dt.date(2026, 11, 27)),  # the early close not yet in
    (dt.datetime(2026, 11, 27, 13, 0), dt.date(2026, 11, 27), dt.date(2026, 11, 30)),   # 13:00: the early close
    (dt.datetime(2026, 12, 24, 14, 0), dt.date(2026, 12, 24), dt.date(2026, 12, 28)),   # Christmas Eve, then Friday off
    (dt.datetime(2027, 1, 1, 12, 0), dt.date(2026, 12, 31), dt.date(2027, 1, 4)),       # New Year's Day
])
def test_weekends_holidays_and_early_closes(now, d, for_day):
    assert FW.run_dates(now) == (d, for_day)
    assert FW.is_session(d) and FW.is_session(for_day) and FW.next_session(d) == for_day


def test_early_closes_are_wheel_lives():
    assert L.close_time(dt.date(2026, 11, 27)) == dt.time(13, 0) and L.close_time(dt.date(2026, 12, 24)) == \
        dt.time(13, 0) and L.close_time(dt.date(2026, 11, 25)) == dt.time(16, 0)
    assert not FW.is_session(dt.date(2026, 11, 26)) and not FW.is_session(dt.date(2027, 1, 18))


def test_every_half_hour_for_three_months_for_is_the_session_after_d():
    now = dt.datetime(2026, 11, 1, 0, 0)
    while now < dt.datetime(2027, 2, 1):
        d, for_day = FW.run_dates(now)
        closes = dt.datetime.combine(d, L.close_time(d))
        assert closes <= now and FW.is_session(d), now
        # no session closes between D's close and now: D is the last close
        assert now < dt.datetime.combine(for_day, L.close_time(for_day)), now
        assert for_day == FW.next_session(d) and not FW.night(d, for_day=for_day)["late"], now
        # the clock-based 'for' agrees exactly when the session after D has not opened
        assert (L.decision_dates(now) == (d, for_day)) == (not FW.session_open(for_day, now)), now
        now += dt.timedelta(minutes=30)


def test_the_retry_s_already_written_check_keys_on_d_alone():
    led = FW.record_night(FW.read_ledger(None), "NVDA", [{"id": "nvda-fw-put-2026-11-23"}], dt.date(2026, 11, 23),
                          dt.date(2026, 11, 24))
    mine = FW.name_ledger(led, "NVDA")
    assert FW.decided_from(mine, dt.date(2026, 11, 23)) == "nvda-fw-put-2026-11-23"
    assert FW.decided_from(mine, "2026-11-23") == "nvda-fw-put-2026-11-23"
    assert FW.decided_from(mine, dt.date(2026, 11, 24)) is None
    assert FW.decided_from(FW.name_ledger(led, "AAPL"), dt.date(2026, 11, 23)) is None
    none = FW.record_night(led, "AAPL", [], dt.date(2026, 11, 23), dt.date(2026, 11, 24))
    assert FW.decided_from(FW.name_ledger(none, "AAPL"), dt.date(2026, 11, 23)) == "no order"
    # a night recorded with another 'for' is still that night: the check does not compare a 'for' from the clock
    odd = {"orders": {}, "nights": {"2026-11-23": {"for": "2026-11-25", "ids": ["nvda-fw-hedge-2026-11-23"]}}}
    assert FW.decided_from(odd, dt.date(2026, 11, 23)) == "nvda-fw-hedge-2026-11-23"
