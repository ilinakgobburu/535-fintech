"""
The far wings' calendar is vol-arb's: the first session after each monthly expiry, held to the next monthly expiry.
From today (2026-11-17) the next entry is Monday 2026-11-23 (the Nov 20 expiry's next session), trading the Dec 18
expiry; every other night up to the next expiry is no entry, and its plan with an empty book is [].
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

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


def test_a_late_run_is_no_entry_and_force_entry_trades_the_next_expiry():
    assert FW.night(dt.date(2026, 11, 23), for_day=dt.date(2026, 11, 25))["late"] is True
    assert FW.night(dt.date(2026, 11, 23), for_day=dt.date(2026, 11, 25))["entry"] is False
    forced = FW.night(dt.date(2026, 11, 16), force_entry=True)
    assert forced == {"entry": True, "expiry": dt.date(2026, 12, 18), "forced": True, "late": False}


def test_decision_dates_around_the_entry():
    # the 22:00 run and the 01:00 retry decide on Monday's close for Tuesday; Tuesday 10:00 is too late for Tuesday
    assert FW.decision_dates(dt.datetime(2026, 11, 23, 22, 0)) == (dt.date(2026, 11, 23), dt.date(2026, 11, 24))
    assert FW.decision_dates(dt.datetime(2026, 11, 24, 1, 0)) == (dt.date(2026, 11, 23), dt.date(2026, 11, 24))
    assert FW.decision_dates(dt.datetime(2026, 11, 24, 10, 0)) == (dt.date(2026, 11, 23), dt.date(2026, 11, 25))
    assert pd.Timestamp(FW.decision_dates(dt.datetime(2026, 11, 27, 22, 0))[1]).date() == dt.date(2026, 11, 30)
