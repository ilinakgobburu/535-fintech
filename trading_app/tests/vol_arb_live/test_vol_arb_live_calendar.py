"""
The live calendar: entry nights are the registered ones (the first session after each monthly expiry), every
other night with an empty book writes nothing. The dates are known from the calendar, not from the code: each
month's third Friday (the session before it when the market is closed), and the NYSE's published holidays.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

import va_live_synth as S
from trading_app.lib import vol_arb_live as VL
from trading_app.lib import wheel_live as L

D = dt.date


def test_the_next_entry_night_after_the_november_expiry_is_monday_the_23rd():
    assert VL.next_entry(D(2026, 11, 20)) == (D(2026, 11, 23), D(2026, 12, 18))
    assert VL.next_entry(D(2026, 11, 21)) == (D(2026, 11, 23), D(2026, 12, 18))     # a Saturday
    assert VL.next_entry(D(2026, 11, 23)) == (D(2026, 11, 23), D(2026, 12, 18))     # on or after
    assert VL.next_entry(D(2026, 11, 24)) == (D(2026, 12, 21), D(2027, 1, 15))


# {entry night: its expiry}, Sept 2026 to June 2027
ENTRIES = {
    D(2026, 9, 21): D(2026, 10, 16),
    D(2026, 10, 19): D(2026, 11, 20),
    D(2026, 11, 23): D(2026, 12, 18),
    D(2026, 12, 21): D(2027, 1, 15),
    D(2027, 1, 19): D(2027, 2, 19),      # Monday Jan 18 is Martin Luther King Day
    D(2027, 2, 22): D(2027, 3, 19),
    D(2027, 3, 22): D(2027, 4, 16),
    D(2027, 4, 19): D(2027, 5, 21),
    D(2027, 5, 24): D(2027, 6, 17),      # Friday June 18 is Juneteenth (observed): Thursday's close
}


def test_entry_nights_are_the_first_session_after_each_monthly_expiry():
    for day in pd.date_range("2026-09-01", "2027-06-17"):
        assert VL.entry_expiry(day.date()) == ENTRIES.get(day.date()), day.date()


def test_a_good_friday_expiry_moves_to_thursday():
    # April 2025's third Friday (Apr 18) was Good Friday: the expiry is Thursday's close, the entry Monday.
    assert VL.next_entry(D(2025, 4, 17)) == (D(2025, 4, 21), D(2025, 5, 16))
    assert VL.previous_expiry(D(2025, 5, 16)) == D(2025, 4, 17)


def test_the_copied_calendar_gives_the_registered_entry_counts():
    # config/vol_arb_ml.yaml part_2 calendar: 59 entries in 2019-01-02..2023-12-29, 33 in 2024-01-02..2026-10-16
    for (lo, hi), n in {("2019-01-02", "2023-12-29"): 59, ("2024-01-02", "2026-10-16"): 33}.items():
        e = VL.entry_sessions(L.sessions(lo, hi), lo, hi)
        assert len(e) == n
    e = VL.entry_sessions(L.sessions("2024-01-02", "2026-10-16"), "2024-01-02", "2026-10-16")
    assert max(e.values()) == pd.Timestamp("2026-10-16")      # the test's last cycle ends on its last day


@pytest.mark.parametrize("day, expected", [
    (D(2026, 12, 18), D(2026, 11, 20)), (D(2027, 1, 15), D(2026, 12, 18)), (D(2027, 6, 17), D(2027, 5, 21)),
])
def test_previous_expiry(day, expected):
    assert VL.previous_expiry(day) == expected


@pytest.mark.parametrize("day, expected", [
    (D(2026, 11, 20), D(2026, 11, 23)), (D(2026, 11, 25), D(2026, 11, 27)),   # Thanksgiving on the 26th
    (D(2026, 12, 24), D(2026, 12, 28)), (D(2026, 11, 22), D(2026, 11, 23)),
])
def test_next_session(day, expected):
    assert VL.next_session(day) == expected


def test_sessions():
    assert VL.is_session(D(2026, 11, 23)) and not VL.is_session(D(2026, 11, 26)) and not VL.is_session(D(2026, 11, 22))


def test_tonights_run_after_the_entry_close_is_an_entry_night():
    day, for_day = L.decision_dates(dt.datetime(2026, 11, 23, 22, 0))     # the 03:00 UTC run
    assert (day, for_day) == (D(2026, 11, 23), D(2026, 11, 24))
    assert VL.night(day) == {"entry": True, "expiry": D(2026, 12, 18), "forced": False, "late": False}
    assert VL.night(day, for_day=for_day) == VL.night(day)
    late = L.decision_dates(dt.datetime(2026, 11, 24, 3, 30))             # a delayed cron, still Nov 23's close
    assert late == (D(2026, 11, 23), D(2026, 11, 24))
    retry = L.decision_dates(dt.datetime(2026, 11, 24, 1, 0))             # the 06:00 UTC retry
    assert retry == (D(2026, 11, 23), D(2026, 11, 24)) and VL.night(*retry[:1], for_day=retry[1])["entry"]


def test_a_run_after_the_next_open_places_no_entry():
    # the Run button on Tuesday 10:00: D is still Monday the 23rd, but the orders would be for Wednesday
    day, for_day = L.decision_dates(dt.datetime(2026, 11, 24, 10, 0))
    assert (day, for_day) == (D(2026, 11, 23), D(2026, 11, 25))
    assert VL.night(day, for_day=for_day) == {"entry": False, "expiry": D(2026, 12, 18), "forced": False, "late": True}
    # just before the open it is still the entry; a holiday weekend's next session counts as next
    assert VL.night(*L.decision_dates(dt.datetime(2026, 11, 24, 9, 29)))["entry"]
    assert VL.night(D(2026, 11, 23), for_day=D(2026, 11, 24))["entry"]
    # a night that is not an entry is never late
    assert VL.night(D(2026, 11, 24), for_day=D(2026, 11, 27))["late"] is False


def test_force_entry_measures_the_next_entrys_expiry():
    assert VL.night(D(2026, 11, 16)) == {"entry": False, "expiry": None, "forced": False, "late": False}
    assert VL.night(D(2026, 11, 16), force_entry=True) == {"entry": True, "expiry": D(2026, 12, 18), "forced": True,
                                                           "late": False}
    assert VL.night(D(2026, 11, 23), force_entry=True)["forced"] is False     # already an entry night


def test_non_entry_nights_with_an_empty_book_give_no_orders():
    cfg = S.config()
    day = D(2026, 11, 24)
    while day <= D(2026, 12, 18):
        if VL.is_session(day):
            assert VL.night(day)["entry"] is False
            plan = VL.plan_night(cfg, day, VL.next_session(day), VL.Book())
            assert plan["orders"] == [] and plan["notes"] == ["no order tonight"]
        day += dt.timedelta(days=1)
