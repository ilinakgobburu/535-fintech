"""
config/wheel_choices.yaml's three rules on test_wheel.py's market: skip a put
when earnings fall before its expiry, sell a put only when its implied vol is
above the stock's recent realised vol, and drop the cost-basis floor once
shares have been held through three call cycles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_app.lib import wheel as W
from test_wheel import END, START, cycle, make_market


def sells(res, cp=None):
    return [e for e in res["blotter"] if e["side"] == "SELL" and e.get("cp") and (cp is None or e["cp"] == cp)]


# --------------------------------------------------------------------------
# choice 10: skip earnings
# --------------------------------------------------------------------------

def test_no_put_is_sold_when_earnings_fall_before_its_expiry():
    m = make_market()
    m["earnings"] = pd.DatetimeIndex(["2019-02-05"])            # inside the Jan 22 -> Feb 15 cycle
    res = W.run(m, start=START, end=END, manage={"skip_earnings": True})
    assert cycle(res, "2019-01-22")["status"] == "skipped: earnings before expiry"
    assert pd.Timestamp("2019-01-22") not in [e["date"] for e in sells(res)]
    assert cycle(res, "2019-02-19")["cp"] == "P"              # the next cycle has no earnings


def test_covered_calls_are_still_written_through_earnings():
    m = make_market(path={"2019-02-14": 45.0})
    m["earnings"] = pd.DatetimeIndex(["2019-03-05"])            # inside the Feb 19 -> Mar 15 call cycle
    res = W.run(m, start=START, end=END, manage={"skip_earnings": True})
    assert cycle(res, "2019-02-19")["cp"] == "C"


# --------------------------------------------------------------------------
# choice 11: implied over realised
# --------------------------------------------------------------------------

def test_a_put_is_sold_when_its_implied_vol_is_above_the_months_realised_vol():
    res = W.run(make_market(), start=START, end=END, manage={"iv_over_rv": 21})    # a flat stock: realised 0
    assert cycle(res, "2019-01-22")["cp"] == "P"


def test_no_put_is_sold_after_a_month_that_moved_more_than_the_option_prices_in():
    m = make_market()
    days = m["stock"].index
    swing = days[(days >= pd.Timestamp("2018-12-20")) & (days < pd.Timestamp("2019-01-22"))]
    m["stock"].loc[swing, "close"] = np.where(np.arange(len(swing)) % 2, 52.5, 47.5)   # about 160% realised
    res = W.run(m, start=START, end=END, manage={"iv_over_rv": 21})
    c = cycle(res, "2019-01-22")
    assert c["status"].startswith("skipped: implied vol") and "rv" in c and c["rv"] > 1.0
    assert pd.Timestamp("2019-01-22") not in [e["date"] for e in sells(res)]


# --------------------------------------------------------------------------
# choice 12: unstick stranded shares
# --------------------------------------------------------------------------

def test_after_three_call_cycles_the_floor_is_dropped():
    m = make_market(path={"2019-02-14": 40.0})                 # assigned near 47.5, then stuck at 40
    held = W.run(m, start=START, end=END)
    for entry in ("2019-02-19", "2019-03-18", "2019-04-22", "2019-05-20"):
        assert cycle(held, entry)["status"] == "held: no qualifying call"
    res = W.run(m, start=START, end=END, manage={"unstick_after": 3})
    for entry in ("2019-02-19", "2019-03-18", "2019-04-22"):
        assert cycle(res, entry)["status"] == "held: no qualifying call"
    fourth = cycle(res, "2019-05-20")
    basis = res["ledger"].set_index("date").at[pd.Timestamp("2019-05-17"), "basis"]
    assert fourth["cp"] == "C" and fourth["unstuck"] and fourth["strike"] < basis



# --------------------------------------------------------------------------
# choice 10b: close before earnings, re-sell after
# --------------------------------------------------------------------------

def test_a_put_is_bought_back_before_earnings_and_a_new_one_sold_after():
    m = make_market()
    m["earnings"] = pd.DatetimeIndex(["2019-02-05"])
    res = W.run(m, start=START, end=END, manage={"close_before_earnings": True})
    back = [e for e in res["blotter"] if e["side"] == "BUY" and e.get("reason") == "earnings"]
    again = [e for e in res["blotter"] if e["side"] == "SELL" and e.get("reason") == "after_earnings"]
    assert [e["date"] for e in back] == [pd.Timestamp("2019-02-04")]
    assert [e["date"] for e in again] == [pd.Timestamp("2019-02-06")]
    assert pd.Timestamp(again[0]["expiry"]) == pd.Timestamp("2019-02-15")
    assert cycle(res, "2019-01-22")["status"] == "closed before earnings"


def test_with_less_than_a_week_left_after_earnings_the_cash_waits_for_the_next_entry():
    m = make_market()
    m["earnings"] = pd.DatetimeIndex(["2019-02-12"])            # Feb 13 leaves 2 days to Feb 15
    res = W.run(m, start=START, end=END, manage={"close_before_earnings": True})
    assert [e["date"] for e in res["blotter"] if e.get("reason") == "earnings"] == [pd.Timestamp("2019-02-11")]
    assert not [e for e in res["blotter"] if e.get("reason") == "after_earnings"]
    assert cycle(res, "2019-02-19")["cp"] == "P"
