"""
The managed wheel (config/wheel_managed.yaml) on test_wheel.py's market, small
enough to check by hand: the early close at half the premium, rolling a
challenged put down and out for a credit, early assignment, and the trend
filter. Every expected number is read off that market's own quotes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_app.lib import wheel as W
from test_wheel import APR, END, FEB, MAR, START, cycle, make_market

ROLLS = {"roll_days": 7, "max_rolls": 2}


def quotes_of(m, ric):
    return m["quotes"][m["quotes"]["ric"] == ric].set_index("date").sort_index()


def events(res, **match):
    return [e for e in res["blotter"] if all(e.get(k) == v for k, v in match.items())]


# --------------------------------------------------------------------------
# nothing changes unless asked
# --------------------------------------------------------------------------

def test_no_management_leaves_the_wheel_exactly_as_registered():
    m = make_market(path={"2019-02-14": 45.0, "2019-03-14": 55.0})
    plain = W.run(m, start=START, end=END)
    for manage in (None, {}):
        again = W.run(m, start=START, end=END, manage=manage)
        assert np.allclose(again["ledger"]["nav"], plain["ledger"]["nav"])
        assert ([(e["date"], e["side"], e["qty"], e["price"]) for e in again["blotter"]]
                == [(e["date"], e["side"], e["qty"], e["price"]) for e in plain["blotter"]])


# --------------------------------------------------------------------------
# the early close
# --------------------------------------------------------------------------

def test_a_put_is_bought_back_on_the_first_close_at_half_its_premium():
    m = make_market()
    res = W.run(m, start=START, end=END, manage={"take_profit": 0.5})
    sold = cycle(res, "2019-01-22")
    q = quotes_of(m, events(res, side="SELL")[0]["instrument"])
    due = q[(q.index > pd.Timestamp("2019-01-22")) & (q["mid"] <= 0.5 * sold["fill"] + 1e-12)]
    back = events(res, side="BUY", reason="take_profit")[0]
    assert back["date"] == due.index[0] and back["price"] == pytest.approx(due["mid"].iloc[0])
    assert sold["status"] == "closed early"
    # nothing new is sold until the next entry session, the one after the February expiry
    sells = [e["date"] for e in events(res, side="SELL")]
    assert not any(back["date"] < d < pd.Timestamp("2019-02-19") for d in sells)
    assert pd.Timestamp("2019-02-19") in sells
    assert res["ledger"].set_index("date").at[back["date"], "option_mv"] == 0.0


def test_in_the_stress_test_the_close_is_triggered_and_paid_at_the_ask():
    m = make_market()
    res = W.run(m, start=START, end=END, fill="cross", manage={"take_profit": 0.5})
    sold = cycle(res, "2019-01-22")
    q = quotes_of(m, events(res, side="SELL")[0]["instrument"])
    due = q[(q.index > pd.Timestamp("2019-01-22")) & (q["ask"] <= 0.5 * sold["fill"] + 1e-12)]
    back = events(res, side="BUY", reason="take_profit")[0]
    assert back["date"] == due.index[0] and back["price"] == pytest.approx(due["ask"].iloc[0])


# --------------------------------------------------------------------------
# rolling a challenged put
# --------------------------------------------------------------------------

def test_a_put_in_the_money_a_week_out_is_rolled_down_and_out_for_a_credit():
    m = make_market(path={"2019-02-04": 46.0})
    res = W.run(m, start=START, end=END, manage=ROLLS)
    first = cycle(res, "2019-01-22")
    roll_day = pd.Timestamp("2019-02-08")       # the last session at least 7 days before Feb 15
    back, new = events(res, side="BUY", reason="roll")[0], events(res, side="SELL", reason="roll")[0]
    assert first["status"] == "rolled" and back["date"] == new["date"] == roll_day
    assert pd.Timestamp(new["expiry"]) == MAR and new["price"] >= back["price"]
    q = m["quotes"]
    chain = q[(q["date"] == roll_day) & (q["expiry"] == MAR) & (q["cp"] == "P") & q["mid"].notna()]
    covers = chain[(chain["strike"] <= first["strike"]) & (chain["mid"] >= back["price"])]
    assert new["strike"] == covers["strike"].max()
    assert not [e for e in events(res, side="ASSIGN") if e["date"] == FEB]


def test_after_two_rolls_in_a_row_the_put_is_held_and_assigned_with_every_credit_in_its_basis():
    m = make_market(path={"2019-02-04": 46.0})
    res = W.run(m, start=START, end=END, manage=ROLLS)
    assert [e["date"] for e in events(res, side="SELL", reason="roll")] == [pd.Timestamp("2019-02-08"),
                                                                            pd.Timestamp("2019-03-08")]
    assigned = events(res, side="ASSIGN")[0]
    assert assigned["date"] == APR
    puts = [e for e in res["blotter"] if e.get("cp") == "P" and e["side"] in ("SELL", "BUY")]
    credit = sum(e["price"] if e["side"] == "SELL" else -e["price"] for e in puts)
    assert res["ledger"].set_index("date").at[APR, "basis"] == pytest.approx(assigned["price"] - credit)
    s = W.summary(res, m["tbill"])
    assert s["rolled"] == 2 and s["assigned"] == 1 and s["closed_early"] == 0


def test_with_no_strike_giving_a_credit_the_put_is_held_to_expiry():
    m = make_market(path={"2019-02-04": 46.0})
    q = m["quotes"]
    m["quotes"] = q[~((q["date"] == pd.Timestamp("2019-02-08")) & (q["expiry"] == MAR))]
    res = W.run(m, start=START, end=END, manage=ROLLS)
    assert not events(res, reason="roll") and cycle(res, "2019-01-22")["status"] == "assigned"


# --------------------------------------------------------------------------
# early assignment
# --------------------------------------------------------------------------

def test_a_put_with_no_time_value_left_is_assigned_early_at_the_strike():
    m = make_market(path={"2019-02-04": 38.0})
    res = W.run(m, start=START, end=END, manage={"early_assignment": True})
    first = cycle(res, "2019-01-22")
    q = quotes_of(m, events(res, side="SELL")[0]["instrument"])
    close = m["stock"]["close"].reindex(q.index)
    time_value = q["mid"] - (first["strike"] - close).clip(lower=0)
    due = q[(q.index > pd.Timestamp("2019-01-22")) & (close < first["strike"]) & (time_value <= 0.05 + 1e-12)]
    a = events(res, side="ASSIGN")[0]
    assert a["date"] == due.index[0] < FEB and a["reason"] == "early"
    assert first["status"] == "assigned early"
    assert res["ledger"].set_index("date").at[a["date"], "shares"] == first["units"]


def test_a_call_deep_in_the_money_is_assigned_the_session_before_an_ex_dividend_date():
    m = make_market(path={"2019-02-14": 45.0, "2019-03-01": 60.0}, dividends={"2019-03-06": 1.0})
    res = W.run(m, start=START, end=END, manage={"early_assignment": True})
    call = cycle(res, "2019-02-19")
    assert call["cp"] == "C" and call["status"] == "called away early"
    a = [e for e in events(res, side="ASSIGN") if e["kind"] == "call"][0]
    assert a["date"] == pd.Timestamp("2019-03-05") and a["reason"] == "early"
    assert not events(res, side="DIVIDEND")
    assert res["ledger"].set_index("date").at[pd.Timestamp("2019-03-05"), "shares"] == 0


# --------------------------------------------------------------------------
# the trend filter
# --------------------------------------------------------------------------

def test_no_put_is_sold_below_the_200_close_average():
    m = make_market()
    before = pd.bdate_range(end=m["stock"].index[0] - pd.Timedelta(days=1), periods=250)
    m["history"] = pd.Series(60.0, index=before)          # the average of the last 200 closes is above 50
    res = W.run(m, start=START, end=END, manage={"trend_sma": 200})
    assert cycle(res, "2019-01-22")["status"] == "skipped: below the 200-close average"
    assert not events(res, side="SELL")
    m["history"] = pd.Series(40.0, index=before)          # now below 50
    res = W.run(m, start=START, end=END, manage={"trend_sma": 200})
    assert cycle(res, "2019-01-22")["cp"] == "P"


def test_the_trend_filter_refuses_to_run_without_200_closes():
    with pytest.raises(ValueError, match="200 closes"):
        W.run(make_market(), start=START, end=END, manage={"trend_sma": 200})


# --------------------------------------------------------------------------
# the books still balance
# --------------------------------------------------------------------------

def test_with_buy_backs_and_rolls_the_pnl_parts_still_sum_to_the_pnl():
    m = make_market(path={"2019-02-04": 46.0, "2019-05-01": 52.0}, dividends={"2019-05-08": 0.5})
    res = W.run(m, start=START, end=END,
                manage={"take_profit": 0.5, "early_assignment": True, **ROLLS})
    a = W.attribution(res)
    assert a["premium"] + a["interest"] + a["dividends"] + a["shares"] + a["open_short"] == pytest.approx(a["pnl"])
    L = res["ledger"]
    rebuilt = L["cash"] + L["shares"] * L["stock_close"] - L["short_units"].fillna(0) * L["short_mark"].fillna(0)
    assert np.allclose(rebuilt, L["nav"])


def test_an_option_nobody_bids_for_can_still_be_bought_back_at_its_ask():
    # LSEG returns an empty bid as <NA>; the stress test buys at the ask regardless
    m = make_market()
    q = m["quotes"].copy()
    q["bid"] = q["bid"].astype(object).where(q["bid"].notna(), pd.NA)
    m["quotes"] = q
    res = W.run(m, start=START, end=END, fill="cross", manage={"take_profit": 0.5})
    back = events(res, side="BUY", reason="take_profit")
    assert back and all(np.isfinite(e["price"]) for e in back)
