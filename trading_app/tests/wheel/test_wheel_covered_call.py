"""run(mode="covered_call"): config/wheel_covered_call.yaml's benchmark, buy and hold's shares plus a monthly call."""

from __future__ import annotations

import pandas as pd
import pytest

from trading_app.lib import wheel as W

from test_wheel import END, FEB, START, cycle, make_market, nearest


def test_it_buys_the_same_whole_shares_as_buy_and_hold():
    m = make_market()
    cc = W.run(m, start=START, end=END, mode="covered_call", floor=False)
    hold = W.run(m, start=START, end=END, mode="buy_and_hold")
    assert cc["ledger"]["shares"].iloc[0] == hold["ledger"]["shares"].iloc[0] == 2000


def test_each_entry_sells_one_call_per_hundred_shares_at_the_target_delta():
    res = W.run(make_market(), start=START, end=END, mode="covered_call", call_delta=0.25, floor=False)
    first = cycle(res, "2019-01-22")
    assert first["cp"] == "C" and first["contracts"] == 20
    assert first["strike"] == nearest(50.0, "2019-01-22", FEB, "C", 0.25)


def test_it_never_sells_a_put():
    res = W.run(make_market(path={"2019-02-14": 45.0, "2019-03-14": 55.0}), start=START, end=END,
                mode="covered_call", floor=False)
    assert all(c["cp"] == "C" for c in res["cycles"] if c.get("cp"))
    assert not any(e["kind"] == "put" for e in res["blotter"])


def test_after_being_called_away_it_buys_the_shares_back_at_the_next_entry():
    res = W.run(make_market(path={"2019-02-14": 55.0}), start=START, end=END, mode="covered_call",
                call_delta=0.25, floor=False)
    assert cycle(res, "2019-01-22")["status"] == "called away"
    led = res["ledger"].set_index("date")
    assert led.at[FEB, "shares"] == 0
    rebuy = [e for e in res["blotter"] if e["kind"] == "stock" and e["side"] == W.BUY]
    assert [e["date"] for e in rebuy] == [pd.Timestamp("2019-01-02"), pd.Timestamp("2019-02-19")]
    assert led.at[pd.Timestamp("2019-02-19"), "shares"] == int(led.at[FEB, "cash"] // 55.0)   # no interest at rate 0
    nxt = cycle(res, "2019-02-19")
    assert nxt["cp"] == "C" and nxt["contracts"] == led.at[pd.Timestamp("2019-02-19"), "shares"] // 100


def test_with_the_stock_flat_every_call_expires_and_the_shares_stay():
    res = W.run(make_market(), start=START, end=END, mode="covered_call", floor=False)
    calls = [c for c in res["cycles"] if c.get("cp") == "C"]
    assert calls and all(c["status"] == "expired" for c in calls)
    assert (res["ledger"]["shares"] == 2000).all()
    assert res["totals"]["premium"] == pytest.approx(sum(c["premium"] for c in calls))


def test_the_wheel_itself_still_starts_in_cash():
    res = W.run(make_market(), start=START, end=END)
    assert res["ledger"]["shares"].iloc[0] == 0
    assert cycle(res, "2019-01-22")["cp"] == "P"


def test_shares_left_over_beyond_the_last_hundred_do_not_stop_the_buy_back():
    # $101,000 at $50 buys 2,020 shares: 20 calls cover 2,000, so 20 are left after the call-away
    res = W.run(make_market(path={"2019-02-14": 55.0}), start=START, end=END, mode="covered_call",
                call_delta=0.25, floor=False, start_cash=101_000.0)
    led = res["ledger"].set_index("date")
    assert led.at[FEB, "shares"] == 20
    assert led.at[pd.Timestamp("2019-02-19"), "shares"] == 20 + int(led.at[FEB, "cash"] // 55.0)
    nxt = cycle(res, "2019-02-19")
    assert nxt["cp"] == "C" and nxt["contracts"] == led.at[pd.Timestamp("2019-02-19"), "shares"] // 100
