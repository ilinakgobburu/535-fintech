"""
The NAV and Reg T views of the wheel's ledger (lib/wheel.py monthly_nav, reg_t),
on test_wheel.py's market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_app.lib import wheel as W
from trading_app.lib.pmcc_daily import performance
from test_wheel import END, START, make_market


def run(**kw):
    m = make_market(path={"2019-02-14": 45.0, "2019-03-14": 55.0}, rate=0.04)
    return m, W.run(m, start=START, end=END, **kw)


def test_the_monthly_table_gives_the_registered_returns_and_sharpe_ratio():
    m, res = run()
    t = W.monthly_nav(res["ledger"], res["start_cash"], m["tbill"])
    perf = performance(res["ledger"], res["start_cash"], m["tbill"])
    assert [r["month"] for r in t] == list(perf["monthly_returns"])
    assert np.allclose([r["ret"] for r in t], list(perf["monthly_returns"].values()))
    ex = np.array([r["excess"] for r in t])
    assert ex.mean() / ex.std(ddof=1) * np.sqrt(12) == pytest.approx(perf["sharpe"])
    assert t[-1]["nav"] == pytest.approx(res["ledger"]["nav"].iloc[-1])
    assert t[-1]["cum_ret"] == pytest.approx(perf["return_pct"] / 100)


def test_month_end_drawdown_is_from_the_highest_close_so_far():
    m, res = run()
    t = W.monthly_nav(res["ledger"], res["start_cash"], m["tbill"])
    nav = res["ledger"].set_index("date")["nav"]
    for r in t:
        upto = nav[nav.index.to_period("M") <= pd.Period(r["month"], "M")]
        peak = max(res["start_cash"], upto.max())
        assert r["drawdown"] == pytest.approx(upto.iloc[-1] / peak - 1)


def test_an_open_put_sets_aside_its_strike_in_cash_and_free_cash_never_goes_below_zero():
    m, res = run()
    R = W.reg_t(res["ledger"]).set_index("date")
    L = res["ledger"].set_index("date")
    puts = L["short"] == "P"
    assert np.allclose(R.loc[puts, "put_collateral"], L.loc[puts, "short_strike"] * L.loc[puts, "short_units"])
    assert (R.loc[~puts, "put_collateral"] == 0).all()
    assert np.allclose(R["free_cash"], L["cash"] - R["put_collateral"])
    assert (R["free_cash"] >= -1e-9).all()


def test_a_call_is_covered_on_every_day_it_is_open():
    m, res = run()
    R = W.reg_t(res["ledger"]).set_index("date")
    L = res["ledger"].set_index("date")
    calls = L["short"] == "C"
    assert calls.any()
    assert (R.loc[calls, "call_shares_needed"] == L.loc[calls, "short_units"]).all()
    assert (R.loc[~calls, "call_shares_needed"] == 0).all()
    assert R["covered"].all()


def test_an_uncovered_call_is_flagged():
    m, res = run()
    led = res["ledger"].copy()
    i = led.index[led["short"] == "C"][0]
    led.loc[i, "shares"] = 0
    assert not W.reg_t(led).loc[i, "covered"]
