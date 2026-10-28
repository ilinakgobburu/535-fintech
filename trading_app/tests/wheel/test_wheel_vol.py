"""
The volatility views of the wheel (lib/wheel.py smile, position_greeks) on
test_wheel.py's market, where every option is priced from one volatility, so
the smile must come back flat at that volatility and the greeks are
Black-Scholes at it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_app.lib import wheel as W
from trading_app.lib.vol import bs_greeks
from test_wheel import END, START, VOL, make_market


def test_the_smile_is_out_of_the_money_puts_below_spot_and_calls_above():
    m = make_market()
    day = pd.Timestamp("2019-01-22")
    sm = W.smile(m, day)
    assert sm is not None and len(sm["points"]) >= 6
    p = sm["points"]
    assert (p.loc[p["cp"] == "P", "strike"] < 50.0).all() and (p.loc[p["cp"] == "C", "strike"] >= 50.0).all()
    assert np.allclose(p["moneyness"], p["strike"] / 50.0)


def test_on_a_market_priced_at_one_volatility_the_smile_is_flat_at_it():
    sm = W.smile(make_market(), pd.Timestamp("2019-01-22"))
    # quotes are rounded to 4 decimals, which moves a far out-of-the-money vol by about 2e-5
    assert np.allclose(sm["points"]["iv"], VOL, atol=1e-4)


def test_the_smile_uses_the_monthly_closest_to_thirty_days_out():
    m = make_market()
    sm = W.smile(m, pd.Timestamp("2019-01-22"))           # Feb 15 is 24 days out, Mar 15 is 52
    assert sm["expiry"] == pd.Timestamp("2019-02-15") and sm["dte"] == 24
    sm = W.smile(m, pd.Timestamp("2019-02-04"))           # Feb 15 is 11 days out, Mar 15 is 39
    assert sm["expiry"] == pd.Timestamp("2019-03-15")


def test_a_short_put_is_long_delta_short_vega_and_collects_theta():
    m = make_market()
    res = W.run(m, start=START, end=END)
    G = W.position_greeks(res, m).set_index("date")
    L = res["ledger"].set_index("date")
    day = pd.Timestamp("2019-01-25")
    row = L.loc[day]
    assert row["short"] == "P" and row["shares"] == 0
    T = (pd.Timestamp(row["short_expiry"]) - day).days / 365.0
    g = bs_greeks(50.0, row["short_strike"], T, VOL, 1.0, "P")
    n = row["short_units"]
    assert G.at[day, "delta"] == pytest.approx(-n * g["delta"], rel=1e-4)
    assert G.at[day, "vega"] == pytest.approx(-n * g["vega"] / 100, rel=1e-4)       # dollars per vol point
    assert G.at[day, "theta"] == pytest.approx(-n * g["theta"] / 365, rel=1e-4)     # dollars per day; bs theta is the holder's
    assert G.at[day, "delta"] > 0 and G.at[day, "vega"] < 0 and G.at[day, "theta"] > 0


def test_shares_alone_carry_one_delta_each_and_no_vega():
    m = make_market(path={"2019-02-14": 30.0})              # assigned, then no call qualifies under the floor
    res = W.run(m, start=START, end=END)
    G = W.position_greeks(res, m).set_index("date")
    L = res["ledger"].set_index("date")
    bare = L[(L["shares"] > 0) & L["short"].isna()].index
    assert len(bare)
    assert (G.loc[bare, "delta"] == L.loc[bare, "shares"]).all() and (G.loc[bare, "vega"] == 0).all()
