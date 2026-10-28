"""
The volatility views of the wheel (lib/wheel.py smile, position_greeks) on
test_wheel.py's market, where every option is priced from one volatility, so
the smile must come back flat at that volatility and the greeks are
Black-Scholes at it; and the lecture's greeks (spot_greeks) against the
Black-Scholes price and the identities the course derives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_app.lib import wheel as W
from trading_app.lib.vol import bs_greeks, bs_price
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


CASES = [(100.0, 100.0, 0.25, 0.20, 0.05), (520.0, 485.0, 30 / 365, 0.22, 0.042),
         (520.0, 560.0, 9 / 365, 0.30, 0.0), (80.0, 60.0, 1.5, 0.35, 0.03)]


def bs_spot(S, K, T, sigma, r, cp):
    return bs_price(S * np.exp(r * T), K, T, sigma, np.exp(-r * T), cp)


@pytest.mark.parametrize("S,K,T,sigma,r", CASES)
def test_the_lecture_greeks_are_the_derivatives_of_the_black_scholes_price(S, K, T, sigma, r):
    g = W.spot_greeks(S, K, T, sigma, r)
    h, dv, dt = S * 1e-4, 1e-5, 1e-6
    for cp, side in (("C", "call"), ("P", "put")):
        price = lambda s=S, t=T, v=sigma: bs_spot(s, K, t, v, r, cp)
        assert g[f"{side}_delta"] == pytest.approx((price(S + h) - price(S - h)) / (2 * h), abs=1e-6)
        assert g["gamma"] == pytest.approx((price(S + h) - 2 * price() + price(S - h)) / h ** 2, rel=1e-3)
        assert g["vega"] == pytest.approx((price(v=sigma + dv) - price(v=sigma - dv)) / (2 * dv) / 100, rel=1e-6)
        # theta is the calendar decay: one day less to expiry, in dollars per day
        assert g[f"{side}_theta"] == pytest.approx(-(price(t=T + dt) - price(t=T - dt)) / (2 * dt) / 365, rel=1e-5)


@pytest.mark.parametrize("S,K,T,sigma,r", CASES)
def test_the_identities_from_the_relationship_slides_hold(S, K, T, sigma, r):
    g = W.spot_greeks(S, K, T, sigma, r)
    assert g["vega"] * 100 == pytest.approx(S * S * sigma * T * g["gamma"])                 # v = S^2 sigma T gamma
    assert g["theta_vol"] * 365 == pytest.approx(-0.5 * sigma ** 2 * S ** 2 * g["gamma"])   # theta_vol = -1/2 sigma^2 S^2 gamma
    assert g["theta_vol"] * 365 == pytest.approx(-(sigma / (2 * T)) * g["vega"] * 100)      # = -(sigma / 2T) v
    assert g["call_delta"] - g["put_delta"] == pytest.approx(1.0)
    assert (g["call_theta"] - g["put_theta"]) * 365 == pytest.approx(-r * K * np.exp(-r * T))


@pytest.mark.parametrize("S,K,T,sigma,r", CASES)
def test_the_lecture_greeks_agree_with_the_engines_forward_greeks(S, K, T, sigma, r):
    g = W.spot_greeks(S, K, T, sigma, r)
    F = S * np.exp(r * T)
    for cp, side in (("C", "call"), ("P", "put")):
        b = bs_greeks(F, K, T, sigma, np.exp(-r * T), cp)
        assert g[f"{side}_delta"] == pytest.approx(b["delta"] * F / S)
        assert g["gamma"] == pytest.approx(b["gamma"] * (F / S) ** 2)
        assert g["vega"] == pytest.approx(b["vega"] / 100)
        assert g["theta_vol"] == pytest.approx(b["theta"] / 365)       # the engine's theta is the vol piece


def test_the_three_clocks_gamma_and_theta_blow_up_into_expiry_and_vega_dies():
    T = np.array([1, 5, 30, 90, 360]) / 365
    g = W.spot_greeks(500.0, 500.0, T, 0.22, 0.04)
    assert (np.diff(g["gamma"]) < 0).all() and (np.diff(np.abs(g["theta_vol"])) < 0).all()
    assert (np.diff(g["vega"]) > 0).all()


def test_a_whole_surface_comes_from_one_call():
    m, days = np.arange(0.7, 1.31, 0.1), np.array([1, 30, 90])
    g = W.spot_greeks(500.0, m[None, :] * 500.0, days[:, None] / 365, 0.22, 0.04)
    assert g["put_theta"].shape == (3, len(m))
    one = W.spot_greeks(500.0, m[2] * 500.0, days[1] / 365, 0.22, 0.04)
    assert g["put_theta"][1, 2] == pytest.approx(float(one["put_theta"]))
