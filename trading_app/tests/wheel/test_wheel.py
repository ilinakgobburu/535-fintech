"""
Tests for the wheel engine, on a market small enough to check by hand: a $50
stock, puts and calls priced from one known volatility with a 10 cent quote
around each price, no dividends and a zero rate unless a test says otherwise.
Every expected number below is arithmetic on those inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_app.lib import wheel as W
from trading_app.lib.pmcc_daily import monthly_expiries
from trading_app.lib.ric import build_option_ric
from trading_app.lib.vol import bs_greeks, bs_price

VOL = 0.30
HALF = 0.05          # half the quoted spread
START, END = "2019-01-02", "2019-06-28"
STRIKES = tuple(np.arange(30.0, 70.01, 2.5))
# 2019 monthly expiries: Jan 18, Feb 15, Mar 15, Apr 18 (Good Friday), May 17, Jun 21
FEB, MAR, APR = pd.Timestamp("2019-02-15"), pd.Timestamp("2019-03-15"), pd.Timestamp("2019-04-18")


def make_market(path: dict | None = None, rate: float = 0.0, dividends: dict | None = None,
                lot: dict | None = None, bump_after: str | None = None) -> dict:
    """
    `path` overrides closes from a date on. `lot` sets the units per contract
    from a date on. `bump_after` doubles every quote after that date, which
    no decision made on or before it may notice.
    """
    days = pd.bdate_range("2018-12-03", "2019-07-31")
    days = days[~days.isin(pd.to_datetime(["2019-01-21", "2019-02-18", "2019-04-19", "2019-05-27",
                                           "2019-07-04"]))]
    close = pd.Series(50.0, index=days)
    for d, v in (path or {}).items():
        close[close.index >= pd.Timestamp(d)] = v
    stock = pd.DataFrame({"close": close, "open": close})

    rows = []
    for expiry in monthly_expiries(days, "2019-01-01", "2019-07-31"):
        for cp in ("P", "C"):
            for k in STRIKES:
                ric = build_option_ric("UUUU", expiry.date(), float(k), cp)
                for d in days[days <= expiry]:
                    T = (expiry - d).days / 365.0
                    fair = (bs_price(close[d], k, T, VOL, 1.0, cp) if T > 0
                            else max((close[d] - k) if cp == "C" else (k - close[d]), 0.0))
                    bump = 2.0 if bump_after and d > pd.Timestamp(bump_after) else 1.0
                    bid = round(fair * bump - HALF, 4)
                    rows.append({"date": d, "ric": ric, "cp": cp, "strike": float(k), "expiry": expiry,
                                 "bid": bid if bid > 0 else np.nan, "ask": round(fair * bump + HALF, 4)})
    quotes = pd.DataFrame(rows)
    quotes["mid"] = np.where(quotes["bid"].notna(), (quotes["bid"] + quotes["ask"]) / 2.0, np.nan)
    lots = pd.Series(100.0, index=days)
    for d, v in (lot or {}).items():
        lots[lots.index >= pd.Timestamp(d)] = v
    div = pd.Series(dividends or {}, dtype=float)
    div.index = pd.DatetimeIndex(pd.to_datetime(div.index))
    return {"quotes": quotes, "stock": stock, "lot": lots, "tbill": pd.Series(rate, index=days),
            "dividends": div, "earnings": pd.DatetimeIndex([])}


def bs_delta(spot, k, day, expiry, cp):
    return bs_greeks(spot, k, (expiry - pd.Timestamp(day)).days / 365.0, VOL, 1.0, cp)["delta"]


def nearest(spot, day, expiry, cp, target, floor=None):
    """The strike the rule should pick, straight from Black-Scholes at the known vol."""
    ks = [k for k in STRIKES if floor is None or k >= floor - 1e-9]
    T = (expiry - pd.Timestamp(day)).days / 365.0
    ks = [k for k in ks if bs_price(spot, k, T, VOL, 1.0, cp) - HALF > 0]
    miss = {k: abs(abs(bs_delta(spot, k, day, expiry, cp)) - target) for k in ks}
    best = min(miss.values())
    tied = [k for k in ks if abs(miss[k] - best) < 1e-12]
    return min(tied) if cp == "P" else max(tied)


def cycle(result, entry):
    return next(c for c in result["cycles"] if c["entry"] == pd.Timestamp(entry))


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def test_put_delta_is_negative_and_inverted_from_the_mid():
    g = W.option_greeks(bs_price(50.0, 47.5, 0.08, VOL, 1.0, "P"), 50.0, 47.5, 0.08, cp="P")
    assert g["iv"] == pytest.approx(VOL, abs=1e-6)
    assert g["delta"] == pytest.approx(bs_greeks(50.0, 47.5, 0.08, VOL, 1.0, "P")["delta"], abs=1e-6)
    assert g["delta"] < 0


@pytest.mark.parametrize("target", [0.15, 0.25, 0.35])
def test_first_put_is_the_strike_with_delta_closest_to_target(target):
    res = W.run(make_market(), start=START, end=END, put_delta=target)
    c = cycle(res, "2019-01-22")
    assert c["cp"] == "P" and c["expiry"] == FEB
    assert c["strike"] == nearest(50.0, "2019-01-22", FEB, "P", target)
    assert abs(c["delta"]) == pytest.approx(abs(bs_delta(50.0, c["strike"], "2019-01-22", FEB, "P")), abs=1e-4)


def test_a_delta_tie_goes_further_out_of_the_money(monkeypatch):
    day, exp = pd.Timestamp("2019-01-22"), FEB
    q = make_market()["quotes"]
    q = q[q["date"] == day]
    fake = {45.0: -0.20, 47.5: -0.30, 50.0: -0.45}
    monkeypatch.setattr(W, "option_greeks",
                        lambda mid, spot, k, T, r, qy, cp: {"iv": VOL, "delta": fake.get(k, -0.99)
                                                            if cp == "P" else -fake.get(k, -0.99)})
    assert W.select(q, "P", exp, 50.0, day, 0.0, 0.0, 0.25)["strike"] == 45.0
    assert W.select(q, "C", exp, 50.0, day, 0.0, 0.0, 0.25)["strike"] == 47.5


def test_a_strike_without_a_bid_cannot_be_selected():
    day = pd.Timestamp("2019-01-22")
    q = make_market()["quotes"]
    q = q[q["date"] == day]
    pick = W.select(q, "P", FEB, 50.0, day, 0.0, 0.0, 0.0001)
    assert pick is not None and np.isfinite(pick["mid"]) and pick["bid"] > 0


# --------------------------------------------------------------------------
# the cycle
# --------------------------------------------------------------------------

def test_flat_stock_every_put_expires_and_the_premium_is_kept():
    res = W.run(make_market(), start=START, end=END)
    assert [c["status"] for c in res["cycles"]] == ["expired"] * 5
    assert all(c["cp"] == "P" for c in res["cycles"])
    assert (res["ledger"]["shares"] == 0).all()
    assert res["ledger"]["cash"].iloc[-1] == pytest.approx(W.START_CASH + sum(c["premium"] for c in res["cycles"]))


def test_puts_are_sized_by_the_cash_already_in_the_account():
    res = W.run(make_market(), start=START, end=END)
    cash = W.START_CASH
    for c in res["cycles"]:
        assert c["contracts"] == int(cash // (100 * c["strike"]))
        assert c["units"] == 100 * c["contracts"]
        cash += c["premium"]


def test_assignment_buys_at_the_strike_and_sets_the_cost_basis():
    res = W.run(make_market(path={"2019-02-14": 45.0}), start=START, end=END)
    first = cycle(res, "2019-01-22")
    assert first["status"] == "assigned" and first["settle"] == 45.0
    led = res["ledger"].set_index("date")
    n = first["units"]
    assert led.at[FEB, "shares"] == n
    assert led.at[FEB, "cash"] == pytest.approx(W.START_CASH + first["premium"] - n * first["strike"])
    assert led.at[FEB, "basis"] == pytest.approx(first["strike"] - first["fill"])


def test_after_assignment_the_book_writes_one_call_per_contract_of_shares():
    res = W.run(make_market(path={"2019-02-14": 45.0}), start=START, end=END)
    first, second = cycle(res, "2019-01-22"), cycle(res, "2019-02-19")
    assert second["cp"] == "C" and second["units"] == first["units"]
    assert second["contracts"] == first["contracts"]


def test_the_floor_keeps_calls_at_or_above_the_cost_basis():
    # At $43 the 0.25-delta call is the 45, below a basis near 46.93; the
    # floor moves the book out to the 47.5, which bids a fifth as much.
    market = make_market(path={"2019-02-14": 43.0})
    on = W.run(market, start=START, end=END, floor=True)
    off = W.run(market, start=START, end=END, floor=False)
    basis = cycle(on, "2019-01-22")["strike"] - cycle(on, "2019-01-22")["fill"]
    assert cycle(on, "2019-02-19")["strike"] >= basis
    assert cycle(on, "2019-02-19")["strike"] == nearest(43.0, "2019-02-19", MAR, "C", 0.25, floor=basis)
    assert cycle(off, "2019-02-19")["strike"] == nearest(43.0, "2019-02-19", MAR, "C", 0.25)
    assert cycle(off, "2019-02-19")["strike"] < basis
    assert cycle(on, "2019-02-19")["premium"] < cycle(off, "2019-02-19")["premium"]


def test_each_call_premium_lowers_the_cost_basis():
    res = W.run(make_market(path={"2019-02-14": 40.0}), start=START, end=END, floor=False)
    first, second = cycle(res, "2019-01-22"), cycle(res, "2019-02-19")
    led = res["ledger"].set_index("date")
    assert led.at[pd.Timestamp("2019-02-19"), "basis"] == pytest.approx(first["strike"] - first["fill"] - second["fill"])


def test_with_no_qualifying_call_the_shares_are_held_uncovered():
    res = W.run(make_market(path={"2019-02-14": 30.0}), start=START, end=END, floor=True)
    c = cycle(res, "2019-02-19")
    assert c["status"] == "held: no qualifying call" and "strike" not in c
    led = res["ledger"].set_index("date")
    assert led.at[pd.Timestamp("2019-02-20"), "option_mv"] == 0.0


def test_called_away_sells_at_the_strike_and_the_next_cycle_sells_puts():
    res = W.run(make_market(path={"2019-02-14": 45.0, "2019-03-14": 55.0}), start=START, end=END)
    call = cycle(res, "2019-02-19")
    assert call["status"] == "called away"
    led = res["ledger"].set_index("date")
    assert led.at[MAR, "shares"] == 0
    before = led.loc[led.index < MAR, "cash"].iloc[-1]
    assert led.at[MAR, "cash"] == pytest.approx(before + call["units"] * call["strike"])
    assert cycle(res, "2019-03-18")["cp"] == "P"


def test_cross_fills_sell_at_the_bid():
    res = W.run(make_market(path={"2019-02-14": 45.0}), start=START, end=END, fill="cross")
    for c in res["cycles"]:
        if "strike" in c:
            assert c["fill"] == c["bid"]
            assert c["premium"] == pytest.approx(c["units"] * c["bid"])


def test_cash_too_small_to_secure_a_contract_stays_idle():
    res = W.run(make_market(), start=START, end=END, start_cash=1_000.0)
    assert all(c["status"] == "idle: cash secures no contract" for c in res["cycles"])
    assert res["ledger"]["nav"].iloc[-1] == pytest.approx(1_000.0)


# --------------------------------------------------------------------------
# what is known when
# --------------------------------------------------------------------------

def test_a_decision_does_not_see_later_quotes():
    plain = W.run(make_market(), start=START, end=END)
    bumped = W.run(make_market(bump_after="2019-01-22"), start=START, end=END)
    a, b = cycle(plain, "2019-01-22"), cycle(bumped, "2019-01-22")
    assert (a["strike"], a["fill"], a["contracts"]) == (b["strike"], b["fill"], b["contracts"])


def test_the_first_cycle_opens_the_session_after_the_first_expiry():
    res = W.run(make_market(), start=START, end=END)
    assert res["cycles"][0]["entry"] == pd.Timestamp("2019-01-22")
    assert [c["expiry"] for c in res["cycles"]][-1] == pd.Timestamp("2019-06-21")


# --------------------------------------------------------------------------
# carry, marks and the ledger
# --------------------------------------------------------------------------

def test_cash_earns_the_bill_over_calendar_days():
    res = W.run(make_market(rate=0.05), start=START, end=END, mode="cash")
    days = res["ledger"]["date"]
    expected = W.START_CASH * np.prod(1.0 + 0.05 * days.diff().dt.days.dropna() / 365.0)
    assert res["ledger"]["nav"].iloc[-1] == pytest.approx(expected)
    assert res["totals"]["interest"] == pytest.approx(expected - W.START_CASH)


def test_buy_and_hold_buys_whole_shares_and_collects_dividends():
    res = W.run(make_market(dividends={"2019-03-01": 0.50}), start=START, end=END, mode="buy_and_hold")
    led = res["ledger"]
    assert led["shares"].iloc[0] == 2000 and led["cash"].iloc[0] == pytest.approx(0.0)
    assert res["totals"]["dividends"] == pytest.approx(2000 * 0.50)


def test_dividends_go_to_shares_held_at_the_prior_close():
    market = make_market(path={"2019-02-14": 45.0}, dividends={"2019-02-15": 1.0, "2019-02-20": 1.0})
    res = W.run(market, start=START, end=END)
    n = cycle(res, "2019-01-22")["units"]
    # Feb 15 is the assignment day: no shares at the prior close. Feb 20: n shares.
    assert res["totals"]["dividends"] == pytest.approx(n * 1.0)


def test_nav_is_cash_plus_shares_plus_the_short_option():
    res = W.run(make_market(path={"2019-02-14": 45.0, "2019-04-10": 52.0}), start=START, end=END)
    led = res["ledger"]
    assert np.allclose(led["nav"], led["cash"] + led["shares"] * led["stock_close"] + led["option_mv"])


def test_the_short_is_marked_at_its_mid_while_open():
    market = make_market()
    res = W.run(market, start=START, end=END)
    c = cycle(res, "2019-01-22")
    day = pd.Timestamp("2019-02-01")
    q = market["quotes"]
    mid = float(q[(q["date"] == day) & (q["strike"] == c["strike"]) & (q["cp"] == "P")
                  & (q["expiry"] == FEB)]["mid"].iloc[0])
    assert res["ledger"].set_index("date").at[day, "option_mv"] == pytest.approx(-c["units"] * mid)


@pytest.mark.parametrize("path", [None, {"2019-02-14": 45.0}, {"2019-02-14": 40.0, "2019-04-10": 55.0}])
def test_attribution_sums_to_the_pnl(path):
    res = W.run(make_market(path=path, rate=0.02, dividends={"2019-03-01": 0.25}), start=START, end=END)
    a = W.attribution(res)
    assert a["premium"] + a["interest"] + a["dividends"] + a["shares"] + a["open_short"] == pytest.approx(a["pnl"])


# --------------------------------------------------------------------------
# units: real contracts across a split
# --------------------------------------------------------------------------

NVDA_SPLITS = (pd.Timestamp("2021-07-20"), [(pd.Timestamp("2021-07-20"), 4.0), (pd.Timestamp("2024-06-10"), 10.0)])


def test_unit_factor_puts_nvda_in_shares_after_the_2021_split():
    f = W.unit_factor(["2021-07-19", "2021-07-20", "2024-06-07", "2024-06-10"], NVDA_SPLITS)
    assert list(f) == [0.25, 1.0, 1.0, 10.0]
    assert list(W.unit_factor(["2021-07-16", "2021-08-20"], None)) == [1.0, 1.0]


def test_a_real_contract_before_the_split_is_four_hundred_units():
    res = W.run(make_market(lot={"2018-12-03": 400.0}), start=START, end=END)
    c = cycle(res, "2019-01-22")
    assert c["contracts"] == int(W.START_CASH // (400 * c["strike"]))
    assert c["units"] == 400 * c["contracts"]


def test_after_the_split_the_same_shares_carry_four_times_the_calls():
    market = make_market(path={"2019-02-14": 45.0}, lot={"2018-12-03": 400.0, "2019-02-19": 100.0})
    res = W.run(market, start=START, end=END)
    put, call = cycle(res, "2019-01-22"), cycle(res, "2019-02-19")
    assert call["units"] == put["units"]
    assert call["contracts"] == 4 * put["contracts"]


# --------------------------------------------------------------------------
# the grid and the pick
# --------------------------------------------------------------------------

def test_the_grid_has_eighteen_cells_in_a_fixed_order():
    cells = W.grid_cells([0.15, 0.25, 0.35], [0.15, 0.25, 0.35], [True, False])
    assert len(cells) == 18
    assert cells[0] == {"put_delta": 0.15, "call_delta": 0.15, "floor": True}
    assert cells[1] == {"put_delta": 0.15, "call_delta": 0.15, "floor": False}
    assert cells[-1] == {"put_delta": 0.35, "call_delta": 0.35, "floor": False}


def test_the_pick_is_the_highest_sharpe_and_a_tie_goes_to_the_earlier_cell():
    scores = [{"cell": 0, "sharpe": 0.5}, {"cell": 1, "sharpe": 0.80001}, {"cell": 2, "sharpe": 0.8},
              {"cell": 3, "sharpe": np.nan}]
    assert W.pick(scores)["cell"] == 1
    assert W.pick([{"cell": 0, "sharpe": 0.8}, {"cell": 1, "sharpe": 0.80004}])["cell"] == 0
