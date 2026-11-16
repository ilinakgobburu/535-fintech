"""
The nightly hedge, expiries and the book. Each held leg's quotes are Black-76 prices at a known vol on each
close, so its IV and delta are known by hand (bs_greeks); the target is -round(net delta) and the orders the
moves from the held shares to it, never one order across zero.
"""

from __future__ import annotations

import datetime as dt
import itertools

import numpy as np
import pandas as pd
import pytest

import va_live_synth as S
from trading_app.lib import vol_arb_live as VL
from trading_app.lib import wheel_live as L
from trading_app.lib.vol import bs_greeks

DAY, FOR = dt.date(2026, 11, 25), dt.date(2026, 11, 27)
NVDA = S.NVDA


def hedge_market(day=DAY):
    c, o, rate, div = S.market_parts(day)
    return VL.market(c, o, rate, div)


def by_hand_delta(mkt, cp, expiry, strike, sigma, day=DAY):
    """Delta to spot per share at D's close at vol `sigma`: Black-76 with F = S e^((r-q)T), D = e^(-rT)."""
    spot = float(mkt["stock"]["close"].loc[pd.Timestamp(day)])
    r = S.RATE_PCT / 100
    q = S.trailing_yield(mkt["dividends"], spot, day)
    T = (pd.Timestamp(expiry) - pd.Timestamp(day)).days / 365.0
    F = spot * np.exp((r - q) * T)
    return bs_greeks(F, strike, T, sigma, np.exp(-r * T), cp)["delta"] * F / spot


def history_at(mkt, cp, expiry, strike, sigma, days=5, missing=(), crossed=()):
    """A leg's last `days` closes of bid/ask, priced at `sigma` on each close's forward (r and the trailing q)."""
    r = S.RATE_PCT / 100
    rows = {}
    for d, s in mkt["stock"]["close"].tail(days).items():
        T = (pd.Timestamp(expiry) - d).days / 365.0
        q = S.trailing_yield(mkt["dividends"], s, d)
        mid = float(VL.VA.bs_price(s * np.exp((r - q) * T), strike, T, sigma, np.exp(-r * T), cp))
        bid, ask = mid * 0.99, mid * 1.01
        if d in missing:
            bid = ask = np.nan
        if d in crossed:
            bid, ask = ask, bid
        rows[d] = {"bid": bid, "ask": ask}
    return pd.DataFrame.from_dict(rows, orient="index")


LEGS = [("P", S.EXPIRY, 165.0, -10, 0.50), ("C", S.EXPIRY, 195.0, 12, 0.40)]


def book_of(legs, shares=0):
    return VL.Book(cash=0.0, shares=shares,
                   options=[{"cp": cp, "expiry": e, "strike": k, "qty": n} for cp, e, k, n, _ in legs])


def histories(mkt, legs, **kw):
    return {(cp, e, k): history_at(mkt, cp, e, k, s, **kw) for cp, e, k, _, s in legs}


def test_the_hedge_is_minus_round_net_delta():
    mkt = hedge_market()
    p = VL.plan_night(S.config(), DAY, FOR, book_of(LEGS), mkt=mkt, leg_history=histories(mkt, LEGS))
    net = sum(n * 100 * by_hand_delta(mkt, cp, e, k, s) for cp, e, k, n, s in LEGS)
    assert p["hedge"]["net_delta"] == pytest.approx(net, abs=1e-6)
    target = -int(round(net))
    assert p["hedge"]["target"] == target
    (o,) = p["orders"]
    assert o == {"id": "nvda-va-hedge-2026-11-25", "for": "2026-11-27", "order_type": "mkt",
                 "action": "BUY" if target > 0 else "SHT", "qty": abs(target), "underlying": NVDA,
                 "contract": "underlying", "ref": "open"}
    for leg, (cp, e, k, n, s) in zip(sorted(p["hedge"]["legs"], key=lambda x: x["cp"], reverse=True), LEGS):
        assert leg["iv"] == pytest.approx(s, abs=1e-6) and leg["iv_day"] == pd.Timestamp(DAY)


def test_already_at_the_target_means_no_order():
    mkt = hedge_market()
    first = VL.plan_night(S.config(), DAY, FOR, book_of(LEGS), mkt=mkt, leg_history=histories(mkt, LEGS))
    again = VL.plan_night(S.config(), DAY, FOR, book_of(LEGS, shares=first["hedge"]["target"]), mkt=mkt,
                          leg_history=histories(mkt, LEGS))
    assert again["orders"] == []


@pytest.mark.parametrize("flip", [1, -1])
def test_crossing_zero_is_two_orders(flip):
    mkt = hedge_market()
    legs = [(cp, e, k, flip * n, s) for cp, e, k, n, s in LEGS]       # flip = -1: the opposite trade
    target = -int(round(sum(n * 100 * by_hand_delta(mkt, cp, e, k, s) for cp, e, k, n, s in legs)))
    shares = -int(np.sign(target)) * 1500                              # held on the other side of zero
    p = VL.plan_night(S.config(), DAY, FOR, book_of(legs, shares=shares), mkt=mkt, leg_history=histories(mkt, legs))
    assert p["hedge"]["target"] == target and target != 0
    if shares > 0:
        want = [("nvda-va-hedge-sel-2026-11-25", "SEL", shares), ("nvda-va-hedge-sht-2026-11-25", "SHT", -target)]
    else:
        want = [("nvda-va-hedge-btc-2026-11-25", "BTC", -shares), ("nvda-va-hedge-buy-2026-11-25", "BUY", target)]
    assert [(o["id"], o["action"], o["qty"]) for o in p["orders"]] == want
    for o in p["orders"]:
        L.validate_order(o)


@pytest.mark.parametrize("held, target, orders", [
    (0, 5, [("BUY", 5)]), (0, -5, [("SHT", 5)]), (5, 8, [("BUY", 3)]), (5, 2, [("SEL", 3)]), (5, 0, [("SEL", 5)]),
    (5, -3, [("SEL", 5), ("SHT", 3)]), (-5, -8, [("SHT", 3)]), (-5, -2, [("BTC", 3)]), (-5, 0, [("BTC", 5)]),
    (-5, 3, [("BTC", 5), ("BUY", 3)]), (4, 4, []), (0, 0, []),
])
def test_transition(held, target, orders):
    assert VL.transition(held, target) == orders


def test_transitions_reach_the_target_without_an_order_crossing_zero():
    sign = {"BUY": 1, "BTC": 1, "SEL": -1, "SHT": -1}
    for held, target in itertools.product(range(-7, 8), repeat=2):
        pos = held
        for action, qty in VL.transition(held, target):
            assert qty > 0
            new = pos + sign[action] * qty
            assert not (pos > 0 and new < 0) and not (pos < 0 and new > 0)        # never through zero
            assert (action in ("SEL", "BTC")) == (pos != 0 and abs(new) < abs(pos))  # SEL/BTC only reduce
            assert not (action == "SEL" and pos <= 0) and not (action == "BTC" and pos >= 0)
            pos = new
        assert pos == target


def test_a_leg_with_no_quote_tonight_keeps_its_last_iv():
    mkt = hedge_market()
    yesterday = mkt["stock"].index[-2]
    legs = [("P", S.EXPIRY, 165.0, -10, 0.50)]
    for kw in ({"missing": (pd.Timestamp(DAY),)}, {"crossed": (pd.Timestamp(DAY),)}):
        hist = {("P", S.EXPIRY, 165.0): history_at(mkt, "P", S.EXPIRY, 165.0, 0.50, **kw)}
        p = VL.plan_night(S.config(), DAY, FOR, book_of(legs), mkt=mkt, leg_history=hist)
        leg = p["hedge"]["legs"][0]
        assert leg["iv_day"] == yesterday and leg["iv"] == pytest.approx(0.50, abs=1e-6)
        assert leg["delta"] == pytest.approx(by_hand_delta(mkt, "P", S.EXPIRY, 165.0, 0.50), abs=1e-9)  # at D's spot


def test_a_leg_with_no_usable_quote_stops_the_plan():
    mkt = hedge_market()
    legs = [("P", S.EXPIRY, 165.0, -10, 0.50)]
    days = tuple(mkt["stock"].index[-5:])
    hist = {("P", S.EXPIRY, 165.0): history_at(mkt, "P", S.EXPIRY, 165.0, 0.50, missing=days)}
    with pytest.raises(ValueError, match="no usable quote"):
        VL.plan_night(S.config(), DAY, FOR, book_of(legs), mkt=mkt, leg_history=hist)
    with pytest.raises(ValueError, match="no usable quote"):
        VL.plan_night(S.config(), DAY, FOR, book_of(legs), mkt=mkt, leg_history={})


def test_quotes_after_d_are_not_read():
    mkt = hedge_market()
    legs = [("P", S.EXPIRY, 165.0, -10, 0.50)]
    hist = history_at(mkt, "P", S.EXPIRY, 165.0, 0.50, missing=(pd.Timestamp(DAY),))
    hist.loc[pd.Timestamp("2026-11-27")] = {"bid": 50.0, "ask": 51.0}       # a later row must not be used
    p = VL.plan_night(S.config(), DAY, FOR, book_of(legs), mkt=mkt, leg_history={("P", S.EXPIRY, 165.0): hist})
    assert p["hedge"]["legs"][0]["iv"] == pytest.approx(0.50, abs=1e-6)


# --------------------------------------------------------------------------- expiry
@pytest.mark.parametrize("expiry", [DAY, DAY - dt.timedelta(days=3)])
def test_an_option_listed_on_or_after_its_expiry_is_a_stale_book(expiry):
    book = book_of([("P", expiry, 165.0, -10, 0.5)])
    assert VL.expired_legs(book, DAY) == book.options
    with pytest.raises(VL.StaleBook):
        VL.plan_night(S.config(), DAY, FOR, book, mkt=hedge_market())
    assert VL.expired_legs(book_of([("P", DAY + dt.timedelta(days=1), 165.0, -10, 0.5)]), DAY) == []


@pytest.mark.parametrize("shares, action", [(-400, "BTC"), (250, "SEL")])
def test_after_an_expiry_the_leftover_hedge_is_closed(shares, action):
    day = dt.date(2026, 12, 18)                       # the expiry's night, Sanji has removed the legs
    p = VL.plan_night(S.config(), day, dt.date(2026, 12, 21), VL.Book(cash=0.0, shares=shares))
    assert [(o["id"], o["action"], o["qty"], o["order_type"]) for o in p["orders"]] == \
        [("nvda-va-hedge-2026-12-18", action, abs(shares), "mkt")]
    assert p["hedge"]["legs"] == [] and p["hedge"]["target"] == 0


def test_no_options_and_no_shares_is_no_order():
    p = VL.plan_night(S.config(), DAY, FOR, VL.Book(cash=100.0))
    assert p["orders"] == []


def test_held_options_need_market_data():
    with pytest.raises(ValueError):
        VL.plan_night(S.config(), DAY, FOR, book_of(LEGS))


# --------------------------------------------------------------------------- the book
def test_no_positions_file_is_an_empty_book():
    b = VL.read_book(None, None, NVDA)
    assert (b.shares, b.options) == (0, [])
    with pytest.raises(L.BookFormatError):
        VL.read_book(None, [{"id": "x", "status": "filled"}], NVDA)


def test_the_book_reads_short_shares_and_both_strike_forms():
    pos = {"cash": 1000.0, "positions": [
        {"underlying": NVDA, "contract": "underlying", "qty": -120},
        {"underlying": NVDA, "contract": "put", "expiry": "2026-12-18", "strike": 165, "qty": -10},
        {"underlying": NVDA, "contract": "call", "expiry": "2026-12-18", "strike": "182.5", "qty": 12},
        {"underlying": NVDA, "contract": "call", "expiry": "2026-12-18", "strike": 200, "qty": 0},
        {"underlying": 999, "contract": "put", "expiry": "2026-12-18", "strike": 50, "qty": -1}]}
    b = VL.read_book(pos, [], NVDA)
    assert b.shares == -120 and b.cash == 1000.0
    assert b.options == [{"cp": "P", "expiry": dt.date(2026, 12, 18), "strike": 165.0, "qty": -10},
                         {"cp": "C", "expiry": dt.date(2026, 12, 18), "strike": 182.5, "qty": 12}]


def test_the_same_contract_twice_is_one_position():
    b = VL.Book(options=[{"cp": "P", "expiry": S.EXPIRY, "strike": 165.0, "qty": -10},
                         {"cp": "P", "expiry": S.EXPIRY, "strike": 165.0, "qty": 4},
                         {"cp": "C", "expiry": S.EXPIRY, "strike": 195.0, "qty": 3},
                         {"cp": "C", "expiry": S.EXPIRY, "strike": 195.0, "qty": -3}])
    assert VL.held_contracts(b) == {("P", S.EXPIRY, 165.0): -6}


@pytest.mark.parametrize("positions, fills", [
    ([], None),
    ({"positions": []}, None),
    ({"cash": 1, "positions": {}}, None),
    ({"cash": 1, "positions": [{"underlying": NVDA, "contract": "put", "expiry": "2026-12-18", "strike": 1,
                                "qty": "-1"}]}, None),
    ({"cash": 1, "positions": [{"underlying": NVDA, "contract": "future", "qty": 1}]}, None),
    ({"cash": 1, "positions": [{"underlying": "4815747", "contract": "underlying", "qty": 5}]}, None),
    ({"cash": 1, "positions": [{"underlying": NVDA, "contract": "put", "strike": 170, "qty": -5}]}, None),
    ({"cash": 1, "positions": [{"underlying": NVDA, "contract": "put", "expiry": "2026-12-18", "qty": -5}]}, None),
    ({"cash": 1, "positions": [{"underlying": NVDA, "contract": "put", "expiry": "2026-12-18", "strike": "otm:1",
                                "qty": -5}]}, None),
    ({"cash": 1, "positions": [{"underlying": NVDA, "contract": "put", "expiry": "2026-12-18", "strike": True,
                                "qty": -5}]}, None),
    ({"cash": 1, "positions": [{"underlying": NVDA, "contract": "put", "expiry": "2026-12-18", "strike": 0,
                                "qty": -5}]}, None),
    (None, {"status": "filled"}),
])
def test_an_unexpected_book_stops_the_script(positions, fills):
    with pytest.raises(L.BookFormatError):
        VL.read_book(positions, fills, NVDA)


def test_order_checks():
    good = VL.stock_orders(root="NVDA", conid=NVDA, day=DAY, for_day=FOR, held=0, target=10)
    VL.check_orders(good)
    with pytest.raises(ValueError):
        VL.check_orders(good + good)                                   # ids repeat
    opt = VL.option_orders(root="NVDA", conid=NVDA, day=DAY, for_day=FOR, cp="P", expiry=S.EXPIRY, strike=165.0,
                           limit=3.2149, held=0, target=-3)
    assert opt[0]["lmt_prc"] == "3.21" and opt[0]["strike"] == "165"
    VL.check_orders(opt)
    for bad in ({**opt[0], "strike": 165.0}, {**opt[0], "lmt_prc": 3.21}, {**opt[0], "action": "SELL"}):
        with pytest.raises(ValueError):
            VL.check_orders([bad])
