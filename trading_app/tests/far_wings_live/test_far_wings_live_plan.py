"""
far_wings_live.plan_night on made-up books and markets: the entry night's two SHT limits at the closing mid (legal by
order.txt, the course's order page, field by field), the band each night options are held (at its boundary exactly,
just past it on either side, the target zero shares-delta, two orders across zero), the night after an expiry (the
shares closed), and the guards on a stale book and on a misread hedge.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import fw_live_synth as S
from trading_app.lib import far_wings_live as FW
from trading_app.lib import wheel as W
from trading_app.lib import wheel_live as L

CFG = S.config()
EXP = S.EXPIRY


def book(options=(), shares=0):
    return FW.Book(cash=100_000.0, shares=shares,
                   options=[{"cp": cp, "expiry": e, "strike": float(k), "qty": q} for cp, e, k, q in options])


def plan(day, b, mkt=None, **kw):
    return FW.plan_night(CFG, day, FW.next_session(day), b, mkt=mkt, **kw)


# --------------------------------------------------------------------------- the entry night
def test_the_entry_night_sells_both_wings_at_the_closing_mid():
    mkt = S.market()
    got = plan(S.ENTRY, book(), mkt, entry_expiry=EXP)
    wings = FW.wings_at_entry(mkt, S.ENTRY, EXP)
    assert got["orders"] == [
        {"id": "nvda-fw-put-2026-11-23", "for": "2026-11-24", "order_type": "lmt", "action": "SHT",
         "qty": wings["P"]["contracts"], "underlying": S.NVDA, "contract": "put", "expiry": "2026-12-18",
         "strike": "145", "ref": "open", "lmt_prc": L.number_token(wings["P"]["mid"])},
        {"id": "nvda-fw-call-2026-11-23", "for": "2026-11-24", "order_type": "lmt", "action": "SHT",
         "qty": wings["C"]["contracts"], "underlying": S.NVDA, "contract": "call", "expiry": "2026-12-18",
         "strike": "220", "ref": "open", "lmt_prc": L.number_token(wings["C"]["mid"])}]
    assert [o["lmt_prc"] for o in got["orders"]] == ["0.53", "0.24"]           # the mids 0.5288 and 0.2393, to the cent
    assert (got["orders"][0]["qty"], got["orders"][1]["qty"]) == (41, 59)
    assert got["hedge"]["legs"] == [] and got["hedge"]["breached"] is None       # nothing held yet: no band check
    assert "entry 2026-11-23 -> 2026-12-18: sell 41 145 puts at 0.53 (|delta| 0.050)" in got["notes"]


def test_every_order_is_legal_by_the_order_page():
    nights = [plan(S.ENTRY, book(), S.market(), entry_expiry=EXP)["orders"]]
    day = dt.date(2026, 12, 1)
    c = S.closes(end=day)
    for shares, last in ((0, 100.0), (300, 100.0), (-300, 400.0)):      # SHT; SEL then SHT; BTC then BUY
        c2 = S.closes(end=day, last=last)
        hist = {("P", EXP, 145.0): S.leg_history(c2, "P", EXP, 145.0, 0.5),
                ("C", EXP, 220.0): S.leg_history(c2, "C", EXP, 220.0, 0.45)}
        nights.append(plan(day, book([("P", EXP, 145, -41), ("C", EXP, 220, -59)], shares),
                           S.market(day, c=c2, chain=False), leg_history=hist)["orders"])
    nights.append(plan(day, book(shares=-250), S.market(day, c=c, chain=False))["orders"])
    every = [o for n in nights for o in n]
    assert {o["action"] for o in every} == {"SHT", "BUY", "SEL", "BTC"}
    assert {o["contract"] for o in every} == {"put", "call", "underlying"}
    for o in every:
        assert S.order_problems(o) == [], o
        L.validate_order(o)
        assert o["underlying"] == S.NVDA and o["ref"] == "open"
        if o["contract"] == "underlying":
            assert o["order_type"] == "mkt" and "lmt_prc" not in o and "strike" not in o and "expiry" not in o
        else:
            assert o["order_type"] == "lmt" and o["action"] == "SHT"
    ids = [o["id"] for o in every]
    assert all(i.startswith("nvda-fw-") for i in ids)


def test_bare_number_strike_and_price_strings():
    mkt = S.market(rows=S.chain_rows(float(S.closes().iloc[-1]), strikes=np.arange(120.0, 260.0, 2.5)))
    for o in plan(S.ENTRY, book(), mkt, entry_expiry=EXP)["orders"]:
        assert isinstance(o["strike"], str) and isinstance(o["lmt_prc"], str)
        assert o["strike"] == L.number_token(float(o["strike"])) and "e" not in o["lmt_prc"]
    o = FW.option_orders(root="NVDA", conid=S.NVDA, day=S.ENTRY, for_day=S.FOR, cp="P", expiry=EXP, strike=147.5,
                         limit="0.4", held=0, target=-3)
    assert o == [{"id": "nvda-fw-put-2026-11-23", "for": "2026-11-24", "order_type": "lmt", "action": "SHT", "qty": 3,
                  "underlying": S.NVDA, "contract": "put", "expiry": "2026-12-18", "strike": "147.5", "ref": "open",
                  "lmt_prc": "0.4"}]


def test_a_wing_with_no_eligible_strike_is_left_out_and_the_other_trades():
    spot = float(S.closes().iloc[-1])
    rows = [r for r in S.chain_rows(spot) if r["cp"] == "P" or r["strike"] <= 195.0]   # no call far enough out
    got = plan(S.ENTRY, book(), S.market(rows=rows), entry_expiry=EXP)
    assert [o["id"] for o in got["orders"]] == ["nvda-fw-put-2026-11-23"]
    assert "entry 2026-11-23 -> 2026-12-18: no call wing (no two-sided call under 0.05 |delta|)" in got["notes"]


def test_a_mid_that_rounds_to_nothing_is_not_sold(monkeypatch):
    mkt = S.market()
    real = FW.wings_at_entry(mkt, S.ENTRY, EXP)
    tiny = {**real, "C": {**real["C"], "mid": 0.004}}                  # a mid of 0.004: a limit of "0"
    monkeypatch.setattr(FW, "wings_at_entry", lambda m, d, e: tiny)
    got = plan(S.ENTRY, book(), mkt, entry_expiry=EXP)
    assert [o["contract"] for o in got["orders"]] == ["put"]
    assert any("no call wing (its mid 0.0040 rounds to a limit of 0)" in n for n in got["notes"])


def test_a_night_that_is_not_an_entry_with_an_empty_book_is_no_order():
    for day in (dt.date(2026, 11, 17), dt.date(2026, 11, 24), dt.date(2026, 12, 18)):
        got = plan(day, book())
        assert got["orders"] == [] and got["notes"] == ["no order tonight"] and got["entry"] is None


# --------------------------------------------------------------------------- the band
def test_breached_is_strictly_outside_5000_dollars():
    assert FW.band_dollars() == 5000.0
    assert not FW.breached(200, 25.0) and not FW.breached(-200, 25.0)          # exactly $5,000: inside
    assert FW.breached(200.001, 25.0) and FW.breached(-200.001, 25.0)
    assert FW.breached(200, 25.0001) and not FW.breached(199.999, 25.0)
    assert not FW.breached(0, 1e6)


@pytest.fixture
def exact_delta(monkeypatch):
    """Every held leg's delta set by hand (an exact binary fraction), so the band's edge can be hit exactly."""
    deltas = {}

    def fake(mkt, cp, expiry, strike, history, day):
        return {"iv": 0.5, "iv_day": pd.Timestamp(day), "mid": 1.0, "delta": deltas[(cp, strike)]}

    monkeypatch.setattr(FW, "leg_delta", fake)
    return deltas


def boundary_market(day, spot):
    return S.market(day, c=S.closes(end=day, last=spot), chain=False)


@pytest.mark.parametrize("cp, delta, action", [("P", -0.0625, "SHT"), ("C", 0.0625, "BUY")])
def test_the_band_at_its_boundary(exact_delta, cp, delta, action):
    day = dt.date(2026, 11, 30)
    exact_delta[(cp, 180.0)] = delta
    b = book([(cp, EXP, 180, -32)])                       # -32 x 100 x delta: +-200 shares, $5,000 at $25
    at = plan(day, b, boundary_market(day, 25.0))
    assert at["hedge"]["option_delta"] == -32 * 100 * delta and at["hedge"]["exposure"] == 5000.0
    assert at["orders"] == [] and at["hedge"]["breached"] is False and at["hedge"]["target"] == 0
    assert "is inside the $5,000 band: no hedge trade (holding 0)" in at["notes"][0]
    past = plan(day, b, boundary_market(day, 25.01))
    assert past["hedge"]["breached"] is True
    assert [(o["id"], o["action"], o["qty"], o["order_type"]) for o in past["orders"]] == [
        ("nvda-fw-hedge-2026-11-30", action, 200, "mkt")]


def test_the_shares_count_in_the_band(exact_delta):
    day = dt.date(2026, 11, 30)
    exact_delta[("P", 180.0)] = -0.0625
    # +200 shares of option delta: holding -190 shares the net is +10 ($250), inside; holding +10 it is 210, outside
    inside = plan(day, book([("P", EXP, 180, -32)], shares=-190), boundary_market(day, 25.0))
    assert inside["hedge"]["net_delta"] == 10 and inside["orders"] == []
    outside = plan(day, book([("P", EXP, 180, -32)], shares=10), boundary_market(day, 25.0))
    assert outside["hedge"]["net_delta"] == 210
    # back to zero shares-delta: -200 shares, so the 10 long are sold and 200 shorted, the close first
    assert [(o["id"], o["action"], o["qty"]) for o in outside["orders"]] == [
        ("nvda-fw-hedge-sel-2026-11-30", "SEL", 10), ("nvda-fw-hedge-sht-2026-11-30", "SHT", 200)]
    # holding -200 already: net 0, nothing
    assert plan(day, book([("P", EXP, 180, -32)], shares=-200), boundary_market(day, 25.0))["orders"] == []


def test_two_orders_when_the_hedge_crosses_zero_from_short_to_long(exact_delta):
    day = dt.date(2026, 11, 30)
    exact_delta[("C", 220.0)] = 0.25
    got = plan(day, book([("C", EXP, 220, -20)], shares=-300), boundary_market(day, 25.0))     # option delta -500
    assert got["hedge"]["target"] == 500
    assert [(o["id"], o["action"], o["qty"]) for o in got["orders"]] == [
        ("nvda-fw-hedge-btc-2026-11-30", "BTC", 300), ("nvda-fw-hedge-buy-2026-11-30", "BUY", 500)]


def test_the_target_is_the_options_delta_rounded(exact_delta):
    day = dt.date(2026, 11, 30)
    exact_delta[("P", 145.0)], exact_delta[("C", 220.0)] = -0.123456, 0.0789
    got = plan(day, book([("P", EXP, 145, -41), ("C", EXP, 220, -59)], shares=37), boundary_market(day, 178.0))
    od = -41 * 100 * -0.123456 + -59 * 100 * 0.0789
    assert got["hedge"]["option_delta"] == pytest.approx(od) and got["hedge"]["net_delta"] == pytest.approx(od + 37)
    assert got["hedge"]["target"] == -round(od) == -41                 # +40.66 shares of option delta
    assert [(o["action"], o["qty"]) for o in got["orders"]] == [("SEL", 37), ("SHT", 41)]


def test_the_band_from_real_deltas_after_the_entry():
    # the wings in the book the night after their fill: deltas from each one's closing mid
    day = dt.date(2026, 11, 25)
    legs = [("P", EXP, 145.0, -41, 0.51), ("C", EXP, 220.0, -59, 0.42)]
    for last, inside in ((None, True), (200.0, False), (150.0, False)):
        c = S.closes(end=day, last=last)
        hist = {(cp, e, k): S.leg_history(c, cp, e, k, vol) for cp, e, k, _, vol in legs}
        mkt = S.market(day, c=c, chain=False)
        got = plan(day, book([(cp, e, k, q) for cp, e, k, q, _ in legs]), mkt, leg_history=hist)
        r, q = W.carry(mkt, pd.Timestamp(day))
        spot = float(c.iloc[-1])
        od = 0.0
        for cp, e, k, qty, _ in legs:
            h = hist[(cp, e, k)].loc[pd.Timestamp(day)]
            od += qty * 100 * W.option_greeks((h["bid"] + h["ask"]) / 2, spot, k, (e - day).days / 365, r, q, cp)["delta"]
        assert got["hedge"]["option_delta"] == pytest.approx(od, rel=1e-9)
        assert got["hedge"]["breached"] is (not inside) and bool(abs(od) * spot > 5000) is (not inside)
        if inside:
            assert got["orders"] == []
        else:
            assert [(o["action"], o["qty"]) for o in got["orders"]] == [("BUY" if od < 0 else "SHT", abs(round(od)))]


def test_a_leg_with_no_mid_tonight_keeps_its_last_iv():
    day = dt.date(2026, 11, 25)
    c = S.closes(end=day, last=200.0)
    hist = S.leg_history(c, "C", EXP, 220.0, 0.42, missing={pd.Timestamp(day)})
    got = plan(day, book([("C", EXP, 220, -59)]), S.market(day, c=c, chain=False),
               leg_history={("C", EXP, 220.0): hist})
    leg = got["hedge"]["legs"][0]
    assert leg["iv_day"] == pd.Timestamp("2026-11-24") and np.isnan(leg["mid"])
    assert leg["iv"] == pytest.approx(0.42, abs=1e-6)
    with pytest.raises(ValueError, match="no usable quote for the 2026-12-18 220 C"):
        plan(day, book([("C", EXP, 220, -59)]), S.market(day, c=c, chain=False), leg_history={})


# --------------------------------------------------------------------------- after the expiry
def test_the_night_of_the_expiry_closes_the_leftover_shares():
    day = EXP                                                 # Sanji has settled the wings: shares only
    for shares, action in ((-180, "BTC"), (4100, "SEL")):
        got = plan(day, book(shares=shares), S.market(day, chain=False))
        assert [(o["id"], o["action"], o["qty"], o["for"]) for o in got["orders"]] == [
            ("nvda-fw-hedge-2026-12-18", action, abs(shares), "2026-12-21")]
        assert got["notes"] == [f"no options held: close the {shares:+d} leftover shares"]
    # and the next session is the next entry
    assert FW.night(dt.date(2026, 12, 21)) == {"entry": True, "expiry": dt.date(2027, 1, 15), "forced": False,
                                               "late": False}


def test_an_option_still_listed_on_its_expiry_stops():
    with pytest.raises(FW.StaleBook, match="expired by 2026-12-18"):
        plan(EXP, book([("P", EXP, 145, -41)]), S.market(EXP, chain=False))
    assert plan(dt.date(2026, 12, 17), book(), None)["orders"] == []


def test_market_data_is_needed_whenever_something_is_held():
    with pytest.raises(ValueError, match="no market data"):
        plan(dt.date(2026, 11, 25), book(shares=10))
    with pytest.raises(ValueError, match="no market data"):
        plan(dt.date(2026, 11, 25), book([("P", EXP, 145, -41)]))
    with pytest.raises(ValueError, match="no market data"):
        plan(S.ENTRY, book(), entry_expiry=EXP)


# --------------------------------------------------------------------------- the hedge limit
def test_the_hedge_limit_is_40_times_the_book(exact_delta):
    assert CFG["limits"]["max_hedge_x_capital"] == 40
    day = dt.date(2026, 11, 30)
    exact_delta[("C", 220.0)] = 1.0
    # -40 calls deep in the money: a 4,000-share hedge; at $1,000 that is exactly 40 x $100,000
    got = plan(day, book([("C", EXP, 220, -40)]), boundary_market(day, 1000.0))
    assert got["orders"][0]["qty"] == 4000
    with pytest.raises(ValueError, match="the hedge target, \\+4000 NVDA shares .* is more than 40 x"):
        plan(day, book([("C", EXP, 220, -40)]), boundary_market(day, 1000.01))


def test_the_limit_checks_the_shares_held_inside_the_band_and_with_no_options(exact_delta):
    day = dt.date(2026, 11, 30)
    exact_delta[("P", 145.0)] = -0.05
    with pytest.raises(ValueError, match="the shares held, -22501 NVDA shares"):
        plan(day, book([("P", EXP, 145, -1)], shares=-22501), boundary_market(day, 178.0))
    with pytest.raises(ValueError, match="the shares held, \\+22501 NVDA shares"):
        plan(day, book(shares=22501), boundary_market(day, 178.0))
    assert plan(day, book(shares=22471), boundary_market(day, 178.0))["orders"][0]["qty"] == 22471
