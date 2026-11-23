"""
The live default since 2026-11-23: no hedge (far_wings_live.HEDGE False, config `hedge: false`). Options are held to
expiry with no stock order however far the stock moves, any shares a book holds (an assignment's) are closed at the
next open, a held leg with no usable quote does not stop the name, the entry night is unchanged, and the config must
agree with the code.
"""

from __future__ import annotations

import datetime as dt

import pytest

import fw_live_synth as S
from trading_app.lib import far_wings_live as FW

EXP = S.EXPIRY
DAY = dt.date(2026, 11, 25)
LEGS = [("P", EXP, 145.0, -41, 0.51), ("C", EXP, 220.0, -59, 0.42)]


def book(options=(), shares=0):
    return FW.Book(cash=100_000.0, shares=shares,
                   options=[{"cp": cp, "expiry": e, "strike": float(k), "qty": q} for cp, e, k, q in options])


def plan(day, b, mkt=None, **kw):
    return FW.plan_night(S.config(), day, FW.next_session(day), b, name=S.NVDA_NAME, mkt=mkt, **kw)


def held(last, shares=0, missing=()):
    c = S.closes(end=DAY, last=last)
    hist = {(cp, e, k): S.leg_history(c, cp, e, k, vol, missing=missing) for cp, e, k, _, vol in LEGS}
    return plan(DAY, book([(cp, e, k, q) for cp, e, k, q, _ in LEGS], shares), S.market(DAY, c=c, chain=False),
                leg_history=hist)


def test_the_default_is_no_hedge_and_the_config_agrees():
    assert FW.HEDGE is False and S.config()["hedge"] is False
    FW.check_config(S.config())
    for wrong in (True, None, "false", 0):
        with pytest.raises(ValueError, match="hedge"):
            FW.check_config(S.config(hedge=wrong))


@pytest.mark.parametrize("last", [None, 100.0, 150.0, 200.0, 400.0])
def test_options_held_write_no_stock_order_however_far_the_stock_moves(last):
    # with the registered hedge, 100, 150, 200 and 400 are all outside the $5,000 band and would each write a hedge
    got = held(last)
    assert got["orders"] == []
    assert got["hedge"]["target"] == 0 and got["hedge"]["band"] is None and got["hedge"]["breached"] is None
    assert any(n.startswith("options held, no hedge") and n.endswith("no stock order") for n in got["notes"])


@pytest.mark.parametrize("shares, action", [(300, "SEL"), (-250, "BTC")])
def test_shares_an_assignment_left_are_closed_while_options_are_held(shares, action):
    got = held(150.0, shares=shares)
    assert [(o["action"], o["qty"]) for o in got["orders"]] == [(action, abs(shares))]
    assert all(o["contract"] == "underlying" and o["order_type"] == "mkt" for o in got["orders"])
    assert got["hedge"]["target"] == 0
    assert any(f"close the {shares:+d} shares an assignment left" in n for n in got["notes"])


def test_a_held_leg_with_no_usable_quote_does_not_stop_the_name():
    c = S.closes(end=DAY, last=200.0)
    got = plan(DAY, book([("C", EXP, 220, -59)]), S.market(DAY, c=c, chain=False), leg_history={})
    assert got["orders"] == [] and got["hedge"]["legs"] == []
    assert any(n.startswith("no usable quote for the held 2026-12-18 220 C") for n in got["notes"])


def test_the_night_of_the_expiry_still_closes_the_leftover_shares():
    for shares, action in ((-180, "BTC"), (4100, "SEL")):
        got = plan(EXP, book(shares=shares), S.market(EXP, chain=False))
        assert [(o["action"], o["qty"]) for o in got["orders"]] == [(action, abs(shares))]


def test_the_entry_night_is_unchanged():
    got = plan(S.ENTRY, book(), S.market(), entry_expiry=EXP)
    assert [(o["action"], o["contract"], o["strike"], o["qty"], o["lmt_prc"]) for o in got["orders"]] == [
        ("SHT", "put", "145", 41, "0.53"), ("SHT", "call", "220", 59, "0.24")]
