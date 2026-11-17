"""
Whose book it is and whether it is up to date, as vol_arb_live has it, for this strategy's ids (nvda-fw-...): the
ledger, the guards on mkt/open_orders.json (rejected or unanswered orders, acknowledged ones, another strategy's), and
the book reconciliation (every option exactly what the filled orders add up to, a short wing assigned, shares with a
fill of their own, no fills file).
"""

from __future__ import annotations

import datetime as dt

import pytest

import fw_live_synth as S
from trading_app.lib import far_wings_live as FW

EXP = S.EXPIRY
PUT = {"id": "nvda-fw-put-2026-11-23", "for": "2026-11-24", "order_type": "lmt", "action": "SHT", "qty": 41,
       "underlying": S.NVDA, "contract": "put", "expiry": "2026-12-18", "strike": "145", "ref": "open", "lmt_prc": "0.53"}
CALL = {**PUT, "id": "nvda-fw-call-2026-11-23", "contract": "call", "qty": 59, "strike": "220", "lmt_prc": "0.24"}
HEDGE = {"id": "nvda-fw-hedge-2026-11-25", "for": "2026-11-26", "order_type": "mkt", "action": "BUY", "qty": 120,
         "underlying": S.NVDA, "contract": "underlying", "ref": "open"}
WHEEL = {"id": "nvda-csp-2026-11-16", "for": "2026-11-16", "order_type": "lmt", "action": "SHT", "qty": 6,
         "underlying": S.NVDA, "contract": "put", "expiry": "2026-12-18", "strike": "155", "ref": "open", "lmt_prc": "7.78"}
DAY = dt.date(2026, 11, 25)


def ledger(*orders):
    led = FW.read_ledger(None)
    return FW.record_night(led, list(orders), dt.date(2026, 11, 23), dt.date(2026, 11, 24))


def book(options=(), shares=0):
    return FW.Book(cash=1.0, shares=shares,
                   options=[{"cp": cp, "expiry": EXP, "strike": float(k), "qty": q} for cp, k, q in options])


def filled(*ids):
    return [{"id": i, "status": "filled"} for i in ids]


# --------------------------------------------------------------------------- the ledger
def test_the_ledger_records_each_night_and_each_order():
    led = ledger(PUT, CALL)
    assert led["format"] == "far_wings_ledger/1" and set(led["orders"]) == {PUT["id"], CALL["id"]}
    assert led["nights"]["2026-11-23"] == {"for": "2026-11-24", "ids": [PUT["id"], CALL["id"]]}
    assert FW.decided(led, dt.date(2026, 11, 23), dt.date(2026, 11, 24)) == f"{PUT['id']}, {CALL['id']}"
    assert FW.decided(led, dt.date(2026, 11, 23), dt.date(2026, 11, 25)) is None
    empty = FW.record_night(led, [], DAY, dt.date(2026, 11, 26))
    assert FW.decided(empty, DAY, dt.date(2026, 11, 26)) == "no order" and set(empty["orders"]) == set(led["orders"])


@pytest.mark.parametrize("obj", [{"format": "vol_arb_ledger/1", "orders": {}, "nights": {}}, [], {"format":
                                 "far_wings_ledger/1", "orders": [], "nights": {}}])
def test_a_bad_ledger_is_an_error(obj):
    with pytest.raises(FW.LedgerError, match="data/far_wings_ledger.json is not a far_wings_ledger/1 file"):
        FW.read_ledger(obj)


# --------------------------------------------------------------------------- the guards
def test_an_unanswered_order_of_its_own_stops_until_acknowledged():
    stops, notes = FW.guard_problems([PUT, CALL], filled(PUT["id"]), DAY, "NVDA", {})
    assert len(stops) == 1 and "no result yet for nvda-fw-call-2026-11-23" in stops[0] and notes == []
    stops, notes = FW.guard_problems([PUT, CALL], filled(PUT["id"]), DAY, "NVDA", {CALL["id"]: "Sanji skipped it"})
    assert stops == [] and "acknowledged in config/far_wings_live.yaml: Sanji skipped it" in notes[0]
    # before Sanji keeps a fills file, nothing is waited for; an order for a later session is not waited for
    assert FW.guard_problems([PUT, CALL], None, DAY, "NVDA", {}) == ([], [])
    assert FW.guard_problems([PUT, CALL], [], dt.date(2026, 11, 23), "NVDA", {}) == ([], [])


def test_a_rejected_order_of_its_own_stops_until_acknowledged():
    fills = [{"id": PUT["id"], "status": "rejected"}, {"id": CALL["id"], "status": "filled"}]
    stops, _ = FW.guard_problems([PUT, CALL], fills, DAY, "NVDA", {})
    assert len(stops) == 1 and stops[0].startswith("Sanji rejected nvda-fw-put-2026-11-23")
    stops, notes = FW.guard_problems([PUT, CALL], fills, DAY, "NVDA", {PUT["id"]: "strike token fixed"})
    assert stops == [] and "Sanji rejected nvda-fw-put-2026-11-23; acknowledged" in notes[0]


def test_another_strategys_orders_are_only_noted():
    va = {**PUT, "id": "nvda-va-put-2026-11-23"}
    stops, notes = FW.guard_problems([WHEEL, va], [{"id": va["id"], "status": "rejected"}], DAY, "NVDA", {})
    assert stops == []
    assert any("Sanji rejected nvda-va-put-2026-11-23, which is not this strategy's" in n for n in notes)
    assert any("no result for nvda-csp-2026-11-16, which is not this strategy's" in n for n in notes)


# --------------------------------------------------------------------------- the book
def test_the_book_of_both_wings_filled_is_its_own():
    led = ledger(PUT, CALL)
    assert FW.book_problems(book([("P", 145, -41), ("C", 220, -59)]), filled(PUT["id"], CALL["id"]), led, "NVDA",
                            DAY) == []
    # one wing unfilled: the book holds only the other
    assert FW.book_problems(book([("C", 220, -59)]), filled(CALL["id"]) + [{"id": PUT["id"], "status": "unfilled"}],
                            led, "NVDA", DAY) == []


def test_a_short_wing_assigned_is_gone_and_its_shares_are_its_own():
    led = ledger(PUT, CALL)
    assert FW.book_problems(book([("C", 220, -59)], shares=4100), filled(PUT["id"], CALL["id"]), led, "NVDA",
                            DAY) == []


def test_anything_else_in_the_book_stops():
    led = ledger(PUT, CALL)
    fills = filled(PUT["id"], CALL["id"], WHEEL["id"])
    got = FW.book_problems(book([("P", 145, -41), ("C", 220, -59), ("P", 155, -6)]), fills, led, "NVDA", DAY)
    assert got == ["the book holds -6 2026-12-18 155 put, which no filled nvda-fw-* order accounts for"]
    got = FW.book_problems(book([("P", 145, -4100), ("C", 220, -59)]), fills, led, "NVDA", DAY)
    assert got == ["the book holds -4100 2026-12-18 145 put, but this strategy's fills add up to -41"]
    got = FW.book_problems(book(shares=300), filled(WHEEL["id"]), led, "NVDA", DAY)
    assert got == ["the book holds +300 NVDA shares, but no nvda-fw-* order has filled"]
    got = FW.book_problems(book([("P", 145, -41)]), None, led, "NVDA", DAY)
    assert "there is no mkt/fills.json" in got[0]
    assert FW.book_problems(book(), None, led, "NVDA", DAY) == []


def test_a_hedge_fill_lets_the_shares_through_and_a_fill_not_in_the_ledger_stops():
    led = ledger(PUT, CALL, HEDGE)
    assert FW.book_problems(book([("P", 145, -41), ("C", 220, -59)], shares=120),
                            filled(PUT["id"], CALL["id"], HEDGE["id"]), led, "NVDA", dt.date(2026, 11, 26)) == []
    with pytest.raises(FW.LedgerError, match="records nvda-fw-put-2026-12-21 filled, but data/far_wings_ledger.json"):
        FW.book_problems(book(), filled("nvda-fw-put-2026-12-21"), led, "NVDA", DAY)


def test_the_fills_add_up_by_action_and_only_for_live_expiries():
    btc = {**PUT, "id": "nvda-fw-put-btc-2026-11-30", "action": "BTC", "qty": 41}
    led = ledger(PUT, CALL, btc)
    want, any_fill = FW.strategy_options(led, filled(PUT["id"], CALL["id"], btc["id"]), "NVDA", DAY)
    assert want == {("C", EXP, 220.0): -59} and any_fill is True
    want, _ = FW.strategy_options(led, filled(PUT["id"], CALL["id"]), "NVDA", EXP)       # the expiry itself: gone
    assert want == {}
    assert FW.strategy_options(led, [{"id": PUT["id"], "status": "unfilled"}], "NVDA", DAY) == ({}, False)
