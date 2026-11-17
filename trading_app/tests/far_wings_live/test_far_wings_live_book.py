"""
Whose book it is and whether it is up to date, as vol_arb_live has it, for this strategy's ids (nvda-fw-...), name by
name: the ledger (each name's part), the guards on mkt/open_orders.json (an unanswered order stops every name, a
rejected one only its own name; acknowledged ones and another strategy's are noted), and the book reconciliation
(every option exactly what the name's filled orders add up to, a short wing assigned, shares with a fill of their
own, no fills file).
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


TICKERS = [n["ticker"] for n in S.config()["names"]]


def ledger(*orders, ticker="NVDA"):
    """A name's part of the ledger after one night with these orders."""
    led = FW.record_night(FW.read_ledger(None), ticker, list(orders), dt.date(2026, 11, 23), dt.date(2026, 11, 24))
    return FW.name_ledger(led, ticker)


def book(options=(), shares=0):
    return FW.Book(cash=1.0, shares=shares,
                   options=[{"cp": cp, "expiry": EXP, "strike": float(k), "qty": q} for cp, k, q in options])


def filled(*ids):
    return [{"id": i, "status": "filled"} for i in ids]


# --------------------------------------------------------------------------- the ledger
def test_the_ledger_records_each_night_and_each_order_name_by_name():
    full = FW.record_night(FW.read_ledger(None), "NVDA", [PUT, CALL], dt.date(2026, 11, 23), dt.date(2026, 11, 24))
    assert full["format"] == "far_wings_ledger/2" and set(full["names"]) == {"NVDA"}
    led = FW.name_ledger(full, "NVDA")
    assert set(led["orders"]) == {PUT["id"], CALL["id"]}
    assert led["nights"]["2026-11-23"] == {"for": "2026-11-24", "ids": [PUT["id"], CALL["id"]]}
    assert FW.decided(led, dt.date(2026, 11, 23), dt.date(2026, 11, 24)) == f"{PUT['id']}, {CALL['id']}"
    assert FW.decided(led, dt.date(2026, 11, 23), dt.date(2026, 11, 25)) is None
    # another name's night is its own: NVDA's is untouched, and a name with no night yet is undecided
    brk = {**PUT, "id": "brkb-fw-put-2026-11-23", "underlying": 72063691, "strike": "500"}
    both = FW.record_night(full, "BRK B", [brk], dt.date(2026, 11, 23), dt.date(2026, 11, 24))
    assert FW.name_ledger(both, "NVDA") == led and set(FW.name_ledger(both, "BRK B")["orders"]) == {brk["id"]}
    assert FW.decided(FW.name_ledger(both, "AAPL"), dt.date(2026, 11, 23), dt.date(2026, 11, 24)) is None
    assert FW.read_ledger(both) is both
    empty = FW.record_night(both, "NVDA", [], DAY, dt.date(2026, 11, 26))
    mine = FW.name_ledger(empty, "NVDA")
    assert FW.decided(mine, DAY, dt.date(2026, 11, 26)) == "no order" and set(mine["orders"]) == set(led["orders"])
    assert FW.name_ledger(empty, "BRK B") == FW.name_ledger(both, "BRK B")
    assert full["names"]["NVDA"]["nights"] == {"2026-11-23": {"for": "2026-11-24", "ids": [PUT["id"], CALL["id"]]}}


@pytest.mark.parametrize("obj", [{"format": "vol_arb_ledger/1", "orders": {}, "nights": {}}, [],
                                 {"format": "far_wings_ledger/1", "orders": {}, "nights": {}},
                                 {"format": "far_wings_ledger/2", "names": []},
                                 {"format": "far_wings_ledger/2", "names": {"NVDA": {"orders": {}}}},
                                 {"format": "far_wings_ledger/2", "names": {"NVDA": {"orders": [], "nights": {}}}},
                                 # a valid names map under another format string, or none: refused all the same
                                 {"format": "far_wings_ledger/3", "names": {"NVDA": {"orders": {}, "nights": {}}}},
                                 {"names": {"NVDA": {"orders": {}, "nights": {}}}}])
def test_a_bad_ledger_is_an_error(obj):
    with pytest.raises(FW.LedgerError, match="data/far_wings_ledger.json is not a far_wings_ledger/2 file"):
        FW.read_ledger(obj)


# --------------------------------------------------------------------------- the guards
def test_an_unanswered_order_of_any_name_stops_every_name_until_acknowledged():
    stale, rejected, notes = FW.guard_problems([PUT, CALL], filled(PUT["id"]), DAY, TICKERS, {})
    assert len(stale) == 1 and "no result yet for nvda-fw-call-2026-11-23" in stale[0] and notes == []
    assert rejected == {}
    stale, rejected, notes = FW.guard_problems([PUT, CALL], filled(PUT["id"]), DAY, TICKERS,
                                               {CALL["id"]: "Sanji skipped it"})
    assert stale == [] and "acknowledged in config/far_wings_live.yaml: Sanji skipped it" in notes[0]
    # another name's unanswered order is the same: the one book file may be stale for every name
    brk = {**CALL, "id": "brkb-fw-call-2026-11-23", "underlying": 72063691}
    stale, rejected, _ = FW.guard_problems([PUT, brk], filled(PUT["id"]), DAY, TICKERS, {})
    assert len(stale) == 1 and "no result yet for brkb-fw-call-2026-11-23" in stale[0] and rejected == {}
    # before Sanji keeps a fills file, nothing is waited for; an order for a later session is not waited for
    assert FW.guard_problems([PUT, CALL], None, DAY, TICKERS, {}) == ([], {}, [])
    assert FW.guard_problems([PUT, CALL], [], dt.date(2026, 11, 23), TICKERS, {}) == ([], {}, [])


def test_a_rejected_order_of_its_own_stops_that_name_until_acknowledged():
    fills = [{"id": PUT["id"], "status": "rejected"}, {"id": CALL["id"], "status": "filled"}]
    stale, rejected, _ = FW.guard_problems([PUT, CALL], fills, DAY, TICKERS, {})
    assert stale == [] and list(rejected) == ["NVDA"] and len(rejected["NVDA"]) == 1
    assert rejected["NVDA"][0].startswith("Sanji rejected nvda-fw-put-2026-11-23")
    stale, rejected, notes = FW.guard_problems([PUT, CALL], fills, DAY, TICKERS, {PUT["id"]: "strike token fixed"})
    assert stale == [] and rejected == {} and "Sanji rejected nvda-fw-put-2026-11-23; acknowledged" in notes[0]
    brk = {**PUT, "id": "brkb-fw-put-2026-11-23", "underlying": 72063691}
    _, rejected, _ = FW.guard_problems([PUT, brk], filled(PUT["id"]) + [{"id": brk["id"], "status": "rejected"}],
                                       DAY, TICKERS, {})
    assert list(rejected) == ["BRK B"] and rejected["BRK B"][0].startswith("Sanji rejected brkb-fw-put-2026-11-23")


def test_another_strategys_orders_are_only_noted():
    va = {**PUT, "id": "nvda-va-put-2026-11-23"}
    stale, rejected, notes = FW.guard_problems([WHEEL, va], [{"id": va["id"], "status": "rejected"}], DAY, TICKERS,
                                               {})
    assert stale == [] and rejected == {}
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


def test_another_names_book_is_checked_against_its_own_fills_and_named():
    # BRK B's book against BRK B's part of the ledger: NVDA's fills do not account for it, and the messages name it
    brk = {**PUT, "id": "brkb-fw-put-2026-11-23", "underlying": 72063691, "strike": "500", "qty": 3}
    led = ledger(brk, ticker="BRK B")
    assert FW.book_problems(book([("P", 500, -3)]), filled(brk["id"]), led, "BRK B", DAY) == []
    assert FW.book_problems(book(shares=300), filled(PUT["id"]), led, "BRK B", DAY) == [
        "the book holds +300 BRK B shares, but no brkb-fw-* order has filled"]
    assert FW.book_problems(book([("P", 145, -41)]), filled(PUT["id"]), ledger(PUT), "BRK B", DAY) == [
        "the book holds -41 2026-12-18 145 put, which no filled brkb-fw-* order accounts for"]
    got = FW.book_problems(book(shares=10), None, led, "BRK B", DAY)
    assert got[0].startswith("mkt/positions.json lists BRK B positions but there is no mkt/fills.json")


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
