"""
Whose book it is: every NVDA option held must be exactly what this strategy's own filled orders add up to (their ids
in mkt/fills.json, looked up in data/vol_arb_ledger.json), a short leg excepted when Sanji has assigned it, and
shares need a fill of its own; and the hedge may not pass limits.max_hedge_x_capital times the book. Books, fills and
ledgers here are made up.
"""

from __future__ import annotations

import datetime as dt

import pytest

import va_live_synth as S
from trading_app.lib import vol_arb_live as VL
from test_vol_arb_live_hedge import FOR, DAY, LEGS, book_of, hedge_market, histories

EXP = S.EXPIRY


def opt(cp, k, n, d="2026-11-23", action=None, expiry=EXP):
    """One of this strategy's option orders, as option_orders writes it (one step: open n contracts)."""
    (o,) = VL.option_orders(root="NVDA", conid=S.NVDA, day=d, for_day=VL.next_session(dt.date.fromisoformat(d)),
                            cp=cp, expiry=expiry, strike=k, limit=2.5, held=0, target=n)
    return o


def ledger_of(*orders):
    led = VL.read_ledger(None)
    for o in orders:
        led["orders"][o["id"]] = o
    return led


def filled(*orders, status="filled"):
    return [{"id": o["id"], "status": status} for o in orders]


def book(options=(), shares=0):
    return VL.Book(cash=0.0, shares=shares, options=[{"cp": cp, "expiry": EXP, "strike": float(k), "qty": n}
                                                     for cp, k, n in options])


PUT_RICH = (opt("P", 165.0, -14), opt("C", 195.0, 14))
CALL_RICH = (opt("P", 165.0, 14), opt("C", 195.0, -14))


def test_the_ledger_file():
    assert VL.read_ledger(None) == {"format": "vol_arb_ledger/1", "orders": {}, "nights": {}}
    for bad in ({"format": "vol_arb_ledger/2", "orders": {}, "nights": {}}, {"format": "vol_arb_ledger/1"},
                {"format": "vol_arb_ledger/1", "orders": [], "nights": {}}, [], "x"):
        with pytest.raises(VL.LedgerError):
            VL.read_ledger(bad)
    led = VL.record_night(VL.read_ledger(None), list(PUT_RICH), dt.date(2026, 11, 23), dt.date(2026, 11, 24))
    assert led["orders"] == {o["id"]: o for o in PUT_RICH}
    assert led["nights"] == {"2026-11-23": {"for": "2026-11-24", "ids": [o["id"] for o in PUT_RICH]}}
    # a second run the same night replaces that night's orders by id; an empty night is recorded too
    again = VL.record_night(led, [{**PUT_RICH[0], "lmt_prc": "2.6"}], dt.date(2026, 11, 23), dt.date(2026, 11, 24))
    assert again["orders"][PUT_RICH[0]["id"]]["lmt_prc"] == "2.6" and len(again["orders"]) == 2
    assert again["nights"]["2026-11-23"]["ids"] == [PUT_RICH[0]["id"]]
    assert VL.record_night(again, [], dt.date(2026, 11, 24), dt.date(2026, 11, 25))["nights"]["2026-11-24"] == \
        {"for": "2026-11-25", "ids": []}
    assert led["orders"][PUT_RICH[0]["id"]]["lmt_prc"] == "2.5"            # the ledger passed in is not changed


def test_strategy_options_adds_up_its_own_fills():
    hedge = VL.stock_orders(root="NVDA", conid=S.NVDA, day="2026-11-24", for_day="2026-11-25", held=0, target=-300)
    led = ledger_of(*PUT_RICH, *hedge)
    want = {("P", EXP, 165.0): -14, ("C", EXP, 195.0): 14}
    assert VL.strategy_options(led, filled(*PUT_RICH), "NVDA", DAY) == (want, True)
    # unfilled orders, other strategies' fills and a repeated record add nothing
    fills = filled(*PUT_RICH) + filled(PUT_RICH[0]) + [{"id": "nvda-csp-2026-11-16", "status": "filled"}]
    assert VL.strategy_options(led, fills, "NVDA", DAY) == (want, True)
    assert VL.strategy_options(led, filled(*PUT_RICH, status="unfilled"), "NVDA", DAY) == ({}, False)
    assert VL.strategy_options(led, [{"id": "nvda-csp-2026-11-16", "status": "filled"}], "NVDA", DAY) == ({}, False)
    # a hedge fill counts as a fill but holds no option; legs that expired by D are gone
    assert VL.strategy_options(led, filled(*hedge), "NVDA", DAY) == ({}, True)
    assert VL.strategy_options(led, filled(*PUT_RICH), "NVDA", EXP) == ({}, True)
    assert VL.strategy_options(led, filled(*PUT_RICH), "NVDA", EXP - dt.timedelta(days=1))[0] == want
    # every action's sign: SHT and SEL take away, BUY and BTC add
    close = [{**PUT_RICH[0], "id": "nvda-va-put-btc-2026-12-01", "action": "BTC", "qty": 4},
             {**PUT_RICH[1], "id": "nvda-va-call-sel-2026-12-01", "action": "SEL", "qty": 14}]
    got, _ = VL.strategy_options(ledger_of(*PUT_RICH, *close), filled(*PUT_RICH, *close), "NVDA", DAY)
    assert got == {("P", EXP, 165.0): -10}
    with pytest.raises(VL.LedgerError, match="nvda-va-put-2026-11-23 filled, but data/vol_arb_ledger.json has no"):
        VL.strategy_options(ledger_of(PUT_RICH[1]), filled(*PUT_RICH), "NVDA", DAY)


@pytest.mark.parametrize("name, held, orders, fills, problems", [
    ("this strategy's legs", [("P", 165, -14), ("C", 195, 14)], PUT_RICH, PUT_RICH, []),
    ("the short put assigned", [("C", 195, 14)], PUT_RICH, PUT_RICH, []),
    ("the short call assigned", [("P", 165, 14)], CALL_RICH, CALL_RICH, []),
    ("an unfilled entry, nothing held", [], PUT_RICH, (), []),
    ("the wheel's put filled before the switch", [("P", 155, -6)], (), (),
     ["the book holds -6 2026-12-18 155 put, which no filled nvda-va-* order accounts for"]),
    ("the wheel's put beside the strategy's", [("P", 155, -6), ("P", 165, -14), ("C", 195, 14)], PUT_RICH, PUT_RICH,
     ["the book holds -6 2026-12-18 155 put, which no filled nvda-va-* order accounts for"]),
    ("the wheel's put at the same strike, same side", [("P", 165, -20), ("C", 195, 14)], PUT_RICH, PUT_RICH,
     ["the book holds -20 2026-12-18 165 put, but this strategy's fills add up to -14"]),
    ("the wheel's put netted against the strategy's long put", [("P", 165, 8), ("C", 195, -14)], CALL_RICH, CALL_RICH,
     ["the book holds +8 2026-12-18 165 put, but this strategy's fills add up to +14"]),
    ("contracts read as shares", [("P", 165, -1400), ("C", 195, 1400)], PUT_RICH, PUT_RICH,
     ["the book holds +1400 2026-12-18 195 call, but this strategy's fills add up to +14",
      "the book holds -1400 2026-12-18 165 put, but this strategy's fills add up to -14"]),
    ("the long leg missing", [("P", 165, -14)], PUT_RICH, PUT_RICH,
     ["this strategy's fills hold +14 2026-12-18 195 call, but the book does not list it"]),
    ("a leg held that never filled", [("P", 165, -14), ("C", 195, 14)], PUT_RICH, PUT_RICH[:1],
     ["the book holds +14 2026-12-18 195 call, which no filled nvda-va-* order accounts for"]),
])
def test_book_problems(name, held, orders, fills, problems):
    got = VL.book_problems(book(held), filled(*fills), ledger_of(*orders), "NVDA", DAY)
    assert got == problems, name


def test_shares_need_a_fill_of_the_strategys_own():
    hedge = VL.stock_orders(root="NVDA", conid=S.NVDA, day="2026-11-24", for_day="2026-11-25", held=0, target=-300)
    assert VL.book_problems(book(shares=600), [{"id": "nvda-csp-2026-11-16", "status": "filled"}], ledger_of(),
                            "NVDA", DAY) == ["the book holds +600 NVDA shares, but no nvda-va-* order has filled"]
    assert VL.book_problems(book(shares=-300), filled(*hedge), ledger_of(*hedge), "NVDA", DAY) == []
    # shares delivered by an assignment after this strategy's legs filled
    assert VL.book_problems(book([("C", 195, 14)], shares=1400), filled(*PUT_RICH), ledger_of(*PUT_RICH), "NVDA",
                            DAY) == []


def test_without_a_fills_file_a_book_with_positions_cannot_be_checked():
    assert VL.book_problems(book(), None, ledger_of(), "NVDA", DAY) == []
    for b in (book([("P", 165, -14)]), book(shares=-300)):
        (p,) = VL.book_problems(b, None, ledger_of(*PUT_RICH), "NVDA", DAY)
        assert p.startswith("mkt/positions.json lists NVDA positions but there is no mkt/fills.json")


# --------------------------------------------------------------------------- the hedge's size
def test_a_hedge_bigger_than_the_limit_stops():
    mkt = hedge_market()
    cfg = S.config()
    assert cfg["limits"] == {"max_hedge_x_capital": 5}
    p = VL.plan_night(cfg, DAY, FOR, book_of(LEGS), mkt=mkt, leg_history=histories(mkt, LEGS))
    spot = float(mkt["stock"]["close"].iloc[-1])
    assert 0 < abs(p["hedge"]["target"]) * spot < 1.5 * 100_000
    # the same legs with qty in shares, as a misread positions.json would give them
    big = [(cp, e, k, n * 100, s) for cp, e, k, n, s in LEGS]
    with pytest.raises(ValueError, match="hedge target.*more than 5 x the \\$100,000 book"):
        VL.plan_night(cfg, DAY, FOR, book_of(big), mkt=mkt, leg_history=histories(mkt, big))
    # shares held worth more than 5 x the book, and the limit's edge
    n = int(5 * 100_000 // spot)
    VL.plan_night(cfg, DAY, FOR, book_of(LEGS, shares=-n), mkt=mkt, leg_history=histories(mkt, LEGS))
    with pytest.raises(ValueError, match="shares held"):
        VL.plan_night(cfg, DAY, FOR, book_of(LEGS, shares=-(n + 1)), mkt=mkt, leg_history=histories(mkt, LEGS))
    with pytest.raises(ValueError, match="shares held"):
        VL.plan_night({**cfg, "limits": {"max_hedge_x_capital": 1}}, DAY, FOR, book_of(LEGS, shares=-(n // 4)),
                      mkt=mkt, leg_history=histories(mkt, LEGS))


@pytest.mark.parametrize("limits", [None, {}, {"max_hedge_x_capital": 0}, {"max_hedge_x_capital": "5"},
                                    {"max_hedge_x_capital": float("nan")}])
def test_the_config_needs_a_positive_limit(limits):
    with pytest.raises(ValueError, match="max_hedge_x_capital"):
        VL.check_config({**S.config(), "limits": limits})
