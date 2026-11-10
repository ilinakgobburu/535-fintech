"""
Tests for lib/wheel_live.py, the nightly order for the course server.

Every answer here is known by construction rather than by running the code
and blessing its output: the NYSE's published holidays, dates counted on a
calendar, Black-Scholes prices made from a known vol so the right strike is
known before select() runs, and order.html's rules for a legal order.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from trading_app.lib import wheel_live as L
from trading_app.lib.vol import bs_greeks, bs_price
from trading_app.lib.wheel import CALL, PUT, select

D = dt.date


# --------------------------------------------------------------------------- calendar
NYSE_2026 = [D(2026, 1, 1), D(2026, 1, 19), D(2026, 2, 16), D(2026, 4, 3), D(2026, 5, 25),
             D(2026, 6, 19), D(2026, 7, 3), D(2026, 9, 7), D(2026, 11, 26), D(2026, 12, 25)]
NYSE_2027 = [D(2027, 1, 1), D(2027, 1, 18), D(2027, 2, 15), D(2027, 3, 26), D(2027, 5, 31),
             D(2027, 6, 18), D(2027, 7, 5), D(2027, 9, 6), D(2027, 11, 25), D(2027, 12, 24)]


@pytest.mark.parametrize("year, expected", [(2026, NYSE_2026), (2027, NYSE_2027)])
def test_holidays_are_the_nyse_list(year, expected):
    days = pd.bdate_range(f"{year}-01-01", f"{year}-12-31")
    closed = sorted(set(d.date() for d in days) - set(d.date() for d in L.sessions(days[0], days[-1])))
    assert closed == expected


def test_new_year_on_a_saturday_is_not_moved_to_friday():
    # 2028-01-01 is a Saturday; the NYSE trades on Friday 2027-12-31.
    assert D(2027, 12, 31) in [d.date() for d in L.sessions("2027-12-27", "2028-01-07")]


@pytest.mark.parametrize("now, expected", [
    (dt.datetime(2026, 11, 10, 21, 0), (D(2026, 11, 10), D(2026, 11, 11))),   # the nightly run
    (dt.datetime(2026, 11, 10, 16, 0), (D(2026, 11, 10), D(2026, 11, 11))),   # at the close
    (dt.datetime(2026, 11, 10, 10, 0), (D(2026, 11, 9), D(2026, 11, 11))),    # mid-session: today's close not final
    (dt.datetime(2026, 11, 10, 9, 30), (D(2026, 11, 9), D(2026, 11, 11))),    # at the open: too late for today
    (dt.datetime(2026, 11, 10, 8, 0), (D(2026, 11, 9), D(2026, 11, 10))),     # before the open
    (dt.datetime(2026, 11, 14, 12, 0), (D(2026, 11, 13), D(2026, 11, 16))),   # Saturday
    (dt.datetime(2026, 11, 25, 21, 0), (D(2026, 11, 25), D(2026, 11, 27))),   # Thanksgiving next day
    (dt.datetime(2026, 7, 2, 21, 0), (D(2026, 7, 2), D(2026, 7, 6))),         # Jul 3 observed, then a weekend
    (dt.datetime(2026, 11, 27, 14, 0), (D(2026, 11, 27), D(2026, 11, 30))),   # early close at 13:00
    (dt.datetime(2026, 11, 27, 12, 0), (D(2026, 11, 25), D(2026, 11, 30))),   # before that early close
    (dt.datetime(2026, 12, 24, 14, 0), (D(2026, 12, 24), D(2026, 12, 28))),   # Christmas Eve, a Thursday
])
def test_decision_dates(now, expected):
    assert L.decision_dates(now) == expected


@pytest.mark.parametrize("day, early", [
    (D(2026, 11, 27), True), (D(2027, 11, 26), True),     # the day after Thanksgiving
    (D(2026, 12, 24), True), (D(2025, 12, 24), True),     # Dec 24 on a Thursday, a Wednesday
    (D(2025, 7, 3), True), (D(2024, 7, 3), True),         # Jul 3 on a Thursday, a Wednesday
    (D(2026, 11, 25), False), (D(2026, 7, 2), False),     # ordinary sessions
])
def test_early_closes(day, early):
    assert (L.close_time(day) == L.EARLY_CLOSE) is early


@pytest.mark.parametrize("for_day, first, expected", [
    (D(2026, 11, 11), True, D(2026, 12, 18)),    # tonight: the first entry, 37 days out
    (D(2026, 11, 11), False, None),              # Nov 20 is 9 days out, Dec 18's cycle has not begun: wait
    (D(2026, 11, 20), False, None),              # Dec 18 is 28 days out but its cycle opens on Nov 23
    (D(2026, 11, 23), False, D(2026, 12, 18)),   # first session after Nov 20: 25 days
    (D(2026, 11, 24), False, D(2026, 12, 18)),   # 24 days: still in the window
    (D(2026, 11, 25), False, None),              # 23 days is too short and Jan 15 is 51: wait
    (D(2026, 12, 11), False, None),              # Jan 15 is 35 days out, but before Dec 18 its cycle has not begun
    (D(2026, 12, 21), False, D(2027, 1, 15)),    # first session after Dec 18: 25 days
    (D(2025, 3, 24), False, D(2025, 4, 17)),     # April 2025's third Friday was Good Friday
])
def test_target_expiry(for_day, first, expected):
    assert L.target_expiry(for_day, 24, 35, first_entry=first) == expected


@pytest.mark.parametrize("nothing_filled, for_day, applies", [
    (True, D(2026, 11, 11), True),
    (True, D(2026, 11, 20), True),
    (True, D(2026, 11, 23), False),      # from Nov 23 the normal rule, even with nothing filled
    (False, D(2026, 11, 11), False),     # once anything has filled, never
])
def test_the_first_entry_exception_ends_on_nov_20(nothing_filled, for_day, applies):
    assert L.first_entry_applies(nothing_filled, for_day, D(2026, 11, 20)) is applies
    assert L.first_entry_applies(nothing_filled, for_day, None) is False


# --------------------------------------------------------------------------- the book
NVDA = 4815747


def test_no_positions_file_is_the_starting_cash():
    book = L.read_book(None, None, NVDA, 100_000)
    assert (book.cash, book.shares, book.open_options, book.first_entry, book.phase) == \
        (100_000.0, 0, [], True, "put")


def test_a_fill_ends_the_first_entry():
    cash_only = {"cash": 100_000, "positions": []}     # e.g. after a put expired worthless
    book = L.read_book(cash_only, [{"id": "x", "status": "unfilled"}, {"id": "y", "status": "filled"}], NVDA, 1e5)
    assert book.first_entry is False
    assert L.read_book(None, [{"id": "x", "status": "unfilled"}], NVDA, 1e5).first_entry is True


def test_a_short_put_is_held():
    pos = {"cash": 100_000, "positions": [
        {"underlying": NVDA, "contract": "put", "expiry": "2026-12-18", "strike": 170, "qty": -5}]}
    book = L.read_book(pos, None, NVDA, 1e5)
    assert book.phase == "hold" and book.first_entry is False


def test_assigned_shares_sell_calls_with_their_basis():
    pos = {"cash": 15_000, "positions": [
        {"underlying": NVDA, "contract": "underlying", "qty": 500, "cost_basis": 161.27},
        {"underlying": 999, "contract": "put", "expiry": "2026-12-18", "strike": 50, "qty": -1}]}
    book = L.read_book(pos, None, NVDA, 1e5)
    assert (book.phase, book.shares, book.cost_basis, book.open_options) == ("call", 500, 161.27, [])


def test_fewer_than_100_shares_is_not_a_call_phase():
    pos = {"cash": 90_000, "positions": [{"underlying": NVDA, "contract": "underlying", "qty": 50}]}
    assert L.read_book(pos, None, NVDA, 1e5).phase == "put"


@pytest.mark.parametrize("positions, fills", [
    ([], None),                                                         # not an object
    ({"positions": []}, None),                                          # no cash
    ({"cash": 1, "positions": {}}, None),                               # positions not a list
    ({"cash": 1, "positions": [{"underlying": NVDA, "contract": "put", "qty": "-1"}]}, None),
    ({"cash": 1, "positions": [{"underlying": NVDA, "contract": "future", "qty": 1}]}, None),
    ({"cash": 1, "positions": [{"underlying": NVDA, "contract": "underlying", "qty": -100}]}, None),
    (None, {"status": "filled"}),                                       # fills not a list
    ({"cash": 1, "positions": [{"underlying": "4815747", "contract": "underlying", "qty": 500}]}, None),
    ({"cash": 1, "positions": [{"conid": NVDA, "contract": "underlying", "qty": 500}]}, None),
    ({"cash": 1, "positions": [{"underlying": NVDA, "contract": "put", "strike": 170, "qty": -5}]}, None),
    (None, [{"id": "nvda-csp-2026-11-11", "status": "filled"}]),        # a fill but no positions file
])
def test_an_unexpected_book_stops_the_script(positions, fills):
    with pytest.raises(L.BookFormatError):
        L.read_book(positions, fills, NVDA, 1e5)


def test_an_option_past_its_expiry_means_the_book_is_stale():
    pos = {"cash": 15_000, "positions": [
        {"underlying": NVDA, "contract": "put", "expiry": "2026-12-18", "strike": 170, "qty": -5}]}
    book = L.read_book(pos, None, NVDA, 1e5)
    assert L.expired_options(book, D(2026, 12, 17)) == []
    assert len(L.expired_options(book, D(2026, 12, 18))) == 1      # expired at that close
    assert len(L.expired_options(book, D(2026, 12, 21))) == 1


# --------------------------------------------------------------------------- results and the file
WRITTEN = [{"id": "nvda-csp-2026-11-11", "for": "2026-11-11"}]


def test_no_fills_file_means_nothing_to_wait_for():
    assert L.unresolved_orders(WRITTEN, None, D(2026, 11, 11)) == []


def test_an_order_without_a_result_stops_the_plan():
    assert L.unresolved_orders(WRITTEN, [], D(2026, 11, 11)) == ["nvda-csp-2026-11-11"]
    assert L.unresolved_orders(WRITTEN, [{"id": "nvda-csp-2026-11-11", "status": "unfilled"}], D(2026, 11, 11)) == []
    assert L.unresolved_orders(WRITTEN, [], D(2026, 11, 10)) == []      # its session has not closed yet


def test_rejections_are_found():
    fills = [{"id": "nvda-csp-2026-11-11", "status": "rejected", "reason": "unknown token"},
             {"id": "someone-else", "status": "rejected"}]
    assert [f["id"] for f in L.rejected_orders(WRITTEN, fills)] == ["nvda-csp-2026-11-11"]


TONIGHT = {"id": "nvda-csp-2026-11-12", "for": "2026-11-12"}


@pytest.mark.parametrize("written, day, for_day, expected", [
    ([], D(2026, 11, 11), D(2026, 11, 12), [TONIGHT]),
    # the 21:00 run: last night's order (for D) is dropped, tonight's goes in
    ([{"id": "nvda-csp-2026-11-11", "for": "2026-11-11"}], D(2026, 11, 11), D(2026, 11, 12), [TONIGHT]),
    # a second run the same night replaces the order with the same session
    ([{"id": "nvda-csp-2026-11-12", "for": "2026-11-12", "qty": 4}], D(2026, 11, 11), D(2026, 11, 12), [TONIGHT]),
    # a run at 14:00 on Nov 12 (D = Nov 11): Nov 12's order is still to be resolved, so it stays alone
    ([{"id": "nvda-csp-2026-11-12", "for": "2026-11-12"}], D(2026, 11, 11), D(2026, 11, 13),
     [{"id": "nvda-csp-2026-11-12", "for": "2026-11-12"}]),
])
def test_merge_orders(written, day, for_day, expected):
    new = [TONIGHT] if for_day == D(2026, 11, 12) else [{"id": "nvda-csp-2026-11-13", "for": "2026-11-13"}]
    assert L.merge_orders(written, new, day, for_day)[0] == expected


@pytest.mark.parametrize("cash, strike, n", [(100_000, 170, 5), (85_000, 170, 5), (84_999, 170, 4), (10_000, 170, 0)])
def test_put_contracts_are_what_the_cash_secures(cash, strike, n):
    assert L.put_contracts(cash, strike) == n


# --------------------------------------------------------------------------- the order
@pytest.mark.parametrize("x, token", [(170.0, "170"), (167.5, "167.5"), (4.351, "4.35"), (4.0, "4"), (0.5, "0.5")])
def test_number_token(x, token):
    assert L.number_token(x) == token


def order(**over):
    base = dict(root="NVDA", conid=NVDA, for_day=D(2026, 11, 11), cp="P", qty=5,
                expiry=D(2026, 12, 18), strike=170.0, limit=8.725001)
    base.update(over)
    return L.build_order(**base)


def test_the_put_order_is_order_html_shaped():
    assert order() == {
        "id": "nvda-csp-2026-11-11", "for": "2026-11-11", "order_type": "lmt", "action": "SHT",
        "qty": 5, "underlying": NVDA, "contract": "put", "expiry": "2026-12-18",
        "strike": "170", "ref": "open", "lmt_prc": "8.73",
    }


def test_the_call_order():
    o = order(cp="C", qty=5, strike=182.5, limit=3.1)
    assert (o["id"], o["contract"], o["strike"], o["lmt_prc"]) == ("nvda-cc-2026-11-11", "call", "182.5", "3.1")


@pytest.mark.parametrize("change", [
    {"extra": 1},                       # unknown field
    {"qty": 0},                         # quantity below 1
    {"qty": 1.0},                       # not an integer
    {"underlying": "NVDA"},             # a ticker, not the conid
    {"for": "next"},                    # 'Not "next"'
    {"strike": "otm:1"},                # a token this script never writes
    {"lmt_prc": "0"},                   # not a positive price
    {"order_type": "mkt"},              # lmt_prc is forbidden on a market order
    {"contract": ""},                   # blank contract
    {"action": "SELL"},                 # not one of BUY SEL SHT BTC
    {"ref": "close"},                   # only open and prev_close
])
def test_illegal_orders_are_refused(change):
    o = {**order(), **change}
    with pytest.raises(ValueError):
        L.validate_order(o)


def test_a_limit_order_needs_its_price_and_an_option_its_expiry():
    for k in ("lmt_prc", "expiry", "strike"):
        o = order()
        del o[k]
        with pytest.raises(ValueError):
            L.validate_order(o)


# --------------------------------------------------------------------------- strike choice, end to end
def synthetic_quotes(cp, spot=100.0, sigma=0.5, r=0.04, days=30, strikes=np.arange(70, 132.5, 2.5)):
    """Mids priced by Black-Scholes at one vol, in the frame wheel_orders.py builds."""
    day, expiry = pd.Timestamp("2026-11-09"), pd.Timestamp("2026-11-09") + pd.Timedelta(days=days)
    T, F, Dc = days / 365.0, spot * np.exp(r * days / 365.0), np.exp(-r * days / 365.0)
    rows = [{"cp": cp, "expiry": expiry, "strike": float(k), "mid": float(bs_price(F, k, T, sigma, Dc, cp))}
            for k in strikes]
    deltas = {float(k): bs_greeks(F, k, T, sigma, Dc, cp)["delta"] * F / spot for k in strikes}
    return pd.DataFrame(rows), day, expiry, deltas


@pytest.mark.parametrize("cp", [PUT, CALL])
def test_select_picks_the_strike_nearest_the_target_delta(cp):
    quotes, day, expiry, deltas = synthetic_quotes(cp)
    best = min(deltas, key=lambda k: abs(abs(deltas[k]) - 0.35))
    pick = select(quotes, cp, expiry, 100.0, day, 0.04, 0.0, 0.35)
    assert float(pick["strike"]) == best
    assert abs(float(pick["iv"]) - 0.5) < 1e-6          # the vol it was priced at comes back


def test_the_floor_keeps_calls_at_or_above_the_basis():
    quotes, day, expiry, _ = synthetic_quotes(CALL)
    pick = select(quotes, CALL, expiry, 100.0, day, 0.04, 0.0, 0.35, floor=111.0)
    assert float(pick["strike"]) == 112.5                # the nearest-delta strike is lower, but below the basis
    assert select(quotes, CALL, expiry, 100.0, day, 0.04, 0.0, 0.35, floor=500.0) is None
