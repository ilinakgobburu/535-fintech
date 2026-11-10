"""
The wheel, live: the next session's order for the course server (Sanji), from
tonight's book.

The rules are config/wheel.yaml's (the backtest), registered for live use in
config/wheel_live.yaml. This module does no I/O. wheel_orders.py at the repo
root pulls the quotes from LSEG, calls these functions and writes
mkt/open_orders.json.

WHAT IS KNOWN WHEN. The script runs after the close of session D. It knows D's
closing quotes and the book after D (mkt/positions.json and mkt/fills.json,
which Sanji writes). The order is for the next session, `for`, and Sanji fills
it against that session's prices. Delta is computed as in the backtest: implied
vol from each option's own closing mid, and delta from that vol, on day D.

THE NIGHT'S DECISION.
    a short option is open    no order. It is held to expiry, as in the backtest.
    nothing open, shares      sell calls, one per 100 shares, at the strike whose
                              delta is closest to the call delta; with the floor,
                              only strikes at or above the cost basis
    nothing open, cash only   sell cash-secured puts at the strike whose |delta|
                              is closest to the put delta, as many as the cash
                              secures (contracts x 100 x strike)

WHICH EXPIRY. As in the backtest, each monthly expiry opens a cycle on the next
session, which sells the next monthly. Live, the entry is also placed on later
nights of the same cycle if it has not filled, as long as the expiry is still
`min_dte` to `max_dte` days after `for` (the backtest's cycles ran 24 to 35).
The first entry, before anything has filled, may sell the cycle after the
current one up to `first_entry_until`. That one exception was chosen on
2026-11-10 so the wheel starts on Nov 11 (Dec 18 expiry) rather than Nov 23.

A STALE BOOK. order.html: "If you run first, you are planning off a stale
book", and Sanji does not check buying power. A plan made before Sanji has
recorded the last order's result could open a second position. So once Sanji
keeps mkt/fills.json, every order this script wrote for a session up to D must
have a result in it before anything new is planned (unresolved_orders), and an
option still listed after its expiry also means the book is not up to date.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass, field

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday,
)

from .pmcc_daily import monthly_expiries

CONTRACT = 100                                      # shares per option contract
OPEN, CLOSE, EARLY_CLOSE = dt.time(9, 30), dt.time(16, 0), dt.time(13, 0)   # New York time


# --------------------------------------------------------------------------- calendar
class NYSEHolidays(AbstractHolidayCalendar):
    """
    The NYSE's full-day closures. New Year's Day falling on a Saturday is not
    moved to the Friday (the NYSE stays open on Dec 31). One-off closures, such as
    a national day of mourning, are not in the rules; Sanji's calendar
    (exchange_calendars) has them.
    """
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def sessions(start, end) -> pd.DatetimeIndex:
    """NYSE sessions in [start, end]: weekdays that are not holidays."""
    days = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
    return days.difference(NYSEHolidays().holidays(days.min(), days.max())) if len(days) else days


def close_time(day: dt.date) -> dt.time:
    """
    13:00 on the NYSE's early-close days: the day after Thanksgiving, and Jul 3 and
    Dec 24 when they fall Monday to Thursday. 16:00 otherwise.
    """
    thanksgiving = USThanksgivingDay.dates(f"{day.year}-11-01", f"{day.year}-11-30")[0].date()
    if day == thanksgiving + dt.timedelta(days=1):
        return EARLY_CLOSE
    if (day.month, day.day) in ((7, 3), (12, 24)) and day.weekday() <= 3:
        return EARLY_CLOSE
    return CLOSE


def decision_dates(now: dt.datetime) -> tuple[dt.date, dt.date]:
    """
    (D, for) for a run at `now`, New York time. D is the last session whose close
    is at or before now, so its closing quotes are final. `for` is the first
    session whose open is after now, so the order cannot be late for it.
    Tuesday 21:00 gives (Tuesday, Wednesday), Tuesday 10:00 gives (Monday,
    Wednesday) and Tuesday 08:00 gives (Monday, Tuesday).
    """
    days = [d.date() for d in sessions(now.date() - dt.timedelta(days=14), now.date() + dt.timedelta(days=14))]
    closed = [d for d in days if dt.datetime.combine(d, close_time(d)) <= now]
    upcoming = [d for d in days if dt.datetime.combine(d, OPEN) > now]
    return closed[-1], upcoming[0]


def target_expiry(for_day: dt.date, min_dte: int, max_dte: int, first_entry: bool) -> dt.date | None:
    """
    The expiry to sell for an entry on `for_day`, or None to wait. The target is
    the nearest monthly expiry at least `min_dte` days after `for_day`. It is used
    if it is at most `max_dte` days away and its cycle has begun, that is the
    monthly expiry before it has already passed. With `first_entry`, neither
    condition applies.
    """
    lo, hi = for_day - dt.timedelta(days=45), for_day + dt.timedelta(days=120)
    expiries = [e.date() for e in monthly_expiries(sessions(lo, hi), lo, hi)]
    for i, e in enumerate(expiries):
        dte = (e - for_day).days
        if dte >= min_dte:
            if first_entry:
                return e
            cycle_begun = i > 0 and expiries[i - 1] < for_day
            return e if (dte <= max_dte and cycle_begun) else None
    return None


def first_entry_applies(nothing_filled: bool, for_day: dt.date, until: dt.date | None) -> bool:
    """The first-entry exception: nothing has filled yet and `for_day` is not after `until`."""
    return nothing_filled and until is not None and for_day <= until


# --------------------------------------------------------------------------- the book
class BookFormatError(ValueError):
    """mkt/positions.json or mkt/fills.json is not in the shape this script reads."""


@dataclass
class Book:
    cash: float
    shares: int = 0
    cost_basis: float | None = None          # per share, for the floor
    open_options: list[dict] = field(default_factory=list)
    first_entry: bool = True                 # nothing has ever filled

    @property
    def phase(self) -> str:
        """'hold' while a short option is open, else 'call' with shares, else 'put'."""
        if any(o["qty"] < 0 for o in self.open_options):
            return "hold"
        return "call" if self.shares >= CONTRACT else "put"


def read_book(positions: dict | None, fills: list | None, conid: int, start_cash: float) -> Book:
    """
    The book for one underlying. mkt/positions.json's format is not published
    yet (order.html only says Sanji writes it), so this reads the shape below and
    refuses anything else rather than guessing:

        {"cash": 98765.43,
         "positions": [
           {"underlying": 4815747, "contract": "underlying", "qty": 600, "cost_basis": 161.20},
           {"underlying": 4815747, "contract": "put", "expiry": "2026-12-18",
            "strike": 165, "qty": -6}]}

    qty < 0 is short. With no positions file (Sanji has not written one yet) the
    book is `start_cash` in cash, as the backtest's was. fills is mkt/fills.json,
    a list of {"id", "status", ...}; the first entry is when nothing in it has
    status "filled".
    """
    if fills is not None and not isinstance(fills, list):
        raise BookFormatError("mkt/fills.json should be a JSON list")
    filled = any(isinstance(f, dict) and str(f.get("status", "")).lower() == "filled" for f in fills or [])

    if positions is None:
        if filled:
            raise BookFormatError("mkt/fills.json records a fill but there is no mkt/positions.json")
        return Book(cash=float(start_cash), first_entry=True)
    if not isinstance(positions, dict) or "cash" not in positions or not isinstance(positions.get("positions"), list):
        raise BookFormatError('mkt/positions.json should look like {"cash": ..., "positions": [...]}')

    book = Book(cash=float(positions["cash"]), first_entry=not filled)
    for p in positions["positions"]:
        if not isinstance(p, dict) or type(p.get("underlying")) is not int:
            raise BookFormatError(f"a position without an integer conid in 'underlying': {p}")
        if p["underlying"] != conid:
            continue
        kind, qty = p.get("contract"), p.get("qty")
        if type(qty) is not int:
            raise BookFormatError(f"position without an integer qty: {p}")
        if kind == "underlying":
            book.shares += qty
            if "cost_basis" in p:
                book.cost_basis = float(p["cost_basis"])
        elif kind in ("put", "call"):
            if not _DATE.match(str(p.get("expiry", ""))):
                raise BookFormatError(f"an option position without a YYYY-MM-DD expiry: {p}")
            if qty != 0:
                book.open_options.append(p)
        else:
            raise BookFormatError(f"unknown contract {kind!r} in {p}")
    if book.shares or book.open_options:
        book.first_entry = False
    if book.shares < 0:
        raise BookFormatError("short shares: the wheel never shorts the stock")
    return book


def expired_options(book: Book, day: dt.date) -> list[dict]:
    """Options still in the book whose expiry is on or before `day`: Sanji has not applied the expiry yet."""
    return [o for o in book.open_options if dt.date.fromisoformat(o["expiry"]) <= day]


def unresolved_orders(written: list[dict], fills: list | None, day: dt.date) -> list[str]:
    """
    Ids this script wrote for sessions up to `day` that have no result in
    mkt/fills.json. Before Sanji keeps a fills file at all (fills is None), there
    is nothing to wait for.
    """
    if fills is None:
        return []
    done = {f.get("id") for f in fills if isinstance(f, dict)}
    return [o["id"] for o in written if o.get("for", "9999") <= day.isoformat() and o.get("id") not in done]


def rejected_orders(written: list[dict], fills: list | None) -> list[dict]:
    """Results of this script's orders that Sanji rejected: the order itself was not legal."""
    ids = {o.get("id") for o in written}
    return [f for f in (fills or []) if isinstance(f, dict) and f.get("id") in ids
            and str(f.get("status", "")).lower() == "rejected"]


def merge_orders(written: list[dict], new: list[dict], day: dt.date, for_day: dt.date) -> tuple[list[dict], str]:
    """
    The orders to leave in mkt/open_orders.json. Orders for sessions up to D are
    dropped (Sanji has resolved them, or is not resolving at all). Tonight's
    decision replaces anything written for `for` or later. An order for a session
    between D and `for` (the session in progress, from a run during market hours)
    is kept, and then no new order is added, so two entries can never be live at once.
    """
    in_progress = [o for o in written if day.isoformat() < o.get("for", "") < for_day.isoformat()]
    if in_progress:
        ids = ", ".join(o["id"] for o in in_progress)
        return in_progress, f"kept {ids}: its session has not been resolved yet, so no new order"
    return list(new), ""


def put_contracts(cash: float, strike: float) -> int:
    """As many puts as the cash already in the account secures at the strike."""
    return int(math.floor(cash / (CONTRACT * strike) + 1e-9))


# --------------------------------------------------------------------------- the order
ORDER_FIELDS = {"id", "for", "order_type", "action", "qty", "underlying", "contract",
                "expiry", "strike", "ref", "lmt_prc"}
_NUMBER = re.compile(r"^\d+(\.\d+)?$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def number_token(x: float) -> str:
    """
    A bare-number token: 165.0 -> '165', 167.5 -> '167.5', 4.351 -> '4.35'.
    Written as a JSON string, like every token in order.html's examples; whether
    Sanji wants bare numbers as strings or as JSON numbers is a question for the
    instructor.
    """
    return f"{round(float(x), 2):.2f}".rstrip("0").rstrip(".")


def build_order(*, root: str, conid: int, for_day: dt.date, cp: str, qty: int,
                expiry: dt.date, strike: float, limit: float) -> dict:
    """One sell-to-open limit order in order.html's shape. cp is 'P' or 'C'."""
    order = {
        "id": f"{root.lower()}-{'csp' if cp == 'P' else 'cc'}-{for_day.isoformat()}",
        "for": for_day.isoformat(),
        "order_type": "lmt",
        "action": "SHT",
        "qty": int(qty),
        "underlying": int(conid),
        "contract": "put" if cp == "P" else "call",
        "expiry": expiry.isoformat(),
        "strike": number_token(strike),
        "ref": "open",
        "lmt_prc": number_token(limit),
    }
    validate_order(order)
    return order


def validate_order(o: dict) -> None:
    """
    The checks order.html says Sanji applies, so a bad order fails here, in the
    Action's log, rather than being rejected on the server. Only the forms this
    script writes are accepted: bare-number strikes and limits, dated expiries.
    """
    unknown = set(o) - ORDER_FIELDS
    if unknown:
        raise ValueError(f"unknown fields {sorted(unknown)}")
    for k in ("id", "for", "order_type", "action", "qty", "underlying", "contract"):
        if k not in o or o[k] in (None, ""):
            raise ValueError(f"missing {k}")
    if not _DATE.match(o["for"]):
        raise ValueError("for must be YYYY-MM-DD")
    if o["order_type"] not in ("mkt", "lmt"):
        raise ValueError("order_type must be mkt or lmt")
    if o["action"] not in ("BUY", "SEL", "SHT", "BTC"):
        raise ValueError("action must be BUY, SEL, SHT or BTC")
    if type(o["qty"]) is not int or o["qty"] < 1:
        raise ValueError("qty must be an integer of at least 1")
    if type(o["underlying"]) is not int:
        raise ValueError("underlying must be the integer conid")
    if o["contract"] not in ("put", "call", "underlying"):
        raise ValueError("contract must be put, call or underlying")
    if o["contract"] in ("put", "call"):
        if not _DATE.match(str(o.get("expiry", ""))):
            raise ValueError("an option needs a dated expiry")
        if not _NUMBER.match(str(o.get("strike", ""))):
            raise ValueError("an option needs a bare-number strike")
    elif "expiry" in o or "strike" in o:
        raise ValueError("omit expiry and strike for the underlying")
    if o.get("ref", "open") not in ("open", "prev_close"):
        raise ValueError("ref must be open or prev_close")
    if o["order_type"] == "lmt":
        if not _NUMBER.match(str(o.get("lmt_prc", ""))) or float(o["lmt_prc"]) <= 0:
            raise ValueError("a limit order needs a positive bare-number lmt_prc")
    elif "lmt_prc" in o:
        raise ValueError("lmt_prc is forbidden on a market order")
