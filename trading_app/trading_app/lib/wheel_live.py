"""
The wheel, live: the next session's order for the course server (Sanji), from
tonight's book.

The rules are config/wheel.yaml's (the backtest), registered for live use in
config/wheel_live.yaml. This module does no I/O. wheel_orders.py at the repo
root pulls the quotes from LSEG, calls these functions and writes
mkt/open_orders.json.

WHAT IS KNOWN WHEN. The script runs after the close of session D. It knows D's
closing quotes and the book after D (mkt/positions.json, which Sanji writes).
The order is for the next session, `for`, and Sanji fills it against that
session's prices. Delta is computed as in the backtest: implied vol from each
option's own closing mid, and delta from that vol, on day D.

THE NIGHT'S DECISION.
    a short option is open    no order. It is held to expiry, as in the backtest.
    nothing open, shares      sell calls, one per 100 shares, at the strike whose
                              delta is closest to the call delta; with the floor,
                              only strikes at or above the cost basis
    nothing open, cash only   sell cash-secured puts at the strike whose |delta|
                              is closest to the put delta, as many as the cash
                              secures (contracts x 100 x strike)

WHICH EXPIRY. The nearest monthly expiry at least `min_dte` calendar days after
`for`. The order is written only if that expiry is at most `max_dte` days away
(the backtest's cycles ran 24 to 35 days). So an order goes in on the first
sessions after each monthly expiry and is placed again the next night if it
did not fill. The first entry, before anything has ever filled, may go further
out. That one exception was chosen on 2026-11-10 so the wheel starts now
rather than on Nov 23.
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

CONTRACT = 100                       # shares per option contract
OPEN, CLOSE = dt.time(9, 30), dt.time(16, 0)   # NYSE regular session, New York time


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


def decision_dates(now: dt.datetime) -> tuple[dt.date, dt.date]:
    """
    (D, for) for a run at `now`, New York time. D is the last session whose close
    is at or before now, so its closing quotes are final. `for` is the first
    session whose open is after now, so the order cannot be late for it.
    Tuesday 21:00 gives (Tuesday, Wednesday), Tuesday 10:00 gives (Monday,
    Wednesday) and Tuesday 08:00 gives (Monday, Tuesday).
    """
    days = sessions(now.date() - dt.timedelta(days=14), now.date() + dt.timedelta(days=14))
    closed = [d.date() for d in days if dt.datetime.combine(d.date(), CLOSE) <= now]
    upcoming = [d.date() for d in days if dt.datetime.combine(d.date(), OPEN) > now]
    return closed[-1], upcoming[0]


def target_expiry(for_day: dt.date, min_dte: int, max_dte: int, first_entry: bool) -> dt.date | None:
    """
    The nearest monthly expiry at least `min_dte` days after `for_day`, or None
    when it is more than `max_dte` days away (wait for the next cycle). With
    `first_entry`, the max is not applied.
    """
    days = sessions(for_day, for_day + dt.timedelta(days=120))
    for e in monthly_expiries(days, for_day, for_day + dt.timedelta(days=120)):
        dte = (e.date() - for_day).days
        if dte >= min_dte:
            return e.date() if (first_entry or dte <= max_dte) else None
    return None


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
    book is `start_cash` in cash, as the backtest's was. fills is mkt/fills.json
    (a list); the first entry is when nothing in it has status "filled".
    """
    filled = False
    if fills is not None:
        if not isinstance(fills, list):
            raise BookFormatError("mkt/fills.json should be a JSON list")
        filled = any(isinstance(f, dict) and str(f.get("status", "")).lower() == "filled" for f in fills)

    if positions is None:
        return Book(cash=float(start_cash), first_entry=not filled)
    if not isinstance(positions, dict) or "cash" not in positions or not isinstance(positions.get("positions"), list):
        raise BookFormatError('mkt/positions.json should look like {"cash": ..., "positions": [...]}')

    book = Book(cash=float(positions["cash"]), first_entry=not filled)
    for p in positions["positions"]:
        if not isinstance(p, dict) or p.get("underlying") != conid:
            continue
        kind, qty = p.get("contract"), p.get("qty")
        if not isinstance(qty, int):
            raise BookFormatError(f"position without an integer qty: {p}")
        if kind == "underlying":
            book.shares += qty
            if "cost_basis" in p:
                book.cost_basis = float(p["cost_basis"])
        elif kind in ("put", "call"):
            if qty != 0:
                book.open_options.append(p)
        else:
            raise BookFormatError(f"unknown contract {kind!r} in {p}")
    if book.shares or book.open_options:
        book.first_entry = False
    if book.shares < 0:
        raise BookFormatError("short shares: the wheel never shorts the stock")
    return book


def put_contracts(cash: float, strike: float) -> int:
    """As many puts as the cash already in the account secures at the strike."""
    return int(math.floor(cash / (CONTRACT * strike) + 1e-9))


# --------------------------------------------------------------------------- the order
ORDER_FIELDS = {"id", "for", "order_type", "action", "qty", "underlying", "contract",
                "expiry", "strike", "ref", "lmt_prc"}
_NUMBER = re.compile(r"^\d+(\.\d+)?$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def number_token(x: float) -> str:
    """A bare-number token: 165.0 -> '165', 167.5 -> '167.5', 4.351 -> '4.35'."""
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
    if not isinstance(o["qty"], int) or o["qty"] < 1:
        raise ValueError("qty must be an integer of at least 1")
    if not isinstance(o["underlying"], int):
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
