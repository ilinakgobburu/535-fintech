"""
The far-wing trade, live: tonight's orders for the course server (Sanji), for a basket of names, each its own book.

The trade (her choices of 2026-11-16): once a month sell one far put (|delta| under 0.10) and one far call (|delta|
under 0.05), delta-hedged with the stock only when the book's net delta leaves a band ("Hedge less often (delta
bands)"), no crash cap ("Hedge only"), each option sold with a limit at the mid ("Yes, limits at the mid"). This
becomes her class algorithm ("Yes, switch to far wings"): it replaces the two-sided variance trade as the one nightly
writer of mkt/open_orders.json. It began on NVDA alone; on 2026-11-17 ("why cant you do more than nvda?") she chose
"The 30-stock basket" and "$100k book per stock": the long/short vol book's 30 names join NVDA, each an independent
$100,000 book under exactly NVDA's rules (no pooling: its own wings, its own $5,000 band, its own hedge limit and its
own reconciliation). This module does no I/O: far_wings_orders.py at the repo root pulls LSEG, calls these functions
and writes the file. It reuses vol_arb_live's calendar, quote, book and order helpers (each tested there) and adds
the far-wing rules; every lib function it uses outside this file is pinned by hash in config/far_wings_live.yaml
(lib_functions), so an edit to one of them stops the nightly run instead of changing it.

A NAME is one entry of config/far_wings_live.yaml's `names`: its ticker (in every order id, made safe: BRK B is
brkb), the stock's LSEG RIC, the LSEG option root, its Interactive Brokers conid (what an order names; null until
confirmed), its strike grid and its lot. A name with no conid, or one Sanji does not track (price_history.json,
tracked: true), gets no orders (untradable).

WHAT IS KNOWN WHEN. The script runs after the close of session D, the last NYSE session whose close is at or before
the run (run_dates), and writes orders for `for`, the next NYSE session after D, always, which Sanji fills at that
session's prices: a market order at the open, a limit sell when the bar's high reaches the limit. order.txt: 'for' is
"The session the order belongs to. Not 'next'. A late run must not fill Monday's order on Tuesday", and Sanji resolves
session S before a script plans S+1. GitHub can start a scheduled run hours late (on 2026-11-17 the 03:00 UTC run ran
at 11:23 UTC, the 06:00 retry at 15:29 UTC), so `for` is never taken from the clock: a run whose `for` has already
opened (09:30 New York, session_open) writes nothing and ends red, the entry of an entry night lost for that cycle and
the band check waiting for the next night. (wheel_live.decision_dates, which the vol-arb and the wheel scripts use,
takes `for` from the clock, the first session opening after the run: run during D+1, it labelled D's decisions for
D+2, a stale order a day late.)

ENTRY NIGHTS are vol-arb's calendar: the first session after each monthly expiry (vol_arb_live.entry_sessions, on
wheel_live's NYSE calendar), each position held to the next monthly expiry. On an entry night, for each name, from
D's closing quotes of that expiry's chain (two-sided as vol_arb.two_sided has it: bid and ask there, ask at or above
bid, mid above zero, one RIC per strike):

    put wing    among the puts whose |delta| is under 0.10 (wheel.option_greeks on each one's own mid, at D's spot,
                T in calendar days / 365, the T-bill and the trailing year's dividend yield: wheel.carry), the one
                whose |delta| is closest to 0.05; a tie goes to the lower strike
    call wing   among the calls whose |delta| is under 0.05, the one closest to 0.03; a tie goes to the higher strike
    size        each wing round(200 / its Black-76 vega per vol point per contract, with D's lot), at least 1
                (vol_arb.leg_vega and contracts_for: about $200 a vol point per wing on that name's $100,000 book)
    orders      SHT each wing, a limit at D's closing mid (to the cent), bare-number strike and price strings, the
                expiry as a date; ids <ticker>-fw-put-<D> and <ticker>-fw-call-<D> (nvda-fw-put-2026-11-23)

No eligible strike is no position in that wing this cycle (said in the log); the other wing still trades. A wing whose
mid rounds to a limit of 0 is not sold either. That rule is for a chain that is there: a right with D rows but not
one two-sided quote (one_sided_problem: D's bids not posted yet) stops that name at 22:00 as data not final, and only
the 01:00 retry, finding it so still, takes it as no eligible strike.

EVERY NIGHT, THE BAND, name by name. The book's net delta in shares: each held option's delta at D's close (its own
closing mid's implied vol; a leg with no usable mid that close keeps its last IV, delta recomputed at D's spot and T:
vol_arb_live.leg_delta) times its contracts and the lot, plus the shares held. When |net delta x D's close| is above
the band (5% of that name's $100,000 book, $5,000; at exactly $5,000 nothing is done), a market order for the next
open moves the shares to -round(the options' delta), zero shares-delta; otherwise no hedge trade. One order cannot
flip a long into a short, so crossing zero is two orders, the close first (ids <ticker>-fw-hedge-sel-<D> and
<ticker>-fw-hedge-sht-<D>; one order is <ticker>-fw-hedge-<D>). The first check is the first night the options are
in the book (a limit may not fill). With no options held, the shares are closed (after an expiry, or after Sanji
assigned both wings).

EXPIRY. Sanji settles the options on the expiry's close (an assignment delivers shares). The night of the expiry the
book holds no options, so the leftover hedge and any delivered shares are closed at the next open; the next night
is the next entry. An option still listed on or after its expiry stops the run (the book is not up to date).

WHOSE BOOK, AND A STALE BOOK: as vol_arb_live has them, name by name, for this strategy's ids (<ticker>-fw-...).
Every option the book holds on a name's conid must be exactly what that name's filled orders add up to (its ids in
mkt/fills.json, looked up in its part of data/far_wings_ledger.json, which keeps every order it wrote), a short leg
excepted when Sanji has assigned it; its shares need a fill of its own (book_problems). Such a problem, or an order of
its own Sanji rejected (until she lists the id in config's `acknowledged`), stops that name only (guard_problems). An
order of any name's with no result once mkt/fills.json exists, for a session up to D, stops every name: the book file
may be stale. Another strategy's orders are only noted. A name's hedge target and shares held may not pass
limits.max_hedge_x_capital times its book (hedge_limit), every night shares or options are held.

ONE FILE. The night's orders for all names go into one mkt/open_orders.json (compose_orders), in config order: each
name decided tonight with its new orders, every other name with its orders as the file had them, then her own orders
for a session after D as she wrote them; the retired strategies' orders, and any for a session up to D, leave it.
"""

from __future__ import annotations

import datetime as dt
import math
import re

import numpy as np
import pandas as pd

from . import vol_arb as VA
from . import vol_arb_live as VL
from . import wheel as W
from . import wheel_live as L
from .ric import build_option_ric

PUT, CALL = W.PUT, W.CALL
LOT = W.CONTRACT                          # shares per contract, for every name (config checks each lot)
TAG = "fw"                                # in every order id: nvda-fw-put-2026-11-23, brkb-fw-call-2026-11-23
CAP = {PUT: 0.10, CALL: 0.05}             # a wing's strikes: |delta| strictly under this
TARGET = {PUT: 0.05, CALL: 0.03}          # ...the one whose |delta| is closest to this
VEGA_DOLLARS = 200.0                      # each wing: round(200 / vega per vol point per contract), at least 1
CAPITAL = 100_000.0                       # each name's book, which its $200 a vol point and band are sized on
BAND_SHARE = 0.05                         # re-hedge a name when its |net delta x close| > 5% of its book
LEDGER_FORMAT = "far_wings_ledger/2"      # by name: {"names": {ticker: {"orders", "nights"}}}
LEDGER_PATH = "data/far_wings_ledger.json"
CONFIG_PATH = "config/far_wings_live.yaml"
SIGNS = VL.SIGNS                          # what one filled order does to a position: BUY/BTC +, SEL/SHT -
FALLBACK_SESSIONS = VL.FALLBACK_SESSIONS  # how far back a held leg's last usable mid may be

# vol_arb_live's and wheel_live's helpers this trade uses as they are, here and in far_wings_orders.py, so that
# lib_functions pins them with the rest: the calendar, the market from the pulls, the chain's checks, the book.
night, next_session, is_session, previous_expiry = VL.night, VL.next_session, VL.is_session, VL.previous_expiry
next_entry, entry_expiry = VL.next_entry, VL.entry_expiry
market, quote_frame, strike_plan, chain_problem = VL.market, VL.quote_frame, VL.strike_plan, VL.chain_problem
Book, read_book, held_contracts, expired_legs = VL.Book, VL.read_book, VL.held_contracts, VL.expired_legs
merge_orders, BookFormatError = L.merge_orders, L.BookFormatError
transition, leg_delta, sha256_hex, option_ric = VL.transition, VL.leg_delta, VL.sha256_hex, build_option_ric


class StaleBook(ValueError):
    """The book is not up to date with what has happened: no order can be planned from it."""


class LedgerError(ValueError):
    """data/far_wings_ledger.json is malformed, or a fill of this strategy's is not in it."""


def _d(x) -> dt.date:
    return pd.Timestamp(x).date()


def band_dollars() -> float:
    """The band, each name's own: 5% of its $100,000 book, $5,000."""
    return BAND_SHARE * CAPITAL


# --------------------------------------------------------------------------- the run's dates
def run_dates(now: dt.datetime) -> tuple[dt.date, dt.date]:
    """
    (D, for) for a run at `now`, New York time, on wheel_live's NYSE calendar (its holidays and its 13:00 early
    closes). D is the last session whose close is at or before `now`, so its closing quotes are final. `for` is the
    next session after D, whatever the clock says: Monday 22:00, Tuesday 01:00 and Tuesday 10:29 all give (Monday,
    Tuesday); a Friday night or a Saturday gives (Friday, Monday). Whether `for` has opened already is session_open's
    question, and the run's to refuse.
    """
    days = [d.date() for d in L.sessions(now.date() - dt.timedelta(days=14), now.date())]
    day = max(d for d in days if dt.datetime.combine(d, L.close_time(d)) <= now)
    return day, next_session(day)


def session_open(for_day, now: dt.datetime) -> bool:
    """Whether `for_day`'s session has opened (09:30 New York) by `now`: D's orders for it would then be a day late."""
    return now >= dt.datetime.combine(_d(for_day), L.OPEN)


def decided_from(ledger: dict, day) -> str | None:
    """
    Whether one name's ledger (name_ledger) records a run that decided it from D's close: the ids it wrote, "no order",
    or None. Keyed on D alone: D's orders are only ever for the session after D (run_dates), so the clock of the run
    asking has no part in it.
    """
    night = ledger["nights"].get(_d(day).isoformat())
    if night is None:
        return None
    return ", ".join(night.get("ids") or []) or "no order"


# --------------------------------------------------------------------------- the wings
def choose(cands: pd.DataFrame, cp: str) -> pd.Series | None:
    """
    The wing among candidates that carry `strike` and `delta` (per share, a put's negative): |delta| strictly under
    CAP[cp], then the |delta| closest to TARGET[cp]; a tie goes to the lower strike for a put, the higher for a call.
    None when no candidate is under the cap.
    """
    c = cands[cands["delta"].notna()]
    c = c[c["delta"].abs() < CAP[cp]]
    if c.empty:
        return None
    c = c.assign(miss=(c["delta"].abs() - TARGET[cp]).abs())
    return c.sort_values(["miss", "strike"], ascending=[True, cp == PUT]).iloc[0]


def contracts_for(vega_per_share: float, lot: int = LOT) -> int:
    """round(200 / (vega per share per 1.00 of vol / 100 x lot)), at least 1: vol_arb.contracts_for at $200."""
    return VA.contracts_for(vega_per_share, lot, VEGA_DOLLARS)


def wing(day_quotes: pd.DataFrame, cp: str, expiry, spot: float, day, r: float, q: float, lot: int = LOT) -> dict | None:
    """
    One wing on D's close: D's two-sided quotes of `expiry` on that right (vol_arb.two_sided), each one's IV and delta
    from its own mid (wheel.option_greeks at D's spot, T = calendar days / 365, r, q), then choose(). With its Black-76
    vega at its own IV (vol_arb.leg_vega) and the contracts it is sized to. None when no strike qualifies.
    """
    day, expiry = pd.Timestamp(day), pd.Timestamp(expiry)
    c = VA.two_sided(day_quotes, expiry)
    c = c[c["cp"] == cp]
    if c.empty:
        return None
    T = (expiry - day).days / W.DAYS_PER_YEAR
    g = [W.option_greeks(float(m), spot, float(k), T, r, q, cp) for m, k in zip(c["mid"], c["strike"])]
    c = c.assign(iv=[x["iv"] for x in g], delta=[x["delta"] for x in g])
    pick = choose(c[c["iv"].notna()], cp)
    if pick is None:
        return None
    F, D = spot * np.exp((r - q) * T), np.exp(-r * T)
    vega = float(VA.leg_vega(F, float(pick["strike"]), T, float(pick["iv"]), D))
    return {"ric": pick["ric"], "strike": float(pick["strike"]), "mid": float(pick["mid"]),
            "bid": W._num(pick["bid"]), "ask": W._num(pick["ask"]), "iv": float(pick["iv"]),
            "delta": float(pick["delta"]), "vega": vega, "contracts": contracts_for(vega, lot)}


def wings_at_entry(mkt: dict, day, expiry) -> dict:
    """Both wings on the entry close of `expiry`'s chain, with the spot, carry, T and lot they were chosen at."""
    day, expiry = pd.Timestamp(day), pd.Timestamp(expiry)
    quotes = mkt["quotes"]
    day_q = quotes[quotes["date"] == day]
    spot = float(mkt["stock"].at[day, "close"])
    r, q = W.carry(mkt, day)
    lot = int(mkt["lot"].get(day, LOT)) if "lot" in mkt else LOT
    out = {"entry": day, "expiry": expiry, "spot": spot, "r": r, "q": q,
           "T": (expiry - day).days / W.DAYS_PER_YEAR, "lot": lot}
    for cp in (PUT, CALL):
        out[cp] = wing(day_q, cp, expiry, spot, day, r, q, lot)
    return out


def one_sided_problem(day_quotes: pd.DataFrame, expiry, day) -> str | None:
    """
    Why D's chain of `expiry` cannot be chosen from yet (None when it can): a right with D rows on LSEG but not one
    two-sided quote among them (vol_arb.two_sided, the rule wing() chooses from: bid and ask there, ask at or above
    bid, mid above zero). A healthy chain has many on each right, so none at all is LSEG part-way through posting D
    (its bids not in yet), not the registered "no eligible strike", which is meant for a chain that is there.
    """
    two = VA.two_sided(day_quotes, expiry)
    rights = [("put" if cp == PUT else "call") for cp in (PUT, CALL) if not (two["cp"] == cp).any()]
    if not rights:
        return None
    return (f"no {_d(expiry)} {' and no '.join(rights)} has a two-sided {_d(day)} quote on LSEG (bid and ask both "
            f"there), though the chain has {_d(day)} rows")


# --------------------------------------------------------------------------- the band
def breached(net_delta_shares: float, spot: float) -> bool:
    """Whether |net delta in shares x D's close| is above the band ($5,000): only then is the book re-hedged."""
    return abs(float(net_delta_shares)) * float(spot) > band_dollars()


def hedge_limit(cfg: dict, spot: float, target: int, shares: int, ticker: str) -> None:
    """
    A name's hedge target and the shares it holds, each times D's close, may not pass limits.max_hedge_x_capital times
    its own $100,000 book. The wings are sized $200 a vol point, so a 0.03-delta call deep in the money after a large rise can
    need a hedge of about 15 to 25 times the book: vol-arb's 5 would stop the run exactly when the hedge matters.
    Shares are not reconciled with the fills (an assignment delivers them), so this is their guard against a misread
    book (a 100-times reading of a normal hedge is far past it).
    """
    x = float(cfg["limits"]["max_hedge_x_capital"])
    for n, what in ((target, "hedge target"), (shares, "shares held")):
        if abs(n) * spot > x * CAPITAL:
            raise ValueError(f"the {what}, {n:+d} {ticker} shares (${abs(n) * spot:,.0f} at {spot:.2f}), is more than "
                             f"{x:g} x the ${CAPITAL:,.0f} book: the book is probably misread")


# --------------------------------------------------------------------------- the orders
def id_root(ticker: str) -> str:
    """A ticker as it goes into an order id: lower case, letters and digits only ('NVDA' -> 'nvda', 'BRK B' -> 'brkb')."""
    return re.sub(r"[^a-z0-9]", "", str(ticker).lower())


def own_prefix(ticker: str) -> str:
    """Every id this strategy writes for a name starts with it: 'nvda-fw-', 'brkb-fw-'. No prefix starts another, as an
    id root has no '-' in it."""
    return f"{id_root(ticker)}-{TAG}-"


def owner(order_id, tickers) -> str | None:
    """Which name of the basket an order id is this strategy's for (None: another strategy's)."""
    return next((t for t in tickers if str(order_id).startswith(own_prefix(t))), None)


def _id(root: str, what: str, action: str | None, day) -> str:
    return f"{own_prefix(root)}{what}{'-' + action.lower() if action else ''}-{_d(day).isoformat()}"


def stock_orders(*, root: str, conid: int, day, for_day, held: int, target: int) -> list[dict]:
    """Market orders for the next open that move a name's shares from `held` to `target`; `root` is its ticker (ids
    nvda-fw-hedge-<D>, brkb-fw-hedge-<D>), `conid` its Interactive Brokers id."""
    steps = transition(held, target)
    return [{"id": _id(root, "hedge", a if len(steps) > 1 else None, day), "for": _d(for_day).isoformat(),
             "order_type": "mkt", "action": a, "qty": int(n), "underlying": int(conid), "contract": "underlying",
             "ref": "open"} for a, n in steps]


def option_orders(*, root: str, conid: int, day, for_day, cp: str, expiry, strike: float, limit: str,
                  held: int, target: int) -> list[dict]:
    """Limit orders at `limit` (a bare-number token) moving one contract from `held` to `target`; `root` is the name's
    ticker (nvda-fw-put-<D>)."""
    steps = transition(held, target)
    what = "put" if cp == PUT else "call"
    return [{"id": _id(root, what, a if len(steps) > 1 else None, day), "for": _d(for_day).isoformat(),
             "order_type": "lmt", "action": a, "qty": int(n), "underlying": int(conid), "contract": what,
             "expiry": _d(expiry).isoformat(), "strike": L.number_token(strike), "ref": "open",
             "lmt_prc": limit} for a, n in steps]


def check_orders(orders: list[dict]) -> None:
    """wheel_live.validate_order on each (order.html's checks), every strike and limit a string token, ids unique."""
    for o in orders:
        L.validate_order(o)
        for k in ("strike", "lmt_prc"):
            if k in o and not isinstance(o[k], str):
                raise ValueError(f"{o['id']}: {k} must be a string token, not {o[k]!r}")
    ids = [o["id"] for o in orders]
    if len(set(ids)) != len(ids):
        raise ValueError(f"order ids repeat: {ids}")


# --------------------------------------------------------------------------- the ledger, the guards, the book
def read_ledger(obj) -> dict:
    """
    data/far_wings_ledger.json, which far_wings_orders.py keeps beside mkt/open_orders.json (Sanji's file keeps only
    the coming orders, so what this strategy wrote earlier, and so what its fills were, is kept here), each name
    separately:

        {"format": "far_wings_ledger/2",
         "names": {"NVDA": {"orders": {"nvda-fw-put-2026-11-23": {the order as written}, ...},
                            "nights": {"2026-11-23": {"for": "2026-11-24", "ids": ["nvda-fw-put-2026-11-23", ...]}}},
                   "BRK B": {"orders": {...}, "nights": {...}}, ...}}

    A name's night is there once a run decided that name for that night (its orders, or none); a name left for the
    retry, skipped or stopped has no night. None (no file yet) is an empty ledger.
    """
    if obj is None:
        return {"format": LEDGER_FORMAT, "names": {}}
    names = obj.get("names") if isinstance(obj, dict) else None
    if (not isinstance(obj, dict) or obj.get("format") != LEDGER_FORMAT or not isinstance(names, dict)
            or not all(isinstance(t, str) and isinstance(v, dict) and isinstance(v.get("orders"), dict)
                       and isinstance(v.get("nights"), dict) for t, v in names.items())):
        raise LedgerError(f"{LEDGER_PATH} is not a {LEDGER_FORMAT} file")
    return obj


def name_ledger(ledger: dict, ticker: str) -> dict:
    """One name's part of the ledger, {"orders", "nights"} (empty before its first night)."""
    return ledger["names"].get(ticker) or {"orders": {}, "nights": {}}


def record_night(ledger: dict, ticker: str, new: list[dict], day, for_day) -> dict:
    """The ledger after a name's orders for tonight were written: each by id (a rerun's version replaces the earlier
    one), and the night; every other name as it was."""
    mine = name_ledger(ledger, ticker)
    mine = {"orders": dict(mine["orders"]), "nights": dict(mine["nights"])}
    for o in new:
        mine["orders"][o["id"]] = o
    mine["nights"][_d(day).isoformat()] = {"for": _d(for_day).isoformat(), "ids": [o["id"] for o in new]}
    return {"format": LEDGER_FORMAT, "names": {**ledger["names"], ticker: mine}}


def strategy_options(ledger: dict, fills: list | None, ticker: str, day) -> tuple[dict, bool]:
    """
    ({(cp, expiry, strike): contracts}, any fill): what one name's filled option orders add up to for the expiries
    after D, and whether any of its orders, a hedge included, has filled (Sanji fills all or nothing). `ledger` is that
    name's part (name_ledger).
    """
    prefix = own_prefix(ticker)
    own = sorted({f.get("id") for f in fills or [] if isinstance(f, dict) and isinstance(f.get("id"), str)
                  and f["id"].startswith(prefix) and str(f.get("status", "")).lower() == "filled"})
    unknown = [i for i in own if i not in ledger["orders"]]
    if unknown:
        raise LedgerError(f"mkt/fills.json records {', '.join(unknown)} filled, but {LEDGER_PATH} has no such "
                          f"order, so what it bought or sold is not known")
    out = {}
    for i in own:
        o = ledger["orders"][i]
        if o.get("contract") not in ("put", "call"):
            continue
        key = (PUT if o["contract"] == "put" else CALL, dt.date.fromisoformat(o["expiry"]), float(o["strike"]))
        out[key] = out.get(key, 0) + SIGNS[o["action"]] * int(o["qty"])
    return {k: v for k, v in out.items() if v != 0 and k[1] > _d(day)}, bool(own)


def guard_problems(written: list[dict], fills: list | None, day, tickers, acknowledged: dict
                   ) -> tuple[list, dict, list]:
    """
    (stale, rejected, notes) for mkt/open_orders.json as it stands, the basket's names `tickers`:
      stale     why every name must stop: an order of any name's written for a session up to D with no result once
                mkt/fills.json exists. Sanji writes one book for the session, so the whole book may be stale
      rejected  {ticker: [why]}: an order of that name's own Sanji rejected; it stops that name only
      notes     another strategy's (the wheel's last put, a vol-arb order), and every acknowledged one
    Only until she lists the id in config's `acknowledged` does one of this strategy's orders stop anything.
    """
    stale, rejected, notes = [], {}, []
    for f in L.rejected_orders(written, fills):
        i = f.get("id")
        t = owner(i, tickers)
        if t is None:
            notes.append(f"Sanji rejected {i}, which is not this strategy's: {f}")
        elif i in acknowledged:
            notes.append(f"Sanji rejected {i}; acknowledged in {CONFIG_PATH}: {acknowledged[i]}")
        else:
            rejected.setdefault(t, []).append(f"Sanji rejected {i}: {f}. Fix the order format; to trade on, add {i} "
                                              f"to {CONFIG_PATH}'s acknowledged with the reason")
    waiting = L.unresolved_orders(written, fills, _d(day))
    for i in [i for i in waiting if owner(i, tickers) is None]:
        notes.append(f"mkt/fills.json has no result for {i}, which is not this strategy's")
    for i in [i for i in waiting if owner(i, tickers) is not None and i in acknowledged]:
        notes.append(f"mkt/fills.json has no result for {i}; acknowledged in {CONFIG_PATH}: {acknowledged[i]}")
    mine = [i for i in waiting if owner(i, tickers) is not None and i not in acknowledged]
    if mine:
        stale.append(f"mkt/fills.json has no result yet for {', '.join(mine)}: the book may be stale. If Sanji has "
                     f"resolved the session and will not answer, add the id to {CONFIG_PATH}'s acknowledged with "
                     f"the reason")
    return stale, rejected, notes


def _leg(key) -> str:
    cp, expiry, strike = key
    return f"{expiry} {strike:g} {'put' if cp == PUT else 'call'}"


def book_problems(book: Book, fills: list | None, ledger: dict, ticker: str, day) -> list[str]:
    """
    Why one name's positions (its conid's, read_book) are not (all) this strategy's, or not up to date with its fills
    (empty when they are): every option held must be exactly what that name's filled orders add up to, except that a
    short leg may be gone (Sanji assigns a short option whole); a long leg its fills hold must be there; shares need
    at least one fill of its own. With no mkt/fills.json at all, a book with positions can't be checked against
    anything. `ledger` is that name's part (name_ledger).
    """
    held, root = held_contracts(book), ticker
    if fills is None:
        if held or book.shares:
            return [f"mkt/positions.json lists {ticker} positions but there is no mkt/fills.json, so it can't be told "
                    "whether they are this strategy's or whether the book is up to date with its orders"]
        return []
    want, any_fill = strategy_options(ledger, fills, root, day)
    out = []
    for key, qty in sorted(held.items()):
        w = want.get(key, 0)
        if w == 0:
            out.append(f"the book holds {qty:+d} {_leg(key)}, which no filled {own_prefix(root)}* order accounts for")
        elif qty != w:
            out.append(f"the book holds {qty:+d} {_leg(key)}, but this strategy's fills add up to {w:+d}")
    for key, w in sorted(want.items()):
        if w > 0 and key not in held:
            out.append(f"this strategy's fills hold {w:+d} {_leg(key)}, but the book does not list it")
    if book.shares and not any_fill:
        out.append(f"the book holds {book.shares:+d} {ticker} shares, but no {own_prefix(root)}* order has filled")
    return out


# --------------------------------------------------------------------------- the basket
def tracked_conids(price_history) -> set[int]:
    """
    The conids price_history.json marks tracked: true, the stocks Sanji prices (and so resolves orders on). Its keys
    are the conids as strings: {"4815747": {"tracked": true, "name": "NVIDIA CORPORATION", "price_history": {...}}}.
    """
    if not isinstance(price_history, dict):
        raise ValueError("price_history.json should map each conid to {tracked, name, price_history}")
    return {int(k) for k, v in price_history.items()
            if isinstance(k, str) and k.isdigit() and isinstance(v, dict) and v.get("tracked") is True}


def untradable(name: dict, tracked: set) -> str | None:
    """Why a name gets no orders tonight whatever its book (None when it can trade): no conid, or one not tracked."""
    if name["conid"] is None:
        return f"no Interactive Brokers conid confirmed yet (conid: null in {CONFIG_PATH})"
    if name["conid"] not in tracked:
        return (f"conid {name['conid']} is not in price_history.json with tracked: true, so Sanji does not price it "
                f"and could not resolve its orders")
    return None


RETIRED_IDS = re.compile(r"^[a-z0-9]+-(?:va|csp|cc)-")   # the vol-arb's ids (nvda-va-put-...) and the wheel's
                                                          # (nvda-csp-..., nvda-cc-...): the far wings replaced both


def compose_orders(written: list[dict], decided: dict, tickers: list, day, for_day) -> tuple[list[dict], str]:
    """
    The night's mkt/open_orders.json, one file for every name, and a note. In config order (`tickers`): a name decided
    tonight (`decided`: {ticker: its orders, perhaps none}) with tonight's orders; every other name of the basket with
    its orders as the file has them, so a name stopped tonight keeps the rejected order the next night's check finds
    again, and a name the 22:00 run decided keeps tonight's orders through the 01:00 retry. Then any other order
    for a session after D, as it was: her own order closing another strategy's position by hand (what a name stopped
    by such a position waits for) is not dropped because the other names were written. The retired strategies' own
    orders (the vol-arb's and the wheel's ids, RETIRED_IDS) leave the file whatever their session, as every other
    order for a session up to D does, as wheel_live.merge_orders has it for the one writer. An order dated between D
    and `for` (merge_orders' note) keeps that date's orders alone and adds nothing, as before; with `for` the session
    right after D (run_dates) that is only an order dated on a weekend or holiday between them, since a run during
    `for`'s session, which this note once handled, now writes nothing at all (session_open).
    """
    new = [o for t in tickers if t in decided for o in decided[t]]
    kept, note = merge_orders(written, new, day, for_day)
    if note:
        return kept, note
    out = []
    for t in tickers:
        out += decided[t] if t in decided else [o for o in written if owner(o.get("id"), tickers) == t]
    later = [o for o in written if owner(o.get("id"), tickers) is None and not RETIRED_IDS.match(str(o.get("id")))
             and str(o.get("for", "")) > _d(day).isoformat()]
    return out + later, ""


# --------------------------------------------------------------------------- the night
def plan_night(cfg: dict, day, for_day, book: Book, *, name: dict, entry_expiry=None, mkt: dict | None = None,
               leg_history: dict | None = None) -> dict:
    """
    Tonight's orders for one name (`name`, an entry of config's names; `book` its positions alone, `mkt` its market).
    The band, every night options are held: the net delta (options plus shares) at D's close, and
    a market order to -round(the options' delta) only when it is outside the band; with no options held, the shares
    are closed. On an entry night (`entry_expiry` set): the two wings' SHT limit orders at D's closing mid.
    `leg_history` maps each held contract (cp, expiry, strike) to its daily bid/ask. `mkt` (D's close at least) is
    needed whenever options or shares are held, and on an entry night.
    """
    root, conid = str(name["ticker"]), int(name["conid"])
    day, for_day = _d(day), _d(for_day)
    late = expired_legs(book, day)
    if late:
        raise StaleBook(f"the book still lists options that expired by {day}: {late}")
    notes, held = [], held_contracts(book)
    if (held or book.shares or entry_expiry is not None) and mkt is None:
        raise ValueError("options or shares are held, or it is an entry night, but no market data was given")

    legs = []
    for (cp, expiry, strike), qty in sorted(held.items()):
        d = leg_delta(mkt, cp, expiry, strike, (leg_history or {}).get((cp, expiry, strike)), day)
        if not np.isfinite(d["delta"]):
            raise ValueError(f"no usable quote for the {expiry} {strike:g} {cp} in the last {FALLBACK_SESSIONS} "
                             f"sessions, so its delta (and the band check) is unknown")
        legs.append({"cp": cp, "expiry": expiry, "strike": strike, "qty": qty, **d})
    option_delta = float(sum(leg["qty"] * LOT * leg["delta"] for leg in legs))
    net = option_delta + book.shares
    hedge = {"legs": legs, "option_delta": option_delta, "net_delta": net, "shares": book.shares,
             "band": band_dollars(), "exposure": None, "breached": None, "target": book.shares}
    spot = float(mkt["stock"].at[pd.Timestamp(day), "close"]) if mkt is not None else np.nan
    if legs:
        hedge["exposure"], hedge["breached"] = abs(net) * spot, breached(net, spot)
        where = "outside" if hedge["breached"] else "inside"
        if hedge["breached"]:
            hedge["target"] = -int(round(option_delta))
        notes.append(f"net delta {net:+.1f} shares (${abs(net) * spot:,.0f} at {spot:.2f}) is {where} the "
                     f"${band_dollars():,.0f} band: " + (f"hedge to {hedge['target']} shares (holding {book.shares})"
                                                         if hedge["breached"] else
                                                         f"no hedge trade (holding {book.shares})"))
    elif book.shares:
        hedge["target"] = 0
        notes.append(f"no options held: close the {book.shares:+d} leftover shares")
    if legs or book.shares:
        hedge_limit(cfg, spot, hedge["target"], book.shares, root)
    stock = stock_orders(root=root, conid=conid, day=day, for_day=for_day, held=book.shares, target=hedge["target"])

    entry, options = None, []
    if entry_expiry is not None:
        entry = wings_at_entry(mkt, day, entry_expiry)
        for cp in (PUT, CALL):
            name = "put" if cp == PUT else "call"
            leg = entry[cp]
            if leg is None:
                notes.append(f"entry {day} -> {_d(entry_expiry)}: no {name} wing (no two-sided {name} under "
                             f"{CAP[cp]:.2f} |delta|)")
                continue
            limit = L.number_token(leg["mid"])
            if not float(limit) > 0:
                notes.append(f"entry {day} -> {_d(entry_expiry)}: no {name} wing (its mid {leg['mid']:.4f} rounds to "
                             f"a limit of {limit})")
                continue
            have = held.get((cp, _d(entry_expiry), float(leg["strike"])), 0)
            options += option_orders(root=root, conid=conid, day=day, for_day=for_day, cp=cp, expiry=entry_expiry,
                                     strike=leg["strike"], limit=limit, held=have, target=have - int(leg["contracts"]))
            notes.append(f"entry {day} -> {_d(entry_expiry)}: sell {leg['contracts']} {leg['strike']:g} {name}s at "
                         f"{limit} (|delta| {abs(leg['delta']):.3f})")
    orders = options + stock
    check_orders(orders)
    if not orders:
        notes.append("no order tonight")
    return {"orders": orders, "notes": notes, "entry": entry, "hedge": hedge}


# --------------------------------------------------------------------------- the config
REGISTERED = {"put_max_abs_delta": CAP[PUT], "put_target_abs_delta": TARGET[PUT],
              "call_max_abs_delta": CAP[CALL], "call_target_abs_delta": TARGET[CALL],
              "vega_dollars": VEGA_DOLLARS, "capital": CAPITAL, "band_share_of_book": BAND_SHARE, "lot": LOT}


def _finite_number(x) -> bool:
    return type(x) in (int, float) and math.isfinite(x)


NAME_KEYS = {"ticker", "ric", "root", "conid", "grid", "lot", "probe"}
_ROOT = re.compile(r"^[A-Z]+$")
_TICKER = re.compile(r"^[A-Z0-9]+([ .\-][A-Z0-9]+)*$")


def _grid_problem(grid) -> str | None:
    """Why a name's strike grid is not [[price below which it applies, increment], ..., [null, increment]]."""
    if not isinstance(grid, list) or not grid:
        return "is not a list of [bound, increment] tiers"
    bounds = []
    for tier in grid:
        if not (isinstance(tier, list) and len(tier) == 2 and _finite_number(tier[1]) and tier[1] > 0
                and (tier[0] is None or (_finite_number(tier[0]) and tier[0] > 0))):
            return f"has a tier {tier!r} that is not [bound or null, a positive increment]"
        bounds.append(tier[0])
    if bounds[-1] is not None or None in bounds[:-1] or bounds[:-1] != sorted(set(bounds[:-1])):
        return "needs rising bounds and null (any price above) last"
    return None


def check_names(names, lot) -> None:
    """config's `names`: each a ticker, the stock's LSEG RIC, its upper-case option root, an integer conid or null,
    a strike grid and a lot equal to the registered one; tickers, id roots and conids each distinct."""
    if not isinstance(names, list) or not names:
        raise ValueError("names must be a list of the basket's names")
    for n in names:
        if not isinstance(n, dict) or set(n) - NAME_KEYS or not NAME_KEYS - {"probe"} <= set(n):
            raise ValueError(f"each name needs exactly ticker, ric, root, conid, grid, lot (and a probe note): {n!r}")
        t = n["ticker"]
        if not (isinstance(t, str) and _TICKER.match(t)):
            raise ValueError(f"name {t!r}: the ticker must be a non-empty string of capitals and digits (a space, a "
                             f"dot or a hyphen between, as in BRK B)")
        if not (isinstance(n["ric"], str) and n["ric"] and isinstance(n["root"], str) and _ROOT.match(n["root"])):
            raise ValueError(f"name {t}: ric must be the stock's LSEG RIC and root its upper-case option root")
        if n["conid"] is not None and type(n["conid"]) is not int:
            raise ValueError(f"name {t}: conid must be an integer, or null until it is confirmed")
        problem = _grid_problem(n["grid"])
        if problem:
            raise ValueError(f"name {t}: grid {problem}")
        if type(n["lot"]) is not int or n["lot"] != lot:
            raise ValueError(f"name {t}: lot is {n['lot']!r}; every name is sized on the registered lot {lot}")
        if "probe" in n and not (isinstance(n["probe"], str) and n["probe"].strip()):
            raise ValueError(f"name {t}: probe, when given, says what the strike probe found")
    for what, keys in (("ticker", [n["ticker"] for n in names]), ("id root", [id_root(n["ticker"]) for n in names]),
                       ("conid", [n["conid"] for n in names if n["conid"] is not None])):
        if len(set(keys)) != len(keys):
            raise ValueError(f"two names have the same {what}: {sorted(k for k in set(keys) if keys.count(k) > 1)}")


def check_config(cfg: dict) -> None:
    """config/far_wings_live.yaml's registered numbers must be this module's, and its other settings well formed."""
    if "underlying" in cfg:
        raise ValueError("underlying is gone: the basket's names are under names")
    check_names(cfg.get("names"), LOT)
    reg = cfg.get("registered") or {}
    bad = [f"{k} is {reg.get(k)!r}, the code's is {v!r}" for k, v in REGISTERED.items()
           if not _finite_number(reg.get(k)) or float(reg[k]) != float(v)]
    if bad:
        raise ValueError(f"{CONFIG_PATH} disagrees with lib/far_wings_live.py: " + "; ".join(bad))
    if type(cfg.get("owns_open_orders")) is not bool:
        raise ValueError("owns_open_orders must be true or false")
    ack = cfg.get("acknowledged", {})
    if not isinstance(ack, dict) or not all(isinstance(k, str) and isinstance(v, str) and v.strip()
                                            for k, v in ack.items()):
        raise ValueError("acknowledged must map order ids to the reason each no longer stops the run")
    share = (cfg.get("chain") or {}).get("min_dated_share")
    if not _finite_number(share) or not 0 < share <= 1:
        raise ValueError(f"chain.min_dated_share must be a share in (0, 1], not {share!r}")
    x = (cfg.get("limits") or {}).get("max_hedge_x_capital")
    if not _finite_number(x) or x <= 0:
        raise ValueError(f"limits.max_hedge_x_capital must be a positive number, not {x!r}")


# --------------------------------------------------------------------------- what this module takes from the lib
def lib_functions(read_source) -> dict:
    """
    {"module.name": sha256 of its source} for every lib function and constant this module uses from another lib
    module, and everything those use in turn (vol_arb_live.dependency_hashes from each of this module's top-level
    names but this checker; this module's own lines are left out). read_source(module) returns lib/<module>.py's
    text. far_wings_orders.py takes its lib helpers through this module, so they are in it too.
    """
    me = __name__.rsplit(".", 1)[-1]
    defs = VL._module_index(read_source(me))[0]
    got = VL.dependency_hashes(read_source, roots=tuple((me, n) for n in sorted(defs) if n != "lib_functions"))
    return {k: v for k, v in got.items() if not k.startswith(me + ".")}


def hash_differences(pinned: dict, got: dict) -> list[str]:
    """What differs between the recorded lib hashes and tonight's: changed, new, or no longer used."""
    return VL.hash_differences(pinned, got)
