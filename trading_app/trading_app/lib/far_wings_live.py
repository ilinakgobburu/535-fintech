"""
The far-wing trade, live: tonight's NVDA orders for the course server (Sanji).

The trade (her choices of 2026-11-16): once a month sell one far put (|delta| under 0.10) and one far call (|delta|
under 0.05), delta-hedged with the stock only when the book's net delta leaves a band ("Hedge less often (delta
bands)"), no crash cap ("Hedge only"), each option sold with a limit at the mid ("Yes, limits at the mid"). This
becomes her class algorithm ("Yes, switch to far wings"): it replaces the two-sided variance trade as the one nightly
writer of mkt/open_orders.json. This module does no I/O: far_wings_orders.py at the repo root pulls LSEG, calls these
functions and writes the file. It reuses vol_arb_live's calendar, quote, book and order helpers (each tested there)
and adds the far-wing rules; every lib function it uses outside this file is pinned by hash in
config/far_wings_live.yaml (lib_functions), so an edit to one of them stops the nightly run instead of changing it.

WHAT IS KNOWN WHEN. The script runs after the close of session D (wheel_live.decision_dates) and writes orders for
the next session, `for`, which Sanji fills at that session's prices: a market order at the open, a limit sell when
the bar's high reaches the limit.

ENTRY NIGHTS are vol-arb's calendar: the first session after each monthly expiry (vol_arb_live.entry_sessions, on
wheel_live's NYSE calendar), each position held to the next monthly expiry. On an entry night, from D's closing
quotes of that expiry's NVDA chain (two-sided as vol_arb.two_sided has it: bid and ask there, ask at or above bid,
mid above zero, one RIC per strike):

    put wing    among the puts whose |delta| is under 0.10 (wheel.option_greeks on each one's own mid, at D's spot,
                T in calendar days / 365, the T-bill and the trailing year's dividend yield: wheel.carry), the one
                whose |delta| is closest to 0.05; a tie goes to the lower strike
    call wing   among the calls whose |delta| is under 0.05, the one closest to 0.03; a tie goes to the higher strike
    size        each wing round(200 / its Black-76 vega per vol point per contract, with D's lot), at least 1
                (vol_arb.leg_vega and contracts_for: about $200 a vol point per wing on the $100,000 book)
    orders      SHT each wing, a limit at D's closing mid (to the cent), bare-number strike and price strings, the
                expiry as a date; ids nvda-fw-put-<D> and nvda-fw-call-<D>

No eligible strike is no position in that wing this cycle (said in the log); the other wing still trades. A wing whose
mid rounds to a limit of 0 is not sold either. That rule is for a chain that is there: a right with D rows but not
one two-sided quote (one_sided_problem: D's bids not posted yet) stops the 22:00 run as data not final, and only the
01:00 retry, finding it so still, takes it as no eligible strike.

EVERY NIGHT, THE BAND. The book's net delta in shares: each held option's delta at D's close (its own closing mid's
implied vol; a leg with no usable mid that close keeps its last IV, delta recomputed at D's spot and T:
vol_arb_live.leg_delta) times its contracts and the lot, plus the shares held. When |net delta x D's close| is above
the band (5% of the $100,000 book, $5,000; at exactly $5,000 nothing is done), a market order for the next open moves
the shares to -round(the options' delta), zero shares-delta; otherwise no hedge trade. One order cannot flip a long
into a short, so crossing zero is two orders, the close first (ids nvda-fw-hedge-sel-<D> and nvda-fw-hedge-sht-<D>;
one order is nvda-fw-hedge-<D>). The first check is the first night the options are in the book (a limit may not
fill). With no options held, the shares are closed (after an expiry, or after Sanji assigned both wings).

EXPIRY. Sanji settles the options on the expiry's close (an assignment delivers shares). The night of the expiry the
book holds no options, so the leftover hedge and any delivered shares are closed at the next open; the next night
is the next entry. An option still listed on or after its expiry stops the run (the book is not up to date).

WHOSE BOOK, AND A STALE BOOK: as vol_arb_live has them, for this strategy's ids (nvda-fw-...). Every NVDA option in
the book must be exactly what this strategy's filled orders add up to (its ids in mkt/fills.json, looked up in
data/far_wings_ledger.json, which keeps every order it wrote), a short leg excepted when Sanji has assigned it; NVDA
shares need a fill of its own (book_problems). Once mkt/fills.json exists, an order of this strategy's for a session
up to D with no result, or one Sanji rejected, stops the run until she lists it in config's `acknowledged`
(guard_problems). Another strategy's orders are only noted. The hedge target and the shares held may not pass
limits.max_hedge_x_capital times the book (hedge_limit), every night shares or options are held.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from . import vol_arb as VA
from . import vol_arb_live as VL
from . import wheel as W
from . import wheel_live as L
from .ric import build_option_ric

PUT, CALL = W.PUT, W.CALL
LOT = W.CONTRACT                          # shares per NVDA contract; no split since 2024-06-10
TAG = "fw"                                # in every order id: nvda-fw-put-2026-11-23
CAP = {PUT: 0.10, CALL: 0.05}             # a wing's strikes: |delta| strictly under this
TARGET = {PUT: 0.05, CALL: 0.03}          # ...the one whose |delta| is closest to this
VEGA_DOLLARS = 200.0                      # each wing: round(200 / vega per vol point per contract), at least 1
CAPITAL = 100_000.0                       # the book the $200 a vol point and the band are sized on
BAND_SHARE = 0.05                         # re-hedge when |net delta x close| > 5% of the book
LEDGER_FORMAT = "far_wings_ledger/1"
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
decided, decision_dates, merge_orders, BookFormatError = VL.decided, L.decision_dates, L.merge_orders, L.BookFormatError
transition, leg_delta, sha256_hex, option_ric = VL.transition, VL.leg_delta, VL.sha256_hex, build_option_ric


class StaleBook(ValueError):
    """The book is not up to date with what has happened: no order can be planned from it."""


class LedgerError(ValueError):
    """data/far_wings_ledger.json is malformed, or a fill of this strategy's is not in it."""


def _d(x) -> dt.date:
    return pd.Timestamp(x).date()


def band_dollars() -> float:
    """The band: 5% of the $100,000 book, $5,000."""
    return BAND_SHARE * CAPITAL


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


def hedge_limit(cfg: dict, spot: float, target: int, shares: int) -> None:
    """
    The hedge's target and the shares held, each times D's close, may not pass limits.max_hedge_x_capital times the
    $100,000 book. The wings are sized $200 a vol point, so a 0.03-delta call deep in the money after a large rise can
    need a hedge of about 15 to 25 times the book: vol-arb's 5 would stop the run exactly when the hedge matters.
    Shares are not reconciled with the fills (an assignment delivers them), so this is their guard against a misread
    book (a 100-times reading of a normal hedge is far past it).
    """
    x = float(cfg["limits"]["max_hedge_x_capital"])
    for n, what in ((target, "hedge target"), (shares, "shares held")):
        if abs(n) * spot > x * CAPITAL:
            raise ValueError(f"the {what}, {n:+d} NVDA shares (${abs(n) * spot:,.0f} at {spot:.2f}), is more than "
                             f"{x:g} x the ${CAPITAL:,.0f} book: the book is probably misread")


# --------------------------------------------------------------------------- the orders
def own_prefix(root: str) -> str:
    """Every id this strategy writes starts with it: 'nvda-fw-'."""
    return f"{root.lower()}-{TAG}-"


def _id(root: str, what: str, action: str | None, day) -> str:
    return f"{own_prefix(root)}{what}{'-' + action.lower() if action else ''}-{_d(day).isoformat()}"


def stock_orders(*, root: str, conid: int, day, for_day, held: int, target: int) -> list[dict]:
    """Market orders for the next open that move the shares from `held` to `target`; ids nvda-fw-hedge-<D>."""
    steps = transition(held, target)
    return [{"id": _id(root, "hedge", a if len(steps) > 1 else None, day), "for": _d(for_day).isoformat(),
             "order_type": "mkt", "action": a, "qty": int(n), "underlying": int(conid), "contract": "underlying",
             "ref": "open"} for a, n in steps]


def option_orders(*, root: str, conid: int, day, for_day, cp: str, expiry, strike: float, limit: str,
                  held: int, target: int) -> list[dict]:
    """Limit orders at `limit` (a bare-number token) moving one contract from `held` to `target`; nvda-fw-put-<D>."""
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
    the coming orders, so what this strategy wrote earlier, and so what its fills were, is kept here):

        {"format": "far_wings_ledger/1",
         "orders": {"nvda-fw-put-2026-11-23": {the order as written}, ...},
         "nights": {"2026-11-23": {"for": "2026-11-24", "ids": ["nvda-fw-put-2026-11-23", ...]}}}

    None (no file yet) is an empty ledger.
    """
    if obj is None:
        return {"format": LEDGER_FORMAT, "orders": {}, "nights": {}}
    if (not isinstance(obj, dict) or obj.get("format") != LEDGER_FORMAT or not isinstance(obj.get("orders"), dict)
            or not isinstance(obj.get("nights"), dict)):
        raise LedgerError(f"{LEDGER_PATH} is not a {LEDGER_FORMAT} file")
    return obj


def record_night(ledger: dict, new: list[dict], day, for_day) -> dict:
    """The ledger after tonight's orders were written: each by id (a rerun's version replaces the earlier one)."""
    out = {"format": LEDGER_FORMAT, "orders": dict(ledger["orders"]), "nights": dict(ledger["nights"])}
    for o in new:
        out["orders"][o["id"]] = o
    out["nights"][_d(day).isoformat()] = {"for": _d(for_day).isoformat(), "ids": [o["id"] for o in new]}
    return out


def strategy_options(ledger: dict, fills: list | None, root: str, day) -> tuple[dict, bool]:
    """
    ({(cp, expiry, strike): contracts}, any fill): what this strategy's filled option orders add up to for the
    expiries after D, and whether any of its orders, a hedge included, has filled (Sanji fills all or nothing).
    """
    prefix = own_prefix(root)
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


def guard_problems(written: list[dict], fills: list | None, day, root: str, acknowledged: dict) -> tuple[list, list]:
    """
    (why the run must stop, notes): an order Sanji rejected, or one written for a session up to D with no result
    once mkt/fills.json exists. Only this strategy's orders stop the run, and only until she lists the id in
    config's `acknowledged`; another strategy's (the wheel's last put, a vol-arb order) is noted.
    """
    prefix, stops, notes = own_prefix(root), [], []
    for f in L.rejected_orders(written, fills):
        i = f.get("id")
        if not str(i).startswith(prefix):
            notes.append(f"Sanji rejected {i}, which is not this strategy's: {f}")
        elif i in acknowledged:
            notes.append(f"Sanji rejected {i}; acknowledged in {CONFIG_PATH}: {acknowledged[i]}")
        else:
            stops.append(f"Sanji rejected {i}: {f}. Fix the order format; to trade on, add {i} to "
                         f"{CONFIG_PATH}'s acknowledged with the reason")
    waiting = L.unresolved_orders(written, fills, _d(day))
    for i in [i for i in waiting if not str(i).startswith(prefix)]:
        notes.append(f"mkt/fills.json has no result for {i}, which is not this strategy's")
    for i in [i for i in waiting if str(i).startswith(prefix) and i in acknowledged]:
        notes.append(f"mkt/fills.json has no result for {i}; acknowledged in {CONFIG_PATH}: {acknowledged[i]}")
    mine = [i for i in waiting if str(i).startswith(prefix) and i not in acknowledged]
    if mine:
        stops.append(f"mkt/fills.json has no result yet for {', '.join(mine)}: the book may be stale. If Sanji has "
                     f"resolved the session and will not answer, add the id to {CONFIG_PATH}'s acknowledged with "
                     f"the reason")
    return stops, notes


def _leg(key) -> str:
    cp, expiry, strike = key
    return f"{expiry} {strike:g} {'put' if cp == PUT else 'call'}"


def book_problems(book: Book, fills: list | None, ledger: dict, root: str, day) -> list[str]:
    """
    Why the book's NVDA positions are not (all) this strategy's, or not up to date with its fills (empty when they
    are): every option held must be exactly what its filled orders add up to, except that a short leg may be gone
    (Sanji assigns a short option whole); a long leg its fills hold must be there; shares need at least one fill of
    its own. With no mkt/fills.json at all, a book with positions can't be checked against anything.
    """
    held = held_contracts(book)
    if fills is None:
        if held or book.shares:
            return ["mkt/positions.json lists NVDA positions but there is no mkt/fills.json, so it can't be told "
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
        out.append(f"the book holds {book.shares:+d} NVDA shares, but no {own_prefix(root)}* order has filled")
    return out


# --------------------------------------------------------------------------- the night
def plan_night(cfg: dict, day, for_day, book: Book, *, entry_expiry=None, mkt: dict | None = None,
               leg_history: dict | None = None) -> dict:
    """
    Tonight's orders. The band, every night options are held: the net delta (options plus shares) at D's close, and
    a market order to -round(the options' delta) only when it is outside the band; with no options held, the shares
    are closed. On an entry night (`entry_expiry` set): the two wings' SHT limit orders at D's closing mid.
    `leg_history` maps each held contract (cp, expiry, strike) to its daily bid/ask. `mkt` (D's close at least) is
    needed whenever options or shares are held, and on an entry night.
    """
    u = cfg["underlying"]
    root, conid = str(u["root"]), int(u["conid"])
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
        hedge_limit(cfg, spot, hedge["target"], book.shares)
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


def check_config(cfg: dict) -> None:
    """config/far_wings_live.yaml's registered numbers must be this module's, and its other settings well formed."""
    u = cfg.get("underlying") or {}
    if type(u.get("conid")) is not int or not u.get("root") or not u.get("ric"):
        raise ValueError("underlying needs root, ric and an integer conid")
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
