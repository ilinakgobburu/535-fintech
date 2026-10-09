"""
The poor man's covered call: a covered call whose 100 shares are replaced by
one deep in-the-money, long-dated call.

It is deliberately the SAME book as lib.covered_call with one leg swapped. The
calendar, the order bar, the short-strike rules, the settlement price and the
fill convention are imported from there, so a difference between the two
books' results is a difference between owning stock and owning a LEAP, and
cannot be a difference between two implementations.

THE RULES THIS IMPLEMENTS
-------------------------
long leg   at the first order bar, buy 1 call in the earliest listed expiry
           at least `min_dte` days out, at the listed strike whose delta is
           closest to `target_delta`. Fill at mid. Held; replaced by the same
           rule once it has fewer than `roll_dte` days left.
short leg  same bar, sell 1 call expiring at the end of that week, by the
           same strike rule the covered call uses -- then raised, if need be,
           to the long leg's break-even (long strike + long debit). Below that
           strike an assignment locks in a loss on the long leg.
no quote   no two-sided quote on a leg -> that leg is not traded. The first
           week needs both; later weeks hold the LEAP uncovered and say so.
expiry     the official close, as in the covered call.
             S_T <= K  the short call expires.
             S_T >  K  assigned. The book never owned shares, so delivery is a
                       short sale of 100 at the strike. NOT flat: short 100
                       shares beside the LEAP over the weekend.
cover      next order bar: buy the 100 shares back at the stock print, then
           write the next call. The LEAP is not exercised -- exercising throws
           away its remaining time value.

`close_itm=True` is the alternative: at the last bar of expiry day, buy the
short call back if it is in the money, so the book is never assigned.

REG T
-----
There is no stock market value to take 50% of. A long option has no loan
value: it is paid for in full. The short call needs no margin of its own while
the long call covers it (expires later, strike at or below the short's).

    NAV       = cash + long call MV + short call MV + stock MV
    options   = long call MV + short call MV     (net; carried at 100%)
    initial   = options + 0.50 x |stock MV|      stock MV is short stock, after
    maint     = options + 0.30 x |stock MV|      an assignment; otherwise zero
    available = NAV - initial
    excess    = NAV - maint

With no stock that reduces to available = cash: the account can spend its
cash and nothing else. Short stock is margined at the plain rates, with no
relief for the long call that hedges it.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .covered_call import (
    ASSIGN, BUY, EXPIRE, INITIAL_RATE, SELL, SHARES_PER_CONTRACT, STRIKE_RULES,
    STRIKE_RULE_META, _bar_at, _last_bar, atm_iv, settlement, stock_marks,
    years_to_expiry,
)
from .ric import occ_symbol, parse_option_ric
from .vol import bs_greeks, bs_price, implied_vol

SHORT_STOCK_MAINT_RATE = 0.30   # FINRA 4210 maintenance on short stock

TARGET_DELTA = 0.80
MIN_DTE = 270
ROLL_DTE = 90

LONG, SHORT = "long", "short"
GREEK_KEYS = ("delta", "gamma", "theta", "vega")


# --------------------------------------------------------------------------
# greeks
# --------------------------------------------------------------------------

def call_greeks(mid: float, spot: float, strike: float, T: float,
                r: float = 0.0, q: float = 0.0) -> dict:
    """
    Implied vol of one quoted call and its sensitivities to SPOT, per share.

        delta   per $1 of stock
        gamma   delta per $1 of stock
        vega    $ per vol POINT (0.01 of sigma)
        theta   $ per calendar day, stock and vol held still

    The pricing is lib.vol's Black-76 with the carry made explicit:
    F = S exp((r-q)T), D = exp(-rT). The covered call sets F = S, D = 1, which
    is right for a week and wrong for a year: at 4% the forward of a one-year
    option sits 4% above spot, and a deep in-the-money call priced without it
    has no implied vol at all.

    A quote with no implied vol -- below intrinsic, or inside the last day --
    returns iv = NaN with delta 1 or 0 by moneyness and the rest zero, which is
    what the option has become.
    """
    itm = bool(np.isfinite(spot) and spot > strike)
    out = {"iv": np.nan, "delta": 1.0 if itm else 0.0,
           "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    if not np.isfinite([mid, spot, strike, T]).all() or T <= 0 or spot <= 0:
        return out
    F, D = spot * np.exp((r - q) * T), np.exp(-r * T)
    iv = implied_vol(mid, F, strike, T, D, "C")
    if not np.isfinite(iv):
        return out
    g = bs_greeks(F, strike, T, iv, D, "C")
    T1 = T - 1.0 / 365.0
    tomorrow = (bs_price(spot * np.exp((r - q) * T1), strike, T1, iv,
                         np.exp(-r * T1), "C") if T1 > 0 else max(spot - strike, 0.0))
    return {
        "iv": float(iv),
        "delta": g["delta"] * F / spot,
        "gamma": g["gamma"] * (F / spot) ** 2,
        "vega": g["vega"] / 100.0,
        "theta": float(tomorrow - mid),
    }


# --------------------------------------------------------------------------
# leg selection
# --------------------------------------------------------------------------

def select_long(chain: pd.DataFrame, spot: float, ts, *,
                target_delta: float = TARGET_DELTA, min_dte: int = MIN_DTE,
                r: float = 0.0, q: float = 0.0) -> dict | None:
    """
    The long leg, from every call quoted at this bar.

    Expiry first, then strike. The expiry is the earliest one at least
    `min_dte` days away. The strike is the one whose delta, computed from its
    own mid at this bar, is closest to `target_delta`; a tie goes to the lower
    strike, the deeper one. Only two-sided quotes are candidates.
    """
    today = pd.Timestamp(ts).date()
    live = chain[chain["mid"].notna() & (chain["mid"] > 0)]
    far = live[[(e - today).days >= min_dte for e in live["expiry"]]]
    if far.empty:
        return None
    expiry = min(far["expiry"])
    T = years_to_expiry(ts, expiry)
    best = None
    for row in far[far["expiry"] == expiry].itertuples():
        g = call_greeks(float(row.mid), spot, float(row.strike), T, r, q)
        if not np.isfinite(g["iv"]):
            continue
        key = (abs(g["delta"] - target_delta), float(row.strike))
        if best is None or key < best[0]:
            best = (key, row, g)
    if best is None:
        return None
    _, row, g = best
    return {"ric": row.ric, "strike": float(row.strike), "expiry": expiry,
            "bid": float(row.bid), "ask": float(row.ask), "mid": float(row.mid),
            "T_years": T, **g}


def strike_for_delta(spot: float, sigma: float, T: float, target: float) -> float:
    """
    The strike whose Black-76 call delta (F = S, D = 1) is exactly `target`:

        N(d1) = target  =>  K = S * exp( s sqrt(T) * N^-1(1 - target) + s^2 T / 2 )
    """
    from scipy.special import ndtri

    return float(spot * np.exp(sigma * np.sqrt(T) * ndtri(1.0 - target)
                               + 0.5 * sigma * sigma * T))


def by_delta(chain: pd.DataFrame, spot: float, *,
             target: float = 0.30, T: float | None = None, **_):
    """
    The lowest listed strike whose call delta is at or under `target` -- the
    desk convention, "sell the 30-delta call". Delta comes from the same
    median at-the-money vol by_assignment_prob uses. Floored at spot.
    """
    if T is None or not np.isfinite(T) or T <= 0:
        return None
    sigma = atm_iv(chain, spot, T)
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    k = max(strike_for_delta(spot, sigma, T, target), spot)
    ks = np.sort(chain["strike"].unique())
    hit = ks[ks >= k - 1e-9]
    return float(hit[0]) if len(hit) else None


# The covered call's rules plus the delta rule. A separate table rather than a
# new entry in STRIKE_RULES, because the published 1.2 page sweeps that one.
SHORT_RULES = {
    **STRIKE_RULES,
    "delta_30": lambda c, s, **k: by_delta(c, s, target=0.30, **k),
}
SHORT_RULE_META = {
    **STRIKE_RULE_META,
    "delta_30": {"label": "first strike at or under 0.30 delta", "target_prob": None},
}


def short_strike(chain: pd.DataFrame, spot: float, T: float, rule: str,
                 floor: float | None) -> tuple[float | None, bool]:
    """
    (strike, floored): the rule's strike, raised to the first listed strike at
    or above `floor` when the rule lands below it.
    """
    strike = SHORT_RULES[rule](chain, float(spot), T=T) if len(chain) else None
    if strike is None or floor is None or strike >= floor - 1e-9:
        return strike, False
    ks = np.sort(chain["strike"].unique())
    hit = ks[ks >= floor - 1e-9]
    return (float(hit[0]) if len(hit) else None), True


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

def _price(bid: float, ask: float, side: str, fill: str) -> float:
    """Mid, or the far side of the quote: buy at the ask, sell at the bid."""
    if fill == "cross":
        return float(ask if side == BUY else bid)
    return (float(bid) + float(ask)) / 2.0


def run_pmcc(
    stock: pd.DataFrame,
    options: pd.DataFrame,
    weeks: list[dict],
    *,
    rule: str = "nearest_otm",
    order_hour: int = 16,
    start_cash: float = 0.0,
    target_delta: float = TARGET_DELTA,
    min_dte: int = MIN_DTE,
    roll_dte: int = ROLL_DTE,
    r: float = 0.0,
    q: float = 0.0,
    close_itm: bool = False,
    breakeven_floor: bool = True,
    fill: str = "mid",
) -> dict:
    """
    Walk the weeks and book what the rules say. Returns the blotter and the
    per-week decision record; the ledger is built from the blotter afterwards.
    """
    stock_ric = stock.attrs.get("ric", "AAPL.O")
    qty = SHARES_PER_CONTRACT
    blotter: list[dict] = []
    cycles: list[dict] = []
    shares = 0
    long = None            # dict(ric, strike, expiry, debit) while held

    def buy_long(ts, spot, bar, why):
        pick = select_long(bar, float(spot), ts, target_delta=target_delta,
                           min_dte=min_dte, r=r, q=q)
        if pick is None:
            return None
        px = _price(pick["bid"], pick["ask"], BUY, fill)
        occ = occ_symbol(parse_option_ric(pick["ric"])["underlying"],
                         pick["expiry"], pick["strike"], "C")
        blotter.append({
            "ts": ts, "instrument": pick["ric"], "occ": occ, "kind": "call",
            "leg": LONG, "side": BUY, "qty": 1, "limit": round(px, 4),
            "fill": round(px, 4), "cash_delta": -qty * px,
            "strike": pick["strike"], "expiry": pick["expiry"],
            "bid": pick["bid"], "ask": pick["ask"],
            "note": (f"{why}: earliest expiry >= {min_dte} days out, strike "
                     f"with delta closest to {target_delta:.2f} (delta "
                     f"{pick['delta']:.2f}, implied vol {100 * pick['iv']:.1f}%); "
                     f"quote {pick['bid']:.2f}/{pick['ask']:.2f}"),
        })
        return {**pick, "occ": occ, "debit": px}

    for w in weeks:
        entry_day, expiry_day = w["entry_date"], w["expiry_date"]
        ts = _bar_at(stock, entry_day, order_hour)
        if ts is None:
            continue
        spot = stock.at[ts, "TRDPRC_1"] if "TRDPRC_1" in stock.columns else np.nan
        if not np.isfinite(spot):
            cycles.append({**w, "status": "no stock print", "order_ts": str(ts)})
            continue
        spot = float(spot)
        bar = options[options["ts"] == ts]
        rec = {**w, "order_ts": str(ts), "spot": spot}

        # ---- cover last week's assignment ----------------------------------
        if shares < 0:
            blotter.append({
                "ts": ts, "instrument": stock_ric, "kind": "stock", "side": BUY,
                "qty": -shares, "limit": spot, "fill": spot,
                "cash_delta": shares * spot,
                "note": f"cover: buy back the {-shares} shares sold short by "
                        f"last week's assignment, at the stock print; the "
                        f"LEAP is not exercised",
            })
            rec["cover_px"] = spot
            shares = 0

        # ---- long leg --------------------------------------------------------
        chain = bar[bar["expiry"] == expiry_day]
        T = years_to_expiry(ts, expiry_day)
        if long is None:
            # The diagonal is one decision. Without a short call to sell this
            # week there is no reason to open it this week.
            first, _ = short_strike(chain, spot, T, rule, None)
            priced = first is not None and np.isfinite(
                chain.loc[np.isclose(chain["strike"], first), "mid"]).any()
            long = buy_long(ts, spot, bar, "long leg") if priced else None
            if long is None:
                cycles.append({**rec, "status": "skipped: no two-sided quote"})
                continue
        elif (long["expiry"] - entry_day).days < roll_dte:
            held = bar[bar["ric"] == long["ric"]]
            if len(held) and np.isfinite(held["mid"].iloc[0]):
                h = held.iloc[0]
                px = _price(h["bid"], h["ask"], SELL, fill)
                blotter.append({
                    "ts": ts, "instrument": long["ric"], "occ": long["occ"],
                    "kind": "call", "leg": LONG, "side": SELL, "qty": -1,
                    "limit": round(px, 4), "fill": round(px, 4),
                    "cash_delta": qty * px, "strike": long["strike"],
                    "expiry": long["expiry"], "bid": float(h["bid"]),
                    "ask": float(h["ask"]),
                    "note": f"roll: long leg has under {roll_dte} days left; "
                            f"sold, quote {h['bid']:.2f}/{h['ask']:.2f}",
                })
                fresh = buy_long(ts, spot, bar, "roll")
                if fresh is None:
                    cycles.append({**rec, "status": "closed: no long leg to roll into"})
                    long = None
                    continue
                long = fresh

        # ---- short leg -------------------------------------------------------
        floor = long["strike"] + long["debit"] if breakeven_floor else None
        strike, floored = short_strike(chain, spot, T, rule, floor)
        row = chain[np.isclose(chain["strike"], strike)] if strike is not None else chain.iloc[:0]
        if not len(row) or not np.isfinite(row["mid"].iloc[0]):
            cycles.append({
                **rec, "status": "skipped: no two-sided quote",
                "strike": strike, "long_strike": long["strike"],
                "held_uncovered": True,
            })
            continue
        s = row.iloc[0]
        px = _price(s["bid"], s["ask"], SELL, fill)
        occ = occ_symbol(parse_option_ric(s["ric"])["underlying"], expiry_day,
                         float(strike), "C")
        short = {"instrument": s["ric"], "occ": occ, "kind": "call", "leg": SHORT,
                 "strike": float(strike), "expiry": expiry_day}
        blotter.append({
            **short, "ts": ts, "side": SELL, "qty": -1,
            "limit": round(px, 4), "fill": round(px, 4), "cash_delta": qty * px,
            "bid": float(s["bid"]), "ask": float(s["ask"]),
            "note": (f"write: {rule} at spot {spot:.2f}"
                     + (f", raised to the long leg's break-even {floor:.2f}"
                        if floored else "")
                     + f"; quote {s['bid']:.2f}/{s['ask']:.2f}; covered by the "
                       f"long {long['strike']:.2f} call"),
        })
        rec.update({
            "strike": float(strike), "floored": floored, "floor": floor,
            "mid": (float(s["bid"]) + float(s["ask"])) / 2.0, "fill": px,
            "bid": float(s["bid"]), "ask": float(s["ask"]), "premium": qty * px,
            "otm_pct": 100.0 * (strike / spot - 1.0), "T_years": T,
            "atm_iv": atm_iv(chain, spot, T),
            "long_strike": long["strike"], "long_expiry": long["expiry"],
        })

        # ---- expiry ------------------------------------------------------------
        ets, s_t, source = settlement(stock, expiry_day)
        if not np.isfinite(s_t):
            cycles.append({**rec, "status": "no expiry print"})
            continue
        rec.update({"settle": s_t, "settle_source": source})

        if close_itm:
            last = float(stock.at[ets, "TRDPRC_1"])
            quote = options[(options["ts"] == ets) & (options["ric"] == s["ric"])]
            if last > strike and len(quote) and np.isfinite(quote["mid"].iloc[0]):
                c = quote.iloc[0]
                bpx = _price(c["bid"], c["ask"], BUY, fill)
                blotter.append({
                    **short, "ts": ets, "side": BUY, "qty": 1,
                    "limit": round(bpx, 4), "fill": round(bpx, 4),
                    "cash_delta": -qty * bpx, "bid": float(c["bid"]),
                    "ask": float(c["ask"]),
                    "note": f"buy to close: last trade {last:.2f} > strike "
                            f"{strike:.2f} in the final bar; quote "
                            f"{c['bid']:.2f}/{c['ask']:.2f}; not assigned",
                })
                cycles.append({**rec, "status": "closed", "close_px": bpx})
                continue

        if s_t > strike:   # ties expire, as in the covered call
            blotter.append({
                **short, "ts": ets, "side": ASSIGN, "qty": 1, "limit": None,
                "fill": float(strike), "cash_delta": 0.0,
                "note": f"assigned: {source} {s_t:.2f} > strike {strike:.2f}; "
                        f"short call closed by assignment. No shares are held, "
                        f"so the next row sells {qty} short at the strike",
            })
            blotter.append({
                "ts": ets, "instrument": stock_ric, "kind": "stock", "side": SELL,
                "qty": qty, "limit": None, "fill": float(strike),
                "cash_delta": qty * float(strike),
                "note": f"delivered against assignment: {qty} shares sold "
                        f"short at the strike {strike:.2f}; short {qty} shares "
                        f"and still long the LEAP, to be covered next session",
            })
            shares -= qty
            cycles.append({**rec, "status": "assigned"})
        else:
            blotter.append({
                **short, "ts": ets, "side": EXPIRE, "qty": 1, "limit": None,
                "fill": 0.0, "cash_delta": 0.0,
                "note": f"expired: {source} {s_t:.2f} <= strike {strike:.2f}; "
                        f"keep the premium, keep the LEAP",
            })
            cycles.append({**rec, "status": "expired"})

    return {"blotter": blotter, "cycles": cycles, "rule": rule,
            "order_hour": order_hour, "start_cash": start_cash,
            "target_delta": target_delta, "min_dte": min_dte, "roll_dte": roll_dte,
            "r": r, "q": q, "close_itm": close_itm, "fill": fill}


def first_debit(run: dict) -> float:
    """Cash the first order bar consumed: the long debit less the first premium."""
    if not run["blotter"]:
        return 0.0
    t0 = min(ev["ts"] for ev in run["blotter"])
    return -sum(ev["cash_delta"] for ev in run["blotter"] if ev["ts"] == t0)


# --------------------------------------------------------------------------
# ledger + Reg T
# --------------------------------------------------------------------------

def build_pmcc_ledger(run: dict, stock: pd.DataFrame, options: pd.DataFrame,
                      greeks: bool = True) -> pd.DataFrame:
    """
    Replay the blotter across every bar and mark the book.

    Derived from the blotter, as the covered call's ledger is, so the two
    cannot drift. Each option is marked at its mid at that bar, or at its last
    mid when the bar has no two-sided quote. Greeks are recomputed only on a
    fresh quote, from that quote and the same bar's stock price.
    """
    blotter = sorted(run["blotter"], key=lambda b: b["ts"])   # stable: booked order
    cash = float(run["start_cash"])
    r, q = float(run.get("r", 0.0)), float(run.get("q", 0.0))
    shares = 0
    legs = {LONG: None, SHORT: None}
    last_mark = {LONG: np.nan, SHORT: np.nan}
    last_greeks = {LONG: None, SHORT: None}
    zero = {"iv": np.nan, **dict.fromkeys(GREEK_KEYS, 0.0)}

    held = {ev["instrument"] for ev in blotter if ev["kind"] == "call"}
    quotes = options[options["ric"].isin(held)]
    mid_by = {(x.ric, x.ts): x.mid for x in quotes.itertuples()}
    marks = stock_marks(stock)

    rows, bi, last_stock = [], 0, np.nan
    for ts in stock.index:
        while bi < len(blotter) and blotter[bi]["ts"] <= ts:
            ev = blotter[bi]
            cash += ev["cash_delta"]
            if ev["kind"] == "stock":
                shares += ev["qty"] if ev["side"] == BUY else -ev["qty"]
            else:
                leg = ev["leg"]
                opening = ev["side"] == (BUY if leg == LONG else SELL)
                legs[leg] = ({"ric": ev["instrument"], "strike": ev["strike"],
                              "expiry": ev["expiry"]} if opening else None)
                last_mark[leg] = ev["fill"] if opening else np.nan
                last_greeks[leg] = None
            bi += 1

        px = marks.at[ts]
        if np.isfinite(px):
            last_stock = float(px)

        mv, g = {}, {}
        for leg, sign in ((LONG, 1), (SHORT, -1)):
            pos = legs[leg]
            if pos is None:
                mv[leg], g[leg] = 0.0, zero
                continue
            m = mid_by.get((pos["ric"], ts), np.nan)
            fresh = m is not None and np.isfinite(m)
            if fresh:
                last_mark[leg] = float(m)
            mv[leg] = sign * SHARES_PER_CONTRACT * last_mark[leg]
            if greeks and (fresh or last_greeks[leg] is None):
                last_greeks[leg] = call_greeks(
                    last_mark[leg], last_stock, pos["strike"],
                    years_to_expiry(ts, pos["expiry"]), r, q)
            g[leg] = last_greeks[leg] or zero

        stock_mv = shares * last_stock if shares else 0.0
        nav = cash + stock_mv + mv[LONG] + mv[SHORT]
        option_req = mv[LONG] + mv[SHORT]
        if legs[LONG] and legs[SHORT]:
            # A short struck below the long is covered only up to the gap.
            option_req += SHARES_PER_CONTRACT * max(
                legs[LONG]["strike"] - legs[SHORT]["strike"], 0.0)
        im = option_req + INITIAL_RATE * abs(stock_mv)
        mm = option_req + SHORT_STOCK_MAINT_RATE * abs(stock_mv)

        row = {
            "ts": ts, "cash": cash, "shares": shares,
            "stock_mark": last_stock, "stock_mv": stock_mv,
            "nav": nav, "option_requirement": option_req,
            "initial_margin": im, "maintenance_margin": mm,
            "available_funds": nav - im, "excess_liquidity": nav - mm,
        }
        for leg in (LONG, SHORT):
            pos = legs[leg]
            row[f"{leg}_ric"] = pos["ric"] if pos else None
            row[f"{leg}_strike"] = pos["strike"] if pos else np.nan
            row[f"{leg}_expiry"] = str(pos["expiry"]) if pos else None
            row[f"{leg}_mark"] = last_mark[leg] if pos else np.nan
            row[f"{leg}_mv"] = mv[leg]
            row[f"{leg}_iv"] = g[leg]["iv"] if pos else np.nan
            row[f"{leg}_delta"] = g[leg]["delta"] if pos else np.nan
        for k in GREEK_KEYS:
            row[k] = (SHARES_PER_CONTRACT * (g[LONG][k] - g[SHORT][k])
                      + (shares if k == "delta" else 0.0))
        rows.append(row)

    led = pd.DataFrame(rows)
    if blotter:
        led = led[led["ts"] >= blotter[0]["ts"]].reset_index(drop=True)
    led["feasible"] = led["available_funds"] >= -1e-9
    return led


def min_start_cash(run: dict, ledger: pd.DataFrame) -> dict:
    """
    Smallest starting cash for which available funds never go negative.

    Available funds move one for one with starting cash -- a dollar of cash is
    a dollar of NAV and changes no requirement -- so this is one subtraction
    from the ledger already built.
    """
    i = int(ledger["available_funds"].idxmin())
    worst = float(ledger.at[i, "available_funds"])
    return {"min_cash": float(run["start_cash"]) - min(worst, 0.0),
            "worst_available": worst, "worst_ts": str(ledger.at[i, "ts"]),
            "binds": worst < 0}


# --------------------------------------------------------------------------
# reading the result
# --------------------------------------------------------------------------

def pnl_by_leg(run: dict, ledger: pd.DataFrame) -> dict:
    """
    Split the P&L three ways: the LEAP, the short calls, and the stock bought
    and sold around assignments. Each is its cash flows plus its closing
    market value, so the three sum to NAV change exactly.
    """
    cash = {LONG: 0.0, SHORT: 0.0, "stock": 0.0}
    for ev in run["blotter"]:
        cash[ev.get("leg", "stock")] += ev["cash_delta"]
    end = ledger.iloc[-1]
    return {
        "long": cash[LONG] + float(end["long_mv"]),
        "short": cash[SHORT] + float(end["short_mv"]),
        "stock": cash["stock"] + float(end["stock_mv"]),
    }


def daily_closes(ledger: pd.DataFrame, col: str = "nav") -> pd.Series:
    """The last bar of each session, indexed by date."""
    s = ledger.set_index("ts")[col]
    return s.groupby([t.date() for t in s.index]).last()


def performance(nav: pd.Series, capital: float, weeks: list[dict]) -> dict:
    """
    Return on `capital`, worst peak-to-trough fall of the daily closing NAV,
    and the weekly simple returns' mean, standard deviation and their ratio
    (a weekly Sharpe ratio with a zero risk-free rate, not annualised).
    Weeks are the book's own: last session of one to last session of the next.
    """
    nav = nav.dropna()
    ends = [w["expiry_date"] for w in weeks if w["expiry_date"] in nav.index]
    pts = pd.concat([pd.Series([capital]), nav.reindex(ends)], ignore_index=True)
    wk = pts.pct_change().dropna()
    path = pd.concat([pd.Series([capital]), nav], ignore_index=True)
    dd = float((path / path.cummax() - 1.0).min())
    sd = float(wk.std(ddof=1)) if len(wk) > 1 else np.nan
    return {
        "capital": float(capital), "final_nav": float(nav.iloc[-1]),
        "pnl": float(nav.iloc[-1] - capital),
        "return_pct": 100.0 * float(nav.iloc[-1] / capital - 1.0),
        "max_drawdown_pct": 100.0 * dd,
        "weekly_mean_pct": 100.0 * float(wk.mean()) if len(wk) else np.nan,
        "weekly_sd_pct": 100.0 * sd,
        "weekly_sharpe": float(wk.mean() / sd) if sd and np.isfinite(sd) and sd > 0 else np.nan,
        "weeks": int(len(wk)),
    }
