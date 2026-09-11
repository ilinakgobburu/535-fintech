"""
The covered-call book: calendar, strike rule, blotter, ledger, Reg T.

One loop over history produces every number the site shows. The counterfactual
strike rules and the fill-hour sweep call this same function with different
arguments, so a comparison can never be a comparison of two implementations.

CONVENTIONS, stated once because the grade is logical consistency
-----------------------------------------------------------------
Bars are hourly and stamped in UTC. A bar labelled 15:00 covers 15:00-16:00,
so its BID/ASK is the quote standing at the END of that hour and its TRDPRC_1
is the last print inside it. We therefore treat the order as sent at the bar's
close and fill BOTH legs out of the SAME bar. That is the whole point: the
stock print and the option quote come from one moment, so the combo is priced
at one moment and no leg gets to see the other's future.

Options quote 13:00-20:00 UTC only (regular session). The 20:00 bar is a
one-trade closing stub, so the tradeable hours are 13:00-19:00.

THE RULES THIS IMPLEMENTS
-------------------------
entry      first session of the week, at the chosen hour. If flat, buy 100
           shares at the stock print. Cash decreases.
write      same bar, sell 1 call expiring at the END of that same week,
           nearest OTM (or ATM when spot sits exactly on a listed strike).
           Fill at mid = (BID+ASK)/2. Cash increases by 100 x mid.
no quote   no BID/ASK on the chosen strike at that bar -> no fill. If we are
           flat we do not buy the stock either, because the combo is one
           decision and half of it is unpriceable. If we already hold shares
           we hold them uncovered for the week and say so.
expiry     last session of the week, at the closing print.
             S_T >  K  assigned: deliver 100 shares, cash += 100 x K, flat.
             S_T <= K  expires worthless: keep shares, keep premium.
           No rolls. No buy-to-close. We wait through expiry.

Note the tie. At S_T exactly equal to K the call has no intrinsic value and is
not rationally exercised, so we treat it as expiring. It is a measure-zero
case on real prices but it has to be decided somewhere rather than falling out
of a floating-point comparison by accident.

REG T
-----
    LMV       = shares x stock mark
    NAV       = cash + stock MV + option MV      (short call MV is NEGATIVE)
    initial   = 0.50 x LMV                       covered short call adds $0
    maint     = 0.25 x LMV                       FINRA
    available = NAV - initial
    excess    = NAV - maint

Cash moves ONLY on blotter events: buy stock, collect premium, expire at zero,
assignment at strike. Marks move NAV; they never move cash.
"""

from __future__ import annotations

import datetime as dt
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .ric import parse_option_ric
from .vol import implied_vol

SHARES_PER_CONTRACT = 100
INITIAL_RATE = 0.50   # Reg T initial on a long equity position
MAINT_RATE = 0.25     # FINRA maintenance

SESSION_START_UTC = 13
SESSION_END_UTC = 20
# 20:00 is a one-trade stub bar; it marks fine but is not a place to send an order.
TRADEABLE_HOURS = tuple(range(SESSION_START_UTC, SESSION_END_UTC))

BUY, SELL, EXPIRE, ASSIGN = "BUY", "SELL", "EXPIRE", "ASSIGN"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_cache(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No cache at {path}. Run scripts/fetch_hw2.py with LSEG Workspace "
            f"running to build it."
        )
    with path.open("rb") as f:
        return pickle.load(f)


def stock_panel(payload: dict) -> pd.DataFrame:
    """Hourly underlying bars, regular session only, one row per timestamp."""
    df = payload["stock"].copy()
    df.index = pd.DatetimeIndex(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        # single-RIC pull: drop the RIC level
        df.columns = [c[-1] for c in df.columns]
    df.columns = [str(c).upper() for c in df.columns]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    hrs = df.index.hour
    df = df[(hrs >= SESSION_START_UTC) & (hrs <= SESSION_END_UTC)]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["date"] = [t.date() for t in df.index]
    return df


def option_panel(payload: dict) -> pd.DataFrame:
    """
    Long table of every call bar: timestamp, strike, expiry, quote, print.

    `mid` is NaN unless BOTH sides are present and the quote is not crossed.
    That is the assignment's "no bid/ask -> no fill" rule expressed once, in
    the data, rather than re-tested at every call site.
    """
    opt = payload["options"]
    if not isinstance(opt.columns, pd.MultiIndex):
        raise ValueError("expected a (RIC, Field) MultiIndex on the options frame")

    idx = pd.DatetimeIndex(opt.index)
    blocks = []
    for ric in opt.columns.get_level_values(0).unique():
        meta = parse_option_ric(ric)
        if meta is None or meta["cp"] != "C":
            continue
        sub = opt[ric]
        frame = pd.DataFrame(index=idx)
        for f in ("BID", "ASK", "TRDPRC_1", "OPEN_PRC", "HIGH_1", "LOW_1",
                  "ACVOL_UNS", "NUM_MOVES"):
            frame[f.lower()] = (pd.to_numeric(sub[f], errors="coerce")
                                if f in sub.columns else np.nan)
        frame = frame.dropna(how="all")
        if frame.empty:
            continue
        frame["ric"] = ric
        frame["strike"] = meta["strike"]
        frame["expiry"] = meta["expiry"]
        blocks.append(frame.reset_index().rename(columns={"index": "ts",
                                                          "Timestamp": "ts"}))
    if not blocks:
        return pd.DataFrame()

    panel = pd.concat(blocks, ignore_index=True)
    panel = panel.rename(columns={panel.columns[0]: "ts"}) if panel.columns[0] != "ts" else panel
    panel["ts"] = pd.DatetimeIndex(panel["ts"])
    hrs = panel["ts"].dt.hour
    panel = panel[(hrs >= SESSION_START_UTC) & (hrs <= SESSION_END_UTC)]

    ok = panel["bid"].notna() & panel["ask"].notna() & (panel["ask"] >= panel["bid"])
    panel["mid"] = np.where(ok, (panel["bid"] + panel["ask"]) / 2.0, np.nan)
    panel["spread"] = np.where(ok, panel["ask"] - panel["bid"], np.nan)
    panel["date"] = panel["ts"].dt.date
    panel["expiry"] = pd.to_datetime(panel["expiry"]).dt.date
    return panel.sort_values(["ts", "expiry", "strike"]).reset_index(drop=True)


def trading_weeks(stock: pd.DataFrame) -> list[dict]:
    """
    Weeks read off the underlying's own session calendar.

    "Buy Monday, expire Friday" is a description, not a rule. Jun 19 2026
    (Juneteenth) and Jul 3 2026 (Jul 4 observed) are closed, and those weeks
    expire on the Thursday -- the Thursday RIC resolves against LSEG and the
    Friday one does not. Taking the first and last session the stock actually
    printed makes a holiday shift the cycle instead of deleting it.
    """
    days = sorted({d for d in stock["date"]})
    out = []
    for key, grp in pd.Series(days).groupby(
            pd.Series(days).map(lambda d: (d.isocalendar().year, d.isocalendar().week))):
        sess = list(grp)
        if len(sess) < 2:
            continue
        out.append({
            "iso": f"{key[0]}-W{key[1]:02d}",
            "entry_date": sess[0],
            "expiry_date": sess[-1],
            "sessions": len(sess),
            "short_week": len(sess) < 5,
        })
    return out


# --------------------------------------------------------------------------
# strike rules
# --------------------------------------------------------------------------

def nearest_otm(chain: pd.DataFrame, spot: float, **_) -> float | None:
    """
    The assignment's baseline: lowest listed strike at or above spot.

    'At or above' rather than 'strictly above' is what makes the ATM clause
    work -- when spot sits exactly on a strike, that strike IS the answer, and
    a strictly-greater test would silently skip to the next one.
    """
    ks = np.sort(chain["strike"].unique())
    hit = ks[ks >= spot - 1e-9]
    return float(hit[0]) if len(hit) else None


def otm_by_offset(chain: pd.DataFrame, spot: float, *, offset_pct: float = 0.0, **_):
    """Nearest listed strike at or above spot x (1 + offset_pct)."""
    ks = np.sort(chain["strike"].unique())
    hit = ks[ks >= spot * (1.0 + offset_pct) - 1e-9]
    return float(hit[0]) if len(hit) else None


def by_premium_floor(chain: pd.DataFrame, spot: float, *, floor: float = 0.50, **_):
    """
    Furthest-OTM strike still paying at least `floor` dollars of mid.

    A cap you are not paid for is not a trade. This rule refuses to sell the
    upside away for pennies, which is the failure mode a fixed nearest-OTM
    rule walks into on a quiet week.
    """
    q = chain[chain["mid"].notna() & (chain["strike"] >= spot - 1e-9)]
    q = q[q["mid"] >= floor].sort_values("strike")
    return float(q["strike"].iloc[-1]) if len(q) else None


# --------------------------------------------------------------------------
# the volatility-aware rule
# --------------------------------------------------------------------------

YEAR_SECONDS = 365.25 * 24 * 3600


def years_to_expiry(ts, expiry: dt.date) -> float:
    """Actual/365.25 from the order bar to the expiry session's close."""
    close = pd.Timestamp(dt.datetime.combine(expiry, dt.time(SESSION_END_UTC)))
    return max((close - pd.Timestamp(ts)).total_seconds(), 0.0) / YEAR_SECONDS


def atm_iv(chain: pd.DataFrame, spot: float, T: float, n: int = 3) -> float:
    """
    Implied vol from the `n` listed strikes closest to spot, median-aggregated.

    F = spot and D = 1. That is not laziness: HW1 fitted put-call parity
    across this same RIC space and found the implied forward lands at spot
    +$0.020, while the discount factor is NOT identified at these maturities
    at all -- raw fits implied annualised rates from -2567% to +394% for
    dte <= 15. At a one-week horizon on a non-dividend payer, F = S and D = 1
    are the honest values, and a fitted D would be fitting noise. This pull is
    calls-only in any case, so parity is unavailable here.

    The median over three strikes rather than a single ATM inversion is for
    robustness: one stale quote at the money should not set the week's strike.
    """
    q = chain[chain["mid"].notna() & (chain["mid"] > 0)]
    if q.empty or T <= 0:
        return float("nan")
    q = q.assign(dist=(q["strike"] - spot).abs()).nsmallest(n, "dist")
    vols = [implied_vol(float(r.mid), float(spot), float(r.strike), T, 1.0, "C")
            for r in q.itertuples()]
    vols = [v for v in vols if v is not None and np.isfinite(v) and v > 0]
    return float(np.median(vols)) if vols else float("nan")


def by_assignment_prob(chain: pd.DataFrame, spot: float, *,
                       target: float = 0.25, T: float | None = None, **_):
    """
    Sell the cap the market's own volatility prices at a `target` chance of
    being breached.

    Under Black-76 with F = S the terminal price is S*exp(-s^2 T/2 + s sqrt(T) Z),
    so P(S_T > K) = 1 - N(d) with d = [ln(K/S) + s^2 T/2] / (s sqrt(T)). Setting
    that equal to `target` and solving:

        K* = S * exp( s sqrt(T) * N^-1(1 - target)  -  s^2 T / 2 )

    then take the lowest listed strike at or above K*.

    READ THE PROBABILITY CORRECTLY. This is the RISK-NEUTRAL probability, not
    a forecast. It is what the option market's prices imply, which is exactly
    what makes it an algorithmic rule -- it needs no view - but a 25% target
    does not predict 25% assignments, and the gap between the two is itself
    worth reporting rather than hiding.

    The point of the rule is that it ADAPTS. A fixed 2%-out rule sells the same
    cap in a calm week and a violent one; this one widens when the week is
    priced to move.
    """
    from scipy.special import ndtri          # inverse standard normal CDF

    if T is None or not np.isfinite(T) or T <= 0:
        return None
    sigma = atm_iv(chain, spot, T)
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    k_star = spot * np.exp(sigma * np.sqrt(T) * ndtri(1.0 - target)
                           - 0.5 * sigma * sigma * T)
    ks = np.sort(chain["strike"].unique())
    hit = ks[ks >= k_star - 1e-9]
    return float(hit[0]) if len(hit) else None


STRIKE_RULES = {
    "nearest_otm": nearest_otm,
    "otm_1pct": lambda c, s, **k: otm_by_offset(c, s, offset_pct=0.01),
    "otm_2pct": lambda c, s, **k: otm_by_offset(c, s, offset_pct=0.02),
    "premium_50c": lambda c, s, **k: by_premium_floor(c, s, floor=0.50),
    "iv_prob_25": lambda c, s, **k: by_assignment_prob(c, s, target=0.25, **k),
    "iv_prob_15": lambda c, s, **k: by_assignment_prob(c, s, target=0.15, **k),
}

# Labels and, where the rule states one, the risk-neutral breach probability it
# is aiming at. Kept beside the rules rather than in the page's JavaScript so a
# new rule cannot appear on the site under its bare function key.
STRIKE_RULE_META = {
    "nearest_otm": {"label": "nearest OTM", "target_prob": None},
    "otm_1pct": {"label": "first strike \u2265 spot \u00d7 1.01", "target_prob": None},
    "otm_2pct": {"label": "first strike \u2265 spot \u00d7 1.02", "target_prob": None},
    "premium_50c": {"label": "furthest strike still paying \u2265 $0.50",
                    "target_prob": None},
    "iv_prob_25": {"label": "implied vol, 25% breach probability", "target_prob": 0.25},
    "iv_prob_15": {"label": "implied vol, 15% breach probability", "target_prob": 0.15},
}


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

def _bar_at(frame: pd.DataFrame, day: dt.date, hour: int) -> pd.Timestamp | None:
    """The bar stamped `hour` on `day`, or the last bar before it that exists."""
    same_day = frame.index[[t.date() == day for t in frame.index]]
    if len(same_day) == 0:
        return None
    upto = same_day[same_day.hour <= hour]
    return upto[-1] if len(upto) else same_day[0]


def _last_bar(frame: pd.DataFrame, day: dt.date) -> pd.Timestamp | None:
    same_day = frame.index[[t.date() == day for t in frame.index]]
    return same_day[-1] if len(same_day) else None


def run_backtest(
    stock: pd.DataFrame,
    options: pd.DataFrame,
    weeks: list[dict],
    *,
    rule: str = "nearest_otm",
    order_hour: int = 15,
    start_cash: float = 50_000.0,
    contracts: int = 1,
) -> dict:
    """
    Walk the weeks and book what the rules say. Returns the blotter and the
    per-week decision record; the ledger is built from the blotter afterwards
    so it cannot disagree with it.
    """
    pick = STRIKE_RULES[rule]
    blotter: list[dict] = []
    cycles: list[dict] = []
    shares = 0
    qty = contracts * SHARES_PER_CONTRACT

    for w in weeks:
        entry_day, expiry_day = w["entry_date"], w["expiry_date"]
        ts = _bar_at(stock, entry_day, order_hour)
        if ts is None:
            continue

        spot = stock.at[ts, "TRDPRC_1"] if "TRDPRC_1" in stock.columns else np.nan
        if not np.isfinite(spot):
            cycles.append({**w, "status": "no stock print", "order_ts": str(ts)})
            continue

        chain = options[(options["ts"] == ts) & (options["expiry"] == expiry_day)]
        T = years_to_expiry(ts, expiry_day)
        strike = pick(chain, float(spot), T=T) if len(chain) else None
        row = chain[np.isclose(chain["strike"], strike)] if strike is not None else chain.iloc[:0]
        mid = float(row["mid"].iloc[0]) if len(row) and np.isfinite(row["mid"].iloc[0]) else np.nan

        if strike is None or not np.isfinite(mid):
            # The combo is one decision and half of it is unpriceable.
            cycles.append({
                **w, "status": "skipped: no two-sided quote",
                "order_ts": str(ts), "spot": float(spot),
                "strike": None if strike is None else float(strike),
                "held_uncovered": shares > 0,
            })
            continue

        r = row.iloc[0]
        if shares == 0:
            blotter.append({
                "ts": ts, "instrument": stock.attrs.get("ric", "AAPL.O"),
                "kind": "stock", "side": BUY, "qty": qty,
                "limit": float(spot), "fill": float(spot),
                "cash_delta": -qty * float(spot),
                "note": "entry: flat at the order bar, buy 100 shares at the stock print",
            })
            shares += qty

        blotter.append({
            "ts": ts, "instrument": r["ric"], "kind": "call", "side": SELL,
            "qty": -contracts, "limit": round(float(mid), 4), "fill": round(float(mid), 4),
            "cash_delta": qty * float(mid),
            "strike": float(strike), "expiry": expiry_day,
            "bid": float(r["bid"]), "ask": float(r["ask"]),
            "note": f"write: {rule}, nearest listed strike >= spot "
                    f"({spot:.2f}); fill at mid of {r['bid']:.2f}/{r['ask']:.2f}",
        })

        # ---- wait through expiry -------------------------------------------
        ets = _last_bar(stock, expiry_day)
        s_t = float(stock.at[ets, "TRDPRC_1"]) if ets is not None else np.nan
        if not np.isfinite(s_t):
            cycles.append({**w, "status": "no expiry print", "strike": float(strike)})
            continue

        itm = s_t > strike  # ties expire: an ATM call has no intrinsic value
        if itm:
            blotter.append({
                "ts": ets, "instrument": r["ric"], "kind": "call", "side": ASSIGN,
                "qty": contracts, "limit": None, "fill": float(strike),
                "cash_delta": qty * float(strike),
                "strike": float(strike), "expiry": expiry_day,
                "note": f"assigned: settle {s_t:.2f} > strike {strike:.2f}; "
                        f"deliver {qty} shares at {strike:.2f}, flat",
            })
            shares -= qty
        else:
            blotter.append({
                "ts": ets, "instrument": r["ric"], "kind": "call", "side": EXPIRE,
                "qty": contracts, "limit": None, "fill": 0.0, "cash_delta": 0.0,
                "strike": float(strike), "expiry": expiry_day,
                "note": f"expired: settle {s_t:.2f} <= strike {strike:.2f}; "
                        f"keep shares, keep premium",
            })

        cycles.append({
            **w, "status": "assigned" if itm else "expired",
            "order_ts": str(ts), "spot": float(spot), "strike": float(strike),
            "mid": float(mid), "bid": float(r["bid"]), "ask": float(r["ask"]),
            "settle": s_t, "premium": qty * float(mid),
            "otm_pct": 100.0 * (strike / spot - 1.0),
            "T_years": T,
            "atm_iv": atm_iv(chain, float(spot), T),
        })

    return {"blotter": blotter, "cycles": cycles,
            "rule": rule, "order_hour": order_hour, "start_cash": start_cash}


# --------------------------------------------------------------------------
# ledger + Reg T
# --------------------------------------------------------------------------

def build_ledger(run: dict, stock: pd.DataFrame, options: pd.DataFrame) -> pd.DataFrame:
    """
    Replay the blotter across every hourly bar and mark the book.

    The ledger is DERIVED from the blotter rather than accumulated alongside
    it, so the two cannot drift. Cash only ever moves on a blotter event;
    marks move NAV and never touch cash.
    """
    blotter = sorted(run["blotter"], key=lambda b: (b["ts"], b["side"] != BUY))
    cash = float(run["start_cash"])
    shares = 0
    call = None            # dict(ric, strike, expiry) while short
    bi = 0

    # Fast lookup of a short call's mid at a given bar.
    mid_by = {(r.ric, r.ts): r.mid for r in options.itertuples()}

    rows = []
    last_stock = np.nan
    last_opt = np.nan
    for ts in stock.index:
        while bi < len(blotter) and blotter[bi]["ts"] <= ts:
            ev = blotter[bi]
            cash += ev["cash_delta"]
            if ev["kind"] == "stock":
                shares += ev["qty"] if ev["side"] == BUY else -ev["qty"]
            elif ev["side"] == SELL:
                call = {"ric": ev["instrument"], "strike": ev["strike"],
                        "expiry": ev["expiry"]}
                last_opt = ev["fill"]
            elif ev["side"] in (EXPIRE, ASSIGN):
                if ev["side"] == ASSIGN:
                    shares -= SHARES_PER_CONTRACT
                call = None
                last_opt = np.nan
            bi += 1

        px = stock.at[ts, "TRDPRC_1"]
        if np.isfinite(px):
            last_stock = float(px)
        stock_mark = last_stock

        if call is None:
            opt_mark = 0.0
        else:
            m = mid_by.get((call["ric"], ts), np.nan)
            if m is not None and np.isfinite(m):
                last_opt = float(m)
            elif not np.isfinite(last_opt):
                # never quoted: fall back to intrinsic, which is a floor, and
                # flag it rather than pretending we had a mark
                last_opt = max(0.0, stock_mark - call["strike"])
            opt_mark = last_opt

        stock_mv = shares * stock_mark
        option_mv = -SHARES_PER_CONTRACT * opt_mark if call is not None else 0.0
        nav = cash + stock_mv + option_mv
        lmv = shares * stock_mark
        im = INITIAL_RATE * lmv
        mm = MAINT_RATE * lmv

        rows.append({
            "ts": ts, "cash": cash, "shares": shares,
            "stock_mark": stock_mark, "stock_mv": stock_mv,
            "call_ric": call["ric"] if call else None,
            "call_strike": call["strike"] if call else np.nan,
            "call_expiry": str(call["expiry"]) if call else None,
            "option_mark": opt_mark if call else np.nan,
            "option_mv": option_mv,
            "nav": nav, "lmv": lmv,
            "initial_margin": im, "maintenance_margin": mm,
            "available_funds": nav - im, "excess_liquidity": nav - mm,
        })

    led = pd.DataFrame(rows)
    led["feasible"] = led["available_funds"] >= 0
    return led


def min_start_cash(run: dict, stock: pd.DataFrame, options: pd.DataFrame,
                   lo: float = 0.0, hi: float = 200_000.0,
                   tol: float = 25.0) -> dict:
    """
    Smallest starting cash for which available funds never go negative.

    Available funds are affine in starting cash -- an extra dollar of cash is
    an extra dollar of NAV and does not move LMV -- so the worst bar is the
    same bar at every level and this is really just one subtraction. We solve
    it by bisection anyway, because the affine argument is a property of the
    accounting we assert on the site and a search that agrees with it is the
    cheapest possible check that the ledger really behaves that way.
    """
    def worst(c: float) -> float:
        led = build_ledger({**run, "start_cash": c}, stock, options)
        return float(led["available_funds"].min())

    if worst(lo) >= 0:
        return {"min_cash": lo, "worst_available": worst(lo), "binds": False}
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if worst(mid) >= 0:
            hi = mid
        else:
            lo = mid
    led = build_ledger({**run, "start_cash": hi}, stock, options)
    i = int(led["available_funds"].idxmin())
    return {
        "min_cash": hi,
        "worst_available": float(led["available_funds"].min()),
        "worst_ts": str(led.at[i, "ts"]),
        "binds": True,
    }
