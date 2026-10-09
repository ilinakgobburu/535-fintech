"""
The registered PMCC backtest: daily bars, a January LEAP, a monthly short call.

Every rule here is a line of config/thresholds.yaml, which was committed
before this module produced any output. Three books run through ONE loop, so
they cannot differ by implementation:

    long_leg="leap"                      the poor man's covered call
    long_leg="stock"                     the covered call, same short-leg rule
    long_leg="stock", write_calls=False  buy and hold

WHAT IS KNOWN WHEN. A decision on day d uses day d's closing quotes and fills
at them. Expiry settles on the expiry day's close. An assignment is covered at
the NEXT session's open. Nothing reads a later row than the one it trades on.

REG T (leap book). A long option has no loan value and is paid in full; the
short call is covered by the LEAP while the LEAP's strike is at or below it.

    NAV       = cash + LEAP MV + short call MV + stock MV
    options   = LEAP MV + short call MV
    initial   = options + 0.50 x |stock MV|      stock MV is short stock, held
    maint     = options + 0.30 x |stock MV|      only from assignment to cover
    available = NAV - initial

For the stock books it is the covered-call form: initial 50% and maintenance
25% of the stock's market value, and a covered short call adds nothing.
"""

from __future__ import annotations

import datetime as dt
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .pmcc import call_greeks
from .ric import parse_option_ric

CONTRACT = 100
LONG, SHORT = "long", "short"
BUY, SELL, EXPIRE, ASSIGN, DIVIDEND = "BUY", "SELL", "EXPIRE", "ASSIGN", "DIVIDEND"

TARGET_LONG_DELTA = 0.80
TARGET_SHORT_DELTA = 0.25
LEAP_TARGET_DAYS = 365
ROLL_DTE = 90
ROLL_DELTA = 0.50           # version 2: replace a LEAP whose delta has fallen below this
COVER_SPREAD = 0.01          # dollars per share added to the open on a buy-back
INITIAL_RATE, MAINT_RATE, SHORT_MAINT_RATE = 0.50, 0.25, 0.30


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_market(cache: str | Path) -> dict:
    """
    The raw pull as the engine reads it.

    `quotes` has one row per contract per day with a `mid` that is NaN unless
    both sides are present and uncrossed. LSEG returns a zero bid as missing,
    so a call nobody bids for has no mid and cannot be selected or sold.
    """
    cache = Path(cache)
    frames = pickle.loads((cache / "options.pkl").read_bytes())
    # Contracts that were open across a spin-off are kept by LSEG under the
    # root with a digit appended (NVS1...), for their whole life. They are the
    # same contracts until the spin-off date, so they are read as the plain root.
    adjusted = cache / "options_adjusted.pkl"
    if adjusted.exists():
        frames.update(pickle.loads(adjusted.read_bytes()))
    blocks = []
    for ric, df in frames.items():
        meta = parse_option_ric(re.sub(r"^([A-Z]+)\d(?=[A-X]\d{9}\.)", r"\1", ric))
        if meta is None or meta["cp"] != "C":
            continue
        q = pd.DataFrame({"bid": pd.to_numeric(df["BID"], errors="coerce"),
                          "ask": pd.to_numeric(df["ASK"], errors="coerce")})
        q.index = pd.DatetimeIndex(df.index).normalize()
        q = q[q.notna().any(axis=1)]
        q["ric"], q["strike"], q["expiry"] = ric, meta["strike"], pd.Timestamp(meta["expiry"])
        blocks.append(q.rename_axis("date").reset_index())
    quotes = pd.concat(blocks, ignore_index=True)
    ok = quotes["bid"].notna() & quotes["ask"].notna() & (quotes["ask"] >= quotes["bid"])
    quotes["mid"] = np.where(ok, (quotes["bid"] + quotes["ask"]) / 2.0, np.nan)

    raw = pd.read_pickle(cache / "stock_unadjusted_ohlc.pkl")
    stock = pd.DataFrame({"close": pd.to_numeric(raw["TRDPRC_1"], errors="coerce"),
                          "open": pd.to_numeric(raw["OPEN_PRC"], errors="coerce")})
    stock.index = pd.DatetimeIndex(raw.index).normalize()
    stock = stock.dropna(subset=["close"]).sort_index()

    aux = pickle.loads((cache / "aux.pkl").read_bytes())
    tb = pd.to_numeric(aux["tbill"]["MID_YLD_1"], errors="coerce") / 100.0
    tb.index = pd.DatetimeIndex(tb.index).normalize()
    div = aux["dividends"].iloc[:, 1:3].copy()
    div.columns = ["ex_date", "amount"]
    div["ex_date"] = pd.to_datetime(div["ex_date"])
    div = div.dropna(subset=["ex_date"])
    div = div[pd.to_numeric(div["amount"], errors="coerce") > 0]
    earnings = pd.DatetimeIndex(pd.to_datetime(aux["earnings"].iloc[:, 1]).dropna()).normalize()
    return {
        "quotes": quotes, "stock": stock,
        "tbill": tb.reindex(stock.index.union(tb.index)).ffill().reindex(stock.index),
        "dividends": pd.Series(div["amount"].astype(float).to_numpy(),
                               index=pd.DatetimeIndex(div["ex_date"]).normalize()).sort_index(),
        "earnings": earnings.sort_values(),
    }


# --------------------------------------------------------------------------
# calendar
# --------------------------------------------------------------------------

def third_friday(year: int, month: int) -> pd.Timestamp:
    d = dt.date(year, month, 15)
    return pd.Timestamp(d + dt.timedelta(days=(4 - d.weekday()) % 7))


def monthly_expiries(sessions: pd.DatetimeIndex, start, end) -> list[pd.Timestamp]:
    """
    Each month's expiry session inside [start, end]: the third Friday, or the
    session before it when the market is closed that day (Good Friday).
    """
    out = []
    for p in pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M"):
        day = third_friday(p.year, p.month)
        upto = sessions[sessions <= day]
        if len(upto) and pd.Timestamp(start) <= upto[-1] <= pd.Timestamp(end):
            out.append(upto[-1])
    return out


def carry(market: dict, day: pd.Timestamp) -> tuple[float, float]:
    """(rate, dividend yield) known at `day`: the bill yield and the trailing year's dividends."""
    r = float(market["tbill"].get(day, np.nan))
    d = market["dividends"]
    paid = float(d[(d.index > day - pd.Timedelta(days=365)) & (d.index <= day)].sum())
    return (r if np.isfinite(r) else 0.0), paid / float(market["stock"].at[day, "close"])


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def with_delta(chain: pd.DataFrame, spot: float, day, r: float, q: float) -> pd.DataFrame:
    """Two-sided quotes of one day, with implied vol and delta from each one's own mid."""
    live = chain[chain["mid"].notna() & (chain["mid"] > 0)].copy()
    if live.empty:
        return live.assign(iv=[], delta=[])
    g = [call_greeks(float(m), spot, float(k), (e - day).days / 365.0, r, q)
         for m, k, e in zip(live["mid"], live["strike"], live["expiry"])]
    live["iv"] = [x["iv"] for x in g]
    live["delta"] = [x["delta"] for x in g]
    return live[live["iv"].notna()]


def select_long(day_quotes: pd.DataFrame, spot: float, day, r: float, q: float,
                target: float = TARGET_LONG_DELTA) -> pd.Series | None:
    """The January expiry nearest 12 months out, then the delta closest to `target`."""
    jan = day_quotes[(day_quotes["expiry"].dt.month == 1) & day_quotes["mid"].notna()]
    if jan.empty:
        return None
    expiries = sorted(jan["expiry"].unique(),
                      key=lambda e: (abs((e - day).days - LEAP_TARGET_DAYS), e))
    c = with_delta(jan[jan["expiry"] == expiries[0]], spot, day, r, q)
    if c.empty:
        return None
    c = c.assign(miss=(c["delta"] - target).abs()).sort_values(["miss", "strike"])
    return c.iloc[0]


def select_short(day_quotes: pd.DataFrame, expiry, spot: float, day, r: float, q: float,
                 floor: float | None, target: float = TARGET_SHORT_DELTA) -> pd.Series | None:
    """Delta closest to `target` among strikes at or above `floor`; a tie goes further out."""
    c = day_quotes[day_quotes["expiry"] == expiry]
    if floor is not None:
        c = c[c["strike"] >= floor - 1e-9]
    c = with_delta(c, spot, day, r, q)
    if c.empty:
        return None
    c = c.assign(miss=(c["delta"] - target).abs()).sort_values(["miss", "strike"],
                                                               ascending=[True, False])
    return c.iloc[0]


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

def run(market: dict, *, start, end, long_leg: str = "leap", write_calls: bool = True,
        skip_earnings: bool = False, fill: str = "cross", ticker: str = "DAL",
        roll_delta: float | None = None, breakeven_floor: bool = False) -> dict:
    """
    Walk the monthly cycles and book what the rules say.

    `fill` is "cross" (sell at the bid, buy at the ask) or "mid". Returns the
    blotter, one record per cycle, the assignments and the dividend-risk log.

    `roll_delta` and `breakeven_floor` are the version 2 rules. With
    `roll_delta` set, the LEAP is also replaced at an entry date when its delta,
    from that day's mid, is below it. With `breakeven_floor`, the short strike
    must be at or above the LEAP's strike plus the price paid for it, where
    version 1 asked only for the LEAP's strike.
    """
    stock, quotes = market["stock"], market["quotes"]
    sessions = stock.index[(stock.index >= pd.Timestamp(start)) & (stock.index <= pd.Timestamp(end))]
    expiries = monthly_expiries(sessions, start, end)
    by_day = {d: g for d, g in quotes[quotes["date"].isin(sessions)].groupby("date")}
    empty = quotes.iloc[:0]
    px = (lambda row, side: float(row["ask"] if side == BUY else row["bid"])) if fill == "cross" \
        else (lambda row, side: float(row["mid"]))

    blotter, cycles, assignments = [], [], []
    shares, long, pending_cover = 0, None, None

    def book(day, when, kind, side, qty, price, cash, note, leg=None, row=None, instrument=None):
        ev = {"date": day, "when": when, "kind": kind, "leg": leg, "side": side,
              "qty": qty, "price": price, "cash_delta": cash, "note": note,
              "instrument": instrument or ticker}
        if row is not None:
            ev.update(instrument=row["ric"], strike=float(row["strike"]),
                      expiry=row["expiry"], bid=float(row["bid"]) if pd.notna(row["bid"]) else None,
                      ask=float(row["ask"]) if pd.notna(row["ask"]) else None)
        blotter.append(ev)

    def buy_leap(day, spot, r, q, why):
        pick = select_long(by_day.get(day, empty), spot, day, r, q)
        if pick is None:
            return None
        p = px(pick, BUY)
        book(day, "close", "call", BUY, 1, p, -CONTRACT * p,
             f"{why}: January expiry nearest 12 months out, delta closest to "
             f"{TARGET_LONG_DELTA:.2f} (delta {pick['delta']:.2f}, implied vol "
             f"{100 * pick['iv']:.1f}%); quote {pick['bid']:.2f}/{pick['ask']:.2f}",
             leg=LONG, row=pick)
        return {"ric": pick["ric"], "strike": float(pick["strike"]), "expiry": pick["expiry"],
                "cost": p, "delta": float(pick["delta"]), "iv": float(pick["iv"])}

    for prev, expiry in zip(expiries[:-1], expiries[1:]):
        day = sessions[sessions > prev][0]
        spot = float(stock.at[day, "close"])
        r, q = carry(market, day)
        rec = {"entry": day, "expiry": expiry, "spot": spot, "dte": (expiry - day).days,
               "earnings_in_cycle": bool(((market["earnings"] > day)
                                          & (market["earnings"] <= expiry)).any())}

        # ---- the open: settle last cycle's assignment --------------------
        if pending_cover is not None:
            o = float(stock.at[day, "open"]) + COVER_SPREAD
            book(day, "open", "stock", BUY, CONTRACT, o, -CONTRACT * o,
                 ("cover: buy back the 100 shares sold short by assignment"
                  if long_leg == "leap" else "re-entry: buy 100 shares after assignment")
                 + f", at the open {o - COVER_SPREAD:.2f} plus {COVER_SPREAD:.2f}")
            shares += CONTRACT
            pending_cover.update(cover_date=day, cover_price=o,
                                 gap=o - COVER_SPREAD - pending_cover["settle"],
                                 cover_cost=CONTRACT * (o - pending_cover["strike"]))
            pending_cover = None

        # ---- long leg ----------------------------------------------------
        if long_leg == "stock":
            if shares == 0:
                book(day, "close", "stock", BUY, CONTRACT, spot, -CONTRACT * spot,
                     "entry: buy 100 shares at the close")
                shares = CONTRACT
        elif long is None:
            long = buy_leap(day, spot, r, q, "long leg")
            if long is None:
                cycles.append({**rec, "status": "skipped: no LEAP quote"})
                continue
        else:
            held = by_day.get(day, empty)
            held = held[(held["ric"] == long["ric"]) & held["mid"].notna()]
            why = None
            if (long["expiry"] - day).days < ROLL_DTE:
                why = f"under {ROLL_DTE} days left on the LEAP"
            elif roll_delta is not None and len(held):
                now = call_greeks(float(held.iloc[0]["mid"]), spot, long["strike"],
                                  (long["expiry"] - day).days / 365.0, r, q)["delta"]
                if now < roll_delta:
                    why = f"LEAP delta {now:.2f} is below {roll_delta:.2f}"
            if why and len(held):
                p = px(held.iloc[0], SELL)
                book(day, "close", "call", SELL, 1, p, CONTRACT * p,
                     f"roll: {why}; sold, quote "
                     f"{held.iloc[0]['bid']:.2f}/{held.iloc[0]['ask']:.2f}",
                     leg=LONG, row=held.iloc[0])
                long = buy_leap(day, spot, r, q, "roll")
                rec["rolled"] = True
                if long is None:
                    cycles.append({**rec, "status": "closed: no LEAP to roll into"})
                    continue
        if long is not None:
            rec.update(long_strike=long["strike"], long_expiry=long["expiry"])

        # ---- short leg ---------------------------------------------------
        if not write_calls:
            cycles.append({**rec, "status": "no call written"})
            continue
        if skip_earnings and rec["earnings_in_cycle"]:
            cycles.append({**rec, "status": "skipped: earnings in cycle"})
            continue
        floor = None
        if long is not None:
            floor = long["strike"] + (long["cost"] if breakeven_floor else 0.0)
            rec["floor"] = floor
        pick = select_short(by_day.get(day, empty), expiry, spot, day, r, q, floor)
        if pick is None:
            cycles.append({**rec, "status": "skipped: no qualifying strike"})
            continue
        p = px(pick, SELL)
        strike = float(pick["strike"])
        book(day, "close", "call", SELL, 1, p, CONTRACT * p,
             f"write: next monthly expiry, delta closest to {TARGET_SHORT_DELTA:.2f} "
             f"(delta {pick['delta']:.2f}, implied vol {100 * pick['iv']:.1f}%), spot "
             f"{spot:.2f}; quote {pick['bid']:.2f}/{pick['ask']:.2f}", leg=SHORT, row=pick)
        rec.update(strike=strike, premium=CONTRACT * p, fill=p, bid=float(pick["bid"]),
                   ask=float(pick["ask"]), short_delta=float(pick["delta"]),
                   short_iv=float(pick["iv"]), otm_pct=100.0 * (strike / spot - 1.0))

        # ---- expiry ------------------------------------------------------
        settle = float(stock.at[expiry, "close"])
        rec["settle"] = settle
        if settle > strike:
            book(expiry, "close", "call", ASSIGN, 1, strike, 0.0,
                 f"assigned: close {settle:.2f} > strike {strike:.2f}", leg=SHORT, row=pick)
            book(expiry, "close", "stock", SELL, CONTRACT, strike, CONTRACT * strike,
                 "delivered against assignment at the strike"
                 + ("; no shares were held, so the book is short 100 until the next open"
                    if long_leg == "leap" else "; flat until the next open"))
            shares -= CONTRACT
            pending_cover = {"entry": day, "expiry": expiry, "strike": strike,
                             "settle": settle, "premium": CONTRACT * p}
            assignments.append(pending_cover)
            cycles.append({**rec, "status": "assigned"})
        else:
            book(expiry, "close", "call", EXPIRE, 1, 0.0, 0.0,
                 f"expired: close {settle:.2f} <= strike {strike:.2f}", leg=SHORT, row=pick)
            cycles.append({**rec, "status": "expired"})

    # An assignment on the last expiry is still covered at the next open.
    if pending_cover is not None:
        nxt = stock.index[stock.index > expiries[-1]]
        if len(nxt) and nxt[0] <= pd.Timestamp(end):
            day = nxt[0]
            o = float(stock.at[day, "open"]) + COVER_SPREAD
            book(day, "open", "stock", BUY, CONTRACT, o, -CONTRACT * o,
                 f"cover after the final assignment, at the open plus {COVER_SPREAD:.2f}")
            shares += CONTRACT
            pending_cover.update(cover_date=day, cover_price=o,
                                 gap=o - COVER_SPREAD - pending_cover["settle"],
                                 cover_cost=CONTRACT * (o - pending_cover["strike"]))

    # Dividends, and the early-exercise watch the rules ask for.
    div_risk = []
    held_short = [(c["entry"], c["expiry"], c["strike"]) for c in cycles if "strike" in c]
    for ex, amount in market["dividends"].items():
        before = sessions[sessions < ex]
        if ex not in sessions or not len(before):
            continue
        eve = before[-1]
        for entry, exp, k in held_short:
            if entry <= eve < exp and float(stock.at[eve, "close"]) > k:
                div_risk.append({"date": eve, "ex_date": ex, "dividend": float(amount),
                                 "close": float(stock.at[eve, "close"]), "strike": k})
    return {"blotter": blotter, "cycles": cycles, "assignments": assignments,
            "dividend_risk": div_risk, "long_leg": long_leg, "write_calls": write_calls,
            "skip_earnings": skip_earnings, "fill": fill, "roll_delta": roll_delta,
            "breakeven_floor": breakeven_floor, "start": pd.Timestamp(start),
            "end": pd.Timestamp(end), "ticker": ticker}


def opening_capital(result: dict) -> float:
    """Cash the first entry day consumed: the long leg less the first premium."""
    first = min(ev["date"] for ev in result["blotter"])
    return -sum(ev["cash_delta"] for ev in result["blotter"] if ev["date"] == first)


# --------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------

def build_ledger(result: dict, market: dict, start_cash: float,
                 greeks: bool = True) -> pd.DataFrame:
    """
    One row per session's close, replayed from the blotter.

    Options are marked at the day's mid, or the last mid when the day has no
    two-sided quote. Dividends are credited on the ex-date to shares held at
    the prior close; they are a cash event the blotter does not carry, so they
    are added here and returned in the `dividend` column.
    """
    stock = market["stock"]
    events = sorted(result["blotter"], key=lambda e: e["date"])   # stable: booked order
    if not events:
        return pd.DataFrame()
    days = stock.index[(stock.index >= events[0]["date"]) & (stock.index <= result["end"])]
    held = {ev["instrument"] for ev in events if ev["kind"] == "call"}
    q = market["quotes"]
    mids = {(a, b): c for a, b, c in
            q.loc[q["ric"].isin(held) & q["mid"].notna(), ["ric", "date", "mid"]].itertuples(index=False)}
    leap_book = result["long_leg"] == "leap"

    cash, shares, i = float(start_cash), 0, 0
    legs = {LONG: None, SHORT: None}
    last = {LONG: np.nan, SHORT: np.nan}
    last_g = {LONG: None, SHORT: None}
    zero = {"iv": np.nan, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    rows, prev_shares = [], 0
    for day in days:
        div = 0.0
        if day in market["dividends"].index and prev_shares > 0:
            div = float(market["dividends"].loc[[day]].sum()) * prev_shares
            cash += div
        while i < len(events) and events[i]["date"] <= day:
            ev = events[i]
            cash += ev["cash_delta"]
            if ev["kind"] == "stock":
                shares += ev["qty"] if ev["side"] == BUY else -ev["qty"]
            else:
                # BUY opens the long leg and SELL opens the short one; every
                # other row on a leg (SELL, EXPIRE, ASSIGN) closes it.
                opening = ev["side"] == (BUY if ev["leg"] == LONG else SELL)
                legs[ev["leg"]] = ({"ric": ev["instrument"], "strike": ev["strike"],
                                    "expiry": ev["expiry"]} if opening else None)
                last[ev["leg"]] = ev["price"] if opening else np.nan
                last_g[ev["leg"]] = None
            i += 1

        spot = float(stock.at[day, "close"])
        r, qy = carry(market, day) if greeks else (0.0, 0.0)
        mv, g = {}, {}
        for leg, sign in ((LONG, 1), (SHORT, -1)):
            pos = legs[leg]
            if pos is None:
                mv[leg], g[leg] = 0.0, zero
                continue
            m = mids.get((pos["ric"], day))
            if m is not None:
                last[leg] = float(m)
            mv[leg] = sign * CONTRACT * last[leg]
            if greeks and (m is not None or last_g[leg] is None):
                last_g[leg] = call_greeks(last[leg], spot, pos["strike"],
                                          (pos["expiry"] - day).days / 365.0, r, qy)
            g[leg] = last_g[leg] or zero

        stock_mv = shares * spot
        nav = cash + stock_mv + mv[LONG] + mv[SHORT]
        if leap_book:
            option_req = mv[LONG] + mv[SHORT]
            im = option_req + INITIAL_RATE * abs(stock_mv)
            mm = option_req + SHORT_MAINT_RATE * abs(stock_mv)
        else:
            im, mm = INITIAL_RATE * stock_mv, MAINT_RATE * stock_mv
        row = {"date": day, "cash": cash, "shares": shares, "stock_close": spot,
               "stock_mv": stock_mv, "long_mv": mv[LONG], "short_mv": mv[SHORT],
               "nav": nav, "initial_margin": im, "maintenance_margin": mm,
               "available_funds": nav - im, "excess_liquidity": nav - mm, "dividend": div,
               "long_strike": legs[LONG]["strike"] if legs[LONG] else np.nan,
               "short_strike": legs[SHORT]["strike"] if legs[SHORT] else np.nan,
               "long_iv": g[LONG]["iv"], "short_iv": g[SHORT]["iv"]}
        for k in ("delta", "gamma", "vega", "theta"):
            row[k] = CONTRACT * (g[LONG][k] - g[SHORT][k]) + (shares if k == "delta" else 0.0)
        rows.append(row)
        prev_shares = shares
    return pd.DataFrame(rows)


def min_start_cash(ledger: pd.DataFrame, start_cash: float) -> dict:
    """Smallest starting cash keeping available funds at or above zero: one subtraction."""
    i = int(ledger["available_funds"].idxmin())
    worst = float(ledger.at[i, "available_funds"])
    return {"min_cash": start_cash - min(worst, 0.0), "worst_available": worst,
            "worst_date": ledger.at[i, "date"]}


# --------------------------------------------------------------------------
# performance
# --------------------------------------------------------------------------

def performance(ledger: pd.DataFrame, capital: float, tbill: pd.Series) -> dict:
    """
    The registered figures. Monthly returns are month-end NAV over the prior
    month-end NAV, the first from `capital`. The Sharpe ratio is the mean
    monthly return over the bill's monthly yield, divided by the standard
    deviation of that excess, times sqrt(12).
    """
    nav = ledger.set_index("date")["nav"]
    month_end = nav.groupby(nav.index.to_period("M")).last()
    ret = pd.concat([pd.Series([capital]), month_end], ignore_index=True).pct_change().dropna()
    ret.index = month_end.index
    rf = (tbill.reindex(nav.index).groupby(nav.index.to_period("M")).mean() / 12.0).reindex(ret.index)
    excess = ret - rf
    path = pd.concat([pd.Series([capital]), nav], ignore_index=True)
    dd = path / path.cummax() - 1.0
    sd = float(excess.std(ddof=1))
    return {
        "capital": float(capital), "final_nav": float(nav.iloc[-1]),
        "pnl": float(nav.iloc[-1] - capital),
        "return_pct": 100.0 * float(nav.iloc[-1] / capital - 1.0),
        "max_drawdown_pct": 100.0 * float(dd.min()),
        "worst_month": str(ret.idxmin()), "worst_month_pct": 100.0 * float(ret.min()),
        "best_month": str(ret.idxmax()), "best_month_pct": 100.0 * float(ret.max()),
        "monthly_mean_pct": 100.0 * float(ret.mean()), "monthly_sd_pct": 100.0 * float(ret.std(ddof=1)),
        "sharpe": float(excess.mean() / sd * np.sqrt(12)) if sd > 0 else np.nan,
        "months": int(len(ret)),
        "monthly_returns": {str(k): float(v) for k, v in ret.items()},
    }


def pnl_by_leg(result: dict, ledger: pd.DataFrame) -> dict:
    """LEAP, short calls, stock and dividends: cash flows plus closing value, summing to P&L."""
    cash = {LONG: 0.0, SHORT: 0.0, "stock": 0.0}
    for ev in result["blotter"]:
        cash[ev["leg"] or "stock"] += ev["cash_delta"]
    end = ledger.iloc[-1]
    return {"long": cash[LONG] + float(end["long_mv"]),
            "short": cash[SHORT] + float(end["short_mv"]),
            "stock": cash["stock"] + float(end["stock_mv"]),
            "dividends": float(ledger["dividend"].sum())}
