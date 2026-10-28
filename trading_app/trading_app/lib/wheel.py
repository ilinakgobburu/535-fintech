"""
The wheel: sell cash-secured puts until one is assigned, then covered calls on
the shares until they are called away, then puts again.

Every rule here is a line of config/wheel.yaml, which was committed before
this module produced any output. Three books run through ONE daily loop, so
they cannot differ by implementation:

    mode="wheel"          the wheel
    mode="buy_and_hold"   the starting cash in whole shares at the first close
    mode="cash"           the starting cash in T-bills, nothing else

WHAT IS KNOWN WHEN. A decision on day d uses day d's closing quotes and fills
at them. Expiry settles on the expiry day's close. Nothing reads a later row
than the one it trades on.

THE CYCLE. Each monthly expiry opens a cycle on the next session:

    no shares   sell puts on the next monthly expiry, as many contracts as the
                cash already in the account secures at the strike
    shares      sell one call per contract's worth of shares
    expiry      put:  close < strike -> assigned: buy the shares at the strike
                call: close > strike -> called away: sell them at the strike
                otherwise the option expires and the next cycle repeats the phase

A short put and cash equal to its strike is the same payoff as 100 shares and
a short call at that strike (put-call parity), so the book is long the stock
below the strike from the moment the put is sold. Assignment changes the form
of that position, not its risk.

COST BASIS. The assigned strike, less that put's premium, less every call
premium collected on the shares since. With `floor`, a call is written only at
or above it, so being called away never realises a loss on the shares; if no
quoted strike qualifies, the shares are held with no call that cycle.

    NAV = cash + shares x close - contracts x shares per contract x option mid

Cash earns the 3-month T-bill over the calendar days between closes: a
cash-secured put's collateral sits in the account, and the bill is what that
cash would earn anyway. Dividends are credited on the ex-date to the shares
held at the prior close.

UNITS. Prices, strikes and dividends are in the shares after every split up to
the cache's `units_after` (NVDA: after the 4:1 split of 2021-07-20). A real
contract covered 100 shares of the day, which is 100 / unit_factor of these
units, so before that split one NVDA contract is 400 units. Sizing in real
contracts keeps every trade one that could have been placed.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .pmcc_daily import carry, monthly_expiries, performance
from .ric import parse_option_ric
from .vol import bs_greeks, implied_vol

CONTRACT = 100
START_CASH = 100_000.0
PUT, CALL = "P", "C"
SELL, ASSIGN, EXPIRE, BUY = "SELL", "ASSIGN", "EXPIRE", "BUY"
DAYS_PER_YEAR = 365.0


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def read_splits(cache: str | Path) -> tuple[pd.Timestamp, list] | None:
    """The cache's splits.json, if it has one: (units_after, [(date, ratio), ...])."""
    path = Path(cache) / "splits.json"
    if not path.exists():
        return None
    s = json.loads(path.read_text())
    return pd.Timestamp(s["units_after"]), [(pd.Timestamp(d), float(r)) for d, r in s["splits"]]


def unit_factor(when, splits: tuple[pd.Timestamp, list] | None) -> np.ndarray:
    """
    What turns a per-share amount quoted on `when` into the backtest's units.
    LSEG quotes stocks and options unadjusted, and lists a contract under its
    strike after every split up to its expiry, so a strike takes the factor of
    its expiry.
    """
    when = pd.DatetimeIndex(pd.to_datetime(when))
    if splits is None:
        return np.ones(len(when))
    units_after, events = splits
    held, unit = np.ones(len(when)), 1.0
    for day, ratio in events:
        held *= np.where(when >= day, ratio, 1.0)
        unit *= ratio if day <= units_after else 1.0
    return held / unit


def load_market(cache: str | Path) -> dict:
    """
    The raw pull as the engine reads it, puts and calls together.

    `quotes` has one row per contract per day, with `cp`, and a `mid` that is
    NaN unless both sides are present and uncrossed. LSEG returns a zero bid
    as missing, so an option nobody bids for has no mid and cannot be sold.
    `lot` is the units one real contract covered on each session.
    """
    cache = Path(cache)
    splits = read_splits(cache)
    frames = pickle.loads((cache / "options.pkl").read_bytes())
    blocks = []
    for ric, df in frames.items():
        meta = parse_option_ric(re.sub(r"^([A-Z]+)\d(?=[A-X]\d{9}\.)", r"\1", ric))
        if meta is None:
            continue
        q = pd.DataFrame({"bid": pd.to_numeric(df["BID"], errors="coerce"),
                          "ask": pd.to_numeric(df["ASK"], errors="coerce")})
        q.index = pd.DatetimeIndex(df.index).normalize()
        q = q[q.notna().any(axis=1)]
        q["ric"], q["cp"] = ric, meta["cp"]
        q["strike"], q["expiry"] = meta["strike"], pd.Timestamp(meta["expiry"])
        blocks.append(q.rename_axis("date").reset_index())
    quotes = pd.concat(blocks, ignore_index=True)
    f = unit_factor(quotes["date"], splits)
    quotes["bid"], quotes["ask"] = quotes["bid"] * f, quotes["ask"] * f
    quotes["strike"] = quotes["strike"] * unit_factor(quotes["expiry"], splits)
    ok = quotes["bid"].notna() & quotes["ask"].notna() & (quotes["ask"] >= quotes["bid"])
    quotes["mid"] = np.where(ok, (quotes["bid"] + quotes["ask"]) / 2.0, np.nan)

    raw = pd.read_pickle(cache / "stock_unadjusted_ohlc.pkl")
    stock = pd.DataFrame({"close": pd.to_numeric(raw["TRDPRC_1"], errors="coerce"),
                          "open": pd.to_numeric(raw["OPEN_PRC"], errors="coerce")})
    stock.index = pd.DatetimeIndex(raw.index).normalize()
    stock = stock.dropna(subset=["close"]).sort_index()
    stock = stock.mul(unit_factor(stock.index, splits), axis=0)

    aux = pickle.loads((cache / "aux.pkl").read_bytes())
    tb = pd.to_numeric(aux["tbill"]["MID_YLD_1"], errors="coerce") / 100.0
    tb.index = pd.DatetimeIndex(tb.index).normalize()
    div = aux["dividends"].iloc[:, 1:3].copy()
    div.columns = ["ex_date", "amount"]
    div["ex_date"] = pd.to_datetime(div["ex_date"])
    div = div.dropna(subset=["ex_date"])
    div = div[pd.to_numeric(div["amount"], errors="coerce") > 0]
    div["amount"] = div["amount"].astype(float) * unit_factor(div["ex_date"], splits)
    earnings = pd.DatetimeIndex(pd.to_datetime(aux["earnings"].iloc[:, 1]).dropna()).normalize()
    return {
        "quotes": quotes, "stock": stock,
        "lot": pd.Series(np.rint(CONTRACT / unit_factor(stock.index, splits)), index=stock.index),
        "tbill": tb.reindex(stock.index.union(tb.index)).ffill().reindex(stock.index),
        "dividends": pd.Series(div["amount"].to_numpy(float),
                               index=pd.DatetimeIndex(div["ex_date"]).normalize()).sort_index(),
        "earnings": earnings.sort_values(),
    }


# --------------------------------------------------------------------------
# greeks and selection
# --------------------------------------------------------------------------

def option_greeks(mid: float, spot: float, strike: float, T: float,
                  r: float = 0.0, q: float = 0.0, cp: str = CALL) -> dict:
    """
    Implied vol of one quoted option and its delta to SPOT, per share: a put's
    delta is negative. lib.vol's Black-76 with F = S exp((r-q)T), D = exp(-rT),
    as lib.pmcc.call_greeks does for calls. A quote with no implied vol returns
    iv = NaN.
    """
    out = {"iv": np.nan, "delta": np.nan}
    if not np.isfinite([mid, spot, strike, T]).all() or T <= 0 or spot <= 0:
        return out
    F, D = spot * np.exp((r - q) * T), np.exp(-r * T)
    iv = implied_vol(mid, F, strike, T, D, cp)
    if not np.isfinite(iv):
        return out
    return {"iv": float(iv), "delta": bs_greeks(F, strike, T, iv, D, cp)["delta"] * F / spot}


def select(day_quotes: pd.DataFrame, cp: str, expiry, spot: float, day, r: float, q: float,
           target: float, floor: float | None = None) -> pd.Series | None:
    """
    The strike of `expiry` whose |delta|, from its own mid, is closest to
    `target`, among two-sided quotes at or above `floor`. A tie goes further
    out of the money: the lower strike for a put, the higher for a call.
    """
    c = day_quotes[(day_quotes["cp"] == cp) & (day_quotes["expiry"] == expiry)
                   & day_quotes["mid"].notna() & (day_quotes["mid"] > 0)]
    if floor is not None:
        c = c[c["strike"] >= floor - 1e-9]
    if c.empty:
        return None
    g = [option_greeks(float(m), spot, float(k), (e - day).days / DAYS_PER_YEAR, r, q, cp)
         for m, k, e in zip(c["mid"], c["strike"], c["expiry"])]
    c = c.assign(iv=[x["iv"] for x in g], delta=[x["delta"] for x in g])
    c = c[c["iv"].notna()]
    if c.empty:
        return None
    c = c.assign(miss=(c["delta"].abs() - target).abs())
    return c.sort_values(["miss", "strike"], ascending=[True, cp == PUT]).iloc[0]


def coverage(market: dict, start, end, lo: float = 0.15, hi: float = 0.35) -> dict:
    """
    config/mr_screen.yaml's data check (d): on each cycle's entry day, as run()
    opens it, do the two-sided puts and the two-sided calls of that cycle's
    expiry each reach |delta| at or below `lo` and at or above `hi`? One record
    per cycle, and how many cycles fall short.
    """
    stock, quotes = market["stock"], market["quotes"]
    sessions = stock.index[(stock.index >= pd.Timestamp(start)) & (stock.index <= pd.Timestamp(end))]
    expiries = monthly_expiries(sessions, start, end)
    out = []
    for prev, expiry in zip(expiries[:-1], expiries[1:]):
        day = sessions[sessions > prev][0]
        spot = float(stock.at[day, "close"])
        r, q = carry(market, day)
        chain = quotes[(quotes["date"] == day) & (quotes["expiry"] == expiry)
                       & quotes["mid"].notna() & (quotes["mid"] > 0)]
        rec = {"entry": day, "expiry": expiry, "spot": spot}
        for cp in (PUT, CALL):
            c = chain[chain["cp"] == cp]
            d = [abs(option_greeks(float(m), spot, float(k), (expiry - day).days / DAYS_PER_YEAR, r, q, cp)["delta"])
                 for m, k in zip(c["mid"], c["strike"])]
            d = [x for x in d if np.isfinite(x)]
            rec[f"{cp}_min"], rec[f"{cp}_max"] = (min(d), max(d)) if d else (np.nan, np.nan)
            rec[f"{cp}_spans"] = bool(d) and min(d) <= lo + 1e-9 and max(d) >= hi - 1e-9
        rec["ok"] = rec["P_spans"] and rec["C_spans"]
        out.append(rec)
    short = sum(not r["ok"] for r in out)
    return {"cycles": out, "short": short, "short_pct": 100.0 * short / len(out) if out else np.nan}


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

def run(market: dict, *, start, end, mode: str = "wheel", put_delta: float = 0.25,
        call_delta: float = 0.25, floor: bool = True, fill: str = "mid",
        start_cash: float = START_CASH, ticker: str = "") -> dict:
    """
    Walk the sessions of [start, end] and book what the rules say.

    `fill` is "mid" or "cross" (sell at the bid; the book never buys an option
    back). Returns the blotter, one record per cycle, and the daily ledger.
    """
    stock, quotes = market["stock"], market["quotes"]
    lot = market.get("lot", pd.Series(float(CONTRACT), index=stock.index))
    sessions = stock.index[(stock.index >= pd.Timestamp(start)) & (stock.index <= pd.Timestamp(end))]
    expiries = monthly_expiries(sessions, start, end)
    entries = {sessions[sessions > prev][0]: exp for prev, exp in zip(expiries[:-1], expiries[1:])}
    by_day = {d: g for d, g in quotes[quotes["date"].isin(sessions)].groupby("date")}
    empty = quotes.iloc[:0]
    marks = {}
    price = (lambda row: float(row["bid"])) if fill == "cross" else (lambda row: float(row["mid"]))

    cash, shares, basis = float(start_cash), 0, np.nan
    option = None            # the open short: ric, cp, strike, expiry, units, cycle
    blotter, cycles, rows = [], [], []
    totals = {"premium": 0.0, "interest": 0.0, "dividends": 0.0}
    prev_day = None

    def book(day, kind, side, qty, px, cash_delta, note, row=None):
        ev = {"date": day, "kind": kind, "side": side, "qty": qty, "price": px,
              "cash_delta": cash_delta, "note": note, "instrument": ticker}
        if row is not None:
            ev.update(instrument=row["ric"], cp=row["cp"], strike=float(row["strike"]),
                      expiry=row["expiry"], bid=float(row["bid"]), ask=float(row["ask"]))
        blotter.append(ev)

    for day in sessions:
        close = float(stock.at[day, "close"])

        # ---- carry since the last close ---------------------------------
        if prev_day is not None:
            r_prev = float(market["tbill"].get(prev_day, np.nan))
            if np.isfinite(r_prev) and cash > 0:
                accrued = cash * r_prev * (day - prev_day).days / DAYS_PER_YEAR
                cash += accrued
                totals["interest"] += accrued
            if shares and day in market["dividends"].index:
                paid = float(market["dividends"].loc[[day]].sum()) * shares
                cash += paid
                totals["dividends"] += paid
                book(day, "stock", "DIVIDEND", shares, paid / shares, paid, "dividend on shares held")

        # ---- the first session: the benchmarks buy -----------------------
        if prev_day is None and mode == "buy_and_hold":
            shares = int(cash // close)
            cash -= shares * close
            book(day, "stock", BUY, shares, close, -shares * close, "buy and hold: whole shares at the close")

        # ---- expiry: settle on the close --------------------------------
        if option is not None and day == option["expiry"]:
            rec, k, n = option["cycle"], option["strike"], option["units"]
            rec["settle"] = close
            if option["cp"] == PUT and close < k:
                book(day, "put", ASSIGN, n, k, 0.0, f"assigned: close {close:.2f} < strike {k:.2f}")
                book(day, "stock", BUY, n, k, -n * k, "bought at the strike against assignment")
                cash -= n * k
                shares += n
                basis = k - option["fill"]
                rec["status"] = "assigned"
            elif option["cp"] == CALL and close > k:
                book(day, "call", ASSIGN, n, k, 0.0, f"called away: close {close:.2f} > strike {k:.2f}")
                book(day, "stock", SELL, n, k, n * k, "delivered at the strike")
                cash += n * k
                shares -= n
                rec["realised_vs_basis"] = n * (k - basis)
                basis = np.nan
                rec["status"] = "called away"
            else:
                book(day, "put" if option["cp"] == PUT else "call", EXPIRE, n, 0.0, 0.0,
                     f"expired: close {close:.2f} vs strike {k:.2f}")
                rec["status"] = "expired"
            option = None

        # ---- entry: open the cycle --------------------------------------
        if mode == "wheel" and day in entries:
            expiry = entries[day]
            r, q = carry(market, day)
            per = int(lot.get(day, CONTRACT))
            rec = {"entry": day, "expiry": expiry, "spot": close, "dte": (expiry - day).days,
                   "earnings_in_cycle": bool(((market["earnings"] > day)
                                              & (market["earnings"] <= expiry)).any()),
                   "shares_before": shares, "basis_before": basis}
            cycles.append(rec)
            if shares == 0:
                cp, target, lo = PUT, put_delta, None
            else:
                cp, target, lo = CALL, call_delta, (basis if floor else None)
                rec["floor"] = lo
            pick = select(by_day.get(day, empty), cp, expiry, close, day, r, q, target, lo)
            if pick is None:
                rec["status"] = ("held: no qualifying call" if cp == CALL else "idle: no quoted put")
            else:
                k = float(pick["strike"])
                contracts = int(cash // (per * k)) if cp == PUT else shares // per
                if contracts == 0:
                    rec["status"] = "idle: cash secures no contract"
                else:
                    n = contracts * per
                    p = price(pick)
                    cash += n * p
                    totals["premium"] += n * p
                    if cp == CALL:
                        basis -= p
                    side = "put" if cp == PUT else "call"
                    book(day, side, SELL, n, p, n * p,
                         f"write {contracts} {side}{'s' if contracts > 1 else ''}: delta closest to "
                         f"{target:.2f} (delta {pick['delta']:.2f}, implied vol {100 * pick['iv']:.1f}%), "
                         f"spot {close:.2f}; quote {pick['bid']:.2f}/{pick['ask']:.2f}", row=pick)
                    rec.update(cp=cp, strike=k, contracts=contracts, units=n, fill=p,
                               bid=float(pick["bid"]), ask=float(pick["ask"]),
                               delta=float(pick["delta"]), iv=float(pick["iv"]), premium=n * p,
                               otm_pct=100.0 * (k / close - 1.0), status="open")
                    option = {"ric": pick["ric"], "cp": cp, "strike": k, "expiry": expiry,
                              "contracts": contracts, "units": n, "fill": p, "cycle": rec}

        # ---- mark --------------------------------------------------------
        option_mv, mark = 0.0, np.nan
        if option is not None:
            today = by_day.get(day, empty)
            m = today.loc[today["ric"] == option["ric"], "mid"].dropna()
            if len(m):
                marks[option["ric"]] = float(m.iloc[0])
            mark = marks.get(option["ric"], option["fill"])
            option_mv = -option["units"] * mark
        rows.append({"date": day, "cash": cash, "shares": shares, "stock_close": close,
                     "stock_mv": shares * close, "option_mv": option_mv,
                     "nav": cash + shares * close + option_mv,
                     "short": None if option is None else option["cp"],
                     "short_strike": np.nan if option is None else option["strike"],
                     "short_ric": None if option is None else option["ric"],
                     "short_expiry": None if option is None else option["expiry"],
                     "short_contracts": np.nan if option is None else option["contracts"],
                     "short_units": np.nan if option is None else option["units"],
                     "short_mark": mark, "basis": basis})
        prev_day = day

    ledger = pd.DataFrame(rows)
    return {"blotter": blotter, "cycles": cycles, "ledger": ledger, "totals": totals,
            "mode": mode, "put_delta": put_delta, "call_delta": call_delta, "floor": floor,
            "fill": fill, "start_cash": float(start_cash), "start": pd.Timestamp(start),
            "end": pd.Timestamp(end), "ticker": ticker}


# --------------------------------------------------------------------------
# what it did
# --------------------------------------------------------------------------

def attribution(result: dict) -> dict:
    """
    P&L in four parts that sum to it: premium written, T-bill interest,
    dividends, and the shares (bought and sold at strikes, marked at the last
    close). An open short at the end is its own fifth part.
    """
    end = result["ledger"].iloc[-1]
    stock = sum(ev["cash_delta"] for ev in result["blotter"]
                if ev["kind"] == "stock" and ev["side"] in (BUY, SELL)) + float(end["stock_mv"])
    t = result["totals"]
    parts = {"premium": t["premium"], "interest": t["interest"], "dividends": t["dividends"],
             "shares": stock, "open_short": float(end["option_mv"])}
    parts["pnl"] = float(end["nav"]) - result["start_cash"]
    return parts


def summary(result: dict, tbill: pd.Series) -> dict:
    """The registered figures for one book."""
    perf = performance(result["ledger"], result["start_cash"], tbill)
    c = result["cycles"]
    status = pd.Series([x.get("status", "") for x in c], dtype=object)
    puts = [x for x in c if x.get("cp") == PUT]
    calls = [x for x in c if x.get("cp") == CALL]
    led = result["ledger"]
    return {
        **{k: v for k, v in perf.items() if k != "monthly_returns"},
        "attribution": attribution(result),
        "cycles": len(c),
        "puts_written": len(puts), "assigned": int(sum(x["status"] == "assigned" for x in puts)),
        "calls_written": len(calls), "called_away": int(sum(x["status"] == "called away" for x in calls)),
        "held_no_call": int((status == "held: no qualifying call").sum()),
        "idle": int(status.str.startswith("idle").sum()),
        "share_days_pct": 100.0 * float((led["shares"] > 0).mean()) if len(led) else np.nan,
        "monthly_returns": perf["monthly_returns"],
    }


def grid_cells(put_deltas, call_deltas, floors) -> list[dict]:
    """Every combination, in a fixed order: put delta, then call delta, then floor on before off."""
    return [{"put_delta": p, "call_delta": c, "floor": f}
            for p in put_deltas for c in call_deltas for f in sorted(floors, reverse=True)]


def pick(scores: list[dict], key: str = "sharpe") -> dict:
    """
    The cell with the highest `key`. A tie, to the printed precision of four
    decimals, goes to the earlier cell in grid order, so the pick never depends
    on floating-point noise.
    """
    finite = [s for s in scores if np.isfinite(s[key])]
    best = max(round(s[key], 4) for s in finite)
    return next(s for s in finite if round(s[key], 4) == best)
