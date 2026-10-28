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
EARLY_PUT_TIME_VALUE = 0.05    # config/wheel_managed.yaml: a put this close to intrinsic is exercised


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
    extra = {}
    if (cache / "stock_history.pkl").exists():      # closes before the window, for the trend average
        h = pd.to_numeric(pd.read_pickle(cache / "stock_history.pkl"), errors="coerce").dropna().sort_index()
        h.index = pd.DatetimeIndex(h.index).normalize()
        extra["history"] = h * unit_factor(h.index, splits)
    return {**extra,
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

def _num(x) -> float:
    """A quote as a float, NaN when LSEG left it empty (a missing bid comes back as <NA>)."""
    return float("nan") if x is None or x is pd.NA or pd.isna(x) else float(x)


def run(market: dict, *, start, end, mode: str = "wheel", put_delta: float = 0.25,
        call_delta: float = 0.25, floor: bool = True, fill: str = "mid",
        start_cash: float = START_CASH, ticker: str = "", manage: dict | None = None) -> dict:
    """
    Walk the sessions of [start, end] and book what the rules say.

    `fill` is "mid" or "cross" (sell at the bid, buy back at the ask). Returns
    the blotter, one record per option written, and the daily ledger.

    `manage` turns on config/wheel_managed.yaml's rules, each by its key; empty
    or None is the wheel as first registered, every option held to expiry:

        take_profit       buy the short back once it costs this fraction of its sale price
        roll_days         a put in the money this many calendar days before expiry is
                          rolled to the next monthly, same or lower strike, for a credit
        max_rolls         rolls in a row before a put is held to assignment
        early_assignment  American exercise: a put with EARLY_PUT_TIME_VALUE or less of
                          time value; a call whose time value is below the next session's dividend
        trend_sma         no new put while the close is below the average of this many closes
        redeploy          after a take-profit close, sell into the next monthly at once
                          (from the first session it is quoted) instead of at the next entry
        roll_lower        roll only to a strike below the current one

    and config/wheel_choices.yaml's, each of which only stops a put being sold or frees a call:

        skip_earnings     no new put when an earnings date falls before its expiry
        close_before_earnings  buy the put back at the close before an earnings date and sell a
                          new one into the same expiry after it, if a week or more remains
        iv_over_rv        no new put unless its implied vol is above the realised vol of this many sessions
        unstick_after     drop the cost-basis floor once shares have been held through this many call cycles
    """
    manage = manage or {}
    stock, quotes = market["stock"], market["quotes"]
    lot = market.get("lot", pd.Series(float(CONTRACT), index=stock.index))
    sessions = stock.index[(stock.index >= pd.Timestamp(start)) & (stock.index <= pd.Timestamp(end))]
    expiries = monthly_expiries(sessions, start, end)
    entries = {sessions[sessions > prev][0]: exp for prev, exp in zip(expiries[:-1], expiries[1:])}
    by_day = {d: g for d, g in quotes[quotes["date"].isin(sessions)].groupby("date")}
    empty = quotes.iloc[:0]
    marks = {}
    price = (lambda row: _num(row["bid"])) if fill == "cross" else (lambda row: _num(row["mid"]))
    buy_price = (lambda row: _num(row["ask"])) if fill == "cross" else (lambda row: _num(row["mid"]))

    # what the managed rules need to know in advance: roll dates, the next monthly,
    # the session before each ex-dividend date, and the trend average
    next_expiry = dict(zip(expiries[:-1], expiries[1:]))
    roll_on = {}
    if manage.get("roll_days"):
        for e in expiries:
            before = sessions[sessions <= e - pd.Timedelta(days=int(manage["roll_days"]))]
            if len(before):
                roll_on[e] = before[-1]
    next_session = dict(zip(sessions[:-1], sessions[1:]))
    ex_dates = set(market["dividends"].index)
    closes = stock["close"]
    if market.get("history") is not None:
        hist = market["history"]
        closes = pd.concat([hist[hist.index < closes.index[0]], closes]).sort_index()
    sma, trend_n = None, manage.get("trend_sma")
    if trend_n:
        sma = closes.rolling(int(trend_n), min_periods=int(trend_n)).mean()
        if entries and not np.isfinite(sma.get(min(entries), np.nan)):
            raise ValueError(f"the trend filter needs {trend_n} closes before {min(entries).date()}")
    rv, rv_n = None, manage.get("iv_over_rv")
    if rv_n:
        rv = np.log(closes.astype(float)).diff().rolling(int(rv_n)).std(ddof=1) * np.sqrt(252)
        if entries and not np.isfinite(rv.get(min(entries), np.nan)):
            raise ValueError(f"the implied-over-realised filter needs {rv_n} returns before {min(entries).date()}")
    unstick_after = manage.get("unstick_after")
    share_cycles = 0         # call-cycle entries since the shares now held were assigned

    cash, shares, basis = float(start_cash), 0, np.nan
    option = None            # the open short: ric, cp, strike, expiry, units, fill, credit, rolls, cycle
    pending = None           # after a take-profit close with `redeploy`: the expiry to re-sell into
    after_earnings = None    # after closing a put before earnings: (earnings date, expiry to re-sell into)
    earnings = pd.DatetimeIndex(market["earnings"]).sort_values()
    blotter, cycles, rows = [], [], []
    totals = {"premium": 0.0, "interest": 0.0, "dividends": 0.0}
    prev_day = None

    def book(day, kind, side, qty, px, cash_delta, note, row=None, reason=None):
        ev = {"date": day, "kind": kind, "side": side, "qty": qty, "price": px,
              "cash_delta": cash_delta, "note": note, "instrument": ticker}
        if row is not None:
            ev.update(instrument=row["ric"], cp=row["cp"], strike=float(row["strike"]),
                      expiry=row["expiry"], bid=_num(row["bid"]), ask=_num(row["ask"]))
        if reason is not None:
            ev["reason"] = reason
        blotter.append(ev)

    def quote(day, ric):
        today = by_day.get(day, empty)
        hit = today[today["ric"] == ric]
        return None if hit.empty else hit.iloc[0]

    def assign_put(day, note, status, reason=None):
        nonlocal cash, shares, basis, share_cycles
        share_cycles = 0
        rec, k, n = option["cycle"], option["strike"], option["units"]
        book(day, "put", ASSIGN, n, k, 0.0, note, reason=reason)
        book(day, "stock", BUY, n, k, -n * k, "bought at the strike against assignment")
        cash -= n * k
        shares += n
        basis = k - option["credit"]
        rec["status"] = status

    def call_away(day, note, status, reason=None):
        nonlocal cash, shares, basis
        rec, k, n = option["cycle"], option["strike"], option["units"]
        book(day, "call", ASSIGN, n, k, 0.0, note, reason=reason)
        book(day, "stock", SELL, n, k, n * k, "delivered at the strike")
        cash += n * k
        shares -= n
        rec["realised_vs_basis"] = n * (k - basis)
        basis = np.nan
        rec["status"] = status

    def buy_back(day, row, p, note, reason):
        nonlocal cash, basis
        n = option["units"]
        cash -= n * p
        totals["premium"] -= n * p
        option["credit"] -= p
        if option["cp"] == CALL:
            basis += p
        book(day, "put" if option["cp"] == PUT else "call", BUY, n, p, -n * p, note, row=row, reason=reason)
        option["cycle"].update(close_day=day, close_price=p)

    def write(day, pick, cp, contracts, n, close, rec, note, reason=None):
        nonlocal cash, basis
        p = price(pick)
        cash += n * p
        totals["premium"] += n * p
        if cp == CALL:
            basis -= p
        book(day, "put" if cp == PUT else "call", SELL, n, p, n * p, note, row=pick, reason=reason)
        rec.update(cp=cp, strike=float(pick["strike"]), contracts=contracts, units=n, fill=p,
                   bid=float(pick["bid"]), ask=float(pick["ask"]),
                   delta=float(pick["delta"]), iv=float(pick["iv"]), premium=n * p,
                   otm_pct=100.0 * (float(pick["strike"]) / close - 1.0), status="open")
        return {"ric": pick["ric"], "cp": cp, "strike": float(pick["strike"]), "expiry": rec["expiry"],
                "contracts": contracts, "units": n, "fill": p, "credit": p, "rolls": 0, "cycle": rec}

    def choose(day, expiry, close):
        """
        What the rules would sell on `day` into `expiry`: (cp, target, floor, pick, contracts,
        per, skip, info). `skip` is why no put is sold, or None; `info` goes on the cycle record.
        """
        r, q = carry(market, day)
        per = int(lot.get(day, CONTRACT))
        info, skip = {}, None
        if shares == 0:
            cp, target, lo = PUT, put_delta, None
        else:
            cp, target, lo = CALL, call_delta, (basis if floor else None)
            if unstick_after and share_cycles >= int(unstick_after) and floor:
                lo = None
                info["unstuck"] = True
        if cp == PUT and sma is not None and close < float(sma.loc[day]):
            skip, info["sma"] = f"skipped: below the {trend_n}-close average", float(sma.loc[day])
        elif cp == PUT and manage.get("skip_earnings") and bool(
                ((market["earnings"] > day) & (market["earnings"] <= expiry)).any()):
            skip = "skipped: earnings before expiry"
        pick = None if skip else select(by_day.get(day, empty), cp, expiry, close, day, r, q, target, lo)
        if pick is not None and cp == PUT and rv is not None:
            info["rv"] = float(rv.loc[day])
            if not float(pick["iv"]) > info["rv"]:
                skip = (f"skipped: implied vol {100 * float(pick['iv']):.1f}% not above "
                        f"realised {100 * info['rv']:.1f}%")
                pick = None
        contracts = 0
        if pick is not None:
            k = float(pick["strike"])
            contracts = int(cash // (per * k)) if cp == PUT else shares // per
        return cp, target, lo, pick, contracts, per, skip, info

    def note_for(cp, contracts, target, pick, close, why=""):
        side = "put" if cp == PUT else "call"
        return (f"write {contracts} {side}{'s' if contracts > 1 else ''}{why}: delta closest to "
                f"{target:.2f} (delta {pick['delta']:.2f}, implied vol {100 * pick['iv']:.1f}%), "
                f"spot {close:.2f}; quote {pick['bid']:.2f}/{pick['ask']:.2f}")

    def new_record(day, expiry, close):
        rec = {"entry": day, "expiry": expiry, "spot": close, "dte": (expiry - day).days,
               "earnings_in_cycle": bool(((market["earnings"] > day) & (market["earnings"] <= expiry)).any()),
               "shares_before": shares, "basis_before": basis}
        cycles.append(rec)
        return rec

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
            k = option["strike"]
            option["cycle"]["settle"] = close
            if option["cp"] == PUT and close < k:
                assign_put(day, f"assigned: close {close:.2f} < strike {k:.2f}", "assigned")
            elif option["cp"] == CALL and close > k:
                call_away(day, f"called away: close {close:.2f} > strike {k:.2f}", "called away")
            else:
                book(day, "put" if option["cp"] == PUT else "call", EXPIRE, option["units"], 0.0, 0.0,
                     f"expired: close {close:.2f} vs strike {k:.2f}")
                option["cycle"]["status"] = "expired"
            option = None

        # ---- early assignment: the holder exercises an American option ---
        if option is not None and manage.get("early_assignment"):
            row = quote(day, option["ric"])
            k = option["strike"]
            if row is not None and np.isfinite(_num(row["mid"])):
                if option["cp"] == PUT and close < k:
                    tv = float(row["mid"]) - (k - close)
                    if tv <= EARLY_PUT_TIME_VALUE + 1e-12:
                        option["cycle"]["settle"] = close
                        assign_put(day, f"assigned early: close {close:.2f} < strike {k:.2f}, "
                                        f"{tv:.2f} of time value left", "assigned early", reason="early")
                        option = None
                elif option["cp"] == CALL and close > k and next_session.get(day) in ex_dates:
                    div = float(market["dividends"].loc[[next_session[day]]].sum())
                    tv = float(row["mid"]) - (close - k)
                    if tv < div:
                        option["cycle"]["settle"] = close
                        call_away(day, f"called away early: close {close:.2f} > strike {k:.2f}, time value "
                                       f"{tv:.2f} below tomorrow's {div:.2f} dividend", "called away early",
                                  reason="early")
                        option = None

        # ---- early close: buy back at the take-profit price -------------
        if option is not None and manage.get("take_profit"):
            row = quote(day, option["ric"])
            if row is not None:
                p = buy_price(row)
                if np.isfinite(p) and p <= manage["take_profit"] * option["fill"] + 1e-12:
                    buy_back(day, row, p, f"close: bought back at {p:.2f}, "
                                          f"{100 * p / option['fill']:.0f}% of the {option['fill']:.2f} it was sold at",
                             "take_profit")
                    option["cycle"]["status"] = "closed early"
                    if manage.get("redeploy") and option["expiry"] in next_expiry:
                        pending = next_expiry[option["expiry"]]
                    option = None

        # ---- step aside for earnings: buy the put back the close before --
        if option is not None and option["cp"] == PUT and manage.get("close_before_earnings"):
            nxt = next_session.get(day, day)
            due = earnings[(earnings > day) & (earnings <= nxt)]
            if len(due) and due[0] <= option["expiry"]:
                row = quote(day, option["ric"])
                # the mid, or the ask when nobody bids: a near-worthless put is still bought back
                p = np.nan if row is None else buy_price(row)
                if row is not None and not np.isfinite(p):
                    p = _num(row["ask"])
                if np.isfinite(p):
                    buy_back(day, row, p, f"close before the {due[0].date()} earnings: bought back at {p:.2f}",
                             "earnings")
                    option["cycle"]["status"] = "closed before earnings"
                    after_earnings = (due[0], option["expiry"])
                    option = None

        # ---- roll a challenged put down and out for a credit ------------
        if (option is not None and option["cp"] == PUT and roll_on.get(option["expiry"]) == day
                and close < option["strike"] and option["rolls"] < int(manage.get("max_rolls", 0))
                and option["expiry"] in next_expiry):
            row, nxt, k = quote(day, option["ric"]), next_expiry[option["expiry"]], option["strike"]
            if row is not None and np.isfinite(buy_price(row)):
                pb = buy_price(row)
                chain = by_day.get(day, empty)
                below = (chain["strike"] < k - 1e-9) if manage.get("roll_lower") else (chain["strike"] <= k + 1e-9)
                c = chain[(chain["cp"] == PUT) & (chain["expiry"] == nxt) & chain["mid"].notna()
                          & (chain["mid"] > 0) & below]
                if not c.empty:
                    c = c[np.array([price(r) for _, r in c.iterrows()]) >= pb - 1e-12]
                if not c.empty:
                    new = c.sort_values("strike").iloc[-1]
                    r, q = carry(market, day)
                    g = option_greeks(price(new), close, float(new["strike"]), (nxt - day).days / DAYS_PER_YEAR,
                                      r, q, PUT)
                    new = new.copy()
                    new["delta"], new["iv"] = g["delta"], g["iv"]
                    old = option
                    buy_back(day, row, pb, f"roll: buy back the {k:.2f} put at {pb:.2f}; close {close:.2f} is "
                                           f"below the strike {(old['expiry'] - day).days} days before expiry", "roll")
                    old["cycle"]["status"] = "rolled"
                    rec = new_record(day, nxt, close)
                    rec["rolled_from"] = old["ric"]
                    option = write(day, new, PUT, old["contracts"], old["units"], close, rec,
                                   f"roll: sell the {nxt.date()} {float(new['strike']):.2f} put at {price(new):.2f}, "
                                   f"a net credit of {price(new) - pb:.2f}", reason="roll")
                    option.update(credit=old["credit"] + option["fill"], rolls=old["rolls"] + 1)

        # ---- re-sell after a take-profit close (redeploy) --------------
        if mode == "wheel" and option is None and pending is not None and day not in entries:
            if day >= pending:
                pending = None
            else:
                cp, target, lo, pick, contracts, per, skip, info = choose(day, pending, close)
                if skip:
                    pending = None              # a filter: the cash waits for the next entry
                elif pick is not None and contracts > 0:
                    rec = new_record(day, pending, close)
                    rec.update(info, redeployed=True)
                    if cp == CALL:
                        rec["floor"] = lo
                    option = write(day, pick, cp, contracts, contracts * per, close, rec,
                                   note_for(cp, contracts, target, pick, close, ", re-sold after the 50% close"),
                                   reason="redeploy")
                    pending = None

        # ---- after earnings: sell a new put into the same expiry ---------
        if (mode == "wheel" and option is None and after_earnings is not None and day > after_earnings[0]
                and day not in entries):
            expiry = after_earnings[1]
            after_earnings = None
            if (expiry - day).days >= 7 and shares == 0:
                cp, target, lo, pick, contracts, per, skip, info = choose(day, expiry, close)
                if not skip and pick is not None and contracts > 0:
                    rec = new_record(day, expiry, close)
                    rec.update(info, after_earnings=True)
                    option = write(day, pick, cp, contracts, contracts * per, close, rec,
                                   note_for(cp, contracts, target, pick, close, ", re-sold after earnings"),
                                   reason="after_earnings")

        # ---- entry: open the cycle --------------------------------------
        if mode == "wheel" and day in entries and option is None:
            expiry = entries[day]
            pending = after_earnings = None
            cp, target, lo, pick, contracts, per, skip, info = choose(day, expiry, close)
            rec = new_record(day, expiry, close)
            rec.update(info)
            if cp == CALL:
                rec["floor"] = lo
                share_cycles += 1
            if skip:
                rec["status"] = skip
            elif pick is None:
                rec["status"] = ("held: no qualifying call" if cp == CALL else "idle: no quoted put")
            elif contracts == 0:
                rec["status"] = "idle: cash secures no contract"
            else:
                option = write(day, pick, cp, contracts, contracts * per, close, rec,
                               note_for(cp, contracts, target, pick, close))

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
            "end": pd.Timestamp(end), "ticker": ticker, "manage": dict(manage)}


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
        "puts_written": len(puts),
        "assigned": int(sum(x["status"] in ("assigned", "assigned early") for x in puts)),
        "calls_written": len(calls),
        "called_away": int(sum(x["status"] in ("called away", "called away early") for x in calls)),
        "held_no_call": int((status == "held: no qualifying call").sum()),
        "idle": int(status.str.startswith("idle").sum()),
        "closed_early": int((status == "closed early").sum()), "rolled": int((status == "rolled").sum()),
        "assigned_early": int((status == "assigned early").sum()),
        "called_away_early": int((status == "called away early").sum()),
        "filtered": int(status.str.startswith("skipped").sum()),
        "share_days_pct": 100.0 * float((led["shares"] > 0).mean()) if len(led) else np.nan,
        "monthly_returns": perf["monthly_returns"],
    }


def monthly_nav(ledger: pd.DataFrame, capital: float, tbill: pd.Series) -> list[dict]:
    """
    The month-end account values the Sharpe ratio is built from, as pmcc_daily.performance
    builds it: each month's return is its last NAV over the last one before it (the first
    from `capital`), less the month's average T-bill yield over twelve; the drawdown is the
    month-end NAV against the highest close so far.
    """
    nav = ledger.set_index("date")["nav"]
    me = nav.groupby(nav.index.to_period("M")).last()
    ret = pd.concat([pd.Series([capital]), me], ignore_index=True).pct_change().dropna()
    ret.index = me.index
    rf = (tbill.reindex(nav.index).groupby(nav.index.to_period("M")).mean() / 12.0).reindex(ret.index)
    peak = nav.cummax().clip(lower=capital).groupby(nav.index.to_period("M")).last()
    return [{"month": str(mo), "nav": float(me[mo]), "ret": float(ret[mo]), "rf": float(rf[mo]),
             "excess": float(ret[mo] - rf[mo]), "cum_ret": float(me[mo] / capital - 1.0),
             "drawdown": float(me[mo] / peak[mo] - 1.0)} for mo in me.index]


def reg_t(ledger: pd.DataFrame) -> pd.DataFrame:
    """
    The account under Reg T as a cash account, which is what a cash-secured wheel is:
    an open put must have its strike times its shares set aside in cash, and an open call
    must be covered by shares held; nothing is borrowed.

        put_collateral      strike x shares under the open put, 0 otherwise
        free_cash           cash - put_collateral; never below zero if every put is secured
        call_shares_needed  shares under the open call, 0 otherwise
        covered             shares held >= call_shares_needed
    """
    put = ledger["short"] == PUT
    call = ledger["short"] == CALL
    collateral = np.where(put, ledger["short_strike"] * ledger["short_units"], 0.0)
    needed = np.where(call, ledger["short_units"], 0).astype(int)
    return pd.DataFrame({"date": ledger["date"], "cash": ledger["cash"], "put_collateral": collateral,
                         "free_cash": ledger["cash"] - collateral, "shares": ledger["shares"],
                         "call_shares_needed": needed, "covered": ledger["shares"] >= needed,
                         "stock_mv": ledger["stock_mv"], "nav": ledger["nav"]})


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
