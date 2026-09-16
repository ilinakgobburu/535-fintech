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
             S_T >  K  assigned. TWO blotter rows, as the assignment specifies:
                       ASSIGN on the call (closes the short, no cash), and a
                       stock SELL of 100 at the strike (cash += 100 x K). Flat.
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

from .ric import occ_symbol, parse_option_ric
from .vol import implied_vol

SHARES_PER_CONTRACT = 100
INITIAL_RATE = 0.50   # Reg T initial on a long equity position
MAINT_RATE = 0.25     # FINRA maintenance

SESSION_START_UTC = 13
SESSION_END_UTC = 20
# The regular session in UTC. The whole backtest window is inside US daylight
# saving time (EDT, UTC-4), so 9:30-16:00 ET is 13:30-20:00 UTC throughout.
SESSION_OPEN = dt.time(13, 30)
SESSION_CLOSE = dt.time(20, 0)

# Used to work out which level of a MultiIndex holds field names. See
# stock_panel: guessing by position is what HW1's loaders got wrong.
KNOWN_FIELDS = {
    "TRDPRC_1", "OPEN_PRC", "HIGH_1", "LOW_1", "BID", "ASK",
    "ACVOL_UNS", "NUM_MOVES", "MID_PRICE", "SETTLE", "CLOSE",
}

# BAR TIMESTAMPS ARE AS-OF, NOT LSEG'S.
#
# LSEG stamps an intraday bar with its START: the bar it calls 15:00 covers
# 15:00-16:00, its TRDPRC_1 is the last trade before 16:00 and its BID/ASK is
# the quote standing at 15:59. Proved against a one-minute pull of the same
# data: the hourly price equals the last minute of [H, H+1) on 300 of 300
# bars, and of [H-1, H) on 1.
#
# The first version of this book used LSEG's labels as if they were the
# moment the prices were observed, which put everything one period early and,
# worse, treated LSEG's "20:00" bar -- 4 to 5pm ET, after-hours trading -- as
# the session close. Every end-of-day mark and every expiry settlement came
# from after-hours prints: 0 of 49 days matched the official close, median
# miss $0.31, worst $15.27.
#
# So the loaders relabel each bar to the END of its window, the instant its
# prices are as of, and keep only bars whose window closes inside the regular
# session. The bar that ends at 20:00 is then genuinely the closing hour, and
# the after-hours bar is gone.
BAR_STEP = {
    "1h": pd.Timedelta(hours=1), "hourly": pd.Timedelta(hours=1),
    "1min": pd.Timedelta(minutes=1), "minute": pd.Timedelta(minutes=1),
}

# Order bars, stamped as-of: the seven hourly bars that close inside the
# session, 14:00 through 20:00 UTC.
TRADEABLE_HOURS = tuple(range(SESSION_START_UTC + 1, SESSION_END_UTC + 1))

BUY, SELL, EXPIRE, ASSIGN = "BUY", "SELL", "EXPIRE", "ASSIGN"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def bar_step(payload: dict) -> pd.Timedelta:
    """The bar width the cache was pulled at. Caches without a label are hourly."""
    iv = str(payload.get("interval", "1h")).lower()
    if iv not in BAR_STEP:
        raise ValueError(f"unknown bar interval {iv!r}; expected one of {sorted(BAR_STEP)}")
    return BAR_STEP[iv]


def in_session_as_of(start: pd.DatetimeIndex, step: pd.Timedelta):
    """
    From LSEG's START stamps, return (keep-mask, as-of stamps).

    A bar is kept when its window ENDS inside the regular session: after the
    open, and no later than the close. That admits the bar that contains the
    opening trades and the bar that ends at the closing bell, and it excludes
    premarket bars and the after-hours bar LSEG labels 20:00.
    """
    start = pd.DatetimeIndex(start)
    end = start + step
    day = start.normalize()
    open_ = day + pd.Timedelta(hours=SESSION_OPEN.hour, minutes=SESSION_OPEN.minute)
    close = day + pd.Timedelta(hours=SESSION_CLOSE.hour, minutes=SESSION_CLOSE.minute)
    keep = np.asarray((end > open_) & (end <= close))
    return keep, end


def is_close_bar(ts) -> bool:
    return pd.Timestamp(ts).time() == SESSION_CLOSE


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
    """
    Underlying bars, regular session only, stamped AS-OF (end of bar), one row
    per timestamp, with the exchange's official close attached to every row of
    its date as OFFICIAL_CLOSE (NaN where the cache has none).
    """
    df = payload["stock"].copy()
    df.index = pd.DatetimeIndex(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        # Which level holds the FIELD names? Decided by counting how many
        # labels on each level are fields we know, not by taking the last one.
        # Taking the last level assumes a (RIC, Field) ordering; handed the
        # (Field, RIC) ordering LSEG also emits, it collapsed every column to
        # the same RIC name and then died inside pd.to_numeric with a TypeError
        # about its argument -- a failure a long way from its cause. This is
        # HW1's loaders lesson, which this module had not applied.
        scores = [
            sum(str(v).upper() in KNOWN_FIELDS
                for v in pd.unique(df.columns.get_level_values(i)))
            for i in range(df.columns.nlevels)
        ]
        if max(scores) == 0:
            raise ValueError(
                f"no level of {list(df.columns)[:3]}... holds recognisable field "
                f"names; refusing to guess which is the RIC")
        lvl = int(np.argmax(scores))
        df.columns = [str(c[lvl]) for c in df.columns]
    df.columns = [str(c).upper() for c in df.columns]
    if df.columns.duplicated().any():
        dupes = sorted({c for c in df.columns[df.columns.duplicated()]})
        raise ValueError(f"duplicate stock columns after flattening: {dupes}")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    keep, as_of = in_session_as_of(df.index, bar_step(payload))
    df = df[keep]
    df.index = pd.DatetimeIndex(as_of[keep])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["date"] = [t.date() for t in df.index]
    closes = payload.get("stock_official_close") or {}
    df["OFFICIAL_CLOSE"] = [closes.get(str(d), np.nan) for d in df["date"]]
    return df


def stock_marks(stock: pd.DataFrame) -> pd.Series:
    """
    The price each bar is marked at: the last trade, except that the bar ending
    at the closing bell is marked at the official close when one is known.

    The closing auction prints at 16:00:00 ET, after the last trade in the
    final hourly bar, and it is the price the market settles on.
    """
    px = pd.to_numeric(stock["TRDPRC_1"], errors="coerce").astype(float)
    if "OFFICIAL_CLOSE" not in stock.columns:
        return px
    official = pd.to_numeric(stock["OFFICIAL_CLOSE"], errors="coerce").astype(float)
    at_close = np.array([is_close_bar(t) for t in stock.index])
    use = at_close & official.notna().to_numpy()
    return px.where(~use, official)


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
    keep, as_of = in_session_as_of(pd.DatetimeIndex(panel["ts"]), bar_step(payload))
    panel = panel[keep].copy()
    panel["ts"] = pd.DatetimeIndex(as_of[keep])

    ok = panel["bid"].notna() & panel["ask"].notna() & (panel["ask"] >= panel["bid"])
    panel["mid"] = np.where(ok, (panel["bid"] + panel["ask"]) / 2.0, np.nan)
    panel["spread"] = np.where(ok, panel["ask"] - panel["bid"], np.nan)
    panel["date"] = panel["ts"].dt.date
    panel["expiry"] = pd.to_datetime(panel["expiry"]).dt.date
    return panel.sort_values(["ts", "expiry", "strike"]).reset_index(drop=True)


def trading_weeks(stock) -> list[dict]:
    """
    Weeks read off the underlying's own session calendar.

    Accepts either the panel this module builds or a bare DatetimeIndex,
    because the fetcher needs the same calendar before a panel exists and this
    logic must not be written twice. It was, briefly: scripts/fetch_hw2.py
    carried a second copy whose guard was spelled `entry == expiry` instead of
    `len(sess) < 2`. The two happened to agree, which is exactly how a
    duplicate survives long enough to drift.

    "Buy Monday, expire Friday" is a description, not a rule. Jun 19 2026
    (Juneteenth) and Jul 3 2026 (Jul 4 observed) are closed, and those weeks
    expire on the Thursday -- the Thursday RIC resolves against LSEG and the
    Friday one does not. Taking the first and last session the stock actually
    printed makes a holiday shift the cycle instead of deleting it.
    """
    if isinstance(stock, pd.DataFrame):
        days = sorted({d for d in stock["date"]})
    else:
        days = sorted({t.date() for t in pd.DatetimeIndex(stock)})
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


def settlement(stock: pd.DataFrame, day: dt.date) -> tuple[pd.Timestamp | None, float, str]:
    """
    (bar, price, source) for settling on `day`.

    The official close when the cache has it; otherwise the last trade in the
    session's final bar, and the source says so rather than passing one off as
    the other.
    """
    ts = _last_bar(stock, day)
    if ts is None:
        return None, float("nan"), "no bar"
    if "OFFICIAL_CLOSE" in stock.columns:
        oc = stock.at[ts, "OFFICIAL_CLOSE"]
        if pd.notna(oc) and np.isfinite(float(oc)):
            return ts, float(oc), "official close"
    return ts, float(stock.at[ts, "TRDPRC_1"]), "last trade"


def run_backtest(
    stock: pd.DataFrame,
    options: pd.DataFrame,
    weeks: list[dict],
    *,
    rule: str = "nearest_otm",
    order_hour: int = 16,
    start_cash: float = 50_000.0,
    contracts: int = 1,
    margin_rate: float = 0.0,
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
        occ = occ_symbol(parse_option_ric(r["ric"])["underlying"], expiry_day,
                         float(strike), "C")
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
            "ts": ts, "instrument": r["ric"], "occ": occ, "kind": "call", "side": SELL,
            "qty": -contracts, "limit": round(float(mid), 4), "fill": round(float(mid), 4),
            "cash_delta": qty * float(mid),
            "strike": float(strike), "expiry": expiry_day,
            "bid": float(r["bid"]), "ask": float(r["ask"]),
            "note": f"write: {rule}, nearest listed strike >= spot "
                    f"({spot:.2f}); fill at mid of {r['bid']:.2f}/{r['ask']:.2f}",
        })

        # ---- wait through expiry -------------------------------------------
        ets, s_t, settle_source = settlement(stock, expiry_day)
        if not np.isfinite(s_t):
            cycles.append({**w, "status": "no expiry print", "strike": float(strike)})
            continue

        itm = s_t > strike  # ties expire: an ATM call has no intrinsic value
        if itm:
            # Two rows, because two things happen and the assignment asks for
            # both: the short call is closed by assignment, and the shares are
            # delivered against it at the strike. Booking it as one row that
            # carried the cash was arithmetically right and still hid the
            # stock leg from anyone reading the blotter for share movements.
            blotter.append({
                "ts": ets, "instrument": r["ric"], "occ": occ, "kind": "call",
                "side": ASSIGN, "qty": contracts, "limit": None,
                "fill": float(strike), "cash_delta": 0.0,
                "strike": float(strike), "expiry": expiry_day,
                # The note points at the stock leg on purpose. Read alone,
                # an ASSIGN row moving $0 looks like the assignment proceeds
                # never arrived -- which is exactly how it was misread.
                "note": f"assigned: {settle_source} {s_t:.2f} > strike "
                        f"{strike:.2f}; short call closed by assignment. The "
                        f"{qty} shares are delivered on the next row, "
                        f"crediting {qty * float(strike):,.2f}",
            })
            blotter.append({
                "ts": ets, "instrument": stock.attrs.get("ric", "AAPL.O"),
                "kind": "stock", "side": SELL, "qty": qty, "limit": None,
                "fill": float(strike), "cash_delta": qty * float(strike),
                "note": f"delivered against assignment: {qty} shares at the "
                        f"strike {strike:.2f}, not the settle {s_t:.2f}; flat",
            })
            shares -= qty
        else:
            blotter.append({
                "ts": ets, "instrument": r["ric"], "occ": occ, "kind": "call",
                "side": EXPIRE,
                "qty": contracts, "limit": None, "fill": 0.0, "cash_delta": 0.0,
                "strike": float(strike), "expiry": expiry_day,
                "note": f"expired: {settle_source} {s_t:.2f} <= strike "
                        f"{strike:.2f}; keep shares, keep premium",
            })

        cycles.append({
            **w, "status": "assigned" if itm else "expired",
            "order_ts": str(ts), "spot": float(spot), "strike": float(strike),
            "mid": float(mid), "bid": float(r["bid"]), "ask": float(r["ask"]),
            "settle": s_t, "settle_source": settle_source,
            "premium": qty * float(mid),
            "otm_pct": 100.0 * (strike / spot - 1.0),
            "T_years": T,
            "atm_iv": atm_iv(chain, float(spot), T),
        })

    return {"blotter": blotter, "cycles": cycles,
            "rule": rule, "order_hour": order_hour, "start_cash": start_cash,
            "margin_rate": margin_rate}


# --------------------------------------------------------------------------
# ledger + Reg T
# --------------------------------------------------------------------------

MARGIN_DAY_COUNT = 360.0   # actual/360, the convention brokers use for margin loans


def margin_interest_schedule(stock: pd.DataFrame):
    """
    For each session's last bar, the calendar days until the next session's
    last bar. A debit balance standing at Friday's close accrues three days.
    """
    last_bar = {}
    for t in stock.index:
        last_bar[t.date()] = t
    dates = sorted(last_bar)
    days = {last_bar[d]: (n - d).days for d, n in zip(dates, dates[1:])}
    return days


def build_ledger(run: dict, stock: pd.DataFrame, options: pd.DataFrame) -> pd.DataFrame:
    """
    Replay the blotter across every hourly bar and mark the book.

    The ledger is DERIVED from the blotter rather than accumulated alongside
    it, so the two cannot drift. Cash only ever moves on a blotter event;
    marks move NAV and never touch cash.

    MARGIN INTEREST. When cash is negative the account is borrowing, and a
    broker charges for that. Interest accrues daily at each session's close on
    the debit balance, at run["margin_rate"] a year, actual/360, for the
    calendar days until the next close. It is carried as an ACCRUED LIABILITY
    that reduces NAV rather than as a cash movement: the assignment is explicit
    that cash moves only on blotter events (buy, premium, expire, assign), and
    a broker debits accrued interest periodically, not bar by bar. The default
    rate is zero, so a run that never borrows is unaffected.
    """
    blotter = sorted(run["blotter"], key=lambda b: (b["ts"], b["side"] != BUY))
    cash = float(run["start_cash"])
    shares = 0
    call = None            # dict(ric, strike, expiry) while short
    bi = 0

    # Fast lookup of a short call's mid at a given bar.
    mid_by = {(r.ric, r.ts): r.mid for r in options.itertuples()}
    # End-of-day bars mark the stock at the official close; see stock_marks.
    marks = stock_marks(stock)
    rate = float(run.get("margin_rate", 0.0) or 0.0)
    accrual_days = margin_interest_schedule(stock) if rate else {}
    accrued = 0.0

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
                # Shares leave on the stock SELL row that follows an ASSIGN,
                # not here. Decrementing in both places would take the book to
                # -100 shares on every assignment.
                call = None
                last_opt = np.nan
            bi += 1

        px = marks.at[ts]
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
        nav = cash + stock_mv + option_mv - accrued
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
            "accrued_interest": accrued,
        })

        # Accrue on the balance standing at this close; it reaches NAV from the
        # next bar on, so a close's own row reflects only earlier accruals.
        if rate and cash < 0 and ts in accrual_days:
            accrued += -cash * rate * accrual_days[ts] / MARGIN_DAY_COUNT

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
