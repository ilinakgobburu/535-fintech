"""
The two-sided variance trade, live: tonight's NVDA orders for the course server (Sanji).

The strategy is config/vol_arb_ml.yaml's (part_2 and its amendments), computed by lib/vol_arb.py and
lib/vol_arb_wf.py, which are byte-identical copies of the frozen backtest code (their sha256 is in
config/vol_arb_live.yaml and checked before anything runs). This module does no I/O: vol_arb_orders.py at the
repo root pulls LSEG and Cboe, calls these functions and writes mkt/open_orders.json.

WHAT IS KNOWN WHEN. The script runs after the close of session D (wheel_live.decision_dates) and writes orders
for the next session, `for`, which Sanji fills at that session's prices: a market order at the open, a limit
order against the bar's high and low.

ENTRY NIGHTS are the registered calendar: the first session after each monthly expiry (entry_sessions, the
backtest's wheel.entry_sessions, on wheel_live's NYSE calendar), the cycle running to the next monthly expiry.
On an entry night, from D's closing quotes of that expiry's NVDA chain:

    measurement   vol_arb.measure(): parity forward, ATM vol, the 1.5-sd corridor, the side variances, the
                  skip rule (no forward, no ATM, few strikes, short reach), the parabola, the flat baseline
    features      vol_arb.features(): the nine, from the trailing closes and dividends (vol_arb.realized),
                  Cboe's VIX close on D and the next report date (TR.ExpectedReportDate)
    the rung      config's `rung`, fit on the development panel (one row per measured 2019-23 stock-cycle,
                  exported from the development output; the 2024-26 final test is not used):
                      rule_1     put_rich when f1 > f2, call_rich otherwise (no fit)
                      logistic   vol_arb_wf.fit_predict on the panel's labelled rows; p > 0.55 put_rich,
                      forest     p < 0.45 call_rich, no trade between (vol_arb_wf.decide)
    the legs      vol_arb.legs_at_entry(): wheel.select()'s put and call nearest 0.25 |delta|, each
                  round(200 / vega per vol point per contract) contracts, at least 1
    the orders    put_rich: SHT the put, BUY the call; call_rich: BUY the put, SHT the call. Limits at the
                  closing mid, bare-number strike and price strings, the expiry as a date

A skipped cycle, a cycle whose nine features are not all finite, a leg missing, or a model's p inside the band
is no trade that cycle, as in the backtest.

EVERY NIGHT, THE HEDGE. The book's NVDA options are this strategy's (the wheel's schedule is retired). Their net
delta at D's close, each leg's from its own closing mid's implied vol (wheel.option_greeks, as the backtest; a
leg with no usable mid that close keeps its last IV, delta recomputed at D's spot and T), sets the target
shares, -round(net delta). A market order for the next open moves the book's shares to it, as the backtest
rebalances at each open from the previous close's deltas. One order cannot flip a long into a short, so
crossing zero is two orders, the close first. With no options held the target is 0: after an expiry the
leftover hedge (and any shares Sanji delivered) is closed. With no options and no shares, nothing.

The entry's legs are hedged from the night after they appear in the book, since a limit order may not fill.

A STALE BOOK stops the run, as in wheel_live: an order this script wrote for a session up to D with no result
yet once mkt/fills.json exists, a rejected order, or an option still listed on or after its expiry.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import vol_arb as VA
from . import vol_arb_wf as WF
from . import wheel as W
from . import wheel_live as L
from .pmcc_daily import monthly_expiries

PUT, CALL = W.PUT, W.CALL
CONTRACT = W.CONTRACT                    # shares per contract; NVDA has not split since 2024-06-10
RUNGS = WF.CANDIDATES                    # rule_1, logistic, forest: the rungs the development run may choose
PANEL_FORMAT = "vol_arb_dev_panel/1"
TAG = "va"                               # in every order id: nvda-va-put-2026-11-23
FALLBACK_SESSIONS = 30                   # how far back a leg's last usable mid may be
_NUMBER = re.compile(r"^\d+(\.\d+)?$")


class StaleBook(ValueError):
    """The book is not up to date with what has happened: no order can be planned from it."""


class PanelError(ValueError):
    """The development panel is missing, malformed, or not the one the config names."""


# --------------------------------------------------------------------------- the calendar
def entry_sessions(stock_index: pd.DatetimeIndex, start, end) -> dict:
    """
    The registered entries: {the first session after each monthly expiry: the next monthly expiry}.
    A verbatim copy of lib/wheel.py's entry_sessions on branch vol-arb-ml (the backtest's calendar),
    which the public lib/wheel.py does not have; wheel.run() builds its entries the same way.
    """
    sessions = stock_index[(stock_index >= pd.Timestamp(start)) & (stock_index <= pd.Timestamp(end))]
    expiries = monthly_expiries(sessions, start, end)
    return {sessions[sessions > prev][0]: exp for prev, exp in zip(expiries[:-1], expiries[1:])}


def _d(x) -> dt.date:
    return pd.Timestamp(x).date()


def entry_calendar(day) -> dict:
    """The registered entries around `day` on the NYSE calendar, {entry date: expiry date}."""
    day = _d(day)
    lo, hi = day - dt.timedelta(days=75), day + dt.timedelta(days=110)
    return {_d(e): _d(x) for e, x in entry_sessions(L.sessions(lo, hi), lo, hi).items()}


def entry_expiry(day) -> dt.date | None:
    """The expiry of the cycle entered on `day`, or None when `day` is not an entry session."""
    return entry_calendar(day).get(_d(day))


def next_entry(day) -> tuple[dt.date, dt.date]:
    """The first entry session on or after `day`, and its expiry."""
    cal = entry_calendar(day)
    e = min(k for k in cal if k >= _d(day))
    return e, cal[e]


def previous_expiry(expiry) -> dt.date:
    """The monthly expiry before `expiry`: the one whose cycle `expiry`'s follows."""
    expiry = _d(expiry)
    lo = expiry - dt.timedelta(days=70)
    ex = [_d(e) for e in monthly_expiries(L.sessions(lo, expiry), lo, expiry)]
    return max(e for e in ex if e < expiry)


def next_session(day) -> dt.date:
    """The first NYSE session after `day`."""
    day = _d(day)
    return min(_d(s) for s in L.sessions(day + dt.timedelta(days=1), day + dt.timedelta(days=14)))


def is_session(day) -> bool:
    day = _d(day)
    return day in {_d(s) for s in L.sessions(day, day)}


def night(day, force_entry: bool = False) -> dict:
    """
    What night `day` is: an entry night (its cycle's expiry) or not. `force_entry` treats a night that is not
    an entry as one, for a one-off dry run on the latest close: it measures the chain the next registered entry
    will trade (that entry's expiry).
    """
    exp = entry_expiry(day)
    if exp is not None:
        return {"entry": True, "expiry": exp, "forced": False}
    if force_entry:
        return {"entry": True, "expiry": next_entry(day)[1], "forced": True}
    return {"entry": False, "expiry": None, "forced": False}


# --------------------------------------------------------------------------- the market, from what was pulled
def mid_of(bid, ask) -> float:
    """load_market's mid: both sides there and ask >= bid, else NaN (two-sided also needs it above zero)."""
    b, a = W._num(bid), W._num(ask)
    return (b + a) / 2.0 if np.isfinite(b) and np.isfinite(a) and a >= b else np.nan


def quote_frame(rows) -> pd.DataFrame:
    """Option quotes in load_market's frame: date, ric, cp, strike, expiry, bid, ask, mid."""
    cols = ["date", "ric", "cp", "strike", "expiry", "bid", "ask", "mid"]
    q = pd.DataFrame(list(rows), columns=cols[:-1])
    q["date"] = pd.to_datetime(q["date"]).dt.normalize()
    q["expiry"] = pd.to_datetime(q["expiry"]).dt.normalize()
    q["strike"] = q["strike"].astype(float)
    q["bid"] = pd.to_numeric(q["bid"], errors="coerce").astype(float)
    q["ask"] = pd.to_numeric(q["ask"], errors="coerce").astype(float)
    q["mid"] = [mid_of(b, a) for b, a in zip(q["bid"], q["ask"])]
    return q[cols]


def market(closes: pd.Series, opens: pd.Series | None, tbill_pct: pd.Series, dividends: pd.Series,
           earnings=(), quotes: pd.DataFrame | None = None) -> dict:
    """
    load_market's dict from the live pulls (NVDA in today's shares; no split since 2024-06-10): the stock's
    closes and opens, the T-bill in decimal carried forward onto the sessions, dividends by ex-date (amounts
    above zero), the report dates, the option quotes, and a lot of 100 shares.
    """
    c = pd.to_numeric(pd.Series(closes), errors="coerce").astype(float)
    c.index = pd.DatetimeIndex(c.index).normalize()
    o = (pd.to_numeric(pd.Series(opens), errors="coerce").astype(float) if opens is not None
         else pd.Series(np.nan, index=c.index))
    o.index = pd.DatetimeIndex(o.index).normalize()
    stock = pd.DataFrame({"close": c, "open": o.reindex(c.index)})
    stock = stock[~stock.index.duplicated(keep="first")].dropna(subset=["close"]).sort_index()
    tb = pd.to_numeric(pd.Series(tbill_pct), errors="coerce").astype(float) / 100.0
    tb.index = pd.DatetimeIndex(tb.index).normalize()
    tb = tb[~tb.index.duplicated(keep="first")].dropna().sort_index()
    div = pd.to_numeric(pd.Series(dividends, dtype=float), errors="coerce")
    div.index = pd.DatetimeIndex(div.index).normalize()
    div = div[div > 0].sort_index()
    return {"stock": stock,
            "tbill": tb.reindex(stock.index.union(tb.index)).ffill().reindex(stock.index),
            "dividends": div.astype(float),
            "earnings": pd.DatetimeIndex(pd.to_datetime(list(earnings))).normalize().sort_values(),
            "quotes": quotes if quotes is not None else quote_frame([]),
            "lot": pd.Series(float(CONTRACT), index=stock.index)}


def strike_plan(closes: pd.Series, prev_expiry, day, bands: dict, grid: list) -> dict:
    """
    The strikes to ask LSEG for, as scripts/fetch_wheel_raw.py's strikes() banded the backtest's chains:
    puts 0.70-1.02 and calls 0.97-1.40 times the closes over the 7 days either side of the previous monthly
    expiry (here only up to D, the later ones are not known yet), on the grid tier of their mean price
    (NVDA_2024's: $1 below $150, $2.50 to $300, $10 above). {cp: [strikes]}.
    """
    prev, day = pd.Timestamp(prev_expiry), pd.Timestamp(day)
    lo, hi = min(prev, day) - pd.Timedelta(days=7), min(prev + pd.Timedelta(days=7), day)
    px = pd.Series(closes, dtype=float)
    px.index = pd.DatetimeIndex(px.index).normalize()
    w = px[(px.index >= lo) & (px.index <= hi)].dropna()
    if w.empty:
        raise ValueError(f"no closes between {lo.date()} and {hi.date()} to band the chain on")
    mean = float(w.mean())
    step = next(float(s) for bound, s in grid if bound is None or mean < float(bound))
    out = {}
    for cp in (PUT, CALL):
        b_lo, b_hi = (float(x) for x in bands[cp])
        k_lo = math.floor(float(w.min()) * b_lo / step) * step
        k_hi = math.ceil(float(w.max()) * b_hi / step) * step
        out[cp] = [round(float(k), 4) for k in np.arange(k_lo, k_hi + step / 2, step)]
    return out


def parse_vix(text: str) -> pd.Series:
    """Cboe's VIX_History.csv (DATE as MM/DD/YYYY, CLOSE), read as the backtest's runner reads it."""
    df = pd.read_csv(io.StringIO(text))
    s = pd.Series(pd.to_numeric(df["CLOSE"], errors="coerce").to_numpy(float),
                  index=pd.to_datetime(df["DATE"], format="%m/%d/%Y")).dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()


def earnings_dates(expected, actual, day, expiry, min_gap_days: int) -> tuple[list, str]:
    """
    The report dates f7 reads, live (vol_arb_ml.yaml live_script_notes: TR.ExpectedReportDate for the next
    report): the expected date and the actual ones. When the expected date is not after D (LSEG has not rolled
    it forward since the last report), a report inside the cycle is still ruled out if the last actual report
    was so recent that the next cannot come before the expiry (`min_gap_days`, a quarterly reporter's least gap);
    otherwise the flag cannot be set and this raises.
    """
    day, expiry = pd.Timestamp(day), pd.Timestamp(expiry)
    exp = sorted({pd.Timestamp(x).normalize() for x in expected if pd.notna(x)})
    act = sorted({pd.Timestamp(x).normalize() for x in actual if pd.notna(x)})
    if any(e > day for e in exp):
        return exp + act, f"next report expected {min(e for e in exp if e > day).date()}"
    last = max([a for a in act if a <= day], default=None)
    if last is not None and last + pd.Timedelta(days=min_gap_days) > expiry:
        return exp + act, (f"no expected report date after {day.date()}; the last report ({last.date()}) was less "
                           f"than {min_gap_days} days before the expiry, so none falls in the cycle")
    raise ValueError(f"TR.ExpectedReportDate gives no date after {day.date()} and the last actual report "
                     f"({last.date() if last is not None else 'none'}) does not rule one out before "
                     f"{expiry.date()}: the earnings feature cannot be set")


# --------------------------------------------------------------------------- the development panel
def export_panel(dev: dict, *, source: str, sha256: str) -> dict:
    """
    The development panel file (data/vol_arb_dev_panel.json) from the development output
    (cache/vol_arb_ml_dev.json): one row per measured 2019-23 stock-cycle, vol_arb_wf.panel()'s id, stock, entry,
    expiry, nine features and label (null with no trade, a leg missing: never trained on), with where it came
    from and the choice the development run recorded for the test.
    """
    if dev.get("stage") != "dev":
        raise PanelError("not a development output (stage is not 'dev')")
    if not dev.get("choice_for_test", {}).get("chosen"):
        raise PanelError("the development output records no choice for the test")
    df = WF.panel(dev.get("cycles") or [], predict=False)
    if df.empty:
        raise PanelError("the development output has no measured cycle")
    rows = []
    for r in df.to_dict("records"):
        rows.append({"id": r["id"], "stock": r["stock"], "entry": str(r["entry"].date()),
                     "expiry": str(r["expiry"].date()), **{f: float(r[f]) for f in VA.FEATURES},
                     "label": None if pd.isna(r["label"]) else int(r["label"])})
    return {"format": PANEL_FORMAT,
            "made_from": {"file": source, "sha256": sha256, "commit": dev.get("commit"),
                          "config": dev.get("config"), "registered": dev.get("registered")},
            "environment": dev.get("environment"), "stocks": dev.get("stocks"), "window": dev.get("window"),
            "choice_for_test": dev["choice_for_test"], "features": list(VA.FEATURES), "rows": rows}


def _finite_number(x) -> bool:
    return type(x) in (int, float) and math.isfinite(x)


def load_panel(obj, cfg: dict) -> pd.DataFrame:
    """
    The panel as vol_arb_wf.panel() frames it (id, stock, entry, expiry, the nine features, label, predict
    False), after checking it is the format above, made from development rows only (the stocks and window of
    config's `development`), every feature a finite number and every label 0, 1 or null, and that it records
    the same rung as config's `rung`.
    """
    if not isinstance(obj, dict) or obj.get("format") != PANEL_FORMAT:
        raise PanelError(f"not a {PANEL_FORMAT} file")
    if obj.get("features") != list(VA.FEATURES):
        raise PanelError("its features are not the nine registered ones, in order")
    chosen = (obj.get("choice_for_test") or {}).get("chosen")
    if chosen != cfg["rung"]:
        raise PanelError(f"the development run chose {chosen!r} but config/vol_arb_live.yaml trades "
                         f"{cfg['rung']!r}")
    dev = cfg["development"]
    stocks, (lo, hi) = set(dev["stocks"]), (pd.Timestamp(str(d)) for d in dev["window"])
    rows = obj.get("rows")
    if not isinstance(rows, list) or not rows:
        raise PanelError("no rows")
    out = []
    for r in rows:
        if not isinstance(r, dict):
            raise PanelError(f"a row is not an object: {r!r}")
        try:
            entry, expiry = pd.Timestamp(r["entry"]), pd.Timestamp(r["expiry"])
        except (KeyError, ValueError, TypeError) as exc:
            raise PanelError(f"a row without dated entry and expiry: {r.get('id')}") from exc
        if r.get("stock") not in stocks:
            raise PanelError(f"{r.get('id')}: {r.get('stock')!r} is not a development stock")
        if not (lo <= entry <= hi) or not expiry > entry:
            raise PanelError(f"{r.get('id')}: entry {entry.date()} is outside the development window "
                             f"or not before its expiry")
        if r.get("id") != WF.row_id(r["stock"], entry):
            raise PanelError(f"{r.get('id')!r} is not the row id of {r['stock']} {entry.date()}")
        bad = [f for f in VA.FEATURES if not _finite_number(r.get(f))]
        if bad:
            raise PanelError(f"{r['id']}: {', '.join(bad)} not a finite number")
        if r.get("label", "missing") not in (0, 1, None) or type(r.get("label")) is bool:
            raise PanelError(f"{r['id']}: label {r.get('label', 'missing')!r} is not 0, 1 or null")
        out.append({"id": r["id"], "stock": r["stock"], "entry": entry, "expiry": expiry,
                    **{f: float(r[f]) for f in VA.FEATURES},
                    "label": np.nan if r["label"] is None else float(r["label"]), "predict": False})
    df = pd.DataFrame(out, columns=["id", "stock", "entry", "expiry", *VA.FEATURES, "label", "predict"])
    if df["id"].duplicated().any():
        raise PanelError("a row id repeats")
    if df["label"].notna().sum() == 0:
        raise PanelError("no labelled row to fit on")
    return df.sort_values(["entry", "stock"]).reset_index(drop=True)


def decide(rung: str, features: dict, panel: pd.DataFrame, day) -> dict:
    """
    The configured rung on tonight's row: rule 1 by its rule; a model fit (vol_arb_wf.fit_predict, the
    registered settings) on the panel's rows finished before D that have a label, in entry then stock order as
    the walk-forward fits, and its p turned into a direction by the registered band (vol_arb_wf.decide). The
    panel must hold the walk-forward's 24 entry months of finished rows (vol_arb_wf.warm_up_start).
    """
    if rung not in RUNGS:
        raise ValueError(f"rung must be one of {RUNGS}, not {rung!r}")
    day = pd.Timestamp(day)
    if WF.warm_up_start(panel, [day]) is None:
        raise PanelError(f"the panel has fewer than {WF.WARM_MONTHS} entry months of finished rows")
    row = pd.DataFrame([{"id": "live", **{f: float(features[f]) for f in VA.FEATURES}}])
    if rung == "rule_1":
        return {"rung": rung, "direction": WF.rule_1(row)["live"], "p": None, "train_rows": 0}
    train = WF.finished(panel, day)
    train = train[train["label"].notna()].sort_values(["entry", "stock"])
    p = float(WF.fit_predict(rung, train[WF.FEATURES].to_numpy(), train["label"].to_numpy(),
                             row[WF.FEATURES].to_numpy())[0])
    return {"rung": rung, "direction": WF.decide(p), "p": p, "train_rows": int(len(train))}


# --------------------------------------------------------------------------- the entry's measurement
def measure_entry(mkt: dict, day, expiry, vix_close: float) -> dict:
    """
    vol_arb.stock_cycles()'s steps for one cycle up to its legs, on D's closing quotes of `expiry`: the status
    (measured, a skip reason, or features_not_finite), the measurement, the nine features, the legs, and
    `no_trade` (the reason, or None when both legs are there to trade).
    """
    entry, expiry = pd.Timestamp(day), pd.Timestamp(expiry)
    quotes = mkt["quotes"]
    q_exp = quotes[quotes["expiry"] == expiry]
    day_q = q_exp[q_exp["date"] == entry]
    spot = float(mkt["stock"].at[entry, "close"])
    r, q = W.carry(mkt, entry)
    T = (expiry - entry).days / VA.DAYS_PER_YEAR
    m = VA.measure(VA.two_sided(day_q), spot, r, q, T)
    rec = {"entry": entry, "expiry": expiry, "status": m["status"], "measure": VA._scalars(m), "features": None,
           "raw_premiums": None, "legs": None, "no_trade": None, "vix_close": float(vix_close)}
    if m["status"] != "measured":
        rec["no_trade"] = m["status"]
        return rec
    real = VA.realized(mkt["stock"]["close"], mkt["dividends"])
    f = VA.features(m, real.loc[entry], float(vix_close), VA.earnings_between(mkt["earnings"], entry, expiry))
    prem = VA.premiums(m, real.loc[entry])
    rec["features"], rec["raw_premiums"] = f, {k: prem[k] for k in VA.RAW}
    if not all(np.isfinite(list(f.values()))):
        rec["status"] = rec["no_trade"] = VA.NOT_FINITE
        return rec
    if not np.isfinite(m["flat_baseline"]):
        raise ValueError(f"{entry.date()}: the nine features are finite but the flat baseline is not")
    legs = VA.legs_at_entry(mkt, day_q, entry, expiry)
    rec["legs"] = legs
    if legs[PUT] is None or legs[CALL] is None:
        rec["no_trade"] = VA.LEG_MISSING
    return rec


# --------------------------------------------------------------------------- the book
@dataclass
class Book:
    cash: float | None = None
    shares: int = 0
    options: list = field(default_factory=list)     # {"cp", "expiry" (date), "strike", "qty"}, qty != 0


def _strike(x) -> float | None:
    if type(x) in (int, float) and math.isfinite(x) and x > 0:
        return float(x)
    if isinstance(x, str) and _NUMBER.match(x) and float(x) > 0:
        return float(x)
    return None


def read_book(positions: dict | None, fills: list | None, conid: int) -> Book:
    """
    The NVDA book from mkt/positions.json, in the shape wheel_live.read_book reads (its format is not published
    yet, so anything else stops the script rather than being guessed at), with short shares allowed (the hedge
    shorts) and each option's strike required (a number, or a bare-number string as the orders write it):

        {"cash": 98765.43,
         "positions": [
           {"underlying": 4815747, "contract": "underlying", "qty": -120},
           {"underlying": 4815747, "contract": "put", "expiry": "2026-12-18", "strike": 165, "qty": -10}]}

    With no positions file (Sanji does not write one yet) the book is empty.
    """
    if fills is not None and not isinstance(fills, list):
        raise L.BookFormatError("mkt/fills.json should be a JSON list")
    filled = any(isinstance(f, dict) and str(f.get("status", "")).lower() == "filled" for f in fills or [])
    if positions is None:
        if filled:
            raise L.BookFormatError("mkt/fills.json records a fill but there is no mkt/positions.json")
        return Book()
    if not isinstance(positions, dict) or "cash" not in positions or not isinstance(positions.get("positions"), list):
        raise L.BookFormatError('mkt/positions.json should look like {"cash": ..., "positions": [...]}')
    book = Book(cash=float(positions["cash"]))
    for p in positions["positions"]:
        if not isinstance(p, dict) or type(p.get("underlying")) is not int:
            raise L.BookFormatError(f"a position without an integer conid in 'underlying': {p}")
        if p["underlying"] != conid:
            continue
        kind, qty = p.get("contract"), p.get("qty")
        if type(qty) is not int:
            raise L.BookFormatError(f"position without an integer qty: {p}")
        if kind == "underlying":
            book.shares += qty
        elif kind in ("put", "call"):
            if not L._DATE.match(str(p.get("expiry", ""))):
                raise L.BookFormatError(f"an option position without a YYYY-MM-DD expiry: {p}")
            k = _strike(p.get("strike"))
            if k is None:
                raise L.BookFormatError(f"an option position without a positive strike: {p}")
            if qty != 0:
                book.options.append({"cp": PUT if kind == "put" else CALL,
                                     "expiry": dt.date.fromisoformat(p["expiry"]), "strike": k, "qty": qty})
        else:
            raise L.BookFormatError(f"unknown contract {kind!r} in {p}")
    return book


def expired_legs(book: Book, day) -> list[dict]:
    """Options still in the book whose expiry is on or before D: Sanji has not applied the expiry yet."""
    return [o for o in book.options if o["expiry"] <= _d(day)]


def held_contracts(book: Book) -> dict:
    """{(cp, expiry, strike): net contracts}, one entry per contract, zeros dropped."""
    out = {}
    for o in book.options:
        key = (o["cp"], o["expiry"], float(o["strike"]))
        out[key] = out.get(key, 0) + int(o["qty"])
    return {k: v for k, v in out.items() if v != 0}


# --------------------------------------------------------------------------- the hedge
def leg_delta(mkt: dict, cp: str, expiry, strike: float, history: pd.DataFrame, day) -> dict:
    """
    One held leg's delta per share at D's close. Its IV is its own mid's (load_market's mid, above zero) as
    wheel.option_greeks computes it at that close's spot, T, r and q: D's, or when D's quote gives none, the
    latest earlier close's within the history (the backtest's "a leg with no mid keeps its last IV"). The delta
    is vol_arb.greeks_at() from that IV at D's spot and T, as the backtest's hedge uses it.
    """
    day, expiry = pd.Timestamp(day), pd.Timestamp(expiry)
    h = history.copy() if history is not None else pd.DataFrame(columns=["bid", "ask"])
    h.index = pd.DatetimeIndex(h.index).normalize()
    h = h[~h.index.duplicated(keep="first")]
    h = h[(h.index <= day)].sort_index().tail(FALLBACK_SESSIONS)
    stock = mkt["stock"]
    iv, iv_day, mid_today = np.nan, None, np.nan
    for d in reversed(h.index):
        mid = mid_of(h.at[d, "bid"], h.at[d, "ask"])
        if d == day:
            mid_today = mid
        if d not in stock.index or not (np.isfinite(mid) and mid > 0):
            continue
        r, q = W.carry(mkt, d)
        own = W.option_greeks(mid, float(stock.at[d, "close"]), float(strike), (expiry - d).days / VA.DAYS_PER_YEAR,
                              r, q, cp)["iv"]
        if np.isfinite(own):
            iv, iv_day = float(own), d
            break
    r, q = W.carry(mkt, day)
    g = VA.greeks_at(float(stock.at[day, "close"]), float(strike), (expiry - day).days / VA.DAYS_PER_YEAR,
                     iv, r, q, cp)
    return {"iv": iv, "iv_day": iv_day, "mid": mid_today, "delta": float(g["delta"])}


def transition(held: int, target: int) -> list[tuple[str, int]]:
    """
    The orders that take a position from `held` to `target` (shares or contracts, negative short): BUY opens or
    adds to a long, SEL sells from a long, SHT opens or adds to a short, BTC buys a short back. One order never
    crosses zero, so long to short (or short to long) is two orders, the close first.
    """
    held, target = int(held), int(target)
    if target == held:
        return []
    if held > 0 and target < held:
        out = [("SEL", held - max(target, 0))]
        return out + ([("SHT", -target)] if target < 0 else [])
    if held < 0 and target > held:
        out = [("BTC", min(target, 0) - held)]
        return out + ([("BUY", target)] if target > 0 else [])
    if target > held:
        return [("BUY", target - held)]
    return [("SHT", held - target)]


# --------------------------------------------------------------------------- the orders
def _id(root: str, what: str, action: str | None, day) -> str:
    return f"{root.lower()}-{TAG}-{what}{'-' + action.lower() if action else ''}-{_d(day).isoformat()}"


def stock_orders(*, root: str, conid: int, day, for_day, held: int, target: int) -> list[dict]:
    """Market orders for the next open that move the shares from `held` to `target`; ids nvda-va-hedge-<D>."""
    steps = transition(held, target)
    return [{"id": _id(root, "hedge", a if len(steps) > 1 else None, day), "for": _d(for_day).isoformat(),
             "order_type": "mkt", "action": a, "qty": int(n), "underlying": int(conid), "contract": "underlying",
             "ref": "open"} for a, n in steps]


def option_orders(*, root: str, conid: int, day, for_day, cp: str, expiry, strike: float, limit: float,
                  held: int, target: int) -> list[dict]:
    """Limit orders at `limit` that move one contract from `held` to `target`; ids nvda-va-put-<D>."""
    steps = transition(held, target)
    what = "put" if cp == PUT else "call"
    return [{"id": _id(root, what, a if len(steps) > 1 else None, day), "for": _d(for_day).isoformat(),
             "order_type": "lmt", "action": a, "qty": int(n), "underlying": int(conid), "contract": what,
             "expiry": _d(expiry).isoformat(), "strike": L.number_token(strike), "ref": "open",
             "lmt_prc": L.number_token(limit)} for a, n in steps]


def check_orders(orders: list[dict]) -> None:
    """
    wheel_live.validate_order on each (order.html's checks), every strike and limit a string token as in
    order.html's examples, and the ids unique.
    """
    for o in orders:
        L.validate_order(o)
        for k in ("strike", "lmt_prc"):
            if k in o and not isinstance(o[k], str):
                raise ValueError(f"{o['id']}: {k} must be a string token, not {o[k]!r}")
    ids = [o["id"] for o in orders]
    if len(set(ids)) != len(ids):
        raise ValueError(f"order ids repeat: {ids}")


# --------------------------------------------------------------------------- the night
def plan_night(cfg: dict, day, for_day, book: Book, *, entry_expiry=None, mkt: dict | None = None,
               leg_history: dict | None = None, vix_close: float = np.nan, panel: pd.DataFrame | None = None) -> dict:
    """
    Tonight's orders. The hedge, every night: target shares -round(net delta of the held options at D's close),
    0 with none held. On an entry night (`entry_expiry` set): the measurement, the features, the rung's
    decision and, unless it is no trade, the two option orders. `leg_history` maps each held contract
    (cp, expiry, strike) to its daily bid/ask.
    """
    u = cfg["underlying"]
    root, conid = str(u["root"]), int(u["conid"])
    day, for_day = _d(day), _d(for_day)
    late = expired_legs(book, day)
    if late:
        raise StaleBook(f"the book still lists options that expired by {day}: {late}")
    notes, held = [], held_contracts(book)

    legs = []
    for (cp, expiry, strike), qty in sorted(held.items()):
        if mkt is None:
            raise ValueError("options are held but no market data was given")
        d = leg_delta(mkt, cp, expiry, strike, (leg_history or {}).get((cp, expiry, strike)), day)
        if not np.isfinite(d["delta"]):
            raise ValueError(f"no usable quote for the {expiry} {strike:g} {cp} in the last {FALLBACK_SESSIONS} "
                             f"sessions, so its delta (and the hedge) is unknown")
        legs.append({"cp": cp, "expiry": expiry, "strike": strike, "qty": qty, **d})
    net = float(sum(leg["qty"] * CONTRACT * leg["delta"] for leg in legs))
    target = -int(round(net)) if legs else 0
    stock = stock_orders(root=root, conid=conid, day=day, for_day=for_day, held=book.shares, target=target)
    if legs:
        notes.append(f"net option delta {net:+.1f} shares: hedge to {target} shares (holding {book.shares})")
    elif book.shares:
        notes.append(f"no options held: close the {book.shares} leftover shares")

    entry, decision, options = None, None, []
    if entry_expiry is not None:
        if mkt is None:
            raise ValueError("an entry night needs the market data")
        entry = measure_entry(mkt, day, entry_expiry, vix_close)
        if entry["no_trade"] is not None:
            notes.append(f"entry {day} -> {_d(entry_expiry)}: no trade ({entry['no_trade']})")
        else:
            if panel is None:
                raise PanelError("an entry night needs the development panel")
            decision = decide(cfg["rung"], entry["features"], panel, day)
            if decision["direction"] is None:
                notes.append(f"entry {day} -> {_d(entry_expiry)}: no trade ({decision['rung']}'s p = "
                             f"{decision['p']:.3f}, inside the band)")
            else:
                signs = VA.DIRECTIONS[decision["direction"]]
                for cp in (PUT, CALL):
                    leg = entry["legs"][cp]
                    have = held.get((cp, _d(entry_expiry), float(leg["strike"])), 0)
                    options += option_orders(root=root, conid=conid, day=day, for_day=for_day, cp=cp,
                                             expiry=entry_expiry, strike=leg["strike"], limit=leg["mid"],
                                             held=have, target=have + signs[cp] * int(leg["contracts"]))
                notes.append(f"entry {day} -> {_d(entry_expiry)}: {decision['direction']} ({decision['rung']})")
    orders = options + stock
    check_orders(orders)
    if not orders:
        notes.append("no order tonight")
    return {"orders": orders, "notes": notes, "entry": entry, "decision": decision,
            "hedge": {"legs": legs, "net_delta": net, "target": target, "shares": book.shares}}


# --------------------------------------------------------------------------- the config
def check_config(cfg: dict) -> None:
    """config/vol_arb_live.yaml's registered numbers must be the frozen library's, and the rung a candidate."""
    if cfg.get("rung") not in RUNGS:
        raise ValueError(f"rung must be one of {RUNGS}, not {cfg.get('rung')!r}")
    u = cfg.get("underlying") or {}
    if type(u.get("conid")) is not int or not u.get("root") or not u.get("ric"):
        raise ValueError("underlying needs root, ric and an integer conid")
    reg = cfg.get("registered") or {}
    want = {"corridor_sd": VA.CORRIDOR_SD, "target_delta": VA.TARGET_DELTA, "vega_dollars": VA.VEGA_DOLLARS,
            "capital": VA.CAPITAL, "band_put_rich_above": WF.P_HI, "band_call_rich_below": WF.P_LO,
            "warm_months": WF.WARM_MONTHS, "lot": CONTRACT}
    bad = [f"{k} is {reg.get(k)!r}, the library's is {v!r}" for k, v in want.items()
           if not _finite_number(reg.get(k)) or float(reg[k]) != float(v)]
    if bad:
        raise ValueError("config/vol_arb_live.yaml disagrees with lib/vol_arb*.py: " + "; ".join(bad))


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
