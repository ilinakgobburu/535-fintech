"""
The two-sided variance trade, live: tonight's NVDA orders for the course server (Sanji).

The strategy is config/vol_arb_ml.yaml's (part_2 and its amendments), computed by lib/vol_arb.py and
lib/vol_arb_wf.py, which are byte-identical copies of the frozen backtest code (their sha256 is in
config/vol_arb_live.yaml and checked before anything runs, as is each lib/wheel.py, lib/vol.py and lib/pmcc_daily.py
function they use: dependency_hashes). This module does no I/O: vol_arb_orders.py at the
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
                  exported from the development output and pinned by panel_pin's sha256; the 2024-26 final test
                  is not used):
                      rule_1     put_rich when f1 > f2, call_rich otherwise (no fit)
                      logistic   vol_arb_wf.fit_predict on the panel's labelled rows; p > 0.55 put_rich,
                      forest     p < 0.45 call_rich, no trade between (vol_arb_wf.decide)
    the legs      vol_arb.legs_at_entry(): wheel.select()'s put and call nearest 0.25 |delta|, each
                  round(200 / vega per vol point per contract) contracts, at least 1
    the orders    put_rich: SHT the put, BUY the call; call_rich: BUY the put, SHT the call. Limits at the
                  closing mid, bare-number strike and price strings, the expiry as a date

A skipped cycle, a cycle whose nine features are not all finite, a leg missing, or a model's p inside the band
is no trade that cycle, as in the backtest.

WHOSE BOOK. Every NVDA option in the book must be exactly what this strategy's filled orders add up to (its ids in
mkt/fills.json, looked up in data/vol_arb_ledger.json, which keeps every order it wrote), a short leg excepted when
Sanji has assigned it; NVDA shares need a fill of its own. Anything else (the wheel's last put filled before the
switch, a book read with contracts in shares) stops the run: book_problems.

EVERY NIGHT, THE HEDGE. The book's NVDA options are this strategy's (checked as above). Their net
delta at D's close, each leg's from its own closing mid's implied vol (wheel.option_greeks, as the backtest; a
leg with no usable mid that close keeps its last IV, delta recomputed at D's spot and T), sets the target
shares, -round(net delta). A market order for the next open moves the book's shares to it, as the backtest
rebalances at each open from the previous close's deltas. One order cannot flip a long into a short, so
crossing zero is two orders, the close first. With no options held the target is 0: after an expiry the
leftover hedge (and any shares Sanji delivered) is closed. With no options and no shares, nothing. A target or a
holding worth more than limits.max_hedge_x_capital times the book stops the run (hedge_limit).

The entry's legs are hedged from the night after they appear in the book, since a limit order may not fill.

A STALE BOOK stops the run, as in wheel_live: an order this script wrote for a session up to D with no result
yet once mkt/fills.json exists, a rejected order, or an option still listed on or after its expiry. Only this
strategy's own orders count (guard_problems); one she has looked at and listed in config's `acknowledged` is noted
instead, so the next run goes on and rewrites mkt/open_orders.json without it.
"""

from __future__ import annotations

import ast
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


def night(day, force_entry: bool = False, for_day=None) -> dict:
    """
    What night `day` is: an entry night (its cycle's expiry) or not. `force_entry` treats a night that is not
    an entry as one, for a one-off dry run on the latest close: it measures the chain the next registered entry
    will trade (that entry's expiry). An entry night whose orders would be for a later session than the one after
    D (`for_day`: a run after the next open, such as a Run button pressed Tuesday morning) is no entry, `late`:
    the backtest enters on the entry session's close, filled the next session, and two-day-old quotes are not that.
    """
    exp = entry_expiry(day)
    if exp is not None and for_day is not None and _d(for_day) != next_session(day):
        return {"entry": False, "expiry": exp, "forced": False, "late": True}
    if exp is not None:
        return {"entry": True, "expiry": exp, "forced": False, "late": False}
    if force_entry:
        return {"entry": True, "expiry": next_entry(day)[1], "forced": True, "late": False}
    return {"entry": False, "expiry": None, "forced": False, "late": False}


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


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def panel_pin_problems(raw: bytes, obj: dict, cfg: dict) -> list[str]:
    """
    Why the panel file is not the one config/vol_arb_live.yaml pins (empty when it is): panel_pin's sha256 of the
    file's bytes, and the development output's sha256 and commit the panel says it was made from (made_from). An
    empty pin (before the development run is exported) is a problem too: the live model may only be fit on the
    exported panel, recorded once.
    """
    pin = cfg.get("panel_pin") or {}
    if not pin.get("sha256"):
        return ["config/vol_arb_live.yaml pins no panel yet (panel_pin.sha256 is empty): set panel_pin from "
                "export_vol_arb_panel.py's output"]
    if not _HEX64.match(str(pin["sha256"])):
        return [f"panel_pin.sha256 {pin['sha256']!r} is not a sha256"]
    out = []
    got = sha256_hex(raw)
    if got != pin["sha256"]:
        out.append(f"its sha256 is {got}, panel_pin records {pin['sha256']}")
    made = (obj.get("made_from") or {}) if isinstance(obj, dict) else {}
    for key, name in (("dev_output_sha256", "sha256"), ("dev_output_commit", "commit")):
        if made.get(name) != pin.get(key):
            out.append(f"it was made from a development output with {name} {made.get(name)!r}, panel_pin records "
                       f"{pin.get(key)!r}")
    return out


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


# --------------------------------------------------------------------------- whose positions they are
LEDGER_FORMAT = "vol_arb_ledger/1"
SIGNS = {"BUY": 1, "BTC": 1, "SEL": -1, "SHT": -1}      # what one filled order does to the position


class LedgerError(ValueError):
    """data/vol_arb_ledger.json is malformed, or a fill of this strategy's is not in it."""


def own_prefix(root: str) -> str:
    """Every id this strategy writes starts with it: 'nvda-va-'."""
    return f"{root.lower()}-{TAG}-"


def read_ledger(obj) -> dict:
    """
    data/vol_arb_ledger.json, which vol_arb_orders.py keeps beside mkt/open_orders.json (Sanji's file keeps only the
    coming orders, so what this strategy wrote earlier, and so what its fills were, is kept here):

        {"format": "vol_arb_ledger/1",
         "orders": {"nvda-va-put-2026-11-23": {the order as written}, ...},
         "nights": {"2026-11-23": {"for": "2026-11-24", "ids": ["nvda-va-put-2026-11-23", ...]}}}

    None (no file yet) is an empty ledger.
    """
    if obj is None:
        return {"format": LEDGER_FORMAT, "orders": {}, "nights": {}}
    if (not isinstance(obj, dict) or obj.get("format") != LEDGER_FORMAT or not isinstance(obj.get("orders"), dict)
            or not isinstance(obj.get("nights"), dict)):
        raise LedgerError(f"data/vol_arb_ledger.json is not a {LEDGER_FORMAT} file")
    return obj


def record_night(ledger: dict, new: list[dict], day, for_day) -> dict:
    """The ledger after tonight's orders were written: each by id (a rerun's version replaces the earlier one)."""
    out = {"format": LEDGER_FORMAT, "orders": dict(ledger["orders"]), "nights": dict(ledger["nights"])}
    for o in new:
        out["orders"][o["id"]] = o
    out["nights"][_d(day).isoformat()] = {"for": _d(for_day).isoformat(), "ids": [o["id"] for o in new]}
    return out


def decided(ledger: dict, day, for_day) -> str | None:
    """Whether the ledger records a run that wrote the orders for `for_day` from D's close: its ids, or None."""
    night = ledger["nights"].get(_d(day).isoformat())
    if night is None or night.get("for") != _d(for_day).isoformat():
        return None
    return ", ".join(night.get("ids") or []) or "no order"


def strategy_options(ledger: dict, fills: list | None, root: str, day) -> tuple[dict, bool]:
    """
    ({(cp, expiry, strike): contracts}, any fill): what this strategy's filled option orders (its ids in
    mkt/fills.json with status filled, looked up in the ledger) add up to for the expiries after D, and whether any
    of its orders, a hedge included, has filled. Sanji fills all or nothing, so a filled order moved its full qty.
    """
    prefix = own_prefix(root)
    own = sorted({f.get("id") for f in fills or [] if isinstance(f, dict) and isinstance(f.get("id"), str)
                  and f["id"].startswith(prefix) and str(f.get("status", "")).lower() == "filled"})
    unknown = [i for i in own if i not in ledger["orders"]]
    if unknown:
        raise LedgerError(f"mkt/fills.json records {', '.join(unknown)} filled, but data/vol_arb_ledger.json has no "
                          f"such order, so what it bought or sold is not known")
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
    (why the run must stop, notes), wheel_live's guards on mkt/open_orders.json as it stands: an order Sanji
    rejected, or one written for a session up to D with no result once mkt/fills.json exists. Only this
    strategy's orders stop the run; another's (the wheel's last put) is noted. An id in config's `acknowledged`
    (her reason, once she has looked at it) is noted instead of stopping the run, so a rejected or unanswered order
    can't keep every later night red: the run that goes on rewrites the file without it.
    """
    prefix, stops, notes = own_prefix(root), [], []
    for f in L.rejected_orders(written, fills):
        i = f.get("id")
        if not str(i).startswith(prefix):
            notes.append(f"Sanji rejected {i}, which is not this strategy's: {f}")
        elif i in acknowledged:
            notes.append(f"Sanji rejected {i}; acknowledged in config/vol_arb_live.yaml: {acknowledged[i]}")
        else:
            stops.append(f"Sanji rejected {i}: {f}. Fix the order format; to trade on, add {i} to "
                         f"config/vol_arb_live.yaml's acknowledged with the reason")
    waiting = L.unresolved_orders(written, fills, _d(day))
    for i in [i for i in waiting if not str(i).startswith(prefix)]:
        notes.append(f"mkt/fills.json has no result for {i}, which is not this strategy's")
    for i in [i for i in waiting if str(i).startswith(prefix) and i in acknowledged]:
        notes.append(f"mkt/fills.json has no result for {i}; acknowledged in config/vol_arb_live.yaml: "
                     f"{acknowledged[i]}")
    mine = [i for i in waiting if str(i).startswith(prefix) and i not in acknowledged]
    if mine:
        stops.append(f"mkt/fills.json has no result yet for {', '.join(mine)}: the book may be stale. If Sanji has "
                     f"resolved the session and will not answer, add the id to config/vol_arb_live.yaml's "
                     f"acknowledged with the reason")
    return stops, notes


def _leg(key) -> str:
    cp, expiry, strike = key
    return f"{expiry} {strike:g} {'put' if cp == PUT else 'call'}"


def book_problems(book: Book, fills: list | None, ledger: dict, root: str, day) -> list[str]:
    """
    Why the book's NVDA positions are not (all) this strategy's, or not up to date with its fills (empty when they
    are). Every option held must be exactly what its filled orders add up to, except that a short leg may be gone
    (Sanji assigns a short option whole, on any close through the strike); a long leg its fills hold must be there;
    shares need at least one fill of its own (a hedge, or a leg whose assignment delivered them). With no
    mkt/fills.json at all, a book with positions can't be checked against anything.
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


def hedge_limit(cfg: dict, spot: float, target: int, shares: int) -> None:
    """
    The hedge's target and the shares held, each times D's close, may not pass `limits.max_hedge_x_capital` times
    the $100,000 book: legs sized $200 a vol point hold about 2,400 / spot contracts, so even one leg deep in the
    money is near 2.4 x, while a book read with contracts in shares would be about 100 times its size.
    """
    cap = float(cfg["limits"]["max_hedge_x_capital"]) * VA.CAPITAL
    for n, what in ((target, "hedge target"), (shares, "shares held")):
        if abs(n) * spot > cap:
            raise ValueError(f"the {what}, {n:+d} NVDA shares (${abs(n) * spot:,.0f} at {spot:.2f}), is more than "
                             f"{cfg['limits']['max_hedge_x_capital']:g} x the ${VA.CAPITAL:,.0f} book: the book is "
                             f"probably misread (contracts in shares?)")


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
    if legs:
        hedge_limit(cfg, float(mkt["stock"].at[pd.Timestamp(day), "close"]), target, book.shares)
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
    ack = cfg.get("acknowledged", {})
    if not isinstance(ack, dict) or not all(isinstance(k, str) and isinstance(v, str) and v.strip()
                                            for k, v in ack.items()):
        raise ValueError("acknowledged must map order ids to the reason each no longer stops the run")
    x = (cfg.get("limits") or {}).get("max_hedge_x_capital")
    if not _finite_number(x) or x <= 0:
        raise ValueError(f"limits.max_hedge_x_capital must be a positive number, not {x!r}")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- what the frozen code calls outside itself
# The lib functions and constants that lib/vol_arb.py and this module take from the public lib/wheel.py, lib/vol.py
# and lib/pmcc_daily.py (wheel.entry_sessions is left out: only vol_arb.stock_cycles, the backtest loop, calls it).
# dependency_hashes() follows each through everything it calls inside trading_app/lib.
BACKTEST_ROOTS = (("wheel", "PUT"), ("wheel", "CALL"), ("wheel", "CONTRACT"), ("wheel", "_num"), ("wheel", "carry"),
                  ("wheel", "option_greeks"), ("wheel", "select"), ("vol", "bs_greeks"), ("vol", "bs_price"),
                  ("pmcc_daily", "monthly_expiries"))


def _module_index(src: str) -> tuple[dict, dict, dict]:
    """
    One lib module's top level: {name: its statement's source lines}, {name: (module, name)} for names it imports
    from another lib module (`from .vol import bs_greeks`), and {alias: module} for `from . import wheel as W`.
    """
    lines, tree = src.splitlines(), ast.parse(src)
    defs, imports, aliases = {}, {}, {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            defs[node.name] = (node, "\n".join(lines[first - 1:node.end_lineno]))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            text = "\n".join(lines[node.lineno - 1:node.end_lineno])
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                for t in ast.walk(target):
                    if isinstance(t, ast.Name):
                        defs[t.id] = (node, text)
        elif isinstance(node, ast.ImportFrom) and node.level == 1:
            for a in node.names:
                if node.module is None:
                    aliases[a.asname or a.name] = a.name
                else:
                    imports[a.asname or a.name] = (node.module, a.name)
    return defs, imports, aliases


def dependency_hashes(read_source, roots=BACKTEST_ROOTS) -> dict:
    """
    {"module.name": sha256 of its source}, for each of `roots` and everything it uses in turn inside trading_app/lib
    (a name its body mentions that the module defines or imports from another lib module), keyed by the module that
    defines it: wheel.carry is pmcc_daily.carry. read_source(module) returns lib/<module>.py's text. A change to any
    of these lines, or a new lib function one of them starts calling, changes the result.
    """
    index, out, todo = {}, {}, list(roots)
    while todo:
        module, name = todo.pop()
        if module not in index:
            index[module] = _module_index(read_source(module))
        defs, imports, aliases = index[module]
        if name in imports and name not in defs:
            todo.append(imports[name])
            continue
        if name not in defs:
            raise KeyError(f"lib/{module}.py defines no {name}")
        key = f"{module}.{name}"
        if key in out:
            continue
        node, text = defs[name]
        out[key] = sha256_hex(text.encode())
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and (sub.id in defs or sub.id in imports):
                todo.append((module, sub.id))
            elif isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id in aliases:
                todo.append((aliases[sub.value.id], sub.attr))
    return dict(sorted(out.items()))


def hash_differences(pinned: dict, got: dict) -> list[str]:
    """What differs between the recorded dependency hashes and tonight's: changed, new, or no longer used."""
    out = [f"{k} changed" for k in sorted(set(pinned) & set(got)) if pinned[k] != got[k]]
    out += [f"{k} is used but not recorded" for k in sorted(set(got) - set(pinned))]
    out += [f"{k} is recorded but no longer used" for k in sorted(set(pinned) - set(got))]
    return out
