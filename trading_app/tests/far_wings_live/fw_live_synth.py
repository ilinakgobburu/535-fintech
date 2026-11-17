"""
Made-up data for the far_wings_live tests (not collected: no test_ prefix). Nothing here is market data:

    closes      alternating +2% / -2% log returns on the NYSE calendar from $175
    the chain   Black-76 mids at a smile sigma(x) = s0 + b x + c x^2, x = ln(K/F)/sqrt(T), on $2.50 strikes from
                about 0.70 to 1.43 times the spot; bid and ask 2% either side of the mid; a mid under 2 cents
                is not quoted
    the rule    rule_as_written(): the wing rule of the technical choices, written as a plain loop over quote rows,
                so the module's pandas version can be compared with it on the same chain
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from trading_app.lib import far_wings_live as FW
from trading_app.lib import wheel as W
from trading_app.lib import wheel_live as L
from trading_app.lib.ric import build_option_ric
from trading_app.lib.vol import bs_price

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "config" / "far_wings_live.yaml"
NVDA = 4815747
ENTRY, EXPIRY, FOR = dt.date(2026, 11, 23), dt.date(2026, 12, 18), dt.date(2026, 11, 24)
RATE_PCT = 4.0
DIVS = pd.Series([0.01, 0.01], index=pd.DatetimeIndex(["2026-06-11", "2026-09-04"]))


def config(**over) -> dict:
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg.update(over)
    return cfg


def closes(end=ENTRY, n=60, start_px=175.0, step=0.02, last=None) -> pd.Series:
    days = L.sessions(pd.Timestamp(end) - pd.Timedelta(days=int(n * 1.6)), end)[-n:]
    rets = np.array([step if i % 2 else -step for i in range(len(days))])
    rets[0] = 0.0
    s = pd.Series(start_px * np.exp(np.cumsum(rets)), index=days)
    if last is not None:
        s.iloc[-1] = float(last)
    return s


def trailing_yield(spot: float, day) -> float:
    """wheel.carry's dividend yield, by hand: the dividends of the 365 days up to `day` over the close."""
    day = pd.Timestamp(day)
    return float(DIVS[(DIVS.index > day - pd.Timedelta(days=365)) & (DIVS.index <= day)].sum()) / spot


def smile(K, F, T, b, s0=0.45, c=0.02):
    x = np.log(np.asarray(K, float) / F) / np.sqrt(T)
    return s0 + b * x + c * x * x


def strikes_around(spot: float, lo=0.70, hi=1.43, step=2.5) -> np.ndarray:
    return np.arange(math.floor(spot * lo / step) * step, math.ceil(spot * hi / step) * step + step / 2, step)


def chain_rows(spot, day=ENTRY, expiry=EXPIRY, b=-0.06, s0=0.45, strikes=None, r=RATE_PCT / 100, q=None,
               spread=0.02, drop=(), one_sided=()):
    """
    Option quote rows as far_wings_orders.chain() builds them. `drop` removes (cp, strike) pairs; `one_sided` keeps
    them with no bid (LSEG gives a zero bid as missing), so they are not two-sided.
    """
    q = trailing_yield(spot, day) if q is None else q
    T = (pd.Timestamp(expiry) - pd.Timestamp(day)).days / 365.0
    F, Dc = spot * np.exp((r - q) * T), np.exp(-r * T)
    ks = strikes_around(spot) if strikes is None else strikes
    rows = []
    for cp in ("P", "C"):
        for k in ks:
            if (cp, float(k)) in drop:
                continue
            mid = float(bs_price(F, float(k), T, float(smile(k, F, T, b, s0)), Dc, cp))
            if not mid > 0.02:
                continue
            bid = np.nan if (cp, float(k)) in one_sided else round(mid * (1 - spread), 6)
            rows.append({"date": pd.Timestamp(day), "ric": build_option_ric("NVDA", expiry, float(k), cp, expired=False),
                         "cp": cp, "strike": float(k), "expiry": pd.Timestamp(expiry),
                         "bid": bid, "ask": round(mid * (1 + spread), 6)})
    return rows


def market(day=ENTRY, rows=None, c=None, chain=True, **kw) -> dict:
    """The night's market dict (far_wings_live.market) from the made-up closes, T-bill, dividends and chain."""
    c = closes(end=day) if c is None else c
    rate = pd.Series(RATE_PCT, index=c.index)
    if rows is None and chain:
        rows = chain_rows(float(c.iloc[-1]), day=day, **kw)
    q = FW.quote_frame(rows) if rows is not None else None
    return FW.market(c, None, rate, DIVS, (), q)


def leg_history(spot_by_day: pd.Series, cp, expiry, strike, sigma, r=RATE_PCT / 100, missing=()):
    """A held leg's daily bid/ask, priced at one vol on each day's close; `missing` days have no quote."""
    rows = {}
    for d, s in spot_by_day.items():
        if d in missing:
            rows[d] = {"bid": np.nan, "ask": np.nan}
            continue
        T = (pd.Timestamp(expiry) - d).days / 365.0
        if T <= 0:
            continue
        q = trailing_yield(float(s), d)
        mid = float(bs_price(s * np.exp((r - q) * T), float(strike), T, sigma, np.exp(-r * T), cp))
        rows[d] = {"bid": mid * 0.99, "ask": mid * 1.01}
    return pd.DataFrame.from_dict(rows, orient="index")


# --------------------------------------------------------------------------- the rule, as the technical choices write it
CAP = {"P": 0.10, "C": 0.05}
TARGET = {"P": 0.05, "C": 0.03}


def rule_as_written(rows, cp, spot, day, expiry, r, q):
    """
    "Among the cycle expiry's two-sided puts at the entry close with |delta| < 0.10 (wheel.option_greeks on its own
    mid), the one with |delta| closest to 0.05 (tie to the lower strike); call wing: among two-sided calls with
    |delta| < 0.05, the one closest to 0.03 (tie to the higher strike). No eligible strike -> that wing has no
    position." A plain loop over the rows: (strike, |delta|), or None.
    """
    day, expiry = pd.Timestamp(day), pd.Timestamp(expiry)
    T = (expiry - day).days / 365.0
    best = None
    for row in rows:
        if row["cp"] != cp or pd.Timestamp(row["expiry"]) != expiry or pd.Timestamp(row["date"]) != day:
            continue
        bid, ask = float(row["bid"]), float(row["ask"])
        if not (math.isfinite(bid) and math.isfinite(ask) and ask >= bid):
            continue
        mid = (bid + ask) / 2
        if not mid > 0:
            continue
        g = W.option_greeks(mid, spot, float(row["strike"]), T, r, q, cp)
        if not math.isfinite(g["iv"]):
            continue
        ad = abs(g["delta"])
        if not ad < CAP[cp]:
            continue
        key = (abs(ad - TARGET[cp]), float(row["strike"]) if cp == "P" else -float(row["strike"]))
        if best is None or key < best[0]:
            best = (key, float(row["strike"]), ad)
    return None if best is None else (best[1], best[2])


# --------------------------------------------------------------------------- order.html, as written
ORDER_FIELDS = {"id", "for", "order_type", "action", "qty", "underlying", "contract", "expiry", "strike", "ref",
                "lmt_prc"}


def order_problems(o: dict) -> list[str]:
    """What order.txt (the course's order page) says Sanji rejects, checked field by field for the forms written."""
    out = []
    if set(o) - ORDER_FIELDS:
        out.append(f"unknown fields {set(o) - ORDER_FIELDS}")
    for k in ("id", "for", "order_type", "action", "qty", "underlying", "contract"):
        if o.get(k) in (None, ""):
            out.append(f"missing {k}")
    if not isinstance(o.get("id"), str):
        out.append("id is not a string")
    try:
        dt.date.fromisoformat(o.get("for", ""))
    except (TypeError, ValueError):
        out.append("for is not YYYY-MM-DD")
    if o.get("order_type") not in ("mkt", "lmt"):
        out.append("order_type")
    if o.get("action") not in ("BUY", "SEL", "SHT", "BTC"):
        out.append("action")
    if type(o.get("qty")) is not int or o["qty"] < 1:
        out.append("qty")
    if type(o.get("underlying")) is not int:
        out.append("underlying is not the integer conid")
    if o.get("contract") not in ("put", "call", "underlying"):
        out.append("contract")
    if o.get("contract") in ("put", "call"):
        try:
            dt.date.fromisoformat(o.get("expiry", ""))
        except (TypeError, ValueError):
            out.append("expiry is not a date")
        s = o.get("strike")
        if not (isinstance(s, str) and s.replace(".", "", 1).isdigit() and float(s) > 0):
            out.append("strike is not a bare positive number string")
    elif "expiry" in o or "strike" in o:
        out.append("expiry or strike on the underlying")
    if o.get("ref", "open") not in ("open", "prev_close"):
        out.append("ref")
    if o.get("order_type") == "lmt":
        p = o.get("lmt_prc")
        if not (isinstance(p, str) and p.replace(".", "", 1).isdigit() and float(p) > 0):
            out.append("lmt_prc is not a bare positive number string")
    elif "lmt_prc" in o:
        out.append("lmt_prc on a market order")
    return out


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))      # a config's YAML dates as strings
    return path
