"""
Made-up data for the vol_arb_live tests (not collected: no test_ prefix). Nothing here is market data:

    closes      alternating +2% / -2% log returns on the NYSE calendar, so the up and down semivariances are
                equal and rule 1's direction is set by the smile alone
    the chain   Black-76 mids at a smile sigma(x) = 0.45 + b x + 0.02 x^2, x = ln(K/F)/sqrt(T): b < 0 a put
                skew (put side rich), b > 0 a call skew; bid and ask 2% either side of the mid
    the panel   5 made-up stocks' worth of rows on the development calendar (59 entries each), features drawn
                from a seeded normal, labels by a stated rule
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from trading_app.lib import vol_arb as VA
from trading_app.lib import vol_arb_live as VL
from trading_app.lib import wheel_live as L
from trading_app.lib.ric import build_option_ric
from trading_app.lib.vol import bs_price

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "config" / "vol_arb_live.yaml"
NVDA = 4815747
ENTRY, EXPIRY = dt.date(2026, 11, 23), dt.date(2026, 12, 18)
RATE_PCT = 4.0
DEV_STOCKS = ["NVDA", "AMD", "DIS", "COST", "DAL"]


def config(**over) -> dict:
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg.update(over)
    return cfg


def closes(end=ENTRY, n=140, start_px=175.0, step=0.02) -> pd.Series:
    days = L.sessions(pd.Timestamp(end) - pd.Timedelta(days=int(n * 1.6)), end)[-n:]
    rets = np.array([step if i % 2 else -step for i in range(len(days))])
    rets[0] = 0.0
    return pd.Series(start_px * np.exp(np.cumsum(rets)), index=days)


def market_parts(end=ENTRY, n=140):
    c = closes(end, n)
    o = c.shift(1).fillna(c.iloc[0]) * 1.001
    rate = pd.Series(RATE_PCT, index=c.index)
    div = pd.Series([0.01, 0.01], index=pd.DatetimeIndex(["2026-06-11", "2026-09-04"]))
    return c, o, rate, div


def smile(K, F, T, b, sigma0=0.45, c=0.02):
    x = np.log(np.asarray(K, float) / F) / np.sqrt(T)
    return sigma0 + b * x + c * x * x


def chain_rows(spot, day=ENTRY, expiry=EXPIRY, b=-0.06, strikes=None, r=RATE_PCT / 100, q=0.0, spread=0.02,
               drop=()):
    """Option quote rows as vol_arb_orders.chain() builds them; `drop` removes (cp, strike) pairs."""
    T = (pd.Timestamp(expiry) - pd.Timestamp(day)).days / 365.0
    F, Dc = spot * np.exp((r - q) * T), np.exp(-r * T)
    if strikes is None:
        strikes = {"P": np.arange(122.5, 185.01, 2.5), "C": np.arange(170.0, 252.51, 2.5)}
    rows = []
    for cp in ("P", "C"):
        for k in strikes[cp]:
            if (cp, float(k)) in drop:
                continue
            mid = float(bs_price(F, float(k), T, float(smile(k, F, T, b)), Dc, cp))
            if not mid > 0.02:
                continue
            rows.append({"date": pd.Timestamp(day), "ric": build_option_ric("NVDA", expiry, float(k), cp, expired=False),
                         "cp": cp, "strike": float(k), "expiry": pd.Timestamp(expiry),
                         "bid": round(mid * (1 - spread), 6), "ask": round(mid * (1 + spread), 6)})
    return rows


def trailing_yield(div: pd.Series, spot: float, day) -> float:
    """wheel.carry's dividend yield, by hand: the dividends of the 365 days up to `day` over the close."""
    day = pd.Timestamp(day)
    return float(div[(div.index > day - pd.Timedelta(days=365)) & (div.index <= day)].sum()) / spot


def entry_market(b=-0.06, day=ENTRY, expiry=EXPIRY, **kw) -> dict:
    """The entry night's market, the chain priced at the forward the measurement uses (r and the trailing q)."""
    c, o, rate, div = market_parts(day)
    spot = float(c.iloc[-1])
    q = VL.quote_frame(chain_rows(spot, day, expiry, b, q=trailing_yield(div, spot, day), **kw))
    return VL.market(c, o, rate, div, earnings=[pd.Timestamp("2026-11-19"), pd.Timestamp("2027-02-25")], quotes=q)


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
        mid = float(bs_price(s * np.exp(r * T), float(strike), T, sigma, np.exp(-r * T), cp))
        rows[d] = {"bid": mid * 0.99, "ask": mid * 1.01}
    return pd.DataFrame.from_dict(rows, orient="index")


# --------------------------------------------------------------------------- the panel
def dev_entries():
    idx = L.sessions("2019-01-02", "2023-12-29")
    return VL.entry_sessions(idx, "2019-01-02", "2023-12-29")


def panel_obj(rule="tilt", chosen="rule_1", seed=7, stocks=DEV_STOCKS, rows=None, constant=False) -> dict:
    """
    A made-up panel file. rule "tilt": label 1 when the tilt f4 is below zero (a put skew), which a depth-3
    forest and the logistic both learn; "separable": label 1 when f1 > f2; "alternating": labels 1, 0, 1, ...
    in row order. constant=True makes every row's features 0.1 (nothing to learn from).
    """
    rng = np.random.default_rng(seed)
    out = []
    i = 0
    for s in stocks:                                  # stock by stock, as the runner's cycles come
        for e, x in dev_entries().items():
            f = {"f1_put_premium": rng.normal(0, 0.4), "f2_call_premium": rng.normal(0, 0.4),
                 "f3_protection_share": rng.normal(-0.05, 0.05), "f4_tilt": rng.normal(0.0, 0.05),
                 "f5_bend": rng.normal(0.02, 0.02), "f6_iv_over_rv": rng.normal(1.3, 0.2),
                 "f7_earnings_in_cycle": float(rng.integers(0, 2)), "f8_vix": rng.normal(0.2, 0.05),
                 "f9_ret21": rng.normal(0, 0.08)}
            if constant:
                f = {k: 0.1 for k in f}
            if rule == "tilt":
                label = int(f["f4_tilt"] < 0)
            elif rule == "separable":
                label = int(f["f1_put_premium"] > f["f2_call_premium"])
            elif rule == "alternating":
                label = i % 2
            else:
                raise ValueError(rule)
            out.append({"id": f"{s}|{e.date()}", "stock": s, "entry": str(e.date()), "expiry": str(x.date()),
                        **{k: float(v) for k, v in f.items()}, "label": label})
            i += 1
    return {"format": VL.PANEL_FORMAT, "made_from": {"file": "synthetic", "sha256": "0" * 64, "commit": None},
            "environment": {}, "stocks": list(stocks), "window": ["2019-01-02", "2023-12-29"],
            "choice_for_test": {"chosen": chosen}, "features": list(VA.FEATURES),
            "rows": rows if rows is not None else out}


def pin_panel(cfg: dict, panel_path: Path) -> dict:
    """cfg with panel_pin set to the panel file at `panel_path`, as export_vol_arb_panel.py prints it."""
    raw = Path(panel_path).read_bytes()
    made = json.loads(raw).get("made_from") or {}
    return {**cfg, "panel_pin": {"sha256": VL.sha256_hex(raw), "dev_output_sha256": made.get("sha256"),
                                 "dev_output_commit": made.get("commit")}}


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))      # a config's YAML dates as strings
    return path
