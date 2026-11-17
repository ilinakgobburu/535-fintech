"""
Made-up data for the nightly-mark tests (nothing here is market data): one NVDA-like history over the real
2026-10-19 -> 2026-11-20 cycle on the NYSE calendar (seeded GBM closes with a 22% gap down on Nov 5, opens near the
previous close, a dividend going ex inside the cycle, a T-bill with missing prints), and the Nov 20 chain priced by
Black-76 on a skewed smile every session, with a few holes of the kinds LSEG has: a leg with no row one day, a leg
with no bid another, and the put's mid under its intrinsic value on Nov 16 (no implied vol). The cycle's legs are
lib/vol_arb.legs_at_entry's on the entry close, written into a copy of config/nvda_marks.yaml; FakeLSEG answers the
script's requests from the same data, as LSEG would after a given close.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from trading_app.lib import nvda_marks as NM
from trading_app.lib import vol_arb as VA
from trading_app.lib import vol_arb_live as VL
from trading_app.lib import wheel_live as L
from trading_app.lib.ric import build_option_ric
from trading_app.lib.vol import bs_price

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "config" / "nvda_marks.yaml"
ENTRY, EXPIRY = pd.Timestamp("2026-10-19"), pd.Timestamp("2026-11-20")
EX_DATE = pd.Timestamp("2026-11-05")
NO_ROW = {(pd.Timestamp("2026-10-29"), "C")}           # the call has no row that close
NO_BID = {(pd.Timestamp("2026-11-02"), "P")}           # the put has a row but no bid that close
UNDER = pd.Timestamp("2026-11-16")                     # the put's mid under intrinsic that close
STRIKES = np.arange(120.0, 300.01, 2.5)


def history(seed: int = 3, s0: float = 200.0, vol: float = 0.50):
    """Closes, opens, T-bill (%), dividends over Sep-Dec 2026."""
    idx = L.sessions("2026-09-01", "2026-12-31")
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, vol / np.sqrt(252), len(idx))
    rets[0] = 0.0
    close = pd.Series(s0 * np.exp(np.cumsum(rets)), index=idx).round(2)
    close[close.index >= EX_DATE] = (close[close.index >= EX_DATE] * 0.78).round(2)
    opens = (close.shift(1).fillna(close.iloc[0]) * np.exp(rng.normal(0, 0.006, len(idx)))).round(3)
    tb = pd.Series(np.round(4.0 + np.cumsum(rng.normal(0, 0.01, len(idx))), 3), index=idx)
    tb = tb.drop([pd.Timestamp("2026-10-30"), pd.Timestamp("2026-11-11")])
    divs = pd.Series([0.40, 0.40], index=pd.DatetimeIndex(["2026-08-06", str(EX_DATE.date())]))
    return close, opens, tb, divs


def smile(K, F, T):
    x = np.log(K / F) / np.sqrt(T)
    return max(0.45 - 0.08 * x + 0.02 * x * x, 0.05)


def option_frames(close, tb, divs, last=EXPIRY) -> dict:
    """{live RIC: DataFrame(BID, ASK) by date} for the Nov 20 chain, sessions from three weeks before the entry to
    `last`, with the holes above."""
    tbf = tb.reindex(close.index.union(tb.index)).ffill()
    frames = {}
    days = close.index[(close.index >= ENTRY - pd.Timedelta(days=21)) & (close.index <= last)]
    for cp in ("P", "C"):
        for k in STRIKES:
            ric = build_option_ric("NVDA", EXPIRY.date(), float(k), cp, expired=False)
            rows = {}
            for d in days:
                if (d, cp) in NO_ROW:
                    continue
                spot, T = float(close[d]), (EXPIRY - d).days / 365.0
                if T <= 0:
                    intrinsic = max(k - spot, 0.0) if cp == "P" else max(spot - k, 0.0)
                    rows[d] = (round(max(intrinsic - 0.05, 0.0), 2) or np.nan, round(intrinsic + 0.05, 2))
                    continue
                r = float(tbf[d]) / 100.0
                q = float(divs[(divs.index > d - pd.Timedelta(days=365)) & (divs.index <= d)].sum()) / spot
                F, D = spot * np.exp((r - q) * T), np.exp(-r * T)
                mid = float(bs_price(F, float(k), T, smile(k, F, T), D, cp))
                if d == UNDER and cp == "P" and k - spot > 1.0:
                    mid = (k - spot) - 0.40                    # under intrinsic: no implied vol
                half = max(0.02 * mid, 0.025)
                bid, ask = round(mid - half, 2), round(mid + half, 2)
                if (d, cp) in NO_BID or bid <= 0:
                    bid = np.nan
                rows[d] = (bid, ask)
            frames[ric] = pd.DataFrame(rows, index=["BID", "ASK"]).T.astype(float)
            frames[ric].index = pd.DatetimeIndex(frames[ric].index)
    return frames


def quotes_of(frames: dict, upto=None) -> pd.DataFrame:
    rows = []
    for ric, f in frames.items():
        meta = NM.parse_option_ric(ric)
        for d, b, a in zip(f.index, f["BID"], f["ASK"]):
            if upto is None or d <= pd.Timestamp(upto):
                rows.append({"date": d, "ric": ric, "cp": meta["cp"], "strike": meta["strike"],
                             "expiry": pd.Timestamp(meta["expiry"]), "bid": b, "ask": a})
    return VL.quote_frame(rows)


def market(upto=None, seed: int = 3):
    """lib/vol_arb_live.market on the made-up data (everything, or up to `upto`), and the option frames."""
    close, opens, tb, divs = history(seed)
    frames = option_frames(close, tb, divs)
    if upto is not None:
        u = pd.Timestamp(upto)
        close, opens, tb = close[close.index <= u], opens[opens.index <= u], tb[tb.index <= u]
    return VL.market(close, opens, tb, divs, (), quotes_of(frames, upto)), frames


def config(mk=None) -> dict:
    """config/nvda_marks.yaml with the cycle's legs replaced by legs_at_entry's on the made-up entry close."""
    cfg = yaml.safe_load(CONFIG.read_text())
    mk = mk if mk is not None else market()[0]
    q = mk["quotes"]
    q = q[(q["date"] == ENTRY) & (q["expiry"] == EXPIRY)]
    legs = VA.legs_at_entry(mk, q, ENTRY, EXPIRY)
    c = cfg["cycle"]
    c["spot_at_entry"] = float(mk["stock"].at[ENTRY, "close"])
    c["atm_iv_at_entry"] = 0.45
    c["lot"] = int(legs["lot"])
    c["legs"] = {cp: {k: legs[cp][k] for k in NM.LEG_KEYS} for cp in ("P", "C")}
    return cfg


def spec(mk=None) -> dict:
    return NM.cycle_spec(config(mk))


def write_config(path: Path, cfg: dict) -> Path:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


class Session:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeLSEG:
    """
    LSEG as the script sees it after `published`'s close: nothing later. `no_close`/`no_open`: sessions whose
    close/open is not there yet; `no_row`: {(ric, day)} with no row yet; `renamed`: live RICs LSEG only knows under
    the expired name; `broken`: RICs whose request fails with another error; `empty`: names (live or expired, as
    asked) LSEG lists but answers with no rows and no error (`empty_as`: an empty frame, or None); `cut`: {live RIC:
    (first, last)} for a history LSEG gives only between those days (None: no bound).
    """

    def __init__(self, published, seed: int = 3, no_close=(), no_open=(), no_row=(), renamed=(), broken=(),
                 empty=(), empty_as="frame", cut=None):
        self.close, self.opens, self.tb, self.divs = history(seed)
        self.frames = option_frames(self.close, self.tb, self.divs)
        self.published = pd.Timestamp(published)
        self.no_close = {pd.Timestamp(d) for d in no_close}
        self.no_open = {pd.Timestamp(d) for d in no_open}
        self.no_row = {(r, pd.Timestamp(d)) for r, d in no_row}
        self.renamed, self.broken = set(renamed), set(broken)
        self.empty, self.empty_as = set(empty), empty_as
        self.cut = {r: tuple(None if d is None else pd.Timestamp(d) for d in v) for r, v in (cut or {}).items()}
        self.calls = []

    def _win(self, df, start, end):
        return df[(df.index >= pd.Timestamp(start)) & (df.index <= min(pd.Timestamp(end), self.published))]

    def get_history(self, universe, fields, interval="daily", start=None, end=None, adjustments=None, count=None):
        self.calls.append(("get_history", universe, tuple(fields), start, end, tuple(adjustments or ())))
        if universe in self.broken:
            raise Exception("Error code 500 | Internal Server Error")
        if universe == "NVDA.O":
            df = pd.DataFrame({"TRDPRC_1": self.close, "OPEN_PRC": self.opens})
            df.loc[df.index.isin(self.no_close), "TRDPRC_1"] = np.nan
            df.loc[df.index.isin(self.no_open), "OPEN_PRC"] = np.nan
            df = df[~(df.index.isin(self.no_close) & df.index.isin(self.no_open))]
            return self._win(df, start, end)
        if universe == "US3MT=RR":
            return self._win(pd.DataFrame({"MID_YLD_1": self.tb}), start, end)
        live = universe.split("^")[0]
        expired = "^" in universe
        if universe in self.empty and live in self.frames:
            return None if self.empty_as is None else pd.DataFrame(columns=["BID", "ASK"], dtype=float)
        if live in self.frames and (expired == (live in self.renamed)):
            f = self.frames[live]
            f = f[[(live, d) not in self.no_row for d in f.index]]
            first, last = self.cut.get(live, (None, None))
            f = f[[(first is None or d >= first) and (last is None or d <= last) for d in f.index]]
            return self._win(f[["BID", "ASK"]], start, end)
        raise Exception(f"Error code -1 | The universe is not found: {universe}")

    def get_data(self, universe, fields, parameters=None):
        self.calls.append(("get_data", universe, tuple(fields)))
        assert fields[0] == "TR.DivExDate"
        d = self.divs[(self.divs.index >= pd.Timestamp(parameters["SDate"]))
                      & (self.divs.index <= pd.Timestamp(parameters["EDate"]))]
        return pd.DataFrame({"Instrument": universe, "Dividend Ex Date": [str(x.date()) for x in d.index],
                             "Gross Dividend Amount": d.to_numpy(), "Dividend Pay Date": None})


# --------------------------------------------------------------------------- the file's schema
def schema_problems(doc) -> list[str]:
    """What is wrong with a docs/data/vol_arb_nvda_mark.json document (empty when nothing is)."""
    out = []

    def need(cond, what):
        if not cond:
            out.append(what)

    def num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)

    def iso(x):
        try:
            return isinstance(x, str) and dt.date.fromisoformat(x).isoformat() == x
        except ValueError:
            return False

    need(isinstance(doc, dict), "not an object")
    if not isinstance(doc, dict):
        return out
    need(set(doc) == {"schema_version", "strategy", "underlying", "as_of", "generated_at", "status", "data_complete",
                      "cycle", "sides", "fills", "nvda", "vol", "legs_now", "pnl", "marks", "settlement", "events",
                      "notes"},
         f"top-level keys {sorted(doc)}")
    need(isinstance(doc.get("data_complete"), bool), "data_complete is true or false")
    need(doc.get("schema_version") == 1, "schema_version")
    need(doc.get("strategy") == "vol_arb", "strategy")
    need(iso(doc.get("as_of")), "as_of")
    try:
        dt.datetime.strptime(doc.get("generated_at", ""), "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        out.append("generated_at is not UTC ISO")
    need(doc.get("status") in ("open", "settled"), "status")
    c = doc.get("cycle") or {}
    need(iso(c.get("entry")) and iso(c.get("expiry")), "cycle dates")
    for leg in ("put", "call"):
        lg = (c.get("legs") or {}).get(leg) or {}
        need(set(lg) == {"ric", "strike", "contracts", "mid", "bid", "ask", "iv", "delta"}, f"cycle.legs.{leg}")
    for reading in ("f7_0", "f7_1"):
        dec = (c.get("decisions") or {}).get(reading) or {}
        need(set(dec) == set(NM.MODELS), f"decisions.{reading}")
        for m, v in dec.items():
            need(v.get("direction") in ("put_rich", "call_rich", None), f"decisions.{reading}.{m}.direction")
            need(v.get("p") is None or (num(v["p"]) and 0 <= v["p"] <= 1), f"decisions.{reading}.{m}.p")
    for leg in ("put", "call"):
        now = (doc.get("legs_now") or {}).get(leg) or {}
        need(set(now) == {"mark", "iv", "iv_from", "mark_from", "intrinsic", "iv_at_entry"}, f"legs_now.{leg}")
        need(all(num(now.get(k)) for k in ("mark", "iv", "intrinsic", "iv_at_entry")), f"legs_now.{leg} numbers")
        need(all(iso(now.get(k)) and now[k] <= doc.get("as_of", "") for k in ("iv_from", "mark_from")),
             f"legs_now.{leg} dates on or before as_of")
    for k in ("entry_close", "close", "change", "change_pct", "high_close", "low_close"):
        need(num((doc.get("nvda") or {}).get(k)), f"nvda.{k}")
    v = doc.get("vol") or {}
    need(num(v.get("atm_iv_at_entry")), "vol.atm_iv_at_entry")
    need(num(v.get("realized_since_entry")) if v.get("returns") else v.get("realized_since_entry") is None,
         "vol.realized_since_entry (null only on the entry close, with no return yet)")
    for side in ("put_rich", "call_rich"):
        for fill in ("mid", "stress"):
            m = ((doc.get("marks") or {}).get(side) or {}).get(fill) or {}
            need(num(m.get("pnl")) and m.get("pnl") == ((doc.get("pnl") or {}).get(side) or {}).get(fill),
                 f"pnl.{side}.{fill}")
            need(set(m.get("split") or {}) == set(VA.SPLIT), f"marks.{side}.{fill}.split")
            need(isinstance(m.get("hedge_trades"), int), f"marks.{side}.{fill}.hedge_trades")
            rows = m.get("daily") or []
            need(len(rows) >= 1 and rows[-1].get("date") == doc.get("as_of"), f"marks.{side}.{fill}.daily ends at as_of")
            need([r.get("date") for r in rows] == sorted(r.get("date") for r in rows), "daily rows in date order")
            for r in rows:
                need(iso(r.get("date")) and all(num(r.get(k)) for k in ("close", "pnl_to_date", "day_pnl", "put_mark",
                                                                      "call_mark")), f"row {r.get('date')}")
    need(isinstance(doc.get("notes"), list) and all(isinstance(n, str) for n in doc["notes"]), "notes")
    st = doc.get("settlement")
    need((st is None) == (doc.get("status") == "open"), "settlement only when settled")
    text = json.dumps(doc)
    need("NaN" not in text and "Infinity" not in text and "-0.0," not in text, "NaN, Infinity or -0.0 in the file")
    return out


def dumps(doc: dict) -> str:
    return json.dumps(doc, indent=1) + "\n"


def copy_cfg(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
