"""
The live entry against the backtest's own loop on the same raw data. One made-up NVDA history (seeded GBM closes,
quarterly dividends, a T-bill with missing prints, report dates, VIX) and a made-up Dec 18 chain (every $2.50
strike, Black-76 mids on a skewed smile, ticks of a cent, zero bids as LSEG sends them: missing) are written as the
backtest's cache (options.pkl, stock_unadjusted_ohlc.pkl, aux.pkl) and run through wheel.load_market and
vol_arb.stock_cycles, the backtest's loop (with vol-arb-ml's wheel.entry_sessions, which the public wheel.py does
not have; vol_arb_live.entry_sessions is its copy). The same raw data, as LSEG would answer the live script's
requests up to D, goes through vol_arb_orders.py's own readers (stock_bars, tbill, dividends, chain) and
vol_arb_live.market / measure_entry. The two must agree: the status, every measurement scalar, the legs (strike,
contracts, mid, IV, delta, vega) and the nine features (to 1e-12: the trailing sums start on different days, as
the reviewer found, so they agree to rounding, not bit for bit). Nothing here is market data.
"""

from __future__ import annotations

import datetime as dt
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

import va_live_synth as S
from trading_app.lib import vol_arb as VA
from trading_app.lib import vol_arb_live as VL
from trading_app.lib import wheel as W
from trading_app.lib import wheel_live as L
from trading_app.lib.ric import build_option_ric
from trading_app.lib.vol import bs_price

_path = list(sys.path)
sys.path.insert(0, str(S.REPO))
import vol_arb_orders as O  # noqa: E402

sys.path[:] = _path

D, EXP = pd.Timestamp("2026-11-23"), pd.Timestamp("2026-12-18")
LISTED = np.arange(50.0, 400.01, 2.5)
ACTUAL = [pd.Timestamp("2026-05-27"), pd.Timestamp("2026-08-26"), pd.Timestamp("2026-11-19")]
EXPECTED = [pd.Timestamp("2027-02-24")]


def history(seed: int, b: float, s0: float, d_holes: float):
    """The raw data both sides read: closes, opens, dividends, T-bill (%), VIX and every quote of the Dec 18 chain."""
    rng = np.random.default_rng(seed)
    idx = L.sessions("2026-03-02", "2026-12-31")
    rets = rng.normal(0.0, 0.45 / np.sqrt(252), len(idx))
    rets[0] = 0.0
    close = pd.Series(180.0 * np.exp(np.cumsum(rets)), index=idx)
    opn = close.shift(1).fillna(close.iloc[0]) * np.exp(rng.normal(0, 0.004, len(idx)))
    divs = pd.Series(0.01, index=pd.DatetimeIndex(["2026-03-11", "2026-06-11", "2026-09-10", "2026-12-04"]))
    tb = pd.Series(np.round(4.0 + np.cumsum(rng.normal(0, 0.01, len(idx))), 3), index=idx)
    tb = tb.drop(tb.index[[-40, -35, -27]])                     # sessions without a print (Nov 13, Nov 20, Dec 2)
    vix = pd.Series(np.round(17 + rng.normal(0, 1, len(idx)), 2), index=idx)
    level = pd.Series(s0 + np.cumsum(rng.normal(0, 0.005, len(idx))), index=idx)
    rows = []
    for day in idx[(idx >= EXP - pd.Timedelta(days=50)) & (idx <= EXP)]:
        spot, T = float(close[day]), (EXP - day).days / 365.0
        if T <= 0:
            continue
        r = float(tb.reindex(idx).ffill()[day]) / 100
        q = float(divs[(divs.index > day - pd.Timedelta(days=365)) & (divs.index <= day)].sum()) / spot
        F, Dc = spot * np.exp((r - q) * T), np.exp(-r * T)
        for cp in ("P", "C"):
            for k in LISTED:
                if day == D and rng.random() < d_holes:
                    continue                                    # no quote that close for this listed strike
                x = np.log(k / F) / np.sqrt(T)
                mid = float(bs_price(F, k, T, max(float(level[day]) + b * x + 0.02 * x * x, 0.05), Dc, cp))
                half = max(mid * 0.015, 0.01)
                bid, ask = round(mid - half, 2), round(mid + half, 2)
                rows.append({"date": day, "ric": build_option_ric("NVDA", EXP.date(), float(k), cp, expired=False),
                             "cp": cp, "strike": float(k), "bid": bid if bid > 0 else np.nan, "ask": ask})
    return close, opn, divs, tb, vix, pd.DataFrame(rows)


def live_plan(close):
    cfg = S.config()
    return VL.strike_plan(close[close.index <= D], VL.previous_expiry(EXP), D.date(), cfg["chain"]["bands"],
                          cfg["chain"]["grid"])


def write_cache(path, close, opn, divs, tb, quotes, plan):
    """The backtest's cache, holding the strikes the live script asks for, on every quote day."""
    path.mkdir()
    frames = {}
    for cp in ("P", "C"):
        for k in plan[cp]:
            sub = quotes[(quotes["cp"] == cp) & (quotes["strike"] == k)]
            if len(sub):
                frames[sub["ric"].iloc[0]] = pd.DataFrame(
                    {"BID": sub["bid"].to_numpy(), "ASK": sub["ask"].to_numpy(), "TRDPRC_1": np.nan},
                    index=pd.DatetimeIndex(sub["date"]))
    (path / "options.pkl").write_bytes(pickle.dumps(frames))
    pd.DataFrame({"TRDPRC_1": close, "OPEN_PRC": opn}).to_pickle(path / "stock_unadjusted_ohlc.pkl")
    aux = {"dividends": pd.DataFrame({"Instrument": "NVDA.O", "Dividend Ex Date": divs.index,
                                      "Dividend Unadjusted Gross": divs.to_numpy(), "Dividend Pay Date": divs.index}),
           "tbill": pd.DataFrame({"MID_YLD_1": tb}),
           "earnings": pd.DataFrame({"Instrument": "NVDA.O", "Event Start Date": ACTUAL + EXPECTED})}
    (path / "aux.pkl").write_bytes(pickle.dumps(aux))
    return frames


class LiveLSEG:
    """LSEG as the live script sees it after D's close: nothing later than D, the windows its requests ask for."""

    def __init__(self, close, opn, divs, tb, frames):
        self.close, self.opn, self.divs, self.tb, self.frames = close, opn, divs, tb, frames

    @staticmethod
    def _win(df, start, end):
        return df[(df.index >= pd.Timestamp(start)) & (df.index <= min(pd.Timestamp(end), D))]

    def get_history(self, universe, fields, interval="daily", start=None, end=None, adjustments=None, count=None):
        if universe == "NVDA.O":
            return self._win(pd.DataFrame({"TRDPRC_1": self.close, "OPEN_PRC": self.opn}), start, end)
        if universe == "US3MT=RR":
            return self._win(pd.DataFrame({"MID_YLD_1": self.tb}), start, end)
        if universe in self.frames:
            return self._win(self.frames[universe][["BID", "ASK"]], start, end)
        raise Exception(f"Error code -1 | The universe is not found: {universe}")

    def get_data(self, universe, fields, parameters=None):
        assert fields[0] == "TR.DivExDate"
        d = self.divs[(self.divs.index >= pd.Timestamp(parameters["SDate"]))
                      & (self.divs.index <= pd.Timestamp(parameters["EDate"]))]
        return pd.DataFrame({"Instrument": universe, "Dividend Ex Date": [str(x.date()) for x in d.index],
                             "Gross Dividend Amount": d.to_numpy(), "Dividend Pay Date": None})


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


@pytest.mark.parametrize("seed, b, s0, d_holes, strict", [
    (5, -0.05, 0.47, 0.0, True),         # a put skew, the chain complete on D (the 22:00 run)
    (11, 0.04, 0.40, 0.0, True),         # a call skew
    (2, -0.05, 0.47, 0.03, False),       # a few listed strikes without a D quote, as the 01:00 retry measures them
])
def test_the_live_entry_is_the_backtests(tmp_path, monkeypatch, capsys, seed, b, s0, d_holes, strict):
    close, opn, divs, tb, vix, quotes = history(seed, b, s0, d_holes)
    plan = live_plan(close)
    frames = write_cache(tmp_path / "cache", close, opn, divs, tb, quotes, plan)

    # the backtest: its loader and its loop over the cycles of Oct-Dec 2026
    monkeypatch.setattr(W, "entry_sessions", VL.entry_sessions, raising=False)
    mk_bt = W.load_market(tmp_path / "cache")
    bt = next(c for c in VA.stock_cycles(mk_bt, "2026-10-01", "2026-12-31", vix, "NVDA")
              if pd.Timestamp(c["entry"]) == D)

    # the live script's readers on LSEG's answers up to D, then the live measurement
    monkeypatch.setattr(O, "ld", LiveLSEG(close, opn, divs, tb, frames))
    monkeypatch.setattr(O.time, "sleep", lambda s: None)
    cfg, data = S.config(), S.config()["data"]
    closes, opens = O.stock_bars("NVDA.O", D.date(), int(data["closes_days"]))
    rate = O.tbill("US3MT=RR", D.date())
    dv = O.dividends("NVDA.O", D.date(), int(data["dividends_days"]))
    earn, _ = VL.earnings_dates(EXPECTED, [a for a in ACTUAL if a <= D], D, EXP, int(data["min_days_between_reports"]))
    chain = O.chain("NVDA", EXP.date(), plan, D.date(), int(cfg["chain"]["history_days"]),
                    float(cfg["chain"]["min_dated_share"]), strict=strict)
    lv = VL.measure_entry(VL.market(closes, opens, rate, dv, earn, chain), D, EXP, float(vix[D]))

    assert bt["status"] == lv["status"] == "measured"
    assert set(bt["measure"]) == set(lv["measure"])
    diff = {k: (v, lv["measure"][k]) for k, v in bt["measure"].items() if not same(v, lv["measure"][k])}
    assert diff == {}
    assert set(bt["features"]) == set(lv["features"]) == set(VA.FEATURES)
    for k in VA.FEATURES:
        assert lv["features"][k] == pytest.approx(bt["features"][k], rel=1e-12, abs=1e-15), k
    for k in VA.RAW:
        assert lv["raw_premiums"][k] == pytest.approx(bt["raw_premiums"][k], rel=1e-12, abs=1e-15), k
    for cp in ("P", "C"):
        a, z = bt["legs"][cp], lv["legs"][cp]
        assert {k: a[k] for k in ("ric", "strike", "contracts")} == {k: z[k] for k in ("ric", "strike", "contracts")}
        for k in ("mid", "bid", "ask", "iv", "delta", "vega"):
            assert same(a[k], z[k]) or z[k] == pytest.approx(a[k], rel=1e-12), (cp, k)
    assert lv["no_trade"] is None and bt["no_trade"] is None
    # and the orders carry the backtest's legs: put_rich here, so SHT the put and BUY the call
    p = VL.plan_night(cfg, D.date(), dt.date(2026, 11, 24), VL.Book(), entry_expiry=EXP.date(),
                      mkt=VL.market(closes, opens, rate, dv, earn, chain), vix_close=float(vix[D]),
                      panel=VL.load_panel(S.panel_obj(), cfg))
    direction = p["decision"]["direction"]
    signs = VA.DIRECTIONS[direction]
    for o in p["orders"]:
        cp = "P" if o["contract"] == "put" else "C"
        assert o["action"] == ("SHT" if signs[cp] < 0 else "BUY") and o["qty"] == bt["legs"][cp]["contracts"]
        assert o["strike"] == L.number_token(bt["legs"][cp]["strike"])
        assert o["lmt_prc"] == L.number_token(bt["legs"][cp]["mid"])
    assert len(p["orders"]) == 2


def test_the_holes_scenario_has_holes_and_the_first_run_stops_on_them(tmp_path, monkeypatch):
    close, opn, divs, tb, vix, quotes = history(2, -0.05, 0.47, 0.03)
    plan = live_plan(close)
    frames = write_cache(tmp_path / "cache", close, opn, divs, tb, quotes, plan)
    at_d = {r for r, f in frames.items() if D in f.index}
    assert 0 < len(frames) - len(at_d) < 0.1 * len(frames)
    monkeypatch.setattr(O, "ld", LiveLSEG(close, opn, divs, tb, frames))
    monkeypatch.setattr(O.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit, match="have earlier rows but none for 2026-11-23"):
        O.chain("NVDA", EXP.date(), plan, D.date(), 14, 0.9, strict=True)
