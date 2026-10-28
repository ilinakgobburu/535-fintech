"""
The Mastercard page's blotter and ledger, audited against the raw LSEG files.

Every option trade is checked against that day's raw bid and ask, every expiry
against that day's raw close, the cash against a rebuild from the blotter and the
raw T-bill, and a random sample of sessions against the raw quotes for the
account's value. The page's numbers are also checked against a fresh run of the
engine, so the page shows what the engine did. Skipped on a fresh clone, where the
git-ignored LSEG cache does not exist.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_app.lib import wheel as W

REPO = Path(__file__).resolve().parents[3]
CACHE = REPO / "cache" / "wheel_raw_MA_2024"
PAGE = REPO / "cache" / "wheel_page_MA.json"
pytestmark = pytest.mark.skipif(not (CACHE / "options.pkl").exists() or not PAGE.exists(),
                                reason="needs the git-ignored LSEG cache")
SAMPLE = 40                           # random sessions per period, from a fixed seed


@pytest.fixture(scope="module")
def raw():
    frames = pickle.loads((CACHE / "options.pkl").read_bytes())
    quotes = {}
    for ric, df in frames.items():
        q = pd.DataFrame({"bid": pd.to_numeric(df["BID"], errors="coerce").astype(float),
                          "ask": pd.to_numeric(df["ASK"], errors="coerce").astype(float)})
        q.index = pd.DatetimeIndex(df.index).normalize()
        quotes[ric] = q
    stock = pd.read_pickle(CACHE / "stock_unadjusted_ohlc.pkl")
    close = pd.to_numeric(stock["TRDPRC_1"], errors="coerce").dropna()
    close.index = pd.DatetimeIndex(close.index).normalize()
    aux = pickle.loads((CACHE / "aux.pkl").read_bytes())
    tb = pd.to_numeric(aux["tbill"]["MID_YLD_1"], errors="coerce") / 100.0
    tb.index = pd.DatetimeIndex(tb.index).normalize()
    tbill = tb.reindex(close.index.union(tb.index)).ffill().reindex(close.index)
    return {"quotes": quotes, "close": close, "tbill": tbill, "page": json.loads(PAGE.read_text())}


def periods(raw):
    return raw["page"]["periods"].items()


def mid(raw, ric, day):
    q = raw["quotes"][ric]
    if day not in q.index:
        return np.nan
    b, a = q.at[day, "bid"], q.at[day, "ask"]
    return (b + a) / 2 if np.isfinite(b) and np.isfinite(a) and a >= b else np.nan


def test_every_option_trade_is_the_raw_mid_of_that_days_quote(raw):
    for _, P in periods(raw):
        for e in P["blotter"]:
            if e.get("cp") and e["side"] in ("SELL", "BUY"):
                day = pd.Timestamp(e["date"])
                q = raw["quotes"][e["instrument"]].loc[day]
                assert e["bid"] == pytest.approx(q["bid"]) and e["ask"] == pytest.approx(q["ask"])
                assert e["price"] == pytest.approx((q["bid"] + q["ask"]) / 2)
                assert e["cash_delta"] == pytest.approx(e["qty"] * e["price"])


def test_every_expiry_follows_the_raw_close_against_the_strike(raw):
    for _, P in periods(raw):
        for e in P["blotter"]:
            if e["kind"] in ("put", "call") and e["side"] in ("EXPIRE", "ASSIGN"):
                c = float(raw["close"][pd.Timestamp(e["date"])])
                assert pd.Timestamp(e["date"]) == pd.Timestamp(e["expiry"])
                itm = c < e["strike"] if e["kind"] == "put" else c > e["strike"]
                assert e["side"] == ("ASSIGN" if itm else "EXPIRE")


def test_cash_rebuilt_from_the_blotter_and_the_raw_t_bill_matches_every_session(raw):
    for _, P in periods(raw):
        cash, prev, events = 100_000.0, None, {}
        for e in P["blotter"]:
            events.setdefault(e["date"], []).append(e)
        for row in P["ledger"]:
            day = pd.Timestamp(row["date"])
            if prev is not None and cash > 0:
                cash += cash * float(raw["tbill"][prev]) * (day - prev).days / 365.0
            cash += sum(e["cash_delta"] for e in events.get(row["date"], []))
            assert row["cash"] == pytest.approx(cash, abs=0.01)
            prev = day


def test_a_random_sample_of_sessions_values_the_account_from_the_raw_quotes(raw):
    rng = np.random.default_rng(535)
    for _, P in periods(raw):
        rows = P["ledger"]
        last_mid = {}
        marks = {}
        for row in rows:                      # the last raw mid of the open contract, as of each session
            day = pd.Timestamp(row["date"])
            if row["ric"]:
                m = mid(raw, row["ric"], day)
                if np.isfinite(m):
                    last_mid[row["ric"]] = m
                marks[row["date"]] = last_mid.get(row["ric"])
        for i in rng.choice(len(rows), size=min(SAMPLE, len(rows)), replace=False):
            row = rows[i]
            day = pd.Timestamp(row["date"])
            assert row["close"] == pytest.approx(float(raw["close"][day]))
            owed = 0.0
            if row["ric"]:
                if marks[row["date"]] is not None:
                    assert row["mark"] == pytest.approx(marks[row["date"]])
                owed = row["units"] * row["mark"]
                assert row["units"] == 100 * row["contracts"]
            assert row["nav"] == pytest.approx(row["cash"] + row["shares"] * row["close"] - owed, abs=0.01)


def test_the_page_shows_exactly_what_a_fresh_run_of_the_engine_books(raw):
    m = W.load_market(CACHE)
    pick = raw["page"]["pick"]
    for name, P in periods(raw):
        res = W.run(m, start=P["window"][0], end=P["window"][1], ticker="MA", **pick)
        L = res["ledger"]
        assert [str(d.date()) for d in L["date"]] == [r["date"] for r in P["ledger"]]
        assert np.allclose(L["nav"], [r["nav"] for r in P["ledger"]])
        assert np.allclose(L["cash"], [r["cash"] for r in P["ledger"]])
        assert [(str(e["date"].date()), e["side"], e["qty"]) for e in res["blotter"]] == \
               [(e["date"], e["side"], e["qty"]) for e in P["blotter"]]


def test_reg_t_every_put_was_secured_and_every_call_covered(raw):
    for _, P in periods(raw):
        closes = {r["date"]: r["close"] for r in P["ledger"]}
        for d, cash, coll, free, shares, stock_value, need, covered in P["regt"]["rows"]:
            assert free == pytest.approx(cash - coll) and free >= -1e-6
            assert shares >= need and covered
            assert stock_value == pytest.approx(shares * closes[d])
