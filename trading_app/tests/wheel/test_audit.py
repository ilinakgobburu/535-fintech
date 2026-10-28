"""
An audit of the blotter and the ledger on random markets.

Each seed draws a random price path (a random walk with jumps, so puts get
assigned and calls get written), prices every option from it as test_wheel.py
does, and runs the wheel at a random setting from the grid. The checks below do
not reuse the engine's logic: they rebuild what the books should say from the
blotter, the quotes and the closes, and compare.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from trading_app.lib import wheel as W
from test_wheel import make_market, nearest

SEEDS = range(12)
HOLIDAYS = pd.to_datetime(["2019-01-21", "2019-02-18", "2019-04-19", "2019-05-27", "2019-07-04"])
START, END = "2019-01-02", "2019-06-28"
MANAGED = {"take_profit": 0.5, "roll_days": 7, "max_rolls": 2, "early_assignment": True,
           "redeploy": True, "roll_lower": True}


def random_market(seed: int, rate: float = 0.0, dividend: bool = False):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2018-12-03", "2019-07-31")
    days = days[~days.isin(HOLIDAYS)]
    steps = rng.normal(0, 0.018, len(days)) + np.where(rng.random(len(days)) < 0.03,
                                                         rng.choice([-0.07, 0.06], len(days)), 0.0)
    px = np.clip(50 * np.exp(np.cumsum(steps)), 36, 64).round(2)
    divs = {"2019-03-06": 0.40, "2019-06-05": 0.40} if dividend else None
    m = make_market(path={str(d.date()): float(v) for d, v in zip(days, px)}, rate=rate, dividends=divs)
    setting = {"put_delta": float(rng.choice([0.15, 0.25, 0.35])), "call_delta": float(rng.choice([0.15, 0.25, 0.35])),
               "floor": bool(rng.choice([True, False]))}
    return m, setting


def by_day(blotter):
    out = defaultdict(list)
    for e in blotter:
        out[e["date"]].append(e)
    return out


def rebuild(res, m):
    """Cash and shares, close by close, from the blotter and the bill alone."""
    cash, shares, prev, rows = res["start_cash"], 0, None, []
    events = by_day(res["blotter"])
    for day in res["ledger"]["date"]:
        if prev is not None and cash > 0:
            cash += cash * float(m["tbill"][prev]) * (day - prev).days / 365.0
        for e in events.get(day, []):
            cash += e["cash_delta"]
            if e["kind"] == "stock" and e["side"] in ("BUY", "SELL"):
                shares += e["qty"] if e["side"] == "BUY" else -e["qty"]
        rows.append((day, cash, shares))
        prev = day
    return pd.DataFrame(rows, columns=["date", "cash", "shares"]).set_index("date")


def option_trades(res):
    """Each option the book wrote, with how it ended."""
    trades, open_ = [], {}
    for e in res["blotter"]:
        if e.get("cp") and e["side"] == "SELL":
            open_[e["instrument"]] = {"sale": e, "end": None}
            trades.append(open_[e["instrument"]])
        elif e["kind"] in ("put", "call") and e["side"] in ("EXPIRE", "ASSIGN", "BUY"):
            ric = e["instrument"] if e.get("cp") else next(r for r, t in open_.items() if t["end"] is None)
            open_[ric]["end"] = e
    return trades


@pytest.mark.parametrize("seed", SEEDS)
def test_cash_and_shares_rebuilt_from_the_blotter_match_the_ledger(seed):
    for manage in (None, MANAGED):
        m, setting = random_market(seed, rate=0.045, dividend=True)
        res = W.run(m, start=START, end=END, manage=manage, **setting)
        L, R = res["ledger"].set_index("date"), rebuild(res, m)
        assert np.allclose(L["cash"], R["cash"], atol=1e-6)
        assert (L["shares"] == R["shares"]).all()
        assert (L["shares"] >= 0).all()


@pytest.mark.parametrize("seed", SEEDS)
def test_account_value_is_cash_plus_shares_less_the_option_owed_at_that_days_mid(seed):
    m, setting = random_market(seed)
    res = W.run(m, start=START, end=END, **setting)
    L = res["ledger"]
    held = L[L["short_ric"].notna()]
    q = m["quotes"].set_index(["date", "ric"])["mid"]
    for _, row in held.iterrows():
        mid = q.get((row["date"], row["short_ric"]), np.nan)
        if np.isfinite(mid):
            assert row["short_mark"] == pytest.approx(mid)
    nav = L["cash"] + L["shares"] * L["stock_close"] - L["short_units"].fillna(0) * L["short_mark"].fillna(0)
    assert np.allclose(nav, L["nav"])
    assert (L["stock_close"].to_numpy() == m["stock"]["close"].reindex(L["date"]).to_numpy()).all()


@pytest.mark.parametrize("seed", SEEDS)
def test_every_option_trade_is_at_that_days_quote(seed):
    for fill in ("mid", "cross"):
        m, setting = random_market(seed)
        res = W.run(m, start=START, end=END, fill=fill, manage=MANAGED, **setting)
        q = m["quotes"].set_index(["date", "ric"])
        for e in res["blotter"]:
            if not e.get("cp") or e["side"] not in ("SELL", "BUY"):
                continue
            row = q.loc[(e["date"], e["instrument"])]
            want = row["mid"] if fill == "mid" else (row["bid"] if e["side"] == "SELL" else row["ask"])
            assert e["price"] == pytest.approx(want)
            if np.isfinite(row["bid"]):
                assert row["bid"] - 1e-9 <= e["price"] <= row["ask"] + 1e-9
            assert e["cash_delta"] == pytest.approx((1 if e["side"] == "SELL" else -1) * e["qty"] * e["price"])


@pytest.mark.parametrize("seed", SEEDS)
def test_every_put_is_cash_secured_and_every_call_covered(seed):
    m, setting = random_market(seed, rate=0.045)
    res = W.run(m, start=START, end=END, **setting)
    L = res["ledger"].set_index("date")
    for t in option_trades(res):
        e = t["sale"]
        if e["cp"] == "P":
            assert L.at[e["date"], "cash"] - e["cash_delta"] >= e["strike"] * e["qty"] - 1e-6
        else:
            assert L.at[e["date"], "shares"] >= e["qty"]
    calls = L[L["short"] == "C"]
    assert (calls["shares"] >= calls["short_units"]).all()


@pytest.mark.parametrize("seed", SEEDS)
def test_each_option_ends_the_way_the_expiry_close_says(seed):
    m, setting = random_market(seed)
    res = W.run(m, start=START, end=END, **setting)
    close = m["stock"]["close"]
    events = by_day(res["blotter"])
    for t in option_trades(res):
        sale, end = t["sale"], t["end"]
        if end is None:                                   # still open at the window's end
            assert pd.Timestamp(sale["expiry"]) > pd.Timestamp(END)
            continue
        k, c = sale["strike"], float(close[end["date"]])
        assert end["date"] == pd.Timestamp(sale["expiry"])
        itm = c < k if sale["cp"] == "P" else c > k
        assert end["side"] == ("ASSIGN" if itm else "EXPIRE")
        if itm:
            assert end["price"] == k
            stock = [e for e in events[end["date"]] if e["kind"] == "stock" and e["side"] in ("BUY", "SELL")]
            assert len(stock) == 1 and stock[0]["price"] == k and stock[0]["qty"] == sale["qty"]
            assert stock[0]["side"] == ("BUY" if sale["cp"] == "P" else "SELL")


@pytest.mark.parametrize("seed", SEEDS)
def test_each_strike_is_the_one_closest_to_the_target_delta_and_calls_respect_the_floor(seed):
    m, setting = random_market(seed)
    res = W.run(m, start=START, end=END, **setting)
    L = res["ledger"].set_index("date")
    for t in option_trades(res):
        e = t["sale"]
        target = setting["put_delta"] if e["cp"] == "P" else setting["call_delta"]
        floor = None
        if e["cp"] == "C" and setting["floor"]:
            floor = float(L.loc[L.index < e["date"], "basis"].iloc[-1])
            assert e["strike"] >= floor - 1e-9
        spot = float(m["stock"]["close"][e["date"]])
        assert e["strike"] == nearest(spot, e["date"], pd.Timestamp(e["expiry"]), e["cp"], target, floor)


@pytest.mark.parametrize("seed", SEEDS)
def test_dividends_are_paid_on_the_shares_held_at_the_prior_close(seed):
    m, setting = random_market(seed, dividend=True)
    res = W.run(m, start=START, end=END, **setting)
    L = res["ledger"].set_index("date")
    paid = {e["date"]: e for e in res["blotter"] if e["side"] == "DIVIDEND"}
    for day, amount in m["dividends"].items():
        prior = L.loc[L.index < day, "shares"]
        held = int(prior.iloc[-1]) if len(prior) else 0
        if held:
            assert paid[day]["cash_delta"] == pytest.approx(held * amount)
        else:
            assert day not in paid


def test_the_random_markets_exercise_every_kind_of_trade():
    # so that the checks above cannot pass because nothing happened
    seen = defaultdict(int)
    for seed in SEEDS:
        m, setting = random_market(seed, rate=0.045, dividend=True)
        for e in W.run(m, start=START, end=END, **setting)["blotter"]:
            seen[(e["kind"], e["side"])] += 1
    assert seen[("put", "ASSIGN")] >= 3 and seen[("call", "SELL")] >= 5
    assert seen[("call", "ASSIGN")] >= 1 and seen[("call", "EXPIRE")] >= 3
    assert seen[("put", "EXPIRE")] >= 10 and seen[("stock", "DIVIDEND")] >= 3
