"""
config/wheel_rsi.yaml's entry filter on test_wheel.py's market: Wilder's RSI(14)
of the closes up to the entry session; no new put while it is above
rsi_put_max, no new covered call while it is below rsi_call_min.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_app.lib import wheel as W
from test_wheel import END, START, cycle, make_market

# up into the first entry, a dip, a drop into the March expiry (assigned), then a bounce
PATH = {"2019-01-08": 51.0, "2019-01-09": 52.0, "2019-01-10": 53.0, "2019-01-11": 54.0, "2019-01-14": 55.0,
        "2019-01-30": 53.0, "2019-03-14": 45.0, "2019-04-01": 46.0, "2019-04-02": 47.0, "2019-04-03": 48.0}
PUT_ENTRIES = ("2019-01-22", "2019-02-19")                  # on PATH: no shares
CALL_ENTRIES = ("2019-03-18", "2019-04-22", "2019-05-20")   # on PATH: assigned at the March expiry
ENTRIES = PUT_ENTRIES + CALL_ENTRIES


def sold_on(res, day):
    return [e for e in res["blotter"] if e["side"] == "SELL" and e.get("cp") and e["date"] == pd.Timestamp(day)]


def fake_rsi(monkeypatch, values: dict, seen: list | None = None):
    """Replace rsi() with fixed values on the named sessions (50 elsewhere)."""
    def fake(closes, n=W.RSI_PERIOD):
        if seen is not None:
            seen.append(n)
        out = pd.Series(50.0, index=closes.index)
        for d, v in values.items():
            out[pd.Timestamp(d)] = v
        return out
    monkeypatch.setattr(W, "rsi", fake)


# --------------------------------------------------------------------------
# the indicator
# --------------------------------------------------------------------------

def test_rsi_matches_wilder_by_hand_over_three_sessions():
    # changes +1, -0.5, +1, -0.5, +1. Seed: gain (1 + 0 + 1) / 3 = 2/3, loss 0.5 / 3 = 1/6, RS 4 -> 80.
    # Then gain (2/3 x 2 + 0) / 3 = 4/9, loss (1/6 x 2 + 0.5) / 3 = 5/18, RS 8/5 -> 100 - 100 / 2.6.
    # Then gain (4/9 x 2 + 1) / 3 = 17/27, loss (5/18 x 2) / 3 = 5/27, RS 17/5 -> 100 - 100 / 4.4.
    r = W.rsi(pd.Series([10.0, 11.0, 10.5, 11.5, 11.0, 12.0]), n=3)
    assert r.iloc[:3].isna().all()
    assert r.iloc[3:].tolist() == pytest.approx([80.0, 100 - 100 / 2.6, 100 - 100 / 4.4], abs=1e-9)


def test_rsi_matches_wilder_by_hand_over_fourteen_sessions():
    # 14 changes: ten of +1, four of -1. Seed: gain 10/14, loss 4/14, RS 2.5 -> 100 - 100 / 3.5.
    # One more -1: gain 10/14 x 13 / 14 = 130/196, loss (4/14 x 13 + 1) / 14 = 66/196 -> 100 - 100 x 66/196.
    steps = [1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1]
    closes = pd.Series(100.0 + np.concatenate([[0.0], np.cumsum(steps)]),
                       index=pd.bdate_range("2019-01-01", periods=len(steps) + 1))
    r = W.rsi(closes)
    assert r.iloc[:14].isna().all()
    assert r.iloc[14] == pytest.approx(100 - 100 / 3.5, abs=1e-9)
    assert r.iloc[15] == pytest.approx(100 - 100 * 66 / 196, abs=1e-9)
    assert (r.index == closes.index).all()


def test_rsi_is_100_with_no_losses_and_50_on_a_flat_stock():
    assert W.rsi(pd.Series(np.arange(1.0, 21.0))).iloc[-1] == 100.0
    assert W.rsi(pd.Series(np.full(20, 50.0))).iloc[-1] == 50.0


def test_rsi_uses_no_later_close():
    closes = pd.Series(50.0 + np.random.default_rng(7).normal(0, 1, 120).cumsum(),
                       index=pd.bdate_range("2019-01-01", periods=120))
    full = W.rsi(closes)
    for k in (20, 47, 90):
        assert W.rsi(closes.iloc[:k + 1]).iloc[-1] == pytest.approx(full.iloc[k], abs=1e-12)
        later = closes.copy()
        later.iloc[k + 1:] = later.iloc[k + 1:] * 3.0
        assert W.rsi(later).iloc[:k + 1].equals(full.iloc[:k + 1])


# --------------------------------------------------------------------------
# no look-ahead in the decision
# --------------------------------------------------------------------------

def test_a_crash_after_the_entry_does_not_change_the_entry_decision():
    up = make_market(path=PATH)
    crash = make_market(path={**PATH, "2019-01-23": 40.0})        # the day after the first entry
    assert W.rsi(up["stock"]["close"])[pd.Timestamp("2019-01-23")] > 70
    assert W.rsi(crash["stock"]["close"])[pd.Timestamp("2019-01-23")] < 30   # a next-day read would sell
    a = W.run(up, start=START, end=END, manage={"rsi_put_max": 70})
    b = W.run(crash, start=START, end=END, manage={"rsi_put_max": 70})
    assert cycle(a, "2019-01-22")["status"] == cycle(b, "2019-01-22")["status"] == "skipped: RSI 100.0 above 70"
    assert cycle(a, "2019-01-22")["rsi"] == cycle(b, "2019-01-22")["rsi"]
    assert not sold_on(b, "2019-01-22")


def test_the_filter_reads_the_entry_sessions_own_close():
    # a jump on the entry session itself is part of that session's RSI
    flat = W.run(make_market(), start=START, end=END, manage={"rsi_put_max": 70})
    jump = W.run(make_market(path={"2019-01-22": 55.0}), start=START, end=END, manage={"rsi_put_max": 70})
    assert cycle(flat, "2019-01-22")["cp"] == "P" and cycle(flat, "2019-01-22")["rsi"] == 50.0
    assert cycle(jump, "2019-01-22")["status"].startswith("skipped: RSI 100.0")


# --------------------------------------------------------------------------
# the rules, exactly at their thresholds
# --------------------------------------------------------------------------

def test_a_put_is_skipped_exactly_when_rsi_is_above_70(monkeypatch):
    seen = []
    values = {"2019-01-22": 70.0, "2019-02-19": 70.0001, "2019-03-18": 95.0, "2019-04-22": 30.0, "2019-05-20": 69.99}
    fake_rsi(monkeypatch, values, seen)
    res = W.run(make_market(), start=START, end=END, manage={"rsi_put_max": 70})      # flat: puts only
    assert seen == [14]
    for day, v in values.items():
        c = cycle(res, day)
        assert c["shares_before"] == 0 and c["rsi"] == v
        if v > 70:
            assert c["status"] == f"skipped: RSI {v:.1f} above 70" and not sold_on(res, day)
        else:
            assert c["cp"] == "P" and len(sold_on(res, day)) == 1
    assert W.summary(res, make_market()["tbill"])["filtered"] == 2


def test_a_call_is_skipped_exactly_when_rsi_is_below_30_and_the_shares_wait(monkeypatch):
    values = {"2019-01-22": 10.0, "2019-02-19": 30.0, "2019-03-18": 29.99, "2019-04-22": 5.0, "2019-05-20": 31.0}
    fake_rsi(monkeypatch, values)
    m = make_market(path={"2019-02-14": 45.0})                   # assigned at the Feb expiry, then flat
    res = W.run(m, start=START, end=END, manage={"rsi_call_min": 30})
    assert cycle(res, "2019-01-22")["cp"] == "P"                  # the call rule leaves puts alone
    assert "rsi" not in cycle(res, "2019-01-22")
    for day in ("2019-02-19", "2019-03-18", "2019-04-22", "2019-05-20"):
        c, v = cycle(res, day), values[day]
        assert c["shares_before"] > 0 and c["rsi"] == v
        if v < 30:
            assert c["status"] == f"skipped: RSI {v:.1f} below 30, shares left uncovered" and not sold_on(res, day)
        else:
            assert c["cp"] == "C" and len(sold_on(res, day)) == 1
    led = res["ledger"].set_index("date")
    waiting = led.loc["2019-03-18":"2019-05-17"]
    assert waiting["short"].isna().all() and (waiting["shares"] == led.at[pd.Timestamp("2019-03-15"), "shares"]).all()


def test_the_put_rule_leaves_calls_alone(monkeypatch):
    fake_rsi(monkeypatch, {d: 95.0 for d in ("2019-02-19", "2019-03-18", "2019-04-22", "2019-05-20")})
    res = W.run(make_market(path={"2019-02-14": 45.0}), start=START, end=END, manage={"rsi_put_max": 70})
    assert all(cycle(res, d)["cp"] == "C" for d in ("2019-02-19", "2019-03-18", "2019-04-22", "2019-05-20"))


@pytest.mark.parametrize("manage", [{"rsi_put_max": 70}, {"rsi_call_min": 30}, {"rsi_put_max": 70, "rsi_call_min": 30}])
def test_on_a_real_path_each_rule_acts_exactly_when_rsi_crosses_its_line(manage):
    m = make_market(path=PATH)
    r = W.rsi(m["stock"]["close"])
    res = W.run(m, start=START, end=END, manage=manage)
    acted = {"P": [], "C": []}
    for day in ENTRIES:
        c, v = cycle(res, day), float(r[pd.Timestamp(day)])
        phase = "P" if c["shares_before"] == 0 else "C"
        assert phase == ("P" if day in PUT_ENTRIES else "C")
        rule = manage.get("rsi_put_max") if phase == "P" else manage.get("rsi_call_min")
        blocked = rule is not None and (v > rule if phase == "P" else v < rule)
        assert c.get("status", "").startswith("skipped: RSI") == blocked
        assert bool(sold_on(res, day)) == (not blocked)
        if rule is not None:
            assert c["rsi"] == pytest.approx(v, abs=1e-12)
            acted[phase].append(blocked)
    for phase, rule in (("P", "rsi_put_max"), ("C", "rsi_call_min")):
        if rule in manage:
            assert set(acted[phase]) == {True, False}          # the path tests both sides of the line


# --------------------------------------------------------------------------
# off unless asked for
# --------------------------------------------------------------------------

def test_without_either_option_run_is_unchanged_and_never_computes_rsi(monkeypatch):
    m = make_market(path=PATH)
    plain = W.run(m, start=START, end=END)

    def boom(*a, **k):
        raise AssertionError("rsi() was called with no RSI option set")

    monkeypatch.setattr(W, "rsi", boom)
    for manage in (None, {}, {"rsi_put_max": None, "rsi_call_min": None}):
        again = W.run(m, start=START, end=END, manage=manage)
        assert again["blotter"] == plain["blotter"] and again["cycles"] == plain["cycles"]
        pd.testing.assert_frame_equal(again["ledger"], plain["ledger"])
    assert all("rsi" not in c for c in plain["cycles"])


def test_thresholds_that_never_bind_trade_exactly_as_the_plain_wheel():
    m = make_market(path=PATH)
    plain = W.run(m, start=START, end=END)
    loose = W.run(m, start=START, end=END, manage={"rsi_put_max": 100, "rsi_call_min": 0})
    assert loose["blotter"] == plain["blotter"]
    pd.testing.assert_frame_equal(loose["ledger"], plain["ledger"])
    assert [{k: v for k, v in c.items() if k != "rsi"} for c in loose["cycles"]] == plain["cycles"]


def test_the_filter_refuses_to_run_without_enough_earlier_closes():
    m = make_market()                       # closes from 2018-12-03; the first entry is then 2018-12-24
    with pytest.raises(ValueError, match="RSI filter needs 30 closes before 2018-12-24, and has 15"):
        W.run(m, start="2018-12-03", end=END, manage={"rsi_call_min": 30})
    W.run(m, start="2018-12-03", end=END)   # the plain wheel does not need them
