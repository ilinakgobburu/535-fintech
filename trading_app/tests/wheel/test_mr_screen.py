"""
scripts/mr_screen.py (config/mr_screen.yaml): the mean-reversion measures, the ranking and the
feasibility checks, on series where every answer is arithmetic.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "mr_screen.py"


def load():
    spec = importlib.util.spec_from_file_location("mr_screen", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load()
START, END = "2014-01-01", "2023-12-31"


def daily_from_monthly(month_returns, first=100.0, start="2013-12-01"):
    """Daily closes, flat within each month, whose month-end closes have these log returns."""
    months = pd.period_range(pd.Period(start, "M"), periods=len(month_returns) + 1, freq="M")
    level = first * np.exp(np.concatenate([[0.0], np.cumsum(month_returns)]))
    days, closes = [], []
    for p, v in zip(months, level):
        for d in pd.bdate_range(p.start_time, p.end_time):
            days.append(d)
            closes.append(v)
    return pd.Series(closes, index=pd.DatetimeIndex(days))


def test_monthly_returns_are_log_changes_between_month_end_closes():
    r = S.monthly_returns(daily_from_monthly(np.full(120, 0.01)), START, END)
    assert len(r) == 120 and r.index[0] == pd.Period("2014-01", "M")
    assert np.allclose(r.to_numpy(), 0.01)


def test_a_missing_month_breaks_the_returns_either_side_of_it():
    close = daily_from_monthly(np.full(120, 0.01))
    close = close[close.index.to_period("M") != pd.Period("2016-06", "M")]
    r = S.monthly_returns(close, START, END)
    assert r.isna().sum() == 2 and S.measures(r)["months"] == 118


def test_alternating_months_reverse_completely():
    # +a, -a, +a, ...: each 3-month sum is +-a, so VR is about a third; rho is -1
    a = 0.05
    r = S.monthly_returns(daily_from_monthly(np.tile([a, -a], 60)), START, END)
    m = S.measures(r)
    assert m["rho"] == pytest.approx(-1.0, abs=1e-9)
    assert m["vr"] == pytest.approx((118 / 117) / (3 * 120 / 119), abs=1e-9)


def test_a_random_walk_has_a_variance_ratio_near_one():
    rng = np.random.default_rng(7)
    vrs = []
    for _ in range(200):
        r = pd.Series(rng.normal(0, 0.06, 120))
        vrs.append(S.measures(r)["vr"])
    assert np.mean(vrs) == pytest.approx(1.0, abs=0.05)
    assert 0.10 < np.std(vrs) < 0.18                     # the registered noise note: about 0.14


def test_a_trending_stock_is_dropped_and_the_rest_rank_by_variance_ratio():
    table = {"TREND": {"months": 120, "vr": 0.60, "rho": +0.20},
             "B": {"months": 120, "vr": 0.80, "rho": -0.10},
             "A": {"months": 120, "vr": 0.70, "rho": -0.05},
             "SHORT": {"months": 100, "vr": 0.50, "rho": -0.30}}
    assert S.ranked(table) == ["A", "B"]


def run_days():
    return pd.bdate_range("2024-01-02", "2026-10-23")


def test_a_two_for_one_split_is_feasible_and_recorded():
    days = run_days()
    unadj = pd.Series(np.where(days < pd.Timestamp("2025-03-03"), 200.0, 100.0), index=days)
    default = pd.Series(100.0, index=days)
    f = S.feasible(default, unadj)
    assert f["feasible"] and f["splits"] == [("2025-03-03", 2.0)]


def test_a_spin_off_style_adjustment_is_not_feasible():
    days = run_days()
    unadj = pd.Series(100.0, index=days)
    default = pd.Series(np.where(days < pd.Timestamp("2025-03-03"), 97.0, 100.0), index=days)
    f = S.feasible(default, unadj)
    assert not f["feasible"] and "other than for a split" in f["why"]


def test_a_stock_that_stopped_trading_or_closed_above_950_is_not_feasible():
    days = run_days()
    early = pd.Series(50.0, index=days[days < pd.Timestamp("2025-07-01")])
    assert "stopped trading" in S.feasible(early, early)["why"]
    high = pd.Series(50.0, index=days)
    high.iloc[-5] = 960.0
    assert "950" in S.feasible(high, high)["why"]


def test_a_frame_with_one_column_per_ric_splits_into_series():
    idx = pd.to_datetime(["2014-01-02", "2014-01-03"])
    got = pd.DataFrame({"AAA.N": [1.0, 2.0], "BBB.OQ": [3.0, np.nan]}, index=idx)
    out = S.frame_to_series(got, ["AAA.N", "BBB.OQ", "CCC.N"])
    assert set(out) == {"AAA.N", "BBB.OQ"} and len(out["BBB.OQ"]) == 1
