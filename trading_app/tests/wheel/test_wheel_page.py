"""
scripts/wheel_page_data.py: the evidence behind the wheel page's fill assumption
and thesis, on inputs where every answer is arithmetic.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "wheel_page_data.py"


def load():
    spec = importlib.util.spec_from_file_location("wheel_page_data", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = load()


def quotes(trade):
    bid = np.array([1.00, 2.00, 3.00, 4.00])
    ask = bid + 0.20
    return pd.DataFrame({"bid": bid, "ask": ask, "mid": (bid + ask) / 2, "trade": trade(bid, ask)})


def test_prints_at_the_mid_sit_halfway_and_inside_a_quarter_spread():
    s = P.print_stats(quotes(lambda b, a: (b + a) / 2))
    assert s["n"] == 4 and s["median_location"] == pytest.approx(0.5)
    assert s["median_abs_dev"] == pytest.approx(0.0) and s["within_quarter_pct"] == pytest.approx(100.0)
    assert s["slope"] == pytest.approx(1.0) and s["r2"] == pytest.approx(1.0)


def test_prints_at_the_bid_are_half_a_spread_from_the_mid():
    s = P.print_stats(quotes(lambda b, a: b))
    assert s["median_location"] == pytest.approx(0.0) and s["at_or_below_bid_pct"] == pytest.approx(100.0)
    assert s["dev_over_spread"] == pytest.approx(0.5) and s["within_quarter_pct"] == pytest.approx(0.0)
    assert s["outside_pct"] == pytest.approx(0.0)


def test_a_print_beyond_the_ask_counts_as_outside_the_quote():
    s = P.print_stats(quotes(lambda b, a: np.where(b == 4.0, a + 0.05, (b + a) / 2)))
    assert s["outside_pct"] == pytest.approx(25.0) and s["within_quarter_pct"] == pytest.approx(75.0)


def test_realised_vol_is_zero_for_a_steady_climb_and_known_for_alternating_moves():
    days = pd.bdate_range("2026-01-02", periods=21)
    steady = pd.Series(100 * np.exp(0.001 * np.arange(21)), index=days)
    assert P.realised_vol(steady, days[0], days[-1]) == pytest.approx(0.0, abs=1e-12)
    a = 0.01
    zigzag = pd.Series(100 * np.exp(np.cumsum([0] + [a, -a] * 10)), index=days)
    r = np.array([a, -a] * 10)
    assert P.realised_vol(zigzag, days[0], days[-1]) == pytest.approx(r.std(ddof=1) * np.sqrt(252))


def test_the_page_data_is_plain_json():
    rows = P.jsonable([{"date": pd.Timestamp("2026-01-02"), "x": np.float64(1.5), "n": np.int64(2),
                        "gone": np.nan, "none": None, "nat": pd.NaT, "na": pd.NA}])
    assert json.loads(json.dumps(rows)) == [{"date": "2026-01-02", "x": 1.5, "n": 2, "gone": None, "none": None,
                                             "nat": None, "na": None}]
