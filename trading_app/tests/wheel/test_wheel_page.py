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


def test_an_expiry_and_an_assignment_carry_the_contract_they_close():
    res = {"blotter": [
        {"date": pd.Timestamp("2026-01-20"), "kind": "put", "side": "SELL", "qty": 200, "price": 4.05,
         "cash_delta": 810.0, "note": "", "instrument": "MAN202649000.U^N26", "cp": "P", "strike": 490.0,
         "expiry": pd.Timestamp("2026-02-20"), "bid": 3.75, "ask": 4.35},
        {"date": pd.Timestamp("2026-02-20"), "kind": "put", "side": "ASSIGN", "qty": 200, "price": 490.0,
         "cash_delta": 0.0, "note": "", "instrument": "MA"},
        {"date": pd.Timestamp("2026-02-20"), "kind": "stock", "side": "BUY", "qty": 200, "price": 490.0,
         "cash_delta": -98000.0, "note": "", "instrument": "MA"}]}
    rows = P.blotter_rows(res)
    assert rows[1]["occ"] == rows[0]["occ"] == "MA    260220P00490000" and rows[1]["contracts"] == 2
    assert rows[2]["occ"] is None and rows[2]["contracts"] is None


def test_an_index_benchmark_starts_at_the_capital_and_moves_with_the_index():
    levels = pd.Series([200.0, 210.0, 189.0, 220.5], index=pd.to_datetime(["2026-01-02", "2026-01-05",
                                                                           "2026-01-06", "2026-01-07"]))
    sessions = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"])
    book = P.index_book(levels, sessions, 100_000.0)
    assert list(book["nav"]) == pytest.approx([100_000.0, 105_000.0, 94_500.0, 110_250.0])


def test_an_index_benchmark_carries_its_last_level_over_a_missing_session():
    levels = pd.Series([200.0, 220.0], index=pd.to_datetime(["2026-01-02", "2026-01-06"]))
    sessions = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    assert list(P.index_book(levels, sessions, 100_000.0)["nav"]) == pytest.approx([100_000.0, 100_000.0, 110_000.0])


def test_the_cboe_file_is_read_as_dated_closing_levels(tmp_path):
    f = tmp_path / "PUT_History.csv"
    f.write_text("DATE,PUT\n01/02/2026,3341.32\n01/05/2026,3350.00\n")
    s = P.read_cboe(f)
    assert list(s.index) == list(pd.to_datetime(["2026-01-02", "2026-01-05"])) and s.iloc[-1] == 3350.0


def test_trade_locations_fall_into_spread_bins_with_the_quote_edges_inside():
    # 0 is the bid, 1 the ask; a print at the bid or the ask is inside the quote
    loc = np.array([-0.3, 0.0, 0.05, 0.5, 0.55, 0.95, 1.0, 1.4])
    b = P.location_bins(loc)
    assert b["labels"][0] == "below the bid" and b["labels"][-1] == "above the ask"
    assert len(b["counts"]) == 12 and sum(b["counts"]) == len(loc)
    assert b["counts"][0] == 1 and b["counts"][-1] == 1            # one each side of the quote
    assert b["counts"][1] == 2                                     # 0.0 and 0.05: from the bid to a tenth of the way
    assert b["counts"][6] == 2                                     # 0.5 and 0.55: the bin starting at the mid
    assert b["counts"][10] == 2                                    # 0.95 and 1.0: the last tenth, the ask included
    assert sum(b["pct"]) == pytest.approx(100.0)


def account(days, values):
    return pd.DataFrame({"date": pd.bdate_range(days, periods=len(values)), "nav": values})


def test_one_fold_joined_is_that_fold():
    from trading_app.lib.pmcc_daily import performance
    bill = pd.Series(0.04, index=pd.bdate_range("2025-01-01", "2026-12-31"))
    f = account("2025-01-02", np.linspace(100_000, 108_000, 250))
    a, b = P.join_folds([f], 100_000, bill), performance(f, 100_000, bill)
    for k in ("return_pct", "sharpe", "max_drawdown_pct", "months"):
        assert a[k] == pytest.approx(b[k])


def test_two_folds_compound_their_returns_and_put_their_months_end_to_end():
    from trading_app.lib.pmcc_daily import performance
    bill = pd.Series(0.04, index=pd.bdate_range("2025-01-01", "2026-12-31"))
    rng = np.random.default_rng(0)
    f1 = account("2025-01-02", 100_000 * np.cumprod(1 + rng.normal(0.0004, 0.004, 255)))
    f2 = account("2026-01-02", 100_000 * np.cumprod(1 + rng.normal(0.0004, 0.004, 200)))
    j = P.join_folds([f1, f2], 100_000, bill)
    p1, p2 = performance(f1, 100_000, bill), performance(f2, 100_000, bill)
    assert j["months"] == p1["months"] + p2["months"]
    assert j["return_pct"] == pytest.approx(100 * ((1 + p1["return_pct"] / 100) * (1 + p2["return_pct"] / 100) - 1))
    # each month's return is the fold's own: the second fold's first month is measured from where the first ended
    assert list(j["monthly_returns"].values()) == pytest.approx(
        list(p1["monthly_returns"].values()) + list(p2["monthly_returns"].values()))
    assert j["max_drawdown_pct"] <= min(p1["max_drawdown_pct"], p2["max_drawdown_pct"]) + 1e-12


def test_folds_that_overlap_are_refused():
    bill = pd.Series(0.04, index=pd.bdate_range("2025-01-01", "2026-12-31"))
    with pytest.raises(ValueError):
        P.join_folds([account("2025-01-02", [100_000.0] * 30), account("2025-01-20", [100_000.0] * 30)], 100_000, bill)


def test_the_tie_count_reads_four_decimals():
    grid = [{"sharpe": 2.68804}, {"sharpe": 2.68801}, {"sharpe": 2.6879}, {"sharpe": 1.5}]
    assert len(P.tied_at_top(grid)) == 2
