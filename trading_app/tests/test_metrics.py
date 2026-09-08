"""
Tests for the statistics the page actually prints.

`metrics.py` had no tests, and it is the module holding the two numbers the
assignment requires. It is also where the fill-location bug lived: the
percentages were computed over the prints that landed inside the quote while
the `n` and the histogram bars printed beside them counted every print, so a
third of a percentage point of drift was invisible and nothing would have
caught it regressing.

Every case here builds a frame whose right answer is known by construction, so
a failure means the statistic changed, not that the data moved.

    python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_app.lib.loaders import MARK_FIELD, PRINT_FIELD  # noqa: E402
from trading_app.lib.metrics import (  # noqa: E402
    arbitrage_audit, infer_strike_step, interpolation_holdout, print_probability,
    sparsity_stats, spread_by_bucket, trade_position_histogram,
)

DATE = pd.Timestamp("2026-07-10")
EXPIRY = pd.Timestamp("2026-08-21")


def _wide(rows: list[dict]) -> pd.DataFrame:
    """A wide frame with the columns build_frames guarantees downstream."""
    df = pd.DataFrame(rows)
    df["date"] = df.get("date", DATE)
    df["expiry"] = df.get("expiry", EXPIRY)
    df["cp"] = df.get("cp", "C")
    df["dte"] = df.get("dte", 42)
    for col in (MARK_FIELD, PRINT_FIELD, "BID", "ASK", "spot", "strike"):
        if col not in df:
            df[col] = np.nan
    df["ric"] = df.get("ric", [f"R{i}" for i in range(len(df))])
    df["has_mark"] = df[MARK_FIELD].notna()
    df["has_print"] = df[PRINT_FIELD].notna()
    df["abs_diff"] = (df[MARK_FIELD] - df[PRINT_FIELD]).abs()
    df["rel_diff"] = df["abs_diff"] / df[MARK_FIELD].replace(0, np.nan)
    spread = df["ASK"] - df["BID"]
    df["spread"] = spread.where(spread >= 0)
    df["spread_pct"] = 100.0 * df["spread"] / df[MARK_FIELD].replace(0, np.nan)
    df["trade_in_spread"] = (df[PRINT_FIELD] - df["BID"]) / df["spread"].replace(0, np.nan)
    df["moneyness"] = df["strike"] / df["spot"].replace(0, np.nan)
    return df


# ---------------------------------------------------------------- requirement 5

def test_the_two_required_numbers():
    """Four series: two marked-only, one with both, one print-only."""
    w = _wide([
        {"strike": 10.0, MARK_FIELD: 1.00},
        {"strike": 10.5, MARK_FIELD: 1.20},
        {"strike": 11.0, MARK_FIELD: 1.50, PRINT_FIELD: 1.40},
        {"strike": 11.5, PRINT_FIELD: 0.80},
    ])
    s = sparsity_stats(w)
    assert s["n_series"] == 4
    assert s["n_mark_only"] == 2
    assert s["pct_mark_no_trade"] == pytest.approx(50.0)
    # only the one series carrying BOTH fields feeds the median
    assert s["median_abs_diff"] == pytest.approx(0.10)


def test_a_row_with_only_a_quote_does_not_enlarge_the_denominator():
    """
    ASK is quoted on more series than MID_PRICE. A row carrying neither of the
    two fields the assignment is about must not dilute the percentage.
    """
    base = [{"strike": 10.0, MARK_FIELD: 1.0}, {"strike": 10.5, MARK_FIELD: 1.2, PRINT_FIELD: 1.1}]
    plain = sparsity_stats(_wide(base))
    padded = sparsity_stats(_wide(base + [{"strike": 11.0, "ASK": 2.0, "BID": 1.0}]))
    assert padded["n_series"] == plain["n_series"] == 2
    assert padded["pct_mark_no_trade"] == pytest.approx(plain["pct_mark_no_trade"])


def test_empty_frame_returns_stats_not_an_exception():
    s = sparsity_stats(_wide([]).iloc[0:0])
    assert s["n_series"] == 0 and s["pct_mark_no_trade"] is None


# ------------------------------------------------------------- fill location

def _fills(positions: list[float]) -> pd.DataFrame:
    """Prints placed at chosen fractions of a $1.00-wide quote around $1.50."""
    return _wide([
        {"strike": 10.0 + i, MARK_FIELD: 1.5, "BID": 1.0, "ASK": 2.0,
         PRINT_FIELD: 1.0 + p}
        for i, p in enumerate(positions)
    ])


def test_fill_histogram_shares_one_denominator_with_its_percentages():
    """
    The regression this file exists for. `n`, the bars, and the percentages
    beside them must all count the same prints.
    """
    h = trade_position_histogram(_fills([0.0, 0.05, 0.5, 0.5, 0.95, 1.0] * 3))
    assert h["n"] == 18
    assert sum(h["counts"]) == h["n"]
    # 6 of 18 sit strictly inside (0.4, 0.6)
    assert h["pct_near_mid"] == pytest.approx(100 * 6 / 18)
    # 12 of 18 sit at or beyond the edges
    assert h["pct_at_edges"] == pytest.approx(100 * 12 / 18)


def test_prints_outside_the_quote_are_clipped_not_dropped():
    """
    A print through the bid is the strongest evidence the mid was not
    achievable. Dropping it from the denominator biases the result toward the
    mid, so it is clipped to the edge and still counted.
    """
    inside = trade_position_histogram(_fills([0.5] * 9 + [0.0]))
    through = trade_position_histogram(_fills([0.5] * 9 + [-0.2]))
    assert through["n"] == inside["n"] == 10
    assert through["pct_near_mid"] == pytest.approx(inside["pct_near_mid"]) == pytest.approx(90.0)
    assert through["pct_at_edges"] == pytest.approx(10.0)


def test_prints_far_outside_the_quote_are_excluded_entirely():
    h = trade_position_histogram(_fills([0.5] * 10 + [3.0, -2.0]))
    assert h["n"] == 10


# --------------------------------------------------------------- arbitrage

def _ladder(mids: list[float], cp: str = "C", step: float = 0.5,
            bids=None, asks=None) -> pd.DataFrame:
    strikes = [10.0 + i * step for i in range(len(mids))]
    return _wide([
        {"strike": k, MARK_FIELD: m, "cp": cp, "spot": 12.0,
         "BID": (bids[i] if bids else m - 0.05),
         "ASK": (asks[i] if asks else m + 0.05)}
        for i, (k, m) in enumerate(zip(strikes, mids))
    ])


def test_a_convex_call_ladder_has_no_violations():
    a = arbitrage_audit(_ladder([2.00, 1.55, 1.20, 0.95, 0.78]), strike_step=0.5, cp="C")
    assert a["n_bfly"] == 3 and a["bfly_mid"] == 0
    assert a["n_mono"] == 4 and a["mono_mid"] == 0


def test_a_concave_middle_is_caught_with_the_right_magnitude():
    # 2.00, 1.90, 1.20 -> 2.00 - 2*1.90 + 1.20 = -0.60
    a = arbitrage_audit(_ladder([2.00, 1.90, 1.20, 0.95]), strike_step=0.5, cp="C")
    assert a["bfly_mid"] == 1
    assert a["worst_bfly"] == pytest.approx(-0.60)


def test_put_monotonicity_uses_the_flipped_sign():
    """
    Puts must RISE in strike. Running the call inequality on a well-behaved put
    ladder would report every strike as a violation.
    """
    rising = arbitrage_audit(_ladder([0.20, 0.45, 0.80, 1.30], cp="P"), strike_step=0.5, cp="P")
    assert rising["n_mono"] == 3 and rising["mono_mid"] == 0
    falling = arbitrage_audit(_ladder([1.30, 0.80, 0.45, 0.20], cp="P"), strike_step=0.5, cp="P")
    assert falling["mono_mid"] == 3


def test_a_missing_strike_cannot_fabricate_a_violation():
    """Only strictly consecutive strikes are compared."""
    w = _ladder([2.00, 1.55, 1.20])                  # strikes 10.0, 10.5, 11.0
    w.loc[w["strike"] == 10.5, "strike"] = 12.0      # -> 10.0, 11.0, 12.0
    a = arbitrage_audit(w, strike_step=0.5, cp="C")
    assert a["n_bfly"] == 0 and a["n_mono"] == 0

    # and one surviving consecutive pair is still compared
    w2 = _ladder([2.00, 1.55, 1.20])
    w2.loc[w2["strike"] == 10.0, "strike"] = 8.0     # -> 8.0, 10.5, 11.0
    a2 = arbitrage_audit(w2, strike_step=0.5, cp="C")
    assert a2["n_bfly"] == 0 and a2["n_mono"] == 1


def test_executable_prices_clear_a_violation_that_the_mid_shows():
    """A butterfly negative at the mid but positive to transact is not free money."""
    a = arbitrage_audit(
        _ladder([2.00, 1.90, 1.20, 0.95],
                bids=[1.70, 1.60, 0.90, 0.65], asks=[2.30, 2.20, 1.50, 1.25]),
        strike_step=0.5, cp="C")
    assert a["bfly_mid"] == 1
    assert a["bfly_exec"] == 0


def test_strike_step_is_read_off_the_data():
    """Hardcoding $0.50 silently tested nothing on the $1.00-grid control."""
    assert infer_strike_step(_ladder([2.0, 1.5, 1.1, 0.8], step=1.0)) == pytest.approx(1.0)
    assert infer_strike_step(_ladder([2.0, 1.5, 1.1, 0.8], step=0.5)) == pytest.approx(0.5)


# ------------------------------------------------------------- interpolation

def _grid(fn) -> pd.DataFrame:
    return _wide([
        {"strike": k, "dte": d, MARK_FIELD: fn(k), "spot": 12.0}
        for k in np.arange(9.0, 15.01, 0.5) for d in (7, 21, 42)
    ])


def test_linear_data_is_interpolated_exactly_and_unbiased():
    h = interpolation_holdout(_grid(lambda k: 3.0 - 0.2 * k))
    assert h["median_abs_err"] == pytest.approx(0.0, abs=1e-9)
    assert h["bias"] == pytest.approx(0.0, abs=1e-9)


def test_convex_data_produces_the_positive_bias_the_page_claims():
    """The chord between two observed strikes sits above a convex curve."""
    h = interpolation_holdout(_grid(lambda k: 0.05 * (k - 12.0) ** 2 + 0.5))
    assert h["bias"] > 0
    assert h["median_abs_err"] > 0


# ------------------------------------------------------------------ buckets

def test_moneyness_labels_are_reversed_for_puts():
    """
    K/S < 1 is in the money for a call and out of the money for a put. The same
    bin edges therefore carry opposite labels.
    """
    rows = [{"strike": k, "dte": 30, "spot": 12.0, MARK_FIELD: 1.0,
             PRINT_FIELD: 1.0 if i % 2 else np.nan, "cp": cp}
            for cp in ("C", "P")
            for i, k in enumerate(np.arange(6.0, 18.01, 0.20))]
    w = _wide(rows)
    calls = print_probability(w, cp="C")
    puts = print_probability(w, cp="P")
    assert calls is not None and puts is not None
    # the lowest K/S bucket is deep ITM for a call and deep OTM for a put
    low_call = next(r for r in calls["marginal"] if r["label"] == "deep ITM")
    low_put = next(r for r in puts["marginal"] if r["label"] == "deep OTM")
    assert low_call["n"] == low_put["n"] > 0


def test_spread_buckets_split_on_whether_the_contract_traded():
    w = _wide([
        {"strike": k, "spot": 12.0, MARK_FIELD: 1.0, "BID": 0.9, "ASK": 1.1,
         PRINT_FIELD: (1.0 if traded else np.nan)}
        for k in np.arange(8.0, 16.01, 0.5) for traded in (True, False)
    ])
    b = spread_by_bucket(w)
    assert b is not None
    assert len(b["traded"]) == len(b["untraded"]) == len(b["moneyness"])
    assert sum(b["counts"]) == len(w)
