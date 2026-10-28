"""scripts/run_portfolio.py: adding the books' daily account values into one portfolio."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_portfolio.py"
spec = importlib.util.spec_from_file_location("run_portfolio", SCRIPT)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def ledger(days, navs):
    return pd.DataFrame({"date": pd.to_datetime(days), "nav": navs})


def test_the_portfolio_is_the_sum_of_the_books_on_each_session():
    out = P.combine([ledger(["2026-01-02", "2026-01-05"], [100.0, 110.0]),
                     ledger(["2026-01-02", "2026-01-05"], [100.0, 90.0])])
    assert list(out["nav"]) == [200.0, 200.0]


def test_a_session_one_book_lacks_carries_its_last_value():
    out = P.combine([ledger(["2026-01-02", "2026-01-05", "2026-01-06"], [100.0, 101.0, 102.0]),
                     ledger(["2026-01-02", "2026-01-06"], [100.0, 105.0])])
    assert list(out["nav"]) == [200.0, 201.0, 207.0]


def test_monthly_returns_start_from_the_capital():
    r = P.monthly_returns(ledger(["2026-01-30", "2026-02-27"], [110.0, 121.0]), 100.0)
    assert list(r.round(10)) == pytest.approx([0.10, 0.10])


def test_five_identical_books_gain_nothing_from_diversification():
    days = pd.to_datetime(["2026-01-30", "2026-02-27", "2026-03-31", "2026-04-30"])
    one = ledger(days, [101.0, 99.0, 104.0, 103.0])
    books = [one] * 5
    rf = pd.Series(0.001, index=pd.PeriodIndex(days, freq="M"))
    port = P.combine(books)
    together = P.sharpe_if_together(books, 100.0, rf)
    r = P.monthly_returns(port, 500.0) - rf
    assert together == pytest.approx(r.mean() / r.std(ddof=1) * 12 ** 0.5)
