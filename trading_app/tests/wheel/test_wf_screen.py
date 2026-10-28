"""
scripts/wf_screen.py's splice of a cached price series onto a newer pull: the old part up to the
cut, the new part from it, and a refusal when the two disagree where they overlap.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "wf_screen.py"
spec = importlib.util.spec_from_file_location("wf_screen", SCRIPT)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

DAYS = pd.bdate_range("2023-08-01", "2024-03-29")


def series(days, level=100.0):
    return pd.Series([level + i * 0.1 for i in range(len(days))], index=days, dtype=float)


def test_the_old_series_runs_to_the_cut_and_the_new_one_from_it():
    old = series(DAYS[DAYS <= "2023-12-29"])
    new = series(DAYS)[DAYS >= "2023-10-02"]
    out = S.extend(old, new, start="2023-10-01")
    assert out.index.is_monotonic_increasing and not out.index.duplicated().any()
    assert out.index[0] == old.index[0] and out.index[-1] == new.index[-1]
    assert (out[out.index < "2023-10-01"] == old[old.index < "2023-10-01"]).all()
    assert (out[out.index >= "2023-10-01"] == new).all()


def test_an_overlap_off_by_more_than_half_a_percent_is_refused():
    old = series(DAYS[DAYS <= "2023-12-29"])
    new = series(DAYS)[DAYS >= "2023-10-02"].copy()
    new.loc["2023-11-15"] *= 1.006
    assert S.extend(old, new, start="2023-10-01") is None
    new.loc["2023-11-15"] /= 1.006
    new *= 0.5                                               # a 2-for-1 split adjusted in since the cache
    assert S.extend(old, new, start="2023-10-01") is None


def test_no_shared_day_is_refused():
    old = series(DAYS[DAYS <= "2023-09-29"])
    new = series(DAYS)[DAYS >= "2023-10-02"]
    assert S.extend(old, new, start="2023-10-01") is None


def test_a_small_rounding_difference_is_accepted():
    old = series(DAYS[DAYS <= "2023-12-29"])
    new = series(DAYS)[DAYS >= "2023-10-02"] * 1.001
    assert S.extend(old, new, start="2023-10-01") is not None
