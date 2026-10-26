"""
scripts/premium_screen.py (config/premium_screen.yaml): the volatility-premium measure and the
pick's feasibility, on series where every answer is arithmetic.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "premium_screen.py"


def load():
    spec = importlib.util.spec_from_file_location("premium_screen", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DAYS = pd.bdate_range("2014-01-01", "2014-06-30")
START, END = pd.Timestamp("2014-01-01"), pd.Timestamp("2014-04-30")


def swinging_close(r: float = 0.01) -> pd.Series:
    """Daily log returns alternating +r and -r: realised vol is r x sqrt(252) up to the sample correction."""
    steps = np.where(np.arange(len(DAYS)) % 2 == 0, r, -r)
    return pd.Series(100.0 * np.exp(np.cumsum(steps)), index=DAYS)


def implied(level: float, days=DAYS) -> pd.DataFrame:
    return pd.DataFrame({"30D_A_IM_C": level, "30D_A_IM_P": level}, index=days)


def test_the_premium_is_implied_minus_realised_over_implied():
    S = load()
    close = swinging_close()
    out = S.measure(implied(20.0), close, START, END)
    # each month: the 21 returns after the entry close alternate +-0.01
    first = [d for d in DAYS if START <= d <= END]
    entries = pd.Series(first).groupby([pd.Series(first).dt.year, pd.Series(first).dt.month]).first()
    rv = []
    logret = np.log(close).diff()
    for e in entries:
        i = list(close.index).index(e)
        rv.append(float(logret.iloc[i + 1:i + 22].std(ddof=1) * np.sqrt(252) * 100))
    assert out["months"] == 4
    assert out["mean_ratio"] == pytest.approx(np.mean([(20.0 - v) / 20.0 for v in rv]))
    assert out["mean_points"] == pytest.approx(np.mean([20.0 - v for v in rv]))
    assert out["coverage"] == 1.0


def test_the_entry_is_the_first_day_of_the_month_with_an_implied_vol():
    S = load()
    iv = implied(20.0)
    iv.loc[pd.Timestamp("2014-02-03")] = np.nan          # 3 Feb has no implied vol, so February enters on 4 Feb
    iv.loc[pd.Timestamp("2014-02-04")] = 30.0
    out = S.measure(iv, swinging_close(), START, END)
    assert out["entries"][1] == "2014-02-04"
    assert out["coverage"] < 1.0


def test_a_stock_missing_implied_vol_on_more_than_five_percent_of_days_is_not_eligible():
    S = load()
    iv = implied(20.0)
    iv.iloc[::10] = np.nan                                 # 10% of days
    out = S.measure(iv, swinging_close(), START, END)
    assert not S.eligible(out, months_needed=4)
    assert S.eligible(S.measure(implied(20.0), swinging_close(), START, END), months_needed=4)


def test_feasibility_flags_a_split_and_a_price_above_950():
    S = load()
    days = pd.bdate_range("2018-06-01", periods=10)
    raw = pd.Series(100.0, index=days)
    assert S.feasible(raw.copy(), raw)["feasible"]
    # a 2:1 split on the sixth day: the default series is 50 throughout, the unadjusted one 100 then 50
    adjusted = pd.Series(50.0, index=days)
    unadjusted = pd.Series(np.where(days < days[5], 100.0, 50.0), index=days)
    f = S.feasible(adjusted, unadjusted)
    assert not f["feasible"] and "adjusted" in f["why"]
    high = raw.copy()
    high[days[-1]] = 960.0
    f = S.feasible(high, high)
    assert not f["feasible"] and "950" in f["why"]
