"""
Pricing-math tests.

The forward, the discount factor and the implied vols are all *derived* — no
rate is assumed anywhere — so an error in the derivation would propagate
silently into the vol surface and into the price-space-vs-vol-space
comparison. These tests build a world where the right answer is known by
construction and check that the code recovers it.

    python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_app.lib.vol import (  # noqa: E402
    bs_price, implied_forward, implied_vol,
)

F_TRUE, D_TRUE, T_TRUE, VOL_TRUE = 13.40, 0.9965, 30 / 365.0, 0.62


def test_put_call_parity_holds_in_the_pricer():
    """C - P = D*(F - K) must fall out of bs_price itself, not be imposed."""
    for K in (11.0, 13.0, 13.5, 16.0):
        c = bs_price(F_TRUE, K, T_TRUE, VOL_TRUE, D_TRUE, "C")
        p = bs_price(F_TRUE, K, T_TRUE, VOL_TRUE, D_TRUE, "P")
        assert c - p == pytest.approx(D_TRUE * (F_TRUE - K), abs=1e-10)


@pytest.mark.parametrize("cp", ["C", "P"])
@pytest.mark.parametrize("K", [10.0, 12.5, 13.4, 15.0, 18.0])
@pytest.mark.parametrize("vol", [0.20, 0.62, 1.40])
def test_implied_vol_round_trips(cp, K, vol):
    px = bs_price(F_TRUE, K, T_TRUE, vol, D_TRUE, cp)
    if not np.isfinite(px) or px <= 1e-8:
        pytest.skip("price numerically zero; vol is not identifiable")
    assert implied_vol(px, F_TRUE, K, T_TRUE, D_TRUE, cp) == pytest.approx(vol, abs=1e-5)


@pytest.mark.parametrize("bad_price,cp", [
    (0.0, "C"),        # zero
    (-1.0, "C"),       # negative
    (0.10, "C"),       # far below intrinsic for F=13.4, K=10
    (99.0, "C"),       # above the forward: no vol reproduces it
])
def test_unpriceable_marks_return_nan_not_a_boundary_vol(bad_price, cp):
    """
    Clamping an impossible mark to 0.1% or 500% vol would launder a bad quote
    into a plausible surface point. It must come back NaN.
    """
    assert np.isnan(implied_vol(bad_price, F_TRUE, 10.0, T_TRUE, D_TRUE, cp))


def test_zero_dte_has_no_implied_vol():
    assert np.isnan(implied_vol(1.0, F_TRUE, 13.0, 0.0, D_TRUE, "C"))


def _synthetic_chain(F=F_TRUE, D=D_TRUE, strikes=(11.0, 12.0, 13.0, 14.0, 15.0)):
    """A clean two-sided chain priced off a known forward."""
    rows = []
    for K in strikes:
        for cp in ("C", "P"):
            rows.append({
                "date": pd.Timestamp("2026-08-03"),
                "expiry": pd.Timestamp("2026-09-02"),
                "dte": 30, "strike": K, "cp": cp, "spot": 13.35,
                "MID_PRICE": bs_price(F, K, T_TRUE, VOL_TRUE, D, cp),
            })
    return pd.DataFrame(rows)


def test_implied_forward_recovers_the_forward_and_discount():
    """
    The whole point: F and D come out of the option prices by regression, with
    no rate and no dividend assumed anywhere.
    """
    fwd = implied_forward(_synthetic_chain())
    assert len(fwd) == 1
    row = fwd.iloc[0]
    assert row["F"] == pytest.approx(F_TRUE, abs=1e-6)
    assert row["D"] == pytest.approx(D_TRUE, abs=1e-6)
    assert row["resid_max"] < 1e-8          # parity is an identity, not a fit
    assert row["n_pairs"] == 5


def test_implied_forward_needs_both_rights():
    """A calls-only chain — the state this project was in — yields no forward."""
    calls = _synthetic_chain()
    calls = calls[calls["cp"] == "C"]
    assert implied_forward(calls).empty


def test_implied_forward_skips_thin_expiries():
    thin = _synthetic_chain(strikes=(13.0,))
    assert implied_forward(thin, min_pairs=3).empty
