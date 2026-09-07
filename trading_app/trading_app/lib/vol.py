"""
The right space to interpolate in, and the price relationship that gets you
there without a model.

The holdout test in `metrics.interpolation_holdout` establishes that linearly
interpolating PRICE across a sparse strike grid misses by a few cents with a
positive bias, and attributes the bias to convexity: price is a convex
function of strike, so a straight chord drawn between two observed strikes
sits above the true curve.

If that diagnosis is right, it implies its own fix. Convexity is a property of
the price, not of the contract. Implied volatility is far flatter in strike --
that is the entire reason practitioners quote a surface in vol rather than in
dollars -- so interpolating vol and converting back to price should shrink the
error. This module makes that a measurement instead of a claim.

Two things are needed to invert Black-Scholes and neither is assumed here:

    the forward F      from put-call parity, per (date, expiry)
    the discount D     from the same regression

    C - P = D * (F - K)

is an identity for European options with no model and no volatility. Fitting
it across the strikes where both rights are quoted returns the slope -D and
the intercept D*F, so the forward and the rate fall out of the option prices
themselves. Nothing here assumes a risk-free rate, a dividend, or that the
forward equals spot -- and the gap between the fitted forward and spot is
itself reported, because on a non-dividend payer over a few weeks it ought to
be small, and if it is not, the marks are lying.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr           # standard normal CDF
from scipy.interpolate import griddata

from .loaders import MARK_FIELD, PRINT_FIELD

DAYS_PER_YEAR = 365.0
MIN_T = 1.0 / DAYS_PER_YEAR      # a zero-DTE contract has no implied vol
VOL_LO, VOL_HI = 1e-3, 5.0       # inversion bracket: 0.1% to 500% annualised


# --------------------------------------------------------------------------
# Black-76: price a European option off the forward
# --------------------------------------------------------------------------

def bs_price(F: float, K: float, T: float, sigma: float, D: float,
             cp: str = "C") -> float:
    """
    Black-76. Working off the forward rather than spot means the dividend and
    the carry are already inside F, so there is nothing left to assume.
    """
    if not (F > 0 and K > 0 and T > 0 and sigma > 0):
        return np.nan
    v = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    if cp == "C":
        return D * (F * ndtr(d1) - K * ndtr(d2))
    return D * (K * ndtr(-d2) - F * ndtr(-d1))


def implied_vol(price: float, F: float, K: float, T: float, D: float,
                cp: str = "C") -> float:
    """
    Invert bs_price for sigma, or return NaN.

    NaN is the honest answer far more often than it is a failure. A quoted mid
    below intrinsic or above the forward has NO implied volatility -- no
    positive sigma reproduces it -- and those are exactly the marks the
    arbitrage audit already flags as unexecutable. Silently clamping them to
    a boundary vol would launder bad marks into a plausible-looking surface.
    """
    if not np.isfinite([price, F, K, T, D]).all():
        return np.nan
    if T < MIN_T or price <= 0 or F <= 0 or K <= 0 or D <= 0:
        return np.nan

    # No-arbitrage bounds on the undiscounted price.
    fwd_price = price / D
    intrinsic = max(F - K, 0.0) if cp == "C" else max(K - F, 0.0)
    ceiling = F if cp == "C" else K
    if fwd_price <= intrinsic + 1e-12 or fwd_price >= ceiling - 1e-12:
        return np.nan

    f = lambda s: bs_price(F, K, T, s, D, cp) - price
    try:
        if f(VOL_LO) > 0 or f(VOL_HI) < 0:
            return np.nan
        return float(brentq(f, VOL_LO, VOL_HI, xtol=1e-8, maxiter=200))
    except Exception:
        return np.nan


# --------------------------------------------------------------------------
# Put-call parity: the forward, model-free
# --------------------------------------------------------------------------

def implied_forward(wide: pd.DataFrame, min_pairs: int = 3) -> pd.DataFrame:
    """
    Fit C - P = D*(F - K) across strikes, once per (date, expiry).

    Returns one row per (date, expiry) with the fitted forward, discount
    factor, the spot for comparison, and the residual scatter -- which is the
    part that matters. Parity is an identity, so on clean marks the residual
    should be pennies. Large residuals mean the call mid and the put mid
    disagree about the same underlying, which cannot both be right.
    """
    cols = ["date", "expiry", "dte", "n_pairs", "F", "D", "spot",
            "basis", "resid_med", "resid_max"]
    if wide is None or wide.empty:
        return pd.DataFrame(columns=cols)

    piv = (wide.dropna(subset=[MARK_FIELD])
                .pivot_table(index=["date", "expiry", "dte", "strike"],
                             columns="cp", values=MARK_FIELD, aggfunc="last")
                .reset_index())
    if "C" not in piv.columns or "P" not in piv.columns:
        return pd.DataFrame(columns=cols)
    piv = piv.dropna(subset=["C", "P"])

    rows = []
    for (d, e, dte), g in piv.groupby(["date", "expiry", "dte"]):
        if len(g) < min_pairs:
            continue
        K = g["strike"].to_numpy(float)
        y = (g["C"] - g["P"]).to_numpy(float)
        if np.ptp(K) == 0:
            continue
        slope, intercept = np.polyfit(K, y, 1)
        D = -slope
        # A few weeks out, the discount factor is within a whisker of 1. A
        # fitted D outside this band is a fit to noise, not to a rate, so we
        # pin D and take the forward from the level instead of the slope.
        if not (0.90 < D < 1.02):
            D = 1.0
            F = float(np.median(y + K))
        else:
            F = intercept / D
        resid = y - (D * (F - K))
        spot = float(g.merge(wide[["date", "spot"]].drop_duplicates("date"),
                             on="date", how="left")["spot"].iloc[0]) \
            if "spot" in wide.columns else np.nan
        rows.append({
            "date": d, "expiry": e, "dte": int(dte), "n_pairs": int(len(g)),
            "F": float(F), "D": float(D), "spot": spot,
            "basis": float(F - spot) if np.isfinite(spot) else np.nan,
            "resid_med": float(np.median(np.abs(resid))),
            "resid_max": float(np.max(np.abs(resid))),
        })
    return pd.DataFrame(rows, columns=cols)


def attach_iv(wide: pd.DataFrame, fwd: pd.DataFrame) -> pd.DataFrame:
    """
    Add F, D, T and the implied vol of the mark to every row we can invert.

    Where parity gives no forward (an expiry with too few two-sided strikes),
    fall back to F = spot, D = 1 and FLAG it, so the page can report how much
    of the surface rests on a fitted forward versus an assumed one.
    """
    out = wide.copy()
    for c in ("F", "D", "T", "iv", "fwd_fitted"):
        out[c] = np.nan
    if out.empty:
        return out

    if fwd is not None and not fwd.empty:
        key = fwd.set_index(["date", "expiry"])[["F", "D"]]
        idx = pd.MultiIndex.from_arrays([out["date"], out["expiry"]])
        out["F"] = key["F"].reindex(idx).to_numpy()
        out["D"] = key["D"].reindex(idx).to_numpy()

    out["fwd_fitted"] = out["F"].notna()
    out["F"] = out["F"].fillna(out["spot"])
    out["D"] = out["D"].fillna(1.0)
    out["T"] = out["dte"] / DAYS_PER_YEAR

    mask = (out[MARK_FIELD].notna() & out["F"].notna()
            & (out["T"] >= MIN_T) & out["strike"].notna())
    out.loc[mask, "iv"] = [
        implied_vol(p, F, K, T, D, cp)
        for p, F, K, T, D, cp in zip(
            out.loc[mask, MARK_FIELD], out.loc[mask, "F"], out.loc[mask, "strike"],
            out.loc[mask, "T"], out.loc[mask, "D"], out.loc[mask, "cp"])
    ]
    return out


def iv_coverage(wide_iv: pd.DataFrame) -> dict:
    """How much of the marked surface actually admits an implied vol."""
    out = {"n_marks": 0, "n_iv": 0, "pct": None, "n_fitted_fwd": 0,
           "pct_fitted_fwd": None, "median_iv": None,
           "iv_lo": None, "iv_hi": None}
    if wide_iv is None or wide_iv.empty or "iv" not in wide_iv:
        return out
    m = wide_iv[wide_iv[MARK_FIELD].notna() & (wide_iv["dte"] >= 1)]
    out["n_marks"] = int(len(m))
    iv = m["iv"].dropna()
    out["n_iv"] = int(len(iv))
    if len(m):
        out["pct"] = float(100.0 * len(iv) / len(m))
        out["n_fitted_fwd"] = int(m["fwd_fitted"].sum())
        out["pct_fitted_fwd"] = float(100.0 * m["fwd_fitted"].mean())
    if len(iv):
        out["median_iv"] = float(iv.median())
        out["iv_lo"] = float(iv.quantile(0.05))
        out["iv_hi"] = float(iv.quantile(0.95))
    return out


# --------------------------------------------------------------------------
# The headline: which space should you interpolate in?
# --------------------------------------------------------------------------

def space_holdout(sl: pd.DataFrame, min_points: int = 20) -> dict | None:
    """
    Run the SAME leave-one-out test twice, changing only the quantity being
    interpolated, and compare the dollar error.

        price space   hide a cell, linearly interpolate MID_PRICE from its
                      neighbours, compare to the real mid
        vol space     hide the same cell, linearly interpolate IMPLIED VOL
                      from the same neighbours, price the result back through
                      Black-76 at that cell's own K/T/F, compare to the real mid

    Everything else is held fixed: the same test cells, the same Delaunay
    triangulation, the same axis rescaling. The only difference is the space
    the straight line is drawn in, so any difference in error is attributable
    to the space and nothing else.

    Both errors are reported in dollars, because a vol error is not
    comparable to a price error and quoting one in vol points would be
    changing the units to win the argument.
    """
    cloud = sl.dropna(subset=["strike", "dte", MARK_FIELD, "iv", "F", "D", "T"])
    cloud = cloud[np.isfinite(cloud["iv"]) & (cloud["T"] >= MIN_T)]
    if len(cloud) < min_points:
        return None

    x = cloud["strike"].to_numpy(float)
    y = cloud["dte"].to_numpy(float)
    price = cloud[MARK_FIELD].to_numpy(float)
    vol = cloud["iv"].to_numpy(float)
    F = cloud["F"].to_numpy(float)
    D = cloud["D"].to_numpy(float)
    T = cloud["T"].to_numpy(float)
    cp = cloud["cp"].to_numpy(str)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None

    # Identical rescaling to metrics.interpolation_holdout, so the two tests
    # are measuring the same geometry.
    y_scale = np.ptp(x) / np.ptp(y)
    pts = np.column_stack([x, y * y_scale])

    err_price, err_vol, rows = [], [], []
    for i in range(len(price)):
        mask = np.ones(len(price), dtype=bool)
        mask[i] = False
        try:
            gp = griddata(pts[mask], price[mask], pts[i:i + 1], method="linear")
            gv = griddata(pts[mask], vol[mask], pts[i:i + 1], method="linear")
        except Exception:
            continue
        if gp is None or gv is None:
            continue
        gp, gv = float(gp[0]), float(gv[0])
        if not (np.isfinite(gp) and np.isfinite(gv)):
            continue          # outside the hull of its own neighbours
        gp_from_vol = bs_price(F[i], x[i], T[i], gv, D[i], cp[i])
        if not np.isfinite(gp_from_vol):
            continue
        err_price.append(gp - price[i])
        err_vol.append(gp_from_vol - price[i])
        rows.append({"strike": float(x[i]), "dte": float(y[i]),
                     "truth": round(float(price[i]), 4),
                     "px": round(gp, 4), "vx": round(gp_from_vol, 4)})

    if len(err_price) < 8:
        return None
    ep, ev = np.array(err_price), np.array(err_vol)
    ap, av = np.abs(ep), np.abs(ev)
    med_p, med_v = float(np.median(ap)), float(np.median(av))

    return {
        "n_tested": int(len(ap)),
        "price": {
            "median_abs_err": med_p,
            "p90_abs_err": float(np.percentile(ap, 90)),
            "bias": float(np.median(ep)),
        },
        "vol": {
            "median_abs_err": med_v,
            "p90_abs_err": float(np.percentile(av, 90)),
            "bias": float(np.median(ev)),
        },
        # Positive = interpolating vol beat interpolating price.
        "improvement_pct": float(100.0 * (med_p - med_v) / med_p) if med_p else None,
        "win_rate": float(100.0 * (av < ap).mean()),
        "rows": rows,
        # kept so slices can be pooled at the level of ERRORS rather than by
        # pooling the observations -- see space_holdout_pooled.
        "_err_price": ep.tolist(),
        "_err_vol": ev.tolist(),
    }


def space_holdout_pooled(wide: pd.DataFrame) -> dict | None:
    """
    Run space_holdout once per (date, right) and pool the ERRORS.

    Pooling the observations instead would be a straightforward mistake and a
    tempting one, because it produces a much bigger number. A surface is a
    single date and a single right: strike and DTE only identify a contract
    within one session. Throwing 53 sessions and both rights into one cloud
    puts many different prices on the same (strike, DTE) coordinate, so the
    triangulation is interpolating between contracts that are not comparable
    and the reported error measures that confusion rather than the geometry.

    So each slice is interpolated on its own, and only the resulting per-cell
    dollar errors are concatenated.
    """
    if wide is None or wide.empty:
        return None
    ep, ev, n_slices, rows = [], [], 0, []
    for (_, _), g in wide.groupby(["date", "cp"]):
        r = space_holdout(g)
        if r is None:
            continue
        n_slices += 1
        ep.extend(r["_err_price"])
        ev.extend(r["_err_vol"])
        rows.extend(r["rows"][:6])          # a thin sample for the scatter
    if len(ep) < 30:
        return None

    ep, ev = np.array(ep), np.array(ev)
    ap, av = np.abs(ep), np.abs(ev)
    med_p, med_v = float(np.median(ap)), float(np.median(av))
    return {
        "n_tested": int(len(ap)),
        "n_slices": int(n_slices),
        "price": {"median_abs_err": med_p,
                  "p90_abs_err": float(np.percentile(ap, 90)),
                  "bias": float(np.median(ep))},
        "vol": {"median_abs_err": med_v,
                "p90_abs_err": float(np.percentile(av, 90)),
                "bias": float(np.median(ev))},
        "improvement_pct": float(100.0 * (med_p - med_v) / med_p) if med_p else None,
        "win_rate": float(100.0 * (av < ap).mean()),
        "rows": rows[:900],
    }


def parity_audit(wide: pd.DataFrame, fwd: pd.DataFrame) -> dict:
    """
    Put-call parity as a structural test, and as a way to FILL holes.

    Two questions, both of which the price-interpolation sections cannot answer:

      consistency  do the call mid and the put mid on the same strike agree
                   about the forward? Parity is an identity, so a residual is
                   not a mispricing to trade -- it is the mid failing to be a
                   price, measured a second independent way.
      recovery     when one right is quoted and the other is not, parity
                   reconstructs the missing mark EXACTLY, with no surface, no
                   volatility and no interpolation. Every such cell is a hole
                   that never needed guessing.
    """
    out = {"n_pairs": 0, "resid_med": None, "resid_p90": None, "n_exec_viol": 0,
           "n_expiries": 0, "median_basis": None, "median_D": None,
           "recoverable": 0, "holes": 0, "recover_pct": None,
           "n_fits": 0, "median_resid_by_fit": None}
    if wide is None or wide.empty or fwd is None or fwd.empty:
        return out

    out["n_fits"] = int(len(fwd))
    out["n_expiries"] = int(fwd["expiry"].nunique())
    out["median_basis"] = float(fwd["basis"].median()) if fwd["basis"].notna().any() else None
    out["median_D"] = float(fwd["D"].median())
    out["median_resid_by_fit"] = float(fwd["resid_med"].median())

    piv = (wide.pivot_table(index=["date", "expiry", "dte", "strike"], columns="cp",
                            values=[MARK_FIELD, "BID", "ASK"], aggfunc="last"))
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()

    cm, pm = f"{MARK_FIELD}_C", f"{MARK_FIELD}_P"
    if cm not in piv or pm not in piv:
        return out

    key = fwd.set_index(["date", "expiry"])[["F", "D"]]
    idx = pd.MultiIndex.from_arrays([piv["date"], piv["expiry"]])
    piv["F"] = key["F"].reindex(idx).to_numpy()
    piv["D"] = key["D"].reindex(idx).to_numpy()

    both = piv.dropna(subset=[cm, pm, "F", "D"])
    if len(both):
        resid = (both[cm] - both[pm]) - both["D"] * (both["F"] - both["strike"])
        out["n_pairs"] = int(len(both))
        out["resid_med"] = float(resid.abs().median())
        out["resid_p90"] = float(resid.abs().quantile(0.90))
        # Would the residual survive paying the spread on all four legs? Buy
        # the conversion at the ask, sell it at the bid.
        need = ["ASK_C", "BID_P", "BID_C", "ASK_P"]
        if all(c in both.columns for c in need):
            q = both.dropna(subset=need)
            if len(q):
                synth_lo = q["ASK_C"] - q["BID_P"]       # cost to buy C-P
                synth_hi = q["BID_C"] - q["ASK_P"]       # proceeds to sell C-P
                fair = q["D"] * (q["F"] - q["strike"])
                out["n_exec_viol"] = int(((synth_lo < fair - 1e-9)
                                          | (synth_hi > fair + 1e-9)).sum())

    # Holes one right can fill for the other.
    one_sided = piv[piv[cm].isna() ^ piv[pm].isna()]
    out["holes"] = int(len(one_sided))
    out["recoverable"] = int(one_sided.dropna(subset=["F", "D"]).shape[0])
    if out["holes"]:
        out["recover_pct"] = float(100.0 * out["recoverable"] / out["holes"])
    return out
