"""
The numbers the page has to print, and the grid the 3D sheet is drawn from.

Definitional note, because this is the whole point of the assignment: a
"listed series" here means a (date, RIC) that carries at least one of the two
fields. We cannot observe the true listed universe -- a contract that was
listed but had neither a quote nor a print leaves no trace in the pull. So the
denominator is the observed universe, which makes the reported sparsity a
*lower bound* on the real sparsity. The holes are at least this bad.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from .loaders import MARK_FIELD, PRINT_FIELD

EMPTY_STATS = {
    "n_series": 0,
    "n_mark_only": 0,
    "n_print_only": 0,
    "n_both": 0,
    "pct_mark_no_trade": None,
    "median_abs_diff": None,
    "median_rel_diff_pct": None,
    "max_abs_diff": None,
    "n_dates": 0,
}


def sparsity_stats(wide: pd.DataFrame) -> dict:
    """
    Requirement 5, plus context.

    pct_mark_no_trade   percent of listed series that day with a mid and no trade
    median_abs_diff     median |MID_PRICE - TRDPRC_1| on series that have both
    """
    if wide is None or wide.empty:
        return dict(EMPTY_STATS)

    n = len(wide)
    both = wide["has_mark"] & wide["has_print"]
    mark_only = wide["has_mark"] & ~wide["has_print"]
    print_only = wide["has_print"] & ~wide["has_mark"]

    diffs = wide.loc[both, "abs_diff"].dropna()
    rels = wide.loc[both, "rel_diff"].dropna()

    return {
        "n_series": int(n),
        "n_mark_only": int(mark_only.sum()),
        "n_print_only": int(print_only.sum()),
        "n_both": int(both.sum()),
        "pct_mark_no_trade": float(100.0 * mark_only.sum() / n) if n else None,
        "median_abs_diff": float(diffs.median()) if len(diffs) else None,
        "median_rel_diff_pct": float(100.0 * rels.median()) if len(rels) else None,
        "max_abs_diff": float(diffs.max()) if len(diffs) else None,
        "n_dates": int(wide["date"].nunique()),
    }


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}%"


def fmt_money(value: float | None, dp: int = 3) -> str:
    return "n/a" if value is None else f"${value:.{dp}f}"


def slice_asof(wide: pd.DataFrame, asof, cp: str | None = None) -> pd.DataFrame:
    """One as-of date, optionally one right. cp=None or 'B' keeps both."""
    if wide is None or wide.empty:
        return wide
    out = wide
    if asof is not None:
        out = out[out["date"] == pd.Timestamp(asof).normalize()]
    if cp in ("C", "P"):
        out = out[out["cp"] == cp]
    return out


def interpolate_grid(
    points: pd.DataFrame,
    value_col: str = MARK_FIELD,
    n_strike: int = 44,
    n_dte: int = 32,
    max_fill_gap: float | None = None,
) -> dict | None:
    """
    Linearly interpolate a sparse cloud onto a regular (strike, dte) grid.

    This is the sheet the assignment warns you about. Two deliberate choices
    keep it honest:

      * `griddata(method="linear")` returns NaN outside the convex hull of the
        observations, so the sheet stops at the edge of the data instead of
        extrapolating into the wings.
      * `max_fill_gap` additionally blanks grid nodes that sit further than
        that distance (in strike units) from any real observation, so a wide
        interior hole reads as a hole rather than a smooth ramp.

    Returns None when there is too little to interpolate at all.
    """
    cloud = points.dropna(subset=["strike", "dte", value_col])
    if len(cloud) < 8:
        return None

    x = cloud["strike"].to_numpy(float)
    y = cloud["dte"].to_numpy(float)
    z = cloud[value_col].to_numpy(float)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None  # collinear cloud, nothing to triangulate

    xi = np.linspace(x.min(), x.max(), n_strike)
    yi = np.linspace(max(0.0, y.min()), y.max(), n_dte)
    xx, yy = np.meshgrid(xi, yi)

    try:
        zz = griddata((x, y), z, (xx, yy), method="linear")
    except Exception:
        return None
    if zz is None or not np.isfinite(zz).any():
        return None

    if max_fill_gap:
        # Distance from each grid node to the nearest observation, with the
        # DTE axis rescaled into strike units so the metric is isotropic.
        y_scale = (np.ptp(x) / np.ptp(y)) if np.ptp(y) else 1.0
        dx = xx[..., None] - x[None, None, :]
        dy = (yy[..., None] - y[None, None, :]) * y_scale
        nearest = np.sqrt(dx**2 + dy**2).min(axis=-1)
        zz = np.where(nearest > max_fill_gap, np.nan, zz)

    return {"x": xi, "y": yi, "z": zz, "coverage": float(np.isfinite(zz).mean())}


def spread_stats(wide: pd.DataFrame) -> dict:
    """
    The error bar on the mark.

    The assignment asks what price you would actually get filled at on a strike
    that has a mark and no print, and answers "you do not know." The bid-ask
    spread is how big that not-knowing is, and it is measurable.

    Split by whether the contract traded, because the interesting claim is that
    the contracts with no print are exactly the ones whose quotes are widest --
    i.e. the mark is least trustworthy precisely where it is the only number
    you have.
    """
    out = {
        "median_spread": None, "median_spread_pct": None,
        "traded_spread_pct": None, "untraded_spread_pct": None,
        "widest_spread_pct": None, "n_quoted": 0,
        "ratio": None,
    }
    if wide is None or wide.empty or "spread" not in wide.columns:
        return out

    q = wide.dropna(subset=["spread", "spread_pct"])
    q = q[np.isfinite(q["spread_pct"])]
    if q.empty:
        return out

    traded = q[q["has_print"]]["spread_pct"]
    untraded = q[~q["has_print"]]["spread_pct"]

    out["n_quoted"] = int(len(q))
    out["median_spread"] = float(q["spread"].median())
    out["median_spread_pct"] = float(q["spread_pct"].median())
    out["widest_spread_pct"] = float(q["spread_pct"].max())
    if len(traded):
        out["traded_spread_pct"] = float(traded.median())
    if len(untraded):
        out["untraded_spread_pct"] = float(untraded.median())
    if out["traded_spread_pct"] and out["untraded_spread_pct"]:
        out["ratio"] = out["untraded_spread_pct"] / out["traded_spread_pct"]
    return out


def spread_by_bucket(wide: pd.DataFrame, n_buckets: int = 7) -> dict | None:
    """
    Median spread-as-%-of-mark bucketed by moneyness, split traded/untraded.
    Shows the smile-shaped liquidity cost: cheapest at the money, blowing out
    in the wings where the only number you have is the mark.
    """
    if wide is None or wide.empty or "spread_pct" not in wide.columns:
        return None
    q = wide.dropna(subset=["spread_pct", "moneyness"])
    q = q[np.isfinite(q["spread_pct"]) & np.isfinite(q["moneyness"])]
    if len(q) < 12:
        return None

    lo, hi = float(q["moneyness"].min()), float(q["moneyness"].max())
    if not (hi > lo):
        return None
    edges = np.linspace(lo, hi, n_buckets + 1)
    centers, traded, untraded, counts = [], [], [], []
    for i in range(n_buckets):
        a, b = edges[i], edges[i + 1]
        sel = q[(q["moneyness"] >= a) & (q["moneyness"] <= b if i == n_buckets - 1
                                        else q["moneyness"] < b)]
        if sel.empty:
            continue
        t = sel[sel["has_print"]]["spread_pct"]
        u = sel[~sel["has_print"]]["spread_pct"]
        centers.append(float((a + b) / 2))
        traded.append(float(t.median()) if len(t) else None)
        untraded.append(float(u.median()) if len(u) else None)
        counts.append(int(len(sel)))
    if not centers:
        return None
    return {"moneyness": centers, "traded": traded,
            "untraded": untraded, "counts": counts}


def trade_position_histogram(wide: pd.DataFrame, n_bins: int = 12) -> dict | None:
    """
    Where inside the bid-ask did the actual print land?

    0 = filled at the bid, 0.5 = filled at the mid, 1 = filled at the ask.
    If prints pile up at the edges rather than the middle, then the mid was
    not an achievable price -- somebody paid the spread to get done. That is
    direct evidence against using the mark as a fill assumption.
    """
    if wide is None or wide.empty or "trade_in_spread" not in wide.columns:
        return None
    v = wide["trade_in_spread"].dropna()
    v = v[np.isfinite(v)]
    # Allow a little outside [0,1]: prints can be stale relative to the close.
    v = v[(v >= -0.25) & (v <= 1.25)]
    if len(v) < 10:
        return None

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(np.clip(v, 0.0, 1.0), bins=edges)
    inside = v[(v >= 0) & (v <= 1)]
    at_edges = float(((inside <= 0.1) | (inside >= 0.9)).mean() * 100) if len(inside) else None
    near_mid = float(((inside > 0.4) & (inside < 0.6)).mean() * 100) if len(inside) else None
    return {
        "centers": [float((edges[i] + edges[i + 1]) / 2) for i in range(n_bins)],
        "counts": [int(c) for c in counts],
        "n": int(len(v)),
        "median": float(v.median()),
        "pct_at_edges": at_edges,
        "pct_near_mid": near_mid,
    }


def interpolation_holdout(
    sl: pd.DataFrame,
    value_col: str = MARK_FIELD,
    min_points: int = 20,
) -> dict | None:
    """
    Quantify the danger the assignment warns about, instead of asserting it.

    Take the cells we DO observe, hide each one in turn, rebuild the linear
    interpolant from its neighbors, and compare the guess to the truth. Errors
    here are a BEST case: these are interior cells surrounded by real data. The
    holes we would actually want to fill are in the wings, with less support,
    so the true error is worse than this.
    """
    cloud = sl.dropna(subset=["strike", "dte", value_col])
    if len(cloud) < min_points:
        return None

    x = cloud["strike"].to_numpy(float)
    y = cloud["dte"].to_numpy(float)
    z = cloud[value_col].to_numpy(float)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None

    # Rescale DTE into strike units so the triangulation is not degenerate.
    y_scale = np.ptp(x) / np.ptp(y)
    pts = np.column_stack([x, y * y_scale])

    truth, guess = [], []
    for i in range(len(z)):
        mask = np.ones(len(z), dtype=bool)
        mask[i] = False
        try:
            g = griddata(pts[mask], z[mask], pts[i:i + 1], method="linear")
        except Exception:
            continue
        if g is None or not np.isfinite(g[0]):
            continue  # outside the hull of its neighbors -- cannot be guessed
        truth.append(float(z[i]))
        guess.append(float(g[0]))

    if len(truth) < 8:
        return None
    truth_a, guess_a = np.array(truth), np.array(guess)
    err = guess_a - truth_a
    abs_err = np.abs(err)
    rel = 100.0 * abs_err / np.where(truth_a == 0, np.nan, truth_a)
    rel = rel[np.isfinite(rel)]

    return {
        "n_tested": int(len(truth)),
        "n_unguessable": int(len(z) - len(truth)),
        "truth": [round(v, 4) for v in truth],
        "guess": [round(v, 4) for v in guess],
        "median_abs_err": float(np.median(abs_err)),
        "p90_abs_err": float(np.percentile(abs_err, 90)),
        "max_abs_err": float(abs_err.max()),
        "median_rel_err": float(np.median(rel)) if len(rel) else None,
        "bias": float(np.median(err)),
    }


def occupancy_matrix(wide: pd.DataFrame, field: str) -> pd.DataFrame | None:
    """
    (expiry x strike) matrix of 1/0 -- did this cell carry a number.

    Rows stay in chronological expiry order, which a plain alphabetical pivot
    on "Aug 21"/"Jul 10" would destroy.
    """
    if wide is None or wide.empty or field not in wide.columns:
        return None
    sl = wide.copy()
    sl["_hit"] = sl[field].notna().astype(int)
    sl["_label"] = sl["expiry"].dt.strftime("%b %d")

    grid = sl.pivot_table(index="_label", columns="strike", values="_hit", aggfunc="max")
    order = sl.drop_duplicates("_label").sort_values("expiry")["_label"].tolist()
    return grid.reindex(order, axis=0).reindex(sorted(grid.columns), axis=1)
