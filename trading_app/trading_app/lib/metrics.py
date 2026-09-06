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
        return None  # collinear cloud, nothing to trianglulate

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
