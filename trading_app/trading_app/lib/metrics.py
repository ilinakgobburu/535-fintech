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

    # The denominator is series carrying at least one of the TWO fields the
    # assignment is about. BID/ASK are pulled as supporting evidence, and a row
    # holding only an ask (ASK is quoted on more series than MID_PRICE) must not
    # quietly enlarge the denominator and dilute the reported percentage.
    listed = wide[wide["has_mark"] | wide["has_print"]]
    if listed.empty:
        return dict(EMPTY_STATS)

    n = len(listed)
    both = listed["has_mark"] & listed["has_print"]
    mark_only = listed["has_mark"] & ~listed["has_print"]
    print_only = listed["has_print"] & ~listed["has_mark"]

    diffs = listed.loc[both, "abs_diff"].dropna()
    rels = listed.loc[both, "rel_diff"].dropna()

    return {
        "n_series": int(n),
        "n_mark_only": int(mark_only.sum()),
        "n_print_only": int(print_only.sum()),
        "n_both": int(both.sum()),
        "pct_mark_no_trade": float(100.0 * mark_only.sum() / n) if n else None,
        "median_abs_diff": float(diffs.median()) if len(diffs) else None,
        "median_rel_diff_pct": float(100.0 * rels.median()) if len(rels) else None,
        "max_abs_diff": float(diffs.max()) if len(diffs) else None,
        "n_dates": int(listed["date"].nunique()),
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
    x_col: str = "strike",
) -> dict | None:
    """
    Linearly interpolate a sparse cloud onto a regular (strike, dte) grid.

    This is the sheet the assignment warns you about. Two deliberate choices
    keep it honest:

      * `griddata(method="linear")` returns NaN outside the convex hull of the
        observations, so the sheet stops at the edge of the data instead of
        extrapolating into the wings.
      * `max_fill_gap` additionally blanks grid nodes that sit further than
        that distance (in `x_col`'s own units) from any real observation, so a
        wide interior hole reads as a hole rather than a smooth ramp. Note the
        units change with the axis: ~$1.25 in strike space is ~0.10 in
        moneyness on a $13 name, so callers must pass the matching value.

    Returns None when there is too little to interpolate at all.
    """
    cloud = points.dropna(subset=[x_col, "dte", value_col])
    if len(cloud) < 8:
        return None

    x = cloud[x_col].to_numpy(float)
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
    # One denominator for the bars and the percentages beside them. Clipping
    # rather than dropping the prints that landed outside the quote matters:
    # they are the strongest evidence against the mid being achievable, and
    # excluding them from the "near the mid" denominator biases toward it.
    clipped = np.clip(v, 0.0, 1.0)
    counts, _ = np.histogram(clipped, bins=edges)
    at_edges = float(((clipped <= 0.1) | (clipped >= 0.9)).mean() * 100)
    near_mid = float(((clipped > 0.4) & (clipped < 0.6)).mean() * 100)
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


# --------------------------------------------------------------------------
# Structural checks: things an option surface must obey regardless of model
# --------------------------------------------------------------------------

def infer_strike_step(wide: pd.DataFrame, default: float = 0.50) -> float:
    """
    The listed strike increment, read off the data instead of assumed.

    Both the butterfly test and the vertical-spread example compare only
    STRICTLY CONSECUTIVE strikes, so they need to know the grid. Hardcoding
    $0.50 is right for UUUU and silently wrong for anything else: on a name
    listed in $1.00 increments no pair is ever one step apart, every test is
    skipped, and the audit reports zero comparisons rather than an error. That
    is what happened to the CCJ control page.

    The modal gap between adjacent listed strikes is the grid.
    """
    if wide is None or wide.empty or "strike" not in wide.columns:
        return default
    gaps = []
    for _, g in wide.groupby(["date", "expiry"]):
        k = np.sort(g["strike"].dropna().unique())
        if len(k) > 1:
            gaps.append(np.diff(k))
    if not gaps:
        return default
    allg = np.concatenate(gaps)
    allg = allg[allg > 1e-9]
    if not len(allg):
        return default
    vals, counts = np.unique(np.round(allg, 4), return_counts=True)
    return float(vals[np.argmax(counts)])


def _intrinsic_floor(df: pd.DataFrame, is_put: bool) -> pd.Series:
    """
    Model-free lower bound for an AMERICAN option.

    Two bounds apply and the option is worth at least both:

      immediate exercise   max(S - K, 0)          calls
                           max(K - S, 0)          puts
      the European bound   D * (F - K)            calls
        off the fitted     D * (K - F)            puts
        forward

    Using spot alone -- which is what this test did before the put wing came
    back and made a parity fit possible -- ignores the second and quietly
    assumes zero carry. Where no forward was fitted, F falls back to spot and
    D to 1, and the second term collapses into the first, so the check never
    gets weaker than it was.
    """
    spot_side = (df["strike"] - df["spot"]) if is_put else (df["spot"] - df["strike"])
    floor = np.maximum(spot_side, 0.0)
    if "F" in df.columns and "D" in df.columns:
        F = df["F"].fillna(df["spot"])
        D = df["D"].fillna(1.0)
        fwd_side = D * ((df["strike"] - F) if is_put else (F - df["strike"]))
        floor = np.maximum(floor, fwd_side.fillna(0.0))
    return floor


def arbitrage_audit(wide: pd.DataFrame, strike_step: float | None = None,
                    cp: str = "C") -> dict:
    """
    Test the surface against two rules that hold for ANY arbitrage-free market,
    with no model and no volatility assumption:

      monotonicity  a call struck higher cannot cost more:  C(K) >= C(K+h)
                    the sign FLIPS for puts: P(K) <= P(K+h). A put struck
                    higher is worth more, so the same test run on puts with
                    the call inequality would report every well-behaved
                    strike as a violation.
      convexity     a butterfly cannot cost negative money:
                    V(K-h) - 2V(K) + V(K+h) >= 0
                    this one does NOT flip -- both rights are convex in K.

    Each is checked twice.

      "mid"        using MID_PRICE, the mark
      "executable" using the prices you could actually transact at -- buy the
                   wings at the ask, sell the body at the bid; sell the low
                   strike at the bid, buy the high strike at the ask

    A violation at the mid is not an arbitrage. It is evidence that the mid is
    not a price. A violation at executable prices WOULD be free money, so if
    the market is functioning there should be none.

    Only strictly consecutive strikes on the listed grid are compared, so a
    missing strike never fabricates a violation.
    """
    out = {
        "n_bfly": 0, "bfly_mid": 0, "bfly_exec": 0, "worst_bfly": None,
        "n_mono": 0, "mono_mid": 0, "mono_exec": 0, "worst_mono": None,
        "n_intr": 0, "intr_mid": 0, "intr_ask": 0, "worst_intr": None,
        "examples": [], "cp": cp, "strike_step": float(strike_step or 0.50),
    }
    if wide is None or wide.empty:
        return out

    calls = wide[wide["cp"] == cp]
    if calls.empty:
        return out
    # Inferred from THIS right's own strikes. A cache whose two rights were
    # pulled on different grids -- which is easy to do by accident across two
    # fetches -- would otherwise have the denser right's step imposed on the
    # sparser one, and the sparser one would silently test nothing.
    if strike_step is None:
        strike_step = infer_strike_step(calls)
    out["strike_step"] = float(strike_step)
    is_put = (cp == "P")

    worst_bf, worst_mn = 0.0, 0.0
    for (_, _), g in calls.groupby(["date", "expiry"]):
        g = g.sort_values("strike")
        K = g["strike"].to_numpy(float)
        M = g[MARK_FIELD].to_numpy(float)
        B = g["BID"].to_numpy(float) if "BID" in g else np.full(len(K), np.nan)
        A = g["ASK"].to_numpy(float) if "ASK" in g else np.full(len(K), np.nan)
        rics = g["ric"].tolist()

        for i in range(len(K) - 1):
            if abs(K[i + 1] - K[i] - strike_step) > 1e-9:
                continue
            if np.isfinite(M[i]) and np.isfinite(M[i + 1]):
                out["n_mono"] += 1
                # calls must fall in K, puts must rise in K
                gap = (M[i] - M[i + 1]) if is_put else (M[i + 1] - M[i])
                if gap > 1e-12:
                    out["mono_mid"] += 1
                    worst_mn = max(worst_mn, gap)
                if is_put:
                    # sell the LOW strike at the bid, buy the HIGH at the ask:
                    # long the higher-struck put is a payoff that is never
                    # negative, so being paid to hold it is free money
                    if (np.isfinite(A[i + 1]) and np.isfinite(B[i])
                            and B[i] - A[i + 1] > 1e-12):
                        out["mono_exec"] += 1
                elif (np.isfinite(A[i]) and np.isfinite(B[i + 1])
                        and B[i + 1] - A[i] > 1e-12):
                    out["mono_exec"] += 1

        for i in range(len(K) - 2):
            if (abs(K[i + 1] - K[i] - strike_step) > 1e-9
                    or abs(K[i + 2] - K[i + 1] - strike_step) > 1e-9):
                continue
            if not (np.isfinite(M[i]) and np.isfinite(M[i + 1]) and np.isfinite(M[i + 2])):
                continue
            out["n_bfly"] += 1
            v = M[i] - 2 * M[i + 1] + M[i + 2]
            if v < -1e-12:
                out["bfly_mid"] += 1
                if v < worst_bf:
                    worst_bf = v
                    out["examples"] = [{
                        "rics": [rics[i], rics[i + 1], rics[i + 2]],
                        "strikes": [float(K[i]), float(K[i + 1]), float(K[i + 2])],
                        "mids": [float(M[i]), float(M[i + 1]), float(M[i + 2])],
                        "cost_mid": float(v),
                        "cost_exec": (float(A[i] - 2 * B[i + 1] + A[i + 2])
                                      if np.isfinite(A[i]) and np.isfinite(B[i + 1])
                                      and np.isfinite(A[i + 2]) else None),
                    }]
            if np.isfinite(A[i]) and np.isfinite(B[i + 1]) and np.isfinite(A[i + 2]):
                if A[i] - 2 * B[i + 1] + A[i + 2] < -1e-12:
                    out["bfly_exec"] += 1

    out["worst_bfly"] = float(worst_bf) if worst_bf < 0 else None
    out["worst_mono"] = float(worst_mn) if worst_mn > 0 else None

    # Intrinsic floor, against the larger of immediate exercise and the
    # parity-fitted forward bound. See _intrinsic_floor.
    q = calls.dropna(subset=[MARK_FIELD, "spot"])
    if len(q):
        intrinsic = _intrinsic_floor(q, is_put)
        below = q[MARK_FIELD] < intrinsic - 1e-12
        out["n_intr"] = int(len(q))
        out["intr_mid"] = int(below.sum())
        if below.any():
            out["worst_intr"] = float((intrinsic - q[MARK_FIELD])[below].max())
        qa = q.dropna(subset=["ASK"]) if "ASK" in q else q.iloc[0:0]
        if len(qa):
            out["intr_ask"] = int((qa["ASK"] < _intrinsic_floor(qa, is_put)
                                   - 1e-12).sum())
    return out


MONEYNESS_BINS = [0, .80, .90, .97, 1.03, 1.10, 1.20, np.inf]
MONEYNESS_LABELS = ["deep ITM", "ITM", "near ITM", "ATM", "near OTM", "OTM", "deep OTM"]
DTE_BINS = [-1, 5, 10, 20, 40, np.inf]
DTE_LABELS = ["0-5d", "6-10d", "11-20d", "21-40d", "40d+"]


def print_probability(wide: pd.DataFrame, cp: str = "C") -> dict | None:
    """
    P(a trade printed) over moneyness x days-to-expiry.

    The hole is not noise: it has a shape. Liquidity peaks around the money and
    falls away on BOTH sides -- steeply into deep ITM, where the option is
    expensive and behaves like the stock, and more gently into deep OTM, where
    it is nearly worthless. That asymmetric hump is a structure you could model,
    which turns "missing" from an absence into a quantity.
    """
    if wide is None or wide.empty or "moneyness" not in wide.columns:
        return None
    w = wide[wide["cp"] == cp].dropna(subset=["moneyness", "dte"]).copy()
    # K/S < 1 is in the money for a CALL and out of the money for a PUT, so the
    # same bucket edges carry opposite meanings. Reverse the labels for puts
    # rather than reversing the edges, which keeps the bins identical.
    labels = MONEYNESS_LABELS if cp == "C" else list(reversed(MONEYNESS_LABELS))
    if len(w) < 50:
        return None
    w["mb"] = pd.cut(w["moneyness"], MONEYNESS_BINS, labels=labels)
    w["db"] = pd.cut(w["dte"], DTE_BINS, labels=DTE_LABELS)

    grid = w.pivot_table(index="mb", columns="db", values="has_print",
                         aggfunc="mean", observed=False) * 100
    counts = w.pivot_table(index="mb", columns="db", values="has_print",
                           aggfunc="size", observed=False)
    order = MONEYNESS_LABELS  # display order is always ITM -> OTM
    grid = grid.reindex(order).reindex(DTE_LABELS, axis=1)
    counts = counts.reindex(order).reindex(DTE_LABELS, axis=1)

    marg = []
    for m in MONEYNESS_LABELS:
        s = w[w["mb"] == m]
        marg.append({
            "label": m,
            "pct": float(100 * s["has_print"].mean()) if len(s) else None,
            "n": int(len(s)),
            "median_mark": float(s[MARK_FIELD].median()) if s[MARK_FIELD].notna().any() else None,
        })
    return {
        "x": DTE_LABELS, "y": MONEYNESS_LABELS,
        "z": [[None if pd.isna(v) else round(float(v), 1) for v in row]
              for row in grid.to_numpy()],
        "n": [[0 if pd.isna(v) else int(v) for v in row] for row in counts.to_numpy()],
        "marginal": marg,
    }


def quote_quality(wide: pd.DataFrame) -> dict:
    """
    Things that would discredit the quotes, tested and mostly NOT found.

    Reporting what was ruled out matters: both the assignment and this page
    describe the mid as 'a midpoint of a possibly-stale quote'. That is testable,
    and it turns out to be largely false here. The mid's problem is not that it
    is stale. It is that it is unexecutable.
    """
    out = {"n": 0, "crossed": 0, "zero_bid": 0,
           "stale_n": 0, "stale": 0, "stale_pct": None}
    if wide is None or wide.empty or "BID" not in wide.columns:
        return out
    q = wide.dropna(subset=["BID", "ASK"])
    out["n"] = int(len(q))
    if len(q):
        out["crossed"] = int((q["BID"] >= q["ASK"]).sum())
        out["zero_bid"] = int((q["BID"] <= 0).sum())

    s = wide.dropna(subset=[MARK_FIELD, "spot"]).sort_values(["ric", "date"])
    if len(s) > 10:
        s = s.assign(dmid=s.groupby("ric")[MARK_FIELD].diff().abs(),
                     dspot=s.groupby("ric")["spot"].diff().abs())
        cand = s.dropna(subset=["dmid", "dspot"])
        cand = cand[cand["dspot"] > 0.05]
        if len(cand):
            out["stale_n"] = int(len(cand))
            out["stale"] = int((cand["dmid"] < 1e-9).sum())
            out["stale_pct"] = float(100 * out["stale"] / len(cand))
    return out


def vertical_spread_example(wide: pd.DataFrame, strike_step: float | None = None,
                            cp: str = "C") -> dict | None:
    """
    One real vertical spread, priced at the mark and at prices you could trade.

    Abstractions about basis points do not land. "This spread is worth X at the
    mid and Y if you actually have to trade it" does.

    The example returned is the one closest to the MEDIAN slippage, not the
    worst. Picking the worst would be cherry-picking; the median says what a
    typical trade costs, and the distribution is reported alongside it.
    """
    if wide is None or wide.empty or "BID" not in wide.columns:
        return None
    rows = []
    side = wide[wide["cp"] == cp]
    if strike_step is None:
        strike_step = infer_strike_step(side)
    for (d, e), g in side.groupby(["date", "expiry"]):
        g = g.sort_values("strike").dropna(subset=[MARK_FIELD, "BID", "ASK"])
        K = g["strike"].to_numpy(float)
        for i in range(len(K) - 1):
            if abs(K[i + 1] - K[i] - strike_step) > 1e-9:
                continue
            lo, hi = g.iloc[i], g.iloc[i + 1]
            # A debit vertical buys the dearer strike. For calls that is the
            # LOWER strike; for puts the HIGHER. Bind both legs once here so
            # the strikes, the RICs and the quotes cannot disagree.
            long_leg, short_leg = (hi, lo) if cp == "P" else (lo, hi)
            # a call vertical is a debit low-minus-high; a put vertical reverses
            mid_val = float(hi[MARK_FIELD] - lo[MARK_FIELD]) if cp == "P" \
                else float(lo[MARK_FIELD] - hi[MARK_FIELD])
            if not (0.05 <= mid_val <= strike_step):
                continue  # a vertical cannot be worth more than its width
            exec_val = (float(hi["ASK"] - lo["BID"]) if cp == "P"
                        else float(lo["ASK"] - hi["BID"]))  # pay ask, receive bid
            rows.append({
                "date": str(pd.Timestamp(d).date()),
                "expiry": str(pd.Timestamp(e).date()),
                "dte": int(lo["dte"]),
                "cp": cp,
                "k_long": float(long_leg["strike"]),
                "k_short": float(short_leg["strike"]),
                "ric_long": str(long_leg["ric"]),
                "ric_short": str(short_leg["ric"]),
                "mid_value": mid_val,
                "exec_value": exec_val,
                "slippage": exec_val - mid_val,
                "slippage_pct": 100.0 * (exec_val - mid_val) / mid_val,
                "legs": {
                    "long": {"mid": float(long_leg[MARK_FIELD]),
                             "bid": float(long_leg["BID"]),
                             "ask": float(long_leg["ASK"])},
                    "short": {"mid": float(short_leg[MARK_FIELD]),
                              "bid": float(short_leg["BID"]),
                              "ask": float(short_leg["ASK"])},
                },
            })

    if not rows:
        return None
    slips = np.array([r["slippage_pct"] for r in rows], dtype=float)
    med = float(np.median(slips))
    pick = min(rows, key=lambda r: abs(r["slippage_pct"] - med))
    pick["n_spreads"] = len(rows)
    pick["median_slippage_pct"] = med
    pick["p90_slippage_pct"] = float(np.percentile(slips, 90))
    pick["pct_worthless"] = float(100.0 * (slips >= 100).mean())
    return pick
