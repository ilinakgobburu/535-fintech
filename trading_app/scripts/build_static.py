"""
Build the GitHub Pages deliverable: one self-contained docs/index.html.

Design note on why this is precomputed rather than computed in the browser:
the Reflex app and the static page must not disagree about the numbers. Every
slice, statistic and interpolated sheet is computed once here, in the same
Python that the Reflex app calls, and embedded as JSON. The browser only
chooses which precomputed slice to draw. There is exactly one implementation
of the math.

    python scripts/build_static.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading_app import theme as T  # noqa: E402
from trading_app.lib.loaders import (  # noqa: E402
    MARK_FIELD, PRINT_FIELD, build_frames, load_payload,
)
from trading_app.lib.metrics import (  # noqa: E402
    arbitrage_audit, fmt_money, fmt_pct, interpolate_grid, interpolation_holdout,
    infer_strike_step, occupancy_matrix, print_probability, quote_quality,
    slice_asof,
    sparsity_stats, spread_by_bucket, spread_stats, trade_position_histogram,
    vertical_spread_example,
)
from trading_app.lib.vol import (  # noqa: E402
    attach_iv, implied_forward, iv_coverage, parity_audit, space_holdout_pooled,
)

DEFAULT_CACHE = ROOT / "trading_app" / "data" / "option_pipeline_data.pkl"
DEFAULT_OUT = ROOT.parent / "docs" / "index.html"
MIN_SERIES = 12          # do not offer a date too thin to say anything about
# Sheet resolution. These are decorative surfaces drawn under a point cloud,
# not data anyone reads a number off, and they dominate the payload: five
# sheets per slice were 52% of every combo. 20x14 halves that and is visually
# indistinguishable at the size the sheet is rendered.
SHEET_NX, SHEET_NY = 20, 14


def _clean(arr, dp: int = 4) -> list:
    """numpy -> JSON, with NaN as null (JSON has no NaN)."""
    out = []
    for v in np.asarray(arr, dtype=object).ravel():
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            out.append(None)
        elif isinstance(v, (np.floating, float)):
            out.append(round(float(v), dp))
        elif isinstance(v, (np.integer, int)):
            out.append(int(v))
        else:
            out.append(str(v))
    return out


def _occ_payload(sl: pd.DataFrame, field: str) -> dict | None:
    grid = occupancy_matrix(sl, field)
    if grid is None or grid.empty:
        return None
    return {
        "x": [f"{c:.2f}" for c in grid.columns],
        "y": [str(i) for i in grid.index],
        "z": [_clean(row) for row in grid.values],
        "filled": round(float(np.nansum(grid.values)) / max(grid.size, 1), 4),
    }


# The sheet's gap threshold is a distance in STRIKE DOLLARS. Hardcoding 1.25
# tuned it to UUUU's ~$13.50-wide grid; on CCJ (~$55 wide) the same number
# blanked 78% of the sheet and made a well-covered surface look 4x sparser than
# its data. Express it as a fraction of the slice's own strike range instead:
# 1.25/13.5 leaves UUUU unchanged and puts CCJ back at ~91% coverage.
GAP_FRACTION = 1.25 / 13.5


def _gap_for(sl: pd.DataFrame, col: str = "strike") -> float:
    span = float(sl[col].max() - sl[col].min()) if len(sl) else 0.0
    return span * GAP_FRACTION if span > 0 else None


class _RicTable:
    """
    Intern RIC strings once instead of repeating them in every slice.

    A RIC is ~20 characters and the same contract appears in most of the 53
    sessions, so the raw strings were 19% of the payload. The table is written
    once at the top level and each slice stores integer indices into it.
    """

    def __init__(self):
        self.index: dict[str, int] = {}

    def ids(self, series) -> list:
        out = []
        for r in series:
            r = str(r)
            i = self.index.get(r)
            if i is None:
                i = len(self.index)
                self.index[r] = i
            out.append(i)
        return out

    def as_list(self) -> list:
        return [r for r, _ in sorted(self.index.items(), key=lambda kv: kv[1])]


def build_combo(sl: pd.DataFrame, rics: "_RicTable") -> dict:
    stats = sparsity_stats(sl)
    spot = sl["spot"].dropna()
    spot_val = round(float(spot.median()), 4) if len(spot) else None

    m = sl.dropna(subset=[MARK_FIELD])
    p = sl.dropna(subset=[PRINT_FIELD])
    both = sl.dropna(subset=[MARK_FIELD, PRINT_FIELD])

    sheet = interpolate_grid(sl, MARK_FIELD, n_strike=SHEET_NX, n_dte=SHEET_NY,
                             max_fill_gap=_gap_for(sl))
    # Same sheet in moneyness space. The gap threshold is in the axis's own
    # units: $1.25 of strike is roughly 0.10 of K/S on a $13 name.
    sheet_mny = interpolate_grid(sl, MARK_FIELD, n_strike=SHEET_NX, n_dte=SHEET_NY,
                                 max_fill_gap=_gap_for(sl, "moneyness"), x_col="moneyness")
    # Bid and ask sheets turn the "surface" into a slab with real thickness.
    sheet_bid = interpolate_grid(sl, "BID", n_strike=SHEET_NX, n_dte=SHEET_NY,
                                 max_fill_gap=_gap_for(sl))
    sheet_ask = interpolate_grid(sl, "ASK", n_strike=SHEET_NX, n_dte=SHEET_NY,
                                 max_fill_gap=_gap_for(sl))
    # The same cloud in the space the surface is actually smooth in.
    iv_pts = sl.dropna(subset=["iv"]) if "iv" in sl.columns else sl.iloc[0:0]
    sheet_iv = (interpolate_grid(iv_pts, "iv", n_strike=SHEET_NX, n_dte=SHEET_NY,
                                 max_fill_gap=_gap_for(sl))
                if len(iv_pts) >= 8 else None)

    return {
        "spot": spot_val,
        "stats": {
            "n_series": stats["n_series"],
            "n_mark_only": stats["n_mark_only"],
            "n_print_only": stats["n_print_only"],
            "n_both": stats["n_both"],
            "pct_mark_no_trade": stats["pct_mark_no_trade"],
            "median_abs_diff": stats["median_abs_diff"],
            "median_rel_diff_pct": stats["median_rel_diff_pct"],
            "max_abs_diff": stats["max_abs_diff"],
            "pct_txt": fmt_pct(stats["pct_mark_no_trade"]),
            "gap_txt": fmt_money(stats["median_abs_diff"]),
        },
        "mark": {
            "k": _clean(m["strike"]), "d": _clean(m["dte"]),
            "v": _clean(m[MARK_FIELD]), "ric": rics.ids(m["ric"]),
            "mny": _clean(m["moneyness"]),
        },
        "print": {
            "k": _clean(p["strike"]), "d": _clean(p["dte"]),
            "v": _clean(p[PRINT_FIELD]), "ric": rics.ids(p["ric"]),
            "mny": _clean(p["moneyness"]),
        },
        "both": {
            "x": _clean(both[PRINT_FIELD]), "y": _clean(both[MARK_FIELD]),
            "k": _clean(both["strike"]), "d": _clean(both["dte"]),
            "gap": _clean(both["abs_diff"]), "ric": rics.ids(both["ric"]),
        },
        "sheet": None if sheet is None else {
            "x": _clean(sheet["x"], 3), "y": _clean(sheet["y"], 3),
            "z": [_clean(row, 3) for row in sheet["z"]],
        },
        "sheet_mny": None if sheet_mny is None else {
            "x": _clean(sheet_mny["x"], 3), "y": _clean(sheet_mny["y"], 3),
            "z": [_clean(row, 3) for row in sheet_mny["z"]],
        },
        "sheet_bid": None if sheet_bid is None else {
            "x": _clean(sheet_bid["x"], 3), "y": _clean(sheet_bid["y"], 3),
            "z": [_clean(row, 3) for row in sheet_bid["z"]],
        },
        "sheet_ask": None if sheet_ask is None else {
            "x": _clean(sheet_ask["x"], 3), "y": _clean(sheet_ask["y"], 3),
            "z": [_clean(row, 3) for row in sheet_ask["z"]],
        },
        "iv": {
            "k": _clean(iv_pts["strike"]), "d": _clean(iv_pts["dte"]),
            "v": _clean(iv_pts["iv"]), "ric": rics.ids(iv_pts["ric"]),
            "mny": _clean(iv_pts["moneyness"]),
        } if len(iv_pts) else None,
        "sheet_iv": None if sheet_iv is None else {
            "x": _clean(sheet_iv["x"], 3), "y": _clean(sheet_iv["y"], 3),
            "z": [_clean(row, 3) for row in sheet_iv["z"]],
        },
        "occ_mark": _occ_payload(sl, MARK_FIELD),
        "occ_print": _occ_payload(sl, PRINT_FIELD),
        "holdout": interpolation_holdout(sl, MARK_FIELD),
        "spread": spread_stats(sl),
    }


def summarize_cache(label: str, path: Path, href: str | None) -> dict | None:
    """One row of the cross-name comparison table."""
    try:
        frames = build_frames(load_payload(path))
    except Exception:
        return None
    wide = frames["wide"]
    if wide.empty:
        return None
    st = sparsity_stats(wide)
    sp = spread_stats(wide)
    hist = trade_position_histogram(wide)
    return {
        "label": label,
        "href": href,
        "underlying": frames["underlying"],
        "n_series": st["n_series"],
        "strike_step": float(infer_strike_step(wide)),
        "n_dates": st["n_dates"],
        "pct_mark_no_trade": st["pct_mark_no_trade"],
        "median_abs_diff": st["median_abs_diff"],
        "median_rel_diff_pct": st["median_rel_diff_pct"],
        "median_spread_pct": sp["median_spread_pct"],
        "median_spread": sp["median_spread"],
        "pct_near_mid": None if hist is None else hist["pct_near_mid"],
        "spot_lo": float(frames["stock"]["LOW_1"].min()),
        "spot_hi": float(frames["stock"]["HIGH_1"].max()),
    }


def build_payload(cache: Path) -> dict:
    frames = build_frames(load_payload(cache))
    wide, stock = frames["wide"], frames["stock"]
    if wide.empty:
        raise SystemExit("Cache produced no option rows — nothing to build.")

    # Put-call parity gives the forward and the discount factor with no rate
    # assumed; implied vols are then inverted off that forward. Both are
    # computed once, here, so every panel sees the same numbers.
    fwd = implied_forward(wide)
    wide = attach_iv(wide, fwd)

    counts = wide.groupby("date")["ric"].nunique()
    dates = [d for d, n in counts.items() if n >= MIN_SERIES]
    if not dates:
        dates = list(counts.index)
    rights = sorted(wide["cp"].unique().tolist())

    ric_table = _RicTable()
    combos, default_date = {}, None
    best = -1
    for d in dates:
        for cp in rights:
            sl = slice_asof(wide, d, cp)
            if sl.empty:
                continue
            key = f"{d.date()}|{cp}"
            combos[key] = build_combo(sl, ric_table)
            score = len(sl) * sl["expiry"].nunique()
            if score > best:
                best, default_date = score, (str(d.date()), cp)

    stock_payload = {}
    if stock is not None and not stock.empty:
        stock_payload = {
            "x": [str(pd.Timestamp(i).date()) for i in stock.index],
            "o": _clean(stock["OPEN_PRC"]), "h": _clean(stock["HIGH_1"]),
            "l": _clean(stock["LOW_1"]), "c": _clean(stock["TRDPRC_1"]),
        }

    overall = sparsity_stats(wide)
    # Aggregated across every session: per-slice these would be too thin to
    # bucket meaningfully, and the claims they support are about the panel as
    # a whole, not about one day.
    aggregate = {
        "spread": spread_stats(wide),
        # K/S means opposite things for calls and puts, so pooling the
        # rights mixes reversed economics inside every bucket and destroys
        # the effect. One bucket set per right.
        "spread_bucket": spread_by_bucket(wide[wide["cp"] == "C"]),
        "spread_bucket_put": spread_by_bucket(wide[wide["cp"] == "P"]),
        "trade_hist": trade_position_histogram(wide),
        "n_dates": int(wide["date"].nunique()),
        "audit": arbitrage_audit(wide, cp="C"),
        "audit_put": arbitrage_audit(wide, cp="P"),
        "print_prob": print_probability(wide, cp="C"),
        "quote_quality": quote_quality(wide),
        "spread_example": vertical_spread_example(wide, cp="C"),
        "parity": parity_audit(wide, fwd),
        "iv_cov": iv_coverage(wide),
        "space": space_holdout_pooled(wide),
        "forward": None if fwd.empty else {
            "date": [str(pd.Timestamp(d).date()) for d in fwd["date"]],
            "expiry": [str(pd.Timestamp(e).date()) for e in fwd["expiry"]],
            "dte": _clean(fwd["dte"]), "F": _clean(fwd["F"]),
            "D": _clean(fwd["D"]), "spot": _clean(fwd["spot"]),
            "basis": _clean(fwd["basis"]), "resid": _clean(fwd["resid_med"]),
            "n_pairs": _clean(fwd["n_pairs"]),
        },
    }
    return {
        "aggregate": aggregate,
        "compare": None,      # filled in by main() when --compare is given
        "sibling": None,
        "meta": {
            "underlying": frames["underlying"],
            "fetched_at": frames["fetched_at"],
            "synthetic": frames["synthetic"],
            "fields_present": frames["fields_present"],
            "has_mark": frames["has_mark"],
            "has_print": frames["has_print"],
            "mark_field": MARK_FIELD,
            "print_field": PRINT_FIELD,
            "n_series_total": int(wide["ric"].nunique()),
            "n_dates_total": int(wide["date"].nunique()),
            "date_min": str(wide["date"].min().date()),
            "date_max": str(wide["date"].max().date()),
            "n_expiries": int(wide["expiry"].nunique()),
        "strike_step": float(infer_strike_step(wide)),
            "n_calls": int(wide[wide["cp"] == "C"]["ric"].nunique()),
            "n_puts": int(wide[wide["cp"] == "P"]["ric"].nunique()),
            "strike_min": float(wide["strike"].min()),
            "strike_max": float(wide["strike"].max()),
            "overall_pct_mark_no_trade": overall["pct_mark_no_trade"],
            "overall_median_gap": overall["median_abs_diff"],
        },
        "dates": [str(d.date()) for d in dates],
        "rights": rights,
        "default": {"date": default_date[0], "cp": default_date[1]},
        "combos": combos,
        "rics": ric_table.as_list(),
        "stock": stock_payload,
        "theme": {
            "base": T.BASE, "panel": T.PANEL, "panelHi": T.PANEL_HI,
            "line": T.LINE, "lineSoft": T.LINE_SOFT,
            "text": T.TEXT, "muted": T.TEXT_MUTED, "faint": T.TEXT_FAINT,
            "mark": T.MARK, "print": T.PRINT, "both": T.BOTH,
            "interp": T.INTERP, "interpScale": T.INTERP_SCALE,
            "interpOpacity": T.INTERP_OPACITY,
            "markScale": T.MARK_SCALE, "printScale": T.PRINT_SCALE,
            "up": T.UP, "down": T.DOWN,
            "font": T.FONT, "mono": T.FONT_MONO,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--sibling", default=None,
                    help="nav link to the other name, as 'Label=href'")
    ap.add_argument("--compare", nargs="*", default=[],
                    help="rows for the comparison table, each 'Label=path=href'")
    args = ap.parse_args()

    payload = build_payload(args.cache)

    if args.sibling and "=" in args.sibling:
        label, href = args.sibling.split("=", 1)
        payload["sibling"] = {"label": label, "href": href}

    rows = []
    for spec in args.compare:
        parts = spec.split("=")
        if len(parts) < 2:
            continue
        label, path = parts[0], Path(parts[1])
        href = parts[2] if len(parts) > 2 else None
        row = summarize_cache(label, path, href)
        if row:
            rows.append(row)
    if len(rows) > 1:
        payload["compare"] = rows
    template = (ROOT / "scripts" / "page_template.html").read_text(encoding="utf-8")

    html = template.replace(
        "/*__PAYLOAD__*/null",
        json.dumps(payload, separators=(",", ":"), allow_nan=False),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")

    meta = payload["meta"]
    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out}  ({size_mb:.2f} MB)")
    print(f"  underlying   {meta['underlying']}  fetched {meta['fetched_at']}")
    print(f"  fields       {meta['fields_present']}")
    print(f"  combos       {len(payload['combos'])} over {len(payload['dates'])} dates")
    print(f"  default      {payload['default']}")
    if not meta["has_mark"]:
        print(f"  WARNING: no {MARK_FIELD} in this cache — requirements 4 and 5 "
              f"cannot be satisfied until it is re-pulled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
