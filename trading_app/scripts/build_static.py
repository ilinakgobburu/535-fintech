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
    fmt_money, fmt_pct, interpolate_grid, interpolation_holdout,
    occupancy_matrix, slice_asof, sparsity_stats, spread_by_bucket,
    spread_stats, trade_position_histogram,
)

DEFAULT_CACHE = ROOT / "trading_app" / "data" / "option_pipeline_data.pkl"
DEFAULT_OUT = ROOT.parent / "docs" / "index.html"
MIN_SERIES = 12          # do not offer a date too thin to say anything about
SHEET_NX, SHEET_NY = 28, 20


def _clean(arr) -> list:
    """numpy -> JSON, with NaN as null (JSON has no NaN)."""
    out = []
    for v in np.asarray(arr, dtype=object).ravel():
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            out.append(None)
        elif isinstance(v, (np.floating, float)):
            out.append(round(float(v), 4))
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


def build_combo(sl: pd.DataFrame) -> dict:
    stats = sparsity_stats(sl)
    spot = sl["spot"].dropna()
    spot_val = round(float(spot.median()), 4) if len(spot) else None

    m = sl.dropna(subset=[MARK_FIELD])
    p = sl.dropna(subset=[PRINT_FIELD])
    both = sl.dropna(subset=[MARK_FIELD, PRINT_FIELD])

    sheet = interpolate_grid(sl, MARK_FIELD, n_strike=SHEET_NX, n_dte=SHEET_NY,
                             max_fill_gap=1.25)
    # Same sheet in moneyness space. The gap threshold is in the axis's own
    # units: $1.25 of strike is roughly 0.10 of K/S on a $13 name.
    sheet_mny = interpolate_grid(sl, MARK_FIELD, n_strike=SHEET_NX, n_dte=SHEET_NY,
                                 max_fill_gap=0.10, x_col="moneyness")

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
            "v": _clean(m[MARK_FIELD]), "ric": _clean(m["ric"]),
            "mny": _clean(m["moneyness"]),
        },
        "print": {
            "k": _clean(p["strike"]), "d": _clean(p["dte"]),
            "v": _clean(p[PRINT_FIELD]), "ric": _clean(p["ric"]),
            "mny": _clean(p["moneyness"]),
        },
        "both": {
            "x": _clean(both[PRINT_FIELD]), "y": _clean(both[MARK_FIELD]),
            "k": _clean(both["strike"]), "d": _clean(both["dte"]),
            "gap": _clean(both["abs_diff"]), "ric": _clean(both["ric"]),
        },
        "sheet": None if sheet is None else {
            "x": _clean(sheet["x"]), "y": _clean(sheet["y"]),
            "z": [_clean(row) for row in sheet["z"]],
        },
        "sheet_mny": None if sheet_mny is None else {
            "x": _clean(sheet_mny["x"]), "y": _clean(sheet_mny["y"]),
            "z": [_clean(row) for row in sheet_mny["z"]],
        },
        "occ_mark": _occ_payload(sl, MARK_FIELD),
        "occ_print": _occ_payload(sl, PRINT_FIELD),
        "holdout": interpolation_holdout(sl, MARK_FIELD),
        "spread": spread_stats(sl),
    }


def build_payload(cache: Path) -> dict:
    frames = build_frames(load_payload(cache))
    wide, stock = frames["wide"], frames["stock"]
    if wide.empty:
        raise SystemExit("Cache produced no option rows — nothing to build.")

    counts = wide.groupby("date")["ric"].nunique()
    dates = [d for d, n in counts.items() if n >= MIN_SERIES]
    if not dates:
        dates = list(counts.index)
    rights = sorted(wide["cp"].unique().tolist())

    combos, default_date = {}, None
    best = -1
    for d in dates:
        for cp in rights:
            sl = slice_asof(wide, d, cp)
            if sl.empty:
                continue
            key = f"{d.date()}|{cp}"
            combos[key] = build_combo(sl)
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
        "spread_bucket": spread_by_bucket(wide),
        "trade_hist": trade_position_histogram(wide),
        "n_dates": int(wide["date"].nunique()),
    }
    return {
        "aggregate": aggregate,
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
            "strike_min": float(wide["strike"].min()),
            "strike_max": float(wide["strike"].max()),
            "overall_pct_mark_no_trade": overall["pct_mark_no_trade"],
            "overall_median_gap": overall["median_abs_diff"],
        },
        "dates": [str(d.date()) for d in dates],
        "rights": rights,
        "default": {"date": default_date[0], "cp": default_date[1]},
        "combos": combos,
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
    args = ap.parse_args()

    payload = build_payload(args.cache)
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
