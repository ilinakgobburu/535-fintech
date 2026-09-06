"""
Turn an LSEG pickle into the tidy long table the rest of the app works from.

The shape LSEG hands back depends on how many fields survived the request,
which is the source of a subtle bug worth spelling out:

  * Two or more fields come back -> MultiIndex columns, (RIC, field) or
    (field, RIC) depending on the call.
  * Exactly ONE field survives  -> FLAT columns of bare RICs, with the field
    name parked on `df.columns.name`.

If you assume the flat case is TRDPRC_1 you will silently mislabel whichever
single field actually came back. We read `columns.name` instead.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .ric import parse_option_ric

# The two fields the assignment cares about.
MARK_FIELD = "MID_PRICE"   # closing NBBO midpoint -- the mark
PRINT_FIELD = "TRDPRC_1"   # last trade -- evidence someone traded

# Anything we recognise as a price field when sniffing a MultiIndex level.
KNOWN_FIELDS = {
    "TRDPRC_1", "MID_PRICE", "SETTLE", "CLOSE",
    "BID", "ASK", "HIGH_1", "LOW_1", "OPEN_PRC",
}

TIDY_COLUMNS = [
    "date", "ric", "field", "value",
    "underlying", "cp", "expiry", "strike", "month_code",
]


def load_payload(path: str | Path) -> dict:
    """Read the cached pickle. Cache-first: this never talks to LSEG."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No cache at {path}. Run scripts/fetch_lseg.py with LSEG Workspace "
            f"running to build it."
        )
    with path.open("rb") as f:
        payload = pickle.load(f)
    payload.setdefault("synthetic", False)
    return payload


def _resolve_levels(cols: pd.MultiIndex) -> tuple[int, int]:
    """Work out which MultiIndex level holds the field name and which the RIC."""
    for i in range(cols.nlevels):
        vals = {str(v).upper() for v in cols.get_level_values(i)}
        if vals & KNOWN_FIELDS:
            return i, (1 if i == 0 else 0)
    # Fall back to the LSEG default ordering, (RIC, field).
    return (cols.nlevels - 1), 0


def _iter_series(df: pd.DataFrame):
    """Yield (ric, field, series) regardless of the column shape LSEG used."""
    cols = df.columns

    if isinstance(cols, pd.MultiIndex):
        field_lvl, ric_lvl = _resolve_levels(cols)
        for col in cols:
            yield str(col[ric_lvl]), str(col[field_lvl]).upper(), df[col]
        return

    # Flat columns: the single surviving field name lives on columns.name.
    field = str(cols.name).upper() if cols.name else PRINT_FIELD
    if cols.name is None:
        warnings.warn(
            "Options frame has flat columns and no columns.name; assuming "
            f"{PRINT_FIELD}. Check the pickle.",
            RuntimeWarning,
        )
    for col in cols:
        label, fld = str(col), field
        if "|" in label:  # "RIC | FIELD"
            label, fld = (p.strip() for p in label.split("|", 1))
        yield label, fld.upper(), df[col]


def flatten_options(df_options: pd.DataFrame) -> pd.DataFrame:
    """Collapse an LSEG options frame into one row per (date, ric, field)."""
    if df_options is None or df_options.empty:
        return pd.DataFrame(columns=TIDY_COLUMNS + ["dte"])

    frame = df_options.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)

    blocks = []
    for ric, field, series in _iter_series(frame):
        parsed = parse_option_ric(ric)
        if parsed is None:
            continue
        vals = pd.to_numeric(series, errors="coerce").dropna()
        vals = vals[np.isfinite(vals)]
        if vals.empty:
            continue
        block = pd.DataFrame(
            {
                "date": pd.DatetimeIndex(vals.index).normalize(),
                "value": vals.to_numpy(dtype=float),
            }
        )
        block["ric"] = ric
        block["field"] = field
        for key in ("underlying", "cp", "expiry", "strike", "month_code"):
            block[key] = parsed[key]
        blocks.append(block)

    if not blocks:
        return pd.DataFrame(columns=TIDY_COLUMNS + ["dte"])

    tidy = pd.concat(blocks, ignore_index=True)
    tidy["expiry"] = pd.to_datetime(tidy["expiry"])
    tidy["dte"] = (tidy["expiry"] - tidy["date"]).dt.days
    # A quote dated after its own expiry is bad data, not a long-dated option.
    return tidy[tidy["dte"] >= 0].reset_index(drop=True)


def attach_underlying(tidy: pd.DataFrame, df_stock: pd.DataFrame) -> pd.DataFrame:
    """Join each option row to that session's underlying close and moneyness."""
    tidy = tidy.copy()
    if tidy.empty or df_stock is None or df_stock.empty:
        tidy["spot"] = np.nan
        tidy["moneyness"] = np.nan
        return tidy

    stock = df_stock.copy()
    if not isinstance(stock.index, pd.DatetimeIndex):
        stock.index = pd.to_datetime(stock.index)
    close_col = PRINT_FIELD if PRINT_FIELD in stock.columns else stock.columns[-1]

    spot = pd.to_numeric(stock[close_col], errors="coerce").dropna()
    spot.index = pd.DatetimeIndex(spot.index).normalize()
    spot = spot[~spot.index.duplicated(keep="last")].sort_index()

    # as-of join: if the option printed on a day the stock did not, walk back
    # to the last session that had a close.
    spot_df = pd.DataFrame({"date": spot.index, "spot": spot.to_numpy(float)})
    tidy = tidy.sort_values("date")
    tidy["spot"] = pd.merge_asof(
        tidy[["date"]], spot_df, on="date", direction="backward"
    )["spot"].to_numpy()

    tidy["moneyness"] = tidy["strike"] / tidy["spot"].replace(0, np.nan)
    return tidy.reset_index(drop=True)


def pivot_fields(tidy: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, ric) with the mark and the print side by side."""
    empty_cols = [
        "date", "ric", "underlying", "cp", "expiry", "strike", "dte",
        "spot", "moneyness", MARK_FIELD, PRINT_FIELD,
        "has_mark", "has_print", "abs_diff", "rel_diff",
    ]
    if tidy.empty:
        return pd.DataFrame(columns=empty_cols)

    keep = tidy[tidy["field"].isin([MARK_FIELD, PRINT_FIELD])]
    if keep.empty:
        return pd.DataFrame(columns=empty_cols)

    idx = [
        c for c in
        ["date", "ric", "underlying", "cp", "expiry", "strike", "dte", "spot", "moneyness"]
        if c in keep.columns
    ]
    wide = (
        keep.pivot_table(index=idx, columns="field", values="value", aggfunc="last")
        .reset_index()
    )
    wide.columns.name = None

    for field in (MARK_FIELD, PRINT_FIELD):
        if field not in wide.columns:
            wide[field] = np.nan

    wide["has_mark"] = wide[MARK_FIELD].notna()
    wide["has_print"] = wide[PRINT_FIELD].notna()
    wide["abs_diff"] = (wide[MARK_FIELD] - wide[PRINT_FIELD]).abs()
    wide["rel_diff"] = wide["abs_diff"] / wide[MARK_FIELD].replace(0, np.nan)
    return wide.sort_values(["date", "expiry", "strike"]).reset_index(drop=True)


def build_frames(payload: dict) -> dict:
    """
    Full cache -> analysis pipeline, plus a note on which fields actually
    survived the pull. The app surfaces that note rather than hiding it.
    """
    tidy = flatten_options(payload.get("options"))
    tidy = attach_underlying(tidy, payload.get("stock"))
    wide = pivot_fields(tidy)

    present = sorted(tidy["field"].unique().tolist()) if not tidy.empty else []
    return {
        "tidy": tidy,
        "wide": wide,
        "stock": payload.get("stock"),
        "underlying": payload.get("ticker", "?"),
        "fetched_at": payload.get("fetched_at", "?"),
        "synthetic": bool(payload.get("synthetic", False)),
        "fields_present": present,
        "has_mark": MARK_FIELD in present,
        "has_print": PRINT_FIELD in present,
    }
