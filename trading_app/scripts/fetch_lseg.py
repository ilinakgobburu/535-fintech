"""
Re-pull the option panel from LSEG and write the pickle the app reads.

Three things this does differently from the starter, each for a concrete reason:

1. ONE FIELD PER REQUEST.
   When you ask for N fields and only one survives, LSEG hands back FLAT
   columns of bare RICs with the field name parked on `columns.name`. Two
   fields collapse to one shape that looks identical to a single-field pull.
   That is exactly how the previous cache ended up as TRDPRC_1-only with no
   record that MID_PRICE had been requested at all. Requesting each field
   separately and assembling the MultiIndex ourselves makes it impossible to
   lose a field label, and makes an empty field loudly empty.

2. STRIKES BANDED PER EXPIRY.
   The starter takes the high and low across the whole window and generates
   every strike in between for every expiry. Most of those contracts never
   existed. We band each expiry to the underlying's range over that contract's
   own life, plus a buffer, which cuts the candidate count substantially.

3. BOTH RIGHTS, WITH THE SUFFIX BUG FIXED.
   The published RIC scheme says the ^ suffix repeats the body's month letter.
   That is true for calls and wrong for puts -- LSEG keys the suffix off the
   expiry month's CALL letter for both rights. Generating ^{put letter} asks
   for contracts that do not resolve, which is why the first pull came back
   calls-only. See trading_app/lib/ric.py. --rights and --merge let you pull
   just the wing that was missed and fold it into an existing cache.

    python scripts/fetch_lseg.py --dry-run     # count candidates, no session
    python scripts/fetch_lseg.py               # real pull
"""

from __future__ import annotations

import argparse
import datetime as dt
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading_app.lib.ric import build_option_ric  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_OUT = ROOT / "trading_app" / "data" / "option_pipeline_data.pkl"

# Requested one at a time -- see docstring note 1.
OPTION_FIELDS = ["MID_PRICE", "TRDPRC_1", "BID", "ASK"]
STOCK_FIELDS = ["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1"]


def candidate_rics(
    df_stock: pd.DataFrame,
    root: str,
    start: dt.date,
    end: dt.date,
    strike_step: float,
    band: float,
    rights: tuple[str, ...] = ("C", "P"),
) -> tuple[list[str], dict]:
    """
    Build the candidate universe, banding strikes per expiry.

    For each Friday expiry we look at where the underlying actually traded
    between `start` and that expiry, and only generate strikes within `band`
    dollars of that range. A contract struck $8 away from anywhere the stock
    ever went is not worth a request.
    """
    lows = pd.to_numeric(df_stock["LOW_1"], errors="coerce")
    highs = pd.to_numeric(df_stock["HIGH_1"], errors="coerce")
    idx = pd.DatetimeIndex(df_stock.index).normalize()

    fridays = pd.date_range(start=start, end=end, freq="W-FRI")
    rics, per_expiry = [], {}

    for f in fridays:
        expiry = f.date()
        mask = idx <= f
        if not mask.any():
            continue
        lo_obs = float(lows[mask].min())
        hi_obs = float(highs[mask].max())
        if not np.isfinite(lo_obs) or not np.isfinite(hi_obs):
            continue

        lo = max(strike_step, np.floor((lo_obs - band) / strike_step) * strike_step)
        hi = np.ceil((hi_obs + band) / strike_step) * strike_step
        strikes = np.arange(lo, hi + strike_step, strike_step)

        made = []
        for k in strikes:
            for cp in rights:
                made.append(build_option_ric(root, expiry, float(k), cp))
        per_expiry[str(expiry)] = len(made)
        rics.extend(made)

    return rics, per_expiry


def pull_field(ld, universe: list[str], field: str, start: str, end: str,
               batch_size: int, pause: float) -> pd.DataFrame | None:
    """
    Pull ONE field across the universe. Batches, with a single-RIC retry when a
    batch throws (a batch containing one bad RIC can fail as a whole).
    """
    frames = []
    batches = [universe[i:i + batch_size] for i in range(0, len(universe), batch_size)]
    n_ok = 0

    for i, batch in enumerate(batches, 1):
        got = None
        try:
            got = ld.get_history(universe=batch, fields=[field],
                                 start=start, end=end, interval="daily")
        except Exception:
            for ric in batch:  # fall back to one at a time
                try:
                    one = ld.get_history(universe=[ric], fields=[field],
                                         start=start, end=end, interval="daily")
                    if one is not None and not one.empty:
                        one = one.dropna(how="all", axis=1)
                        if not one.empty:
                            frames.append(one)
                            n_ok += 1
                except Exception:
                    continue
        if got is not None and not got.empty:
            got = got.dropna(how="all", axis=1)
            if not got.empty:
                frames.append(got)
                n_ok += got.shape[1]

        if i % 20 == 0 or i == len(batches):
            print(f"    {field:<10} batch {i}/{len(batches)}  series so far {n_ok}",
                  flush=True)
        if pause:
            time.sleep(pause)

    if not frames:
        return None
    out = pd.concat(frames, axis=1)
    # A single-field response has flat RIC columns; that is what we want here,
    # because we already know which field we asked for.
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="UUUU", help="option root, e.g. UUUU")
    ap.add_argument("--stock", default="UUUU.K", help="underlying RIC")
    ap.add_argument("--weeks", type=int, default=12)
    ap.add_argument("--strike-step", type=float, default=0.50)
    ap.add_argument("--band", type=float, default=3.0,
                    help="dollars beyond the observed range to still generate strikes for")
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--pause", type=float, default=0.0)
    ap.add_argument("--fields", nargs="+", default=OPTION_FIELDS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--rights", nargs="+", default=["C", "P"], choices=["C", "P"],
                    help="which rights to generate; use 'P' to backfill a calls-only cache")
    ap.add_argument("--merge", type=Path, default=None,
                    help="existing pickle to fold this pull into, instead of "
                         "writing a fresh one (keeps its stock frame and window)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the candidate list and stop; no LSEG session")
    args = ap.parse_args()

    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(weeks=args.weeks)
    start_str, end_str = start_date.isoformat(), end_date.isoformat()
    print(f"window {start_str} -> {end_str}  root={args.root}  stock={args.stock}")

    if args.dry_run:
        # Synthesise a plausible range so we can count candidates offline.
        days = pd.bdate_range(start_date, end_date)
        fake = pd.DataFrame({"LOW_1": 7.0, "HIGH_1": 20.0}, index=days)
        rics, per_expiry = candidate_rics(
            fake, args.root, start_date, end_date, args.strike_step, args.band,
            tuple(args.rights))
        print(f"candidates: {len(rics)} across {len(per_expiry)} expiries")
        print(f"  ~{-(-len(rics)//args.batch_size)} batches per field "
              f"x {len(args.fields)} fields")
        print("  (dry run uses an assumed $7-$20 range; the real pull bands to "
              "the actual underlying)")
        return 0

    merge_payload = None
    if args.merge:
        with args.merge.open("rb") as fh:
            merge_payload = pickle.load(fh)
        win = merge_payload.get("window")
        if win:
            start_str, end_str = win
            start_date = dt.date.fromisoformat(start_str)
            end_date = dt.date.fromisoformat(end_str)
            print(f"merging into {args.merge} — reusing its window {start_str} -> {end_str}")

    try:
        import lseg.data as ld
    except Exception as exc:
        print(f"cannot import lseg.data: {exc}")
        return 1

    try:
        ld.open_session()
    except Exception as exc:
        print(f"open_session failed: {exc}")
        return 2

    if merge_payload is not None:
        df_stock = merge_payload["stock"]
        print(f"reusing cached underlying ({len(df_stock)} sessions)")
    else:
        print("fetching underlying...")
        df_stock = ld.get_history(universe=[args.stock], fields=STOCK_FIELDS,
                                  start=start_str, end=end_str, interval="daily")
    if df_stock is None or df_stock.empty:
        print("no underlying history -- check the RIC and your entitlements")
        ld.close_session()
        return 3
    print(f"  {len(df_stock)} sessions, "
          f"${float(df_stock['LOW_1'].min()):.2f} - ${float(df_stock['HIGH_1'].max()):.2f}")

    rics, per_expiry = candidate_rics(
        df_stock, args.root, start_date, end_date, args.strike_step, args.band,
        tuple(args.rights))
    print(f"candidate universe: {len(rics)} RICs ({'+'.join(args.rights)}) "
          f"across {len(per_expiry)} expiries")

    by_field: dict[str, pd.DataFrame] = {}
    for field in args.fields:
        print(f"  pulling {field}...")
        got = pull_field(ld, rics, field, start_str, end_str,
                         args.batch_size, args.pause)
        if got is None or got.empty:
            print(f"    {field}: NOTHING CAME BACK")
            continue
        by_field[field] = got
        print(f"    {field}: {got.shape[1]} series, "
              f"{int(got.notna().sum().sum())} observations")

    ld.close_session()

    if not by_field:
        print("no option data at all -- nothing written")
        return 4

    # Assemble an explicit (RIC, field) MultiIndex so the labels cannot be lost.
    pieces = []
    for field, frame in by_field.items():
        f = frame.copy()
        f.columns = pd.MultiIndex.from_product([[c for c in f.columns], [field]])
        pieces.append(f)
    df_options = pd.concat(pieces, axis=1).sort_index(axis=1)
    df_options.columns.names = ["RIC", "Field"]

    if merge_payload is not None:
        prior = merge_payload["options"]
        before = prior.columns.get_level_values(0).nunique()
        df_options = pd.concat([prior, df_options], axis=1).sort_index(axis=1)
        df_options = df_options.loc[:, ~df_options.columns.duplicated()]
        df_options.columns.names = ["RIC", "Field"]
        after = df_options.columns.get_level_values(0).nunique()
        print(f"merged: {before} existing series + new -> {after} total")
        args.fields = sorted(set(merge_payload.get("fields_requested", [])) |
                             set(args.fields))

    payload = {
        "stock": df_stock,
        "options": df_options,
        "ticker": args.root,
        "fetched_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "synthetic": False,
        "fields_requested": list(args.fields),
        "fields_returned": sorted(by_field.keys()),
        "window": [start_str, end_str],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(payload, f)

    print(f"\nwrote {args.out}  ({args.out.stat().st_size/1e6:.2f} MB)")
    print(f"  fields returned : {payload['fields_returned']}")
    print(f"  series          : {df_options.columns.get_level_values(0).nunique()}")
    print(f"  observations    : {int(df_options.notna().sum().sum())}")
    missing = [f for f in args.fields if f not in by_field]
    if missing:
        print(f"  NOT returned    : {missing}")
    print("\nnext: python scripts/build_static.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
