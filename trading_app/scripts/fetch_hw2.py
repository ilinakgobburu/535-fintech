"""
Pull the covered-call panel: AAPL hourly stock bars plus the near-the-money
call chain for each weekly expiry in the window.

Three things here are not in the HW1 fetcher, each because the data said so.

1. ZERO-PADDED EXPIRY DAY.
   The HW2 handout says "DAY not zero-padded". That is wrong, and two of the
   three AAPL sample RICs it prints do not resolve because of it:

       AAPLF52619000.U^F26   handout form   LDError
       AAPLF052619000.U^F26  zero-padded    400 observations
       AAPLH72620500.U^H26   handout form   LDError
       AAPLH072620500.U^H26  zero-padded    478 observations

   The third sample, AAPLG172620000.U^G26, expires on the 17th, so the rule
   never bites and it resolves either way. Every testable case fails; every
   padded correction works. This matters here specifically: Aug 7 and Sep 4
   are single-digit Fridays, so a literal reading loses 2 of 10 cycles with no
   error message at all. lib/ric.py already builds the padded form.

2. THE WEEK IS DERIVED FROM THE TAPE, NOT FROM date_range(freq="W-FRI").
   "Buy Monday, expire Friday" is not a calendar rule. In this window Jun 19
   (Juneteenth) and Jul 3 (Jul 4 observed) are closed, and those weeks expire
   on the THURSDAY -- AAPLG0226*.U^G26 resolves, the Friday RIC does not. We
   take the first and last session the underlying actually printed in each
   week, so a holiday shifts both legs instead of silently dropping a cycle.

3. MULTI-FIELD PULLS ARE CHECKED, NOT AVOIDED.
   HW1 pulled one field per request because a multi-field response that loses
   all but one field collapses to FLAT columns of bare RICs -- a shape
   indistinguishable from a single-field pull, which is how a cache once ended
   up TRDPRC_1-only with no record of the loss. The collapse is detectable:
   a healthy multi-field response has columns.nlevels == 2. We assert that,
   and fall back to per-field requests only for the batches that collapse.
   Same guarantee as HW1, at a fifth of the requests.

    python3 scripts/fetch_hw2.py --dry-run    # candidate count, no session
    python3 scripts/fetch_hw2.py              # real pull
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

DEFAULT_OUT = ROOT / "trading_app" / "data" / "covered_call_AAPL.pkl"

OPTION_FIELDS = ["BID", "ASK", "TRDPRC_1", "OPEN_PRC", "HIGH_1", "LOW_1",
                 "ACVOL_UNS", "NUM_MOVES"]
STOCK_FIELDS = ["TRDPRC_1", "OPEN_PRC", "HIGH_1", "LOW_1", "BID", "ASK",
                "ACVOL_UNS"]

# Options quote only during the regular session. Probed on AAPL: option bars
# run 13:00-20:00 UTC while the stock carries 08:00-23:00, and the 20:00 option
# bar is a one-trade closing stub (ACVOL_UNS=4, NUM_MOVES=1 on 2026-08-31).
SESSION_START_UTC = 13
SESSION_END_UTC = 20


def trading_weeks(stock_index: pd.DatetimeIndex) -> list[dict]:
    """
    Split the underlying's own session calendar into (entry, expiry) weeks.

    Entry is the first session of the ISO week, expiry the last. This is the
    whole holiday story: no hardcoded Friday, no assumption that a week has
    five days.
    """
    days = pd.Series(sorted({d.date() for d in stock_index}))
    iso = days.map(lambda d: (d.isocalendar().year, d.isocalendar().week))
    weeks = []
    for key, grp in days.groupby(iso):
        sessions = list(grp)
        entry, expiry = sessions[0], sessions[-1]
        if entry == expiry:
            continue  # a one-session week cannot hold a Monday->Friday cycle
        weeks.append({
            "iso": f"{key[0]}-W{key[1]:02d}",
            "entry": entry,
            "expiry": expiry,
            "sessions": len(sessions),
            "short_week": len(sessions) < 5,
        })
    return weeks


def chain_rics(root: str, expiry: dt.date, lo: float, hi: float,
               step: float, band: float) -> list[str]:
    """Calls along the chain, banded to where the stock actually went."""
    k_lo = np.floor((lo - band) / step) * step
    k_hi = np.ceil((hi + band) / step) * step
    strikes = np.arange(k_lo, k_hi + step / 2, step)
    return [build_option_ric(root, expiry, float(k), "C") for k in strikes]


def _as_ric_field(got, batch: list[str], fields: list[str]) -> pd.DataFrame | None:
    """
    Normalise ANY LSEG response into a (RIC, Field) MultiIndex, or return None
    when the shape genuinely cannot be resolved.

    This exists because a FLAT response's labels mean the opposite thing
    depending on how many RICs you asked for. Probed directly:

        1 RIC,  3 fields -> columns ['BID','ASK','TRDPRC_1'], columns.name = the RIC
        1 RIC,  1 field  -> columns ['BID'],                   columns.name = the RIC
        2 RICs, 1 field  -> columns [ric, ric],                columns.name = 'BID'

    So flat columns carry whichever axis has more than one member, and when
    BOTH are singletons the columns are FIELDS. That is a trap for the
    bisection above, which is a pure performance optimisation and yet drives
    batches down to size one -- flipping the meaning of the response
    underneath it. The first version of this fetcher labelled field names as
    RICs for every bisected batch and reported 26 live series out of 20
    requested, which is how the bug was noticed at all.

    The defence is to decide by MEMBERSHIP, not by position: we know exactly
    which RICs we asked for and exactly which fields, so we test the labels
    against those two sets and refuse to guess when neither matches.
    """
    if got is None or got.empty:
        return None
    cols = got.columns
    fset, bset = set(fields), set(batch)

    if cols.nlevels == 2:
        lvl0 = {str(v) for v in cols.get_level_values(0)}
        got = got.copy()
        if lvl0 <= fset and not (lvl0 <= bset):   # (Field, RIC) ordering
            got.columns = cols.swaplevel(0, 1)
        got.columns.names = ["RIC", "Field"]
        return got

    labels = [str(c) for c in cols]
    name = str(cols.name) if cols.name is not None else ""

    if set(labels) <= fset and name in bset:      # one RIC, columns are fields
        got = got.copy()
        got.columns = pd.MultiIndex.from_tuples([(name, l) for l in labels],
                                                names=["RIC", "Field"])
        return got

    if set(labels) <= bset and name in fset:      # one field, columns are RICs
        got = got.copy()
        got.columns = pd.MultiIndex.from_tuples([(l, name) for l in labels],
                                                names=["RIC", "Field"])
        return got

    if len(batch) == 1 and set(labels) <= fset:   # RIC absent from columns.name
        got = got.copy()
        got.columns = pd.MultiIndex.from_tuples([(batch[0], l) for l in labels],
                                                names=["RIC", "Field"])
        return got

    return None


def _relabel_per_field(ld, batch, fields, start, end):
    """
    Re-request a batch one field at a time and label locally.

    Only used when a MULTI-RIC multi-field response came back flat, which
    means fields were lost and the response cannot tell us which. HW1's fix,
    applied to the cases that actually need it rather than to every request.
    """
    pieces = []
    for f in fields:
        try:
            one = ld.get_history(universe=batch, fields=[f],
                                 start=start, end=end, interval="1h")
        except Exception:
            continue
        norm = _as_ric_field(one, batch, [f])
        if norm is None:
            continue
        norm = norm.dropna(how="all", axis=1)
        if not norm.empty:
            pieces.append(norm)
    if not pieces:
        return None
    out = pd.concat(pieces, axis=1).sort_index(axis=1)
    out.columns.names = ["RIC", "Field"]
    return out


def pull_batch(ld, batch: list[str], fields: list[str],
               start: str, end: str, stats: dict) -> pd.DataFrame | None:
    """
    One request per batch, with the failure modes handled SEPARATELY -- which
    is the whole trick, because they look alike and want opposite fixes.

      request THROWS   -> the batch contains a RIC that never existed. Almost
                          every batch has some, since we generate strikes past
                          where the stock actually went. BISECT: split and
                          retry each half, so N dead RICs cost O(N log B)
                          requests instead of collapsing the batch to one
                          request per (RIC, field). Only a batch of one is
                          genuinely dead.

      response is FLAT -> ambiguous. For a batch of ONE this is the normal,
                          healthy shape and _as_ric_field resolves it. For a
                          batch of MANY it means fields were lost and cannot
                          be identified from the response, so we re-request
                          per field and label locally.
    """
    try:
        got = ld.get_history(universe=batch, fields=fields,
                             start=start, end=end, interval="1h")
    except Exception:
        stats["threw"] += 1
        if len(batch) == 1:
            stats["dead_rics"].append(batch[0])
            return None
        h = len(batch) // 2
        parts = [x for x in (pull_batch(ld, batch[:h], fields, start, end, stats),
                             pull_batch(ld, batch[h:], fields, start, end, stats))
                 if x is not None and not x.empty]
        if not parts:
            return None
        out = pd.concat(parts, axis=1).sort_index(axis=1)
        out.columns.names = ["RIC", "Field"]
        return out

    if got is None or got.empty:
        return None

    if got.columns.nlevels == 1 and len(batch) > 1:
        stats["collapsed"] += 1
        return _relabel_per_field(ld, batch, fields, start, end)

    norm = _as_ric_field(got, batch, fields)
    if norm is None:
        stats["unresolved"] += 1
        return _relabel_per_field(ld, batch, fields, start, end)
    return norm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="AAPL")
    ap.add_argument("--stock", default="AAPL.O")
    ap.add_argument("--start", default="2026-06-26")
    ap.add_argument("--end", default="2026-09-05")
    ap.add_argument("--strike-step", type=float, default=2.50)
    ap.add_argument("--band", type=float, default=20.0,
                    help="dollars past the week's stock range to still generate strikes")
    ap.add_argument("--lookback-days", type=int, default=21,
                    help="how far before expiry to pull each contract")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--pause", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"window {args.start} -> {args.end}  root={args.root}  stock={args.stock}")

    if args.dry_run:
        days = pd.bdate_range(args.start, args.end)
        weeks = trading_weeks(pd.DatetimeIndex(days))
        n = sum(len(chain_rics(args.root, w["expiry"], 290, 320,
                               args.strike_step, args.band)) for w in weeks)
        print(f"{len(weeks)} weeks, ~{n} call RICs "
              f"(~{-(-n // args.batch_size)} batches) using an assumed $290-320 range")
        for w in weeks:
            print(f"  {w['iso']}  {w['entry']} -> {w['expiry']}")
        return 0

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

    print("fetching underlying (hourly)...")
    df_stock = ld.get_history(universe=[args.stock], fields=STOCK_FIELDS,
                              start=args.start, end=args.end, interval="1h")
    if df_stock is None or df_stock.empty:
        print("no underlying history")
        ld.close_session()
        return 3
    df_stock.index = pd.DatetimeIndex(df_stock.index)
    print(f"  {len(df_stock)} hourly bars over "
          f"{df_stock.index.min()} -> {df_stock.index.max()}")

    # Regular session only, matched to the option bars.
    hrs = df_stock.index.hour
    rth = df_stock[(hrs >= SESSION_START_UTC) & (hrs <= SESSION_END_UTC)]
    print(f"  {len(rth)} bars inside {SESSION_START_UTC}:00-{SESSION_END_UTC}:00 UTC")

    weeks = trading_weeks(pd.DatetimeIndex(rth.index))
    print(f"\n{len(weeks)} trading weeks:")
    for w in weeks:
        flag = "  <- SHORT WEEK" if w["short_week"] else ""
        print(f"  {w['iso']}  {w['entry']} ({w['entry'].strftime('%a')})"
              f" -> {w['expiry']} ({w['expiry'].strftime('%a')})"
              f"  {w['sessions']} sessions{flag}")

    highs = pd.to_numeric(rth["HIGH_1"], errors="coerce")
    lows = pd.to_numeric(rth["LOW_1"], errors="coerce")
    idx_dates = pd.Series([d.date() for d in rth.index], index=rth.index)

    stats = {"threw": 0, "collapsed": 0, "unresolved": 0, "dead_rics": []}
    all_frames, per_expiry = [], {}
    for w in weeks:
        expiry = w["expiry"]
        first = expiry - dt.timedelta(days=args.lookback_days)
        mask = (idx_dates >= first) & (idx_dates <= expiry)
        if not mask.any():
            continue
        lo, hi = float(lows[mask.values].min()), float(highs[mask.values].max())
        rics = chain_rics(args.root, expiry, lo, hi, args.strike_step, args.band)
        print(f"\n{w['iso']} expiry {expiry}: stock ${lo:.2f}-${hi:.2f}, "
              f"{len(rics)} strikes")

        batches = [rics[i:i + args.batch_size]
                   for i in range(0, len(rics), args.batch_size)]
        got_series = 0
        for i, batch in enumerate(batches, 1):
            frame = pull_batch(ld, batch, OPTION_FIELDS,
                               str(first), str(expiry + dt.timedelta(days=1)),
                               stats)
            if frame is not None and not frame.empty:
                # The check that caught the label-flip bug, kept permanently:
                # every RIC label must be one we actually asked for. A frame
                # that reports more series than the batch had RICs is not a
                # lucky bonus, it is mislabelled columns.
                back = set(map(str, frame.columns.get_level_values(0)))
                stray = back - set(batch)
                if stray:
                    raise RuntimeError(
                        f"response carried {len(stray)} label(s) not in the "
                        f"requested batch: {sorted(stray)[:5]} -- refusing to "
                        f"write a mislabelled cache")
                all_frames.append(frame)
                got_series += len(back)
            print(f"    batch {i}/{len(batches)}  live series so far {got_series}",
                  flush=True)
            if args.pause:
                time.sleep(args.pause)
        per_expiry[str(expiry)] = got_series

    ld.close_session()

    if not all_frames:
        print("no option data at all -- nothing written")
        return 4

    df_options = pd.concat(all_frames, axis=1).sort_index(axis=1)
    df_options = df_options.loc[:, ~df_options.columns.duplicated()]
    df_options.columns.names = ["RIC", "Field"]

    payload = {
        "stock": df_stock,
        "options": df_options,
        "ticker": args.root,
        "stock_ric": args.stock,
        "fetched_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "window": [args.start, args.end],
        "interval": "1h",
        "session_utc": [SESSION_START_UTC, SESSION_END_UTC],
        "weeks": weeks,
        "series_per_expiry": per_expiry,
        "fetch_stats": {k: (v if k != "dead_rics" else sorted(set(v)))
                        for k, v in stats.items()},
        "option_fields": OPTION_FIELDS,
        "stock_fields": STOCK_FIELDS,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(payload, f)

    n_series = df_options.columns.get_level_values(0).nunique()
    print(f"\nwrote {args.out}  ({args.out.stat().st_size / 1e6:.2f} MB)")
    print(f"  call series  : {n_series}")
    print(f"  observations : {int(df_options.notna().sum().sum())}")
    print(f"  fields       : {sorted(set(df_options.columns.get_level_values(1)))}")
    print(f"  requests that threw : {stats['threw']}  "
          f"(bisected; {len(set(stats['dead_rics']))} RICs never existed)")
    print(f"  flat responses      : {stats['collapsed']}  "
          f"(re-pulled per field and relabelled locally)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
