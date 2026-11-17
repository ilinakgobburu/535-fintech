#!/usr/bin/env python3
"""
Tonight's far-wing orders (NVDA) for the course server (Sanji), from LSEG.

Run by .github/workflows/far-wings-orders.yml every weeknight at 22:00 New York time (03:00 UTC), again with
--retry at 01:00 (06:00 UTC; it does nothing if the first run wrote tonight's orders, so a failed first run is
retried before the open), and from its Run workflow button:
  1. read config/far_wings_live.yaml, check its numbers against lib/far_wings_live.py and every lib function that
     module uses against the recorded hashes (lib_functions), and read the book (mkt/positions.json and
     mkt/fills.json, which Sanji writes). Stop if the book is not up to date with the orders already written, or
     holds anything this strategy's own fills do not account for;
  2. on an entry night (the first session after a monthly expiry), log in to LSEG (lseg_login.py) for NVDA's closes,
     dividends, the 3-month T-bill and the closing bid/ask of the next monthly expiry's chain, one RIC per request,
     each request paced; on any night with options in the book, the closes, dividends, T-bill and each held leg's
     quotes; on a night with only shares, the closes;
  3. decide with trading_app/trading_app/lib/far_wings_live.py (no I/O; tested in trading_app/tests/far_wings_live/):
     the two wings' SHT limits at the closing mid on an entry night, and every night the band check on the book's
     net delta, a market order to zero shares-delta at the next open only when it is outside the $5,000 band;
  4. write mkt/open_orders.json: the next session's orders, or [] with the reason printed, and record them in
     data/far_wings_ledger.json (every order this script has written, by id), which step 1 uses to check that every
     NVDA position in the book is what this strategy's own fills add up to (lib/far_wings_live.book_problems).

Every failure exits non-zero before mkt/open_orders.json is touched, so the Action goes red and the file keeps
what it had.

When Sanji rejects one of this strategy's orders, or gives no result for one, every night stops (the file keeps
the order, so the same check fails again) until she has looked at it: fix what was wrong, then add the id to
config/far_wings_live.yaml's `acknowledged` with the reason. The next run notes it, goes on and rewrites the file
without it. Another strategy's orders (the wheel's last put) are only noted.

    python3 far_wings_orders.py                       # tonight, as the Action runs it (LSEG from the Keychain)
    python3 far_wings_orders.py --dry-run             # print tonight's orders, write nothing
    python3 far_wings_orders.py --retry               # the 01:00 run: only if tonight's orders were not written
    python3 far_wings_orders.py --dry-run --as-of 2026-11-16 --force-entry
                                                      # a one-off check: treat the latest close as an entry night
    python3 far_wings_orders.py --print-lib-functions # the lib_functions lines, to re-record after an intended edit
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "trading_app"))
from trading_app.lib import far_wings_live as FW  # noqa: E402

try:
    import lseg.data as ld
except ImportError:          # the tests replace it; the Action installs lseg-data
    ld = None

CONFIG = ROOT / "config" / "far_wings_live.yaml"
MKT = ROOT / "mkt"
LEDGER = Path(FW.LEDGER_PATH)          # under ROOT: every order this script has written, by id
NY = ZoneInfo("America/New_York")
PAUSE = 0.4         # seconds after every LSEG request (_paced): the course guideline is at most 3 a second
UNLISTED = "the universe is not found"              # LSEG's answer for a RIC it does not know (lower case)
LIB = Path(FW.__file__).resolve().parent            # trading_app/trading_app/lib


def open_lseg():
    """The LSEG platform session (lseg_login.py: GitHub secrets, or the macOS Keychain)."""
    from lseg_login import open_session
    return open_session()


def now_ny() -> dt.datetime:
    return dt.datetime.now(NY).replace(tzinfo=None)


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def lib_hashes() -> dict:
    return FW.lib_functions(lambda module: (LIB / f"{module}.py").read_text())


def check_libs(cfg: dict) -> None:
    """
    Every lib function and constant lib/far_wings_live.py takes from another lib module (vol_arb_live's calendar,
    book and orders, wheel's option_greeks, wheel_live's order checks, ...), and everything those use in turn, must
    hash to what the config records, so an edit there cannot change tonight's wings, deltas or orders unnoticed.
    """
    bad = FW.hash_differences(cfg.get("lib_functions") or {}, lib_hashes())
    if bad:
        sys.exit("The lib functions lib/far_wings_live.py uses are not the recorded ones (config/far_wings_live.yaml "
                 "lib_functions): " + "; ".join(bad) + ". No order written. After an intended change, re-record them "
                 "with python3 far_wings_orders.py --print-lib-functions.")


# --------------------------------------------------------------------------- LSEG
def _paced(call, *args, **kwargs):
    """One LSEG request, then the pause, whatever it returned or raised: at most 3 requests a second."""
    try:
        return call(*args, **kwargs)
    finally:
        time.sleep(PAUSE)


def _frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).normalize()
    return df[~df.index.duplicated(keep="first")].sort_index()


def stock_closes(ric: str, day: dt.date, days: int) -> pd.Series:
    """NVDA's unadjusted closes up to D, as the backtest's cache was pulled; D's must be there."""
    raw = _frame(_paced(ld.get_history, universe=ric, fields=["TRDPRC_1"], interval="daily",
                        start=str(day - dt.timedelta(days=days)), end=str(day), adjustments=["exchangeCorrection"]))
    if raw.empty or "TRDPRC_1" not in raw:
        sys.exit(f"LSEG returned no closes for {ric}: no order written.")
    raw = raw[raw.index.date <= day]
    closes = pd.to_numeric(raw["TRDPRC_1"], errors="coerce").dropna()
    if closes.empty or closes.index[-1].date() != day:
        last = closes.index[-1].date() if len(closes) else None
        sys.exit(f"LSEG's last close for {ric} is {last}, not {day}: data not final yet, no order written.")
    return closes


def tbill(ric: str, day: dt.date) -> pd.Series:
    df = _frame(_paced(ld.get_history, universe=ric, fields=["MID_YLD_1"], interval="daily",
                       start=str(day - dt.timedelta(days=30)), end=str(day)))
    s = pd.to_numeric(df.get("MID_YLD_1", pd.Series(dtype=float)), errors="coerce").dropna()
    s = s[s.index.date <= day]
    if s.empty:
        sys.exit(f"LSEG returned no {ric} yield up to {day}: no order written.")
    return s


def dividends(ric: str, day: dt.date, days: int) -> pd.Series:
    """Dividends as paid, the backtest's fields (aux.pkl): ex-date and unadjusted gross amount."""
    df = _paced(ld.get_data, ric, ["TR.DivExDate", "TR.DivUnadjustedGross", "TR.DivPayDate"],
                {"SDate": str(day - dt.timedelta(days=days)), "EDate": str(day + dt.timedelta(days=60))})
    if df is None or df.empty or df.shape[1] < 3:
        return pd.Series(dtype=float)
    d = df.iloc[:, 1:3].copy()
    d.columns = ["ex_date", "amount"]
    d["ex_date"] = pd.to_datetime(d["ex_date"], errors="coerce")
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce")
    d = d.dropna()
    return pd.Series(d["amount"].to_numpy(float), index=pd.DatetimeIndex(d["ex_date"]).normalize())


def option_quotes(ric: str, start: dt.date, end: dt.date) -> pd.DataFrame | None:
    """
    One option's daily closing BID and ASK, one RIC per request, paced. None when LSEG does not know the RIC (its
    own "The universe is not found": not listed); any other error, a transient "404 Not Found" included, stops the
    run, so a partial chain or book is never used.
    """
    try:
        df = _paced(ld.get_history, universe=ric, fields=["BID", "ASK"], interval="daily", start=str(start),
                    end=str(end))
    except Exception as exc:
        msg = str(exc)
        if UNLISTED in msg.lower():
            return None
        first = msg.splitlines()[0][:200] if msg else repr(exc)
        sys.exit(f"LSEG error on {ric}, so no order was written: {first}")
    df = _frame(df)
    if df.empty or not {"BID", "ASK"} <= set(df.columns):     # listed, but no quote in the window
        return pd.DataFrame(columns=["bid", "ask"], index=pd.DatetimeIndex([]), dtype=float)
    return pd.DataFrame({"bid": pd.to_numeric(df["BID"], errors="coerce"),
                         "ask": pd.to_numeric(df["ASK"], errors="coerce")}, index=df.index)


def chain(root: str, expiry: dt.date, plan: dict, day: dt.date, days: int, min_share: float,
          strict: bool = True) -> pd.DataFrame:
    """
    The expiry's puts and calls on D's close, in load_market's frame. Stops when LSEG looks part-way through
    publishing D (FW.chain_problem): no contract with a D row, fewer than `min_share` of the listed contracts with
    a D row, or (`strict`, the 22:00 run) a listed contract with earlier rows but none for D. The 01:00 retry is
    not strict: three hours on, such a contract is taken as a hole in D's chain (no quote that close), and the
    wings are chosen without it.
    """
    rows, unlisted, dated, stale = [], 0, 0, []
    for cp in (FW.PUT, FW.CALL):
        for k in plan[cp]:
            ric = FW.option_ric(root, expiry, k, cp, expired=False)
            q = option_quotes(ric, day - dt.timedelta(days=days), day)
            if q is None:
                unlisted += 1
                continue
            at = q[q.index.date == day]
            if at.empty:
                if (q.index.date < day).any():
                    stale.append(ric)
                continue
            dated += 1
            rows.append({"date": pd.Timestamp(day), "ric": ric, "cp": cp, "strike": float(k),
                         "expiry": pd.Timestamp(expiry), "bid": at["bid"].iloc[0], "ask": at["ask"].iloc[0]})
    asked = len(plan[FW.PUT]) + len(plan[FW.CALL])
    frame = FW.quote_frame(rows)
    two = int((frame["mid"] > 0).sum())
    print(f"  {asked} asked: {dated} with a {day} row ({two} two-sided), {unlisted} not listed"
          + (f", {len(stale)} with earlier rows but none for {day}" if stale else ""))
    problem = FW.chain_problem(expiry, day, asked - unlisted, dated, stale if strict else [], min_share)
    if problem:
        sys.exit(f"{problem}: no order written.")
    if stale:
        print(f"  Note: chose without {len(stale)} listed contract(s) with no {day} quote ({', '.join(stale[:4])}"
              f"{', ...' if len(stale) > 4 else ''}): the retry takes them as holes in {day}'s chain")
    return frame


# --------------------------------------------------------------------------- the file
def write_orders(written: list[dict], new: list[dict], day: dt.date, for_day: dt.date, dry_run: bool,
                 ledger: dict, root: str) -> None:
    """
    mkt/open_orders.json after wheel_live.merge_orders, and data/far_wings_ledger.json with tonight's orders. Every
    order that leaves the file is named, another strategy's (the wheel's last put, a vol-arb order) said to be so.
    """
    orders, note = FW.merge_orders(written, new, day, for_day)
    if note:
        print(note)
    kept = {o.get("id") for o in orders}
    for o in written:
        if o.get("id") not in kept:
            whose = "" if str(o.get("id", "")).startswith(FW.own_prefix(root)) else ", not this strategy's"
            print(f"Dropped {o.get('id')} (for {o.get('for')}{whose}) from mkt/open_orders.json")
    if dry_run:
        print("Dry run, nothing written. mkt/open_orders.json would be:")
        print(json.dumps(orders, indent=2))
        return
    MKT.mkdir(exist_ok=True)
    (MKT / "open_orders.json").write_text(json.dumps(orders, indent=2) + "\n")
    print(f"Wrote mkt/open_orders.json: {len(orders)} order(s): {', '.join(o['id'] for o in orders) or '[]'}")
    if not note:
        path = ROOT / LEDGER
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(FW.record_night(ledger, new, day, for_day), indent=1) + "\n")
        print(f"Recorded tonight's {len(new)} order(s) in {LEDGER}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"for_day={for_day.isoformat()}\n")


def report(plan: dict) -> None:
    e = plan["entry"]
    if e is not None:
        print(f"Entry {e['entry'].date()} -> {e['expiry'].date()}: spot {e['spot']:.2f}, r {e['r']:.2%}, "
              f"q {e['q']:.2%}, T {e['T'] * 365:.0f} days, lot {e['lot']}")
        for cp in (FW.PUT, FW.CALL):
            leg = e[cp]
            if leg:
                print(f"  {'put ' if cp == FW.PUT else 'call'} {leg['strike']:g} ({leg['ric']}): bid {leg['bid']:.2f} "
                      f"ask {leg['ask']:.2f} mid {leg['mid']:.4f}, IV {leg['iv']:.1%}, delta {leg['delta']:+.4f}, "
                      f"vega/share {leg['vega']:.4f} -> {leg['contracts']} contracts")
    h = plan["hedge"]
    for leg in h["legs"]:
        print(f"  held {leg['qty']:+d} {leg['expiry']} {leg['strike']:g} {leg['cp']}: IV {leg['iv']:.1%} "
              f"(from {leg['iv_day'].date()}), delta {leg['delta']:+.4f}")
    for n in plan["notes"]:
        print(n)


# --------------------------------------------------------------------------- the night
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dry-run", action="store_true", help="print the orders, write nothing")
    ap.add_argument("--as-of", type=dt.date.fromisoformat, help="decide on this session's close (dry run only)")
    ap.add_argument("--force-entry", action="store_true",
                    help="treat the night as an entry night (dry run only), choosing from the next entry's expiry")
    ap.add_argument("--retry", action="store_true",
                    help="the second nightly run: do nothing if tonight's orders were already written")
    ap.add_argument("--print-lib-functions", action="store_true",
                    help="print the lib_functions lines for config/far_wings_live.yaml and stop")
    args = ap.parse_args(argv)
    if args.print_lib_functions:
        print("lib_functions:")
        for k, v in lib_hashes().items():
            print(f"  {k}: {v}")
        return 0
    if (args.as_of or args.force_entry) and not args.dry_run:
        sys.exit("--as-of and --force-entry are for a one-off check: add --dry-run.")

    cfg = yaml.safe_load(CONFIG.read_text())
    try:
        FW.check_config(cfg)
    except ValueError as exc:
        sys.exit(f"{exc}. No order written.")
    if not cfg["owns_open_orders"] and not args.dry_run:
        sys.exit("config/far_wings_live.yaml says owns_open_orders: false (the far-wing trade is retired from "
                 "mkt/open_orders.json): nothing written.")
    check_libs(cfg)
    u = cfg["underlying"]

    if args.as_of:
        if not FW.is_session(args.as_of):
            sys.exit(f"{args.as_of} is not an NYSE session.")
        day, for_day = args.as_of, FW.next_session(args.as_of)
    else:
        day, for_day = FW.decision_dates(now_ny())
    if args.retry:
        try:
            done = FW.decided(FW.read_ledger(read_json(ROOT / LEDGER)), day, for_day)
        except FW.LedgerError as exc:
            sys.exit(f"{exc}: no order written.")
        if done:
            print(f"Retry: the orders for {for_day} from {day}'s close were already written ({done}); nothing to do.")
            return 0
        print(f"Retry: nothing was written for {for_day} from {day}'s close yet, so this run decides.")
    tonight = FW.night(day, args.force_entry, for_day)
    print(f"Quotes from the close of {day}, orders for {for_day}: "
          + (f"entry night, expiry {tonight['expiry']}{' (forced)' if tonight['forced'] else ''}"
             if tonight["entry"] else "not an entry night"))
    if tonight["late"]:
        print(f"Note: {day} was an entry night, but {FW.next_session(day)} has opened since, so its wings are not "
              f"placed from {day}'s quotes for {for_day} (the backtest enters on the entry close, filled the next "
              f"session): no entry this cycle, only the band check.")

    # The book, and whether it is up to date with what this script already wrote.
    written = read_json(MKT / "open_orders.json") or []
    if not isinstance(written, list):
        sys.exit("mkt/open_orders.json is not a JSON list")
    fills = read_json(MKT / "fills.json")
    if fills is not None and not isinstance(fills, list):
        sys.exit("mkt/fills.json is not a JSON list")
    stops, notes = FW.guard_problems(written, fills, day, u["root"], cfg.get("acknowledged") or {})
    for n in notes:
        print(f"Note: {n}")
    if stops:
        sys.exit(". ".join(stops) + ". No order written.")
    try:
        book = FW.read_book(read_json(MKT / "positions.json"), fills, u["conid"])
    except FW.BookFormatError as exc:
        sys.exit(f"Can't read the book, so no order was written: {exc}")
    late = FW.expired_legs(book, day)
    if late:
        sys.exit(f"mkt/positions.json still lists options that expired by {day}: {late}. Book not updated yet.")
    try:
        ledger = FW.read_ledger(read_json(ROOT / LEDGER))
        problems = FW.book_problems(book, fills, ledger, u["root"], day)
    except FW.LedgerError as exc:
        sys.exit(f"{exc}: no order written.")
    if problems:
        sys.exit("The book is not this strategy's alone, or not up to date with its fills, so no order was written: "
                 + "; ".join(problems) + ". If another strategy's position is in it (the wheel's last put), close it "
                 "by hand (a BTC or SEL order of your own in mkt/open_orders.json) and run again once Sanji has filled it.")
    held = FW.held_contracts(book)
    print(f"Book: {book.shares} shares, {len(held)} option contract(s) held")
    try:
        return _tonight(args, cfg, day, for_day, tonight, written, book, held, ledger)
    except SystemExit as exc:
        # an entry night's input failing also leaves the hedge unwritten: say so, so the shares are not forgotten
        if tonight["entry"] and book.shares and isinstance(exc.code, str):
            raise SystemExit(f"{exc.code} Tonight's hedge was not written either: the book's {book.shares:+d} NVDA "
                             f"shares stay as they are until a run succeeds (the retry, or the Run button).") from None
        raise


def _tonight(args, cfg, day, for_day, tonight, written, book, held, ledger) -> int:
    """The night's decision once the book has passed its checks: the inputs, the plan, the file."""
    u, data = cfg["underlying"], cfg["data"]
    if not tonight["entry"] and not held and not book.shares:
        plan = FW.plan_night(cfg, day, for_day, book)
        report(plan)
        write_orders(written, plan["orders"], day, for_day, args.dry_run, ledger, u["root"])
        return 0

    session = open_lseg()
    try:
        closes = stock_closes(u["ric"], day, int(data["closes_days"]))
        rate, divs, quotes, history = pd.Series(dtype=float), pd.Series(dtype=float), None, {}
        if tonight["entry"] or held:
            rate = tbill(data["rate_ric"], day)
            divs = dividends(u["ric"], day, int(data["dividends_days"]))
        if tonight["entry"]:
            strikes = FW.strike_plan(closes, FW.previous_expiry(tonight["expiry"]), day, cfg["chain"]["bands"],
                                     cfg["chain"]["grid"])
            print(f"{u['ric']} closed at {closes.iloc[-1]:.2f} on {day}; asking LSEG for {len(strikes['P'])} puts "
                  f"{strikes['P'][0]:g}-{strikes['P'][-1]:g} and {len(strikes['C'])} calls "
                  f"{strikes['C'][0]:g}-{strikes['C'][-1]:g} of {tonight['expiry']}")
            quotes = chain(u["root"], tonight["expiry"], strikes, day, int(cfg["chain"]["history_days"]),
                           float(cfg["chain"]["min_dated_share"]), strict=not args.retry)
        for (cp, expiry, strike) in held:
            ric = FW.option_ric(u["root"], expiry, strike, cp, expired=False)
            q = option_quotes(ric, day - dt.timedelta(days=int(data["leg_history_days"])), day)
            if q is None:
                sys.exit(f"LSEG does not know {ric}, a contract the book holds: no order written.")
            history[(cp, expiry, strike)] = q
    finally:
        session.close()
        print("LSEG session closed")

    mkt = FW.market(closes, None, rate, divs, (), quotes)
    try:
        plan = FW.plan_night(cfg, day, for_day, book, entry_expiry=tonight["expiry"] if tonight["entry"] else None,
                             mkt=mkt, leg_history=history)
    except (ValueError, FW.StaleBook) as exc:
        sys.exit(f"{exc}. No order written.")
    report(plan)
    write_orders(written, plan["orders"], day, for_day, args.dry_run, ledger, u["root"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
