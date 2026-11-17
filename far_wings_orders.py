#!/usr/bin/env python3
"""
Tonight's far-wing orders for the basket (NVDA and the long/short vol book's 30 names, each its own $100,000 book)
for the course server (Sanji), from LSEG.

Run by .github/workflows/far-wings-orders.yml every weeknight at 22:00 New York time (03:00 UTC), again with
--retry at 01:00 (06:00 UTC; it decides only the names the first run did not, so a name whose data was not final is
retried before the open), and from its Run workflow button:
  1. read config/far_wings_live.yaml, check its numbers against lib/far_wings_live.py and every lib function that
     module uses against the recorded hashes (lib_functions), read price_history.json (the conids Sanji tracks) and
     the book (mkt/positions.json and mkt/fills.json, which Sanji writes). A name with no conid, or one Sanji does
     not track, gets no orders (one line says why). Stop everything, writing nothing, if a file is malformed or the
     book is stale (an order of this strategy's with no result, an option listed past its expiry). A name whose own
     book does not add up (another strategy's position on it, a rejected order of its own) gets no orders tonight;
  2. for each name in turn, as the single-name script did for NVDA: on an entry night (the first session after a
     monthly expiry) log in to LSEG (lseg_login.py, one session for the night) for its closes, dividends, the
     3-month T-bill (once a night) and the closing bid/ask of the next monthly expiry's chain, one RIC per request,
     each request paced; on any night with options in its book, the closes, dividends, T-bill and each held leg's
     quotes; on a night with only shares, the closes. A request LSEG fails on (anything but "The universe is not
     found") is asked once more after RETRY_PAUSE. A name whose LSEG data is not final at 22:00 is left for the
     01:00 retry; still not final at the retry, it is skipped with a note and the run ends red;
  3. decide each name with trading_app/trading_app/lib/far_wings_live.py (no I/O; tested in
     trading_app/tests/far_wings_live/): its two wings' SHT limits at the closing mid on an entry night, and every
     night the band check on its book's net delta, a market order to zero shares-delta at the next open only when it
     is outside its $5,000 band;
  4. write mkt/open_orders.json once for every name (far_wings_live.compose_orders: the names decided tonight with
     their new orders, every other name with what the file had), or [] with the reasons printed, and record each
     decided name's night in data/far_wings_ledger.json (every order this script has written, by name and id), which
     step 1 uses to check that every position a name holds is what its own fills add up to.

Every failure that puts the whole book in doubt exits non-zero before mkt/open_orders.json is touched, so the Action
goes red and the file keeps what it had; so does a night on which no name could be decided (as when NVDA alone was
not final). The file and the ledger are each written to a temporary file first and renamed into place, the ledger
first, so a failure while writing them leaves both as they were. A name stopped by its own book (or by an unexpected
error on its data), or skipped at the retry (its entry or its hedge was not written), does not keep the others from
being written: the run ends 0 with `stopped=` or `skipped=` in GITHUB_OUTPUT, and the workflow's last step turns red
after the commit; a retry that decides nothing more but skipped or stopped a name exits non-zero. With
owns_open_orders: false in the config (the far wings retired) it writes nothing and exits 0.

THE DATES. D is the last NYSE session whose close is at or before the run, and every order is for the next session
after D, always (far_wings_live.run_dates; order.txt: 'for' is "The session the order belongs to. Not 'next'. A late
run must not fill Monday's order on Tuesday"). GitHub can start a scheduled run hours late (on 2026-11-17 the vol-arb's
03:00 UTC run ran at 11:23 UTC and its 06:00 retry at 15:29 UTC, which wrote "Quotes from the close of 2026-11-16,
orders for 2026-11-18": a stale order a day late). So a run, the 22:00 one or the retry, that finds the session for
D's orders already open (09:30 New York) when it would write, writes nothing to mkt/ or the ledger and exits non-zero,
before asking LSEG anything, and again if the session opens while it is asking: on an entry night that cycle's entry
is lost (it is placed only from the entry session's close, for the session after it), and every name's band check
waits for the next night's run. The retry's check for what is already written keys on D alone.

THE ONE-OFF FIRST ENTRY (config/far_wings_live.yaml's first_entry, her answer "Tomorrow (Wednesday)"): 2026-11-17's
close is an entry night once, into the 2026-12-18 expiry, orders for 2026-11-18, its chain banded on the closes of
2026-11-10 to 2026-11-17 (far_wings_live.night_of). The run that writes that night records in the ledger every name
it found tracked; those names are hedged only on 2026-11-23 (far_wings_live.entry_for), the others enter then. A
first-entry night refused as late writes nothing, and the cycle falls back to the regular 2026-11-23 entry.

When Sanji rejects one of a name's orders, that name stops every night (the file keeps the order, so the same check
fails again) until she has looked at it: fix what was wrong, then add the id to config/far_wings_live.yaml's
`acknowledged` with the reason. When he gives no result for one, every name stops the same way. The next run notes
it, goes on and rewrites the file without it. Another strategy's orders (the wheel's last put) are only noted.

    python3 far_wings_orders.py                       # tonight, as the Action runs it (LSEG from the Keychain)
    python3 far_wings_orders.py --dry-run             # print tonight's orders, write nothing
    python3 far_wings_orders.py --retry               # the 01:00 run: only the names tonight's run did not decide
    python3 far_wings_orders.py --dry-run --as-of 2026-11-16 --force-entry
                                                      # a one-off check: treat the latest close as an entry night
    python3 far_wings_orders.py --dry-run --only "AAPL,BRK B" --include-untracked
                                                      # a few names only, even before Sanji tracks their conids
    python3 far_wings_orders.py --print-lib-functions # the lib_functions lines, to re-record after an intended edit
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import time
import traceback
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
PRICE_HISTORY = Path("price_history.json")         # under ROOT: Sanji's prices, and which conids he tracks
LEDGER = Path(FW.LEDGER_PATH)          # under ROOT: every order this script has written, by name and id
NY = ZoneInfo("America/New_York")
PAUSE = 0.4         # seconds after every LSEG request (_paced): the course guideline is at most 3 a second
RETRY_PAUSE = 5.0   # seconds before the one repeat of a request LSEG failed on (any error but UNLISTED)
UNLISTED = "the universe is not found"              # LSEG's answer for a RIC it does not know (lower case)
LIB = Path(FW.__file__).resolve().parent            # trading_app/trading_app/lib


LATE = ("the session for {day}'s orders ({for_day}) is already open: nothing written (a late run must not fill a day "
        "late)")


class NotFinal(Exception):
    """One name's LSEG data is not final yet, or LSEG failed on one of its requests: no order for it from this run."""


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
def _once(call, *args, **kwargs):
    """One LSEG request, then the pause, whatever it returned or raised: at most 3 requests a second."""
    try:
        return call(*args, **kwargs)
    finally:
        time.sleep(PAUSE)


def _paced(call, *args, **kwargs):
    """
    One LSEG request, paced (_once). An error other than LSEG's "The universe is not found" (its answer for a RIC it
    does not know, which asking again cannot change) is asked once more, RETRY_PAUSE seconds later: one transient
    error (a 404, a timeout) among a name's 20 to 120 requests would otherwise leave the whole name for the 01:00
    retry. A second error is the caller's, as the first was before.
    """
    try:
        return _once(call, *args, **kwargs)
    except Exception as exc:
        if UNLISTED in str(exc).lower():
            raise
        what = kwargs.get("universe", args[0] if args else "?")
        print(f"  LSEG error on {what}: {_first_line(exc)}; asking once more in {RETRY_PAUSE:g} s")
        time.sleep(RETRY_PAUSE)
        return _once(call, *args, **kwargs)


def _first_line(exc: Exception) -> str:
    msg = str(exc)
    return msg.splitlines()[0][:200] if msg else repr(exc)


def _frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).normalize()
    return df[~df.index.duplicated(keep="first")].sort_index()


def stock_closes(ric: str, day: dt.date, days: int) -> pd.Series:
    """A name's unadjusted closes up to D, as the backtest's cache was pulled; D's must be there."""
    try:
        raw = _frame(_paced(ld.get_history, universe=ric, fields=["TRDPRC_1"], interval="daily",
                            start=str(day - dt.timedelta(days=days)), end=str(day), adjustments=["exchangeCorrection"]))
    except Exception as exc:
        raise NotFinal(f"LSEG error on {ric}: {_first_line(exc)}") from None
    if raw.empty or "TRDPRC_1" not in raw:
        raise NotFinal(f"LSEG returned no closes for {ric}")
    raw = raw[raw.index.date <= day]
    closes = pd.to_numeric(raw["TRDPRC_1"], errors="coerce").dropna()
    if closes.empty or closes.index[-1].date() != day:
        last = closes.index[-1].date() if len(closes) else None
        raise NotFinal(f"LSEG's last close for {ric} is {last}, not {day}: data not final yet")
    return closes


def tbill(ric: str, day: dt.date) -> pd.Series:
    """The 3-month T-bill up to D, every name's rate: without it no name can be decided, so it stops the run."""
    try:
        df = _frame(_paced(ld.get_history, universe=ric, fields=["MID_YLD_1"], interval="daily",
                           start=str(day - dt.timedelta(days=30)), end=str(day)))
    except Exception as exc:
        sys.exit(f"LSEG error on {ric} (every name's rate): {_first_line(exc)}: no order written.")
    if df.empty or "MID_YLD_1" not in df:
        sys.exit(f"LSEG returned no {ric} yield up to {day}: no order written.")
    s = pd.to_numeric(df["MID_YLD_1"], errors="coerce").dropna()
    s = s[s.index.date <= day]
    if s.empty:
        sys.exit(f"LSEG returned no {ric} yield up to {day}: no order written.")
    return s


def dividends(ric: str, day: dt.date, days: int) -> pd.Series:
    """Dividends as paid, the backtest's fields (aux.pkl): ex-date and unadjusted gross amount."""
    try:
        df = _paced(ld.get_data, ric, ["TR.DivExDate", "TR.DivUnadjustedGross", "TR.DivPayDate"],
                    {"SDate": str(day - dt.timedelta(days=days)), "EDate": str(day + dt.timedelta(days=60))})
    except Exception as exc:
        raise NotFinal(f"LSEG error on {ric}'s dividends: {_first_line(exc)}") from None
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
    own "The universe is not found", code 70005, under "No data to return": not listed); any other error, a transient
    "404 Not Found" included, is that name's data not final (NotFinal), so a partial chain or book is never used.
    """
    try:
        df = _paced(ld.get_history, universe=ric, fields=["BID", "ASK"], interval="daily", start=str(start),
                    end=str(end))
    except Exception as exc:
        if UNLISTED in str(exc).lower():
            return None
        raise NotFinal(f"LSEG error on {ric}: {_first_line(exc)}") from None
    df = _frame(df)
    if df.empty or not {"BID", "ASK"} <= set(df.columns):     # listed, but no quote in the window
        return pd.DataFrame(columns=["bid", "ask"], index=pd.DatetimeIndex([]), dtype=float)
    return pd.DataFrame({"bid": pd.to_numeric(df["BID"], errors="coerce"),
                         "ask": pd.to_numeric(df["ASK"], errors="coerce")}, index=df.index)


def chain(root: str, expiry: dt.date, plan: dict, day: dt.date, days: int, min_share: float,
          strict: bool = True) -> pd.DataFrame:
    """
    One name's expiry's puts and calls on D's close, in load_market's frame. Not final (NotFinal) when LSEG looks
    part-way through publishing D (FW.chain_problem): no contract with a D row, fewer than `min_share` of the listed
    contracts with a D row, or (`strict`, the 22:00 run) a listed contract with earlier rows but none for D; and
    (`strict`) when a right has D rows but not one two-sided quote (FW.one_sided_problem: D's bids not posted yet),
    which would otherwise lose the cycle's wing with a green run and a ledger night the retry takes as decided. The
    01:00 retry is not strict: three hours on, such a contract is taken as a hole in D's chain (no quote that close),
    and a right still with no two-sided quote as no eligible strike (no position in that wing this cycle, as the
    backtest has it), each said in the log.
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
    two = {cp: int(((frame["mid"] > 0) & (frame["cp"] == cp)).sum()) for cp in (FW.PUT, FW.CALL)}
    print(f"  {asked} asked: {dated} with a {day} row ({two[FW.PUT] + two[FW.CALL]} two-sided: {two[FW.PUT]} puts, "
          f"{two[FW.CALL]} calls), {unlisted} not listed"
          + (f", {len(stale)} with earlier rows but none for {day}" if stale else ""))
    problem = FW.chain_problem(expiry, day, asked - unlisted, dated, stale if strict else [], min_share)
    if problem:
        raise NotFinal(problem)
    one_sided = FW.one_sided_problem(frame, expiry, day)
    if one_sided and strict:
        raise NotFinal(f"{one_sided[0].upper()}{one_sided[1:]}: LSEG's {day} data is not final")
    if stale:
        print(f"  Note: chose without {len(stale)} listed contract(s) with no {day} quote ({', '.join(stale[:4])}"
              f"{', ...' if len(stale) > 4 else ''}): the retry takes them as holes in {day}'s chain")
    if one_sided:
        print(f"  Note: the retry still finds that {one_sided}: it takes that as no eligible strike, so that wing "
              f"has no position this cycle")
    return frame


class Lseg:
    """The night's one LSEG session, opened when the first name needs it, and the T-bill, asked for once."""

    def __init__(self):
        self.session, self.rate = None, None

    def open(self) -> None:
        # a login that fails stops the night (lseg_login exits itself on a refused login), rather than counting as
        # the first name's trouble and being tried again for every name after it
        if self.session is None:
            try:
                self.session = open_lseg()
            except Exception as exc:
                sys.exit(f"Can't open the LSEG session ({type(exc).__name__}): no order written.")

    def tbill(self, ric: str, day: dt.date) -> pd.Series:
        if self.rate is None:
            self.rate = tbill(ric, day)
        return self.rate

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
            print("LSEG session closed")


# --------------------------------------------------------------------------- the file
def _staged(path: Path, text: str) -> Path:
    """`text` in a new temporary file beside `path` (the same directory, so os.replace onto `path` is one rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.chmod(tmp, 0o644)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return tmp


def replace_files(contents: list[tuple[Path, str]]) -> None:
    """
    Each (path, text) written whole: every text is first written to a temporary file beside its path, and only once
    all are written is each moved onto its path (os.replace, a rename), in the order given. A failure before the
    renames leaves every path as it was, and no temporary file behind; one between two renames in the same checkout
    (most unlikely) leaves the earlier paths replaced, which is why the ledger goes first.
    """
    staged = []
    try:
        for path, text in contents:
            staged.append((_staged(path, text), path))
        for tmp, path in staged:
            os.replace(tmp, path)
    finally:
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)


def write_orders(written: list[dict], decided: dict, tickers: list, day: dt.date, for_day: dt.date, dry_run: bool,
                 ledger: dict, stopped: dict, skipped: dict | None = None, first: tuple | None = None) -> None:
    """
    mkt/open_orders.json after far_wings_live.compose_orders (every name in one file), and data/far_wings_ledger.json
    with tonight's night for each name decided. Every order that leaves the file is named, another strategy's (the
    wheel's last put, a vol-arb order) said to be so. Both files' contents are built before either is touched, and
    each is written to a temporary file beside it and then renamed onto it, the ledger first and
    mkt/open_orders.json last (replace_files): a failure before the renames leaves both as they were, so the file
    never holds orders the ledger does not know (which would stop every later night with a LedgerError once Sanji
    recorded their fills). `stopped` (names that got no orders because of their own book, an unexpected error on
    their data, or positions Sanji does not track) and `skipped` (names whose LSEG data was still not final at the
    01:00 retry, so their entry or hedge was not written) go to GITHUB_OUTPUT, for the workflow's last step to turn
    red after the commit. `first` (the one-off first entry's night): (its expiry, every name this run found tracked),
    recorded in the ledger's first_entry, so those names do not enter that cycle again.
    """
    orders, note = FW.compose_orders(written, decided, tickers, day, for_day)
    if note:
        print(note)
    else:
        FW.check_orders([o for o in orders if FW.owner(o.get("id"), tickers)])
        ids = [o.get("id") for o in orders]
        if len(set(ids)) != len(ids):
            raise ValueError(f"order ids repeat in mkt/open_orders.json: {ids}")
    kept = {o.get("id") for o in orders}
    for o in written:
        if o.get("id") not in kept:
            whose = "" if FW.owner(o.get("id"), tickers) else ", not this strategy's"
            print(f"Dropped {o.get('id')} (for {o.get('for')}{whose}) from mkt/open_orders.json")
    for o in orders:
        if not note and FW.owner(o.get("id"), tickers) is None:
            print(f"Kept {o.get('id')} (for {o.get('for')}, not this strategy's): an order for a session after {day}, "
                  f"left as it was written")
    if dry_run:
        print("Dry run, nothing written. mkt/open_orders.json would be:")
        print(json.dumps(orders, indent=2))
        return
    files = []
    if not note:
        for t in tickers:
            if t in decided:
                ledger = FW.record_night(ledger, t, decided[t], day, for_day)
        if first is not None:
            ledger = FW.record_first_entry(ledger, day, for_day, first[0], first[1])
        files.append((ROOT / LEDGER, json.dumps(ledger, indent=1) + "\n"))
    files.append((MKT / "open_orders.json", json.dumps(orders, indent=2) + "\n"))
    replace_files(files)
    if not note:
        n = sum(len(v) for v in decided.values())
        print(f"Recorded tonight's {n} order(s) for {len(decided)} name(s) in {LEDGER}")
        if first is not None:
            print(f"Recorded the one-off first entry in {LEDGER}: the {first[0]} cycle of "
                  f"{', '.join(ledger['first_entry']['names'])} began tonight; no name of these enters it again, and "
                  f"a name not tracked tonight enters at the regular entry")
    print(f"Wrote mkt/open_orders.json: {len(orders)} order(s): {', '.join(o['id'] for o in orders) or '[]'}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"for_day={for_day.isoformat()}\n")
            if stopped:
                f.write(f"stopped={', '.join(stopped)}\n")
            if skipped:
                f.write(f"skipped={', '.join(skipped)}\n")


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
def _holds(book) -> bool:
    return bool(book.shares or FW.held_contracts(book))


def _left(ledger, tradable, tonight) -> list:
    """On the first entry's night, the names a retry was to decide that the 22:00 run recorded as tracked."""
    rec = ledger.get("first_entry") if tonight.get("first") else None
    return [n["ticker"] for n in tradable if rec and n["ticker"] in rec["names"]]


def late_run(day, for_day, now, tonight, holders, earlier, left=()) -> str:
    """
    Why a run whose orders' session (`for_day`) had already opened at `now` (FW.session_open) writes nothing, and what
    that costs: on an entry night the cycle's entry, which is placed only from the entry session's close for the
    session after it (the registered rule: no entry from older quotes), and for every name holding positions
    (`holders`) its band check, which waits for the next night's run. `earlier` are the names the 22:00 run already
    wrote (a retry's), whose orders stand. On the one-off first entry's night: with nothing written that night, the
    first entry is lost and the cycle falls back to the regular entry; after a 22:00 run that wrote, the names it
    left for the retry (`left`) began their cycle that night and get no wings this cycle.
    """
    out = [f"Late run ({now:%Y-%m-%d %H:%M} New York time): " + LATE.format(day=day, for_day=for_day)]
    if earlier:
        out.append(f"the 22:00 run's orders for {', '.join(earlier)} stand as it wrote them")
    if tonight.get("first") and not earlier:
        e, _ = FW.next_entry(for_day)
        out.append(f"{day} was the one-off first entry's night, so the early first entry is lost: no wing is placed "
                   f"from {day}'s close for {for_day}, and the {tonight['expiry']} cycle falls back to the regular "
                   f"entry on {e} (orders for {FW.next_session(e)})")
    elif tonight.get("first"):
        if left:
            e, _ = FW.next_entry(for_day)
            out.append(f"{', '.join(left)}, left for this retry by the 22:00 run that found them tracked, began their "
                       f"{tonight['expiry']} cycle at the one-off first entry and get no wings this cycle (not on {e} "
                       f"either), as a name not decided at a regular entry's retry gets none")
    elif tonight["entry"]:
        e, x = FW.next_entry(for_day)
        out.append(f"{day} was an entry night, so this cycle's entry is lost: its wings are placed only from {day}'s "
                   f"close, for {for_day}, and not again this cycle (the next entry is {e}, for the {x} expiry)")
    if holders:
        out.append(f"the band check from {day}'s close for {', '.join(holders)} (any hedge, or the close of leftover "
                   f"shares) is not written either: it waits for the next night's run, on {for_day}'s close")
    return ". ".join(out) + "."


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dry-run", action="store_true", help="print the orders, write nothing")
    ap.add_argument("--as-of", type=dt.date.fromisoformat, help="decide on this session's close (dry run only)")
    ap.add_argument("--force-entry", action="store_true",
                    help="treat the night as an entry night (dry run only), choosing from the next entry's expiry")
    ap.add_argument("--retry", action="store_true",
                    help="the second nightly run: decide only the names tonight's orders were not written for")
    ap.add_argument("--only", help="comma-separated tickers to decide (dry run only), e.g. \"AAPL,BRK B,META\"")
    ap.add_argument("--include-untracked", action="store_true",
                    help="also decide names whose conid price_history.json does not track (dry run only)")
    ap.add_argument("--print-lib-functions", action="store_true",
                    help="print the lib_functions lines for config/far_wings_live.yaml and stop")
    args = ap.parse_args(argv)
    if args.print_lib_functions:
        print("lib_functions:")
        for k, v in lib_hashes().items():
            print(f"  {k}: {v}")
        return 0
    if (args.as_of or args.force_entry or args.only or args.include_untracked) and not args.dry_run:
        sys.exit("--as-of, --force-entry, --only and --include-untracked are for a one-off check: add --dry-run.")

    cfg = yaml.safe_load(CONFIG.read_text())
    try:
        FW.check_config(cfg)
    except ValueError as exc:
        sys.exit(f"{exc}. No order written.")
    if not cfg["owns_open_orders"] and not args.dry_run:
        # retired on purpose: a green run that writes nothing, so the schedule does not go red twice a night
        print("config/far_wings_live.yaml says owns_open_orders: false (the far-wing trade is retired from "
              "mkt/open_orders.json): nothing written. Delete the schedule in .github/workflows/far-wings-orders.yml "
              "to stop these runs.")
        return 0
    check_libs(cfg)
    tickers = [n["ticker"] for n in cfg["names"]]
    names = list(cfg["names"])
    if args.only:
        pick = [t.strip() for t in args.only.split(",") if t.strip()]
        unknown = [t for t in pick if t not in tickers]
        if unknown or not pick:
            sys.exit(f"--only names {unknown or pick} not in config/far_wings_live.yaml's names ({', '.join(tickers)})")
        names = [n for n in names if n["ticker"] in pick]

    now = now_ny()
    if args.as_of:
        if not FW.is_session(args.as_of):
            sys.exit(f"{args.as_of} is not an NYSE session.")
        day, for_day = args.as_of, FW.next_session(args.as_of)
    else:
        day, for_day = FW.run_dates(now)
    try:
        tracked = FW.tracked_conids(read_json(ROOT / PRICE_HISTORY))
    except ValueError as exc:
        sys.exit(f"Can't tell which conids Sanji tracks ({exc}; is {PRICE_HISTORY} there?): no order written.")
    try:
        ledger = FW.read_ledger(read_json(ROOT / LEDGER))
    except FW.LedgerError as exc:
        sys.exit(f"{exc}: no order written.")

    tradable, untracked = [], []
    for n in names:
        why = FW.untradable(n, tracked)
        if why is None:
            tradable.append(n)
        elif args.include_untracked and n["conid"] is not None:
            print(f"{n['ticker']}: {why}; decided anyway (--include-untracked): a live run gives it no orders")
            tradable.append(n)
        else:
            print(f"{n['ticker']}: {why}: no orders tonight")
            if n["conid"] is not None:
                untracked.append(n)
    if args.retry:
        done = {n["ticker"]: FW.decided_from(FW.name_ledger(ledger, n["ticker"]), day) for n in tradable}
        if all(done.values()):
            said = "; ".join(f"{t}: {d}" for t, d in done.items()) if len(done) != 1 else next(iter(done.values()))
            print(f"Retry: the orders for {for_day} from {day}'s close were already written ({said}); nothing to do.")
            return 0
        if any(done.values()):
            print(f"Retry: {', '.join(t for t, d in done.items() if d)} already written for {for_day} from {day}'s "
                  f"close; this run decides {', '.join(t for t, d in done.items() if not d)}.")
        else:
            print(f"Retry: nothing was written for {for_day} from {day}'s close yet, so this run decides.")
        tradable = [n for n in tradable if not done[n["ticker"]]]
    earlier = [t for t, d in done.items() if d] if args.retry else []
    first = FW.first_entry(cfg)
    tonight = FW.night_of(day, args.force_entry, first)
    print(f"Quotes from the close of {day}, orders for {for_day}: "
          + (f"entry night, expiry {tonight['expiry']}{' (forced)' if tonight['forced'] else ''}"
             if tonight["entry"] else "not an entry night")
          + (f" (the one-off first entry, her decision of 2026-11-17: \"{first['answer']}\"; the regular entry of "
             f"this cycle is {first['replaces']}; chain banded on the closes of {day - dt.timedelta(days=7)} to {day})"
             if tonight["first"] else ""))

    # The book, and whether it is up to date with what this script already wrote: the file's checks stop every name,
    # a name's own problems stop that name.
    written = read_json(MKT / "open_orders.json") or []
    if not isinstance(written, list):
        sys.exit("mkt/open_orders.json is not a JSON list")
    fills = read_json(MKT / "fills.json")
    if fills is not None and not isinstance(fills, list):
        sys.exit("mkt/fills.json is not a JSON list")
    stale, rejected, notes = FW.guard_problems(written, fills, day, tickers, cfg.get("acknowledged") or {})
    for n in notes:
        print(f"Note: {n}")
    if stale:
        sys.exit(". ".join(stale) + ". No order written.")
    positions = read_json(MKT / "positions.json")
    books = {}
    try:
        for n in cfg["names"]:
            if n["conid"] is not None:
                books[n["ticker"]] = FW.read_book(positions, fills, n["conid"])
    except FW.BookFormatError as exc:
        sys.exit(f"Can't read the book, so no order was written: {exc}")
    late = {t: FW.expired_legs(b, day) for t, b in books.items() if FW.expired_legs(b, day)}
    if late:
        listed = next(iter(late.values())) if len(late) == 1 else "; ".join(f"{t} {v}" for t, v in late.items())
        sys.exit(f"mkt/positions.json still lists options that expired by {day}: {listed}. Book not updated yet.")

    stopped = {}           # ticker: why it got no orders tonight and the run ends red
    for n in untracked:
        if _holds(books[n["ticker"]]):
            stopped[n["ticker"]] = (f"it holds positions ({books[n['ticker']].shares:+d} shares, "
                                    f"{len(FW.held_contracts(books[n['ticker']]))} option contract(s)) but Sanji does "
                                    f"not track its conid, so they were not hedged")
    for n in tradable:
        t = n["ticker"]
        why = list(rejected.get(t, []))
        try:
            problems = FW.book_problems(books[t], fills, FW.name_ledger(ledger, t), t, day)
        except FW.LedgerError as exc:
            problems = [str(exc)]
        if problems:
            why.append("The book is not this strategy's alone, or not up to date with its fills, so no order was "
                       "written: " + "; ".join(problems) + ". If another strategy's position is in it (the wheel's "
                       "last put), close it by hand (a BTC or SEL order of your own in mkt/open_orders.json) and run "
                       "again once Sanji has filled it")
        if why:
            stopped[t] = ". ".join(why)
    if not args.as_of and FW.session_open(for_day, now):
        if not args.dry_run:
            # before any LSEG request: the orders could only be written a day late
            sys.exit(late_run(day, for_day, now, tonight, [n["ticker"] for n in tradable if _holds(books[n["ticker"]])],
                              earlier, _left(ledger, tradable, tonight)))
        print(f"Note (dry run): the session for {day}'s orders ({for_day}) is already open, so a run now would write "
              f"nothing and end red")
    return _tonight(args, cfg, day, for_day, tonight, written, books, tradable, tickers, ledger, stopped, earlier)


def _decide(args, cfg, name, day, for_day, tonight, book, held, lseg: Lseg) -> dict:
    """One name's decision once its book has passed its checks: its inputs from LSEG, then its plan."""
    data = cfg["data"]
    if not tonight["entry"] and not held and not book.shares:
        return FW.plan_night(cfg, day, for_day, book, name=name)
    lseg.open()
    closes = stock_closes(name["ric"], day, int(data["closes_days"]))
    rate, divs, quotes, history = pd.Series(dtype=float), pd.Series(dtype=float), None, {}
    if tonight["entry"] or held:
        rate = lseg.tbill(data["rate_ric"], day)
        divs = dividends(name["ric"], day, int(data["dividends_days"]))
    if tonight["entry"]:
        try:
            strikes = FW.strike_plan(closes, tonight["band_ref"], day, cfg["chain"]["bands"], name["grid"])
        except ValueError as exc:
            raise NotFinal(str(exc)) from None
        print(f"{name['ric']} closed at {closes.iloc[-1]:.2f} on {day}; asking LSEG for {len(strikes['P'])} puts "
              f"{strikes['P'][0]:g}-{strikes['P'][-1]:g} and {len(strikes['C'])} calls "
              f"{strikes['C'][0]:g}-{strikes['C'][-1]:g} of {tonight['expiry']}")
        quotes = chain(name["root"], tonight["expiry"], strikes, day, int(cfg["chain"]["history_days"]),
                       float(cfg["chain"]["min_dated_share"]), strict=not args.retry)
    for (cp, expiry, strike) in held:
        ric = FW.option_ric(name["root"], expiry, strike, cp, expired=False)
        q = option_quotes(ric, day - dt.timedelta(days=int(data["leg_history_days"])), day)
        if q is None:
            raise NotFinal(f"LSEG does not know {ric}, a contract the book holds")
        history[(cp, expiry, strike)] = q
    mkt = FW.market(closes, None, rate, divs, (), quotes)
    return FW.plan_night(cfg, day, for_day, book, name=name, entry_expiry=tonight["expiry"] if tonight["entry"] else None,
                         mkt=mkt, leg_history=history)


def _tonight(args, cfg, day, for_day, tonight, written, books, tradable, tickers, ledger, stopped, earlier) -> int:
    """
    Each name in turn, then the one file: what was decided, what waits for the retry, what was skipped. `earlier`
    are the names the 22:00 run already decided (the retry leaves their orders as written).
    """
    decided, waiting, skipped, lseg = {}, {}, {}, Lseg()
    try:
        for n in tradable:
            t = n["ticker"]
            book = books[t]
            held = FW.held_contracts(book)
            print(f"{t} ({n['ric']}, conid {n['conid']}): book {book.shares} shares, {len(held)} option contract(s) held")
            if t in stopped:
                print(f"  {t} stopped, no orders tonight: {stopped[t]}")
                continue
            expiry, why = FW.entry_for(tonight, t, FW.name_ledger(ledger, t), book, ledger, day)
            if why:
                print(f"  {t}: {why}")
            mine = {**tonight, "entry": expiry is not None}
            try:
                plan = _decide(args, cfg, n, day, for_day, mine, book, held, lseg)
            except NotFinal as exc:
                after = "the Run button" if args.retry else "the retry, or the Run button"
                hedge = (f" Tonight's hedge was not written either: the book's {book.shares:+d} {t} shares stay as "
                         f"they are until a run succeeds ({after})."
                         if mine["entry"] and book.shares else
                         (" Its hedge was not written either." if _holds(book) else ""))
                if args.retry:
                    # the last run before the open: its entry (or its hedge) is lost for the night, so the run ends
                    # red (skipped= in GITHUB_OUTPUT, or the exit below when nothing more was decided)
                    skipped[t] = f"still not final at the retry ({exc}): no order written.{hedge}"
                    print(f"  {t} skipped: still not final at the retry ({exc}), so no order for it from "
                          f"{day}'s close.{hedge}")
                else:
                    waiting[t] = f"{exc}: no order written.{hedge}"
                    print(f"  {t} not final: {exc}; left for the 01:00 retry.{hedge}")
                continue
            except (ValueError, FW.StaleBook) as exc:
                stopped[t] = f"{exc}"
                print(f"  {t} stopped, no orders tonight: {exc}")
                continue
            except Exception as exc:
                # anything else from one name's data or plan (an odd LSEG answer) stops that name only, so the
                # others are still written and the run ends red after the commit; SystemExit (the T-bill, the
                # login) is not an Exception and still stops the whole night
                stopped[t] = f"unexpected {type(exc).__name__}: {_first_line(exc)}"
                print(f"  {t} stopped, no orders tonight: unexpected {type(exc).__name__} on its data or plan:")
                traceback.print_exc(file=sys.stdout)
                continue
            report(plan)
            decided[t] = plan["orders"]
    finally:
        lseg.close()

    summary = [f"{len(decided)} decided" + (f" ({', '.join(f'{t} {len(o)}' for t, o in decided.items())})"
                                             if decided else "")]
    if earlier:
        summary.append(f"written by the 22:00 run: {', '.join(earlier)}")
    if waiting:
        summary.append(f"left for the 01:00 retry: {', '.join(waiting)}")
    red = "the Action goes red after the commit" if decided else "red"
    if skipped:
        summary.append(f"SKIPPED at the retry ({red}): {', '.join(skipped)}")
    if stopped:
        summary.append(f"STOPPED ({red}): {', '.join(stopped)}")
    if not decided:
        print("Tonight by name: " + "; ".join(summary))
        stops = {t: why if "no order" in why.lower() else f"{why}. No order written." for t, why in stopped.items()}
        if earlier:
            # the retry decided nothing more: the 22:00 run's orders stand, nothing to write; red if a name was
            # skipped or stopped, since its orders for tonight were not written
            problems = {t: skipped.get(t) or stops[t] for t in tickers if t in skipped or t in stops}
            if problems:
                sys.exit("The retry decided nothing more, and "
                         + " ".join(f"{t}: {why}" for t, why in problems.items()))
            print("Retry: nothing more could be decided (see above): the 22:00 run's orders stand, nothing written.")
            return 0
        # no name could be decided: as the single-name script stopped, red with nothing written
        problems = {**waiting, **skipped, **stops}
        if len(problems) == 1:
            sys.exit(next(iter(problems.values())))
        if problems:
            sys.exit("No name could be decided tonight, so no order was written. "
                     + " ".join(f"{t}: {why}" for t, why in problems.items()))
        print("No name can trade tonight (see above): nothing written.")
        return 0
    now = now_ny()
    if not args.dry_run and FW.session_open(for_day, now):
        # the session opened while LSEG was being asked: what was decided would be a day late
        print("Tonight by name: " + "; ".join(summary))
        sys.exit(late_run(day, for_day, now, tonight, [n["ticker"] for n in tradable if _holds(books[n["ticker"]])],
                          earlier, _left(ledger, tradable, tonight)))
    first = (tonight["expiry"], [n["ticker"] for n in tradable]) if tonight["first"] else None
    write_orders(written, decided, tickers, day, for_day, args.dry_run, ledger, stopped, skipped, first)
    print("Tonight by name: " + "; ".join(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
