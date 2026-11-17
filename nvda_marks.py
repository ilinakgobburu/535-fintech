#!/usr/bin/env python3
"""
Tonight's mark of the two-sided variance trade's open NVDA cycle, from LSEG, for the vol-arb page.

Run by .github/workflows/nvda-marks.yml every weeknight at 23:45 New York time in winter (04:45 UTC), again with
--retry at 04:15 (09:15 UTC; it does nothing if the first run marked tonight's close), and from its Run workflow
button:
  1. read config/nvda_marks.yaml (the cycle as decided on its entry night: entry, expiry, legs, contracts, each
     model's p and direction; lib/nvda_marks.cycle_spec checks it, nothing is decided again), and check that
     lib/vol_arb.py and lib/vol_arb_wf.py, and every lib function they use, are the frozen backtest code
     config/vol_arb_live.yaml pins;
  2. log in to LSEG (lseg_login.py) for NVDA's unadjusted closes and opens from a week before the entry to D, the
     3-month T-bill, dividends, and both legs' daily closing bid and ask, one RIC per request, each request paced;
  3. mark the cycle with trading_app/trading_app/lib/nvda_marks.py (no I/O; tested in trading_app/tests/nvda_marks/),
     which runs lib/vol_arb.py's own cycle_path and run_trade on the market lib/vol_arb_live.market builds from the
     pulls: both sides at both fills, to D's close while the cycle is open, settled at the expiry;
  4. write docs/data/vol_arb_nvda_mark.json, to a temporary file beside it and renamed into place, so a failure
     leaves the old file as it was.

D is the last NYSE session whose close has passed (--as-of to pick another). When D's NVDA close or open, or a leg's
D row (at the expiry: its row on the session before, the expiry's close settling at intrinsic), is not on LSEG yet,
the data is not final: the run exits non-zero and the file keeps what it had. The retry (--retry) marks with what
LSEG has, a leg's missing quote carried as the backtest carries it (its last mark and IV), or the last complete
session when D's close is still missing, and says so in the file's notes and with data_complete: false. A leg whose
D row has no bid (nobody bidding) is final data, carried the same way with a note, on either run. A leg LSEG gives
no quotes for, or whose quotes start after the entry or stop before the last two closes the mark reads, stops every
run, the retry too (lib/nvda_marks.history_problems): marked over, its whole history would be carried. Once the file
holds the cycle settled with data_complete: true, later runs leave it as it is; one the retry settled is settled
again by the next run.

    python3 nvda_marks.py                         # tonight, as the Action runs it (LSEG from the Keychain)
    python3 nvda_marks.py --dry-run               # print tonight's mark, write nothing
    python3 nvda_marks.py --retry                 # the 04:15 run: only if tonight's close is not marked yet
    python3 nvda_marks.py --as-of 2026-11-16 --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "trading_app"))
from trading_app.lib import nvda_marks as NM  # noqa: E402
from trading_app.lib import vol_arb as VA  # noqa: E402
from trading_app.lib import vol_arb_live as VL  # noqa: E402
from trading_app.lib import vol_arb_wf as WF  # noqa: E402
from trading_app.lib import wheel_live as L  # noqa: E402
from trading_app.lib.ric import build_option_ric  # noqa: E402

try:
    import lseg.data as ld
except ImportError:          # the tests replace it; the Action installs lseg-data
    ld = None

CONFIG = ROOT / "config" / "nvda_marks.yaml"
PINS = ROOT / "config" / "vol_arb_live.yaml"         # the frozen backtest code's sha256 and lib function hashes
OUT = ROOT / "docs" / "data" / "vol_arb_nvda_mark.json"
NY = ZoneInfo("America/New_York")
PAUSE = 0.4         # seconds after every LSEG request (_paced): the course guideline is at most 3 a second
UNLISTED = "the universe is not found"              # LSEG's answer for a RIC it does not know (lower case)
LIBS = {"trading_app/trading_app/lib/vol_arb.py": VA, "trading_app/trading_app/lib/vol_arb_wf.py": WF}
LIB = Path(VA.__file__).resolve().parent


class NotFinal(Exception):
    """LSEG's data for D is not all there yet."""


def open_lseg():
    """The LSEG platform session (lseg_login.py: GitHub secrets, or the macOS Keychain)."""
    from lseg_login import open_session
    return open_session()


def now_ny() -> dt.datetime:
    return dt.datetime.now(NY).replace(tzinfo=None)


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def check_libs() -> None:
    """lib/vol_arb.py and lib/vol_arb_wf.py, and each lib function they use, must be the pinned frozen code."""
    lib = (yaml.safe_load(PINS.read_text()) or {}).get("lib") or {}
    want = lib.get("sha256") or {}
    for name, mod in LIBS.items():
        got = VL.sha256_hex(Path(mod.__file__).read_bytes())
        if want.get(name) != got:
            sys.exit(f"{name} is not the frozen backtest code (sha256 {got}, config/vol_arb_live.yaml records "
                     f"{want.get(name)}): nothing marked.")
    bad = VL.hash_differences(lib.get("functions") or {},
                              VL.dependency_hashes(lambda module: (LIB / f"{module}.py").read_text()))
    if bad:
        sys.exit("The backtest code's lib functions are not the recorded ones (config/vol_arb_live.yaml "
                 "lib.functions): " + "; ".join(bad) + ". Nothing marked.")


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


def stock_bars(ric: str, start: dt.date, end: dt.date) -> tuple[pd.Series, pd.Series]:
    """NVDA's unadjusted closes and opens from `start` to `end` (vol_arb_orders.stock_bars's request)."""
    raw = _frame(_paced(ld.get_history, universe=ric, fields=["TRDPRC_1", "OPEN_PRC"], interval="daily",
                        start=str(start), end=str(end), adjustments=["exchangeCorrection"]))
    if raw.empty or "TRDPRC_1" not in raw:
        raise NotFinal(f"LSEG returned no closes for {ric} from {start} to {end}")
    raw = raw[raw.index.date <= end]
    closes = pd.to_numeric(raw["TRDPRC_1"], errors="coerce")
    opens = pd.to_numeric(raw["OPEN_PRC"], errors="coerce") if "OPEN_PRC" in raw else pd.Series(float("nan"),
                                                                                                index=raw.index)
    return closes, opens


def tbill(ric: str, start: dt.date, end: dt.date) -> pd.Series:
    df = _frame(_paced(ld.get_history, universe=ric, fields=["MID_YLD_1"], interval="daily", start=str(start),
                       end=str(end)))
    s = pd.to_numeric(df.get("MID_YLD_1", pd.Series(dtype=float)), errors="coerce").dropna()
    s = s[s.index.date <= end]
    if s.empty:
        sys.exit(f"LSEG returned no {ric} yield from {start} to {end}: nothing marked.")
    return s


def dividends(ric: str, start: dt.date, end: dt.date) -> pd.Series:
    """Dividends as paid, the backtest's fields (aux.pkl): ex-date and unadjusted gross amount."""
    df = _paced(ld.get_data, ric, ["TR.DivExDate", "TR.DivUnadjustedGross", "TR.DivPayDate"],
                {"SDate": str(start), "EDate": str(end + dt.timedelta(days=60))})
    if df is None or df.empty or df.shape[1] < 3:
        return pd.Series(dtype=float)
    d = df.iloc[:, 1:3].copy()
    d.columns = ["ex_date", "amount"]
    d["ex_date"] = pd.to_datetime(d["ex_date"], errors="coerce")
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce")
    d = d.dropna()
    return pd.Series(d["amount"].to_numpy(float), index=pd.DatetimeIndex(d["ex_date"]).normalize())


def _quotes(ric: str, start: dt.date, end: dt.date) -> pd.DataFrame | None:
    """One option's daily closing BID and ASK; None when LSEG does not know the RIC. Any other error stops."""
    try:
        df = _paced(ld.get_history, universe=ric, fields=["BID", "ASK"], interval="daily", start=str(start),
                    end=str(end))
    except Exception as exc:
        msg = str(exc)
        if UNLISTED in msg.lower():
            return None
        first = msg.splitlines()[0][:200] if msg else repr(exc)
        sys.exit(f"LSEG error on {ric}, so nothing was marked: {first}")
    df = _frame(df)
    if df.empty or not {"BID", "ASK"} <= set(df.columns):
        return pd.DataFrame(columns=["bid", "ask"], index=pd.DatetimeIndex([]), dtype=float)
    df = df[df.index.date <= end]
    return pd.DataFrame({"bid": pd.to_numeric(df["BID"], errors="coerce"),
                         "ask": pd.to_numeric(df["ASK"], errors="coerce")}, index=df.index)


def leg_quotes(root: str, leg: dict, cp: str, expiry: dt.date, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    A leg's quotes from `start` to `end` under its live RIC, or, once LSEG has renamed it after the expiry, under the
    expired RIC (with ^ and the expiry's month and year), in load_market's frame with the live RIC, as the cycle
    names it. After the expiry an empty answer for the live RIC is taken as the rename too, and the expired RIC is
    asked; when that is not known either, the empty answer stands, and history_problems() stops the run.
    """
    q = _quotes(leg["ric"], start, end)
    if q is None or (q.empty and end >= expiry):
        expired = build_option_ric(root, expiry, leg["strike"], cp, expired=True)
        old = _quotes(expired, start, end)
        if old is None and q is None:
            sys.exit(f"LSEG knows neither {leg['ric']} nor {expired}, the cycle's {NM.LEG[cp]}: nothing marked.")
        if old is not None:
            q = old
            print(f"  {leg['ric']} is listed as {expired} now (expired)")
    rows = [{"date": d, "ric": leg["ric"], "cp": cp, "strike": leg["strike"], "expiry": pd.Timestamp(expiry),
             "bid": b, "ask": a} for d, b, a in zip(q.index, q["bid"], q["ask"])]
    return VL.quote_frame(rows)


# --------------------------------------------------------------------------- the file
def read_existing(path: Path):
    """The file as it stands, or None when it is missing or not a JSON object."""
    try:
        obj = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def same_cycle(obj, spec: dict) -> bool:
    c = (obj or {}).get("cycle") or {}
    return (obj or {}).get("strategy") == NM.STRATEGY and c.get("entry") == str(spec["entry"].date()) \
        and c.get("expiry") == str(spec["expiry"].date())


def write_atomic(path: Path, text: str) -> None:
    """`text` written whole: to a temporary file beside `path`, then renamed onto it. A failure leaves `path` as it
    was and no temporary file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def report(doc: dict) -> None:
    print(f"Mark at the {doc['as_of']} close ({doc['status']}): NVDA {doc['nvda']['close']:.2f} "
          f"({doc['nvda']['change_pct']:+.1f}% since {doc['cycle']['entry']}); realized vol since entry "
          f"{doc['vol']['realized_since_entry']:.1%}, ATM IV at entry {doc['vol']['atm_iv_at_entry']:.1%}")
    for d in NM.DIRECTIONS:
        m = doc["marks"][d]
        print(f"  {d:9}  mid {m['mid']['pnl']:+10,.2f}   stress {m['stress']['pnl']:+10,.2f}   "
              f"{m['mid']['hedge_trades']} hedge trades, {m['mid']['shares']:+d} shares")
    for n in doc["notes"]:
        print(f"  note: {n}")


# --------------------------------------------------------------------------- the night
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dry-run", action="store_true", help="print the mark, write nothing")
    ap.add_argument("--as-of", type=dt.date.fromisoformat, help="mark this session's close instead of D's")
    ap.add_argument("--retry", action="store_true",
                    help="the second nightly run: do nothing if D's close is already marked, else mark with what "
                         "LSEG has")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(CONFIG.read_text())
    try:
        spec = NM.cycle_spec(cfg)
    except NM.SpecError as exc:
        sys.exit(f"config/nvda_marks.yaml: {exc}. Nothing marked.")
    check_libs()
    entry, expiry = spec["entry"].date(), spec["expiry"].date()

    if args.as_of:
        if not VL.is_session(args.as_of):
            sys.exit(f"{args.as_of} is not an NYSE session.")
        day = args.as_of
    else:
        day = L.decision_dates(now_ny())[0]
    if day < entry:
        sys.exit(f"{day} is before the cycle's entry {entry}: nothing to mark.")
    target = min(day, expiry)
    existing = read_existing(OUT)
    if not args.dry_run and same_cycle(existing, spec) and existing.get("status") == "settled":
        if existing.get("data_complete") is True:
            print(f"{OUT.relative_to(ROOT)} already holds the cycle settled at its {expiry} expiry: nothing to do.")
            return 0
        if not args.retry:
            print(f"{OUT.relative_to(ROOT)} holds the cycle settled by the retry with a leg's missing close carried: "
                  f"settling it again")
    if args.retry and same_cycle(existing, spec) and existing.get("as_of") == str(target):
        print(f"Retry: the {target} close is already marked in {OUT.relative_to(ROOT)}; nothing to do.")
        return 0
    print(f"Marking the {entry} -> {expiry} cycle at the {target} close"
          + (" (the retry: with what LSEG has)" if args.retry else ""))

    u, data = cfg["underlying"], cfg["lseg"]
    start = entry - dt.timedelta(days=int(data["days_before_entry"]))
    session = open_lseg()
    try:
        try:
            closes, opens = stock_bars(u["ric"], start, target)
        except NotFinal as exc:
            sys.exit(f"{exc}: data not final, nothing marked.")
        rate = tbill(data["rate_ric"], entry - dt.timedelta(days=int(data["rate_days_before_entry"])), target)
        divs = dividends(u["ric"], entry - dt.timedelta(days=int(data["dividend_days_before_entry"])), target)
        quotes = pd.concat([leg_quotes(u["root"], spec["legs"][cp], cp, expiry, start, target)
                            for cp in (NM.PUT, NM.CALL)], ignore_index=True)
    finally:
        session.close()
        print("LSEG session closed")

    market = VL.market(closes, opens, rate, divs, (), quotes)
    problems = NM.data_problems(market, spec, target)
    extra = []
    as_of = pd.Timestamp(target)
    if problems and args.retry:
        print("Retry: " + "; ".join(problems))
        last = NM.last_complete_session(market, entry, target)
        if last is None:
            sys.exit("; ".join(problems) + ", and no complete session since the entry: nothing marked.")
        if last != as_of:
            extra.append(f"Marked by the retry at the {NM._day(last)} close: LSEG had no complete "
                         f"{NM._day(as_of)} bar for NVDA at the retry.")
            as_of = last
            problems = NM.data_problems(market, spec, as_of)          # the legs at that close
    # A leg with no quotes, or quotes that start late or stop early, is never marked over, not even by the retry:
    # its mark and IV would be carried from wherever the quotes stop, and every hedge trade after it would change.
    broken = NM.history_problems(market, spec, as_of)
    if broken:
        sys.exit("; ".join(broken) + ": the legs' quote history is not complete, so nothing was marked (the retry "
                 "does not mark over this).")
    if problems and not args.retry:
        sys.exit("; ".join(problems) + ": data not final, nothing marked (the retry asks again).")
    if problems:
        again = "a later run settles the cycle again once LSEG has it" if as_of >= spec["expiry"] else \
            "the next night's mark reads it again"
        extra.append(f"Marked by the retry with LSEG's data as it stood: " + "; ".join(problems)
                     + f" (the leg's last mark is carried, as the backtest carries it; {again}).")
    extra = [*NM.entry_differences(market, spec), *extra]
    try:
        mark = NM.mark_open_cycle(market, spec, as_of)
    except ValueError as exc:
        sys.exit(f"{exc}: nothing marked.")
    doc = NM.document(mark, spec, cfg, extra_notes=extra, generated_at=now_utc(), data_complete=not problems)
    report(doc)
    text = json.dumps(doc, indent=1) + "\n"
    if args.dry_run:
        print("Dry run, nothing written. docs/data/vol_arb_nvda_mark.json would be:")
        print(text, end="")
        return 0
    write_atomic(OUT, text)
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(text):,} bytes): {doc['status']}, as of {doc['as_of']}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"as_of={doc['as_of']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
