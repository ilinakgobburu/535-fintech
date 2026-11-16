#!/usr/bin/env python3
"""
Tonight's two-sided variance trade orders for the course server (Sanji), from LSEG and Cboe.

Run by .github/workflows/vol-arb-orders.yml every weeknight at 22:00 New York time (03:00 UTC) and from its
Run workflow button:
  1. read config/vol_arb_live.yaml, check lib/vol_arb.py and lib/vol_arb_wf.py against its hashes, and read
     the book (mkt/positions.json and mkt/fills.json, which Sanji writes). Stop if the book is not up to date
     with the orders already written;
  2. on an entry night (the first session after a monthly expiry), read the development panel and Cboe's VIX
     close, then log in to LSEG (lseg_login.py) for NVDA's closes and opens, dividends, the 3-month T-bill,
     the report dates and the closing bid/ask of the next monthly expiry's chain, one RIC per request;
     on any night with options in the book, the closes, dividends, T-bill and each held leg's quotes;
  3. decide with trading_app/trading_app/lib/vol_arb_live.py (no I/O; tested in
     trading_app/tests/vol_arb_live/), which computes with the backtest's own functions;
  4. write mkt/open_orders.json: the next session's orders, or [] with the reason printed.

Every failure exits non-zero before mkt/open_orders.json is touched, so the Action goes red and the file keeps
what it had.

    python3 vol_arb_orders.py                       # tonight, as the Action runs it (LSEG from the Keychain)
    python3 vol_arb_orders.py --dry-run             # print tonight's orders, write nothing
    python3 vol_arb_orders.py --dry-run --as-of 2026-11-16 --force-entry
                                                    # a one-off check: treat the latest close as an entry night
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "trading_app"))
from trading_app.lib import vol_arb as VA  # noqa: E402
from trading_app.lib import vol_arb_live as VL  # noqa: E402
from trading_app.lib import vol_arb_wf as WF  # noqa: E402
from trading_app.lib import wheel_live as L  # noqa: E402
from trading_app.lib.ric import build_option_ric  # noqa: E402

try:
    import lseg.data as ld
except ImportError:          # the tests replace it; the Action installs lseg-data
    ld = None

CONFIG = ROOT / "config" / "vol_arb_live.yaml"
MKT = ROOT / "mkt"
NY = ZoneInfo("America/New_York")
PAUSE = 0.4         # seconds between LSEG requests: the course guideline is at most 3 a second
LIBS = {"trading_app/trading_app/lib/vol_arb.py": VA, "trading_app/trading_app/lib/vol_arb_wf.py": WF}


def open_lseg():
    """The LSEG platform session (lseg_login.py: GitHub secrets, or the macOS Keychain)."""
    from lseg_login import open_session
    return open_session()


def now_ny() -> dt.datetime:
    return dt.datetime.now(NY).replace(tzinfo=None)


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def check_libs(cfg: dict) -> None:
    """The copies of the backtest code this process imported must hash to what the config records."""
    want = (cfg.get("lib") or {}).get("sha256") or {}
    for name, mod in LIBS.items():
        got = VL.sha256_hex(Path(mod.__file__).read_bytes())
        if want.get(name) != got:
            sys.exit(f"{name} is not the frozen backtest code (sha256 {got}, config/vol_arb_live.yaml records "
                     f"{want.get(name)}): no order written.")


# --------------------------------------------------------------------------- LSEG
def _frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).normalize()
    return df[~df.index.duplicated(keep="first")].sort_index()


def stock_bars(ric: str, day: dt.date, days: int) -> tuple[pd.Series, pd.Series]:
    """NVDA's unadjusted closes and opens up to D, as the backtest's cache was pulled; D's must be there."""
    raw = _frame(ld.get_history(universe=ric, fields=["TRDPRC_1", "OPEN_PRC"], interval="daily",
                                start=str(day - dt.timedelta(days=days)), end=str(day),
                                adjustments=["exchangeCorrection"]))
    if raw.empty or "TRDPRC_1" not in raw:
        sys.exit(f"LSEG returned no closes for {ric}: no order written.")
    raw = raw[raw.index.date <= day]
    closes = pd.to_numeric(raw["TRDPRC_1"], errors="coerce").dropna()
    if closes.empty or closes.index[-1].date() != day:
        last = closes.index[-1].date() if len(closes) else None
        sys.exit(f"LSEG's last close for {ric} is {last}, not {day}: data not final yet, no order written.")
    opens = pd.to_numeric(raw["OPEN_PRC"], errors="coerce") if "OPEN_PRC" in raw else None
    return closes, opens


def tbill(ric: str, day: dt.date) -> pd.Series:
    df = _frame(ld.get_history(universe=ric, fields=["MID_YLD_1"], interval="daily",
                               start=str(day - dt.timedelta(days=30)), end=str(day)))
    s = pd.to_numeric(df.get("MID_YLD_1", pd.Series(dtype=float)), errors="coerce").dropna()
    s = s[s.index.date <= day]
    if s.empty:
        sys.exit(f"LSEG returned no {ric} yield up to {day}: no order written.")
    return s


def dividends(ric: str, day: dt.date, days: int) -> pd.Series:
    """Dividends as paid, the backtest's fields (aux.pkl): ex-date and unadjusted gross amount."""
    df = ld.get_data(ric, ["TR.DivExDate", "TR.DivUnadjustedGross", "TR.DivPayDate"],
                     {"SDate": str(day - dt.timedelta(days=days)), "EDate": str(day + dt.timedelta(days=60))})
    if df is None or df.empty or df.shape[1] < 3:
        return pd.Series(dtype=float)
    d = df.iloc[:, 1:3].copy()
    d.columns = ["ex_date", "amount"]
    d["ex_date"] = pd.to_datetime(d["ex_date"], errors="coerce")
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce")
    d = d.dropna()
    return pd.Series(d["amount"].to_numpy(float), index=pd.DatetimeIndex(d["ex_date"]).normalize())


def _dates(df: pd.DataFrame | None) -> list:
    if df is None or df.empty:
        return []
    vals = df.drop(columns=[c for c in df.columns if str(c).lower() == "instrument"]).to_numpy().ravel()
    return [t for t in pd.to_datetime(pd.Series(vals), errors="coerce") if pd.notna(t)]


def report_dates(ric: str, day: dt.date) -> tuple[list, list]:
    """(the next expected report date, the actual report dates of the last two years)."""
    expected = _dates(ld.get_data(ric, ["TR.ExpectedReportDate"]))
    actual = _dates(ld.get_data(ric, ["TR.EPSActReportDate"],
                                {"SDate": str(day - dt.timedelta(days=730)), "EDate": str(day), "Frq": "FQ"}))
    return expected, actual


def option_quotes(ric: str, start: dt.date, end: dt.date) -> pd.DataFrame | None:
    """
    One option's daily closing BID and ASK, one RIC per request, paced. None when LSEG does not know the RIC
    (not listed); any other error stops the run, so a partial chain or book is never used.
    """
    try:
        df = ld.get_history(universe=ric, fields=["BID", "ASK"], interval="daily", start=str(start), end=str(end))
    except Exception as exc:
        msg = str(exc)
        if "not found" in msg.lower():
            return None
        first = msg.splitlines()[0][:200] if msg else repr(exc)
        sys.exit(f"LSEG error on {ric}, so no order was written: {first}")
    finally:
        time.sleep(PAUSE)
    df = _frame(df)
    if df.empty or not {"BID", "ASK"} <= set(df.columns):
        return pd.DataFrame(columns=["bid", "ask"])
    return pd.DataFrame({"bid": pd.to_numeric(df["BID"], errors="coerce"),
                         "ask": pd.to_numeric(df["ASK"], errors="coerce")}, index=df.index)


def chain(root: str, expiry: dt.date, plan: dict, day: dt.date, days: int) -> pd.DataFrame:
    """The expiry's puts and calls on D's close, in load_market's frame. Stops if no contract has a D row."""
    rows, unlisted, dated = [], 0, 0
    for cp in (VL.PUT, VL.CALL):
        for k in plan[cp]:
            ric = build_option_ric(root, expiry, k, cp, expired=False)
            q = option_quotes(ric, day - dt.timedelta(days=days), day)
            if q is None:
                unlisted += 1
                continue
            at = q[q.index.date == day]
            if at.empty:
                continue
            dated += 1
            rows.append({"date": pd.Timestamp(day), "ric": ric, "cp": cp, "strike": float(k),
                         "expiry": pd.Timestamp(expiry), "bid": at["bid"].iloc[0], "ask": at["ask"].iloc[0]})
    asked = len(plan[VL.PUT]) + len(plan[VL.CALL])
    frame = VL.quote_frame(rows)
    two = int((frame["mid"] > 0).sum())
    print(f"  {asked} asked: {dated} with a {day} row ({two} two-sided), {unlisted} not listed")
    if dated == 0:
        sys.exit(f"No {expiry} option has a {day} row on LSEG yet: data not final, no order written.")
    return frame


def vix_close(url: str, day: dt.date) -> float:
    """Cboe's VIX close on D, from its public history file."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode("utf-8")
    except Exception as exc:
        sys.exit(f"Cboe's VIX file could not be read ({exc}): no order written.")
    s = VL.parse_vix(text)
    if pd.Timestamp(day) not in s.index:
        last = s.index[-1].date() if len(s) else None
        sys.exit(f"Cboe's VIX file ends {last}, without {day}: the VIX feature can't be set, no order written.")
    return float(s.loc[pd.Timestamp(day)])


# --------------------------------------------------------------------------- the panel and the file
def load_panel(cfg: dict, path: Path):
    obj = read_json(path)
    if obj is None:
        sys.exit(f"{path} does not exist: export the development panel (export_vol_arb_panel.py) before an "
                 f"entry night. No order written.")
    try:
        panel = VL.load_panel(obj, cfg)
    except VL.PanelError as exc:
        sys.exit(f"The development panel {path} can't be used: {exc}. No order written.")
    env = (obj.get("environment") or {}).get("sklearn")
    try:
        import sklearn
        if env and env != sklearn.__version__:
            print(f"Note: the panel's development run used scikit-learn {env}, this is {sklearn.__version__}")
    except ImportError:
        pass
    print(f"Panel {path.name}: {len(panel)} rows ({int(panel['label'].notna().sum())} labelled), "
          f"{panel['entry'].min().date()} to {panel['entry'].max().date()}; rung {cfg['rung']}")
    return panel


def write_orders(written: list[dict], new: list[dict], day: dt.date, for_day: dt.date, dry_run: bool) -> None:
    orders, note = L.merge_orders(written, new, day, for_day)
    if note:
        print(note)
    if dry_run:
        print("Dry run, nothing written. mkt/open_orders.json would be:")
        print(json.dumps(orders, indent=2))
        return
    MKT.mkdir(exist_ok=True)
    (MKT / "open_orders.json").write_text(json.dumps(orders, indent=2) + "\n")
    print(f"Wrote mkt/open_orders.json: {len(orders)} order(s): {', '.join(o['id'] for o in orders) or '[]'}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"for_day={for_day.isoformat()}\n")


def report(plan: dict) -> None:
    e = plan["entry"]
    if e is not None:
        m = e["measure"]
        print(f"Entry {e['entry'].date()} -> {e['expiry'].date()}: {e['status']}"
              + (f"; F {m['F']:.2f}, sigma_ATM {m['sigma_atm']:.1%}, K0 {m['K0']:g}, {m.get('n_put')} puts and "
                 f"{m.get('n_call')} calls in the corridor" if "sigma_atm" in m else ""))
        if e["features"]:
            print("  features: " + ", ".join(f"{k} {v:+.4f}" for k, v in e["features"].items()))
        if e["legs"]:
            for cp in (VL.PUT, VL.CALL):
                leg = e["legs"][cp]
                if leg:
                    print(f"  {'put ' if cp == VL.PUT else 'call'} {leg['strike']:g} ({leg['ric']}): mid "
                          f"{leg['mid']:.2f}, IV {leg['iv']:.1%}, delta {leg['delta']:+.3f}, vega/share "
                          f"{leg['vega']:.3f} -> {leg['contracts']} contracts")
    d = plan["decision"]
    if d is not None:
        print(f"  {d['rung']}: {d['direction'] or 'no trade'}"
              + (f" (p = {d['p']:.3f} from {d['train_rows']} panel rows)" if d["p"] is not None else ""))
    for leg in plan["hedge"]["legs"]:
        print(f"  held {leg['qty']:+d} {leg['expiry']} {leg['strike']:g} {leg['cp']}: IV {leg['iv']:.1%} "
              f"(from {leg['iv_day'].date()}), delta {leg['delta']:+.3f}")
    for n in plan["notes"]:
        print(n)


# --------------------------------------------------------------------------- the night
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dry-run", action="store_true", help="print the orders, write nothing")
    ap.add_argument("--as-of", type=dt.date.fromisoformat, help="decide on this session's close (dry run only)")
    ap.add_argument("--force-entry", action="store_true",
                    help="treat the night as an entry night (dry run only), measuring the next entry's expiry")
    ap.add_argument("--panel", type=Path, help="another development panel file (default: the config's)")
    args = ap.parse_args(argv)
    if (args.as_of or args.force_entry) and not args.dry_run:
        sys.exit("--as-of and --force-entry are for a one-off check: add --dry-run.")

    cfg = yaml.safe_load(CONFIG.read_text())
    try:
        VL.check_config(cfg)
    except ValueError as exc:
        sys.exit(f"{exc}. No order written.")
    check_libs(cfg)
    u, data = cfg["underlying"], cfg["data"]

    if args.as_of:
        if not VL.is_session(args.as_of):
            sys.exit(f"{args.as_of} is not an NYSE session.")
        day, for_day = args.as_of, VL.next_session(args.as_of)
    else:
        day, for_day = L.decision_dates(now_ny())
    tonight = VL.night(day, args.force_entry)
    print(f"Quotes from the close of {day}, orders for {for_day}: "
          + (f"entry night, expiry {tonight['expiry']}{' (forced)' if tonight['forced'] else ''}"
             if tonight["entry"] else "not an entry night"))

    # The book, and whether it is up to date with what this script already wrote.
    written = read_json(MKT / "open_orders.json") or []
    if not isinstance(written, list):
        sys.exit("mkt/open_orders.json is not a JSON list")
    fills = read_json(MKT / "fills.json")
    for f in L.rejected_orders(written, fills):
        sys.exit(f"Sanji rejected {f.get('id')}: {f}. Fix the order format before trading on.")
    waiting = L.unresolved_orders(written, fills, day)
    if waiting:
        sys.exit(f"mkt/fills.json has no result yet for {', '.join(waiting)}: the book may be stale, no order written.")
    try:
        book = VL.read_book(read_json(MKT / "positions.json"), fills, u["conid"])
    except L.BookFormatError as exc:
        sys.exit(f"Can't read the book, so no order was written: {exc}")
    late = VL.expired_legs(book, day)
    if late:
        sys.exit(f"mkt/positions.json still lists options that expired by {day}: {late}. Book not updated yet.")
    held = VL.held_contracts(book)
    print(f"Book: {book.shares} shares, {len(held)} option contract(s) held")

    panel = None
    if tonight["entry"]:
        panel = load_panel(cfg, args.panel or (ROOT / cfg["panel"]))
    if not tonight["entry"] and not held:
        plan = VL.plan_night(cfg, day, for_day, book)
        report(plan)
        write_orders(written, plan["orders"], day, for_day, args.dry_run)
        return 0

    vix = np.nan
    if tonight["entry"]:
        vix = vix_close(data["vix_csv"], day)
        print(f"VIX closed at {vix:.2f} on {day} (Cboe)")
    session = open_lseg()
    try:
        closes, opens = stock_bars(u["ric"], day, int(data["closes_days"]))
        rate = tbill(data["rate_ric"], day)
        divs = dividends(u["ric"], day, int(data["dividends_days"]))
        earn, quotes = [], None
        if tonight["entry"]:
            expected, actual = report_dates(u["ric"], day)
            try:
                earn, why = VL.earnings_dates(expected, actual, day, tonight["expiry"],
                                              int(data["min_days_between_reports"]))
            except ValueError as exc:
                sys.exit(f"{exc}. No order written.")
            print(f"Report dates: {why}")
            strikes = VL.strike_plan(closes, VL.previous_expiry(tonight["expiry"]), day, cfg["chain"]["bands"],
                                     cfg["chain"]["grid"])
            print(f"{u['ric']} closed at {closes.iloc[-1]:.2f} on {day}; asking LSEG for {len(strikes['P'])} puts "
                  f"{strikes['P'][0]:g}-{strikes['P'][-1]:g} and {len(strikes['C'])} calls "
                  f"{strikes['C'][0]:g}-{strikes['C'][-1]:g} of {tonight['expiry']}")
            quotes = chain(u["root"], tonight["expiry"], strikes, day, int(cfg["chain"]["history_days"]))
        history = {}
        for (cp, expiry, strike) in held:
            ric = build_option_ric(u["root"], expiry, strike, cp, expired=False)
            q = option_quotes(ric, day - dt.timedelta(days=int(data["leg_history_days"])), day)
            if q is None:
                sys.exit(f"LSEG does not know {ric}, a contract the book holds: no order written.")
            history[(cp, expiry, strike)] = q
    finally:
        session.close()
        print("LSEG session closed")

    mkt = VL.market(closes, opens, rate, divs, earn, quotes)
    try:
        plan = VL.plan_night(cfg, day, for_day, book, entry_expiry=tonight["expiry"], mkt=mkt,
                             leg_history=history, vix_close=vix, panel=panel)
    except (ValueError, VL.StaleBook) as exc:
        sys.exit(f"{exc}. No order written.")
    report(plan)
    write_orders(written, plan["orders"], day, for_day, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
