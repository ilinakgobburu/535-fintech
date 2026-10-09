"""
Raw daily pull for the PMCC backtest: every monthly AAPL call expiry and every
January LEAP the backtest could touch in 2019-2023, with no strategy rule
applied. Selection happens later, from quotes on the decision date only.

One request per contract, so every contract ends in exactly one state:

    returned       rows came back
    empty          the contract exists but has no rows in the window asked for
    never_listed   LSEG error 70005, "The universe is not found"
    failed         anything else, after retries with backoff

    python3 scripts/fetch_pmcc_raw.py --root DAL   # pull, resuming from the log
    python3 scripts/fetch_pmcc_raw.py --plan       # count contracts, no session
    python3 scripts/fetch_pmcc_raw.py --retry-failed

STRIKES. Listed AAPL strikes are $5 apart a month or more from expiry and
$2.50 strikes are added later, so monthlies are generated on a $2.50 grid and
LEAPs on a $5 grid. $1 strikes, which appear only in the last days, are not
pulled.

THE 2020 SPLIT. A contract that was open on 2020-08-31 carries its POST-split
strike in the RIC for its whole life, with unadjusted pre-split prices before
that date. Strikes are therefore generated in pre-split dollars where the
band dates are pre-split, then divided by 4 for the RIC. Stock closes here are
unadjusted throughout.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
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

START, END = dt.date(2019, 1, 2), dt.date(2023, 12, 29)
FIELDS = ["BID", "ASK", "TRDPRC_1", "ACVOL_UNS", "OPINT_1", "IMP_VOLT"]

# Per underlying: the stock RIC, any split inside the window, and the strike
# grids to generate. A grid is a tuple of increments; a strike is generated if
# it is a multiple of any of them.
UNDERLYINGS = {
    "AAPL": {"stock": "AAPL.O", "split": (dt.date(2020, 8, 31), 4.0),
             "monthly": ((0.85, 1.35), (2.5,)), "leap": ((0.50, 1.05), (5.0,))},
    # DAL lists $0.50 and $1 strikes near the money and $1/$2.50/$5 on LEAPs.
    # Every short call this book can write is out of the money, hence the band.
    "DAL": {"stock": "DAL", "split": None,
            "monthly": ((0.97, 1.40), (0.5,)), "leap": ((0.50, 1.05), (1.0, 2.5))},
    # BIIB trades between about $200 and $480 with $5 strikes, $2.50 in places.
    "BIIB": {"stock": "BIIB.O", "split": None,
             "monthly": ((0.97, 1.40), (2.5,)), "leap": ((0.50, 1.05), (5.0,))},
}
ROOT_SYM = "AAPL"
CACHE = ROOT.parent / "cache" / "pmcc_raw"


def configure(root: str) -> None:
    """Point the module at one underlying. AAPL keeps its original cache path."""
    global ROOT_SYM, STOCK_RIC, SPLIT, SPLIT_RATIO, MONTHLY_BAND, MONTHLY_STEP
    global LEAP_BAND, LEAP_STEP, CACHE, LOG
    cfg = UNDERLYINGS[root]
    ROOT_SYM, STOCK_RIC = root, cfg["stock"]
    SPLIT, SPLIT_RATIO = cfg["split"] or (dt.date.max, 1.0)
    MONTHLY_BAND, MONTHLY_STEP = cfg["monthly"]
    LEAP_BAND, LEAP_STEP = cfg["leap"]
    CACHE = ROOT.parent / "cache" / ("pmcc_raw" if root == "AAPL" else f"pmcc_raw_{root}")
    LOG = CACHE / "requests.jsonl"


configure(ROOT_SYM)
RETRIES = 4


def third_friday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 15)
    return d + dt.timedelta(days=(4 - d.weekday()) % 7)


def expiry_session(day: dt.date, sessions: list[dt.date]) -> dt.date:
    """The third Friday, or the session before it when the market is closed."""
    known = [s for s in sessions if s <= day]
    return known[-1] if known and day <= sessions[-1] else day


def plan(closes: pd.Series) -> list[dict]:
    """Every contract to request, with the window to request it over."""
    sessions = [t.date() for t in closes.index]
    px = pd.Series(closes.to_numpy(float), index=sessions)

    def strikes(lo_day, hi_day, band, steps, expiry):
        w = px[(px.index >= lo_day) & (px.index <= hi_day)]
        if w.empty:
            return []
        out = set()
        # Pre-split and post-split sessions are banded separately, each in its
        # own dollars, then both expressed as the strike the RIC carries.
        for era, part in (("pre", w[w.index < SPLIT]), ("post", w[w.index >= SPLIT])):
            if part.empty:
                continue
            scale = SPLIT_RATIO if (era == "pre" and expiry >= SPLIT) else 1.0
            for step in steps:
                lo = np.floor(part.min() * band[0] / step) * step
                hi = np.ceil(part.max() * band[1] / step) * step
                out |= {round(float(k) / scale, 4)
                        for k in np.arange(lo, hi + step / 2, step)}
        return sorted(out)

    jobs = []
    months = pd.period_range("2019-01", "2024-01", freq="M")
    expiries = [expiry_session(third_friday(p.year, p.month), sessions) for p in months]
    for prev, exp in zip(expiries[:-1], expiries[1:]):
        # Band on the sessions around the previous expiry, when this contract
        # would be opened; request its last 50 days.
        lo_day, hi_day = prev - dt.timedelta(days=7), prev + dt.timedelta(days=7)
        for k in strikes(lo_day, hi_day, MONTHLY_BAND, MONTHLY_STEP, exp):
            jobs.append({"kind": "monthly", "expiry": exp, "strike": k,
                         "start": exp - dt.timedelta(days=50), "end": exp})
    for year in range(2020, 2026):
        exp = third_friday(year, 1)
        # A January LEAP can be opened from about 16 to 9 months before expiry.
        lo_day, hi_day = exp - dt.timedelta(days=490), exp - dt.timedelta(days=270)
        for k in strikes(max(lo_day, START), hi_day, LEAP_BAND, LEAP_STEP, exp):
            jobs.append({"kind": "leap", "expiry": exp, "strike": k,
                         "start": max(lo_day, START) - dt.timedelta(days=5),
                         "end": min(exp, END + dt.timedelta(days=5))})
    # A January expiry is both a monthly and a LEAP. One request per contract,
    # over the union of the two windows.
    merged: dict[str, dict] = {}
    for j in jobs:
        j["ric"] = build_option_ric(ROOT_SYM, j["expiry"], j["strike"], "C")
        seen = merged.get(j["ric"])
        if seen is None:
            merged[j["ric"]] = j
        else:
            seen.update(kind="leap", start=min(seen["start"], j["start"]),
                        end=max(seen["end"], j["end"]))
    return list(merged.values())


def request(ld, job: dict) -> tuple[str, pd.DataFrame | None, str]:
    """(status, frame, detail) for one contract, retrying dropped connections."""
    detail = ""
    for attempt in range(RETRIES):
        try:
            got = ld.get_history(universe=job["ric"], fields=FIELDS, interval="daily",
                                 start=str(job["start"]),
                                 end=str(job["end"] + dt.timedelta(days=1)))
        except Exception as exc:
            detail = f"{type(exc).__name__}: {str(exc)[:200]}".replace("\n", " ")
            if "70005" in detail:
                return "never_listed", None, detail
            time.sleep(2.0 * 2 ** attempt)
            continue
        if got is None or got.empty or not got.notna().any().any():
            return "empty", None, ""
        return "returned", got, ""
    return "failed", None, detail


def read_log() -> dict:
    done = {}
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            rec = json.loads(line)
            done[rec["ric"]] = rec
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="AAPL", choices=sorted(UNDERLYINGS))
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    configure(args.root)
    CACHE.mkdir(parents=True, exist_ok=True)

    stock_path = CACHE / "stock_unadjusted.pkl"
    if args.plan and stock_path.exists():
        closes = pd.read_pickle(stock_path)
        ld = None
    else:
        import lseg.data as ld
        ld.open_session()
        raw = ld.get_history(universe=STOCK_RIC, fields=["TRDPRC_1", "OPEN_PRC"], interval="daily",
                             start="2018-06-01", end="2024-02-01",
                             adjustments=["exchangeCorrection"])
        closes = pd.to_numeric(raw["TRDPRC_1"], errors="coerce").dropna()
        closes.index = pd.DatetimeIndex(closes.index)
        closes.to_pickle(stock_path)
        raw.to_pickle(CACHE / "stock_unadjusted_ohlc.pkl")

    jobs = plan(closes)
    print(f"{len(jobs)} contracts: "
          f"{sum(j['kind'] == 'monthly' for j in jobs)} monthly, "
          f"{sum(j['kind'] == 'leap' for j in jobs)} LEAP", flush=True)
    if args.plan:
        return 0

    done = read_log()
    skip = {"returned", "empty", "never_listed"} | (set() if args.retry_failed else {"failed"})
    todo = [j for j in jobs if done.get(j["ric"], {}).get("status") not in skip]
    print(f"{len(todo)} to request, {len(jobs) - len(todo)} already logged", flush=True)

    frames_path = CACHE / "options.pkl"
    frames = pickle.loads(frames_path.read_bytes()) if frames_path.exists() else {}
    try:
        with LOG.open("a") as log:
            for i, job in enumerate(todo, 1):
                status, got, detail = request(ld, job)
                if got is not None:
                    frames[job["ric"]] = got
                log.write(json.dumps({
                    "ric": job["ric"], "kind": job["kind"], "expiry": str(job["expiry"]),
                    "strike": job["strike"], "status": status, "detail": detail,
                    "rows": 0 if got is None else int(len(got))}) + "\n")
                log.flush()
                if i % 100 == 0 or i == len(todo):
                    frames_path.write_bytes(pickle.dumps(frames))
                    print(f"  {i}/{len(todo)}  last {job['ric']} {status}", flush=True)
    finally:
        frames_path.write_bytes(pickle.dumps(frames))
        ld.close_session()

    counts = pd.Series([r["status"] for r in read_log().values()]).value_counts()
    print(counts.to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
