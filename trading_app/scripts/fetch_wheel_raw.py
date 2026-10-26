"""
Raw daily pull for the wheel: every monthly put and call the wheel could write
on one stock in 2019-2023, with no strategy rule applied. Selection happens
later, from quotes on the decision date only. Same request, log and states as
scripts/fetch_pmcc_raw.py, whose request() this reuses:

    returned       rows came back
    empty          the contract exists but has no rows in the window asked for
    never_listed   LSEG error 70005, "The universe is not found"
    failed         anything else, after retries with backoff

    python3 scripts/fetch_wheel_raw.py --root AMD --plan     # count contracts, no session
    python3 scripts/fetch_wheel_raw.py --root AMD --stop-after-failures 25
    python3 scripts/fetch_wheel_raw.py --root AMD --retry-failed

BANDS. A put is written below spot and a call above it, so each monthly
expiry gets puts from 0.70 to 1.02 times spot and calls from 0.97 to 1.40
times, over the sessions around the previous expiry, when the contract would
be opened. The put band reaches a 0.15-delta put at the implied vols of March
2020; the call band is the PMCC pull's.

STEPS. A grid is a list of (price below which it applies, increments), in the
dollars of the day; a strike is generated if it is a multiple of any of that
tier's increments. Only increments every listed series includes are asked for,
so a coarser grid costs precision in delta, not never-listed requests.

SEED. NVDA's calls were pulled for the PMCC into cache/pmcc_raw_NVDA on the
same band and grid. A contract already logged there is copied, with its rows,
instead of being requested again; so are the stock, dividend, earnings and
T-bill pulls.

SPLITS. A contract carries its strike after every split up to its expiry, with
unadjusted prices before the split. Sessions between splits are banded in
their own dollars, then expressed as the strike the RIC carries.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pickle
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_pmcc_raw import FIELDS, expiry_session, request, third_friday  # noqa: E402,F401
from trading_app.lib.ric import build_option_ric  # noqa: E402

START, END = dt.date(2019, 1, 2), dt.date(2023, 12, 29)
CALL_BAND, PUT_BAND = (0.97, 1.40), (0.70, 1.02)

UNDERLYINGS = {
    # 4:1 on 2021-07-20, inside the window; 10:1 on 2024-06-10, after every
    # monthly here. The calls repeat the PMCC pull's $2.50 grid so they seed
    # from it. Puts take $5 above $200, where every NVDA series lists $5.
    "NVDA": {"stock": "NVDA.O", "splits": [(dt.date(2021, 7, 20), 4.0), (dt.date(2024, 6, 10), 10.0)],
             "units_after": dt.date(2021, 7, 20), "seed": "pmcc_raw_NVDA",
             "C": [(math.inf, (2.5,))],
             "P": [(200.0, (2.5,)), (math.inf, (5.0,))]},
    # AMD: no split. Grids from the probe of listed strikes (see the commit).
    "AMD": {"stock": "AMD.O", "splits": [], "units_after": None, "seed": None,
            "C": [(25.0, (0.5,)), (100.0, (1.0,)), (math.inf, (5.0,))],
            "P": [(25.0, (0.5,)), (100.0, (1.0,)), (math.inf, (5.0,))]},
}


def cache_dir(root: str) -> Path:
    return ROOT.parent / "cache" / f"wheel_raw_{root}"


def steps_at(tiers, price: float) -> tuple:
    """The increments of the first tier whose bound is above `price`."""
    return next(steps for bound, steps in tiers if price < bound)


def strikes(px: pd.Series, lo_day, hi_day, band, tiers, expiry, splits) -> list[float]:
    """Every strike to request for one expiry, as the RIC carries it."""
    w = px[(px.index >= lo_day) & (px.index <= hi_day)]
    out = set()
    bounds = [dt.date.min] + [d for d, _ in splits] + [dt.date.max]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        part = w[(w.index >= lo) & (w.index < hi)]
        if part.empty:
            continue
        scale = float(np.prod([r for d, r in splits if hi <= d <= expiry]))
        for step in steps_at(tiers, float(part.mean())):
            k_lo = np.floor(part.min() * band[0] / step) * step
            k_hi = np.ceil(part.max() * band[1] / step) * step
            out |= {round(float(k) / scale, 4) for k in np.arange(k_lo, k_hi + step / 2, step)}
    return sorted(out)


def plan(closes: pd.Series, root: str) -> list[dict]:
    """Every contract to request, with the window to request it over."""
    cfg = UNDERLYINGS[root]
    sessions = [t.date() for t in closes.index]
    px = pd.Series(closes.to_numpy(float), index=sessions)
    months = pd.period_range(f"{START.year}-{START.month:02d}", f"{END.year + 1}-01", freq="M")
    expiries = [expiry_session(third_friday(p.year, p.month), sessions) for p in months]
    jobs = []
    for prev, exp in zip(expiries[:-1], expiries[1:]):
        lo_day, hi_day = prev - dt.timedelta(days=7), prev + dt.timedelta(days=7)
        for cp, band in (("P", PUT_BAND), ("C", CALL_BAND)):
            for k in strikes(px, lo_day, hi_day, band, cfg[cp], exp, cfg["splits"]):
                jobs.append({"kind": "monthly", "cp": cp, "expiry": exp, "strike": k,
                             "start": exp - dt.timedelta(days=50), "end": exp,
                             "ric": build_option_ric(root, exp, k, cp)})
    return jobs


def read_log(log: Path) -> dict:
    done = {}
    if log.exists():
        for line in log.read_text().splitlines():
            rec = json.loads(line)
            done[rec["ric"]] = rec
    return done


SEED_FILES = ("stock_unadjusted.pkl", "stock_unadjusted_ohlc.pkl", "aux.pkl", "splits.json")


def seed_files(cache: Path, seed_dir: Path) -> None:
    """The seed's stock, aux and splits pulls, where this cache has none."""
    for name in SEED_FILES:
        if (seed_dir / name).exists() and not (cache / name).exists():
            shutil.copy2(seed_dir / name, cache / name)


def seed(cache: Path, seed_dir: Path, jobs: list[dict]) -> int:
    """Copy every planned contract the seed already logged, with its rows."""
    theirs = read_log(seed_dir / "requests.jsonl")
    ours = read_log(cache / "requests.jsonl")
    take = [j for j in jobs if j["ric"] not in ours
            and theirs.get(j["ric"], {}).get("status") in ("returned", "empty", "never_listed")]
    if not take:
        return 0
    their_frames = pickle.loads((seed_dir / "options.pkl").read_bytes())
    frames_path = cache / "options.pkl"
    frames = pickle.loads(frames_path.read_bytes()) if frames_path.exists() else {}
    with (cache / "requests.jsonl").open("a") as log:
        for j in take:
            rec = {**theirs[j["ric"]], "kind": "monthly", "cp": j["cp"], "seeded_from": seed_dir.name}
            if j["ric"] in their_frames:
                frames[j["ric"]] = their_frames[j["ric"]]
            log.write(json.dumps(rec) + "\n")
    frames_path.write_bytes(pickle.dumps(frames))
    return len(take)


def pull_aux(ld, cache: Path, stock_ric: str) -> None:
    """Dividends as paid, earnings dates and the 3-month T-bill, in the form lib.wheel reads."""
    lo, hi = str(START - dt.timedelta(days=400)), str(END + dt.timedelta(days=60))
    div = ld.get_data(stock_ric, ["TR.DivExDate", "TR.DivUnadjustedGross", "TR.DivPayDate"],
                      {"SDate": lo, "EDate": hi})
    earnings = ld.get_data(stock_ric, ["TR.EventStartDate"], {"SDate": lo, "EDate": hi, "EventType": "RES"})
    tbill = ld.get_history(universe="US3MT=RR", fields=["MID_YLD_1"], interval="daily", start=lo, end=hi)
    (cache / "aux.pkl").write_bytes(pickle.dumps({"dividends": div, "tbill": tbill, "earnings": earnings,
                                                  "pulled_at": str(pd.Timestamp.now())}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, choices=sorted(UNDERLYINGS))
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--stop-after-failures", type=int, default=None,
                    help="stop after this many failures in a row (e.g. LSEG's daily cap); resume with --retry-failed")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    cfg = UNDERLYINGS[args.root]
    cache = cache_dir(args.root)
    cache.mkdir(parents=True, exist_ok=True)
    seed_dir = ROOT.parent / "cache" / cfg["seed"] if cfg["seed"] else None
    if seed_dir is not None:
        seed_files(cache, seed_dir)

    ld = None
    stock_path = cache / "stock_unadjusted.pkl"
    if not stock_path.exists() or not (cache / "aux.pkl").exists():
        if args.plan and stock_path.exists():
            pass
        else:
            import lseg.data as ld
            ld.open_session()
            if not stock_path.exists():
                raw = ld.get_history(universe=cfg["stock"], fields=["TRDPRC_1", "OPEN_PRC"], interval="daily",
                                     start=str(START - dt.timedelta(days=215)),
                                     end=str(END + dt.timedelta(days=34)),
                                     adjustments=["exchangeCorrection"])
                closes = pd.to_numeric(raw["TRDPRC_1"], errors="coerce").dropna()
                closes.index = pd.DatetimeIndex(closes.index)
                closes.to_pickle(stock_path)
                raw.to_pickle(cache / "stock_unadjusted_ohlc.pkl")
            if not (cache / "aux.pkl").exists():
                pull_aux(ld, cache, cfg["stock"])
    if cfg["units_after"] is not None and not (cache / "splits.json").exists():
        (cache / "splits.json").write_text(json.dumps(
            {"units_after": str(cfg["units_after"]), "splits": [[str(d), r] for d, r in cfg["splits"]]}))

    jobs = plan(pd.read_pickle(stock_path), args.root)
    print(f"{len(jobs)} contracts: {sum(j['cp'] == 'P' for j in jobs)} puts, "
          f"{sum(j['cp'] == 'C' for j in jobs)} calls", flush=True)
    if seed_dir is not None and not args.plan:
        print(f"{seed(cache, seed_dir, jobs)} copied from {seed_dir.name}", flush=True)

    log_path = cache / "requests.jsonl"
    done = read_log(log_path)
    if seed_dir is not None and args.plan:
        theirs = read_log(seed_dir / "requests.jsonl")
        done = {**{k: v for k, v in theirs.items() if v["status"] != "failed"}, **done}
    skip = {"returned", "empty", "never_listed"} | (set() if args.retry_failed else {"failed"})
    todo = [j for j in jobs if done.get(j["ric"], {}).get("status") not in skip]
    print(f"{len(todo)} to request ({sum(j['cp'] == 'P' for j in todo)} puts, "
          f"{sum(j['cp'] == 'C' for j in todo)} calls), {len(jobs) - len(todo)} already logged", flush=True)
    if args.plan or not todo:
        if ld is not None:
            ld.close_session()
        return 0

    if ld is None:
        import lseg.data as ld
        ld.open_session()
    frames_path = cache / "options.pkl"
    frames = pickle.loads(frames_path.read_bytes()) if frames_path.exists() else {}
    try:
        with log_path.open("a") as log:
            run_of_failures = 0
            for i, job in enumerate(todo, 1):
                status, got, detail = request(ld, job)
                run_of_failures = run_of_failures + 1 if status == "failed" else 0
                if got is not None:
                    frames[job["ric"]] = got
                log.write(json.dumps({
                    "ric": job["ric"], "kind": job["kind"], "cp": job["cp"], "expiry": str(job["expiry"]),
                    "strike": job["strike"], "status": status, "detail": detail,
                    "rows": 0 if got is None else int(len(got))}) + "\n")
                log.flush()
                if i % 100 == 0 or i == len(todo):
                    frames_path.write_bytes(pickle.dumps(frames))
                    print(f"  {i}/{len(todo)}  last {job['ric']} {status}", flush=True)
                if args.stop_after_failures and run_of_failures >= args.stop_after_failures:
                    print(f"  stopped after {run_of_failures} failures in a row at {i}/{len(todo)}: {detail}",
                          flush=True)
                    break
    finally:
        frames_path.write_bytes(pickle.dumps(frames))
        ld.close_session()

    counts = pd.Series([r["status"] for r in read_log(log_path).values()]).value_counts()
    print(counts.to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
