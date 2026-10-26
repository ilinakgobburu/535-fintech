"""
Rank config/mr_screen.yaml's universe, the S&P 500 as it stood on 2023-12-29, by how strongly its
monthly moves reversed over 2014-2023, and pick the five stocks for the wheel's class run
(config/wheel_mr.yaml). Every choice is read from the YAML, committed before this first runs.

    python3 scripts/mr_screen.py          # pull (resuming from the cache), rank, check feasibility, pick

The measures, on monthly log returns between month-end closes:

    VR  = var(overlapping 3-month returns) / (3 x var(monthly returns))     below 1: moves reverse
    rho = lag-1 autocorrelation of monthly returns                          below 0: moves reverse

A stock ranks only if both agree; the ranking is by VR, lowest first.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_wheel_raw import open_session  # noqa: E402

CFG_PATH = REPO / "config" / "mr_screen.yaml"
RAW = REPO / "cache" / "mr_screen_raw.pkl"
OUT = REPO / "cache" / "mr_screen.json"
AS_OF = "2023-12-29"
RUN_SPAN = ("2024-01-02", "2026-10-16")
MAX_CLOSE = 950.0
MIN_MONTHS = 114
SPLIT_RATIOS = (1.5, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0)
BATCH = 50


# --------------------------------------------------------------------------
# the measures
# --------------------------------------------------------------------------

def monthly_returns(close: pd.Series, start, end) -> pd.Series:
    """
    Log returns between month-end closes, one per month of [start, end], the first from the
    month-end before `start`. A month without a close is NaN, and so are the returns either side
    of it: a return never spans two months.
    """
    c = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    c.index = pd.DatetimeIndex(c.index)
    months = pd.period_range(pd.Period(pd.Timestamp(start), "M") - 1, pd.Period(pd.Timestamp(end), "M"), freq="M")
    me = c.groupby(c.index.to_period("M")).last().reindex(months)
    return np.log(me.astype(float)).diff().iloc[1:]


def measures(r: pd.Series) -> dict:
    """Variance ratio over three months, lag-1 autocorrelation, months scored, annualised volatility."""
    one = r.dropna()
    if len(one) < 3:
        return {"months": len(one), "vr": np.nan, "rho": np.nan, "vol": np.nan}
    three = r.rolling(3).sum().dropna()          # consecutive months only: a NaN breaks the window
    return {"months": int(len(one)),
            "vr": float(three.var(ddof=1) / (3.0 * one.var(ddof=1))),
            "rho": float(r.autocorr(lag=1)),
            "vol": float(one.std(ddof=1) * np.sqrt(12))}


def ranked(table: dict) -> list[str]:
    """Eligible stocks on which both measures say 'reverses', lowest variance ratio first."""
    keep = [t for t, m in table.items()
            if m["months"] >= MIN_MONTHS and np.isfinite(m["vr"]) and np.isfinite(m["rho"]) and m["rho"] < 0]
    return sorted(keep, key=lambda t: (table[t]["vr"], t))


# --------------------------------------------------------------------------
# feasibility over the run window
# --------------------------------------------------------------------------

def adjustments(default_close: pd.Series, unadjusted_close: pd.Series) -> dict:
    """
    Days on which the ratio of LSEG's default close to its unadjusted close moved by more than 1%,
    each classed as a split (a whole ratio from SPLIT_RATIOS, either way) or as another adjustment.
    """
    ratio = (pd.to_numeric(default_close, errors="coerce")
             / pd.to_numeric(unadjusted_close, errors="coerce")).dropna().sort_index()
    step = (ratio / ratio.shift(1)).dropna()
    splits, other = [], []
    for day, f in step[(step - 1.0).abs() > 0.01].items():
        x = f if f >= 1 else 1.0 / f
        is_split = any(abs(x / s - 1.0) < 0.005 for s in SPLIT_RATIOS)
        (splits if is_split else other).append((str(pd.Timestamp(day).date()), float(f)))
    return {"splits": splits, "other": other}


def feasible(default_close: pd.Series, unadjusted_close: pd.Series, run_end: str = RUN_SPAN[1]) -> dict:
    """Conditions (a) to (c) of config/mr_screen.yaml; (d) is checked after the option pull."""
    unadj = pd.to_numeric(unadjusted_close, errors="coerce").dropna().sort_index()
    why = []
    last = pd.Timestamp(unadj.index.max()) if len(unadj) else pd.NaT
    if pd.isna(last) or last < pd.Timestamp(run_end):
        why.append(f"stopped trading before {run_end} (last close {last.date() if pd.notna(last) else 'none'})")
    adj = adjustments(default_close, unadjusted_close)
    if adj["other"]:
        why.append("contracts adjusted other than for a split: " + ", ".join(f"{d} x{f:.3f}" for d, f in adj["other"]))
    top = float(unadj.max()) if len(unadj) else np.nan
    if not np.isfinite(top) or top > MAX_CLOSE:
        why.append(f"closed above ${MAX_CLOSE:.0f} (highest {top:.2f})")
    return {"feasible": not why, "why": "; ".join(why), "splits": adj["splits"],
            "max_close": top, "last_close_date": str(last.date()) if pd.notna(last) else None}


# --------------------------------------------------------------------------
# pulls
# --------------------------------------------------------------------------

def frame_to_series(got: pd.DataFrame, rics: list[str]) -> dict:
    """One TRDPRC_1 series per RIC from a get_history frame, whatever its column layout."""
    out = {}
    if got is None or got.empty:
        return out
    if isinstance(got.columns, pd.MultiIndex):
        for ric in rics:
            if ric in got.columns.get_level_values(0):
                out[ric] = pd.to_numeric(got[ric]["TRDPRC_1"], errors="coerce").dropna()
    elif len(rics) == 1:
        out[rics[0]] = pd.to_numeric(got.iloc[:, 0], errors="coerce").dropna()
    else:
        for ric in rics:
            if ric in got.columns:
                out[ric] = pd.to_numeric(got[ric], errors="coerce").dropna()
    return {k: v for k, v in out.items() if len(v)}


def pull_prices(ld, raw: dict, start: str, end: str, max_failures: int = 5) -> None:
    """Daily closes for every constituent not yet cached, in batches, one by one when a batch fails."""
    todo = [r for r in raw["constituents"] if r not in raw["px"] and r not in raw["missing"]]
    failures = 0
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        try:
            got = frame_to_series(ld.get_history(universe=batch, fields=["TRDPRC_1"], interval="daily",
                                                 start=start, end=end), batch)
        except Exception:
            got = {}
            for ric in batch:
                try:
                    got.update(frame_to_series(ld.get_history(universe=ric, fields=["TRDPRC_1"],
                                                              interval="daily", start=start, end=end), [ric]))
                except Exception as exc:
                    if "429" in str(exc) or "Too many" in str(exc):
                        failures = max_failures
                        break
                time.sleep(0.2)
        raw["px"].update(got)
        raw["missing"].extend(r for r in batch if r not in got and failures < max_failures)
        pd.to_pickle(raw, RAW)
        print(f"  prices {min(i + BATCH, len(todo))}/{len(todo)}", flush=True)
        if failures >= max_failures:
            print("  stopped: request cap reached; run again to resume", flush=True)
            return


def pull_feasibility(ld, ric: str) -> dict:
    lo, hi = RUN_SPAN[0], "2026-10-23"
    default = ld.get_history(universe=ric, fields=["TRDPRC_1"], interval="daily", start=lo, end=hi)
    unadjusted = ld.get_history(universe=ric, fields=["TRDPRC_1"], interval="daily", start=lo, end=hi,
                                adjustments=["exchangeCorrection"])
    return {"default": frame_to_series(default, [ric]).get(ric, pd.Series(dtype=float)),
            "unadjusted": frame_to_series(unadjusted, [ric]).get(ric, pd.Series(dtype=float))}


def main() -> int:
    warnings.filterwarnings("ignore")
    cfg = yaml.safe_load(CFG_PATH.read_text())
    start, end = (str(x) for x in cfg["window"])
    raw = pd.read_pickle(RAW) if RAW.exists() else None
    import lseg.data as ld
    open_session(ld)
    try:
        if raw is None:
            cons = ld.get_data(".SPX", ["TR.IndexConstituentRIC", "TR.IndexConstituentName"], {"SDate": AS_OF})
            raw = {"as_of": AS_OF, "constituents": cons.iloc[:, 1].dropna().tolist(),
                   "names": dict(zip(cons.iloc[:, 1], cons.iloc[:, 2])), "px": {}, "missing": [], "feasibility": {}}
            pd.to_pickle(raw, RAW)
        print(f"{len(raw['constituents'])} constituents on {raw['as_of']}", flush=True)
        pull_prices(ld, raw, str(pd.Timestamp(start) - pd.Timedelta(days=40)), end)
        if len(raw["px"]) + len(raw["missing"]) < len(raw["constituents"]):
            return 0

        table = {r: measures(monthly_returns(raw["px"][r], start, end)) for r in raw["px"]}
        for r in raw["missing"]:
            table[r] = {"months": 0, "vr": np.nan, "rho": np.nan, "vol": np.nan}
        order = ranked(table)
        chosen = []
        for ric in order:
            if len(chosen) == 7:
                break
            if ric not in raw["feasibility"]:
                raw["feasibility"][ric] = pull_feasibility(ld, ric)
                pd.to_pickle(raw, RAW)
            f = raw["feasibility"][ric]
            table[ric]["feasibility"] = feasible(f["default"], f["unadjusted"])
            if table[ric]["feasibility"]["feasible"]:
                chosen.append(ric)
    finally:
        ld.close_session()

    result = {"as_of": raw["as_of"], "window": [start, end], "universe": len(raw["constituents"]),
              "priced": len(raw["px"]), "eligible": sum(m["months"] >= MIN_MONTHS for m in table.values()),
              "ranked": order, "picks": chosen[:5], "follow_ups": chosen[5:7],
              "table": {r: {**m, "name": raw["names"].get(r)} for r, m in table.items()}}
    OUT.write_text(json.dumps(result, indent=1, default=str))
    print(f"priced {result['priced']}, eligible {result['eligible']}, ranked (both measures agree) {len(order)}")
    for i, r in enumerate(order[:25], 1):
        m = table[r]
        f = m.get("feasibility")
        tag = "" if f is None else ("  PICK" if r in chosen[:5] else "  follow-up" if r in chosen[5:] else
                                    "  skipped: " + f["why"]) + (f"  splits {f['splits']}" if f and f["splits"] else "")
        print(f"{i:3} {r:10} {str(raw['names'].get(r))[:28]:28} VR {m['vr']:.3f}  rho {m['rho']:+.3f}  vol {100 * m['vol']:4.0f}%  months {m['months']}{tag}")
    print(f"\npicks: {result['picks']}; follow-ups: {result['follow_ups']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
