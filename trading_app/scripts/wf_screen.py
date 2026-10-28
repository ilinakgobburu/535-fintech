"""
The stock walk-forward's screens (config/wf_screen.yaml): config/mr_screen.yaml's ranking, re-run at
the end of 2024 and of 2025 on the S&P 500 as it stood then and the ten years before, each picking
the stock its fold trades the next year. The measures, ranking and feasibility tests are
scripts/mr_screen.py's own functions; only the dates and the price pull differ.

    python3 scripts/wf_screen.py          # pull (resuming from the cache), rank, check feasibility, pick

Prices already pulled for config/mr_screen.yaml (to 2023-12-29) are extended from 2023-10-01 rather
than pulled again; a series whose overlap disagrees by more than 0.5% is pulled again in full.
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
import mr_screen as M  # noqa: E402

CFG_PATH = REPO / "config" / "wf_screen.yaml"
RAW = REPO / "cache" / "wf_screen_raw.pkl"
OUT = REPO / "cache" / "wf_screen.json"
SEED = M.RAW                                   # config/mr_screen.yaml's pull, 2013-11 to 2023-12
EXTEND_FROM, FULL_FROM, PULL_TO = "2023-10-01", "2014-11-20", "2025-12-31"
OVERLAP_TOL = 0.005


def extend(old: pd.Series, new: pd.Series, start: str = EXTEND_FROM, tol: float = OVERLAP_TOL) -> pd.Series | None:
    """
    `old` up to the day before `start`, then `new` from it; None if the two disagree by more than
    `tol` on any day both cover from `start`, or share no day there, so the series must be pulled again.
    """
    old, new = old.sort_index(), new.sort_index()
    cut = pd.Timestamp(start)
    both = old[old.index >= cut].index.intersection(new.index)
    if not len(both) or float((new[both] / old[both] - 1.0).abs().max()) > tol:
        return None
    return pd.concat([old[old.index < cut], new[new.index >= cut]])


def pull(ld, rics: list[str], start: str, end: str) -> tuple[dict, bool]:
    """Daily closes for `rics`, in batches, one by one when a batch fails; True if the request cap stopped it."""
    out = {}
    for i in range(0, len(rics), M.BATCH):
        batch = rics[i:i + M.BATCH]
        try:
            out.update(M.frame_to_series(ld.get_history(universe=batch, fields=["TRDPRC_1"], interval="daily",
                                                        start=start, end=end), batch))
        except Exception:
            for ric in batch:
                try:
                    out.update(M.frame_to_series(ld.get_history(universe=ric, fields=["TRDPRC_1"], interval="daily",
                                                                start=start, end=end), [ric]))
                except Exception as exc:
                    if "429" in str(exc) or "Too many" in str(exc):
                        return out, True
                time.sleep(0.2)
        print(f"  {start[:4]}- prices {min(i + M.BATCH, len(rics))}/{len(rics)}", flush=True)
    return out, False


def pull_feasibility(ld, ric: str, lo: str, hi: str) -> dict:
    default = ld.get_history(universe=ric, fields=["TRDPRC_1"], interval="daily", start=lo, end=hi)
    unadjusted = ld.get_history(universe=ric, fields=["TRDPRC_1"], interval="daily", start=lo, end=hi,
                                adjustments=["exchangeCorrection"])
    return {"default": M.frame_to_series(default, [ric]).get(ric, pd.Series(dtype=float)),
            "unadjusted": M.frame_to_series(unadjusted, [ric]).get(ric, pd.Series(dtype=float))}


def within(s: pd.Series, lo: str, hi: str) -> pd.Series:
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index)
    return s[(s.index >= pd.Timestamp(lo)) & (s.index <= pd.Timestamp(hi))]


def screen(raw: dict, fold: dict, ld) -> dict:
    """One fold: rank its universe on its window, then test feasibility down the ranking until three pass."""
    lo_w, hi_w = (str(x) for x in fold["screen_window"])
    as_of = str(fold["universe_as_of"])
    span = (str(fold["train"]["start"]), str(fold["test"]["end"]))
    members = raw["constituents"][as_of]
    table = {r: M.measures(M.monthly_returns(raw["px"][r], lo_w, hi_w)) if r in raw["px"]
             else {"months": 0, "vr": np.nan, "rho": np.nan, "vol": np.nan} for r in members}
    order = M.ranked(table)
    chosen = []
    for ric in order:
        if len(chosen) == 3:
            break
        if ric not in raw["feasibility"]:
            raw["feasibility"][ric] = pull_feasibility(ld, ric, "2024-01-02", "2026-10-23")
            pd.to_pickle(raw, RAW)
        f = raw["feasibility"][ric]
        table[ric]["feasibility"] = M.feasible(within(f["default"], *span), within(f["unadjusted"], *span), span[1])
        if table[ric]["feasibility"]["feasible"]:
            chosen.append(ric)
    return {"fold": fold["fold"], "as_of": as_of, "window": [lo_w, hi_w], "span": list(span),
            "universe": len(members), "priced": sum(r in raw["px"] for r in members),
            "eligible": sum(m["months"] >= M.MIN_MONTHS for m in table.values()),
            "ranked": order, "pick": chosen[0] if chosen else None, "follow_ups": chosen[1:3],
            "table": {r: {**table[r], "name": raw["names"].get(r)} for r in order[:25]}}


def main() -> int:
    warnings.filterwarnings("ignore")
    cfg = yaml.safe_load(CFG_PATH.read_text())
    if RAW.exists():
        raw = pd.read_pickle(RAW)
    else:
        seed = pd.read_pickle(SEED)
        raw = {"constituents": {}, "names": dict(seed["names"]), "seed": dict(seed["px"]), "px": {},
               "missing": [], "repulled": [], "feasibility": dict(seed["feasibility"])}
    import lseg.data as ld
    open_session(ld)
    try:
        for fold in cfg["folds"]:
            as_of = str(fold["universe_as_of"])
            if as_of not in raw["constituents"]:
                cons = ld.get_data(".SPX", ["TR.IndexConstituentRIC", "TR.IndexConstituentName"], {"SDate": as_of})
                raw["constituents"][as_of] = cons.iloc[:, 1].dropna().tolist()
                raw["names"].update({k: v for k, v in zip(cons.iloc[:, 1], cons.iloc[:, 2]) if k not in raw["names"]})
                pd.to_pickle(raw, RAW)
            print(f"{len(raw['constituents'][as_of])} constituents on {as_of}", flush=True)
        need = sorted(set().union(*raw["constituents"].values()) - set(raw["px"]) - set(raw["missing"]))
        seeded = [r for r in need if r in raw["seed"]]
        got, capped = pull(ld, seeded, EXTEND_FROM, PULL_TO)
        again = []
        for r in seeded:
            if r in got:
                joined = extend(raw["seed"][r], got[r])
                if joined is None:
                    again.append(r)
                else:
                    raw["px"][r] = joined
        raw["repulled"] = sorted(set(raw["repulled"]) | set(again))
        pd.to_pickle(raw, RAW)
        if capped:
            print("stopped: request cap reached; run again to resume")
            return 0
        fresh = [r for r in need if r not in raw["seed"]] + again
        got, capped = pull(ld, fresh, FULL_FROM, PULL_TO)
        raw["px"].update(got)
        if not capped:
            raw["missing"] = sorted(set(raw["missing"]) | {r for r in need if r not in raw["px"]})
        pd.to_pickle(raw, RAW)
        if capped:
            print("stopped: request cap reached; run again to resume")
            return 0
        print(f"priced {len(raw['px'])}; extended {len(seeded) - len(again)}, pulled in full {len(fresh)} "
              f"({len(again)} because the overlap disagreed); missing {len(raw['missing'])}", flush=True)
        results = [screen(raw, fold, ld) for fold in cfg["folds"]]
    finally:
        ld.close_session()

    OUT.write_text(json.dumps({"folds": results, "repulled": raw["repulled"], "missing": raw["missing"]},
                              indent=1, default=str))
    for res in results:
        print(f"\nfold {res['fold']}: universe of {res['as_of']} ({res['universe']}), screened on "
              f"{res['window'][0]} to {res['window'][1]}; priced {res['priced']}, eligible {res['eligible']}, "
              f"ranked {len(res['ranked'])}")
        for i, r in enumerate(res["ranked"][:12], 1):
            m = res["table"][r]
            f = m.get("feasibility")
            tag = "" if f is None else ("  PICK" if r == res["pick"] else "  follow-up" if r in res["follow_ups"]
                                        else "  skipped: " + f["why"])
            print(f"{i:3} {r:10} {str(m['name'])[:28]:28} VR {m['vr']:.3f}  rho {m['rho']:+.3f}  "
                  f"vol {100 * m['vol']:4.0f}%  months {m['months']}{tag}")
        print(f"pick: {res['pick']}; follow-ups: {res['follow_ups']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
