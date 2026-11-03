"""
Run config/wheel_rsi.yaml: the RSI entry filter against v1, the plain wheel at each
stock-window's committed pick, on config/wheel_choices.yaml's windows.

    python3 scripts/run_rsi.py --stage train   # rsi_put, rsi_call, rsi_both on the 9 training windows
    python3 scripts/run_rsi.py --stage test    # once, only the versions recorded as passing

The windows, the runs and the figures are run_choices.py's (run_versions, figures).
What this adds is the registered bar (7 of 9 at mid and at the stress fill), the
median changes against v1, and the entries each rule filtered. The test stage refuses
to run until a training_result_* key in the config names a passing version, and
refuses to run a second time. Results go to cache/wheel_rsi_{stage}.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_choices import FILLS, REPO, W, figures, run_versions, stocks  # noqa: E402

CONFIG = REPO / "config" / "wheel_rsi.yaml"
NEED = 7          # of 9, at mid and at the stress fill


def filtered(sw: dict, market: dict, manage: dict, fill: str) -> dict:
    """The entries of one run, by phase, and how many of each the RSI rules stopped."""
    res = W.run(market, start=sw["window"][0], end=sw["window"][1], fill=fill, manage=manage,
                ticker=sw["stock"], **sw["pick"])
    out = {}
    for phase, cycles in (("put", [c for c in res["cycles"] if c["shares_before"] == 0]),
                          ("call", [c for c in res["cycles"] if c["shares_before"] > 0])):
        out[f"{phase}_entries"] = len(cycles)
        out[f"{phase}_filtered"] = sum(str(c.get("status", "")).startswith("skipped: RSI") for c in cycles)
    out["entries"] = out["put_entries"] + out["call_entries"]
    out["filtered"] = out["put_filtered"] + out["call_filtered"]
    return out


def tally(rows: list[dict], versions: dict) -> dict:
    out = {}
    for v in versions:
        if v == "v1":
            continue
        t = {}
        for fill in FILLS:
            ds = [r[f"{v}_{fill}"]["sharpe"] - r[f"v1_{fill}"]["sharpe"] for r in rows]
            dr = [r[f"{v}_{fill}"]["return_pct"] - r[f"v1_{fill}"]["return_pct"] for r in rows]
            t[fill] = {"wins": int(sum(d > 0 for d in ds)), "ties": int(sum(d == 0 for d in ds)),
                       "median_sharpe_change": float(np.median(ds)), "median_return_change_pts": float(np.median(dr))}
            n = {k: sum(r[f"{v}_{fill}_entries"][k] for r in rows)
                 for k in ("entries", "filtered", "put_entries", "put_filtered", "call_entries", "call_filtered")}
            t[fill]["entries"] = {**n, "filtered_pct": 100.0 * n["filtered"] / n["entries"] if n["entries"] else np.nan}
        t["passes"] = all(t[fill]["wins"] >= NEED for fill in FILLS)
        out[v] = t
    return out


def passed_in_training(cfg: dict) -> list[str]:
    keys = sorted(k for k in cfg if str(k).startswith("training_result"))
    return list((cfg[keys[-1]] or {}).get("passed") or []) if keys else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("train", "test"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    cfg = yaml.safe_load(CONFIG.read_text())
    path = REPO / "cache" / f"wheel_rsi_{args.stage}.json"

    if args.stage == "train":
        versions = {"v1": {}, **cfg["versions"]}
    else:
        passed = passed_in_training(cfg)
        if not passed:
            raise SystemExit("config/wheel_rsi.yaml records no version that passed training: the test stage is not run")
        if path.exists():
            raise SystemExit(f"the test stage has been run once already; its result is {path}")
        versions = {"v1": {}, **{v: cfg["versions"][v] for v in passed}}

    windows = [sw for sw in stocks() if sw["window_name"] == args.stage]
    rows = run_versions(windows, versions)
    markets = {}
    for sw, row in zip(windows, rows):
        m = markets.setdefault(sw["cache"], W.load_market(REPO / sw["cache"]))
        for v, manage in versions.items():
            if v != "v1":
                for fill in FILLS:
                    row[f"{v}_{fill}_entries"] = filtered(sw, m, manage, fill)

    t = tally(rows, versions)
    print(f"\nagainst v1 on {len(rows)} {args.stage} windows (passes with {NEED} Sharpe wins at mid and at the stress fill):")
    for v, x in t.items():
        print(f"  {v:9} " + "   ".join(
            f"{fill}: wins {x[fill]['wins']} (ties {x[fill]['ties']}), median Sharpe {x[fill]['median_sharpe_change']:+.3f}, "
            f"median return {x[fill]['median_return_change_pts']:+.1f} pts, filtered {x[fill]['entries']['filtered']}"
            f"/{x[fill]['entries']['entries']} ({x[fill]['entries']['filtered_pct']:.1f}%)" for fill in FILLS)
            + f"  -> {'passes' if x['passes'] else 'does not pass'}")
    out = {"stage": args.stage, "config": "config/wheel_rsi.yaml", "need": NEED, "versions": versions,
           "rows": rows, "tally": t}
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
