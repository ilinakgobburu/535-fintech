"""
Run config/wheel_choices.yaml in its two stages:

    python3 scripts/run_choices.py --stage train   # each rule alone (and all three) on the 9 training windows
    python3 scripts/run_choices.py --stage test    # the version under `decisions`, once, on the 9 test windows

The test stage refuses to run until `decisions` in the config names a version,
committed after the training stage was seen. Results go to
cache/wheel_choices_{stage}.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trading_app.lib import wheel as W  # noqa: E402
from run_managed import stocks  # noqa: E402

CONFIG = REPO / "config" / "wheel_choices.yaml"
# config/wheel_choices.yaml's rules as run() reads them
RULES = {"skip_earnings": {"skip_earnings": True}, "iv_over_rv": {"iv_over_rv": 21},
         "unstick": {"unstick_after": 3}}
FILLS = ("mid", "cross")
NEED = 5          # of 9, at mid and at the stress fills
COUNTS = ("puts_written", "assigned", "calls_written", "called_away", "filtered", "share_days_pct")


def figures(s: dict) -> dict:
    return {"return_pct": s["return_pct"], "sharpe": s["sharpe"], "max_drawdown_pct": s["max_drawdown_pct"],
            "pnl": s["pnl"], **{k: s[k] for k in COUNTS}}


def run_versions(windows: list[dict], versions: dict) -> list[dict]:
    rows, markets = [], {}
    for sw in windows:
        m = markets.setdefault(sw["cache"], W.load_market(REPO / sw["cache"]))
        row = {k: sw[k] for k in ("stock", "window_name", "window", "pick")}
        for v, manage in versions.items():
            for fill in FILLS:
                res = W.run(m, start=sw["window"][0], end=sw["window"][1], fill=fill, manage=manage,
                            ticker=sw["stock"], **sw["pick"])
                row[f"{v}_{fill}"] = figures(W.summary(res, m["tbill"]))
                if v != "v1" and fill == "mid":
                    row[f"{v}_unstuck"] = sum(bool(c.get("unstuck")) for c in res["cycles"])
        rows.append(row)
        print(f"{sw['stock']:5} {sw['window_name']:5}  Sharpe at mid: "
              + "  ".join(f"{v} {row[v + '_mid']['sharpe']:5.2f}" for v in versions)
              + "   return: " + "  ".join(f"{v} {row[v + '_mid']['return_pct']:6.1f}%" for v in versions), flush=True)
    return rows


def tally(rows: list[dict], versions) -> dict:
    out = {}
    for v in versions:
        if v == "v1":
            continue
        wins = {fill: sum(r[f"{v}_{fill}"]["sharpe"] > r[f"v1_{fill}"]["sharpe"] for r in rows) for fill in FILLS}
        out[v] = {**wins, "helps": all(w >= NEED for w in wins.values())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("train", "test"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    cfg = yaml.safe_load(CONFIG.read_text())
    windows = [sw for sw in stocks() if sw["window_name"] == args.stage]

    if args.stage == "train":
        versions = {"v1": {}, **RULES, "all_three": {k: v for r in RULES.values() for k, v in r.items()}}
    else:
        decided = cfg.get("decisions") or {}
        if not decided.get("version"):
            raise SystemExit("config/wheel_choices.yaml has no decisions.version: decide on the training "
                             "stage and commit it first")
        versions = {"v1": {}, "decided": decided["version"]}

    rows = run_versions(windows, versions)
    t = tally(rows, versions)
    print(f"\nbeat the plain wheel's Sharpe, out of {len(rows)} (needs {NEED} at mid and at the stress fills):")
    for v, x in t.items():
        print(f"  {v:14} mid {x['mid']}  stress {x['cross']}  -> {'helps' if x['helps'] else 'does not help'}")
    out = {"stage": args.stage, "versions": versions, "rows": rows, "tally": t}
    if args.stage == "test":
        ma = next(r for r in rows if r["stock"] == "MA")
        out["ma_assigned"] = ma["decided_mid"]["assigned"]
        print(f"  MA 2026: assigned {ma['decided_mid']['assigned']} times "
              f"({'can' if ma['decided_mid']['assigned'] else 'cannot'} go on the page)")
    path = REPO / "cache" / f"wheel_choices_{args.stage}.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
