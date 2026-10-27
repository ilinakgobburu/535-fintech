"""
Run config/wheel.yaml on one stock, in two stages, so the pick exists on disk
before its test window is run:

    python3 scripts/run_wheel.py --root NVDA --stage train   # 18 settings on 2019-2021 -> the pick
    python3 scripts/run_wheel.py --root NVDA --stage test    # the pick, the baseline and benchmarks on 2022-2023
    python3 scripts/run_wheel.py --config config/wheel_mr.yaml --root XYZ --stage check   # data check (d)
    python3 scripts/run_wheel.py --config config/wheel_mr.yaml --root XYZ --stage train

The train stage writes config/{STEM}_picks/{ROOT}.json, which is committed
before the test stage is run; the test stage refuses to run without it, and
never re-picks. Results go to cache/{STEM}_results_{ROOT}.json. STEM is the
config's file name: wheel for config/wheel.yaml.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
from trading_app.lib import wheel as W  # noqa: E402

FILLS = ("mid", "cross")


def use_config(path: Path) -> None:
    """Point the run at one registered config: its windows, grid and stocks, its picks and results."""
    global CFG, STEM, PICKS, UNDERLYINGS
    CFG = yaml.safe_load(Path(path).read_text())
    STEM = Path(path).stem
    PICKS = REPO / "config" / f"{STEM}_picks"
    # The registered stocks, and any added later under a dated key, each with its cache.
    UNDERLYINGS = {**(CFG.get("underlyings") or {}),
                   **{t: v for k, block in CFG.items() if k.startswith("added_")
                      for t, v in block.items() if isinstance(v, dict) and "cache" in v}}


use_config(REPO / "config" / "wheel.yaml")


def window(name: str) -> tuple[str, str]:
    w = CFG["windows"][name]
    return str(w["start"]), str(w["end"])


def book(market, name, fill, **setting):
    start, end = window(name)
    res = W.run(market, start=start, end=end, fill=fill, **setting)
    return res, W.summary(res, market["tbill"])


def cell_row(setting: dict, s: dict) -> dict:
    a = s["attribution"]
    return {**setting, "sharpe": s["sharpe"], "return_pct": s["return_pct"], "pnl": s["pnl"],
            "max_drawdown_pct": s["max_drawdown_pct"], "worst_month_pct": s["worst_month_pct"],
            "premium": a["premium"], "interest": a["interest"], "shares_pnl": a["shares"],
            "puts": s["puts_written"], "assigned": s["assigned"], "calls": s["calls_written"],
            "called_away": s["called_away"], "held_no_call": s["held_no_call"], "idle": s["idle"],
            "share_days_pct": s["share_days_pct"]}


def show(rows: list[dict], title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"{'put':>5} {'call':>5} {'floor':>5} {'Sharpe':>7} {'return':>8} {'max DD':>7} "
          f"{'premium':>9} {'shares':>9} {'interest':>8} {'asgn/puts':>9} {'away/calls':>10} {'held':>4} {'in shares':>9}")
    for r in rows:
        print(f"{r.get('put_delta', '-')!s:>5} {r.get('call_delta', '-')!s:>5} {str(r.get('floor', '-')):>5} "
              f"{r['sharpe']:7.2f} {r['return_pct']:7.1f}% {r['max_drawdown_pct']:6.1f}% "
              f"{r['premium']:9.0f} {r['shares_pnl']:9.0f} {r['interest']:8.0f} "
              f"{r['assigned']:>4}/{r['puts']:<4} {r['called_away']:>5}/{r['calls']:<4} {r['held_no_call']:>4} "
              f"{r['share_days_pct']:8.0f}%")


def check(market, root: str) -> None:
    """config/mr_screen.yaml (d): more than 10% of cycles without puts and calls spanning 0.15-0.35 delta fails."""
    cycles = []
    for name in ("train", "test"):
        cycles += W.coverage(market, *window(name))["cycles"]
    short = [c for c in cycles if not c["ok"]]
    pct = 100.0 * len(short) / len(cycles)
    for c in short:
        print(f"  short {c['entry'].date()} -> {c['expiry'].date()}  spot {c['spot']:.2f}  "
              f"puts |delta| {c['P_min']:.2f}-{c['P_max']:.2f}  calls {c['C_min']:.2f}-{c['C_max']:.2f}")
    verdict = "fails" if pct > 10.0 else "passes"
    print(f"{root}: {len(short)} of {len(cycles)} cycles short ({pct:.1f}%): {verdict} the data check")
    (REPO / "cache" / f"{STEM}_coverage_{root}.json").write_text(json.dumps({
        "root": root, "cycles": len(cycles), "short": len(short), "short_pct": pct, "verdict": verdict,
        "detail": [{k: (str(v.date()) if isinstance(v, pd.Timestamp) else v) for k, v in c.items()}
                   for c in cycles]}, indent=1, default=float))


def train(market, root: str) -> None:
    g = CFG["grid"]
    cells = W.grid_cells(g["put_delta"], g["call_delta"], g["floor"])
    out = {fill: [] for fill in FILLS}
    for fill in FILLS:
        for c in cells:
            out[fill].append(cell_row(c, book(market, "train", fill, **c)[1]))
    chosen = W.pick(out["mid"], CFG["selection"]["score"].split()[0].lower())
    setting = {k: chosen[k] for k in ("put_delta", "call_delta", "floor")}
    show(out["mid"], f"{root} train {window('train')} at mid")
    show(out["cross"], f"{root} train at bid")
    print(f"\npick: {setting}, training Sharpe {chosen['sharpe']:.4f}")
    PICKS.mkdir(exist_ok=True)
    (PICKS / f"{root}.json").write_text(json.dumps({
        "root": root, "registered": str(CFG["registered"]), "train": window("train"),
        "pick": setting, "train_sharpe_mid": chosen["sharpe"],
        "train_grid_mid": out["mid"], "train_grid_bid": out["cross"]}, indent=1, default=float) + "\n")
    print(f"wrote {PICKS / f'{root}.json'}; commit it before --stage test")


def test(market, root: str) -> None:
    path = PICKS / f"{root}.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing: run --stage train and commit the pick first")
    picked = json.loads(path.read_text())
    setting, base = picked["pick"], CFG["baseline"]["setting"]
    results, books = {}, {}
    for fill in FILLS:
        for name, s in (("pick", setting), ("baseline", base)):
            res, summ = book(market, "test", fill, **s)
            results[f"{name}_{fill}"] = cell_row(s, summ)
            books[f"{name}_{fill}"] = res
    for mode in ("buy_and_hold", "cash"):
        res, summ = book(market, "test", "mid", mode=mode)
        results[mode] = cell_row({}, summ)
        books[mode] = res

    # After the pick's result exists: every setting on the test window, for the
    # pick's rank and how well the training ranks predicted the test ranks.
    g = CFG["grid"]
    cells = W.grid_cells(g["put_delta"], g["call_delta"], g["floor"])
    test_grid = [cell_row(c, book(market, "test", "mid", **c)[1]) for c in cells]
    train_sharpe = [r["sharpe"] for r in picked["train_grid_mid"]]
    test_sharpe = [r["sharpe"] for r in test_grid]
    rank = 1 + sum(t > results["pick_mid"]["sharpe"] + 1e-12 for t in test_sharpe)
    rho = float(pd.Series(train_sharpe).rank().corr(pd.Series(test_sharpe).rank()))

    show([results[k] for k in ("pick_mid", "baseline_mid", "pick_cross", "baseline_cross")],
         f"{root} test {window('test')}: pick {setting} and baseline {base}, at mid then bid")
    show([results["buy_and_hold"], results["cash"]], "buy and hold, T-bills")
    show(test_grid, "all 18 settings on the test window, at mid (diagnostic only)")
    helped = all(results[f"pick_{f}"]["sharpe"] > results[f"baseline_{f}"]["sharpe"] for f in FILLS)
    print(f"\npick ranks {rank} of 18 on the test window; Spearman of train vs test Sharpe {rho:.2f}")
    print(f"selection helped on {root} (pick beats baseline at mid and at bid): {helped}")

    def jsonable(res):
        return {"cycles": [{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in c.items()}
                           for c in res["cycles"]],
                "nav": {str(d.date()): float(v) for d, v in zip(res["ledger"]["date"], res["ledger"]["nav"])}}

    (REPO / "cache" / f"{STEM}_results_{root}.json").write_text(json.dumps({
        "root": root, "pick": setting, "baseline": base, "test": window("test"), "results": results,
        "test_grid_mid": test_grid, "pick_rank_of_18": rank, "spearman_train_test": rho,
        "selection_helped": helped, "books": {k: jsonable(v) for k, v in books.items()}},
        indent=1, default=float))


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=REPO / "config" / "wheel.yaml")
    known, _ = pre.parse_known_args()
    use_config(known.config if known.config.is_absolute() else REPO / known.config)
    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--root", required=True, choices=sorted(UNDERLYINGS))
    ap.add_argument("--stage", required=True, choices=("check", "train", "test"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    market = W.load_market(REPO / UNDERLYINGS[args.root]["cache"])
    {"check": check, "train": train, "test": test}[args.stage](market, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
