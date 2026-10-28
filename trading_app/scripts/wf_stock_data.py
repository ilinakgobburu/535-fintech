"""
The data behind Part II's stock walk-forward section (config/wf_screen.yaml): each fold's screen,
its stock and setting, its test books re-run and checked against the recorded results, and the
folds joined end to end.

    python3 scripts/wf_stock_data.py      # writes cache/wf_stock_page.json
"""

from __future__ import annotations

import importlib.util
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
from trading_app.lib.pmcc_daily import performance  # noqa: E402

spec = importlib.util.spec_from_file_location("wheel_page_data", Path(__file__).with_name("wheel_page_data.py"))
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

SCREEN_CFG = REPO / "config" / "wf_screen.yaml"
SCREEN = REPO / "cache" / "wf_screen.json"
OUT = REPO / "cache" / "wf_stock_page.json"
BOOKS = ("wheel", "wheel_bid", "baseline", "hold", "cash", "putw")
STORED = {"wheel": "pick_mid", "wheel_bid": "pick_cross", "baseline": "baseline_mid", "hold": "buy_and_hold", "cash": "cash"}


def fold_books(market: dict, window: tuple[str, str], pick: dict, base: dict, ticker: str, bench) -> tuple[dict, dict]:
    """Each book's summary on one fold's test window, and its account value on every session."""
    start, end = window
    runs = {"wheel": W.run(market, start=start, end=end, ticker=ticker, **pick),
            "wheel_bid": W.run(market, start=start, end=end, fill="cross", ticker=ticker, **pick),
            "baseline": W.run(market, start=start, end=end, ticker=ticker, **base),
            "hold": W.run(market, start=start, end=end, mode="buy_and_hold", ticker=ticker),
            "cash": W.run(market, start=start, end=end, mode="cash", ticker=ticker)}
    books = {k: P.book_summary(v, market["tbill"]) for k, v in runs.items()}
    dates = runs["wheel"]["ledger"]["date"]
    navs = {k: v["ledger"]["nav"].to_numpy() for k, v in runs.items()}
    capital = runs["wheel"]["start_cash"]
    if bench is not None:
        book = P.index_book(bench, dates, capital)
        perf = performance(book, capital, market["tbill"])
        books["putw"] = {**{k: v for k, v in perf.items() if k != "monthly_returns"},
                         "vol_pct": perf["monthly_sd_pct"] * np.sqrt(12)}
        navs["putw"] = book["nav"].to_numpy()
    return books, {"date": list(dates), **navs, "capital": capital}


def main() -> int:
    warnings.filterwarnings("ignore")
    design, screen = yaml.safe_load(SCREEN_CFG.read_text()), json.loads(SCREEN.read_text())
    bench = P.read_cboe(P.CBOE_PUT) if P.CBOE_PUT.exists() else None
    folds, tbill = [], None
    for f, s in zip(design["folds"], screen["folds"]):
        cfg_path = REPO / f["run_config"]
        stem, cfg = cfg_path.stem, yaml.safe_load(cfg_path.read_text())
        ric = s["pick"]
        ticker = ric.split(".")[0]
        entry = next(v for k, block in cfg.items() if k.startswith("added_") for t, v in block.items() if t == ticker)
        picked = json.loads((REPO / "config" / f"{stem}_picks" / f"{ticker}.json").read_text())
        stored = json.loads((REPO / "cache" / f"{stem}_results_{ticker}.json").read_text())
        market = W.load_market(REPO / entry["cache"])
        tbill = market["tbill"] if tbill is None else tbill.combine_first(market["tbill"])
        window = (str(cfg["windows"]["test"]["start"]), str(cfg["windows"]["test"]["end"]))
        books, nav = fold_books(market, window, picked["pick"], cfg["baseline"]["setting"], ticker, bench)
        for book, key in STORED.items():
            for k in ("sharpe", "return_pct", "max_drawdown_pct"):
                a, b = books[book][k], stored["results"][key][k]
                if not (abs(a - b) < P.TOL or (np.isnan(a) and np.isnan(b))):
                    raise SystemExit(f"fold {f['fold']} {key} {k}: {a} now, {b} recorded: the run does not reproduce")
        tied = P.tied_at_top(picked["train_grid_mid"])
        top = [{"rank": i + 1, "ric": r, "name": s["table"][r]["name"], "vr": s["table"][r]["vr"], "rho": s["table"][r]["rho"], "vol": s["table"][r]["vol"],
                "status": ("pick" if r == s["pick"] else "follow-up" if r in s["follow_ups"] else
                           (s["table"][r].get("feasibility") or {}).get("why") or "")}
               for i, r in enumerate(s["ranked"][:10])]
        folds.append({"fold": f["fold"], "ticker": ticker, "ric": ric, "as_of": s["as_of"], "screen_window": s["window"],
                      "universe": s["universe"], "eligible": s["eligible"], "ranked": len(s["ranked"]),
                      "ma_rank": s["ranked"].index("MA.N") + 1 if "MA.N" in s["ranked"] else None, "top": top,
                      "train": picked["train"], "test": list(window), "pick": picked["pick"],
                      "train_sharpe": picked["train_sharpe_mid"], "tied": len(tied),
                      "tied_calls": len({r["call_delta"] for r in tied}),
                      **{f"train_{k}": tied[0][k] for k in ("puts", "assigned", "calls")},
                      "rank": stored["pick_rank_of_18"], "books": books, "nav": nav})
    capital = folds[0]["nav"]["capital"]
    joined, path = {}, {}
    for k in BOOKS:
        if all(k in fo["nav"] for fo in folds):
            parts = [pd.DataFrame({"date": fo["nav"]["date"], "nav": fo["nav"][k]}) for fo in folds]
            perf = P.join_folds(parts, capital, tbill)
            joined[k] = {x: perf[x] for x in ("return_pct", "sharpe", "max_drawdown_pct", "months")}
            scale, vals = 1.0, []
            for part in parts:
                vals += list(scale * part["nav"].to_numpy(dtype=float))
                scale *= float(part["nav"].iloc[-1]) / capital
            path[k] = vals
    path["date"] = [d for fo in folds for d in fo["nav"]["date"]]
    for fo in folds:
        fo.pop("nav")
    out = {"folds": folds, "joined": joined, "path": path,
           "commits": {"design": P.commit(SCREEN_CFG), "picks": P.commit_adding(REPO / design["folds"][0]["run_config"], "added_2026_10_28:"),
                       "settings": P.commit(REPO / "config" / f"{Path(design['folds'][0]['run_config']).stem}_picks" / f"{folds[0]['ticker']}.json")}}
    OUT.write_text(json.dumps(P.jsonable(out), separators=(",", ":")))
    for fo in folds:
        b = fo["books"]
        print(f"fold {fo['fold']}: {fo['ticker']} {fo['pick']} on {fo['test']}: wheel {b['wheel']['return_pct']:+.1f}% "
              f"Sharpe {b['wheel']['sharpe']:.2f}; hold {b['hold']['return_pct']:+.1f}% {b['hold']['sharpe']:.2f}; "
              f"PutWrite {b['putw']['sharpe']:.2f}; MA ranked {fo['ma_rank']}")
    for k, v in joined.items():
        print(f"joined {k:10} {v['return_pct']:+6.1f}%  Sharpe {v['sharpe']:5.2f}  max DD {v['max_drawdown_pct']:6.1f}%  months {v['months']}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
