"""
config/random_wheel_2026.yaml, a curiosity run: the plain wheel at config/wheel.yaml's default setting in 2026, on
random S&P 100 stocks (cache/random_wheel_2026_draw.json, in draw order, skipping any that fails data check (d))
and on the mean-reversion screen's five, so that only the stock differs.

    python3 scripts/random_wheel_2026.py     # -> cache/random_wheel_2026.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trading_app.lib import wheel as W  # noqa: E402
from trading_app.lib.pmcc_daily import performance  # noqa: E402
from run_portfolio import combine, figures, premium_vs_realised  # noqa: E402

START, END = "2026-01-02", "2026-10-16"
DEFAULT = {"put_delta": 0.25, "call_delta": 0.25, "floor": True}
WANT = 8
SCREEN = ["MA", "APA", "V", "FISV", "ADI"]
DRAW = REPO / "cache" / "random_wheel_2026_draw.json"
OUT = REPO / "cache" / "random_wheel_2026.json"


def book(market: dict, ticker: str) -> dict:
    """The default wheel, buy and hold, and T-bills on one stock in 2026, with the options sold against realized vol."""
    wheel = W.run(market, start=START, end=END, ticker=ticker, **DEFAULT)
    hold = W.run(market, start=START, end=END, mode="buy_and_hold", ticker=ticker)
    perf = lambda r: figures(performance(r["ledger"], 100_000.0, market["tbill"]))
    return {"wheel": perf(wheel), "hold": perf(hold), "vol": {k: v for k, v in premium_vs_realised(
                wheel, market["stock"]["close"]).items() if k != "options"},
            "ledgers": {"wheel": wheel["ledger"], "hold": hold["ledger"]}, "tbill": market["tbill"]}


def short_pct(market: dict) -> float:
    cycles = W.coverage(market, START, END)["cycles"]
    return 100.0 * sum(not c["ok"] for c in cycles) / len(cycles)


def group(books: dict) -> dict:
    """Equal-weight averages, and the wheels and holdings added into one portfolio, $100,000 each."""
    out = {"average": {b: {k: float(np.mean([x[b][k] for x in books.values()]))
                           for k in ("return_pct", "sharpe", "max_drawdown_pct", "vol_pct")} for b in ("wheel", "hold")}}
    tbill = next(iter(books.values()))["tbill"]
    for b in ("wheel", "hold"):
        port = combine([x["ledgers"][b] for x in books.values()])
        out[f"portfolio_{b}"] = figures(performance(port, 100_000.0 * len(books), tbill))
    out["wheel_beats_hold_sharpe"] = sum(x["wheel"]["sharpe"] > x["hold"]["sharpe"] for x in books.values())
    out["vol_above"] = sum(x["vol"]["above"] for x in books.values())
    out["vol_sold"] = sum(x["vol"]["sold"] for x in books.values())
    return out


def main() -> int:
    warnings.filterwarnings("ignore")
    draw = json.loads(DRAW.read_text())
    random_books, skipped = {}, []
    for ric in draw["order"]:
        if len(random_books) == WANT:
            break
        t = ric.split(".")[0]
        f = draw.get("feasibility", {}).get(ric)
        if f is not None and not f["feasible"]:
            skipped.append({"ric": ric, "why": f["why"]})
            continue
        cache = REPO / "cache" / f"wheel_raw_{t}_R26"
        if not cache.exists():
            skipped.append({"ric": ric, "why": "no option pull"})
            continue
        m = W.load_market(cache)
        pct = short_pct(m)
        if pct > 10.0:
            skipped.append({"ric": ric, "why": f"data check (d): {pct:.1f}% of cycles short"})
            continue
        random_books[t] = {**book(m, t), "short_pct": pct}
    screen_books = {t: book(W.load_market(REPO / "cache" / f"wheel_raw_{t}_2024"), t) for t in SCREEN}
    for t, b in screen_books.items():            # the class run recorded these: check they reproduce
        rec = json.loads((REPO / "cache" / f"wheel_mr_results_{t}.json").read_text())["results"]["baseline_mid"]
        assert abs(rec["sharpe"] - b["wheel"]["sharpe"]) < 1e-9, f"{t} baseline does not reproduce"
    strip = lambda bs: {t: {k: v for k, v in b.items() if k not in ("ledgers", "tbill")} for t, b in bs.items()}
    out = {"window": [START, END], "setting": DEFAULT, "skipped": skipped,
           "random": strip(random_books), "screen": strip(screen_books),
           "random_group": group(random_books) if random_books else None, "screen_group": group(screen_books)}
    OUT.write_text(json.dumps(out, indent=1, default=float))
    row = lambda t, b: (f"  {t:5} wheel {b['wheel']['return_pct']:+6.1f}% Sharpe {b['wheel']['sharpe']:5.2f} DD "
                        f"{b['wheel']['max_drawdown_pct']:6.1f}% | hold {b['hold']['return_pct']:+6.1f}% "
                        f"{b['hold']['sharpe']:5.2f} | sold above realized {b['vol']['above']}/{b['vol']['sold']}")
    print("random S&P 100 stocks (default setting, 2026):")
    for t, b in random_books.items():
        print(row(t, b))
    print("skipped:", skipped)
    print("the screen's five (same setting, 2026):")
    for t, b in screen_books.items():
        print(row(t, b))
    for name, g in (("random", out["random_group"]), ("screen", out["screen_group"])):
        if g:
            print(f"{name}: avg wheel Sharpe {g['average']['wheel']['sharpe']:.2f} (hold {g['average']['hold']['sharpe']:.2f}); "
                  f"portfolio wheel {g['portfolio_wheel']['return_pct']:+.1f}% Sharpe {g['portfolio_wheel']['sharpe']:.2f} "
                  f"DD {g['portfolio_wheel']['max_drawdown_pct']:.1f}% vs hold {g['portfolio_hold']['return_pct']:+.1f}% "
                  f"{g['portfolio_hold']['sharpe']:.2f}; wheel beat hold on Sharpe {g['wheel_beats_hold_sharpe']}; "
                  f"sold above realized {g['vol_above']}/{g['vol_sold']}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
