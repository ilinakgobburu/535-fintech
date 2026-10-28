"""
Run config/wheel_portfolio.yaml: the plain wheel on the five class-run stocks
side by side, $100,000 each, added into one $500,000 portfolio.

    python3 scripts/run_portfolio.py      # -> cache/wheel_portfolio_results.json
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

OUT = REPO / "cache" / "wheel_portfolio_results.json"
STOCKS = ["MA", "APA", "V", "FISV", "ADI"]
FILLS = ("mid", "cross")


def combine(ledgers: list[pd.DataFrame]) -> pd.DataFrame:
    """One ledger whose account value is the books' added together; a missing session carries the last value."""
    navs = pd.concat([l.set_index("date")["nav"] for l in ledgers], axis=1).sort_index().ffill()
    return pd.DataFrame({"date": navs.index, "nav": navs.sum(axis=1).to_numpy()})


def monthly_returns(ledger: pd.DataFrame, capital: float) -> pd.Series:
    """Month-end account value over the prior month-end, the first from `capital` (as pmcc_daily.performance)."""
    nav = ledger.set_index("date")["nav"]
    me = nav.groupby(nav.index.to_period("M")).last()
    r = pd.concat([pd.Series([capital]), me], ignore_index=True).pct_change().dropna()
    r.index = me.index
    return r


def figures(p: dict) -> dict:
    return {k: p[k] for k in ("return_pct", "sharpe", "max_drawdown_pct", "worst_month", "worst_month_pct",
                              "monthly_sd_pct")} | {"vol_pct": p["monthly_sd_pct"] * np.sqrt(12)}


def main() -> int:
    warnings.filterwarnings("ignore")
    from run_managed import stocks
    windows = {}
    for sw in stocks():
        if sw["stock"] in STOCKS:
            windows.setdefault(sw["window_name"], []).append(sw)
    out = {"stocks": STOCKS, "windows": {}}
    markets = {}
    for name, sws in windows.items():
        lo, hi = sws[0]["window"]
        res = {"wheel_mid": {}, "wheel_cross": {}, "hold": {}}
        for sw in sws:
            m = markets.setdefault(sw["cache"], W.load_market(REPO / sw["cache"]))
            res["wheel_mid"][sw["stock"]] = W.run(m, start=lo, end=hi, ticker=sw["stock"], **sw["pick"])
            res["wheel_cross"][sw["stock"]] = W.run(m, start=lo, end=hi, fill="cross", ticker=sw["stock"], **sw["pick"])
            res["hold"][sw["stock"]] = W.run(m, start=lo, end=hi, mode="buy_and_hold", ticker=sw["stock"])
        tbill = markets[sws[0]["cache"]]["tbill"]
        cash = W.run(markets[sws[0]["cache"]], start=lo, end=hi, mode="cash")
        w = {"window": [lo, hi]}
        for book, runs in res.items():
            ledgers = [r["ledger"] for r in runs.values()]
            port = combine(ledgers)
            w[book] = {"portfolio": figures(performance(port, 100_000.0 * len(ledgers), tbill)),
                       "each": {s: figures(performance(r["ledger"], 100_000.0, tbill)) for s, r in runs.items()},
                       "nav": {"date": [str(d.date()) for d in port["date"]], "value": port["nav"].round(2).tolist()}}
            each = w[book]["each"].values()
            w[book]["average_of_five"] = {k: float(np.mean([e[k] for e in each]))
                                          for k in ("return_pct", "sharpe", "max_drawdown_pct", "vol_pct")}
            if book == "wheel_mid":
                rets = pd.DataFrame({s: monthly_returns(r["ledger"], 100_000.0) for s, r in runs.items()})
                corr = rets.corr()
                w["correlations"] = corr.round(3).to_dict()
                w["median_correlation"] = float(np.median(corr.to_numpy()[np.triu_indices(len(corr), 1)]))
        w["cash"] = figures(performance(cash["ledger"], 100_000.0, tbill))
        out["windows"][name] = w
        p, a, h = w["wheel_mid"]["portfolio"], w["wheel_mid"]["average_of_five"], w["hold"]["portfolio"]
        print(f"\n{name} {lo} to {hi}")
        print(f"  five wheels together: return {p['return_pct']:+.1f}%  Sharpe {p['sharpe']:.2f}  max drawdown "
              f"{p['max_drawdown_pct']:.1f}%  vol {p['vol_pct']:.1f}%   (at the stress fills: return "
              f"{w['wheel_cross']['portfolio']['return_pct']:+.1f}%, Sharpe {w['wheel_cross']['portfolio']['sharpe']:.2f})")
        print(f"  average single wheel: return {a['return_pct']:+.1f}%  Sharpe {a['sharpe']:.2f}  max drawdown "
              f"{a['max_drawdown_pct']:.1f}%  vol {a['vol_pct']:.1f}%")
        print(f"  five stocks held:     return {h['return_pct']:+.1f}%  Sharpe {h['sharpe']:.2f}  max drawdown "
              f"{h['max_drawdown_pct']:.1f}%  vol {h['vol_pct']:.1f}%")
        print(f"  T-bills: {w['cash']['return_pct']:+.1f}%;  median correlation of the wheels' monthly returns "
              f"{w['median_correlation']:.2f}")
        for s, e in w["wheel_mid"]["each"].items():
            print(f"    {s:5} wheel {e['return_pct']:+6.1f}%  Sharpe {e['sharpe']:5.2f}  max drawdown {e['max_drawdown_pct']:6.1f}%")
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
