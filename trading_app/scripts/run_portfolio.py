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
from wheel_page_data import CBOE_PUT, index_book, read_cboe  # noqa: E402

OUT = REPO / "cache" / "wheel_portfolio_results.json"
STOCKS = ["MA", "APA", "V", "FISV", "ADI"]
FILLS = ("mid", "cross")
COUNTS = ("puts_written", "assigned", "calls_written", "called_away", "held_no_call")


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


def sharpe_if_together(ledgers: list[pd.DataFrame], capital: float, rf: pd.Series) -> float:
    """
    The portfolio's Sharpe ratio had the books moved in lockstep: its own mean excess return
    over the average of the books' monthly volatilities, annualised.
    """
    port = monthly_returns(combine(ledgers), capital * len(ledgers))
    excess = port - rf.reindex(port.index)
    avg_sd = float(np.mean([monthly_returns(l, capital).std(ddof=1) for l in ledgers]))
    return float(excess.mean() / avg_sd * np.sqrt(12))


def figures(p: dict) -> dict:
    return {k: p[k] for k in ("return_pct", "sharpe", "max_drawdown_pct", "worst_month", "worst_month_pct",
                              "monthly_sd_pct")} | {"vol_pct": p["monthly_sd_pct"] * np.sqrt(12)}


def trades(runs: dict, tbill: pd.Series) -> dict:
    """Each wheel's premium written, T-bill interest and option counts, and the five added together."""
    each = {}
    for s, r in runs.items():
        sm = W.summary(r, tbill)
        each[s] = {"premium": sm["attribution"]["premium"], "interest": sm["attribution"]["interest"],
                   **{k: sm[k] for k in COUNTS}}
    return {"each": each, "total": {k: sum(e[k] for e in each.values()) for k in next(iter(each.values()))}}


def main() -> int:
    warnings.filterwarnings("ignore")
    from run_managed import stocks
    windows = {}
    for sw in stocks():
        if sw["stock"] in STOCKS:
            windows.setdefault(sw["window_name"], []).append(sw)
    out = {"stocks": STOCKS, "windows": {},
           "picks": {sw["stock"]: sw["pick"] for sws in windows.values() for sw in sws}}
    bench = read_cboe(CBOE_PUT) if CBOE_PUT.exists() else None
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
            rf = (tbill.groupby(tbill.index.to_period("M")).mean() / 12.0)
            w[book]["sharpe_if_together"] = sharpe_if_together(ledgers, 100_000.0, rf)
            if book == "wheel_mid":
                rets = pd.DataFrame({s: monthly_returns(r["ledger"], 100_000.0) for s, r in runs.items()})
                corr = rets.corr()
                w["correlations"] = corr.round(3).to_dict()
                w["median_correlation"] = float(np.median(corr.to_numpy()[np.triu_indices(len(corr), 1)]))
                w[book]["trades"] = trades(runs, tbill)
        w["cash"] = figures(performance(cash["ledger"], 100_000.0, tbill))
        w["cash"]["nav"] = {"date": [str(d.date()) for d in cash["ledger"]["date"]],
                            "value": (cash["ledger"]["nav"] * len(sws)).round(2).tolist()}
        if bench is not None:       # the Cboe PutWrite index, added after the test as on Part I (config/wheel_mr.yaml)
            capital = 100_000.0 * len(sws)
            idx = index_book(bench, pd.to_datetime(w["wheel_mid"]["nav"]["date"]), capital)
            w["putw"] = figures(performance(idx, capital, tbill))
            w["putw"]["nav"] = {"date": w["wheel_mid"]["nav"]["date"], "value": idx["nav"].round(2).tolist()}
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
