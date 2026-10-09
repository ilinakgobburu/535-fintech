"""
Run the registered PMCC backtest and its comparisons from the raw cache, and
write every figure the page shows to one JSON file.

    python3 scripts/run_pmcc.py                 # DAL, 2019-01-02 to 2023-12-29

Books, all through lib.pmcc_daily.run:

    pmcc        version 2 rules, mid fills, sell through earnings        (primary)
    pmcc_cross  the primary book filled at the bid and ask               (stress test)
    pmcc_skip   the primary book, skipping cycles with an earnings date
    pmcc_v1     the version 1 rules as first run: bid/ask fills, time-only LEAP
                roll, short strike at or above the LEAP strike
    cc          100 shares + the same delta rule for the short call, mid fills
    cc_cross    the covered call at the bid and ask
    hold        100 shares, dividends as cash
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trading_app.lib import pmcc_daily as E  # noqa: E402

REPO = ROOT.parent
RULES = yaml.safe_load((REPO / "config" / "thresholds.yaml").read_text())

V2 = dict(roll_delta=E.ROLL_DELTA, breakeven_floor=True)
BOOKS = {
    "pmcc": dict(long_leg="leap", fill="mid", **V2),
    "pmcc_cross": dict(long_leg="leap", fill="cross", **V2),
    "pmcc_skip": dict(long_leg="leap", fill="mid", skip_earnings=True, **V2),
    "pmcc_v1": dict(long_leg="leap", fill="cross"),
    "cc": dict(long_leg="stock", fill="mid"),
    "cc_cross": dict(long_leg="stock", fill="cross"),
    "hold": dict(long_leg="stock", write_calls=False),
}


def clean(x):
    """JSON-safe: timestamps to ISO dates, NaN to null, numpy scalars to Python."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (pd.Timestamp, pd.Period)):
        return str(x)[:10]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), 6)
    return x


def short_leg_pnl(cycle: dict, assignments: dict) -> float | None:
    """Premium kept, less what covering an assignment cost over the strike."""
    if "premium" not in cycle:
        return None
    a = assignments.get(cycle["expiry"])
    return cycle["premium"] - (a.get("cover_cost", 0.0) if a else 0.0)


def payoff_example(market: dict, res: dict) -> dict | None:
    """
    Profit at the first short call's expiry, across stock prices, for the PMCC
    and for a covered call writing the same call on the same day.

    The covered call's line is arithmetic. The PMCC's needs a value for the
    LEAP on that date at each stock price; it is priced with Black-Scholes at
    the implied volatility the LEAP had on the entry date, with the rate and
    dividend yield of that date. That is a model value, and the chart says so.
    """
    from trading_app.lib.pmcc import call_greeks
    from trading_app.lib.vol import bs_price

    cyc = next((c for c in res["cycles"] if "premium" in c), None)
    leap = next((e for e in res["blotter"] if e["leg"] == "long" and e["side"] == "BUY"), None)
    if cyc is None or leap is None or leap["date"] != cyc["entry"]:
        return None
    day, spot = cyc["entry"], cyc["spot"]
    r, q = E.carry(market, day)
    iv = call_greeks(leap["price"], spot, leap["strike"], (leap["expiry"] - day).days / 365.0, r, q)["iv"]
    T = (leap["expiry"] - cyc["expiry"]).days / 365.0
    grid = np.linspace(0.70 * spot, 1.30 * spot, 61)
    short = np.maximum(grid - cyc["strike"], 0.0)
    value = np.array([bs_price(s * np.exp((r - q) * T), leap["strike"], T, iv, np.exp(-r * T), "C")
                      for s in grid])
    return {
        "entry": day, "expiry": cyc["expiry"], "spot": spot, "short_strike": cyc["strike"],
        "premium": cyc["premium"], "leap_strike": leap["strike"], "leap_expiry": leap["expiry"],
        "leap_cost": E.CONTRACT * leap["price"], "leap_iv": iv,
        "stock": [round(float(x), 2) for x in grid],
        "pmcc": [round(float(x), 2) for x in E.CONTRACT * (value - leap["price"] - short) + cyc["premium"]],
        "cc": [round(float(x), 2) for x in E.CONTRACT * (grid - spot - short) + cyc["premium"]],
    }


def one_book(name: str, market: dict, start, end, ticker: str) -> dict:
    res = E.run(market, start=start, end=end, ticker=ticker, **BOOKS[name])
    capital = E.opening_capital(res)
    led = E.build_ledger(res, market, capital)
    perf = E.performance(led, capital, market["tbill"])
    floor = E.min_start_cash(led, capital)
    by_expiry = {a["expiry"]: a for a in res["assignments"]}
    cycles = []
    for c in res["cycles"]:
        c = dict(c)
        c["short_pnl"] = short_leg_pnl(c, by_expiry)
        cycles.append(c)
    status = pd.Series([c["status"] for c in cycles]).value_counts().to_dict()
    written = [c for c in cycles if "premium" in c]
    # The same book funded with the cash Reg T actually required. Every NAV
    # shifts by the extra cash, so P&L is identical and only the base changes.
    funded = led.assign(nav=led["nav"] + (floor["min_cash"] - capital))
    perf_funded = E.performance(funded, floor["min_cash"], market["tbill"])
    nav = funded.set_index("date")["nav"]
    path = pd.concat([pd.Series([floor["min_cash"]]), nav], ignore_index=True)
    return {
        "name": name, "rules": BOOKS[name], "capital": capital, "performance": perf,
        "payoff": payoff_example(market, res) if name == "pmcc" else None,
        "min_start_cash": floor, "performance_funded": perf_funded,
        "days_cash_negative": int((led["cash"] < -1e-9).sum()),
        "status_counts": status,
        "premium_collected": float(sum(c["premium"] for c in written)),
        "cover_cost": float(sum(a.get("cover_cost", 0.0) for a in res["assignments"])),
        "pnl_by_leg": E.pnl_by_leg(res, led),
        "cycles": cycles, "assignments": res["assignments"],
        "dividend_risk": res["dividend_risk"], "blotter": res["blotter"],
        "ledger": {
            "date": [str(d)[:10] for d in led["date"]],
            **{k: [None if not np.isfinite(v) else round(float(v), 4) for v in led[k]]
               for k in ("nav", "cash", "shares", "stock_close", "long_mv", "short_mv",
                         "available_funds", "excess_liquidity", "initial_margin",
                         "delta", "vega", "theta", "long_iv", "short_iv")},
            "nav_funded": [round(float(v), 4) for v in funded["nav"]],
            "drawdown_pct": [round(100.0 * float(v), 4)
                             for v in (path / path.cummax() - 1.0).iloc[1:]],
        },
    }


def earnings_split(book: dict, earnings: pd.DatetimeIndex) -> list[dict]:
    """
    The short leg's result by whether the cycle contained an earnings date.
    Not a registered test; it is context for the earnings comparison book.
    """
    def group(c):
        return "an earnings date" if c.get("earnings_in_cycle") else "no earnings date"

    rows = []
    df = pd.DataFrame([{"group": group(c), "pnl": c["short_pnl"], "premium": c["premium"],
                        "assigned": c["status"] == "assigned"}
                       for c in book["cycles"] if c.get("short_pnl") is not None])
    for g, d in df.groupby("group"):
        rows.append({"group": g, "cycles": int(len(d)), "mean_premium": float(d["premium"].mean()),
                     "mean_pnl": float(d["pnl"].mean()), "median_pnl": float(d["pnl"].median()),
                     "worst_pnl": float(d["pnl"].min()), "assigned": int(d["assigned"].sum())})
    return rows


def data_summary(cache: Path, market: dict, start, end) -> dict:
    lines = (cache / "requests.jsonl").read_text().splitlines()
    if (cache / "requests_adjusted.jsonl").exists():
        lines += (cache / "requests_adjusted.jsonl").read_text().splitlines()
    log = pd.DataFrame([json.loads(l) for l in lines])
    log = log.drop_duplicates("ric", keep="last")
    q = market["quotes"]
    q = q[(q["date"] >= pd.Timestamp(start)) & (q["date"] <= pd.Timestamp(end))]
    two = q[q["mid"].notna()]
    return {
        "requests": int(len(log)), "by_status": log["status"].value_counts().to_dict(),
        "quote_rows": int(len(q)), "two_sided_rows": int(len(two)),
        "crossed_rows": int((q["bid"] > q["ask"]).sum()),
        "sessions": int(((market["stock"].index >= pd.Timestamp(start))
                         & (market["stock"].index <= pd.Timestamp(end))).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=RULES["underlying"])
    args = ap.parse_args()
    start, end = RULES["period"]["start"], RULES["period"]["end"]
    cache = REPO / "cache" / f"pmcc_raw_{args.ticker}"
    market = E.load_market(cache)

    books = {name: one_book(name, market, start, end, args.ticker) for name in BOOKS}
    out = {
        "ticker": args.ticker, "period": [str(start), str(end)],
        "rules_commit": subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", "config/thresholds.yaml"],
            cwd=REPO, capture_output=True, text=True).stdout.strip(),
        "data": data_summary(cache, market, start, end),
        "books": books,
        "earnings_split": {k: earnings_split(books[k], market["earnings"]) for k in ("pmcc", "cc")},
        "stock": {"first": float(market["stock"]["close"].loc[pd.Timestamp(start):].iloc[0]),
                  "last": float(market["stock"]["close"].loc[:pd.Timestamp(end)].iloc[-1])},
    }
    dest = REPO / "cache" / f"pmcc_results_{args.ticker}.json"
    dest.write_text(json.dumps(clean(out)))
    print(f"wrote {dest}")
    print(f"{'book':10s} {'open cap':>8s} {'min cash':>8s} {'P&L':>7s} {'ret/open%':>9s} "
          f"{'ret/min%':>8s} {'maxDD%':>7s} {'worst mo%':>9s} {'Sharpe':>6s}  cycles")
    for k, b in books.items():
        p, f = b["performance"], b["performance_funded"]
        print(f"{k:10s} {p['capital']:8.0f} {f['capital']:8.0f} {p['pnl']:7.0f} "
              f"{p['return_pct']:9.1f} {f['return_pct']:8.1f} {f['max_drawdown_pct']:7.1f} "
              f"{f['worst_month_pct']:9.1f} {f['sharpe']:6.2f}  {b['status_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
