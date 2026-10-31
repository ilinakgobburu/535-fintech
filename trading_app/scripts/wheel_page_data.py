"""
Everything docs/hw4.html shows about the wheel on Mastercard, in one file:

    python3 scripts/wheel_page_data.py          # -> cache/wheel_page_MA.json

The setting is config/wheel_mr_picks/MA.json, committed before the 2026 test
was run. This script re-runs that pick, the baseline and the benchmarks on both
windows of config/wheel_mr.yaml, and stops unless every 2026 figure reproduces
the one cache/wheel_mr_results_MA.json recorded when the test was run once, and
the training figure reproduces the pick file's. Nothing here chooses anything.

It adds what the page needs beyond the registered figures:

    the daily ledger and the blotter of each book, for the portfolio on any date
    the fill evidence: where LSEG's last trade of the day printed against the
        closing bid and ask, for the contracts the wheel trades and for each
        contract it sold
    the thesis evidence: implied against realised volatility each cycle, and the
        screen's mean-reversion measures over the run window
"""

from __future__ import annotations

import json
import pickle
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trading_app.lib import wheel as W  # noqa: E402
from trading_app.lib.pmcc_daily import carry, monthly_expiries, performance  # noqa: E402
from trading_app.lib.ric import occ_symbol  # noqa: E402

TICKER, RIC, COMPANY = "MA", "MA.N", "Mastercard"
CONFIG = REPO / "config" / "wheel_mr.yaml"
PICK = REPO / "config" / "wheel_mr_picks" / f"{TICKER}.json"
RESULTS = REPO / "cache" / f"wheel_mr_results_{TICKER}.json"
COVERAGE = REPO / "cache" / f"wheel_mr_coverage_{TICKER}.json"
SCREEN = REPO / "cache" / "mr_screen.json"
CBOE_PUT = REPO / "cache" / "cboe" / "PUT_History.csv"     # config/wheel_mr.yaml added_2026_10_28_benchmark
OUT = REPO / "cache" / f"wheel_page_{TICKER}.json"
WF_CONFIG = REPO / "config" / "wheel_wf.yaml"                          # the walk-forward of the setting
WF_PICK = REPO / "config" / "wheel_wf_picks" / f"{TICKER}.json"
WF_RESULTS = REPO / "cache" / f"wheel_wf_results_{TICKER}.json"
BAND = {"delta": (0.10, 0.40), "dte": (20, 36)}     # the contracts the wheel sells: monthly, out of the money
TOL = 1e-6


# --------------------------------------------------------------------------
# the measures (tests/wheel/test_wheel_page.py)
# --------------------------------------------------------------------------

def print_stats(df: pd.DataFrame) -> dict:
    """
    Where the day's last trade printed against the closing quote. `df` has bid,
    ask, mid and trade. Location 0 is the bid, 0.5 the mid, 1 the ask.
    """
    d = df[(df["ask"] > df["bid"])].copy()
    spread = d["ask"] - d["bid"]
    dev = d["trade"] - d["mid"]
    slope, intercept = np.polyfit(d["mid"], d["trade"], 1)
    fit = intercept + slope * d["mid"]
    ss_tot = float(((d["trade"] - d["trade"].mean()) ** 2).sum())
    return {
        "n": int(len(d)),
        "r2": 1.0 - float(((d["trade"] - fit) ** 2).sum()) / ss_tot if ss_tot > 0 else 1.0,
        "slope": float(slope), "intercept": float(intercept),
        "median_abs_dev": float(dev.abs().median()), "median_spread": float(spread.median()),
        "dev_over_spread": float((dev.abs() / spread).median()),
        "within_quarter_pct": 100.0 * float((dev.abs() <= spread / 4 + 1e-9).mean()),
        "outside_pct": 100.0 * float(((d["trade"] < d["bid"] - 1e-9) | (d["trade"] > d["ask"] + 1e-9)).mean()),
        "at_or_below_bid_pct": 100.0 * float((d["trade"] <= d["bid"] + 1e-9).mean()),
        "median_location": float(((d["trade"] - d["bid"]) / spread).median()),
        "median_spread_pct_of_mid": 100.0 * float((spread / d["mid"]).median()),
    }


def location_bins(loc) -> dict:
    """
    Where trades printed in the quote, 0 the bid and 1 the ask, in tenths, with a bin either side for
    prints outside it. A print at the bid or the ask is inside the quote.
    """
    loc = np.asarray(loc, float)
    inside = loc[(loc >= 0) & (loc <= 1)]
    tenths = np.minimum((inside * 10).astype(int), 9)          # the ask itself goes in the last tenth
    counts = [int((loc < 0).sum())] + [int((tenths == i).sum()) for i in range(10)] + [int((loc > 1).sum())]
    labels = ["below the bid"] + [f"{i / 10:.1f}–{(i + 1) / 10:.1f}" for i in range(10)] + ["above the ask"]
    return {"labels": labels, "counts": counts, "pct": [100.0 * c / max(len(loc), 1) for c in counts]}


def realised_vol(close: pd.Series, start, end) -> float:
    """Annualised standard deviation of daily log returns from `start`'s close to `end`'s."""
    c = close[(close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))]
    r = np.log(c.astype(float)).diff().dropna()
    return float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else float("nan")


def jsonable(x):
    """Timestamps as dates, numpy numbers as Python ones, NaN as null, all the way down."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if x is pd.NaT or x is pd.NA:
        return None
    if isinstance(x, pd.Timestamp):
        return str(x.date())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# --------------------------------------------------------------------------
# the industry benchmark
# --------------------------------------------------------------------------

def read_cboe(path: Path) -> pd.Series:
    """Cboe's published daily history: a DATE column (month/day/year) and the index's closing level."""
    d = pd.read_csv(path)
    s = pd.Series(pd.to_numeric(d.iloc[:, 1], errors="coerce").to_numpy(),
                  index=pd.to_datetime(d["DATE"], format="%m/%d/%Y")).dropna().sort_index()
    return s


def index_book(levels: pd.Series, sessions, capital: float) -> pd.DataFrame:
    """`capital` invested in the index at the first session, valued at its closing level on each session."""
    lv = levels.reindex(levels.index.union(pd.DatetimeIndex(sessions))).ffill().reindex(pd.DatetimeIndex(sessions))
    return pd.DataFrame({"date": pd.DatetimeIndex(sessions), "nav": (capital * lv / lv.iloc[0]).to_numpy()})


# --------------------------------------------------------------------------
# the books
# --------------------------------------------------------------------------

def occ(row_cp, strike, expiry) -> str:
    return occ_symbol(TICKER, pd.Timestamp(expiry).date(), float(strike), row_cp)


def ledger_rows(res: dict) -> list[dict]:
    out = []
    for r in res["ledger"].to_dict("records"):
        held = r["short_ric"] is not None
        out.append({
            "date": r["date"], "close": r["stock_close"], "cash": r["cash"], "shares": int(r["shares"]),
            "basis": r["basis"], "stock_mv": r["stock_mv"],
            "short": r["short"], "ric": r["short_ric"],
            "occ": occ(r["short"], r["short_strike"], r["short_expiry"]) if held else None,
            "strike": r["short_strike"], "expiry": r["short_expiry"],
            "contracts": r["short_contracts"], "units": r["short_units"],
            "mark": r["short_mark"], "option_mv": r["option_mv"], "nav": r["nav"]})
    return out


def blotter_rows(res: dict) -> list[dict]:
    """The engine's blotter, with the contract named on its expiry or assignment as well as its sale."""
    out, open_ = [], None
    for e in res["blotter"]:
        row = {k: e.get(k) for k in ("date", "kind", "side", "qty", "price", "cash_delta", "note",
                                     "instrument", "cp", "strike", "expiry", "bid", "ask")}
        if e.get("cp") and e["side"] == W.SELL:
            open_ = e
        elif e["kind"] in ("put", "call") and open_ is not None:
            row.update({k: open_[k] for k in ("instrument", "cp", "strike", "expiry")})
        row["occ"] = occ(row["cp"], row["strike"], row["expiry"]) if row.get("cp") else None
        row["contracts"] = e["qty"] // W.CONTRACT if row.get("cp") else None
        out.append(row)
    return out


def book_summary(res: dict, tbill: pd.Series) -> dict:
    s = W.summary(res, tbill)
    s["vol_pct"] = s["monthly_sd_pct"] * np.sqrt(12)
    return s


SURFACE_GRID = np.round(np.arange(0.80, 1.2001, 0.025), 3)


def vol_views(market: dict, result: dict, by_day: dict) -> dict:
    """The smile on every session, the surface (strike/spot against date) and the greek surfaces."""
    days, smiles, surface = [], [], []
    for day in result["ledger"]["date"]:
        sm = W.smile(market, day, by_day)
        if sm is None or len(sm["points"]) < 3:
            continue
        pts = sm["points"]
        days.append(day)
        smiles.append({"expiry": sm["expiry"], "dte": sm["dte"], "spot": round(sm["spot"], 2),
                       "m": pts["moneyness"].round(4).tolist(), "iv": pts["iv"].round(4).tolist()})
        m, iv = pts["moneyness"].to_numpy(), pts["iv"].to_numpy()
        row = np.interp(SURFACE_GRID, m, iv)
        row[(SURFACE_GRID < m.min()) | (SURFACE_GRID > m.max())] = np.nan
        surface.append([None if not np.isfinite(v) else round(float(v), 4) for v in row])
    return {"smile": {"date": days, "s": smiles},
            "surface": {"date": days, "grid": SURFACE_GRID.tolist(), "iv": surface},
            "greeks": greek_views(market, result)}


# strike / spot, as the lecture's surfaces; finer near the money, where a few days out delta is a cliff
# and a coarser grid leaves the drawn surface off the options traced across it
GREEK_MONEYNESS = np.unique(np.round(np.r_[np.arange(0.70, 0.94, 0.01), np.arange(0.94, 1.06, 0.0025),
                                           np.arange(1.06, 1.3001, 0.01)], 4))
GREEK_DAYS = np.r_[1:11, 12:41:2, 45:121:5]                            # finer where the wheel's options live
CLOCK_DAYS = np.arange(1, 361)


def greek_views(market: dict, result: dict) -> dict:
    """
    The lecture's greek surfaces over strike/spot and days to expiry, held at this window's typical
    Mastercard (median close, median implied vol of the options sold, average bill rate, no yield),
    with every option sold traced across them from its sale to the session before its expiry; and
    the three clocks of one at-the-money option.
    """
    led = result["ledger"]
    spot = float(led["stock_close"].median())
    sold = [c for c in result["cycles"] if c.get("cp")]
    sigma = float(np.median([c["iv"] for c in sold]))
    rate = float(market["tbill"].reindex(led["date"]).mean())
    g = W.spot_greeks(spot, GREEK_MONEYNESS[None, :] * spot, GREEK_DAYS[:, None] / W.DAYS_PER_YEAR, sigma, rate)
    keys = ("call_delta", "put_delta", "call_theta", "put_theta", "gamma", "vega")
    paths = []
    for c in sold:
        held = led[(led["short"] == c["cp"]) & (led["short_expiry"] == c["expiry"])
                   & (led["date"] >= c["entry"]) & (led["date"] < c["expiry"])]
        m = c["strike"] / held["stock_close"].to_numpy(dtype=float)
        dte = (pd.Timestamp(c["expiry"]) - held["date"]).dt.days.to_numpy()
        p = W.spot_greeks(spot, m * spot, dte / W.DAYS_PER_YEAR, sigma, rate)
        side = "call" if c["cp"] == W.CALL else "put"
        paths.append({"occ": occ(c["cp"], c["strike"], c["expiry"]), "cp": c["cp"], "status": c.get("status"),
                      "date": list(held["date"]), "close": held["stock_close"].round(2).tolist(),
                      "m": np.round(m, 4).tolist(), "dte": dte.tolist(),
                      **{k: np.round(p[f"{side}_{k}" if k in ("delta", "theta") else k], 6).tolist()
                         for k in ("delta", "theta", "gamma", "vega")}})
    atm = W.spot_greeks(spot, spot, CLOCK_DAYS / W.DAYS_PER_YEAR, sigma, rate)
    return {"spot": spot, "sigma": sigma, "rate": rate, "m": GREEK_MONEYNESS.tolist(), "dte": GREEK_DAYS.tolist(),
            "surfaces": {k: np.round(g[k], 6).tolist() for k in keys}, "sold": paths,
            "sold_days": [int(min(c["dte"] for c in sold)), int(max(c["dte"] for c in sold))],
            "clocks": {"dte": CLOCK_DAYS.tolist(), "gamma": np.round(atm["gamma"], 6).tolist(),
                       "theta_vol": np.round(atm["theta_vol"], 6).tolist(), "vega": np.round(atm["vega"], 6).tolist()}}


def run_period(market: dict, window: tuple[str, str], pick: dict, base: dict, bench: pd.Series | None = None,
               by_day: dict | None = None) -> dict:
    start, end = window
    runs = {"wheel": W.run(market, start=start, end=end, ticker=TICKER, **pick),
            "wheel_bid": W.run(market, start=start, end=end, fill="cross", ticker=TICKER, **pick),
            "baseline": W.run(market, start=start, end=end, ticker=TICKER, **base),
            "baseline_bid": W.run(market, start=start, end=end, fill="cross", ticker=TICKER, **base),
            "hold": W.run(market, start=start, end=end, mode="buy_and_hold", ticker=TICKER),
            "cash": W.run(market, start=start, end=end, mode="cash", ticker=TICKER)}
    tb = market["tbill"]
    books = {k: book_summary(v, tb) for k, v in runs.items()}
    nav = {"date": list(runs["wheel"]["ledger"]["date"]),
           **{k: list(v["ledger"]["nav"]) for k, v in runs.items()}}
    if bench is not None:
        book = index_book(bench, runs["wheel"]["ledger"]["date"], runs["wheel"]["start_cash"])
        perf = performance(book, runs["wheel"]["start_cash"], tb)
        books["putw"] = {**{k: v for k, v in perf.items() if k != "monthly_returns"},
                         "vol_pct": perf["monthly_sd_pct"] * np.sqrt(12)}
        nav["putw"] = list(book["nav"])
    close = market["stock"]["close"]
    cycles = []
    for c in runs["wheel"]["cycles"]:
        c = dict(c)
        if c.get("cp"):
            c["occ"] = occ(c["cp"], c["strike"], c["expiry"])
        c["realised_vol"] = realised_vol(close, c["entry"], c["expiry"])
        cycles.append(c)
    led = runs["wheel"]["ledger"]
    navs = led.set_index("date")["nav"]
    rt = W.reg_t(led)
    low = int(rt["free_cash"].idxmin())
    top = int(rt["put_collateral"].idxmax())
    regt = {"rows": [[r.date, r.cash, r.put_collateral, r.free_cash, int(r.shares), r.stock_mv,
                      int(r.call_shares_needed), bool(r.covered)] for r in rt.itertuples()],
            "min_free_cash": float(rt.at[low, "free_cash"]), "min_free_date": rt.at[low, "date"],
            "max_collateral": float(rt.at[top, "put_collateral"]), "max_collateral_date": rt.at[top, "date"],
            "max_collateral_contracts": int(led.at[top, "short_contracts"]),
            "max_collateral_strike": float(led.at[top, "short_strike"]), "min_cash": float(rt["cash"].min()),
            "put_days": int((led["short"] == W.PUT).sum()), "call_days": int((led["short"] == W.CALL).sum()),
            "share_days": int((led["shares"] > 0).sum()), "sessions": int(len(led)),
            "always_covered": bool(rt["covered"].all())}
    drawdown = {"date": list(navs.index), "dd": list(navs / navs.cummax().clip(lower=runs["wheel"]["start_cash"]) - 1)}
    return {"window": [start, end], "books": books, "nav": nav, "runs": runs,
            "ledger": ledger_rows(runs["wheel"]), "blotter": blotter_rows(runs["wheel"]),
            "hold_blotter": blotter_rows(runs["hold"]), "cycles": cycles,
            "monthly": W.monthly_nav(led, runs["wheel"]["start_cash"], market["tbill"]),
            "drawdown": drawdown, "regt": regt,
            "vol": vol_views(market, runs["wheel"], by_day) if by_day is not None else None}


def reproduce(periods: dict, stored: dict, picked: dict) -> list[str]:
    """Every registered 2026 figure, and the training Sharpe that chose the pick, as recorded."""
    checks = []
    names = {"pick_mid": "wheel", "pick_cross": "wheel_bid", "baseline_mid": "baseline",
             "baseline_cross": "baseline_bid", "buy_and_hold": "hold", "cash": "cash"}
    for key, book in names.items():
        for f in ("sharpe", "return_pct", "pnl", "max_drawdown_pct"):
            a, b = periods["test"]["books"][book][f], stored["results"][key][f]
            if not (abs(a - b) < TOL or (np.isnan(a) and np.isnan(b))):
                raise SystemExit(f"test {key} {f}: {a} now, {b} recorded: the run does not reproduce")
            checks.append(f"test {key} {f}")
    row = next(r for r in picked["train_grid_mid"]
               if all(r[k] == picked["pick"][k] for k in ("put_delta", "call_delta", "floor")))
    for f in ("sharpe", "return_pct", "pnl"):
        a = periods["train"]["books"]["wheel"][f]
        if abs(a - row[f]) >= TOL:
            raise SystemExit(f"train pick {f}: {a} now, {row[f]} recorded: the run does not reproduce")
        checks.append(f"train pick {f}")
    return checks


def join_folds(navs: list[pd.DataFrame], capital: float, tbill: pd.Series) -> dict:
    """
    Out-of-sample folds, each run from `capital`, as one record: each fold's account values are
    scaled by where the folds before it ended, so performance() puts their month-end returns end to
    end for the Sharpe ratio, compounds their returns and measures the drawdown along the whole path.
    """
    parts, scale, last = [], 1.0, None
    for f in navs:
        d = pd.to_datetime(f["date"])
        if last is not None and d.iloc[0] <= last:
            raise ValueError("folds must follow one another without overlapping")
        parts.append(pd.DataFrame({"date": d, "nav": scale * f["nav"].to_numpy(dtype=float)}))
        scale *= float(f["nav"].iloc[-1]) / capital
        last = d.iloc[-1]
    return performance(pd.concat(parts, ignore_index=True), capital, tbill)


def tied_at_top(grid: list[dict]) -> list[dict]:
    """The settings that share the best training Sharpe, to the 4 decimals the tie rule uses."""
    best = max(round(r["sharpe"], 4) for r in grid)
    return [r for r in grid if round(r["sharpe"], 4) == best]


WF_BOOKS = ("wheel", "wheel_bid", "baseline", "hold", "cash", "putw")


def walk_forward(market: dict, periods: dict, picked: dict, stored: dict, base: dict,
                 bench: pd.Series | None) -> dict | None:
    """
    config/wheel_wf.yaml: the setting chosen again each year with only the years before it. Fold 1
    (picked on 2024, run on 2025) is run here and checked against its recorded result; fold 2 is
    the class run's 2026 test. Each fold's books, and the two joined end to end.
    """
    if not (WF_PICK.exists() and WF_RESULTS.exists()):
        return None
    w = yaml.safe_load(WF_CONFIG.read_text())["windows"]["test"]
    wf_pick, wf_stored = json.loads(WF_PICK.read_text()), json.loads(WF_RESULTS.read_text())
    one = run_period(market, (str(w["start"]), str(w["end"])), wf_pick["pick"], base, bench)
    names = {"pick_mid": "wheel", "pick_cross": "wheel_bid", "baseline_mid": "baseline",
             "buy_and_hold": "hold", "cash": "cash"}
    for key, book in names.items():
        for f in ("sharpe", "return_pct", "max_drawdown_pct"):
            a, b = one["books"][book][f], wf_stored["results"][key][f]
            if not (abs(a - b) < TOL or (np.isnan(a) and np.isnan(b))):
                raise SystemExit(f"walk-forward fold 1 {key} {f}: {a} now, {b} recorded: the run does not reproduce")
    capital = float(one["runs"]["wheel"]["start_cash"])
    two = periods["test"]
    folds = [(one, wf_pick, wf_stored), (two, picked, stored)]
    out = [{"fold": i + 1, "train": P["train"], "test": per["window"], "pick": P["pick"],
            "train_sharpe": P["train_sharpe_mid"], "tied": len(tied_at_top(P["train_grid_mid"])),
            "tied_calls": len({r["call_delta"] for r in tied_at_top(P["train_grid_mid"])}),
            **{f"train_{k}": tied_at_top(P["train_grid_mid"])[0][k] for k in ("puts", "assigned", "calls")},
            "rank": S["pick_rank_of_18"], "books": {k: per["books"][k] for k in WF_BOOKS if k in per["books"]}}
           for i, (per, P, S) in enumerate(folds)]
    joined = {}
    for k in WF_BOOKS:
        if all(k in per["nav"] for per, _, _ in folds):
            perf = join_folds([pd.DataFrame({"date": per["nav"]["date"], "nav": per["nav"][k]}) for per, _, _ in folds],
                              capital, market["tbill"])
            joined[k] = {f: perf[f] for f in ("return_pct", "sharpe", "max_drawdown_pct", "months")}
    return {"folds": out, "joined": joined, "same_pick": wf_pick["pick"] == picked["pick"]}


# --------------------------------------------------------------------------
# fill evidence
# --------------------------------------------------------------------------

def trade_frame(cache: Path, market: dict) -> pd.DataFrame:
    """The engine's quotes with LSEG's last trade, volume and open interest of each day."""
    frames = pickle.loads((cache / "options.pkl").read_bytes())
    extra = []
    for ric, df in frames.items():
        e = pd.DataFrame({"trade": pd.to_numeric(df["TRDPRC_1"], errors="coerce"),
                          "volume": pd.to_numeric(df["ACVOL_UNS"], errors="coerce"),
                          "oi": pd.to_numeric(df["OPINT_1"], errors="coerce")})
        e.index = pd.DatetimeIndex(df.index).normalize()
        e["ric"] = ric
        extra.append(e.rename_axis("date").reset_index())
    q = market["quotes"].merge(pd.concat(extra, ignore_index=True), on=["date", "ric"], how="left")
    imp = []
    for ric, df in frames.items():
        e = pd.DataFrame({"lseg_iv": pd.to_numeric(df["IMP_VOLT"], errors="coerce") / 100.0})
        e.index = pd.DatetimeIndex(df.index).normalize()
        e["ric"] = ric
        imp.append(e.rename_axis("date").reset_index())
    return q.merge(pd.concat(imp, ignore_index=True), on=["date", "ric"], how="left")


def band_rows(q: pd.DataFrame, market: dict, start, end) -> pd.DataFrame:
    """Two-sided, traded, out-of-the-money monthlies 20-36 days out, with delta from the mid."""
    stock = market["stock"]["close"]
    d = q[(q["date"] >= pd.Timestamp(start)) & (q["date"] <= pd.Timestamp(end))
          & q["mid"].notna() & (q["mid"] > 0) & (q["volume"] > 0) & q["trade"].notna()].copy()
    d["spot"] = d["date"].map(stock)
    d["dte"] = (d["expiry"] - d["date"]).dt.days
    d = d[d["dte"].between(*BAND["dte"])
          & np.where(d["cp"] == W.PUT, d["strike"] < d["spot"], d["strike"] > d["spot"])]
    deltas, ivs = [], []
    for r in d.itertuples():
        rate, q_ = carry(market, r.date)
        g = W.option_greeks(r.mid, r.spot, r.strike, r.dte / W.DAYS_PER_YEAR, rate, q_, r.cp)
        deltas.append(abs(g["delta"]))
        ivs.append(g["iv"])
    d["abs_delta"], d["iv"] = deltas, ivs
    return d[d["abs_delta"].between(*BAND["delta"])]


def fill_evidence(q: pd.DataFrame, market: dict, periods: dict) -> dict:
    lo, hi = periods["train"]["window"][0], periods["test"]["window"][1]
    every = q[(q["date"] >= pd.Timestamp(lo)) & (q["date"] <= pd.Timestamp(hi))
              & q["mid"].notna() & (q["mid"] > 0) & (q["volume"] > 0) & q["trade"].notna()]
    band = band_rows(q, market, lo, hi)
    test_lo = pd.Timestamp(periods["test"]["window"][0])
    out = {"band": BAND,
           "all": print_stats(every), "in_band": print_stats(band),
           "in_band_puts": print_stats(band[band["cp"] == W.PUT]),
           "in_band_calls": print_stats(band[band["cp"] == W.CALL]),
           "in_band_train": print_stats(band[band["date"] < test_lo]),
           "in_band_test": print_stats(band[band["date"] >= test_lo]),
           "in_band_volume_median": float(band["volume"].median()),
           "in_band_oi_median": float(band["oi"].median())}
    # where in the quote each trade of the band printed: 0 the bid, 0.5 the mid, 1 the ask
    both = band[band["iv"].notna() & band["lseg_iv"].notna() & (band["lseg_iv"] > 0)]
    out["iv_check"] = {"n": int(len(both)), "corr": float(np.corrcoef(both["iv"], both["lseg_iv"])[0, 1]),
                       "median_abs_diff_pts": float(100 * (both["iv"] - both["lseg_iv"]).abs().median())}
    two = band[band["ask"] > band["bid"]]
    out["locations"] = location_bins((two["trade"] - two["bid"]) / (two["ask"] - two["bid"]))
    # the same trades as points, with the typical half-spread for the scatter's guide lines
    out["scatter"] = {"mid": band["mid"].round(3).tolist(), "trade": band["trade"].round(3).tolist(),
                      "cp": band["cp"].tolist(), "date": [str(d.date()) for d in band["date"]],
                      "half_spread": float(((two["ask"] - two["bid"]) / 2 / two["mid"]).median()),
                      "zoom": float(np.percentile(np.maximum(band["mid"], band["trade"]), 98))}
    # the book's own fills: each contract it sold, on the day it sold it
    idx = q.set_index(["date", "ric"])
    own = []
    for name in ("train", "test"):
        for e in periods[name]["blotter"]:
            if e["side"] != W.SELL or not e.get("cp"):
                continue
            r = idx.loc[(pd.Timestamp(e["date"]), e["instrument"])]
            spread = float(r["ask"] - r["bid"])
            own.append({"period": name, "date": e["date"], "occ": e["occ"], "cp": e["cp"],
                        "strike": e["strike"], "contracts": e["contracts"],
                        "bid": float(r["bid"]), "ask": float(r["ask"]), "mid": float(r["mid"]),
                        "fill": e["price"], "spread": spread, "spread_pct": 100.0 * spread / float(r["mid"]),
                        "trade": r["trade"], "volume": r["volume"], "oi": r["oi"],
                        "location": (float(r["trade"]) - float(r["bid"])) / spread
                        if np.isfinite(r["trade"]) and spread > 0 else np.nan,
                        "mid_minus_bid_dollars": (float(r["mid"]) - float(r["bid"])) * e["qty"]})
    out["own"] = own
    return out


# --------------------------------------------------------------------------
# thesis evidence
# --------------------------------------------------------------------------

def premium_by_cycle(market: dict, start, end) -> list[dict]:
    """Each cycle: the 0.25-delta put's implied vol on entry against the vol the stock then realised."""
    stock = market["stock"]
    sessions = stock.index[(stock.index >= pd.Timestamp(start)) & (stock.index <= pd.Timestamp(end))]
    expiries = monthly_expiries(sessions, start, end)
    by_day = {d: g for d, g in market["quotes"].groupby("date")}
    out = []
    for prev, expiry in zip(expiries[:-1], expiries[1:]):
        day = sessions[sessions > prev][0]
        spot = float(stock.at[day, "close"])
        r, q = carry(market, day)
        p = W.select(by_day.get(day, market["quotes"].iloc[:0]), W.PUT, expiry, spot, day, r, q, 0.25)
        out.append({"entry": day, "expiry": expiry, "iv": float(p["iv"]) if p is not None else np.nan,
                    "realised": realised_vol(stock["close"], day, expiry)})
    return out


def reversal(market: dict) -> dict:
    import mr_screen as S
    scr = json.loads(SCREEN.read_text())
    row = scr["table"][RIC]
    run = S.measures(S.monthly_returns(market["stock"]["close"], "2024-01-01", "2026-09-30"))
    rng = np.random.default_rng(1)
    sims = [S.measures(pd.Series(rng.normal(0, 0.06, run["months"])))["vr"] for _ in range(4000)]
    top = []
    for i, r in enumerate(scr["ranked"][:10], 1):
        m = scr["table"][r]
        f = m.get("feasibility") or {}
        status = ("pick" if r in scr["picks"] else "follow-up" if r in scr["follow_ups"]
                  else ("skipped: " + f["why"]) if f.get("why") else "")
        top.append({"rank": i, "ric": r, "name": m.get("name"), "vr": m["vr"], "rho": m["rho"],
                    "vol": m["vol"], "status": status})
    return {"screen_window": scr["window"], "universe": scr["universe"], "priced": scr["priced"],
            "eligible": scr["eligible"], "ranked": len(scr["ranked"]), "rank": scr["ranked"].index(RIC) + 1,
            "screen": {k: row[k] for k in ("months", "vr", "rho", "vol")}, "run": run,
            "random_walk_vr_mean": float(np.mean(sims)), "random_walk_vr_sd": float(np.std(sims)),
            "top": top, "median_vr": float(np.median([scr["table"][r]["vr"] for r in scr["table"]
                                                    if scr["table"][r]["months"] >= 114]))}


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def commit(path: Path, added: bool = True) -> str:
    args = ["git", "log", "--format=%h %s", "-1"] + (["--diff-filter=A"] if added else []) + ["--", str(path)]
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True).stdout.strip()


def commit_adding(path: Path, text: str) -> str:
    """The commit that first put `text` into `path`: for a dated key added to a config later."""
    args = ["git", "log", "--format=%h %s", "--reverse", "-S", text, "--", str(path)]
    out = subprocess.run(args, cwd=REPO, capture_output=True, text=True).stdout.strip().splitlines()
    return out[0] if out else ""


def data_counts(cache: Path, market: dict) -> dict:
    last = {}
    for line in (cache / "requests.jsonl").read_text().splitlines():
        r = json.loads(line)
        last[r["ric"]] = r["status"]
    q = market["quotes"]
    return {"requests": len(last), **pd.Series(list(last.values())).value_counts().to_dict(),
            "quote_rows": int(len(q)), "two_sided_rows": int(q["mid"].notna().sum()),
            "stock_sessions": int(len(market["stock"]))}


def count_tests() -> int:
    """The tests pytest collects in tests/wheel, the number a reader running it would see."""
    got = subprocess.run([sys.executable, "-m", "pytest", "tests/wheel", "--collect-only", "-q"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    return int(re.search(r"(\d+) tests? collected", got).group(1))


def main() -> int:
    warnings.filterwarnings("ignore")
    cfg = yaml.safe_load(CONFIG.read_text())
    entry = cfg["added_2026_10_26"][TICKER]
    cache = REPO / entry["cache"]
    picked = json.loads(PICK.read_text())
    stored = json.loads(RESULTS.read_text())
    pick, base = picked["pick"], cfg["baseline"]["setting"]
    market = W.load_market(cache)

    bench = read_cboe(CBOE_PUT) if CBOE_PUT.exists() else None
    by_day = {d: g for d, g in market["quotes"].groupby("date")}
    periods = {name: run_period(market, (str(cfg["windows"][name]["start"]), str(cfg["windows"][name]["end"])),
                                pick, base, bench, by_day) for name in ("train", "test")}
    checks = reproduce(periods, stored, picked)
    print(f"reproduced {len(checks)} recorded figures")
    wf = walk_forward(market, periods, picked, stored, base, bench)
    if wf:
        j = wf["joined"]
        print(f"walk-forward 2025-2026: wheel {j['wheel']['return_pct']:+.1f}% Sharpe {j['wheel']['sharpe']:.2f}; "
              f"hold {j['hold']['return_pct']:+.1f}% Sharpe {j['hold']['sharpe']:.2f}")

    q = trade_frame(cache, market)
    fills = fill_evidence(q, market, periods)
    lo, hi = periods["train"]["window"][0], periods["test"]["window"][1]
    prem = premium_by_cycle(market, lo, hi)
    for p in periods.values():
        p.pop("runs")

    out = {
        "ticker": TICKER, "ric": RIC, "company": COMPANY, "pick": pick, "baseline": base,
        "periods": periods, "fills": fills, "premium": prem, "reversal": reversal(market),
        "selection": {"train_grid_mid": picked["train_grid_mid"], "train_grid_bid": picked["train_grid_bid"],
                      "test_grid_mid": stored["test_grid_mid"], "rank": stored["pick_rank_of_18"],
                      "spearman": stored["spearman_train_test"], "helped": stored["selection_helped"]},
        "coverage": json.loads(COVERAGE.read_text()) if COVERAGE.exists() else None,
        "walk_forward": wf,
        "data": data_counts(cache, market), "tests": count_tests(), "reproduced": checks,
        "commits": {"rules": commit(REPO / "config" / "wheel.yaml"),
                    "class_run": commit(CONFIG), "screen_picks": commit_adding(CONFIG, "added_2026_10_26:"),
                    "pick": commit(PICK), "wf_config": commit(WF_CONFIG), "wf_pick": commit(WF_PICK)},
    }
    OUT.write_text(json.dumps(jsonable(out), separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")
    t, f = periods["test"]["books"], fills["in_band"]
    print(f"test: wheel {t['wheel']['return_pct']:+.1f}% Sharpe {t['wheel']['sharpe']:.2f}; "
          f"hold {t['hold']['return_pct']:+.1f}% Sharpe {t['hold']['sharpe']:.2f}")
    print(f"fills, in band: n {f['n']}, median location {f['median_location']:.2f}, "
          f"within a quarter-spread of mid {f['within_quarter_pct']:.0f}%, outside {f['outside_pct']:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
