"""
Term-structure seasonality screen: does a stock's 30-day minus 90-day implied
volatility peak in the same calendar months each year, and do those months pay
a larger variance risk premium?

Every threshold is read from config/screen_thresholds.yaml, which is committed
before this script is first run. Raw pulls go to cache/, which is not committed.

    python3 scripts/term_structure_screen.py            # pull (or reuse cache) and screen
    python3 scripts/term_structure_screen.py --refresh  # force a new pull
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((REPO / "config" / "screen_thresholds.yaml").read_text())

RIC = CFG["universe"][0]
TICKER = RIC.split(".")[0]
IV_RIC = f"{TICKER}ATMIV.U"
CACHE = REPO / "cache" / f"term_structure_{TICKER}.pkl"
OUT = REPO / "cache" / f"term_structure_results_{TICKER}.json"
IV_FIELDS = ["30D_A_IM_C", "30D_A_IM_P", "90D_A_IM_C", "90D_A_IM_P"]
RV_DAYS = 21
EVENT_DAYS = 45


def pull(ric: str) -> dict:
    import lseg.data as ld
    warnings.filterwarnings("ignore", category=FutureWarning)
    (d0, _), (_, v1) = CFG["periods"]["discovery"], CFG["periods"]["validation"]
    ld.open_session()
    try:
        iv = ld.get_history(universe=IV_RIC, fields=IV_FIELDS, interval="daily",
                            start=f"{d0}-01-01", end=f"{v1}-12-31")
        # Adjusted close: returns are what matter here, and a split is not one.
        px = ld.get_history(universe=ric, fields=["TRDPRC_1"], interval="daily",
                            start=f"{d0 - 1}-12-31", end=f"{v1 + 1}-03-01")
        ev = ld.get_data(ric, ["TR.EventStartDate"],
                         {"SDate": f"{d0}-01-01", "EDate": f"{v1 + 1}-03-01",
                          "EventType": "RES"})
    finally:
        ld.close_session()
    return {"iv": iv, "px": px, "earnings": ev,
            "pulled_at": pd.Timestamp.now().isoformat(timespec="seconds")}


def screen(raw: dict) -> dict:
    (d0, d1), (v0, v1) = CFG["periods"]["discovery"], CFG["periods"]["validation"]
    iv = raw["iv"].apply(pd.to_numeric, errors="coerce")
    iv.index = pd.DatetimeIndex(iv.index)
    close = pd.to_numeric(raw["px"]["TRDPRC_1"], errors="coerce").dropna()
    close.index = pd.DatetimeIndex(close.index)
    earnings = pd.DatetimeIndex(pd.to_datetime(raw["earnings"].iloc[:, 1]).dropna()).normalize()

    days = close.index[(close.index.year >= d0) & (close.index.year <= v1)]
    iv = iv.reindex(days)
    iv30 = iv[["30D_A_IM_C", "30D_A_IM_P"]].mean(axis=1, skipna=False)
    iv90 = iv[["90D_A_IM_C", "90D_A_IM_P"]].mean(axis=1, skipna=False)
    spread = iv30 - iv90

    # ---- coverage -----------------------------------------------------------
    missing = spread.isna()
    coverage = [{"year": int(y), "trading_days": int(len(g)), "missing": int(g.sum())}
                for y, g in missing.groupby(days.year)]
    gap = {"discovery": float(missing[(days.year >= d0) & (days.year <= d1)].mean()),
           "validation": float(missing[(days.year >= v0) & (days.year <= v1)].mean())}
    out = {"pulled_at": raw["pulled_at"], "coverage": coverage, "gap_share": gap,
           "earnings_dates": int(len(earnings))}
    if max(gap.values()) > CFG["stop_if_gap_share_above"]:
        return {**out, "stopped": "gap share above the registered limit"}

    # ---- discovery: which months sit in the year's top quartile -------------
    monthly = spread.groupby([days.year, days.month]).mean().unstack()
    ranks = monthly.rank(axis=1, ascending=False, method="min")
    top = ranks <= CFG["season_rule"]["top_quartile"]
    disc = top.loc[d0:d1]
    counts = disc.sum()
    peak = [int(m) for m in counts.index if counts[m] >= CFG["season_rule"]["min_years"]]
    out["discovery"] = [
        {"month": int(m), "years_in_top_quartile": int(counts[m]),
         "mean_spread": float(monthly.loc[d0:d1, m].mean()), "peak": int(m) in peak}
        for m in counts.index]
    out["peak_months"] = peak

    # ---- validation: one entry per month, premium over the next 21 sessions --
    logret = np.log(close).diff()
    pos = {t: i for i, t in enumerate(close.index)}
    obs = []
    for (y, m), g in iv30.dropna().groupby([iv30.dropna().index.year,
                                            iv30.dropna().index.month]):
        if not v0 <= y <= v1:
            continue
        entry = g.index[0]
        fwd = logret.iloc[pos[entry] + 1: pos[entry] + 1 + RV_DAYS]
        if len(fwd) < RV_DAYS:
            continue
        rv = float(fwd.std(ddof=1) * np.sqrt(252) * 100)
        window_end = entry + pd.Timedelta(days=EVENT_DAYS)
        obs.append({"year": int(y), "month": int(m), "entry": str(entry.date()),
                    "iv30": float(g.iloc[0]), "rv": rv, "premium": float(g.iloc[0]) - rv,
                    "peak": int(m) in peak,
                    "earnings_in_window": bool(((earnings >= entry) & (earnings <= window_end)).any()),
                    "spread_month_mean": float(monthly.loc[y, m])})
    out["observations"] = obs
    df = pd.DataFrame(obs)

    def summarize(x: pd.Series) -> dict:
        if len(x) == 0:
            return {"n": 0}
        t = stats.ttest_1samp(x, 0.0, alternative="greater") if len(x) > 1 else None
        return {"n": int(len(x)), "mean": float(x.mean()), "median": float(x.median()),
                "share_positive": float((x > 0).mean()),
                "t": float(t.statistic) if t else None,
                "p_one_sided": float(t.pvalue) if t else None}

    out["validation"] = {
        "peak": summarize(df.loc[df["peak"], "premium"]),
        "non_peak": summarize(df.loc[~df["peak"], "premium"]),
        "peak_no_earnings": summarize(df.loc[df["peak"] & ~df["earnings_in_window"], "premium"]),
        "non_peak_no_earnings": summarize(df.loc[~df["peak"] & ~df["earnings_in_window"], "premium"]),
        "peak_flagged": int((df["peak"] & df["earnings_in_window"]).sum()),
        "all_flagged": int(df["earnings_in_window"].sum()),
    }
    # How the discovery months ranked out of sample. Not a registered rule.
    out["validation_top_quartile_counts"] = {
        int(m): int(top.loc[v0:v1, m].sum()) for m in top.columns}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    if args.refresh or not CACHE.exists():
        CACHE.parent.mkdir(exist_ok=True)
        pd.to_pickle(pull(RIC), CACHE)
    res = screen(pd.read_pickle(CACHE))
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "observations"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
