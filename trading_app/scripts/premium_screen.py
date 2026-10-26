"""
Rank config/premium_screen.yaml's universe by its 2010-2018 volatility premium and pick the wheel's
next stock (config/wheel.yaml). Every choice is read from the YAML. The measure, universe, window
and pick rule are unchanged from the screen's registration of 2026-10-23.

    python3 scripts/premium_screen.py

The measure, per month: (iv30 at the month's first trading day with one - realised vol over the
next 21 trading days) / iv30, averaged over the months. The pick is the eligible stock with the
highest mean whose option contracts were not adjusted, and which stayed under $950, over the span
the pull needs. The raw pull is cached; with it present, this makes no LSEG request.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_wheel_raw import open_session  # noqa: E402

CFG_PATH = REPO / "config" / "premium_screen.yaml"
RAW = REPO / "cache" / "premium_screen_raw.pkl"
OUT = REPO / "cache" / "wheel_screen.json"
IV_FIELDS = ["30D_A_IM_C", "30D_A_IM_P"]
FEASIBILITY_SPAN = ("2018-06-01", "2025-01-31")
MAX_CLOSE = 950.0


def measure(iv: pd.DataFrame, close: pd.Series, start, end) -> dict:
    """The registered premium measure for one stock, with what is reported beside it."""
    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    days = close.index[(close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))]
    iv30 = iv[IV_FIELDS].apply(pd.to_numeric, errors="coerce").reindex(days).mean(axis=1, skipna=False)
    logret = np.log(close).diff()
    pos = {t: i for i, t in enumerate(close.index)}
    ratios, points, entries = [], [], []
    have = iv30.dropna()
    for _, g in have.groupby([have.index.year, have.index.month]):
        entry = g.index[0]
        fwd = logret.iloc[pos[entry] + 1: pos[entry] + 22]
        if len(fwd) < 21 or fwd.isna().any():
            continue
        rv = float(fwd.std(ddof=1) * np.sqrt(252) * 100)
        level = float(g.iloc[0])
        ratios.append((level - rv) / level)
        points.append(level - rv)
        entries.append(str(entry.date()))
    return {"coverage": float(iv30.notna().mean()) if len(days) else 0.0, "months": len(ratios),
            "mean_ratio": float(np.mean(ratios)) if ratios else np.nan,
            "mean_points": float(np.mean(points)) if points else np.nan,
            "share_positive": float(np.mean(np.array(points) > 0)) if points else np.nan,
            "entries": entries}


def eligible(m: dict, months_needed: int = 100) -> bool:
    return m["coverage"] >= 0.95 and m["months"] >= months_needed


def feasible(default_close: pd.Series, unadjusted_close: pd.Series) -> dict:
    """No corporate action that adjusted the contracts, and no close above $950, over the pull's span."""
    ratio = (pd.to_numeric(default_close, errors="coerce") / pd.to_numeric(unadjusted_close, errors="coerce")).dropna()
    moved = float(ratio.max() / ratio.min() - 1.0) if len(ratio) else np.nan
    top = float(pd.to_numeric(unadjusted_close, errors="coerce").max())
    why = []
    if not np.isfinite(moved) or moved > 0.01:
        why.append(f"contracts adjusted: default/unadjusted ratio moved {100 * moved:.1f}%")
    if top > MAX_CLOSE:
        why.append(f"closed above ${MAX_CLOSE:.0f} (highest {top:.2f})")
    return {"feasible": not why, "why": "; ".join(why), "ratio_moved_pct": 100 * moved, "max_close": top}


def universe(cfg: dict) -> dict:
    return {t: ric for group in cfg["universe"].values() if isinstance(group, dict) for t, ric in group.items()}


def history(ld, rics, **kwargs):
    """The first RIC that returns rows, and its frame."""
    for ric in rics:
        try:
            got = ld.get_history(interval="daily", universe=ric, **kwargs)
            if got is not None and not got.empty:
                return ric, got
        except Exception:
            continue
    return None, None


def pull_screen(cfg: dict) -> dict:
    import lseg.data as ld
    warnings.filterwarnings("ignore", category=FutureWarning)
    start, end = (str(x) for x in cfg["screen"]["window"])
    out = {}
    open_session(ld)
    try:
        for ticker, ric in universe(cfg).items():
            _, iv = history(ld, [f"{ticker}ATMIV.U"], fields=IV_FIELDS, start=start, end=end)
            used, px = history(ld, [ric, f"{ticker}.N", f"{ticker}.O"], fields=["TRDPRC_1"],
                               start=start, end="2019-02-15")
            out[ticker] = {"iv": iv, "px": px, "ric": used}
    finally:
        ld.close_session()
    return out


def pull_feasibility(ric: str) -> dict:
    import lseg.data as ld
    warnings.filterwarnings("ignore", category=FutureWarning)
    lo, hi = FEASIBILITY_SPAN
    open_session(ld)
    try:
        _, adj = history(ld, [ric], fields=["TRDPRC_1"], start=lo, end=hi)
        _, raw = history(ld, [ric], fields=["TRDPRC_1"], start=lo, end=hi, adjustments=["exchangeCorrection"])
    finally:
        ld.close_session()
    return {"default": adj, "unadjusted": raw}


def main() -> int:
    cfg = yaml.safe_load(CFG_PATH.read_text())
    raw = pd.read_pickle(RAW) if RAW.exists() else {"screen": pull_screen(cfg), "feasibility": {}}
    pd.to_pickle(raw, RAW)
    start, end = cfg["screen"]["window"]

    table = {}
    for ticker, rec in raw["screen"].items():
        if rec["iv"] is None or rec["px"] is None:
            table[ticker] = {"coverage": 0.0, "months": 0, "mean_ratio": np.nan, "ric": rec["ric"], "eligible": False}
            continue
        m = measure(rec["iv"], rec["px"]["TRDPRC_1"], start, end)
        table[ticker] = {**{k: v for k, v in m.items() if k != "entries"}, "ric": rec["ric"], "eligible": eligible(m)}
    ranked = sorted((t for t in table if table[t]["eligible"]), key=lambda t: -table[t]["mean_ratio"])

    chosen = []
    for ticker in ranked:
        if len(chosen) == 3:
            break
        if ticker not in raw["feasibility"]:
            raw["feasibility"][ticker] = pull_feasibility(table[ticker]["ric"])
            pd.to_pickle(raw, RAW)
        f = raw["feasibility"][ticker]
        check = feasible(f["default"]["TRDPRC_1"], f["unadjusted"]["TRDPRC_1"])
        unadj = pd.to_numeric(f["unadjusted"]["TRDPRC_1"], errors="coerce")
        check["close_2018_12_31"] = float(unadj[unadj.index <= pd.Timestamp("2018-12-31")].iloc[-1])
        table[ticker]["feasibility"] = check
        if check["feasible"]:
            chosen.append(ticker)

    result = {"table": table, "ranked": ranked, "pick": chosen[0] if chosen else None, "follow_ups": chosen[1:]}
    OUT.write_text(json.dumps(result, indent=1, default=str))
    rows = pd.DataFrame(table).T.loc[ranked + [t for t in table if t not in ranked]]
    pd.set_option("display.width", 200)
    print(rows[["ric", "coverage", "months", "mean_ratio", "mean_points", "share_positive", "eligible"]].to_string())
    for t in ranked:
        if "feasibility" in table[t]:
            print(f"{t}: {'feasible' if table[t]['feasibility']['feasible'] else 'NOT feasible: ' + table[t]['feasibility']['why']}"
                  f" (2018-12-31 close {table[t]['feasibility']['close_2018_12_31']:.2f})")
    print(f"\npick: {result['pick']}; follow-ups: {result['follow_ups']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
