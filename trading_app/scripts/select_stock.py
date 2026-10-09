"""
Rank the registered candidates by their 2010-2018 term spread and print the
table. Every choice here is read from config/selection.yaml, which is
committed before this script is first run.

    python3 scripts/select_stock.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((REPO / "config" / "selection.yaml").read_text())
CACHE = REPO / "cache" / "selection.pkl"
FIELDS = ["30D_A_IM_C", "30D_A_IM_P", "90D_A_IM_C", "90D_A_IM_P"]


def pull() -> dict:
    import lseg.data as ld
    warnings.filterwarnings("ignore", category=FutureWarning)
    start, end = (str(x) for x in CFG["window"])
    out = {}
    ld.open_session()
    try:
        for ticker, ric in CFG["candidates"].items():
            rec = {}
            for key, kwargs in (("iv", dict(universe=f"{ticker}ATMIV.U", fields=FIELDS)),
                                ("px", dict(universe=ric, fields=["TRDPRC_1"]))):
                try:
                    rec[key] = ld.get_history(interval="daily", start=start,
                                              end="2019-02-15" if key == "px" else end, **kwargs)
                except Exception as exc:
                    rec[key] = None
                    rec[f"{key}_error"] = str(exc)[:160].replace("\n", " ")
            out[ticker] = rec
    finally:
        ld.close_session()
    return out


def measure(rec: dict) -> dict:
    start, end = (pd.Timestamp(str(x)) for x in CFG["window"])
    if rec.get("px") is None or rec.get("iv") is None:
        return {"coverage": 0.0}
    close = pd.to_numeric(rec["px"]["TRDPRC_1"], errors="coerce").dropna()
    close.index = pd.DatetimeIndex(close.index)
    days = close.index[(close.index >= start) & (close.index <= end)]
    iv = rec["iv"].apply(pd.to_numeric, errors="coerce")
    iv.index = pd.DatetimeIndex(iv.index)
    iv = iv.reindex(days)
    iv30 = iv[FIELDS[:2]].mean(axis=1, skipna=False)
    iv90 = iv[FIELDS[2:]].mean(axis=1, skipna=False)
    spread = iv30 - iv90
    logret = np.log(close).diff()
    pos = {t: i for i, t in enumerate(close.index)}
    prem = []
    have = iv30.dropna()
    for _, g in have.groupby([have.index.year, have.index.month]):
        fwd = logret.iloc[pos[g.index[0]] + 1: pos[g.index[0]] + 22]
        if len(fwd) == 21:
            prem.append(float(g.iloc[0]) - float(fwd.std(ddof=1) * np.sqrt(252) * 100))
    return {"coverage": float(spread.notna().mean()) if len(days) else 0.0,
            "mean_spread": float(spread.mean()), "share_days_positive": float((spread > 0).mean()),
            "mean_iv30": float(iv30.mean()), "mean_premium": float(np.mean(prem)) if prem else np.nan,
            "months": len(prem)}


def main() -> int:
    if not CACHE.exists():
        pd.to_pickle(pull(), CACHE)
    raw = pd.read_pickle(CACHE)
    table = pd.DataFrame({t: measure(r) for t, r in raw.items()}).T
    table["eligible"] = table["coverage"] >= 0.95
    table = table.sort_values(["eligible", "mean_spread"], ascending=[False, False])
    (REPO / "cache" / "selection_results.json").write_text(table.to_json(orient="index"))
    pd.set_option("display.width", 200)
    print(table.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
