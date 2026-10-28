"""
Run config/wheel_managed.yaml: the hold-to-expiry wheel (v1) against the managed
wheel (A) and the managed wheel with the trend filter (B), on all 18
stock-windows already registered, each at its stock's committed pick.

    python3 scripts/run_managed.py          # -> cache/wheel_managed_results.json

Nothing is chosen from these results except what the config names in advance:
which of A and B leads the Mastercard page, decided on the other 8 stocks.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
from trading_app.lib import wheel as W  # noqa: E402

OUT = REPO / "cache" / "wheel_managed_results.json"
# config/wheel_managed.yaml's numbers, as run() reads them
VERSIONS = {
    "v1": {},
    "A": {"take_profit": 0.5, "roll_days": 7, "max_rolls": 2, "early_assignment": True},
    "B": {"take_profit": 0.5, "roll_days": 7, "max_rolls": 2, "early_assignment": True, "trend_sma": 200},
    # added_2026_10_27_fixed: designed after A's and B's results, so not out of sample
    "A2": {"take_profit": 0.5, "roll_days": 7, "max_rolls": 2, "early_assignment": True,
           "redeploy": True, "roll_lower": True},
    "B2": {"take_profit": 0.5, "roll_days": 7, "max_rolls": 2, "early_assignment": True, "trend_sma": 200,
           "redeploy": True, "roll_lower": True},
}
FILLS = ("mid", "cross")
COUNTS = ("puts_written", "assigned", "calls_written", "called_away", "closed_early", "rolled",
          "assigned_early", "called_away_early", "filtered", "share_days_pct")
GROUPS = (("wheel", ["NVDA", "AMD", "DIS", "COST"]), ("wheel_mr", ["MA", "APA", "V", "FISV", "ADI"]))
PAGE_STOCK = "MA"


def stocks() -> list[dict]:
    out = []
    for stem, roots in GROUPS:
        cfg = yaml.safe_load((REPO / "config" / f"{stem}.yaml").read_text())
        und = {**(cfg.get("underlyings") or {}),
               **{t: v for k, b in cfg.items() if k.startswith("added_") for t, v in b.items()
                  if isinstance(v, dict) and "cache" in v}}
        for t in roots:
            pick = json.loads((REPO / "config" / f"{stem}_picks" / f"{t}.json").read_text())["pick"]
            for name in ("train", "test"):
                w = cfg["windows"][name]
                out.append({"stock": t, "window_name": name, "window": [str(w["start"]), str(w["end"])],
                            "cache": und[t]["cache"], "pick": pick})
    return out


def figures(s: dict) -> dict:
    return {"return_pct": s["return_pct"], "sharpe": s["sharpe"], "max_drawdown_pct": s["max_drawdown_pct"],
            "pnl": s["pnl"], **{k: s[k] for k in COUNTS}, "premium": s["attribution"]["premium"]}


def main() -> int:
    warnings.filterwarnings("ignore")
    rows, markets = [], {}
    for sw in stocks():
        m = markets.setdefault(sw["cache"], W.load_market(REPO / sw["cache"]))
        row = {k: sw[k] for k in ("stock", "window_name", "window", "pick")}
        for v, manage in VERSIONS.items():
            for fill in FILLS:
                res = W.run(m, start=sw["window"][0], end=sw["window"][1], fill=fill, manage=manage,
                            ticker=sw["stock"], **sw["pick"])
                row[f"{v}_{fill}"] = figures(W.summary(res, m["tbill"]))
        hold = W.run(m, start=sw["window"][0], end=sw["window"][1], mode="buy_and_hold", ticker=sw["stock"])
        row["hold"] = figures(W.summary(hold, m["tbill"]))
        rows.append(row)
        print(f"{sw['stock']:5} {sw['window_name']:5} Sharpe at mid: v1 {row['v1_mid']['sharpe']:5.2f}  "
              + "  ".join(f"{v} {row[v + '_mid']['sharpe']:5.2f}" for v in VERSIONS if v != "v1")
              + "   return " + "  ".join(f"{v} {row[v + '_mid']['return_pct']:6.1f}%" for v in VERSIONS),
              flush=True)

    helped = {f"{v}_{fill}": sum(r[f"{v}_{fill}"]["sharpe"] > r[f"v1_{fill}"]["sharpe"] for r in rows)
              for v in VERSIONS if v != "v1" for fill in FILLS}
    others = [r for r in rows if r["stock"] != PAGE_STOCK]
    med = {v: float(np.median([r[f"{v}_mid"]["sharpe"] for r in others])) for v in ("A", "B")}
    lead = "B" if med["B"] > med["A"] else "A"
    out = {"versions": VERSIONS, "rows": rows, "helped_of": len(rows), "helped": helped,
           "lead_rule": "the higher median Sharpe at mid across the 16 stock-windows of the other 8 stocks; a tie to A",
           "median_sharpe_others": med, "lead": lead}
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nhelped (Sharpe above v1) out of {len(rows)}: {helped}")
    print(f"median Sharpe at mid on the other 8 stocks: A {med['A']:.3f}, B {med['B']:.3f}: the page leads with {lead}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
