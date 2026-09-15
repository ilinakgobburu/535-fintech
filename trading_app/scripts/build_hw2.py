"""
Build the covered-call page: one self-contained docs/hw2.html.

Same discipline as the HW1 builder. Every number is computed once here, in the
Python the tests cover, and embedded as JSON; the browser only draws what it
is given. There is exactly one implementation of the arithmetic, so the
blotter, the ledger, the Reg T panel and the write-up cannot disagree.

    python3 scripts/build_hw2.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading_app import theme as T  # noqa: E402
from trading_app.lib import cc_analysis as A  # noqa: E402
from trading_app.lib.covered_call import (  # noqa: E402
    INITIAL_RATE, MAINT_RATE, SHARES_PER_CONTRACT, TRADEABLE_HOURS,
    build_ledger, load_cache, min_start_cash, option_panel, run_backtest,
    stock_panel, trading_weeks,
)

DEFAULT_CACHE = ROOT / "trading_app" / "data" / "covered_call_AAPL.pkl"
DEFAULT_OUT = ROOT.parent / "docs" / "hw2.html"

ORDER_HOUR = 15      # 15:00-16:00 UTC = 11:00 ET. See note in build_payload.
START_CASH = 50_000.0
RULE = "nearest_otm"
SCATTER_CAP = 4000   # points drawn; every statistic uses the full sample


def money(x: float, dp: int = 0) -> str:
    """
    Format a dollar figure the way the PAGE formats it.

    Not decoration. Python's format() rounds halves to even while JavaScript's
    toLocaleString rounds them away from zero, so a premium of exactly
    $3,566.50 prints as 3,566 in this script's console output and 3,567 on the
    page built by it. Two implementations of the arithmetic disagreeing by a
    dollar is precisely the failure this project is arranged to prevent, and a
    build log that contradicts its own artefact is the least excusable version
    of it. So the console rounds the way the browser does.
    """
    q = Decimal(10) ** -dp
    return f"{Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP):,}"


def jnum(x):
    """JSON-safe scalar: NaN/inf become null rather than blowing up the dump."""
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)          # counts must stay ints; "n = 26361.0" reads as a bug
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if not np.isfinite(f) else round(f, 6)
    if isinstance(x, (dt.date, dt.datetime, pd.Timestamp)):
        return str(x)
    return x


def jrec(d: dict) -> dict:
    return {k: jnum(v) for k, v in d.items()}


def jlist(xs) -> list:
    return [jnum(x) for x in xs]


def _strike_step(options: pd.DataFrame, cycles: list[dict]) -> float:
    booked = [c for c in cycles if c.get("strike") is not None]
    steps = []
    for c in booked:
        ks = np.sort(options.loc[options["expiry"] == c["expiry_date"], "strike"].unique())
        near = ks[np.abs(ks - c["strike"]) <= 10.0]
        d = np.diff(near)
        if len(d):
            steps.append(float(d[d > 0].min()))
    return float(np.median(steps)) if steps else float("nan")


def suite_facts() -> dict:
    """
    Count the tests and the mutations by parsing the source.

    The page makes claims about its own test coverage, and a hand-typed "29
    tests" goes stale the first time one is added. HW1 shipped prose hardcoded
    to one ticker's numbers and it silently became false; a count is the same
    hazard in miniature, so it is derived rather than asserted.
    """
    import ast as _ast

    def count(node):
        return sum(
            (isinstance(c, (_ast.FunctionDef, _ast.AsyncFunctionDef))
             and c.name.startswith("test_")) + (count(c) if isinstance(c, _ast.ClassDef) else 0)
            for c in _ast.iter_child_nodes(node))

    # Every test file, discovered rather than listed. Naming one file meant the
    # page under-reported itself the moment a second and third were added --
    # the same staleness this function exists to prevent, one level up.
    per_file = {}
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        per_file[f.name] = count(_ast.parse(f.read_text(encoding="utf-8")))

    mut = ROOT / "scripts" / "mutation_check.py"
    n_mut = 0
    for node in _ast.walk(_ast.parse(mut.read_text(encoding="utf-8"))):
        if isinstance(node, _ast.Assign) and any(
                getattr(t, "id", "") == "MUTATIONS" for t in node.targets):
            n_mut = len(node.value.elts)
    # This assignment's own test files, distinguished from Assignment 1.1's,
    # because the section on the page is about THIS book's coverage and
    # quietly absorbing 1.1's numbers would overstate it.
    mine = {"test_covered_call.py", "test_fetch_shapes.py", "test_page.py"}
    return {
        # "functions", not "tests": parametrised ones expand into several
        # cases each, so this is a floor and never an overstatement. pytest
        # collects more than this number, not fewer.
        "functions": sum(per_file.values()),
        "hw2_functions": sum(n for f, n in per_file.items() if f in mine),
        "hw2_files": sorted(f for f in per_file if f in mine),
        "per_file": per_file,
        "files": len(per_file),
        "mutations": n_mut,
    }


def build_payload(cache: Path) -> dict:
    payload = load_cache(cache)
    stock = stock_panel(payload)
    options = option_panel(payload)
    weeks = trading_weeks(stock)

    run = run_backtest(stock, options, weeks, rule=RULE,
                       order_hour=ORDER_HOUR, start_cash=START_CASH)
    ledger = build_ledger(run, stock, options)
    bh = A.buy_and_hold(stock, ledger, START_CASH)
    fit = A.mid_vs_print(options, stock)
    hours = A.fill_hour_sweep(stock, options, weeks, rule=RULE, start_cash=START_CASH)
    rules = A.rule_sweep(stock, options, weeks, order_hour=ORDER_HOUR,
                         start_cash=START_CASH)
    floor = min_start_cash(run, stock, options)
    ohlc = A.ohlc_integrity(stock)

    # The 1-minute study is precomputed into a small JSON because the minute
    # cache itself is ~390 MB and is not committed. Absent, the page simply
    # omits the section rather than inventing one.
    study_path = ROOT / "trading_app" / "data" / "bar_size_study.json"
    bar_study = (json.loads(study_path.read_text(encoding="utf-8"))
                 if study_path.exists() else None)

    cyc = pd.DataFrame(run["cycles"])
    booked = cyc[cyc["status"].isin(["assigned", "expired"])] if len(cyc) else cyc

    # ---- scatter -------------------------------------------------------
    d = fit["frame"]
    if len(d) > SCATTER_CAP:
        d_draw = d.sample(SCATTER_CAP, random_state=0).sort_values("mid")
    else:
        d_draw = d.sort_values("mid")

    # ---- headline ------------------------------------------------------
    final_nav = float(ledger["nav"].iloc[-1])
    bh_final = float(bh["nav"].iloc[-1]) if len(bh) else np.nan
    premium = float(booked["premium"].sum()) if len(booked) else 0.0

    meta = {
        "ticker": payload.get("ticker", "AAPL"),
        "stock_ric": payload.get("stock_ric", "AAPL.O"),
        "fetched_at": payload.get("fetched_at", "?"),
        "window": payload.get("window", []),
        "interval": payload.get("interval", "1h"),
        "bars": int(len(stock)),
        "option_series": int(options["ric"].nunique()) if len(options) else 0,
        "option_obs": int(len(options)),
        # Strike spacing near the money, measured rather than assumed: the
        # smallest gap between listed strikes on the contracts actually written.
        "strike_step": _strike_step(options, run["cycles"]),
        "weeks": len(weeks),
        "order_hour": ORDER_HOUR,
        "start_cash": START_CASH,
        "rule": RULE,
        "initial_rate": INITIAL_RATE,
        "maint_rate": MAINT_RATE,
        "shares": SHARES_PER_CONTRACT,
        "fetch_stats": payload.get("fetch_stats", {}),
        "suite": suite_facts(),
    }

    return {
        "meta": meta,
        "headline": {
            "final_nav": jnum(final_nav),
            "pnl": jnum(final_nav - START_CASH),
            "pnl_pct": jnum(100.0 * (final_nav / START_CASH - 1.0)),
            "premium": jnum(premium),
            "weeks_booked": int(len(booked)),
            "weeks_skipped": int(len(cyc) - len(booked)) if len(cyc) else 0,
            "assignments": int((booked["status"] == "assigned").sum()) if len(booked) else 0,
            "bh_final": jnum(bh_final),
            "bh_pnl": jnum(bh_final - START_CASH),
            "gap": jnum(final_nav - bh_final),
            "min_available": jnum(float(ledger["available_funds"].min())),
            "min_excess": jnum(float(ledger["excess_liquidity"].min())),
            "ever_infeasible": bool((~ledger["feasible"]).any()),
        },
        "blotter": [jrec(b) for b in run["blotter"]],
        "cycles": [jrec(c) for c in run["cycles"]],
        "ledger": {
            "ts": [str(t) for t in ledger["ts"]],
            "cash": jlist(ledger["cash"]),
            "shares": jlist(ledger["shares"]),
            "stock_mark": jlist(ledger["stock_mark"]),
            "stock_mv": jlist(ledger["stock_mv"]),
            "option_mark": jlist(ledger["option_mark"]),
            "option_mv": jlist(ledger["option_mv"]),
            "call_strike": jlist(ledger["call_strike"]),
            "call_expiry": [x if x else None for x in ledger["call_expiry"]],
            "nav": jlist(ledger["nav"]),
            "lmv": jlist(ledger["lmv"]),
            "initial_margin": jlist(ledger["initial_margin"]),
            "maintenance_margin": jlist(ledger["maintenance_margin"]),
            "available_funds": jlist(ledger["available_funds"]),
            "excess_liquidity": jlist(ledger["excess_liquidity"]),
        },
        "buy_hold": {"ts": [str(t) for t in bh["ts"]], "nav": jlist(bh["nav"])} if len(bh) else None,
        "scatter": {
            "mid": jlist(d_draw["mid"]),
            "trd": jlist(d_draw["trdprc_1"]),
            "moneyness": jlist(d_draw["moneyness"]),
            "moves": jlist(d_draw["num_moves"]),
            "spread": jlist(d_draw["spread"]),
            "strike": jlist(d_draw["strike"]),
            "drawn": int(len(d_draw)),
            "total": int(len(d)),
        },
        "fit": {
            "pooled": jrec(fit["pooled"]),
            "buckets": [jrec(b) for b in fit["buckets"]],
            "by_price": [jrec(b) for b in fit["by_price"]],
            "written": jrec(fit["written"]) if fit["written"] else None,
            "by_moves": [jrec(b) for b in fit["by_moves"]],
            "resid": jrec(fit["resid"]),
        },
        "bar_study": bar_study,
        "ohlc": {
            "bars": ohlc["bars"],
            "thresholds_pct": jlist(ohlc["thresholds_pct"]),
            "worst_high": jrec(ohlc["worst_high"]) if ohlc["worst_high"] else None,
            "high": jrec(ohlc["high"]),
            "low": jrec(ohlc["low"]),
            "close_move": jrec(ohlc["close_move"]),
        },
        "hour_sweep": [jrec(h) for h in hours],
        "rule_sweep": [jrec(r) for r in rules],
        "min_cash": jrec(floor),
        "theme": {
            "base": T.BASE, "panel": T.PANEL, "panel_hi": T.PANEL_HI,
            "line": T.LINE, "line_soft": T.LINE_SOFT,
            "text": T.TEXT, "muted": T.TEXT_MUTED, "faint": T.TEXT_FAINT,
            "mark": T.MARK, "print": T.PRINT, "both": T.BOTH,
            "up": T.UP, "down": T.DOWN, "warn": T.WARN,
            "font": T.FONT, "mono": T.FONT_MONO,
        },
    }


def render_data_page(css: str, meta: dict) -> str:
    """
    Static by construction. Pages cannot reach LSEG, and a page that pretended
    to would be the defect the assignment names ("Live LSEG on Pages is a
    defect"). It says what is baked into the book and how to rebuild it.
    """
    from html import escape as e
    fetched = e(str(meta.get("fetched_at", "?")))
    window = " \u2192 ".join(e(str(x)) for x in meta.get("window", []))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Covered Call \u00b7 Data</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{css}
  .callout{{border-left:3px solid var(--print);background:rgba(201,133,0,.07);
    padding:16px 20px;border-radius:0 8px 8px 0;margin:22px 0;line-height:1.65}}
</style>
</head>
<body>
<div class="wrap">
<nav>
  <div class="brand">Covered <span>Call</span></div>
  <a href="hw2.html">Blotter</a>
  <a class="on" href="data.html">Data</a>
  <a class="ticker-link" href="index.html">\u2190 Assignment 1.1 \u00b7 Surface Lab</a>
</nav>
<header>
  <div class="eyebrow">MEng FinTech \u00b7 Algorithmic Trading II \u00b7 Assignment 2</div>
  <h1>Data connection required</h1>
  <p class="lede">This page cannot reach LSEG from GitHub Pages.</p>
</header>
<div class="callout" id="data-state"><strong>Data connection required.</strong> Pages is static
hosting. LSEG Workspace and the Python desktop session exist only on a local machine, so nothing
here fetches data at view time. The book at <a href="hw2.html" style="color:var(--both)">hw2.html</a>
still works \u2014 that is what gets graded.</div>
<h2>What is baked into the book</h2>
<ul class="notes">
  <li>{e(str(meta.get("stock_ric", "")))} and its call chain, {e(str(meta.get("interval", "")))}
      bars, {window}.</li>
  <li>{meta.get("option_series", 0)} call series, {meta.get("option_obs", 0):,} option bars,
      {meta.get("bars", 0)} underlying bars. LSEG pull of {fetched}.</li>
</ul>
<h2>How to re-pull it</h2>
<p class="notes">With LSEG Workspace running, from a local checkout:</p>
<pre class="eqn" style="font-family:var(--mono);background:var(--panel-hi);border:1px solid var(--line);border-radius:8px;padding:14px 18px;overflow-x:auto">cd trading_app
python3 scripts/fetch_hw2.py --dry-run    # counts candidates, no session needed
python3 scripts/fetch_hw2.py              # the real pull, writes the cache
python3 scripts/build_hw2.py              # bakes it into docs/hw2.html</pre>
<p class="notes"><a href="hw2.html" style="color:var(--both)">\u2190 Back to the blotter</a></p>
</div>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dump-json", type=Path, default=None,
                    help="write the payload as JSON only (for inspection)")
    args = ap.parse_args()

    p = build_payload(args.cache)
    blob = json.dumps(p, separators=(",", ":"), allow_nan=False)

    if args.dump_json:
        args.dump_json.write_text(blob, encoding="utf-8")
        print(f"wrote {args.dump_json} ({len(blob)/1e6:.2f} MB)")

    html = (ROOT / "scripts" / "hw2_template.html").read_text(encoding="utf-8")

    # The stylesheet has ONE source of truth: the HW1 template. Copying it
    # would let the two pages drift into looking like different sites, and
    # HW1 is finished, so it is read rather than edited.
    hw1 = (ROOT / "scripts" / "page_template.html").read_text(encoding="utf-8")
    css = re.search(r"<style>(.*?)</style>", hw1, re.S)
    if not css:
        raise RuntimeError("could not find the shared <style> block in page_template.html")
    html = html.replace("/*__STYLE__*/", css.group(1))

    app = (ROOT / "scripts" / "hw2_app.js").read_text(encoding="utf-8")
    html = html.replace("/*__APP__*/", app)
    html = html.replace("/*__PAYLOAD__*/null", blob)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size/1e6:.2f} MB)")

    # The assignment: "A Data page on github.io must show Data connection
    # required." A section inside the book is not a page, so this writes one,
    # beside the book, in the same stylesheet. Only on the default output:
    # a build to a scratch path (the staleness test) must not touch docs/.
    if args.out.resolve() == DEFAULT_OUT.resolve():
        data_page = args.out.parent / "data.html"
        data_page.write_text(render_data_page(css.group(1), p["meta"]), encoding="utf-8")
        print(f"wrote {data_page}")

    m, h = p["meta"], p["headline"]
    print(f"  {m['ticker']}  {m['weeks']} weeks  {m['bars']} bars  "
          f"{m['option_series']} call series / {m['option_obs']} option bars")
    print(f"  booked {h['weeks_booked']}  skipped {h['weeks_skipped']}  "
          f"assigned {h['assignments']}")
    print(f"  premium ${money(h['premium'])}   NAV ${money(h['final_nav'])}  "
          f"(P&L ${money(h['pnl'])})")
    print(f"  buy & hold ${money(h['bh_final'])}   gap ${money(h['gap'])}")
    print(f"  pooled R2 {p['fit']['pooled']['r2']}  n={p['fit']['pooled']['n']}")
    print(f"  min available funds ${money(h['min_available'])}  "
          f"min start cash ${money(p['min_cash']['min_cash'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
