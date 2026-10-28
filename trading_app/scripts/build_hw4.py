"""
Build docs/hw4.html, the wheel page, from two inputs:

    cache/wheel_page_MA.json   every number, written by scripts/wheel_page_data.py
    scripts/hw4_prose.md       every sentence of prose

As on the PMCC page, prose is never generated: `{{name}}` in hw4_prose.md is
replaced by a computed figure, and an unknown placeholder stops the build.
The page carries its own stylesheet (STYLE below, colours first), so restyling
it touches no other page.

    python3 scripts/wheel_page_data.py && python3 scripts/build_hw4.py
    python3 scripts/build_hw4.py --facts      # the placeholders and their values
    python3 scripts/build_hw4.py --draft      # mark unwritten prose on the page
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from html import escape
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DATA = REPO / "cache" / "wheel_page_MA.json"
PORTFOLIO = REPO / "cache" / "wheel_portfolio_results.json"      # Part II, scripts/run_portfolio.py
PROSE = ROOT / "scripts" / "hw4_prose.md"
RULES = REPO / "config" / "wheel.yaml"
OUT = REPO / "docs" / "hw4.html"
REPO_URL = "https://github.com/ilinakgobburu/535-fintech/tree/main/trading_app"
PERIOD_LABEL = {"test": "2026 test: the result", "train": "2024–25 training: where the setting was chosen"}


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

def usd(x, dp=0, sign=False) -> str:
    if x is None:
        return "–"
    lead = "−" if x < 0 else ("+" if sign and x > 0 else "")
    return f"{lead}${abs(x):,.{dp}f}"


def pct(x, dp=1, sign=False) -> str:
    if x is None:
        return "–"
    lead = "−" if x < 0 else ("+" if sign and x > 0 else "")
    return f"{lead}{abs(x):.{dp}f}%"


def num(x, dp=2) -> str:
    return "–" if x is None else f"{x:,.{dp}f}".replace("-", "−")


def short_commit(s: str) -> str:
    return s.split()[0] if s else "–"


def table(headers: list[str], rows: list[list], left: int = 1, cls: str = "") -> str:
    """`left` leading columns are text; the rest are right-aligned figures."""
    th = "".join(f'<th class="{"l" if i < left else ""}">{escape(h)}</th>' for i, h in enumerate(headers))
    body = "".join("<tr" + (' class="hi"' if r and r[-1] == "__hi__" else "") + ">" + "".join(
        f'<td class="{"l" if i < left else ""}">{c}</td>'
        for i, c in enumerate(r[:-1] if r and r[-1] == "__hi__" else r)) + "</tr>" for r in rows)
    return f'<div class="tbl-wrap"><table class="tbl {cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


# --------------------------------------------------------------------------
# facts and prose
# --------------------------------------------------------------------------

def premium_split(D: dict, period: str) -> tuple[int, int, float, float]:
    test_lo = D["periods"]["test"]["window"][0]
    rows = [p for p in D["premium"] if p["iv"] is not None and p["realised"] is not None
            and ((p["entry"] >= test_lo) == (period == "test"))]
    over = sum(p["iv"] > p["realised"] for p in rows)
    return over, len(rows), sum(p["iv"] for p in rows) / len(rows), sum(p["realised"] for p in rows) / len(rows)


def portfolio_commit() -> str:
    import subprocess
    out = subprocess.run(["git", "log", "--format=%h", "--diff-filter=A", "-1", "--", "config/wheel_portfolio.yaml"],
                         cwd=REPO, capture_output=True, text=True).stdout.strip()
    return out or "–"


def part2_facts(P2: dict) -> dict[str, str]:
    f = {"p2_commit": portfolio_commit()}
    for n, tag in (("train", "train"), ("test", "test")):
        w = P2["windows"][n]
        port, avg, hold = w["wheel_mid"]["portfolio"], w["wheel_mid"]["average_of_five"], w["hold"]["portfolio"]
        f.update({f"p2_{tag}_return": pct(port["return_pct"], sign=True), f"p2_{tag}_sharpe": num(port["sharpe"]),
                  f"p2_{tag}_vol": pct(port["vol_pct"]), f"p2_{tag}_avg_vol": pct(avg["vol_pct"]),
                  f"p2_{tag}_dd": pct(port["max_drawdown_pct"]), f"p2_{tag}_avg_dd": pct(avg["max_drawdown_pct"]),
                  f"p2_{tag}_together": num(w["wheel_mid"]["sharpe_if_together"]),
                  f"p2_{tag}_avg_sharpe": num(avg["sharpe"]),
                  f"p2_{tag}_hold_return": pct(hold["return_pct"], sign=True), f"p2_{tag}_hold_sharpe": num(hold["sharpe"]),
                  f"p2_corr_{tag}": num(w["median_correlation"])})
    f["p2_fisv_train"] = pct(P2["windows"]["train"]["wheel_mid"]["each"]["FISV"]["return_pct"], sign=True)
    return f


def facts(D: dict) -> dict[str, str]:
    t = D["periods"]["test"]["books"]
    w, h, c, b, wb = t["wheel"], t["hold"], t["cash"], t["baseline"], t["wheel_bid"]
    f = D["fills"]["in_band"]
    rv = D["reversal"]
    over, n, _, _ = premium_split(D, "test")
    return {
        "test_wheel_return": pct(w["return_pct"], sign=True), "test_hold_return": pct(h["return_pct"], sign=True),
        "test_cash_return": pct(c["return_pct"], sign=True), "test_bid_return": pct(wb["return_pct"], sign=True),
        "test_wheel_dd": pct(w["max_drawdown_pct"]), "test_hold_dd": pct(h["max_drawdown_pct"]),
        "test_wheel_sharpe": num(w["sharpe"]), "test_hold_sharpe": num(h["sharpe"]),
        "test_baseline_sharpe": num(b["sharpe"]),
        "test_puts": str(w["puts_written"]), "test_assigned": str(w["assigned"]),
        "test_calls": str(w["calls_written"]), "test_called_away": str(w["called_away"]),
        "test_cycles": str(w["cycles"]),
        "test_iv_over_rv": str(over), "test_premium_cycles": str(n),
        "vr_screen": num(rv["screen"]["vr"]), "vr_run": num(rv["run"]["vr"]),
        "vr_median": num(rv["median_vr"]),
        "rw_vr": f'{num(rv["random_walk_vr_mean"])} ± {num(rv["random_walk_vr_sd"])}',
        "universe": str(rv["universe"]), "rank": str(D["selection"]["rank"]),
        "pick_commit": short_commit(D["commits"]["pick"]), "screen_commit": short_commit(D["commits"]["screen_picks"]),
        "band_n": f'{f["n"]:,}', "band_location": num(f["median_location"]), "band_slope": num(f["slope"], 3),
        "band_r2": num(f["r2"], 3), "band_within": pct(f["within_quarter_pct"], 0),
        "band_at_bid": pct(f["at_or_below_bid_pct"], 0), "band_outside": pct(f["outside_pct"], 0),
        "band_volume": f'{D["fills"]["in_band_volume_median"]:,.0f}', "band_oi": f'{D["fills"]["in_band_oi_median"]:,.0f}',
        "tests": str(D["tests"]), "reproduced": str(len(D["reproduced"])),
    }


def load_prose(f: dict[str, str]) -> dict[str, str]:
    """`## key` opens a section; blank line = paragraph; "- " = bullet; "//" = note, never shown."""
    sections: dict[str, list[str]] = {}
    key = None
    for line in PROSE.read_text(encoding="utf-8").splitlines():
        if line.startswith("//"):
            continue
        if line.startswith("## "):
            key = line[3:].strip()
            sections[key] = []
        elif key is not None:
            sections[key].append(line)

    def sub(m):
        name = m.group(1).strip()
        if name not in f:
            raise SystemExit(f"hw4_prose.md uses {{{{{name}}}}}, which is not a computed figure. "
                             f"Run build_hw4.py --facts for the list.")
        return f[name]

    out = {}
    for key, lines in sections.items():
        html, para, items = [], [], []

        def flush():
            if para:
                html.append("<p>" + " ".join(para) + "</p>")
                para.clear()
            if items:
                html.append("<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
                items.clear()

        for line in lines:
            text = re.sub(r"\{\{(.*?)\}\}", sub, escape(line.strip(), quote=False))
            text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
            if not text:
                flush()
            elif text.startswith("- "):
                if para:
                    flush()
                items.append(text[2:])
            else:
                if items:
                    flush()
                para.append(text)
        flush()
        out[key] = "".join(html)
    return out


# --------------------------------------------------------------------------
# static blocks
# --------------------------------------------------------------------------

def verdict_table(D: dict) -> str:
    t = D["periods"]["test"]["books"]
    w, h, c, b = t["wheel"], t["hold"], t["cash"], t["baseline"]
    over, n, iv, rv = premium_split(D, "test")
    run = D["reversal"]["run"]
    held, missed = '<span class="ok">Held</span>', '<span class="no">Did not hold</span>'
    gap = w["return_pct"] - h["return_pct"]
    rows = [
        ["Options are priced above the moves that follow",
         "0.25-delta put's implied vol on entry against the vol realised to expiry",
         f"{over} of {n} cycles; {pct(100 * iv)} against {pct(100 * rv)} on average",
         held if over > n / 2 else missed],
        ["A higher Sharpe ratio than buying the stock", "Sharpe ratio, monthly, T-bill as risk-free",
         f"{num(w['sharpe'])} against {num(h['sharpe'])}", held if w["sharpe"] > h["sharpe"] else missed],
        ["Less risk than buying the stock", "Maximum drawdown; annualised volatility of monthly returns",
         f"{pct(w['max_drawdown_pct'])} against {pct(h['max_drawdown_pct'])}; "
         f"{pct(w['vol_pct'])} against {pct(h['vol_pct'])}",
         held if w["max_drawdown_pct"] > h["max_drawdown_pct"] else missed],
        ["A return close to the stock's", "Return over the test",
         f"{pct(w['return_pct'], sign=True)} against {pct(h['return_pct'], sign=True)}",
         f'<span class="mid">{"ahead" if gap >= 0 else "behind"} by {abs(gap):.1f} points</span>'],
        ["Beats T-bills", "Return over the test", f"{pct(w['return_pct'], sign=True)} against {pct(c['return_pct'], sign=True)}",
         held if w["return_pct"] > c["return_pct"] else missed],
        ["Mastercard's monthly moves keep reversing",
         "The screen's own rule: variance ratio below 1 and autocorrelation below 0, over 2024-01 to 2026-09",
         f"variance ratio {num(run['vr'])}, autocorrelation {num(run['rho'], 2)}",
         held if run["vr"] < 1 and run["rho"] < 0 else missed],
        ["Choosing the setting on 2024–25 helps", "Registered rule: the pick beats the default setting's Sharpe at mid and at bid",
         f"{num(w['sharpe'])} against {num(b['sharpe'])} at mid; ranked {D['selection']['rank']} of 18",
         held if D["selection"]["helped"] else missed],
    ]
    return table(["Claim", "Measured by", "2026 test", ""], rows, left=3, cls="verdict")


def screen_table(D: dict) -> str:
    rows = []
    for r in D["reversal"]["top"]:
        status = r["status"]
        why = ("closed above $950" if "950" in status else "contracts adjusted, not a split"
               if "adjusted" in status else "stopped trading" if "stopped" in status else "")
        tag = ('<span class="ok">pick</span>' if status == "pick" else
               '<span class="mid">follow-up</span>' if status == "follow-up" else
               f'<span class="faint">skipped: {why}</span>' if why else "")
        name = re.sub(r"\s+Ord$", "", str(r["name"]).title())
        row = [str(r["rank"]), escape(r["ric"]), escape(name), num(r["vr"], 3), num(r["rho"], 2),
               pct(100 * r["vol"], 0), tag]
        rows.append(row + (["__hi__"] if r["ric"] == D["ric"] else []))
    return table(["#", "RIC", "Company", "Variance ratio", "Autocorr.", "Vol", ""], rows, left=3, cls="small")


def grid_table(D: dict) -> str:
    S = D["selection"]
    test = {(r["put_delta"], r["call_delta"], r["floor"]): r for r in S["test_grid_mid"]}
    p = D["pick"]
    rows = []
    for r in S["train_grid_mid"]:
        k = (r["put_delta"], r["call_delta"], r["floor"])
        tr = test[k]
        row = [num(r["put_delta"]), num(r["call_delta"]), "on" if r["floor"] else "off",
               num(r["sharpe"]), pct(r["return_pct"], sign=True), num(tr["sharpe"]), pct(tr["return_pct"], sign=True)]
        if k == (p["put_delta"], p["call_delta"], p["floor"]):
            row.append("__hi__")
        rows.append(row)
    return table(["Put Δ", "Call Δ", "Floor", "Train Sharpe", "Train return", "Test Sharpe", "Test return"],
                 rows, left=3, cls="small")


def fill_tables(D: dict) -> tuple[str, str]:
    F = D["fills"]
    names = [("all", "Every traded contract-day"), ("in_band", "The band the wheel sells"),
             ("in_band_puts", "  puts"), ("in_band_calls", "  calls"),
             ("in_band_train", "  2024–25"), ("in_band_test", "  2026")]
    rows = [[escape(label), f'{F[k]["n"]:,}', num(F[k]["median_location"]), num(F[k]["slope"], 3), num(F[k]["r2"], 3),
             usd(F[k]["median_spread"], 2), pct(F[k]["median_spread_pct_of_mid"]), pct(F[k]["within_quarter_pct"], 0),
             pct(F[k]["at_or_below_bid_pct"], 0), pct(F[k]["outside_pct"], 0)] for k, label in names]
    stats = table(["Sample", "n", "Median location", "Slope", "R²", "Median spread", "of mid",
                   "Within ¼-spread", "At or below bid", "Outside quote"], rows, cls="small")
    own = [[o["date"], escape(o["occ"]), "put" if o["cp"] == "P" else "call", str(o["contracts"]),
            f'{num(o["bid"])} / {num(o["ask"])}', num(o["fill"], 3), pct(o["spread_pct"]),
            num(o["trade"]), f'{o["volume"]:,.0f}' if o["volume"] is not None else "–",
            f'{o["oi"]:,.0f}' if o["oi"] is not None else "–", num(o["location"])] for o in F["own"]]
    own_t = table(["Date", "Contract", "Type", "Contracts", "Bid / ask", "Fill (mid)", "Spread", "Last trade",
                   "Volume", "Open int.", "Location"], own, left=3, cls="small")
    return stats, own_t


def rules_list(D: dict) -> str:
    r = yaml.safe_load(RULES.read_text())["rules"]
    p = D["pick"]
    lines = [
        ("Setting", f'puts at {p["put_delta"]:.2f} delta, calls at {p["call_delta"]:.2f} delta, cost-basis floor '
                    f'{"on" if p["floor"] else "off"}; chosen on 2024–25 out of 18 settings'),
        ("Cycle", r["cycle"]), ("Cash phase", r["cash_phase"]), ("Share phase", r["share_phase"]),
        ("Cost basis", r["cost_basis"]), ("Expiry", r["expiry"]), ("Quotes", r["listed"]),
        ("Volatility", r["iv_and_delta"]), ("Rate", r["rate"]), ("Cash", r["cash"]), ("Dividends", r["dividends"]),
        ("Earnings", r["earnings"]), ("Contracts", "whole real contracts of 100 shares"),
        ("Fills", "mid of the closing bid and ask; stress test at the bid (the book never buys an option back)"),
        ("Commissions", r["commissions"]),
    ]
    cap = lambda s: " ".join(str(s).split())[:1].upper() + " ".join(str(s).split())[1:]
    return '<dl class="rules">' + "".join(f"<dt>{escape(a)}</dt><dd>{escape(cap(b))}</dd>" for a, b in lines) + "</dl>"


# --------------------------------------------------------------------------
# data for the page's script
# --------------------------------------------------------------------------

def rounded(x, dp=4):
    if isinstance(x, float):
        return round(x, dp)
    if isinstance(x, list):
        return [rounded(v, dp) for v in x]
    if isinstance(x, dict):
        return {k: rounded(v, dp) for k, v in x.items()}
    return x


def page_data(D: dict) -> dict:
    out = {"ticker": D["ticker"], "periods": {}}
    keep = ("return_pct", "pnl", "sharpe", "max_drawdown_pct", "vol_pct", "worst_month", "worst_month_pct",
            "puts_written", "assigned", "calls_written", "called_away", "held_no_call", "share_days_pct", "cycles")
    for name, P in D["periods"].items():
        books = {k: {**{f: v.get(f) for f in keep}, "attribution": v["attribution"]} for k, v in P["books"].items()}
        L = P["ledger"]
        out["periods"][name] = {
            "window": P["window"], "books": books,
            "nav": {k: [round(x, 2) for x in v] if k != "date" else v for k, v in P["nav"].items()},
            "ledger": [[r["date"], r["close"], r["cash"], r["shares"], r["basis"], r["short"], r["occ"], r["strike"],
                        r["expiry"], r["contracts"], r["mark"], r["option_mv"], r["nav"]] for r in L],
            "blotter": P["blotter"], "cycles": P["cycles"]}
    F = D["fills"]["scatter"]
    out["scatter"] = F
    return rounded(out)


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------

def part2_tables(P2: dict) -> tuple[str, str, str]:
    W = P2["windows"]
    books = [("Five wheels together", lambda w: w["wheel_mid"]["portfolio"]),
             ("Five wheels, fills at the bid", lambda w: w["wheel_cross"]["portfolio"]),
             ("One wheel, average of the five", lambda w: w["wheel_mid"]["average_of_five"]),
             ("Five stocks bought and held", lambda w: w["hold"]["portfolio"]),
             ("T-bills", lambda w: w["cash"])]
    rows = []
    for label, get in books:
        row = [escape(label)]
        for n in ("train", "test"):
            x = get(W[n])
            row += [pct(x["return_pct"], sign=True), "–" if label == "T-bills" else num(x["sharpe"]),
                    pct(x["max_drawdown_pct"]), pct(x["vol_pct"])]
        rows.append(row + (["__hi__"] if label == "Five wheels together" else []))
    hdr = ["", "2024–25 return", "Sharpe", "Max drawdown", "Volatility", "2026 return", "Sharpe", "Max drawdown", "Volatility"]
    results = table(hdr, rows, cls="small")
    each = []
    for s_ in P2["stocks"]:
        row = [escape(s_)]
        for n in ("train", "test"):
            e = W[n]["wheel_mid"]["each"][s_]
            row += [pct(e["return_pct"], sign=True), num(e["sharpe"]), pct(e["max_drawdown_pct"])]
        each.append(row)
    each_t = table(["Stock", "2024–25 return", "Sharpe", "Max drawdown", "2026 return", "Sharpe", "Max drawdown"],
                   each, cls="small")
    corr = ""
    for n, label in (("train", "2024–25"), ("test", "2026")):
        c = W[n]["correlations"]
        names = list(c)
        corr += f'<div class="sub">{label}</div>' + table(
            [""] + names, [[escape(a)] + [num(c[a][b]) for b in names] for a in names], cls="small")
    return results, each_t, corr


def part2_data(P2: dict) -> dict:
    out = {}
    for n in ("train", "test"):
        w = P2["windows"][n]
        out[n] = {"date": w["wheel_mid"]["nav"]["date"], "wheel": w["wheel_mid"]["nav"]["value"],
                  "hold": w["hold"]["nav"]["value"], "cash": w["cash"]["nav"]["value"]}
    return out


PART2_SCRIPT = r"""
const P2 = /*__P2__*/null;
if (typeof Plotly !== 'undefined') {
  const css2 = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  [['train', 'p2-train'], ['test', 'p2-test']].forEach(([k, id]) => {
    const d = P2[k];
    Plotly.newPlot(id, [
      {x: d.date, y: d.cash, name: 'T-bills', mode: 'lines', line: {color: css2('--cash'), width: 1.4}},
      {x: d.date, y: d.hold, name: 'Five stocks held', mode: 'lines', line: {color: css2('--hold'), width: 1.4}},
      {x: d.date, y: d.wheel, name: 'Five wheels together', mode: 'lines', line: {color: css2('--wheel'), width: 2.2}}],
      {paper_bgcolor: css2('--panel'), plot_bgcolor: css2('--panel'), font: {family: css2('--font'), color: css2('--muted'), size: 12},
       margin: {l: 70, r: 12, t: 10, b: 36}, hovermode: 'x unified', legend: {orientation: 'h', y: -0.16},
       xaxis: {gridcolor: css2('--line-soft')}, yaxis: {tickformat: '$,.0f', gridcolor: css2('--line-soft')}},
      {displayModeBar: false, responsive: true});
  });
}
"""


STYLE = """
  /* ---- colours and type: restyle the page here ----------------------- */
  :root{
    --base:#0B1020; --panel:#141A2E; --panel-hi:#1B2340; --line:#26304A; --line-soft:#1D2438;
    --text:#E8ECF5; --muted:#8792AB; --faint:#5A6580;
    --accent:#9085e9; --wheel:#19C3FF; --wheel-bid:#7FDBFF; --hold:#FF3DA6; --cash:#8792AB;
    --up:#199e70; --down:#e5604d; --warn:#c98500;
    --font:'Inter',ui-sans-serif,system-ui,-apple-system,sans-serif;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    --fs-xs:12px; --fs-sm:13px; --fs-base:15px;
  }
  *{box-sizing:border-box} html,body{margin:0;padding:0}
  body{background:var(--base);color:var(--text);font-family:var(--font);font-size:var(--fs-base);
    line-height:1.6;-webkit-font-smoothing:antialiased}
  .wrap{max-width:1440px;margin:0 auto;padding:0 24px 80px}
  @media (max-width:640px){.wrap{padding:0 16px 60px}}
  a{color:var(--wheel)}

  .chip{font-family:var(--mono);font-size:var(--fs-xs);color:var(--muted);border:1px solid var(--line);
    border-radius:999px;padding:4px 10px;background:var(--panel);white-space:nowrap}
  .chip b{color:var(--text);font-weight:600} .chip a{color:inherit;text-decoration:none}
  .chip.hot{background:var(--hold);border-color:var(--hold);color:#0B1020;font-weight:700}

  .grid{display:grid;gap:12px;margin:12px 0}
  .g-2-1{grid-template-columns:minmax(0,2fr) minmax(0,1fr)}
  .g-1-1{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}
  .g-3-2{grid-template-columns:minmax(0,3fr) minmax(0,2fr)}
  .g-4{grid-template-columns:repeat(4,minmax(0,1fr))}
  @media (max-width:1180px){.g-4{grid-template-columns:repeat(2,minmax(0,1fr))}}
  @media (max-width:980px){.g-2-1,.g-1-1,.g-3-2,.g-4{grid-template-columns:minmax(0,1fr)}}

  .panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;min-width:0;display:flex;flex-direction:column}
  .panel > .ph{display:flex;align-items:center;gap:10px;padding:9px 14px;border-bottom:1px solid var(--line);
    font-size:11px;letter-spacing:.11em;text-transform:uppercase;color:var(--muted);font-weight:600}
  .panel > .ph .tag{margin-left:auto;font-family:var(--mono);letter-spacing:0;text-transform:none;font-weight:400;color:var(--faint)}
  .panel > .pb{padding:12px 14px;min-width:0}
  .panel > .pb.flush{padding:4px}
  .prose p{margin:0 0 10px} .prose p:last-child{margin-bottom:0}
  .prose ul{padding-left:20px;margin:0} .prose li{margin-bottom:6px}
  .prose li strong:first-child{color:var(--accent)}
  .lead{font-size:16.5px;line-height:1.6}
  .sub{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:600;margin:14px 0 8px}
  .cap{font-size:var(--fs-sm);color:var(--muted);margin-top:8px}
  .todo{border:1px dashed var(--warn);color:var(--warn);padding:8px 12px;border-radius:6px;font-size:var(--fs-sm)}

  .switch{display:flex;gap:6px;flex-wrap:wrap;margin:18px 0 0}
  .switch button{font:inherit;font-size:var(--fs-sm);color:var(--muted);background:var(--panel);border:1px solid var(--line);
    border-radius:999px;padding:6px 14px;cursor:pointer}
  .switch button.on{color:#0B1020;background:var(--wheel);border-color:var(--wheel);font-weight:600}

  .kpis{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:10px;margin:12px 0}
  @media (max-width:1280px){.kpis{grid-template-columns:repeat(4,minmax(0,1fr))}}
  @media (max-width:640px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
  .kpi{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px;min-width:0;position:relative}
  .kpi .k{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);padding-right:18px}
  .kpi .v{font-family:var(--mono);font-size:20px;font-weight:700;margin-top:5px}
  .kpi .d{font-family:var(--mono);font-size:var(--fs-xs);color:var(--faint);margin-top:4px}
  .kpi .tip{position:absolute;top:10px;right:10px;margin:0}
  .pos{color:var(--up)} .neg{color:var(--down)}

  .tbl-wrap{overflow-x:auto;max-width:100%}
  table.tbl{border-collapse:collapse;width:100%;font-size:var(--fs-sm)}
  table.tbl th,table.tbl td{padding:6px 10px;border-bottom:1px solid var(--line-soft);text-align:right;white-space:nowrap}
  table.tbl th{color:var(--muted);font-weight:600;font-size:var(--fs-xs);text-transform:uppercase;letter-spacing:.05em}
  table.tbl td{font-family:var(--mono)}
  table.tbl .l{text-align:left}
  table.tbl td.l{font-family:var(--font)}
  table.tbl tr.hi td{background:rgba(25,195,255,.10)}
  table.tbl.verdict td{white-space:normal} table.tbl.verdict td:nth-child(3){font-family:var(--mono)}
  table.tbl.small td,table.tbl.small th{padding:5px 8px}
  table.tbl tbody tr.click{cursor:pointer} table.tbl tbody tr.click:hover td{background:var(--panel-hi)}
  table.tbl tbody tr.sel td{background:rgba(144,133,233,.18)}
  .note{display:block;min-width:280px;white-space:normal;color:var(--faint);font-size:var(--fs-xs);font-family:var(--font);line-height:1.45}
  .ok{color:var(--up);font-weight:600} .no{color:var(--down);font-weight:600} .mid{color:var(--warn)} .faint{color:var(--faint)}
  .ev-open{color:var(--wheel)} .ev-close{color:var(--muted)} .ev-assign{color:var(--warn)} .ev-stock{color:var(--up)}
  .ev-div{color:var(--accent)}

  dl.rules{margin:0;display:grid;grid-template-columns:auto minmax(0,1fr);gap:10px 18px}
  dl.rules dt{font-family:var(--mono);font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--accent);padding-top:2px;white-space:nowrap}
  dl.rules dd{margin:0;line-height:1.5}
  @media (max-width:640px){dl.rules{grid-template-columns:minmax(0,1fr)} dl.rules dt{padding-top:8px}}

  .tabs{display:flex;gap:2px;flex-wrap:wrap;padding:6px 8px 0;border-bottom:1px solid var(--line)}
  .tabs button{font:inherit;font-size:var(--fs-sm);color:var(--muted);background:none;border:none;
    border-bottom:2px solid transparent;padding:8px 12px;cursor:pointer}
  .tabs button.on{color:var(--text);border-bottom-color:var(--wheel)}
  .pane{display:none;padding:10px} .pane.on{display:block}
  .pane .tbl-wrap{max-height:560px;overflow-y:auto}
  .pane table.tbl thead th{position:sticky;top:0;background:var(--panel-hi);z-index:1}
  details summary{cursor:pointer;color:var(--muted);font-size:var(--fs-sm);margin:10px 0 6px}

  /* the portfolio on any date */
  .pf-controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  .pf-controls input[type=range]{flex:1 1 260px;accent-color:var(--wheel)}
  .pf-controls input[type=date]{font:inherit;font-size:var(--fs-sm);background:var(--panel-hi);color:var(--text);
    border:1px solid var(--line);border-radius:6px;padding:4px 8px;color-scheme:dark}
  .pf-controls button{font:inherit;font-size:var(--fs-sm);background:var(--panel-hi);color:var(--text);
    border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}
  .pf-controls button:hover{border-color:var(--wheel)}
  .pf{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:12px}
  @media (max-width:980px){.pf{grid-template-columns:repeat(2,minmax(0,1fr))}}
  @media (max-width:520px){.pf{grid-template-columns:minmax(0,1fr)}}
  .pf .cell{background:var(--panel-hi);border:1px solid var(--line);border-radius:8px;padding:10px 12px;min-width:0}
  .pf .cell .k{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
  .pf .cell .v{font-family:var(--mono);font-size:19px;font-weight:700;margin-top:4px}
  .pf .cell .d{font-family:var(--mono);font-size:var(--fs-xs);color:var(--faint);margin-top:3px;line-height:1.5}
  .pf-eq{font-family:var(--mono);font-size:var(--fs-sm);color:var(--muted);margin-top:10px;overflow-wrap:anywhere}
  .pf-eq b{color:var(--text)}
  .pf-events{margin-top:10px;font-size:var(--fs-sm);color:var(--muted)}
  .pf-events div{padding:3px 0;border-top:1px solid var(--line-soft)}

  /* Part I and Part II */
  html{scroll-behavior:smooth}
  .site{position:sticky;top:0;z-index:30;display:flex;align-items:center;gap:16px;flex-wrap:wrap;
    padding:10px 0;background:var(--base);border-bottom:1px solid var(--line)}
  .site .brand{display:flex;align-items:baseline;gap:10px;color:var(--text);text-decoration:none}
  .site .brand .sym{font-family:var(--mono);font-weight:700;font-size:18px}
  .site .brand .co{font-weight:600}
  .site .brand .name{color:var(--muted);font-size:var(--fs-sm)}
  .site .right{margin-left:auto;display:flex;gap:8px;flex-wrap:wrap}
  .site a.chip{text-decoration:none}
  .menus{display:flex;gap:4px;flex-wrap:wrap}
  .menu{position:relative}
  .menu > button{font:inherit;font-size:var(--fs-sm);font-weight:600;color:var(--text);background:none;
    border:1px solid transparent;border-radius:8px;padding:6px 12px;cursor:pointer}
  .menu > button:hover,.menu.open > button,.menu > button:focus-visible{background:var(--panel-hi);border-color:var(--line);outline:none}
  .menu .caret{color:var(--muted);margin-left:6px;font-size:11px}
  .dropdown{display:none;position:absolute;top:calc(100% + 6px);left:0;min-width:250px;padding:6px;z-index:40;
    background:var(--panel-hi);border:1px solid var(--line);border-radius:10px;box-shadow:0 14px 34px rgba(0,0,0,.5)}
  .dropdown::before{content:"";position:absolute;left:0;right:0;top:-8px;height:8px}
  .menu.open .dropdown{display:block}
  @media (hover:hover) and (min-width:641px){.menu:hover:not(.shut) .dropdown{display:block}}
  .dropdown a{display:block;padding:7px 10px;border-radius:6px;color:var(--text);text-decoration:none;font-size:var(--fs-sm)}
  .dropdown a:hover,.dropdown a:focus{background:var(--panel);outline:none}
  .dropdown a.top{color:var(--accent);font-weight:600}
  .dropdown .grp{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);padding:9px 10px 3px}
  .meta{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 0}
  section.panel,.switch,.grid{scroll-margin-top:76px}
  @media (max-width:640px){.site{position:static}.menus{width:100%}.menu{flex:1 1 auto}
    .menu > button{width:100%;text-align:left;border-color:var(--line)}
    .dropdown{position:static;margin-top:6px;min-width:0;box-shadow:none}.site .right{margin-left:0}}
  .part{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin:30px 0 4px;padding-top:18px;
    border-top:2px solid var(--line);scroll-margin-top:70px}
  .part:first-of-type{border-top:none;padding-top:0;margin-top:18px}
  .part .no{font-family:var(--mono);font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#0B1020;
    background:var(--accent);border-radius:6px;padding:3px 9px;font-weight:700}
  .part h2{margin:0;font-size:22px;font-weight:600;letter-spacing:-.01em}
  .part .what{color:var(--muted);font-size:var(--fs-sm);flex-basis:100%}
  .tip{position:relative;display:inline-flex;align-items:center;justify-content:center;width:15px;height:15px;
    margin-left:7px;border:1px solid var(--line);border-radius:50%;font:600 10px/1 var(--mono);color:var(--muted);
    cursor:help;text-transform:none;letter-spacing:0;flex:none}
  .tip .bubble{display:none;position:absolute;top:calc(100% + 8px);right:-10px;z-index:40;width:290px;max-width:74vw;
    padding:10px 12px;background:var(--panel-hi);border:1px solid var(--line);border-radius:8px;
    box-shadow:0 10px 28px rgba(0,0,0,.45);color:var(--text);font:400 var(--fs-sm)/1.5 var(--font);text-align:left;white-space:normal}
  .tip:hover .bubble,.tip:focus .bubble{display:block}
"""

SCRIPT = r"""
const D = /*__DATA__*/null;
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const money = (x, dp = 0, sign = false) => x == null ? '–' :
  (x < 0 ? '−' : (sign && x > 0 ? '+' : '')) + '$' + Math.abs(x).toLocaleString('en-US', {minimumFractionDigits: dp, maximumFractionDigits: dp});
const pct = (x, dp = 1, sign = false) => x == null ? '–' : (x < 0 ? '−' : (sign && x > 0 ? '+' : '')) + Math.abs(x).toFixed(dp) + '%';
const num = (x, dp = 2) => x == null ? '–' : x.toFixed(dp).replace('-', '−');
const cls = x => x == null ? '' : (x < 0 ? 'neg' : 'pos');
let period = 'test';

const LABELS = {wheel: 'Wheel (mid fills)', wheel_bid: 'Wheel (fills at the bid)', hold: 'Buy and hold', cash: 'T-bills'};
const COLORS = {wheel: '--wheel', wheel_bid: '--wheel-bid', hold: '--hold', cash: '--cash'};

function eventName(e) {
  if (e.cp && e.side === 'SELL') return ['Sell to open: ' + (e.cp === 'P' ? 'cash-secured put' : 'covered call'), 'ev-open'];
  if (e.side === 'EXPIRE') return ['Close: ' + (e.kind === 'put' ? 'put' : 'call') + ' expired', 'ev-close'];
  if (e.side === 'ASSIGN') return [e.kind === 'put' ? 'Assigned: put' : 'Called away: call', 'ev-assign'];
  if (e.kind === 'stock' && e.side === 'BUY') return ['Buy shares at the strike', 'ev-stock'];
  if (e.kind === 'stock' && e.side === 'SELL') return ['Deliver shares at the strike', 'ev-stock'];
  if (e.side === 'DIVIDEND') return ['Dividend', 'ev-div'];
  return [e.side, ''];
}

function kpis() {
  const B = D.periods[period].books, w = B.wheel, h = B.hold, c = B.cash, b = B.wheel_bid;
  const tiles = [
    ['Wheel return', pct(w.return_pct, 1, true), 'buy and hold ' + pct(h.return_pct, 1, true), w.return_pct],
    ['Sharpe ratio', num(w.sharpe), 'buy and hold ' + num(h.sharpe), w.sharpe],
    ['Max drawdown', pct(w.max_drawdown_pct), 'buy and hold ' + pct(h.max_drawdown_pct), w.max_drawdown_pct],
    ['Return at the bid', pct(b.return_pct, 1, true), 'stress test on fills', b.return_pct],
    ['Premium collected', money(w.attribution.premium), (w.puts_written + w.calls_written) + ' options sold', null],
    ['T-bill interest', money(w.attribution.interest), 'T-bills alone ' + pct(c.return_pct, 1, true), null],
    ['Puts sold / assigned', w.puts_written + ' / ' + w.assigned, 'in shares on ' + pct(w.share_days_pct, 0) + ' of sessions', null],
    ['Calls sold / called away', w.calls_written + ' / ' + w.called_away, w.held_no_call + ' cycles held with no call', null],
  ];
  $('kpis').innerHTML = tiles.map(([k, v, d, x]) =>
    `<div class="kpi"><div class="k">${esc(k)}${TIPS[k] || ''}</div><div class="v ${cls(x)}">${v}</div><div class="d">${esc(d)}</div></div>`).join('');
}

function navChart() {
  const P = D.periods[period], N = P.nav;
  const traces = ['cash', 'hold', 'wheel_bid', 'wheel'].map(k => ({
    x: N.date, y: N[k], name: LABELS[k], mode: 'lines',
    line: {color: css(COLORS[k]), width: k === 'wheel' ? 2.2 : 1.4, dash: k === 'wheel_bid' ? 'dot' : 'solid'},
    hovertemplate: '%{y:$,.0f}<extra>' + LABELS[k] + '</extra>'}));
  const idx = new Map(N.date.map((d, i) => [d, i]));
  const groups = [['Put sold', e => e.cp === 'P' && e.side === 'SELL', 'triangle-down'],
                  ['Call sold', e => e.cp === 'C' && e.side === 'SELL', 'triangle-up'],
                  ['Assigned or called away', e => e.side === 'ASSIGN', 'x']];
  groups.forEach(([name, f, sym]) => {
    const ev = P.blotter.filter(f);
    traces.push({x: ev.map(e => e.date), y: ev.map(e => N.wheel[idx.get(e.date)]), name, mode: 'markers',
      marker: {symbol: sym, size: 9, color: sym === 'x' ? css('--warn') : css('--text')},
      text: ev.map(e => esc(e.note)), hovertemplate: '%{text}<extra>' + name + '</extra>'});
  });
  Plotly.react('plot-nav', traces, {
    paper_bgcolor: css('--panel'), plot_bgcolor: css('--panel'),
    font: {family: css('--font'), color: css('--muted'), size: 12},
    margin: {l: 64, r: 16, t: 10, b: 36}, hovermode: 'closest', legend: {orientation: 'h', y: -0.14},
    xaxis: {gridcolor: css('--line-soft')}, yaxis: {title: 'account value', tickformat: '$,.0f', gridcolor: css('--line-soft')},
  }, {displayModeBar: false, responsive: true});
}

// ---- the portfolio on any date -------------------------------------------
let pfIndex = 0;
function setDay(i) {
  const L = D.periods[period].ledger;
  pfIndex = Math.max(0, Math.min(L.length - 1, i));
  const [date, close, cash, shares, basis, short, occ, strike, expiry, contracts, mark, optMv, nav] = L[pfIndex];
  $('pf-range').value = pfIndex; $('pf-date').value = date;
  const optVal = optMv == null ? 0 : -optMv;
  const opt = short == null
    ? `<div class="v">none</div><div class="d">no option open after this close</div>`
    : `<div class="v">${contracts} ${short === 'P' ? 'short put' : 'covered call'}${contracts > 1 ? 's' : ''}</div>
       <div class="d">${esc(occ)}<br>strike ${money(strike, 2)}, expires ${esc(expiry)}<br>
       mark ${money(mark, 3)} per share, owed ${money(optVal, 0)}</div>`;
  const prem = D.periods[period].blotter.filter(e => e.cp && e.side === 'SELL' && e.date <= date)
    .reduce((s, e) => s + e.cash_delta, 0);
  $('pf').innerHTML = `
    <div class="cell"><div class="k">Cash</div><div class="v">${money(cash)}</div>
      <div class="d">${short === 'P' ? 'of which ' + money(strike * contracts * 100) + ' secures the put' : 'earning the T-bill'}</div></div>
    <div class="cell"><div class="k">Shares</div><div class="v">${shares.toLocaleString()}</div>
      <div class="d">${shares ? 'worth ' + money(shares * close) + ' at ' + money(close, 2) + '<br>cost basis ' + money(basis, 2) + ' per share' : 'close ' + money(close, 2)}</div></div>
    <div class="cell"><div class="k">Open option</div>${opt}</div>
    <div class="cell"><div class="k">Account value</div><div class="v">${money(nav)}</div>
      <div class="d">${pct(100 * (nav / 100000 - 1), 2, true)} since the start<br>premium collected so far ${money(prem)}</div></div>`;
  $('pf-eq').innerHTML = `${esc(date)}: cash <b>${money(cash)}</b> + shares <b>${money(shares * close)}</b> − option owed <b>${money(optVal)}</b> = account value <b>${money(nav)}</b>`;
  const evs = D.periods[period].blotter.filter(e => e.date <= date).slice(-3).reverse();
  $('pf-events').innerHTML = '<div class="sub" style="margin:4px 0">Latest trades on or before this date</div>' +
    (evs.length ? evs.map(e => `<div>${esc(e.date)} · ${esc(eventName(e)[0])} · ${esc(e.occ || 'MA')} · ${money(e.cash_delta, 2, true)}</div>`).join('') : '<div>none yet</div>');
  document.querySelectorAll('#ledger tbody tr').forEach((tr, j) => tr.classList.toggle('sel', j === pfIndex));
}

function jump(dir) {
  const P = D.periods[period], L = P.ledger, cur = L[pfIndex][0];
  const days = [...new Set(P.blotter.map(e => e.date))].sort();
  const t = dir > 0 ? days.find(d => d > cur) : [...days].reverse().find(d => d < cur);
  if (t) setDay(L.findIndex(r => r[0] === t));
}

// ---- tables --------------------------------------------------------------
function tables() {
  const P = D.periods[period];
  $('blotter').innerHTML = '<div class="tbl-wrap"><table class="tbl"><thead><tr>' +
    ['Date', 'Event', 'Instrument', 'Side', 'Quantity', 'Bid / ask', 'Fill', 'Cash', 'Rule that fired']
      .map((h, i) => `<th class="${i < 4 ? 'l' : ''}">${h}</th>`).join('') + '</tr></thead><tbody>' +
    P.blotter.map(e => {
      const [name, c] = eventName(e);
      const qty = e.cp ? `${e.contracts} contract${e.contracts > 1 ? 's' : ''} (${e.qty} sh)` : `${e.qty.toLocaleString()} sh`;
      return `<tr><td class="l">${esc(e.date)}</td><td class="l ${c}">${esc(name)}</td><td class="l">${esc(e.occ || 'MA (stock)')}</td>
        <td class="l">${esc(e.side)}</td><td>${qty}</td><td>${e.bid != null ? num(e.bid) + ' / ' + num(e.ask) : '–'}</td>
        <td>${e.price ? num(e.price, 3) : '–'}</td><td class="${cls(e.cash_delta)}">${money(e.cash_delta, 2, true)}</td>
        <td class="l"><span class="note">${esc(e.note)}</span></td></tr>`;
    }).join('') + '</tbody></table></div>';

  $('ledger').innerHTML = '<div class="tbl-wrap"><table class="tbl"><thead><tr>' +
    ['Date', 'Close', 'Cash', 'Shares', 'Cost basis', 'Open option', 'Contracts', 'Mark', 'Option owed', 'Account value']
      .map((h, i) => `<th class="${i === 0 || i === 5 ? 'l' : ''}">${h}</th>`).join('') + '</tr></thead><tbody>' +
    P.ledger.map(([date, close, cash, shares, basis, short, occ, strike, expiry, contracts, mark, optMv, nav], j) =>
      `<tr class="click" data-i="${j}"><td class="l">${date}</td><td>${num(close)}</td><td>${money(cash)}</td><td>${shares}</td>
       <td>${shares ? num(basis) : '–'}</td><td class="l">${occ ? esc(occ) : '–'}</td><td>${contracts ?? '–'}</td>
       <td>${mark != null ? num(mark, 3) : '–'}</td><td>${optMv ? money(-optMv) : '–'}</td><td>${money(nav)}</td></tr>`).join('') +
    '</tbody></table></div>';
  $('ledger').querySelectorAll('tr.click').forEach(tr => tr.addEventListener('click', () => {
    setDay(+tr.dataset.i); $('portfolio').scrollIntoView({behavior: 'smooth', block: 'start'});
  }));

  $('cycles').innerHTML = '<div class="tbl-wrap"><table class="tbl"><thead><tr>' +
    ['Entry', 'Expiry', 'Days', 'Sold', 'Contract', 'Strike', 'From spot', 'Delta', 'Implied vol', 'Realised vol',
     'Premium', 'Close at expiry', 'Outcome'].map((h, i) => `<th class="${i < 2 || i === 3 || i === 4 || i === 12 ? 'l' : ''}">${h}</th>`).join('') +
    '</tr></thead><tbody>' + P.cycles.map(c => `<tr><td class="l">${c.entry}</td><td class="l">${c.expiry}</td><td>${c.dte}</td>
      <td class="l">${c.cp === 'P' ? 'put' : c.cp === 'C' ? 'call' : '–'}</td><td class="l">${esc(c.occ || '–')}</td>
      <td>${c.strike != null ? num(c.strike) : '–'}</td><td>${c.otm_pct != null ? pct(c.otm_pct, 1, true) : '–'}</td>
      <td>${c.delta != null ? num(c.delta) : '–'}</td><td>${c.iv != null ? pct(100 * c.iv) : '–'}</td>
      <td>${c.realised_vol != null ? pct(100 * c.realised_vol) : '–'}</td><td>${c.premium != null ? money(c.premium) : '–'}</td>
      <td>${c.settle != null ? num(c.settle) : '–'}</td><td class="l">${esc(c.status || '')}</td></tr>`).join('') +
    '</tbody></table></div>';
}

function scatter() {
  const S = D.scatter, put = S.cp.map(c => c === 'P');
  const pick = (arr, want) => arr.filter((_, i) => put[i] === want);
  const hi = Math.max(...S.mid, ...S.trade);
  Plotly.newPlot('plot-fill', [
    {x: pick(S.mid, true), y: pick(S.trade, true), name: 'puts', mode: 'markers', marker: {size: 4, color: css('--wheel'), opacity: .55},
     text: pick(S.date, true), hovertemplate: 'mid %{x:$.2f} → trade %{y:$.2f}<br>%{text}<extra>put</extra>'},
    {x: pick(S.mid, false), y: pick(S.trade, false), name: 'calls', mode: 'markers', marker: {size: 4, color: css('--hold'), opacity: .55},
     text: pick(S.date, false), hovertemplate: 'mid %{x:$.2f} → trade %{y:$.2f}<br>%{text}<extra>call</extra>'},
    {x: [0, hi], y: [0, hi], name: 'trade = mid', mode: 'lines', line: {color: css('--faint'), dash: 'dot', width: 1}, hoverinfo: 'skip'}],
    {paper_bgcolor: css('--panel'), plot_bgcolor: css('--panel'), font: {family: css('--font'), color: css('--muted'), size: 12},
     margin: {l: 56, r: 12, t: 10, b: 44}, hovermode: 'closest', legend: {orientation: 'h', y: -0.2},
     xaxis: {title: 'closing mid', tickformat: '$,.0f', gridcolor: css('--line-soft')},
     yaxis: {title: 'last trade of the day', tickformat: '$,.0f', gridcolor: css('--line-soft')}},
    {displayModeBar: false, responsive: true});
}

function show(p) {
  period = p;
  document.querySelectorAll('.switch button').forEach(b => b.classList.toggle('on', b.dataset.period === p));
  document.querySelectorAll('[data-window]').forEach(el => el.textContent = D.periods[p].window.join(' to '));
  const L = D.periods[p].ledger;
  $('pf-range').max = L.length - 1; $('pf-date').min = L[0][0]; $('pf-date').max = L[L.length - 1][0];
  kpis(); tables();
  if (typeof Plotly !== 'undefined') navChart();
  setDay(L.length - 1);
}

document.querySelectorAll('.switch button').forEach(b => b.addEventListener('click', () => show(b.dataset.period)));
$('pf-range').addEventListener('input', e => setDay(+e.target.value));
$('pf-date').addEventListener('change', e => {
  const L = D.periods[period].ledger, v = e.target.value;
  let j = L.findIndex(r => r[0] >= v); if (j < 0) j = L.length - 1;
  if (L[j][0] > v && j > 0) j -= 1;          // the last close on or before the chosen day
  setDay(j);
});
$('pf-prev').addEventListener('click', () => jump(-1));
$('pf-next').addEventListener('click', () => jump(1));
document.querySelectorAll('.tabs button').forEach(btn => btn.addEventListener('click', () => {
  const root = btn.closest('.panel');
  root.querySelectorAll('.tabs button').forEach(b => b.classList.toggle('on', b === btn));
  root.querySelectorAll('.pane').forEach(p => p.classList.toggle('on', p.dataset.pane === btn.dataset.tab));
}));
show('test');
if (typeof Plotly !== 'undefined') scatter();

// ---- the header's dropdowns: hover on a laptop, tap anywhere; Escape or a click elsewhere closes
const menus = [...document.querySelectorAll('.menu')];
const shutAll = () => menus.forEach(m => { m.classList.remove('open'); m.querySelector('button').setAttribute('aria-expanded', 'false'); });
menus.forEach(m => {
  const b = m.querySelector('button');
  b.addEventListener('click', e => {
    e.stopPropagation();
    const opening = !m.classList.contains('open');
    shutAll();
    if (opening) { m.classList.add('open'); b.setAttribute('aria-expanded', 'true'); }
  });
  m.querySelectorAll('.dropdown a').forEach(a => a.addEventListener('click', () => {
    if (a.dataset.tab != null) {
      const t = document.querySelector(`#records .tabs button[data-tab="${a.dataset.tab}"]`);
      if (t) t.click();
    }
    shutAll();
    m.classList.add('shut');                       // stays closed under the pointer until it leaves
    m.addEventListener('mouseleave', () => m.classList.remove('shut'), {once: true});
  }));
});
document.addEventListener('click', shutAll);
document.addEventListener('keydown', e => { if (e.key === 'Escape') shutAll(); });
"""


MENU_PART1 = [
    ("Results", [("Thesis", "#s-thesis"), ("Did it work?", "#s-did-it-work"), ("Returns and account value", "#results"),
                 ("Portfolio on any date", "#portfolio")]),
    ("Trades", [("Blotter", "#records", 0), ("Ledger", "#records", 1), ("Cycles", "#records", 2), ("Rules", "#records", 3)]),
    ("Method", [("Fill prices", "#s-fill-prices"), ("Stock choice", "#s-stock-choice"), ("Setting choice", "#s-setting-choice"),
                ("Accuracy", "#s-accuracy"), ("Reporting", "#s-reporting"), ("Limitations", "#s-limitations")]),
]
MENU_PART2 = [
    ("", [("Motivation", "#s-motivation"), ("Results", "#s-results"), ("Account value", "#p2-charts"),
          ("What it shows", "#s-what-it-shows"), ("Each stock's wheel", "#s-each-stock-s-wheel")]),
]


def menus(part2: bool) -> str:
    """The header's dropdowns: one per part, each listing its sections (a Trades item also opens its tab)."""
    def one(label, top, groups):
        items = "".join(
            (f'<div class="grp">{escape(g)}</div>' if g else "")
            + "".join(f'<a href="{it[1]}"' + (f' data-tab="{it[2]}"' if len(it) > 2 else "") + f'>{escape(it[0])}</a>'
                      for it in links)
            for g, links in groups)
        return (f'<div class="menu"><button type="button" aria-haspopup="true" aria-expanded="false">{escape(label)}'
                f'<span class="caret">▾</span></button><div class="dropdown"><a class="top" href="{top}">'
                f'{escape(label)}: start</a>{items}</div></div>')
    return (one("Part I · The assignment", "#part1", MENU_PART1)
            + (one("Part II · Five stocks", "#part2", MENU_PART2) if part2 else ""))


def tip(prose: dict[str, str], label: str) -> str:
    body = prose.get(f"tip: {label}", "").strip()
    if not body:
        return ""
    return f'<span class="tip" tabindex="0" aria-label="About {escape(label)}">i<span class="bubble">{body}</span></span>'


def render(D: dict, prose: dict[str, str], draft: bool, P2: dict | None = None) -> str:
    missing = []

    def say(key: str, cls: str = "") -> str:
        body = prose.get(key, "").strip()
        if body:
            return f'<div class="prose {cls}">{body}</div>'
        missing.append(key)
        return f'<div class="todo">Not yet written: {escape(key)}</div>' if draft else ""

    def panel(title: str, body: str, tag: str = "", flush: bool = False, pid: str = "") -> str:
        pid = pid or "s-" + re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")       # the header menus link here
        return (f'<section class="panel" id="{pid}"><div class="ph">{escape(title)}{tip(prose, title)}'
                + (f'<span class="tag">{tag}</span>' if tag else "")
                + f'</div><div class="pb{" flush" if flush else ""}">{body}</div></section>')

    T, R = D["periods"]["test"], D["periods"]["train"]
    tips = {k: tip(prose, k) for k in ("Sharpe ratio", "Max drawdown", "Premium collected", "T-bill interest")}
    stats, own = fill_tables(D)
    data = D["data"]
    tabs = [("Blotter", say("blotter") + '<div id="blotter" style="margin-top:8px"></div>'),
            ("Ledger", '<div class="cap" style="margin:0 0 8px">One row per session, at the close. '
                       'Click a row to see that day in the portfolio panel.</div><div id="ledger"></div>'),
            ("Cycles", '<div id="cycles"></div>'),
            ("Rules", rules_list(D))]
    tab_html = ('<div class="tabs">' + "".join(
        f'<button data-tab="{i}" class="{"on" if i == 0 else ""}">{escape(n)}</button>' for i, (n, _) in enumerate(tabs))
        + "</div>" + "".join(f'<div class="pane{" on" if i == 0 else ""}" data-pane="{i}">{b}</div>'
                             for i, (_, b) in enumerate(tabs)))
    script = (SCRIPT.replace("/*__DATA__*/null", json.dumps(page_data(D), separators=(",", ":")))
              .replace("const D =", "const TIPS = " + json.dumps(tips) + ";\nconst D ="))

    parts = [f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wheel Strategy · MA</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>{STYLE}</style></head><body><div class="wrap" id="top">
<header class="site"><a class="brand" href="#top"><span class="sym">MA</span><span class="co">{escape(D["company"])}</span>
<span class="name">Wheel strategy · backtest</span></a>
<nav class="menus" aria-label="Sections">{menus(bool(P2))}</nav>
<div class="right"><a class="chip hot" href="{REPO_URL}">Code (to GitHub Repo) ↗</a>
<a class="chip" href="hw3.html">← PMCC page</a></div></header>
<div class="meta"><span class="chip">test <b>{T["window"][0]}</b> to <b>{T["window"][1]}</b></span>
<span class="chip">trained <b>{R["window"][0]}</b> to <b>{R["window"][1]}</b></span>
<span class="chip">daily closes · LSEG</span></div>""",
        '<div class="part" id="part1"><span class="no">Part I</span><h2>The assignment: the wheel on Mastercard</h2>'
        '<span class="what">Everything the assignment asks for: the thesis and whether it held, the blotter, '
        'the ledger, the portfolio on any date, and the evidence for the fill prices.</span></div>',
        '<div style="height:12px"></div>',
        panel("Thesis", say("thesis", "lead")),
        '<div style="height:12px"></div>',
        panel("Did it work?", verdict_table(D) + '<div style="height:10px"></div>' + say("verdict"), "2026 test"),
        '<div class="switch" id="results"><button data-period="test" class="on">' + escape(PERIOD_LABEL["test"]) + '</button>'
        '<button data-period="train">' + escape(PERIOD_LABEL["train"]) + '</button></div>',
        '<div id="kpis" class="kpis"></div>',
        '<div class="grid g-2-1" style="margin-top:0">',
        panel("Account value", '<div id="plot-nav" style="height:400px"></div>',
              'from $100,000 · <span data-window></span>', flush=True),
        panel("Summary", say("summary"), "2026 test"),
        "</div>",
        panel("Portfolio on any date",
              '<div class="pf-controls"><button id="pf-prev">◀ previous trade</button>'
              '<input id="pf-range" type="range" min="0" value="0" aria-label="session">'
              '<button id="pf-next">next trade ▶</button><input id="pf-date" type="date" aria-label="date"></div>'
              '<div id="pf" class="pf"></div><div id="pf-eq" class="pf-eq"></div><div id="pf-events" class="pf-events"></div>'
              + '<div class="cap">' + re.sub(r"</?p>", "", prose.get("portfolio", "")) + "</div>",
              '<span data-window></span>', pid="portfolio"),
        '<div style="height:12px"></div>',
        f'<section class="panel" id="records">{tab_html}</section>',
        '<div class="grid g-1-1">',
        panel("Fill prices", say("fills") + '<div class="sub">Notes</div>' + say("fills-note")),
        panel("Last trade against the closing mid", '<div id="plot-fill" style="height:380px"></div>'
              f'<div class="cap" style="padding:0 10px 8px">The {D["fills"]["in_band"]["n"]:,} contract-days in the band the '
              'wheel sells, 2024 to October 2026. Points on the dotted line traded exactly at the mid.</div>',
              "LSEG TRDPRC_1 vs (BID + ASK) / 2", flush=True),
        "</div>",
        panel("Where trades printed, by sample", stats
              + "<details><summary>Each option the wheel sold, on the day it sold it</summary>" + own + "</details>",
              "location: 0 = bid, 0.5 = mid, 1 = ask"),
        '<div class="grid g-1-1">',
        panel("Stock choice", say("stock") + '<div class="sub">The screen, top ten</div>' + screen_table(D),
              f'{D["reversal"]["ranked"]} of {D["reversal"]["eligible"]} eligible stocks ranked'),
        panel("Setting choice", say("setting") + '<div class="sub">All 18 settings, at mid fills</div>' + grid_table(D),
              "highlighted: the pick"),
        "</div>",
        '<div class="grid g-1-1">',
        panel("Accuracy", say("accuracy")
              + f'<div class="cap">Data: {data["requests"]:,} contracts requested from LSEG, {data.get("returned", 0):,} returned, '
                f'{data.get("never_listed", 0):,} never listed; {data["quote_rows"]:,} daily quotes, '
                f'{data["two_sided_rows"]:,} with both a bid and an ask.</div>'),
        panel("Reporting", say("honest")),
        "</div>",
        panel("Limitations", say("limitations")),
        *(part2_html(P2, say, panel) if P2 else []),
        f"<script>{script}</script>",
        "</div></body></html>",
    ]
    if missing:
        print(f"  prose sections still empty: {', '.join(missing)}", file=sys.stderr)
    return "\n".join(parts)


def part2_html(P2: dict, say, panel) -> list[str]:
    results, each_t, corr = part2_tables(P2)
    lo, hi = P2["windows"]["train"]["window"], P2["windows"]["test"]["window"]
    return [
        '<div class="part" id="part2"><span class="no">Part II</span><h2>Further exploration: the same wheel on '
        'five stocks</h2><span class="what">Not required by the assignment. The plain wheel of Part I, run on all five '
        'stocks the screen picked at once.</span></div>',
        '<div style="height:10px"></div>',
        panel("Motivation", say("part2-motivation") + '<div style="height:8px"></div>' + say("part2-setup")),
        '<div style="height:12px"></div>',
        panel("Results", results, "$500,000 to start in each period; each period starts from cash"),
        '<div class="grid g-1-1" id="p2-charts">',
        panel("Account value, 2024–25", '<div id="p2-train" style="height:320px"></div>', f"{lo[0]} to {lo[1]}", flush=True),
        panel("Account value, 2026", '<div id="p2-test" style="height:320px"></div>', f"{hi[0]} to {hi[1]}", flush=True),
        "</div>",
        panel("What it shows", say("part2-findings")),
        '<div style="height:12px"></div>',
        panel("Each stock's wheel", each_t + "<details><summary>Correlations of the five wheels' monthly returns"
              "</summary>" + corr + "</details>", "$100,000 each"),
        f"<script>{PART2_SCRIPT.replace('/*__P2__*/null', json.dumps(part2_data(P2), separators=(',', ':')))}</script>",
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", action="store_true", help="print the placeholders and exit")
    ap.add_argument("--draft", action="store_true", help="mark unwritten prose sections on the page")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    D = json.loads(DATA.read_text())
    P2 = json.loads(PORTFOLIO.read_text()) if PORTFOLIO.exists() else None
    f = {**facts(D), **(part2_facts(P2) if P2 else {})}
    if args.facts:
        for k, v in f.items():
            print(f"{{{{{k}}}}}  =  {v}")
        return 0
    html = render(D, load_prose(f), args.draft, P2)
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
