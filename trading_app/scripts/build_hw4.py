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

import pandas as pd
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
        "audit_markets": str(len(eval(re.search(r"SEEDS = (range\(\d+\))",
                                                (ROOT / "tests" / "wheel" / "test_audit.py").read_text()).group(1)))),
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

HELD, MISSED = '<span class="st st-good">✓ Held</span>', '<span class="st st-bad">✕ Did not hold</span>'


def verdict_table(D: dict) -> str:
    """The thesis's two reasons and its claim, each against the 2026 test."""
    t = D["periods"]["test"]["books"]
    w, h = t["wheel"], t["hold"]
    over, n, iv, rv = premium_split(D, "test")
    run = D["reversal"]["run"]
    rows = [
        ["Reason 1: options are priced above the moves that follow",
         "0.25-delta put's implied vol on entry against the vol realised to expiry",
         f"{over} of {n} cycles; {pct(100 * iv)} against {pct(100 * rv)} on average",
         HELD if over > n / 2 else MISSED],
        ["Reason 2: Mastercard's monthly moves reverse",
         "The screen's own rule: variance ratio below 1 and autocorrelation below 0, over 2024-01 to 2026-09",
         f"variance ratio {num(run['vr'])}, autocorrelation {num(run['rho'], 2)}",
         HELD if run["vr"] < 1 and run["rho"] < 0 else MISSED],
        ["The claim: a higher Sharpe ratio than holding Mastercard", "Sharpe ratio, monthly, T-bill as risk-free",
         f"{num(w['sharpe'])} against {num(h['sharpe'])}", HELD if w["sharpe"] > h["sharpe"] else MISSED],
    ]
    return table(["", "Measured by", "2026 test", ""], rows, left=3, cls="verdict")


def setting_verdict(D: dict) -> str:
    """The registered test of the setting choice, beside the 18-setting table it belongs to."""
    t, S = D["periods"]["test"]["books"], D["selection"]
    return (f'<div class="cap">Registered test of the choice: the pick beats the default setting\'s Sharpe ratio in 2026, '
            f'at mid and at the bid. {num(t["wheel"]["sharpe"])} against {num(t["baseline"]["sharpe"])} at mid, '
            f'{num(t["wheel_bid"]["sharpe"])} against {num(t["baseline_bid"]["sharpe"])} at the bid; ranked '
            f'{S["rank"]} of 18. {HELD if S["helped"] else MISSED}</div>')


def fill_stress_table(D: dict) -> str:
    """The wheel at mid fills against the same wheel selling every option at the bid, both periods."""
    rows = []
    for label, key in (("Mid fills (as reported)", "wheel"), ("Every option sold at the bid", "wheel_bid")):
        row = [escape(label)]
        for n in ("test", "train"):
            b = D["periods"][n]["books"][key]
            row += [pct(b["return_pct"], sign=True), num(b["sharpe"])]
        rows.append(row)
    gap = ["Cost of selling at the bid"]
    for n in ("test", "train"):
        m, b = D["periods"][n]["books"]["wheel"], D["periods"][n]["books"]["wheel_bid"]
        gap += [f'{b["return_pct"] - m["return_pct"]:+.1f} pts'.replace("-", "−"), num(b["sharpe"] - m["sharpe"])]
    rows.append(gap)
    return table(["", "2026 return", "Sharpe", "2024–25 return", "Sharpe"], rows, cls="small")


def screen_table(D: dict) -> str:
    rows, skipped = [], []
    for r in D["reversal"]["top"]:
        status = r["status"]
        why = ("closed above $950, which the option symbols cannot encode" if "950" in status
               else "its contracts were adjusted for something other than a split" if "adjusted" in status
               else "it stopped trading" if "stopped" in status else "")
        tag = ('<span class="st st-good">✓ Pick</span>' if status == "pick" else
               '<span class="st st-neutral">Follow-up</span>' if status == "follow-up" else
               '<span class="faint">Skipped</span>' if why else "")
        name = re.sub(r"\s+Ord$", "", str(r["name"]).title())
        if why:
            skipped.append(f"{escape(r['ric'].split('.')[0])}: {why}")
        row = [str(r["rank"]), f'<span class="code">{escape(r["ric"])}</span>', escape(name), num(r["vr"], 3),
               num(r["rho"], 2), pct(100 * r["vol"], 0), tag]
        rows.append(row + (["__hi__"] if r["ric"] == D["ric"] else []))
    note = (f'<div class="cap">Skipped, by rules fixed before the screen ran: {"; ".join(skipped)}.</div>'
            if skipped else "")
    return table(["#", "RIC", "Company", "Variance ratio", "Autocorr.", "Vol", ""], rows, left=3, cls="small screen") + note


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
    own = [[o["date"], f'<span class="code">{escape(o["occ"])}</span>', "put" if o["cp"] == "P" else "call", str(o["contracts"]),
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
        books = {k: {**{f: v.get(f) for f in keep}, "attribution": v.get("attribution")} for k, v in P["books"].items()}
        L = P["ledger"]
        out["periods"][name] = {
            "window": P["window"], "books": books,
            "nav": {k: [round(x, 2) for x in v] if k != "date" else v for k, v in P["nav"].items()},
            "ledger": [[r["date"], r["close"], r["cash"], r["shares"], r["basis"], r["short"], r["occ"], r["strike"],
                        r["expiry"], r["contracts"], r["mark"], r["option_mv"], r["nav"]] for r in L],
            "blotter": P["blotter"], "cycles": P["cycles"],
            "monthly": P["monthly"], "drawdown": P["drawdown"], "regt": P["regt"], "vol": P.get("vol"),
            # the window ends before its month does (2026 stops on 16 October)
            "partial_last_month": bool((pd.Timestamp(P["window"][1]) + pd.offsets.BDay(1)).month
                                       == pd.Timestamp(P["window"][1]).month)}
    out["locations"] = D["fills"]["locations"]
    out["iv_check"] = D["fills"].get("iv_check")
    out["scatter"] = D["fills"]["scatter"]
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
    const axis = {gridcolor: css2('--line-soft'), linecolor: css2('--axis'), zeroline: false,
                  tickfont: {family: css2('--mono'), color: css2('--muted'), size: 10.5}};
    Plotly.newPlot(id, [
      {x: d.date, y: d.cash, name: 'T-bills', mode: 'lines', line: {color: css2('--cash'), width: 1.5},
       hovertemplate: '%{y:$,.0f}<extra>T-bills</extra>'},
      {x: d.date, y: d.hold, name: 'Five stocks held', mode: 'lines', line: {color: css2('--hold'), width: 1.5},
       hovertemplate: '%{y:$,.0f}<extra>Five stocks held</extra>'},
      {x: d.date, y: d.wheel, name: 'Five wheels together', mode: 'lines', line: {color: css2('--wheel'), width: 2},
       hovertemplate: '%{y:$,.0f}<extra>Five wheels together</extra>'}],
      {paper_bgcolor: css2('--panel'), plot_bgcolor: css2('--panel'), font: {family: css2('--font'), color: css2('--muted'), size: 11},
       margin: {l: 74, r: 16, t: 30, b: 40}, hovermode: 'x unified',
       hoverlabel: {bgcolor: css2('--panel-hover'), bordercolor: css2('--axis'), font: {family: css2('--mono'), color: css2('--text'), size: 11.5}},
       legend: {orientation: 'h', x: 0, xanchor: 'left', y: 1.0, yanchor: 'bottom', font: {color: css2('--text-2'), size: 11.5}},
       xaxis: axis, yaxis: {...axis, tickformat: '$,.0f'}},
      {displayModeBar: false, responsive: true});
  });
}
"""


STYLE = """
  /* ---- design tokens: restyle the page here -------------------------------
     Terminal-style dark: crisp 1px rules, square corners, IBM Plex Sans for words
     and IBM Plex Mono for every figure. Series colours validated for this surface
     with the dataviz palette script (blue, orange, aqua pass every check). */
  :root{
    color-scheme:dark;
    --base:#0A0D11; --panel:#0F1318; --panel-hi:#151A21; --panel-hover:#1A2029;
    --line:#1F2630; --line-soft:#19202A; --axis:#2A323D;
    --text:#E6E9EE; --text-2:#9BA3AE; --muted:#7D8590; --faint:#5B636E;
    --accent:#3987e5; --accent-soft:rgba(57,135,229,.12);
    --wheel:#3987e5; --wheel-bid:#7FB0EE; --hold:#d95926; --cash:#7D8590; --aqua:#199e70;
    --up:#16C784; --down:#F0616D; --warn:#E8A93B;
    --mk-put:#E8A93B; --mk-call:#d55181;   /* trade markers, checked against the blue wheel line */
    --font:'IBM Plex Sans',ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif;
    --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
    --radius:2px;
  }
  *{box-sizing:border-box} html,body{margin:0;padding:0}
  body{background:var(--base);color:var(--text);font-family:var(--font);font-size:13.5px;line-height:1.6;
    -webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
  .wrap{max-width:1440px;margin:0 auto;padding:0 24px 64px}
  @media (max-width:640px){.wrap{padding:0 16px 48px}}
  a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
  ::selection{background:var(--accent-soft)}
  .num,.code{font-family:var(--mono)}
  .code{font-size:11.5px;color:var(--text-2)}

  /* ---- site header: a terminal tab bar ------------------------------------- */
  .site{position:sticky;top:0;z-index:30;display:flex;align-items:stretch;gap:28px;flex-wrap:wrap;
    margin:0 -24px;padding:0 24px;min-height:50px;background:var(--base);border-bottom:1px solid var(--line)}
  .site .brand{display:flex;align-items:center;gap:10px;color:var(--text);text-decoration:none}
  .site .brand:hover{text-decoration:none}
  .site .brand .sym{font:600 12px/1 var(--mono);color:var(--accent);border:1px solid var(--accent);padding:5px 7px;
    border-radius:var(--radius);letter-spacing:.04em}
  .site .brand .co{font-weight:600;font-size:14px}
  .site .brand .name{color:var(--muted);font-size:12.5px;padding-left:10px;border-left:1px solid var(--line)}
  .pages{display:flex;align-items:stretch;gap:22px}
  .pages a{display:flex;align-items:center;color:var(--muted);font-size:12px;font-weight:500;letter-spacing:.06em;
    text-transform:uppercase;border-bottom:2px solid transparent;margin-bottom:-1px;white-space:nowrap}
  .pages a:hover{color:var(--text);text-decoration:none}
  .pages a.on{color:var(--text);border-bottom-color:var(--accent)}
  .site .right{margin-left:auto;display:flex;gap:8px;align-items:center}
  .site .right a{font-size:11.5px;font-weight:500;letter-spacing:.05em;text-transform:uppercase;padding:6px 10px;
    border-radius:var(--radius);white-space:nowrap;border:1px solid var(--line);color:var(--text-2)}
  .site .right a:hover{text-decoration:none;color:var(--text);border-color:var(--axis);background:var(--panel)}
  .site .right a.hot{color:var(--accent);border-color:rgba(57,135,229,.55)}
  .site .right a.hot:hover{background:var(--accent-soft);color:#fff}
  @media (max-width:900px){.site{position:static;margin:0 -16px;padding:8px 16px;gap:10px 22px}
    .site .right{margin-left:0} .pages{flex-wrap:wrap;gap:0 18px;max-width:100%} .pages a{padding:8px 0;white-space:normal}}

  /* ---- page heading and the facts strip ----------------------------------- */
  .part{margin:22px 0 12px;scroll-margin-top:66px}
  .part .no{display:block;font-size:11px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
  .part h2{margin:6px 0 4px;font-size:22px;line-height:1.25;font-weight:600;letter-spacing:-.005em}
  .part .what{display:block;color:var(--text-2);font-size:13.5px;max-width:90ch}
  .meta{display:flex;flex-wrap:wrap;margin:14px 0 4px;border:1px solid var(--line);border-radius:var(--radius);
    background:var(--panel)}
  .meta .chip{display:flex;flex-direction:column;gap:3px;padding:9px 16px;border-right:1px solid var(--line)}
  .meta .chip i{font-style:normal;font-size:10.5px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
  .meta .chip b{font:500 12.5px/1.3 var(--mono);color:var(--text)}

  /* ---- layout and panels -------------------------------------------------- */
  .grid{display:grid;gap:12px;margin:12px 0}
  .g-2-1{grid-template-columns:minmax(0,2fr) minmax(0,1fr)}
  .g-1-1{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}
  @media (max-width:1300px){.g-2-1,.g-1-1{grid-template-columns:minmax(0,1fr)}}
  .g-3{display:grid;gap:12px;grid-template-columns:repeat(3,minmax(0,1fr))}
  @media (max-width:800px){.g-3{grid-template-columns:minmax(0,1fr)}}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);min-width:0;
    display:flex;flex-direction:column;scroll-margin-top:66px}
  .panel > .ph{display:flex;align-items:center;gap:10px;min-height:36px;padding:0 14px;border-bottom:1px solid var(--line);
    font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--text-2)}
  .panel > .ph .tag{margin-left:auto;font:400 11px var(--mono);letter-spacing:0;text-transform:none;color:var(--muted);text-align:right}
  .panel > .pb{padding:14px;min-width:0}
  .panel > .pb.flush{padding:4px 6px 6px}
  .prose{color:var(--text-2);line-height:1.7;max-width:none}
  .prose strong{color:var(--text);font-weight:600}
  .prose p{margin:0 0 10px} .prose p:last-child{margin-bottom:0}
  .prose ul{padding-left:16px;margin:0} .prose li{margin-bottom:8px} .prose li::marker{color:var(--faint)}
  .prose.lead{font-size:14.5px;color:var(--text);line-height:1.7}
  .sub{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:600;margin:18px 0 8px}
  .cap{font-size:12.5px;color:var(--muted);margin-top:10px;line-height:1.55}
  .todo{border:1px dashed var(--warn);color:var(--warn);padding:8px 12px;font-size:12.5px}

  /* ---- period switch and the KPI strip ------------------------------------ */
  .switch{display:inline-flex;margin:18px 0 0;border:1px solid var(--line);border-radius:var(--radius);
    background:var(--panel);scroll-margin-top:66px;max-width:100%;overflow-x:auto}
  .switch button{font:inherit;font-size:12px;font-weight:500;letter-spacing:.04em;color:var(--muted);background:none;border:none;
    border-right:1px solid var(--line);padding:8px 14px;cursor:pointer;white-space:nowrap}
  .switch button:last-child{border-right:none}
  .switch button:hover{color:var(--text)}
  .switch button.on{color:var(--text);background:var(--panel-hover);box-shadow:inset 0 -2px 0 var(--accent)}
  .kpis{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));margin:12px 0;border:1px solid var(--line);
    border-radius:var(--radius);background:var(--panel)}
  @media (max-width:1280px){.kpis{grid-template-columns:repeat(4,minmax(0,1fr))}}
  @media (max-width:640px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
  .kpi{padding:11px 14px 12px;min-width:0;position:relative;border-right:1px solid var(--line);border-bottom:1px solid var(--line);
    margin:0 -1px -1px 0}
  .kpi .k{font-size:10.5px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);
    padding-right:20px;line-height:1.35}
  .kpi .v{font:500 20px/1.2 var(--mono);letter-spacing:-.02em;margin-top:7px}
  .kpi .d{font:400 11.5px/1.4 var(--mono);color:var(--muted);margin-top:5px}
  .kpi .tip{position:absolute;top:10px;right:10px;margin:0}
  .pos{color:var(--up)} .neg{color:var(--down)}

  /* ---- tables ------------------------------------------------------------- */
  .tbl-wrap{overflow-x:auto;max-width:100%}
  table.tbl{border-collapse:collapse;width:100%;font-size:12.5px}
  table.tbl th,table.tbl td{padding:7px 10px;border-bottom:1px solid var(--line-soft);text-align:right;white-space:nowrap}
  table.tbl th{color:var(--muted);font-weight:600;font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;
    background:var(--panel-hi);border-bottom:1px solid var(--line)}
  table.tbl td{color:var(--text);font-family:var(--mono);font-size:12px}
  table.tbl td.l{font-family:var(--font);font-size:12.5px}
  table.tbl td.l.num{font-family:var(--mono);font-size:12px}
  table.tbl .l{text-align:left}
  table.tbl tbody tr:hover td{background:var(--panel-hi)}
  table.tbl tbody tr:last-child td{border-bottom:none}
  table.tbl tr.hi td{background:var(--accent-soft)} table.tbl tr.hi td:first-child{box-shadow:inset 2px 0 0 var(--accent)}
  table.tbl.verdict td{white-space:normal;vertical-align:middle;padding-top:9px;padding-bottom:9px}
  table.tbl.verdict td:first-child{color:var(--text);font-weight:500;width:24%}
  table.tbl.verdict td:nth-child(2){color:var(--text-2)}
  table.tbl.verdict td:nth-child(3){font-family:var(--mono);font-size:12px}
  table.tbl.small td,table.tbl.small th{padding:6px 9px}
  table.tbl.screen th,table.tbl.screen td{padding:6px 8px} table.tbl.screen td:last-child{text-align:left}
  table.tbl tbody tr.click{cursor:pointer}
  table.tbl tbody tr.sel td{background:var(--accent-soft)}
  .note{display:block;min-width:280px;white-space:normal;color:var(--muted);font:400 12px/1.45 var(--font)}
  .st{display:inline-flex;align-items:center;gap:5px;padding:2px 7px;border-radius:var(--radius);font:600 10.5px/1.5 var(--font);
    letter-spacing:.07em;text-transform:uppercase;white-space:nowrap;border:1px solid transparent}
  .st-good{color:var(--up);background:rgba(22,199,132,.08);border-color:rgba(22,199,132,.35)}
  .st-bad{color:var(--down);background:rgba(240,97,109,.08);border-color:rgba(240,97,109,.35)}
  .st-neutral{color:var(--warn);background:rgba(232,169,59,.08);border-color:rgba(232,169,59,.35)}
  .faint{color:var(--muted)}
  .ev-open{color:var(--accent)} .ev-close{color:var(--muted)} .ev-assign{color:var(--warn)} .ev-stock{color:var(--up)}
  .ev-div{color:var(--text-2)}

  /* ---- rules, tabs, details ----------------------------------------------- */
  dl.rules{margin:0;display:grid;grid-template-columns:150px minmax(0,1fr);font-size:12.5px}
  dl.rules dt,dl.rules dd{padding:8px 0;border-bottom:1px solid var(--line-soft)}
  dl.rules dt{color:var(--muted);font-size:10.5px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;padding-top:10px;padding-right:14px}
  dl.rules dd{margin:0;color:var(--text-2);line-height:1.55}
  @media (max-width:640px){dl.rules{grid-template-columns:minmax(0,1fr)} dl.rules dt{border-bottom:none;padding-bottom:0}}
  .tabs{display:flex;gap:22px;flex-wrap:wrap;padding:0 14px;border-bottom:1px solid var(--line)}
  .tabs button{font:inherit;font-size:11.5px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);
    background:none;border:none;border-bottom:2px solid transparent;padding:11px 0 10px;cursor:pointer;margin-bottom:-1px}
  .tabs button:hover{color:var(--text)}
  .tabs button.on{color:var(--text);border-bottom-color:var(--accent)}
  .pane{display:none;padding:12px 14px 14px} .pane.on{display:block}
  .pane .tbl-wrap{max-height:560px;overflow-y:auto;border:1px solid var(--line)}
  .pane table.tbl thead th{position:sticky;top:0;z-index:1}
  details{margin-top:12px}
  details summary{cursor:pointer;color:var(--text-2);font-size:12.5px;font-weight:500;padding:6px 0;list-style:none}
  details summary::before{content:"+";display:inline-block;width:16px;font-family:var(--mono);color:var(--accent)}
  details[open] summary::before{content:"−"}
  details summary::-webkit-details-marker{display:none}

  /* ---- the portfolio on any date ----------------------------------------- */
  .pf-controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  .pf-controls input[type=range]{flex:1 1 260px;accent-color:var(--accent)}
  .pf-controls input[type=date],.pf-controls button{font:500 12px var(--font);background:var(--panel-hi);color:var(--text);
    border:1px solid var(--line);border-radius:var(--radius);padding:6px 10px;cursor:pointer;color-scheme:dark}
  .pf-controls input[type=date]{font-family:var(--mono)}
  .pf-controls button:hover{border-color:var(--axis);background:var(--panel-hover)}
  .pf{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));margin-top:12px;border:1px solid var(--line);border-radius:var(--radius)}
  @media (max-width:1000px){.pf{grid-template-columns:repeat(2,minmax(0,1fr))}}
  @media (max-width:520px){.pf{grid-template-columns:minmax(0,1fr)}}
  .pf .cell{padding:11px 14px;min-width:0;border-right:1px solid var(--line);border-bottom:1px solid var(--line);margin:0 -1px -1px 0}
  .pf .cell .k{font-size:10.5px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
  .pf .cell .v{font:500 18px/1.25 var(--mono);letter-spacing:-.02em;margin-top:6px}
  .pf .cell .d{font:400 11.5px/1.6 var(--mono);color:var(--muted);margin-top:4px}
  .pf-eq{font:400 12px/1.6 var(--mono);color:var(--muted);margin-top:12px;overflow-wrap:anywhere}
  .pf-eq b{color:var(--text);font-weight:500}
  .pf-events{margin-top:10px;font:400 12px/1.6 var(--mono);color:var(--text-2)}
  .pf-events div{padding:5px 0;border-top:1px solid var(--line-soft)}
  .pf-events .sub{border-top:none;font-family:var(--font)}

  /* ---- next / previous page ----------------------------------------------- */
  a.next{display:flex;align-items:center;margin:16px 0 0;padding:13px 16px;border:1px solid var(--line);
    border-radius:var(--radius);background:var(--panel);color:var(--text-2);font-size:12px;font-weight:600;
    letter-spacing:.07em;text-transform:uppercase}
  a.next:hover{text-decoration:none;color:var(--text);border-color:var(--accent)}

  /* ---- hover notes -------------------------------------------------------- */
  .fv{display:none} .fv.on{display:block}
  .tip{position:relative;display:inline-flex;align-items:center;justify-content:center;width:15px;height:15px;
    margin-left:6px;border:1px solid var(--axis);border-radius:50%;font:600 9.5px/1 var(--mono);color:var(--muted);
    cursor:help;text-transform:none;letter-spacing:0;flex:none}
  .tip:hover,.tip:focus{color:var(--text);border-color:var(--accent);outline:none}
  .tip .bubble{display:none;position:absolute;top:calc(100% + 8px);right:-10px;z-index:40;width:290px;max-width:74vw;
    padding:10px 12px;background:var(--panel-hover);border:1px solid var(--axis);border-radius:var(--radius);
    box-shadow:0 12px 32px rgba(0,0,0,.6);color:var(--text-2);font:400 12.5px/1.55 var(--font);text-align:left;
    white-space:normal;letter-spacing:0;text-transform:none}
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

const LABELS = {wheel: 'Wheel', wheel_bid: 'Wheel (fills at the bid)', hold: 'Buy and hold MA', cash: 'T-bills',
                putw: 'Cboe PutWrite index'};
const COLORS = {wheel: '--wheel', wheel_bid: '--wheel-bid', hold: '--hold', cash: '--cash', putw: '--aqua'};

function eventName(e) {
  if (e.cp && e.side === 'SELL') return ['Sell to open: ' + (e.cp === 'P' ? 'cash-secured put' : 'covered call'), 'ev-open'];
  if (e.side === 'EXPIRE') return ['Close: ' + (e.kind === 'put' ? 'put' : 'call') + ' expired', 'ev-close'];
  if (e.side === 'ASSIGN') return [e.kind === 'put' ? 'Assigned: put' : 'Called away: call', 'ev-assign'];
  if (e.kind === 'stock' && e.side === 'BUY') return ['Buy shares at the strike', 'ev-stock'];
  if (e.kind === 'stock' && e.side === 'SELL') return ['Deliver shares at the strike', 'ev-stock'];
  if (e.side === 'DIVIDEND') return ['Dividend', 'ev-div'];
  return [e.side, ''];
}

function bench() {
  const B = D.periods[period].books;
  const rows = [['wheel', 'Wheel'], ['putw', 'Cboe PutWrite'],
                ['hold', 'Buy and hold MA'], ['cash', 'T-bills']].filter(([k]) => B[k]);
  $('bench').innerHTML = '<div class="tbl-wrap"><table class="tbl small"><thead><tr>' +
    ['', 'Return', 'Sharpe', 'Max DD', 'Vol'].map((h, i) => `<th class="${i ? '' : 'l'}">${h}</th>`).join('') +
    '</tr></thead><tbody>' + rows.map(([k, label]) => `<tr${k === 'wheel' ? ' class="hi"' : ''}><td class="l">${label}</td>
      <td class="${cls(B[k].return_pct)}">${pct(B[k].return_pct, 1, true)}</td><td>${k === 'cash' ? '–' : num(B[k].sharpe)}</td>
      <td>${pct(B[k].max_drawdown_pct)}</td><td>${pct(B[k].vol_pct)}</td></tr>`).join('') + '</tbody></table></div>';
}

function kpis() {
  const B = D.periods[period].books, w = B.wheel, h = B.hold, c = B.cash, b = B.wheel_bid, u = B.putw;
  const vs = (f, g) => 'buy and hold ' + f(h) + (u ? ' · PutWrite ' + f(u) : '');
  const tiles = [
    ['Wheel return', pct(w.return_pct, 1, true), vs(x => pct(x.return_pct, 1, true)), w.return_pct],
    ['Sharpe ratio', num(w.sharpe), vs(x => num(x.sharpe)), w.sharpe],
    ['Max drawdown', pct(w.max_drawdown_pct), vs(x => pct(x.max_drawdown_pct)), w.max_drawdown_pct],
    ['Return at the bid', pct(b.return_pct, 1, true), 'stress test on fills', b.return_pct],
    ['Premium collected', money(w.attribution.premium), (w.puts_written + w.calls_written) + ' options sold', null],
    ['T-bill interest', money(w.attribution.interest), 'T-bills alone ' + pct(c.return_pct, 1, true), null],
    ['Puts sold / assigned', w.puts_written + ' / ' + w.assigned, 'in shares on ' + pct(w.share_days_pct, 0) + ' of sessions', null],
    ['Calls sold / called away', w.calls_written + ' / ' + w.called_away, w.held_no_call + ' cycles held with no call', null],
  ];
  $('kpis').innerHTML = tiles.map(([k, v, d, x]) =>
    `<div class="kpi"><div class="k">${esc(k)}${TIPS[k] || ''}</div><div class="v ${cls(x)}">${v}</div><div class="d">${esc(d)}</div></div>`).join('');
}

// One chart style: the page's surface and type, hairline grid, legend above the plot, a dark tooltip.
function layout(o = {}) {
  const axis = {gridcolor: css('--line-soft'), linecolor: css('--axis'), zeroline: false,
                tickfont: {family: css('--mono'), color: css('--muted'), size: 10.5},
                title: {font: {color: css('--muted'), size: 11}}};
  return {paper_bgcolor: css('--panel'), plot_bgcolor: css('--panel'),
    font: {family: css('--font'), color: css('--muted'), size: 11}, margin: {l: 70, r: 16, t: 30, b: 40},
    hoverlabel: {bgcolor: css('--panel-hover'), bordercolor: css('--axis'), font: {family: css('--mono'), color: css('--text'), size: 11.5}},
    legend: {orientation: 'h', x: 0, xanchor: 'left', y: 1.0, yanchor: 'bottom', font: {color: css('--text-2'), size: 11.5}},
    ...o, xaxis: {...axis, ...(o.xaxis || {})}, yaxis: {...axis, ...(o.yaxis || {})}};
}

// ▼ put sold, ▲ call sold, ✕ assigned or called away, each on the account-value line at its date
function tradeMarkers(blotter, dates, values) {
  const idx = new Map(dates.map((d, i) => [d, i]));
  return [['Put sold', e => e.cp === 'P' && e.side === 'SELL', 'triangle-down', '--mk-put', 12],
          ['Call sold', e => e.cp === 'C' && e.side === 'SELL', 'triangle-up', '--mk-call', 12],
          ['Assigned or called away', e => e.side === 'ASSIGN', 'x', '--text', 9]].map(([name, f, sym, color, size]) => {
    const ev = blotter.filter(f);
    return {x: ev.map(e => e.date), y: ev.map(e => values[idx.get(e.date)]), name, mode: 'markers',
      marker: {symbol: sym, size, color: css(color), line: {color: css(sym === 'x' ? '--panel' : '--base'), width: sym === 'x' ? 2 : 1.5}},
      text: ev.map(e => esc((e.kind === 'put' ? 'put ' : 'call ') + (e.occ || '') + ' at ' + num(e.price))),
      hovertemplate: '%{text}<extra>' + name + '</extra>'};
  });
}

function navChart() {
  const P = D.periods[period], N = P.nav;
  const traces = ['cash', 'hold', 'putw', 'wheel'].filter(k => N[k]).map(k => ({
    x: N.date, y: N[k], name: LABELS[k], mode: 'lines',
    line: {color: css(COLORS[k]), width: k === 'wheel' ? 2 : 1.5, dash: k === 'wheel_bid' ? 'dot' : 'solid'},
    hovertemplate: '%{y:$,.0f}<extra>' + LABELS[k] + '</extra>'}));
  traces.push(...tradeMarkers(P.blotter, N.date, N.wheel));
  Plotly.react('plot-nav', traces, layout({hovermode: 'x unified', yaxis: {tickformat: '$,.0f'}}),
    {displayModeBar: false, responsive: true});
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
       <div class="d"><span class="code">${esc(occ)}</span><br>strike ${money(strike, 2)}, expires ${esc(expiry)}<br>
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
      .map((h, i) => `<th class="${i < 4 || i === 8 ? 'l' : ''}">${h}</th>`).join('') + '</tr></thead><tbody>' +
    P.blotter.map(e => {
      const [name, c] = eventName(e);
      const qty = e.cp ? `${e.contracts} contract${e.contracts > 1 ? 's' : ''} (${e.qty} sh)` : `${e.qty.toLocaleString()} sh`;
      return `<tr><td class="l num">${esc(e.date)}</td><td class="l ${c}">${esc(name)}</td><td class="l">${e.occ ? '<span class="code">' + esc(e.occ) + '</span>' : 'MA (stock)'}</td>
        <td class="l">${esc(e.side)}</td><td>${qty}</td><td>${e.bid != null ? num(e.bid) + ' / ' + num(e.ask) : '–'}</td>
        <td>${e.price ? num(e.price, 3) : '–'}</td><td class="${cls(e.cash_delta)}">${money(e.cash_delta, 2, true)}</td>
        <td class="l"><span class="note">${esc(e.note)}</span></td></tr>`;
    }).join('') + '</tbody></table></div>';

  $('ledger').innerHTML = '<div class="tbl-wrap"><table class="tbl"><thead><tr>' +
    ['Date', 'Close', 'Cash', 'Shares', 'Cost basis', 'Open option', 'Contracts', 'Mark', 'Option owed', 'Account value']
      .map((h, i) => `<th class="${i === 0 || i === 5 ? 'l' : ''}">${h}</th>`).join('') + '</tr></thead><tbody>' +
    P.ledger.map(([date, close, cash, shares, basis, short, occ, strike, expiry, contracts, mark, optMv, nav], j) =>
      `<tr class="click" data-i="${j}"><td class="l num">${date}</td><td>${num(close)}</td><td>${money(cash)}</td><td>${shares}</td>
       <td>${shares ? num(basis) : '–'}</td><td class="l">${occ ? '<span class="code">' + esc(occ) + '</span>' : '–'}</td><td>${contracts ?? '–'}</td>
       <td>${mark != null ? num(mark, 3) : '–'}</td><td>${optMv ? money(-optMv) : '–'}</td><td>${money(nav)}</td></tr>`).join('') +
    '</tbody></table></div>';
  $('ledger').querySelectorAll('tr.click').forEach(tr => tr.addEventListener('click', () => {
    setDay(+tr.dataset.i); $('portfolio').scrollIntoView({behavior: 'smooth', block: 'start'});
  }));

  $('cycles').innerHTML = '<div class="tbl-wrap"><table class="tbl"><thead><tr>' +
    ['Entry', 'Expiry', 'Days', 'Sold', 'Contract', 'Strike', 'From spot', 'Delta', 'Implied vol', 'Realised vol',
     'Premium', 'Close at expiry', 'Outcome'].map((h, i) => `<th class="${i < 2 || i === 3 || i === 4 || i === 12 ? 'l' : ''}">${h}</th>`).join('') +
    '</tr></thead><tbody>' + P.cycles.map(c => `<tr><td class="l num">${c.entry}</td><td class="l num">${c.expiry}</td><td>${c.dte}</td>
      <td class="l">${c.cp === 'P' ? 'put' : c.cp === 'C' ? 'call' : '–'}</td><td class="l">${c.occ ? '<span class="code">' + esc(c.occ) + '</span>' : '–'}</td>
      <td>${c.strike != null ? num(c.strike) : '–'}</td><td>${c.otm_pct != null ? pct(c.otm_pct, 1, true) : '–'}</td>
      <td>${c.delta != null ? num(c.delta) : '–'}</td><td>${c.iv != null ? pct(100 * c.iv) : '–'}</td>
      <td>${c.realised_vol != null ? pct(100 * c.realised_vol) : '–'}</td><td>${c.premium != null ? money(c.premium) : '–'}</td>
      <td>${c.settle != null ? num(c.settle) : '–'}</td><td class="l">${esc(c.status || '')}</td></tr>`).join('') +
    '</tbody></table></div>';
}

function navTab() {
  const P = D.periods[period], rows = P.monthly;
  $('navtable').innerHTML = '<div class="tbl-wrap"><table class="tbl small"><thead><tr>' +
    ['Month', 'Month-end NAV', 'Return', 'T-bill return (month)', 'Excess return', 'Cumulative return', 'Drawdown at month-end']
      .map((h, i) => `<th class="${i === 0 ? 'l' : ''}">${h}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map(r => `<tr><td class="l num">${r.month}</td><td>${money(r.nav)}</td>
      <td class="${cls(r.ret)}">${pct(100 * r.ret, 2, true)}</td><td>${pct(100 * r.rf, 2)}</td>
      <td class="${cls(r.excess)}">${pct(100 * r.excess, 2, true)}</td><td>${pct(100 * r.cum_ret, 2, true)}</td>
      <td>${pct(100 * r.drawdown, 2)}</td></tr>`).join('') + '</tbody></table></div>';
  const ex = rows.map(r => r.excess), n = ex.length, mean = ex.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(ex.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1));
  $('sharpe-line').innerHTML = `Sharpe ratio = mean monthly excess return ${pct(100 * mean, 3, true)} ÷ its standard deviation `
    + `${pct(100 * sd, 3)} × √12 = <b class="num">${num(mean / sd * Math.sqrt(12))}</b>, over ${n} months.`
    + (D.periods[period].partial_last_month ? ' The last month is partial but is charged a full month of the T-bill.' : '');
}

function regtTab() {
  const R = D.periods[period].regt;
  $('regt-summary').innerHTML = '<div class="pf" style="margin-top:0">' + [
    ['Free cash, lowest', money(R.min_free_cash), 'on ' + R.min_free_date + (R.min_free_cash >= 0 ? ', never below zero' : '')],
    ['Cash set aside, largest', money(R.max_collateral), `for ${R.max_collateral_contracts} put contract${R.max_collateral_contracts > 1 ? 's' : ''} at ${money(R.max_collateral_strike)}, ${R.max_collateral_date}`],
    ['Sessions with a put / a call open', R.put_days + ' / ' + R.call_days, 'of ' + R.sessions + ' sessions'],
    ['Every call covered', R.always_covered ? '✓ Yes' : '✕ No', 'shares held on ' + R.share_days + ' sessions'],
  ].map(([k, v, d]) => `<div class="cell"><div class="k">${esc(k)}</div><div class="v">${v}</div><div class="d">${esc(d)}</div></div>`)
   .join('') + '</div>';
  $('regt-table').innerHTML = '<div class="tbl-wrap"><table class="tbl small"><thead><tr>' +
    ['Date', 'Cash', 'Set aside for the put', 'Free cash', 'Shares', 'Shares value', 'Call covered']
      .map((h, i) => `<th class="${i === 0 ? 'l' : ''}">${h}</th>`).join('') + '</tr></thead><tbody>' +
    R.rows.map(([d, cash, coll, free, sh, smv, need, cov]) => `<tr><td class="l num">${d}</td><td>${money(cash)}</td>
      <td>${coll ? money(coll) : '–'}</td><td>${money(free)}</td><td>${sh}</td><td>${sh ? money(smv) : '–'}</td>
      <td>${need ? (cov ? 'yes, ' + sh + ' of ' + need : 'NO') : '–'}</td></tr>`).join('') + '</tbody></table></div>';
}

function drawTabCharts() {
  if (typeof Plotly === 'undefined') return;
  const P = D.periods[period], cfg = {displayModeBar: false, responsive: true};
  if ($('plot-navc').offsetParent) {
    // a line, so its axis fits the data rather than starting at zero
    const L = P.ledger, x = L.map(r => r[0]);
    const nav = L.map(r => r[12]);
    Plotly.react('plot-navc', [
      {x, y: nav, name: 'Account value (NAV)', mode: 'lines', line: {color: css('--wheel'), width: 2},
       hovertemplate: '%{y:$,.0f}<extra>NAV</extra>'},
      ...tradeMarkers(P.blotter, x, nav)],
      layout({hovermode: 'x unified', yaxis: {tickformat: '$,.0f', autorange: true}}), cfg);
  }
  if ($('plot-hold').offsetParent) {
    // a breakdown, so it stays stacked from zero: the share of cash plus shares held in each
    const L = P.ledger, x = L.map(r => r[0]);
    const total = L.map(r => r[2] + r[3] * r[1]);
    Plotly.react('plot-hold', [
      {x, y: L.map((r, i) => 100 * r[2] / total[i]), name: 'Cash (incl. put collateral)', stackgroup: 'held', mode: 'none',
       fillcolor: 'rgba(125,133,144,0.45)', hovertemplate: '%{y:.0f}%<extra>cash</extra>'},
      {x, y: L.map((r, i) => 100 * r[3] * r[1] / total[i]), name: 'Mastercard shares', stackgroup: 'held', mode: 'none',
       fillcolor: 'rgba(25,158,112,0.65)', hovertemplate: '%{y:.0f}%<extra>shares</extra>'}],
      layout({hovermode: 'x unified', margin: {l: 70, r: 16, t: 30, b: 36}, yaxis: {ticksuffix: '%', range: [0, 100]}}), cfg);
  }
  if ($('plot-dd').offsetParent) {
    Plotly.react('plot-dd', [{x: P.drawdown.date, y: P.drawdown.dd.map(v => 100 * v), mode: 'lines', name: 'Drawdown',
      fill: 'tozeroy', line: {color: css('--wheel'), width: 1.5}, fillcolor: 'rgba(57,135,229,0.15)',
      hovertemplate: '%{y:.2f}%<extra>drawdown</extra>'}],
      layout({showlegend: false, hovermode: 'x unified', margin: {l: 56, r: 16, t: 10, b: 36}, yaxis: {ticksuffix: '%'}}), cfg);
  }
  if ($('plot-regt').offsetParent) {
    // one bar per cycle, on the close its option was sold (or, with nothing sold, its entry close)
    const byDate = new Map(P.regt.rows.map(r => [r[0], r]));
    const sold = new Map(P.blotter.filter(e => e.cp && e.side === 'SELL').map(e => [e.date, e]));
    const bars = P.cycles.map(c => byDate.get(c.entry)).filter(Boolean);
    const x = bars.map(r => r[0]);
    const seg = (name, y, color, what) => ({type: 'bar', name, x, y, marker: {color, line: {color: css('--panel'), width: 1}},
      hovertemplate: '%{y:$,.0f}<extra>' + what + '</extra>'});
    Plotly.react('plot-regt', [
      seg('Set aside for the put', bars.map(r => r[2]), css('--wheel'), 'set aside for the put'),
      seg('Shares held (value)', bars.map(r => r[5]), css('--aqua'), 'shares'),
      seg('Free cash', bars.map(r => r[3]), 'rgba(125,133,144,0.45)', 'free cash')],
      layout({barmode: 'stack', bargap: 0.35, hovermode: 'x unified', yaxis: {tickformat: '$,.0f'}}), cfg);
  }
}

// ---- volatility ------------------------------------------------------------
let smIndex = 0, surfView = 'heat';
function smileInterp(sm, m) {
  const xs = sm.m, ys = sm.iv;
  if (m < xs[0] || m > xs[xs.length - 1]) return null;
  for (let i = 1; i < xs.length; i++) if (xs[i] >= m) return ys[i - 1] + (ys[i] - ys[i - 1]) * (m - xs[i - 1]) / (xs[i] - xs[i - 1]);
  return ys[ys.length - 1];
}
function smileAt(i) {
  const V = D.periods[period].vol; if (!V) return;
  smIndex = Math.max(0, Math.min(V.smile.date.length - 1, i));
  const day = V.smile.date[smIndex], sm = V.smile.s[smIndex];
  $('sm-range').value = smIndex; $('sm-date').value = day;
  const atm = smileInterp(sm, 1.0), wing = smileInterp(sm, 0.9);
  $('sm-info').innerHTML = `${day} · expiry <b>${sm.expiry}</b> (${sm.dte} days) · MA close <b>${money(sm.spot, 2)}</b>`
    + (atm != null ? ` · at-the-money vol <b>${pct(100 * atm)}</b>` : '')
    + (wing != null && atm != null ? ` · 90% strike <b>${pct(100 * wing)}</b> (put skew ${num(100 * (wing - atm), 1)} pts)` : '');
  if (typeof Plotly === 'undefined' || !$('plot-smile').offsetParent) return;
  const isPut = sm.m.map(m => m < 1);
  const traces = [
    {x: sm.m.map(m => 100 * m), y: sm.iv.map(v => 100 * v), name: 'Smile', mode: 'lines', line: {color: css('--text-2'), width: 1.5},
     hoverinfo: 'skip', showlegend: false},
    {x: sm.m.filter((_, i) => isPut[i]).map(m => 100 * m), y: sm.iv.filter((_, i) => isPut[i]).map(v => 100 * v),
     name: 'Puts (below the close)', mode: 'markers', marker: {size: 8, color: css('--wheel')},
     hovertemplate: 'strike %{x:.1f}% of close<br>implied vol %{y:.1f}%<extra>put</extra>'},
    {x: sm.m.filter((_, i) => !isPut[i]).map(m => 100 * m), y: sm.iv.filter((_, i) => !isPut[i]).map(v => 100 * v),
     name: 'Calls (above the close)', mode: 'markers', marker: {size: 8, color: css('--aqua')},
     hovertemplate: 'strike %{x:.1f}% of close<br>implied vol %{y:.1f}%<extra>call</extra>'}];
  const row = D.periods[period].ledger.find(r => r[0] === day);
  if (row && row[5] && row[8] === sm.expiry) {
    const m = row[7] / sm.spot, iv = smileInterp(sm, m);
    if (iv != null) traces.push({x: [100 * m], y: [100 * iv], name: row[5] === 'P' ? "The wheel's short put" : "The wheel's covered call",
      mode: 'markers', marker: {symbol: row[5] === 'P' ? 'triangle-down' : 'triangle-up', size: 15,
      color: css(row[5] === 'P' ? '--mk-put' : '--mk-call'), line: {color: css('--base'), width: 1.5}},
      hovertemplate: 'strike ' + money(row[7], 0) + ' (%{x:.1f}% of close)<br>implied vol %{y:.1f}%<extra>the wheel</extra>'});
  }
  Plotly.react('plot-smile', traces, layout({hovermode: 'closest',
    shapes: [{type: 'line', x0: 100, x1: 100, y0: 0, y1: 1, yref: 'paper', line: {color: css('--axis'), width: 1}}],
    annotations: [{x: 100, y: 1, yref: 'paper', text: 'close', showarrow: false, yanchor: 'bottom',
                   font: {family: css('--mono'), size: 10.5, color: css('--muted')}}],
    xaxis: {title: {text: 'strike, % of the close'}, ticksuffix: '%'}, yaxis: {title: {text: 'implied vol'}, ticksuffix: '%'}}),
    {displayModeBar: false, responsive: true});
}
function smileJump(dir) {
  const V = D.periods[period].vol, cur = V.smile.date[smIndex];
  const sales = [...new Set(D.periods[period].blotter.filter(e => e.cp && e.side === 'SELL').map(e => e.date))].sort();
  const t = dir > 0 ? sales.find(d => d > cur) : [...sales].reverse().find(d => d < cur);
  if (t) { const j = V.smile.date.findIndex(d => d >= t); if (j >= 0) smileAt(j); }
}
function volCharts() {
  const P = D.periods[period], V = P.vol, cfg = {displayModeBar: false, responsive: true};
  if (!V || typeof Plotly === 'undefined') return;
  if ($('plot-smile').offsetParent) smileAt(smIndex);
  if ($('plot-surface').offsetParent && surfView === '3d') {
    const S = V.surface, z = S.grid.map((_, j) => S.iv.map(row => row[j] == null ? null : 100 * row[j]));
    const all = z.flat().filter(v => v != null).sort((a, b) => a - b);
    const q = f => all[Math.floor(f * (all.length - 1))];
    const idx = new Map(S.date.map((d, i) => [d, i]));
    const close = new Map(P.ledger.map(r => [r[0], r[1]]));
    const onSurface = e => {                                  // the sold strike, lifted onto that day's surface
      const i = idx.get(e.date); if (i == null) return null;
      const m = e.strike / close.get(e.date), g = S.grid, row = S.iv[i];
      const j = g.findIndex((v, k) => k > 0 && g[k - 1] <= m && m <= v);     // the grid cell the strike falls in
      if (j < 1 || row[j] == null || row[j - 1] == null) return null;
      return 100 * (row[j - 1] + (row[j] - row[j - 1]) * (m - g[j - 1]) / (g[j] - g[j - 1]));
    };
    const sold = P.blotter.filter(e => e.cp && e.side === 'SELL').map(e => ({e, z: onSurface(e)})).filter(o => o.z != null);
    const zLo = q(0.01), zHi = q(0.995) + 2;
    // each option sold as a pin from the floor up through the surface, with a dot where it crosses it
    const pins = (cp, name, color) => {
      const o = sold.filter(x => x.e.cp === cp), X = [], Y = [], Z = [];
      o.forEach(x => { const m = 100 * x.e.strike / close.get(x.e.date);
                       X.push(m, m, null); Y.push(x.e.date, x.e.date, null); Z.push(zLo, zHi, null); });
      return [
        {type: 'scatter3d', mode: 'lines', name, legendgroup: cp, x: X, y: Y, z: Z, hoverinfo: 'skip',
         line: {color: css(color), width: 6}},
        {type: 'scatter3d', mode: 'markers', name, legendgroup: cp, showlegend: false,
         x: o.map(x => 100 * x.e.strike / close.get(x.e.date)), y: o.map(x => x.e.date), z: o.map(x => x.z),
         marker: {size: 6, color: css(color), line: {color: css('--base'), width: 1}}, text: o.map(x => x.e.occ),
         hovertemplate: '%{text}<br>strike %{x:.1f}% of close<br>implied vol %{z:.1f}%<extra></extra>'}];
    };
    const ax = (title, extra = {}) => ({title: {text: title, font: {color: css('--muted'), size: 11}}, color: css('--muted'),
      gridcolor: css('--line'), zerolinecolor: css('--axis'), backgroundcolor: css('--panel'), showbackground: true,
      tickfont: {family: css('--mono'), size: 10, color: css('--muted')}, ...extra});
    Plotly.react('plot-surface', [
      {type: 'surface', x: S.grid.map(g => 100 * g), y: S.date, z: S.iv.map(row => row.map(v => v == null ? null : 100 * v)),
       cmin: q(0.03), cmax: q(0.97), connectgaps: false,
       colorscale: [[0, '#0d366b'], [0.35, '#256abf'], [0.7, '#6da7ec'], [1, '#cde2fb']], showscale: true,
       colorbar: {title: {text: 'implied vol', side: 'right'}, ticksuffix: '%', thickness: 10, outlinewidth: 0, len: 0.7,
                  tickfont: {family: css('--mono'), color: css('--muted'), size: 10}},
       contours: {z: {show: false}}, lighting: {ambient: 0.85, diffuse: 0.4, specular: 0.05},
       hovertemplate: '%{y}<br>strike %{x:.1f}% of close<br>implied vol %{z:.1f}%<extra></extra>'},
      ...pins('P', 'Puts sold', '--mk-put'), ...pins('C', 'Calls sold', '--mk-call')],
      {paper_bgcolor: css('--panel'), font: {family: css('--font'), color: css('--muted'), size: 11},
       margin: {l: 0, r: 0, t: 10, b: 0}, legend: {orientation: 'h', x: 0, y: 1.02, font: {color: css('--text-2')}},
       hoverlabel: {bgcolor: css('--panel-hover'), bordercolor: css('--axis'), font: {family: css('--mono'), color: css('--text'), size: 11.5}},
       scene: {xaxis: ax('strike, % of close', {ticksuffix: '%', autorange: 'reversed'}), yaxis: ax('date'),
               zaxis: ax('implied vol', {ticksuffix: '%', range: [zLo, zHi]}),
               aspectmode: 'manual', aspectratio: {x: 1.25, y: 1.9, z: 0.75},
               camera: {eye: {x: 1.75, y: -1.25, z: 0.85}, center: {x: 0, y: 0, z: -0.15}},
               domain: {x: [0, 1], y: [0, 1]}, bgcolor: css('--panel')}},
      cfg);
  } else if ($('plot-surface').offsetParent) {
    const S = V.surface, z = S.grid.map((_, j) => S.iv.map(row => row[j] == null ? null : 100 * row[j]));
    const all = z.flat().filter(v => v != null).sort((a, b) => a - b);
    const q = f => all[Math.floor(f * (all.length - 1))];
    const close = new Map(P.ledger.map(r => [r[0], r[1]]));
    const sold = P.blotter.filter(e => e.cp && e.side === 'SELL');
    const mk = (cp, name, sym, color) => {
      const ev = sold.filter(e => e.cp === cp);
      return {x: ev.map(e => e.date), y: ev.map(e => 100 * e.strike / close.get(e.date)), name, mode: 'markers', type: 'scatter',
        marker: {symbol: sym, size: 10, color: css(color), line: {color: css('--base'), width: 1.5}},
        text: ev.map(e => e.occ), hovertemplate: '%{text}<br>strike %{y:.1f}% of close<extra>' + name.toLowerCase() + '</extra>'};
    };
    Plotly.react('plot-surface', [
      {type: 'heatmap', x: S.date, y: S.grid.map(g => 100 * g), z, zmin: q(0.03), zmax: q(0.97),
       colorscale: [[0, '#0d366b'], [0.35, '#256abf'], [0.7, '#6da7ec'], [1, '#cde2fb']],
       colorbar: {title: {text: 'implied vol', side: 'right'}, ticksuffix: '%', thickness: 10, outlinewidth: 0,
                  tickfont: {family: css('--mono'), color: css('--muted'), size: 10}},
       hovertemplate: '%{x}<br>strike %{y:.1f}% of close<br>implied vol %{z:.1f}%<extra></extra>'},
      mk('P', 'Puts sold', 'triangle-down', '--mk-put'), mk('C', 'Calls sold', 'triangle-up', '--mk-call')],
      layout({hovermode: 'closest', margin: {l: 58, r: 16, t: 34, b: 40},
              yaxis: {title: {text: 'strike, % of the close'}, ticksuffix: '%'}}), cfg);
  }
  if ($('plot-vrp').offsetParent) {
    const c = P.cycles.filter(x => x.iv != null && x.realised_vol != null && x.cp);
    const over = c.filter(x => x.iv > x.realised_vol).length;
    const gap = c.reduce((a, x) => a + (x.iv - x.realised_vol), 0) / Math.max(c.length, 1);
    $('vrp-info').innerHTML = `Implied above realised on <b>${over} of ${c.length}</b> options sold; on average `
      + `<b>${num(100 * gap, 1)} vol points</b> higher.`;
    const seg = {x: [], y: []};
    c.forEach(x => { seg.x.push(x.entry, x.entry, null); seg.y.push(100 * x.iv, 100 * x.realised_vol, null); });
    Plotly.react('plot-vrp', [
      {x: seg.x, y: seg.y, mode: 'lines', line: {color: css('--axis'), width: 2}, hoverinfo: 'skip', showlegend: false},
      {x: c.map(x => x.entry), y: c.map(x => 100 * x.iv), name: 'Implied vol, when sold', mode: 'markers',
       marker: {size: 9, color: css('--wheel')}, text: c.map(x => x.occ || ''),
       hovertemplate: '%{text}<br>implied %{y:.1f}%<extra></extra>'},
      {x: c.map(x => x.entry), y: c.map(x => 100 * x.realised_vol), name: 'Realised vol, to expiry', mode: 'markers',
       marker: {size: 9, color: css('--text-2')}, hovertemplate: 'realised %{y:.1f}%<extra></extra>'}],
      layout({hovermode: 'closest', yaxis: {ticksuffix: '%', rangemode: 'tozero'}}), cfg);
  }
  if ($('plot-gsurf').offsetParent) greekCharts(V.greeks, cfg);
}

// the lecture's surfaces: one greek over strike/spot and days to expiry, held long, per share
const GK = {
  put_delta:  {title: 'PUT   Δ = N(d₁) − 1', z: 'Δ per share', cp: 'P', key: 'delta', fmt: '%{z:.3f}'},
  call_delta: {title: 'CALL   Δ = N(d₁)', z: 'Δ per share', cp: 'C', key: 'delta', fmt: '%{z:.3f}'},
  put_theta:  {title: 'PUT   Θ   ($ / day)', z: 'Θ  $ / day', cp: 'P', key: 'theta', fmt: '%{z:$.3f} per day'},
  call_theta: {title: 'CALL   Θ   ($ / day)', z: 'Θ  $ / day', cp: 'C', key: 'theta', fmt: '%{z:$.3f} per day'},
  gamma:      {title: 'Γ   call = put', z: 'Γ  per $1', cp: null, key: 'gamma', fmt: '%{z:.4f} per $1'},
  vega:       {title: 'ν   call = put   ($ / vol point)', z: 'ν  $ / vol point', cp: null, key: 'vega', fmt: '%{z:$.3f} per vol point'},
};
let gkView = 'put_delta';

function greekCharts(G, cfg) {
  const k = GK[gkView], Z = G.surfaces[gkView], flat = Z.flat();
  const lo = Math.min(...flat), hi = Math.max(...flat), lift = 0.015 * (hi - lo);   // keeps a path drawn above the surface it sits on
  const x0 = G.m[0], x1 = G.m[G.m.length - 1], y0 = G.dte[0], y1 = G.dte[G.dte.length - 1];
  const far = $('plot-gsurf').clientWidth < 700 ? 1.75 : 1;            // a phone sees the whole surface
  $('gk-info').innerHTML = `Held fixed, as in the lecture: Mastercard at <b>${money(G.spot, 2)}</b> (median close), implied vol `
    + `<b>${pct(100 * G.sigma)}</b> (median of the options sold), rate <b>${pct(100 * G.rate)}</b>, no dividend yield.`;
  // each option sold, traced from its sale (dot) to the session before expiry
  const traced = (cp, name, color) => {
    const o = G.sold.filter(s => s.cp === cp), X = [], Y = [], H = [], T = [];
    o.forEach(s => { s.m.forEach((m, i) => { X.push(m); Y.push(s.dte[i]); H.push(s[k.key][i] + lift);
                       T.push(`${s.occ}<br>${s.date[i]} · close ${money(s.close[i], 2)}<br>${s.dte[i]} days left · K/S ${num(m, 3)}`
                              + `<br>${s.status}`); });
                     X.push(null); Y.push(null); H.push(null); T.push(null); });
    return [{type: 'scatter3d', mode: 'lines', name, legendgroup: cp, x: X, y: Y, z: H, text: T,
             line: {color: css(color), width: 5}, hovertemplate: '%{text}<br>' + k.fmt + '<extra></extra>'},
            {type: 'scatter3d', mode: 'markers', name, legendgroup: cp, showlegend: false,
             x: o.map(s => s.m[0]), y: o.map(s => s.dte[0]), z: o.map(s => s[k.key][0] + lift), text: o.map(s => s.occ + ' sold ' + s.date[0]),
             marker: {size: 5, color: css(color), line: {color: css('--base'), width: 1}}, hovertemplate: '%{text}<br>' + k.fmt + '<extra></extra>'}];
  };
  const ax = (title, extra = {}) => ({title: {text: title, font: {color: css('--muted'), size: 11}}, color: css('--muted'),
    gridcolor: css('--line'), zerolinecolor: css('--axis'), backgroundcolor: css('--panel'), showbackground: true,
    tickfont: {family: css('--mono'), size: 10, color: css('--muted')}, ...extra});
  Plotly.react('plot-gsurf', [
    {type: 'surface', x: G.m, y: G.dte, z: Z, showscale: false,
     colorscale: [[0, '#0d366b'], [0.35, '#256abf'], [0.7, '#6da7ec'], [1, '#cde2fb']],
     contours: {z: {show: false}}, lighting: {ambient: 0.85, diffuse: 0.4, specular: 0.05},
     hovertemplate: 'K/S %{x:.2f}<br>%{y} days to expiry<br>' + k.fmt + '<extra></extra>'},
    {type: 'scatter3d', mode: 'lines', x: [x0, x1, x1, x0, x0], y: [y0, y0, y1, y1, y0], z: [0, 0, 0, 0, 0],
     line: {color: css('--text'), width: 4}, hoverinfo: 'skip', showlegend: false},
    ...(k.cp !== 'C' ? traced('P', 'Puts sold', '--mk-put') : []), ...(k.cp !== 'P' ? traced('C', 'Calls sold', '--mk-call') : [])],
    {paper_bgcolor: css('--panel'), font: {family: css('--font'), color: css('--muted'), size: 11},
     title: {text: k.title, x: 0.5, y: 0.97, font: {family: css('--font'), color: css('--text'), size: 14}},
     margin: {l: 0, r: 0, t: 34, b: 0}, legend: {orientation: 'h', x: 0, y: 1.0, font: {color: css('--text-2')}},
     hoverlabel: {bgcolor: css('--panel-hover'), bordercolor: css('--axis'), font: {family: css('--mono'), color: css('--text'), size: 11.5}},
     scene: {xaxis: ax('moneyness  K / S', {range: [x0, x1]}), yaxis: ax('expiry (DTE)', {range: [0, y1]}),
             zaxis: ax(k.z, {range: [Math.min(lo, 0) - 4 * lift, Math.max(hi, 0) + 4 * lift]}),
             aspectmode: 'manual', aspectratio: {x: 1.35, y: 1.35, z: 0.8},
             camera: {eye: {x: 1.2 * far, y: -1.5 * far, z: 0.85 * far}, center: {x: 0, y: 0, z: -0.22}}, bgcolor: css('--panel')}},
    cfg);
  // three clocks, one at-the-money option: gamma and the vol piece of theta blow up into expiry, vega dies
  const C = G.clocks, band = G.sold_days[1];
  const clock = (id, y, title, fmt) => Plotly.react(id, [{x: C.dte, y, mode: 'lines', line: {color: css('--wheel'), width: 2},
      hovertemplate: '%{x} days<br>' + fmt + '<extra></extra>'}],
    layout({showlegend: false, hovermode: 'closest', margin: {l: 58, r: 12, t: 30, b: 38},
      title: {text: title, x: 0.03, y: 0.96, font: {family: css('--font'), color: css('--text'), size: 12.5}},
      xaxis: {title: {text: 'days to expiry'}, range: [0, 360]}, yaxis: {rangemode: 'tozero'},
      shapes: [{type: 'rect', x0: 0, x1: band, xref: 'x', y0: 0, y1: 1, yref: 'paper', layer: 'below',
                fillcolor: css('--accent-soft'), line: {width: 0}}],
      annotations: [{x: band, xref: 'x', y: 1, yref: 'paper', xanchor: 'left', yanchor: 'top', showarrow: false,
                     text: ' the wheel', font: {family: css('--mono'), size: 10, color: css('--text-2')}}]}), cfg);
  clock('plot-gclk-gamma', C.gamma, 'Γ   ATM', '%{y:.4f} per $1');
  clock('plot-gclk-theta', C.theta_vol.map(v => -v), '|Θ_vol|   ATM   ($ / day)', '%{y:$.3f} per day');
  clock('plot-gclk-vega', C.vega, 'ν   ATM   ($ / vol point)', '%{y:$.3f} per vol point');
}

function fillHist() {
  const H = D.locations, n = H.labels.length, inside = H.labels.map((_, i) => i > 0 && i < n - 1);
  const bar = (keep, name, color) => ({type: 'bar', name, x: H.labels.map((l, i) => keep[i] ? l : null).filter(v => v !== null),
    y: H.pct.filter((_, i) => keep[i]), marker: {color, line: {color: css('--panel'), width: 1}},
    hovertemplate: '%{y:.1f}% of trades<extra>%{x}</extra>'});
  const at = (pos, text) => ({x: pos, xref: 'x', y: 1.02, yref: 'paper', text, showarrow: false, yanchor: 'bottom',
    font: {family: css('--mono'), size: 11, color: css('--text-2')}});
  const rule = pos => ({type: 'line', x0: pos, x1: pos, xref: 'x', y0: 0, y1: 1, yref: 'paper', layer: 'above',
    line: {color: css('--muted'), width: 1}});
  Plotly.newPlot('plot-fill', [
    bar(inside, 'Inside the quote', css('--wheel')),
    bar(inside.map(v => !v), 'Outside the quote', 'rgba(125,133,144,0.55)')],
    layout({bargap: 0.18, hovermode: 'closest', margin: {l: 52, r: 16, t: 54, b: 70},
            legend: {orientation: 'h', x: 0, xanchor: 'left', y: 1.14, yanchor: 'bottom'},
            xaxis: {type: 'category', categoryorder: 'array', categoryarray: H.labels, tickangle: -35},
            yaxis: {ticksuffix: '%', rangemode: 'tozero'},
            shapes: [0.5, 5.5, 10.5].map(rule), annotations: [at(0.5, 'bid'), at(5.5, 'mid'), at(10.5, 'ask')]}),
    {displayModeBar: false, responsive: true});
}

function fillScatter() {
  const S = D.scatter, put = S.cp.map(c => c === 'P');
  const pick = (arr, want) => arr.filter((_, i) => put[i] === want);
  const z = S.zoom, h = S.half_spread;
  const guideColor = k => css(k > 0 ? '--mk-call' : '--mk-put');        // ask pink, bid amber
  const guide = (k, name) => ({x: [0, z * 1.05], y: [0, z * 1.05 * (1 + k * h)], name, mode: 'lines', hoverinfo: 'skip',
    line: {color: guideColor(k), width: 1.5}});
  const pts = (want, name, color) => ({x: pick(S.mid, want), y: pick(S.trade, want), name, mode: 'markers',
    marker: {size: 5, color, opacity: 0.55}, text: pick(S.date, want),
    hovertemplate: 'mid %{x:$.2f} → trade %{y:$.2f}<br>%{text}<extra>' + name.toLowerCase() + '</extra>'});
  Plotly.newPlot('plot-scatter', [
    pts(true, 'Puts', css('--wheel')), pts(false, 'Calls', css('--aqua')),
    {x: [0, z * 1.05], y: [0, z * 1.05], name: 'Trade = mid', mode: 'lines', line: {color: css('--text-2'), width: 1}, hoverinfo: 'skip'},
    guide(-1, 'Typical bid'), guide(1, 'Typical ask')],
    layout({hovermode: 'closest', margin: {l: 58, r: 16, t: 34, b: 46},
            annotations: [[1, 'typical ask'], [-1, 'typical bid']].map(([k, text]) => ({x: z * 0.8, y: z * 0.8 * (1 + k * h),
              text, showarrow: true, arrowhead: 0, arrowcolor: guideColor(k), ax: k > 0 ? -46 : 46, ay: k > 0 ? -30 : 30,
              bgcolor: css('--panel'), bordercolor: guideColor(k), borderpad: 3,
              font: {family: css('--mono'), size: 10.5, color: guideColor(k)}})),
            xaxis: {title: {text: 'closing mid'}, tickformat: '$,.0f', range: [0, z]},
            yaxis: {title: {text: 'last trade of the day'}, tickformat: '$,.0f', range: [0, z]}}),
    {displayModeBar: false, responsive: true});
}

document.querySelectorAll('.fillview button').forEach(btn => btn.addEventListener('click', () => {
  document.querySelectorAll('.fillview button').forEach(b => b.classList.toggle('on', b === btn));
  document.querySelectorAll('.fv').forEach(v => v.classList.toggle('on', v.dataset.view === btn.dataset.view));
  if (typeof Plotly === 'undefined') return;
  if (btn.dataset.view === 'scatter' && !$('plot-scatter').data) fillScatter();   // drawn once it has a width
  Plotly.Plots.resize($(btn.dataset.view === 'scatter' ? 'plot-scatter' : 'plot-fill'));
}));

function show(p) {
  period = p;
  document.querySelectorAll('.switch button[data-period]').forEach(b => b.classList.toggle('on', b.dataset.period === p));
  document.querySelectorAll('[data-window]').forEach(el => el.textContent = D.periods[p].window.join(' to '));
  const L = D.periods[p].ledger;
  $('pf-range').max = L.length - 1; $('pf-date').min = L[0][0]; $('pf-date').max = L[L.length - 1][0];
  kpis(); bench(); tables(); navTab(); regtTab();
  const V = D.periods[p].vol;
  if (V) { $('sm-range').max = V.smile.date.length - 1; $('sm-date').min = V.smile.date[0];
           $('sm-date').max = V.smile.date[V.smile.date.length - 1]; smIndex = V.smile.date.length - 1;
           smileJump(-1); smileAt(smIndex); }
  if (typeof Plotly !== 'undefined') { navChart(); drawTabCharts(); volCharts(); }
  setDay(L.length - 1);
}

document.querySelectorAll('.switch button[data-period]').forEach(b => b.addEventListener('click', () => show(b.dataset.period)));
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
  drawTabCharts(); volCharts();       // a chart drawn in a hidden tab has no width, so tabs draw when shown
}));
$('sm-range').addEventListener('input', e => smileAt(+e.target.value));
document.querySelectorAll('.surfview button').forEach(btn => btn.addEventListener('click', () => {
  surfView = btn.dataset.sv;
  document.querySelectorAll('.surfview button').forEach(b => b.classList.toggle('on', b === btn));
  $('surf-hint').style.display = surfView === '3d' ? '' : 'none';
  if (typeof Plotly !== 'undefined') { Plotly.purge('plot-surface'); volCharts(); }
}));
document.querySelectorAll('.gkview button').forEach(btn => btn.addEventListener('click', () => {
  gkView = btn.dataset.gk;
  document.querySelectorAll('.gkview button').forEach(b => b.classList.toggle('on', b === btn));
  if (typeof Plotly !== 'undefined') volCharts();
}));
$('sm-date').addEventListener('change', e => {
  const ds = D.periods[period].vol.smile.date; let j = ds.findIndex(d => d >= e.target.value);
  if (j < 0) j = ds.length - 1; if (ds[j] > e.target.value && j > 0) j -= 1; smileAt(j);
});
$('sm-prev').addEventListener('click', () => smileJump(-1));
$('sm-next').addEventListener('click', () => smileJump(1));
show('test');
if (typeof Plotly !== 'undefined') fillHist();
"""


def vol_tabs(D: dict) -> str:
    """The Volatility section's four views; each draws when its tab is shown."""
    chk = D["fills"].get("iv_check") or {}
    note = (f'Implied vols are inverted from each option\'s mid with Black-76. Across the {chk.get("n", 0):,} contract-days '
            f'in the band the wheel sells they match LSEG\'s own implied-vol field with a correlation of '
            f'{chk.get("corr", float("nan")):.3f} (median difference {chk.get("median_abs_diff_pts", float("nan")):.2f} vol points).')
    panes = [
        ("Smile", '<div class="cap" style="margin:0 0 10px">The implied vol of every out-of-the-money option (puts below '
                  'the close, calls above it) of the monthly expiry closest to 30 days out, on one session. Puts below '
                  'the money cost more vol than calls above it: the skew the wheel\'s puts are sold into. The option the '
                  'account is short that day is marked when it belongs to the same expiry.</div>'
                  '<div class="pf-controls"><button id="sm-prev">◀ previous sale</button>'
                  '<input id="sm-range" type="range" min="0" value="0" aria-label="session">'
                  '<button id="sm-next">next sale ▶</button><input id="sm-date" type="date" aria-label="date"></div>'
                  '<div id="sm-info" class="pf-eq"></div><div id="plot-smile" style="height:340px"></div>'),
        ("Surface over time", '<div class="cap" style="margin:0 0 10px">Every session\'s smile side by side: the date '
                  'across, strike as a share of the close down, colour the implied vol of the monthly closest to 30 days '
                  'out. The markers are the options the wheel sold, at their strike against the close on the day of sale. '
                  'Only one or two monthlies are quoted on any day, so this is the surface at a fixed maturity over time '
                  'rather than across maturities.</div>'
                  '<div class="switch surfview" style="margin:0 0 8px"><button data-sv="heat" class="on">Heatmap</button>'
                  '<button data-sv="3d">3D surface</button></div>'
                  '<div id="plot-surface" style="height:470px"></div>'
                  '<div class="cap" id="surf-hint" style="display:none">Drag to rotate, scroll to zoom, double-click to reset. '
                  'Gaps are strikes with no quote that day.</div>'),
        ("Implied vs realised", '<div class="cap" style="margin:0 0 10px">For each option the wheel sold: the implied '
                  'vol it was sold at, and the volatility Mastercard then realised to its expiry (daily closes, '
                  'annualised). Selling above what follows is the volatility risk premium the strategy is paid for. These are the '
                  'options actually sold; the thesis table uses a fixed yardstick instead, the 0.25-delta put of every '
                  'cycle, so its count can differ.</div>'
                  '<div id="vrp-info" class="pf-eq" style="margin-top:0"></div><div id="plot-vrp" style="height:320px"></div>'),
        ("Greeks", '<div class="cap" style="margin:0 0 10px">The four greeks the course uses, drawn as the lecture '
                  'draws them: each over strike as a share of the close (K/S) and days to expiry, for one option held '
                  'long, per share, with the white frame at zero. Θ is in dollars per day and ν in dollars per vol point. '
                  'Each option the wheel sold is traced across its surface from the day it was sold (dot) to the session '
                  'before its expiry.</div>'
                  '<div class="switch gkview" style="margin:0 0 8px">'
                  + "".join(f'<button data-gk="{k}" class="{"on" if i == 0 else ""}">{n}</button>' for i, (k, n) in enumerate(
                      [("put_delta", "Put Δ"), ("call_delta", "Call Δ"), ("put_theta", "Put Θ"), ("call_theta", "Call Θ"),
                       ("gamma", "Γ"), ("vega", "ν")]))
                  + '</div><div id="gk-info" class="pf-eq" style="margin-top:0"></div>'
                  '<div id="plot-gsurf" style="height:560px"></div>'
                  '<div class="cap">Drag to rotate, scroll to zoom, double-click to reset.</div>'
                  '<div class="sub">Three clocks, one at-the-money option</div>'
                  '<div class="g-3"><div id="plot-gclk-gamma" style="height:220px"></div>'
                  '<div id="plot-gclk-theta" style="height:220px"></div><div id="plot-gclk-vega" style="height:220px"></div></div>'
                  '<div class="cap">Γ and the vol piece of Θ blow up into expiry (1/√T); ν dies (√T). The shaded band is '
                  'where the wheel\'s monthly options live. These are long-option surfaces, as in the lecture; the wheel '
                  'sold them, so its book carries the opposite signs: Θ &gt; 0, Γ &lt; 0, ν &lt; 0. Buying theta is '
                  'selling gamma and vega.</div>'),
    ]
    return ('<div class="tabs">' + "".join(f'<button data-tab="{i}" class="{"on" if i == 0 else ""}">{escape(n)}</button>'
                                         for i, (n, _) in enumerate(panes)) + "</div>"
            + "".join(f'<div class="pane{" on" if i == 0 else ""}" data-pane="{i}">{b}</div>' for i, (_, b) in enumerate(panes))
            + f'<div class="cap" style="padding:0 14px 12px">{escape(note)}</div>')


def tip(prose: dict[str, str], label: str) -> str:
    body = prose.get(f"tip: {label}", "").strip()
    if not body:
        return ""
    return f'<span class="tip" tabindex="0" aria-label="About {escape(label)}">i<span class="bubble">{body}</span></span>'


def render(D: dict, prose: dict[str, str], draft: bool, P2: dict | None = None,
           names: tuple[str, str] = ("hw4.html", "hw4_part2.html")) -> tuple[str, str | None]:
    """Part I's page and, with P2, Part II's: `names` are their file names, for the links between them."""
    missing = []

    def say(key: str, cls: str = "") -> str:
        body = prose.get(key, "").strip()
        if body:
            return f'<div class="prose {cls}">{body}</div>'
        missing.append(key)
        return f'<div class="todo">Not yet written: {escape(key)}</div>' if draft else ""

    def panel(title: str, body: str, tag: str = "", flush: bool = False, pid: str = "") -> str:
        return (f'<section class="panel"{f" id={chr(34)}{pid}{chr(34)}" if pid else ""}><div class="ph">{escape(title)}{tip(prose, title)}'
                + (f'<span class="tag">{tag}</span>' if tag else "")
                + f'</div><div class="pb{" flush" if flush else ""}">{body}</div></section>')

    T, R = D["periods"]["test"], D["periods"]["train"]
    tips = {k: tip(prose, k) for k in ("Sharpe ratio", "Max drawdown", "Premium collected", "T-bill interest")}
    stats, own = fill_tables(D)
    data = D["data"]
    tabs = [("Blotter", say("blotter") + '<div id="blotter" style="margin-top:8px"></div>'),
            ("Ledger", '<div class="cap" style="margin:0 0 8px">One row per session, at the close. '
                       'Click a row to see that day in the portfolio panel.</div><div id="ledger"></div>'),
            ("NAV", '<div class="cap" style="margin:0 0 10px">The account value (NAV) at every close and what it is '
                    'made of: cash, which includes the cash securing an open put, and Mastercard shares at the close, '
                    'less the open option at its mid. Then how far it stood below its highest close, and the month-end '
                    'values the Sharpe ratio is computed from.</div>'
                    '<div class="sub" style="margin-top:0">Account value</div>'
                    '<div id="plot-navc" class="tabplot" style="height:300px"></div>'
                    '<div class="sub">What the account holds, as a share of its value</div>'
                    '<div id="plot-hold" class="tabplot" style="height:170px"></div>'
                    '<div class="sub">Drawdown from the highest close so far</div>'
                    '<div id="plot-dd" class="tabplot" style="height:200px"></div>'
                    '<div class="sub">Month-end account value</div><div id="navtable"></div>'
                    '<div id="sharpe-line" class="cap"></div>'),
            ("Reg T", '<div class="cap" style="margin:0 0 10px">The wheel runs in a cash account under Reg T: an open '
                      'put must have its strike times its shares set aside in cash, and an open call must be covered '
                      'by shares held. Nothing is borrowed, so there is no margin requirement; what Reg T asks is that '
                      'free cash never goes below zero and that every call is covered.</div>'
                      '<div id="regt-summary"></div>'
                      '<div class="sub">What the account held at the start of each monthly cycle</div>'
                      '<div id="plot-regt" class="tabplot" style="height:300px"></div>'
                      '<div class="cap">One bar per cycle, at the close on which its option was sold. Between one put '
                      'expiring on a Friday and the next being sold on the Monday, the account holds only cash for a '
                      'session; the daily table below shows those sessions.</div>'
                      '<div class="sub">Every session</div><div id="regt-table"></div>'),
            ("Cycles", '<div id="cycles"></div>'),
            ("Rules", rules_list(D))]
    tab_html = ('<div class="tabs">' + "".join(
        f'<button data-tab="{i}" class="{"on" if i == 0 else ""}">{escape(n)}</button>' for i, (n, _) in enumerate(tabs))
        + "</div>" + "".join(f'<div class="pane{" on" if i == 0 else ""}" data-pane="{i}">{b}</div>'
                             for i, (_, b) in enumerate(tabs)))
    script = (SCRIPT.replace("/*__DATA__*/null", json.dumps(page_data(D), separators=(",", ":")))
              .replace("const D =", "const TIPS = " + json.dumps(tips) + ";\nconst D ="))

    def page_top(title: str, active: str) -> str:
        """The head and the site header both pages share: Part I and Part II as two pages of one site."""
        links = [("1", "Part I · The assignment", names[0])] + ([("2", "Part II · Further exploration", names[1])] if P2 else [])
        nav = "".join(f'<a href="{href}"' + (' class="on" aria-current="page"' if k == active else "") + f'>{escape(label)}</a>'
                      for k, label, href in links)
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>{STYLE}</style></head><body><div class="wrap">
<header class="site"><a class="brand" href="{names[0]}"><span class="sym">MA</span><span class="co">{escape(D["company"])}</span>
<span class="name">Wheel strategy · backtest</span></a>
<nav class="pages" aria-label="Pages">{nav}</nav>
<div class="right"><a class="chip hot" href="{REPO_URL}">Code (to GitHub Repo) ↗</a>
<a class="chip" href="hw3.html">← PMCC page</a></div></header>"""

    def heading(anchor: str, eyebrow: str, title: str, what: str, facts: list[tuple[str, str]]) -> str:
        """A page's heading: what part it is, its title, one line on what it holds, and its key facts."""
        return (f'<div class="part" id="{anchor}"><span class="no">{escape(eyebrow)}</span><h2>{escape(title)}</h2>'
                f'<span class="what">{escape(what)}</span></div>'
                + '<div class="meta">' + "".join(f'<span class="chip"><i>{escape(k)}</i><b>{escape(v)}</b></span>'
                                               for k, v in facts) + "</div>")

    parts = [
        page_top("Wheel Strategy · MA", "1"),
        heading("part1", "Part I · The assignment", "The wheel on Mastercard (MA)",
                "Everything the assignment asks for: the thesis and whether it held, the blotter, the ledger, "
                "the portfolio on any date, and the evidence for the fill prices.",
                [("Test period", f'{T["window"][0]} → {T["window"][1]}'),
                 ("Setting chosen on", f'{R["window"][0]} → {R["window"][1]}'),
                 ("Starting cash", "$100,000"), ("Data", "Daily closes · LSEG")]),
        '<div style="height:12px"></div>',
        panel("Thesis and whether it held", say("thesis", "lead") + '<div style="height:12px"></div>'
              + verdict_table(D) + '<div style="height:10px"></div>' + say("verdict"), "2026 test"),
        '<div class="switch"><button data-period="test" class="on">' + escape(PERIOD_LABEL["test"]) + '</button>'
        '<button data-period="train">' + escape(PERIOD_LABEL["train"]) + '</button></div>',
        '<div id="kpis" class="kpis"></div>',
        '<div class="grid g-2-1" style="margin-top:0">',
        panel("Account value", '<div id="plot-nav" style="height:400px"></div>',
              'from $100,000 · <span data-window></span>', flush=True),
        panel("Benchmarks", '<div id="bench"></div><div class="cap">The Cboe S&amp;P 500 PutWrite index sells a '
              'one-month at-the-money S&amp;P 500 put each month, secured by cash in T-bills: the industry standard for '
              'systematic put selling. Added as a benchmark after the test.</div>',
              '<span data-window></span>'),
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
        '<div style="height:12px"></div>',
        f'<section class="panel" id="volatility"><div class="ph">Volatility<span class="tag"><span data-window></span></span></div>{vol_tabs(D)}</section>',
        '<div class="grid g-1-1">',
        panel("Fill prices", say("fills") + '<div class="sub">Notes</div>' + say("fills-note")
              + '<div class="sub">The stress test: the same wheel selling every option at the bid</div>'
              + fill_stress_table(D)),
        panel("Last trade against the closing quote",
              '<div class="switch fillview" style="margin:8px 10px 0"><button data-view="hist" class="on">Where in the spread</button>'
              '<button data-view="scatter">Trade against mid</button></div>'
              '<div class="fv on" data-view="hist"><div id="plot-fill" style="height:340px"></div>'
              f'<div class="cap" style="padding:0 10px 8px">Each bar is the share of the {D["fills"]["in_band"]["n"]:,} '
              'contract-days in the band the wheel sells, 2024 to October 2026, whose last trade printed in that part of '
              'the closing quote: 0 is the bid, 0.5 the mid, 1 the ask. Inside the quote the prints centre on the mid. '
              'Prints outside it mostly reflect the last trade coming hours before the closing quote.</div></div>'
              '<div class="fv" data-view="scatter"><div id="plot-scatter" style="height:340px"></div>'
              '<div class="cap" style="padding:0 10px 8px">Each point is one of the same trades: the closing mid against '
              'the day\'s last trade. Points on the diagonal traded at the mid; the guide lines sit half the median spread '
              f'({100 * D["fills"]["scatter"]["half_spread"]:.1f}% of the mid) either side, roughly where the bid and the ask '
              f'were. Zoomed to the 98% of points under ${D["fills"]["scatter"]["zoom"]:.0f}; drag to zoom, double-click to reset.</div></div>',
              "LSEG TRDPRC_1 against the closing BID and ASK", flush=True),
        "</div>",
        panel("Where trades printed, by sample", stats
              + "<details><summary>Each option the wheel sold, on the day it sold it</summary>" + own + "</details>",
              "location: 0 = bid, 0.5 = mid, 1 = ask"),
        '<div class="grid g-1-1">',
        panel("Stock choice", say("stock") + '<div class="sub">The screen, top ten</div>' + screen_table(D),
              f'{D["reversal"]["ranked"]} of {D["reversal"]["eligible"]} eligible stocks ranked'),
        panel("Setting choice", say("setting") + setting_verdict(D) + '<div class="sub">All 18 settings, at mid fills</div>' + grid_table(D),
              "highlighted: the pick"),
        "</div>",
        (f'<a class="next" href="{names[1]}">Next: Part II · Further exploration, the same wheel on five stocks →</a>'
         if P2 else ""),
        f"<script>{script}</script>",
        "</div></body></html>",
    ]
    second = None
    if P2:
        second = "\n".join([page_top("Wheel Strategy · Five Stocks", "2"), *part2_html(P2, say, panel, heading),
                            f'<a class="next" href="{names[0]}">← Back to Part I · The assignment, the wheel on Mastercard</a>',
                            "</div></body></html>"])
    if missing:
        print(f"  prose sections still empty: {', '.join(missing)}", file=sys.stderr)
    return "\n".join(parts), second


def part2_html(P2: dict, say, panel, heading) -> list[str]:
    results, each_t, corr = part2_tables(P2)
    lo, hi = P2["windows"]["train"]["window"], P2["windows"]["test"]["window"]
    return [
        heading("part2", "Part II · Further exploration", "The same wheel on five stocks",
                "Not required by the assignment. The plain wheel of Part I, run on all five stocks the screen "
                "picked at once.",
                [("Periods", f"{lo[0]} → {lo[1]}  ·  {hi[0]} → {hi[1]}"), ("Stocks", " · ".join(P2["stocks"])),
                 ("Starting cash", "5 × $100,000"), ("Data", "Daily closes · LSEG")]),
        '<div style="height:6px"></div>',
        panel("Motivation", say("part2-motivation") + '<div style="height:8px"></div>' + say("part2-setup")),
        '<div style="height:12px"></div>',
        panel("Results", results, "$500,000 to start in each period; each period starts from cash"),
        '<div class="grid g-1-1">',
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
    second_path = args.out.with_name(args.out.stem + "_part2.html")
    first, second = render(D, load_prose(f), args.draft, P2, (args.out.name, second_path.name))
    for path, html in ((args.out, first), (second_path, second)):
        if html is not None:
            path.write_text(html, encoding="utf-8")
            print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
