"""
Build docs/hw3.html, the PMCC page, from two inputs:

    cache/pmcc_results_<TICKER>.json   every number, written by run_pmcc.py
    scripts/hw3_prose.md               every sentence of prose

Tables and charts are generated here from the results file. Prose is never
generated: it is read from hw3_prose.md, where `{{name}}` is replaced by a
computed figure so a number in a sentence cannot drift from the table beside
it. An unknown placeholder stops the build.

    python3 scripts/run_pmcc.py && python3 scripts/build_hw3.py
    python3 scripts/build_hw3.py --facts     # list the placeholders and their values
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from html import escape
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PROSE = ROOT / "scripts" / "hw3_prose.md"
OUT = REPO / "docs" / "hw3.html"
RULES_FILE = REPO / "config" / "thresholds.yaml"

BOOK_LABELS = {
    "pmcc": "PMCC (primary: mid fills)",
    "pmcc_cross": "PMCC, bid/ask fills (stress test)",
    "pmcc_skip": "PMCC, skip earnings cycles",
    "pmcc_v1": "PMCC, first version of the rules",
    "cc": "Covered call",
    "cc_cross": "Covered call, bid/ask fills",
    "hold": "Buy and hold, with dividends",
}
COLORS = {"pmcc": "#199e70", "pmcc_v1": "#9085e9", "cc": "#c98500", "hold": "#8792AB"}


def usd(x, dp=0, sign=False) -> str:
    if x is None:
        return "–"
    s = f"{abs(x):,.{dp}f}"
    lead = "−" if x < 0 else ("+" if sign and x > 0 else "")
    return f"{lead}${s}"


def pct(x, dp=1, sign=False) -> str:
    if x is None:
        return "–"
    lead = "−" if x < 0 else ("+" if sign and x > 0 else "")
    return f"{lead}{abs(x):.{dp}f}%"


def num(x, dp=2) -> str:
    return "–" if x is None else f"{x:,.{dp}f}".replace("-", "−")


def table(headers: list[str], rows: list[list], left: int = 1, cls: str = "") -> str:
    """`left` leading columns are text; the rest are right-aligned figures."""
    th = "".join(f'<th class="{"l" if i < left else ""}">{escape(h)}</th>'
                 for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(f'<td class="{"l" if i < left else ""}">{c}</td>'
                         for i, c in enumerate(r)) + "</tr>" for r in rows)
    return (f'<div class="card"><div class="tbl-wrap"><table class="tbl {cls}">'
            f"<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div></div>")


# --------------------------------------------------------------------------
# facts: the figures prose may quote
# --------------------------------------------------------------------------

def count_tests() -> int:
    tree = ast.parse((ROOT / "tests" / "pmcc" / "test_pmcc_daily.py").read_text())
    return sum(isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
               for n in ast.walk(tree))


def facts(R: dict) -> dict[str, str]:
    f = {"ticker": R["ticker"], "start": R["period"][0], "end": R["period"][1],
         "rules_commit": R["rules_commit"], "tests": str(count_tests()),
         "stock_first": usd(R["stock"]["first"], 2), "stock_last": usd(R["stock"]["last"], 2),
         "stock_change": pct(100 * (R["stock"]["last"] / R["stock"]["first"] - 1), 1, True)}
    for k, b in R["books"].items():
        p, m = b["performance"], b["performance_funded"]
        sc = b["status_counts"]
        f.update({
            f"{k}_open_cash": usd(p["capital"]), f"{k}_regt_cash": usd(m["capital"]),
            f"{k}_pnl": usd(p["pnl"], sign=True),
            f"{k}_return_open": pct(p["return_pct"], sign=True),
            f"{k}_return_regt": pct(m["return_pct"], sign=True),
            f"{k}_drawdown": pct(m["max_drawdown_pct"]),
            f"{k}_worst_month": pct(m["worst_month_pct"]),
            f"{k}_worst_month_name": m["worst_month"],
            f"{k}_sharpe": num(m["sharpe"]),
            f"{k}_premium": usd(b["premium_collected"]),
            f"{k}_cover_cost": usd(b["cover_cost"]),
            f"{k}_written": str(sc.get("expired", 0) + sc.get("assigned", 0)),
            f"{k}_assigned": str(sc.get("assigned", 0)),
            f"{k}_expired": str(sc.get("expired", 0)),
            f"{k}_skipped": str(sum(v for s, v in sc.items() if s.startswith("skipped"))),
            f"{k}_cycles": str(len(b["cycles"])),
            f"{k}_days_cash_negative": str(b["days_cash_negative"]),
            f"{k}_worst_available": usd(b["min_start_cash"]["worst_available"]),
            f"{k}_worst_available_date": b["min_start_cash"]["worst_date"],
            f"{k}_leap_pnl": usd(b["pnl_by_leg"]["long"], sign=True),
            f"{k}_short_pnl": usd(b["pnl_by_leg"]["short"], sign=True),
            f"{k}_stock_pnl": usd(b["pnl_by_leg"]["stock"], sign=True),
        })
    d = R["data"]
    f.update({"requests": f"{d['requests']:,}",
              "returned": f"{d['by_status'].get('returned', 0):,}",
              "never_listed": f"{d['by_status'].get('never_listed', 0):,}",
              "empty": f"{d['by_status'].get('empty', 0):,}",
              "failed": f"{d['by_status'].get('failed', 0):,}",
              "quote_rows": f"{d['quote_rows']:,}", "crossed_rows": f"{d['crossed_rows']:,}",
              "sessions": f"{d['sessions']:,}"})
    for row in R["earnings_split"]["pmcc"]:
        key = {"an earnings date": "earn", "no earnings date": "no_earn"}[row["group"]]
        f.update({f"{key}_cycles": str(row["cycles"]), f"{key}_assigned": str(row["assigned"]),
                  f"{key}_mean_pnl": usd(row["mean_pnl"], sign=True),
                  f"{key}_worst_pnl": usd(row["worst_pnl"], sign=True),
                  f"{key}_mean_premium": usd(row["mean_premium"])})
    return f


# --------------------------------------------------------------------------
# prose
# --------------------------------------------------------------------------

def load_prose(f: dict[str, str]) -> dict[str, str]:
    """
    `## key` opens a section. Blank lines separate paragraphs, lines starting
    with "- " make a list, and lines starting with "//" are notes to the
    author that never reach the page.
    """
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
            raise SystemExit(f"hw3_prose.md uses {{{{{name}}}}}, which is not a computed figure. "
                             f"Run build_hw3.py --facts for the list.")
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
# generated blocks
# --------------------------------------------------------------------------

def summary_table(R: dict) -> str:
    rows = []
    for k, label in BOOK_LABELS.items():
        b = R["books"][k]
        p, m = b["performance"], b["performance_funded"]
        rows.append([escape(label), usd(p["capital"]), usd(m["capital"]), usd(p["pnl"], sign=True),
                     pct(p["return_pct"], sign=True), pct(m["return_pct"], sign=True),
                     pct(m["max_drawdown_pct"]), f'{pct(m["worst_month_pct"])} ({m["worst_month"]})',
                     num(m["sharpe"])])
    return table(["Book", "Cash to open", "Reg T cash", "P&L", "Return on cash to open",
                  "Return on Reg T cash", "Worst fall", "Worst month", "Sharpe"], rows)


def rules_table() -> str:
    rules = yaml.safe_load(RULES_FILE.read_text())
    rows = []

    def walk(prefix, v):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(f"{prefix} · {k}" if prefix else k, x)
        elif isinstance(v, list):
            rows.append([escape(prefix.replace("_", " ")),
                         "<br>".join(escape(str(x)) for x in v)])
        else:
            rows.append([escape(prefix.replace("_", " ")), escape(" ".join(str(v).split()))])

    walk("", rules)
    return table(["Rule", "Registered value"], rows, left=2, cls="rules")


def cycles_table(R: dict, book: str = "pmcc") -> str:
    rows = []
    for c in R["books"][book]["cycles"]:
        wrote = c.get("premium") is not None
        rows.append([
            c["entry"], c["expiry"], num(c["spot"]),
            num(c.get("long_strike"), 1) if c.get("long_strike") else "–",
            num(c["strike"], 1) if wrote else "–",
            num(c["short_delta"]) if wrote else "–",
            f'{num(c["bid"])} / {num(c["ask"])}' if wrote else "–",
            usd(c["premium"]) if wrote else "–",
            num(c["settle"]) if wrote else "–",
            escape(c["status"]) + (" · earnings" if c.get("earnings_in_cycle") else ""),
            usd(c["short_pnl"], sign=True) if wrote else "–",
        ])
    return table(["Entry", "Expiry", "Stock", "LEAP strike", "Short strike", "Delta",
                  "Bid / ask", "Premium", "Close at expiry", "Outcome", "Short-call P&L"],
                 rows, left=2)


def assignments_table(R: dict, book: str = "pmcc") -> str:
    rows = [[a["expiry"], num(a["strike"], 1), num(a["settle"]), a.get("cover_date", "–"),
             num(a.get("cover_price")), num(a.get("gap")), usd(a["premium"]),
             usd(a.get("cover_cost"), sign=True)]
            for a in R["books"][book]["assignments"]]
    return table(["Expiry", "Strike", "Close", "Covered", "Buy-back price",
                  "Close-to-open gap", "Premium kept", "Cost over strike"], rows)


def leap_table(R: dict) -> str:
    rows = [[e["date"], e["side"], escape(e["instrument"]), num(e["strike"], 1), e["expiry"],
             f'{num(e["bid"])} / {num(e["ask"])}', num(e["price"]), usd(e["cash_delta"], sign=True)]
            for e in R["books"]["pmcc"]["blotter"] if e["leg"] == "long"]
    return table(["Date", "Side", "Contract", "Strike", "Expiry", "Bid / ask", "Fill", "Cash"],
                 rows, left=3)


def earnings_table(R: dict) -> str:
    rows = [[escape(r["group"]), str(r["cycles"]), str(r["assigned"]), usd(r["mean_premium"]),
             usd(r["mean_pnl"], sign=True), usd(r["median_pnl"], sign=True),
             usd(r["worst_pnl"], sign=True)] for r in R["earnings_split"]["pmcc"]]
    return table(["Cycle contains", "Cycles", "Assigned", "Mean premium",
                  "Mean short-call P&L", "Median", "Worst"], rows)


def blotter_table(R: dict) -> str:
    rows = []
    for e in R["books"]["pmcc"]["blotter"]:
        quote = (f'{num(e["bid"])} / {num(e["ask"])}'
                 if e.get("bid") is not None and e.get("ask") is not None else "–")
        rows.append([e["date"], e["when"], escape(e["instrument"]),
                     f'<span class="s-{e["side"]}">{e["side"]}</span>', str(e["qty"]),
                     quote, num(e["price"]), usd(e["cash_delta"], 2, sign=True),
                     f'<span class="note">{escape(e["note"])}</span>'])
    return table(["Date", "At", "Instrument", "Side", "Qty", "Bid / ask", "Fill", "Cash", "Rule"],
                 rows, left=4, cls="blotter")


def chart_data(R: dict) -> dict:
    out = {"books": {}}
    for k in ("pmcc", "pmcc_v1", "cc", "hold"):
        b = R["books"][k]
        cap = b["performance"]["capital"]
        L = b["ledger"]
        out["books"][k] = {"label": BOOK_LABELS[k], "color": COLORS[k], "date": L["date"],
                           "pnl": [round(v - cap, 2) for v in L["nav"]]}
    L = R["books"]["pmcc"]["ledger"]
    out["pmcc"] = {k: L[k] for k in ("date", "stock_close", "long_mv", "cash",
                                     "available_funds", "delta", "vega")}
    return out


CHART_JS = r"""
const D = /*__DATA__*/null;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const base = (title, ytitle) => ({
  paper_bgcolor: css('--panel'), plot_bgcolor: css('--panel'),
  font: {family: css('--font'), color: css('--muted'), size: 12},
  margin: {l: 64, r: 56, t: 14, b: 36}, hovermode: 'x unified',
  legend: {orientation: 'h', y: -0.14},
  xaxis: {gridcolor: css('--line-soft'), linecolor: css('--line')},
  yaxis: {title: ytitle, gridcolor: css('--line-soft'), zerolinecolor: css('--line')},
});
const CFG = {displayModeBar: false, responsive: true};
if (typeof Plotly !== 'undefined') {
  Plotly.newPlot('plot-pnl', Object.values(D.books).map(b => ({
    x: b.date, y: b.pnl, name: b.label, mode: 'lines', line: {color: b.color, width: 1.8},
    hovertemplate: '%{y:$,.0f}<extra>' + b.label + '</extra>'})),
    base('Profit and loss since the first trade, one contract or 100 shares', 'dollars'), CFG);
  const P = D.pmcc;
  Plotly.newPlot('plot-leap', [
    {x: P.date, y: P.long_mv, name: 'LEAP market value', mode: 'lines', line: {color: css('--mark'), width: 1.8},
     hovertemplate: '%{y:$,.0f}<extra>LEAP</extra>'},
    {x: P.date, y: P.stock_close, name: 'Stock close (right axis)', mode: 'lines', yaxis: 'y2',
     line: {color: css('--muted'), width: 1.2}, hovertemplate: '%{y:$.2f}<extra>stock</extra>'}],
    {...base('The LEAP and the stock', 'dollars'),
     yaxis2: {overlaying: 'y', side: 'right', showgrid: false, title: 'stock, dollars'}}, CFG);
  Plotly.newPlot('plot-funds', [
    {x: P.date, y: P.available_funds, name: 'Available funds', mode: 'lines',
     line: {color: css('--both'), width: 1.4}, hovertemplate: '%{y:$,.0f}<extra>available funds</extra>'}],
    base('Reg T available funds when the account holds only the cash to open', 'dollars'), CFG);
  Plotly.newPlot('plot-delta', [
    {x: P.date, y: P.delta, name: 'Net delta, share equivalents', mode: 'lines',
     line: {color: css('--mark'), width: 1.4}, hovertemplate: '%{y:.0f} shares<extra>delta</extra>'},
    {x: P.date, y: P.vega, name: 'Net vega, dollars per vol point (right axis)', mode: 'lines', yaxis: 'y2',
     line: {color: css('--print'), width: 1.2}, hovertemplate: '%{y:$.1f}<extra>vega</extra>'}],
    {...base('Net delta and vega of the PMCC', 'share equivalents'),
     yaxis2: {overlaying: 'y', side: 'right', showgrid: false, title: 'dollars per vol point'}}, CFG);
}
// Tabs. A chart drawn in a hidden pane has no width until the pane is shown.
document.querySelectorAll('.tabs button').forEach(btn => btn.addEventListener('click', () => {
  const root = btn.closest('.panel');
  root.querySelectorAll('.tabs button').forEach(b => b.classList.toggle('on', b === btn));
  root.querySelectorAll('.pane').forEach(p => p.classList.toggle('on', p.dataset.pane === btn.dataset.tab));
  if (typeof Plotly !== 'undefined')
    root.querySelectorAll('.pane.on .plot').forEach(el => Plotly.Plots.resize(el));
}));
"""

EXTRA_CSS = """
  .wrap{max-width:1440px}
  .tbl-wrap{overflow-x:auto;border-radius:8px;max-width:100%}
  table.tbl td.l, table.tbl th.l{text-align:left}
  table.tbl td{white-space:nowrap}
  table.tbl td:first-child{white-space:normal}
  table.tbl.rules td{white-space:normal;font-family:var(--font)}
  table.tbl.blotter td .note{display:block;min-width:260px;white-space:normal;
    color:var(--faint);font-size:var(--fs-xs);font-family:var(--font);line-height:1.45}
  .s-BUY{color:var(--up)} .s-SELL{color:var(--mark)}
  .s-EXPIRE{color:var(--muted)} .s-ASSIGN{color:var(--print)}
  .neg{color:var(--down)} .posv{color:var(--mark)}

  /* ---- dashboard shell ---------------------------------------------- */
  .bar{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:14px;flex-wrap:wrap;
    padding:12px 0;background:var(--base);border-bottom:1px solid var(--line)}
  .bar .sym{font-family:var(--mono);font-weight:700;font-size:18px;letter-spacing:.02em}
  .bar .name{color:var(--muted);font-size:var(--fs-sm)}
  .bar .right{margin-left:auto;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
  .chip{font-family:var(--mono);font-size:var(--fs-xs);color:var(--muted);border:1px solid var(--line);
    border-radius:999px;padding:4px 10px;background:var(--panel);white-space:nowrap}
  .chip b{color:var(--text);font-weight:600}
  .chip a{color:inherit;text-decoration:none}
  :root{--up:#199e70;--down:#e5604d}
  .kpis{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:10px;margin:14px 0}
  @media (max-width:1280px){.kpis{grid-template-columns:repeat(4,minmax(0,1fr))}}
  @media (max-width:640px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
  .kpi{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px;min-width:0}
  .kpi .k{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
  .kpi .v{font-family:var(--mono);font-size:20px;font-weight:700;letter-spacing:-.02em;margin-top:5px}
  .kpi .d{font-family:var(--mono);font-size:var(--fs-xs);color:var(--faint);margin-top:4px}
  .grid{display:grid;gap:12px;margin-bottom:12px}
  .g-2-1{grid-template-columns:minmax(0,2fr) minmax(0,1fr)}
  .g-1-1{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}
  @media (max-width:980px){.g-2-1,.g-1-1{grid-template-columns:minmax(0,1fr)}}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;min-width:0;
    display:flex;flex-direction:column}
  .panel > .ph{display:flex;align-items:center;gap:10px;padding:9px 14px;border-bottom:1px solid var(--line);
    font-size:11px;letter-spacing:.11em;text-transform:uppercase;color:var(--muted);font-weight:600}
  .panel > .ph .tag{margin-left:auto;font-family:var(--mono);letter-spacing:0;text-transform:none;
    font-weight:400;color:var(--faint)}
  .panel > .pb{padding:12px 14px;min-width:0}
  .panel > .pb.flush{padding:4px}
  .panel .card{border:none;background:none;padding:0;margin:0}
  .prose{color:var(--text);font-size:var(--fs-base);line-height:1.62;max-width:92ch}
  .prose p{margin:0 0 10px} .prose p:last-child{margin-bottom:0}
  .prose ul{padding-left:20px;margin:0 0 10px} .prose li{margin-bottom:6px}
  .todo{border:1px dashed var(--print);color:var(--print);padding:8px 12px;border-radius:6px;
    font-size:var(--fs-sm)}
  dl.rules{margin:0;display:grid;grid-template-columns:auto minmax(0,1fr);gap:9px 14px;font-size:var(--fs-sm)}
  dl.rules dt{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;
    color:var(--both);padding-top:2px;white-space:nowrap}
  dl.rules dd{margin:0;color:var(--text);line-height:1.5}
  .tabs{display:flex;gap:2px;flex-wrap:wrap;padding:6px 8px 0;border-bottom:1px solid var(--line)}
  .tabs button{font:inherit;font-size:var(--fs-sm);color:var(--muted);background:none;border:none;
    border-bottom:2px solid transparent;padding:8px 12px;cursor:pointer}
  .tabs button:hover{color:var(--text)}
  .tabs button.on{color:var(--text);border-bottom-color:var(--mark)}
  .pane{display:none;padding:10px} .pane.on{display:block}
  .pane .tbl-wrap{max-height:520px;overflow-y:auto}
  .pane table.tbl thead th{position:sticky;top:0;background:var(--panel-hi);z-index:1}
"""


def rule_card() -> str:
    """The strategy in five lines, each the registered text from the rules file."""
    r = yaml.safe_load(RULES_FILE.read_text())
    lines = [
        ("Long leg", f'{r["long_leg"]["expiry"]}; {r["long_leg"]["strike"]}. Roll: {r["long_leg"]["roll"]}'),
        ("Short leg", f'{r["short_leg"]["expiry"]}; {r["short_leg"]["strike"]}'),
        ("Earnings", f'{r["earnings"]["primary"]} (comparison: {r["earnings"]["comparison"]})'),
        ("Fills", f'{r["fills"]["primary"]} (stress test: {r["fills"]["stress_test"]})'),
        ("Assigned", f'{r["assignment"]["rule"]}, at {r["assignment"]["buy_back_price"]}'),
    ]
    return '<dl class="rules">' + "".join(
        f"<dt>{escape(a)}</dt><dd>{escape(b)}</dd>" for a, b in lines) + "</dl>"


def kpis(R: dict) -> str:
    b, c, h = R["books"]["pmcc"], R["books"]["cc"], R["books"]["hold"]
    p, m = b["performance"], b["performance_funded"]
    sc = b["status_counts"]
    written = sc.get("expired", 0) + sc.get("assigned", 0)
    items = [
        ("PMCC P&L", usd(p["pnl"], sign=True), f'{pct(p["return_pct"], sign=True)} of cash to open', p["pnl"]),
        ("Covered call P&L", usd(c["performance"]["pnl"], sign=True),
         f'buy and hold {usd(h["performance"]["pnl"], sign=True)}', c["performance"]["pnl"]),
        ("Cash to open", usd(p["capital"]), f'covered call {usd(c["performance"]["capital"])}', None),
        ("Reg T cash needed", usd(m["capital"]), f'covered call {usd(c["performance_funded"]["capital"])}', None),
        ("Worst fall", pct(m["max_drawdown_pct"]), "on Reg T cash", m["max_drawdown_pct"]),
        ("Worst month", pct(m["worst_month_pct"]), m["worst_month"], m["worst_month_pct"]),
        ("Calls assigned", f'{sc.get("assigned", 0)} / {written}',
         f'{sum(v for k, v in sc.items() if k.startswith("skipped"))} cycles skipped', None),
        ("Premium vs cover cost", usd(b["premium_collected"]), f'cover cost {usd(b["cover_cost"])}', None),
    ]
    return '<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="k">{escape(k)}</div>'
        f'<div class="v {"" if x is None else ("neg" if x < 0 else "posv")}">{v}</div>'
        f'<div class="d">{escape(d)}</div></div>' for k, v, d, x in items) + "</div>"


def render(R: dict, prose: dict[str, str], draft: bool, blotter: bool = False) -> str:
    shared = (ROOT / "scripts" / "page_template.html").read_text(encoding="utf-8")
    css = re.search(r"<style>(.*?)</style>", shared, re.S).group(1)
    missing = []

    def say(key: str, optional: bool = False) -> str:
        body = prose.get(key, "").strip()
        if body:
            return f'<div class="prose">{body}</div>'
        if optional:
            return ""
        missing.append(key)
        return f'<div class="todo">Not yet written: {escape(key)}</div>' if draft else ""

    def panel(title: str, body: str, tag: str = "", flush: bool = False) -> str:
        return (f'<section class="panel"><div class="ph">{escape(title)}'
                + (f'<span class="tag">{escape(tag)}</span>' if tag else "")
                + f'</div><div class="pb{" flush" if flush else ""}">{body}</div></section>')

    def plot(pid: str, height: int) -> str:
        return f'<div id="{pid}" class="plot" style="height:{height}px"></div>'

    t = escape(R["ticker"])
    book = R["books"]["pmcc"]
    tabs = [
        ("Cycles", say("trades", optional=True) + cycles_table(R)),
        ("Assignments", assignments_table(R)),
        ("LEAP trades", leap_table(R)),
        ("Earnings cycles", say("earnings", optional=True) + earnings_table(R)),
        ("Exposure", say("greeks", optional=True) + plot("plot-delta", 360)),
        ("Registered rules", rules_table()),
    ]
    if blotter:
        tabs.append(("Blotter", blotter_table(R)))
    tab_html = ('<div class="tabs">' + "".join(
        f'<button data-tab="{i}" class="{"on" if i == 0 else ""}">{escape(name)}</button>'
        for i, (name, _) in enumerate(tabs)) + "</div>" + "".join(
        f'<div class="pane{" on" if i == 0 else ""}" data-pane="{i}">{body}</div>'
        for i, (_, body) in enumerate(tabs)))

    parts = [
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Poor Man's Covered Call · {t}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>{css}{EXTRA_CSS}</style></head><body><div class="wrap">
<div class="bar"><span class="sym">{t}</span>
<span class="name">Poor Man's Covered Call · backtest</span>
<div class="right"><span class="chip"><b>{R["period"][0]}</b> to <b>{R["period"][1]}</b></span>
<span class="chip">daily bars · LSEG</span>
<span class="chip">rules revised after a first run · <b>{escape(R["rules_commit"])}</b></span>
<span class="chip"><a href="hw2.html">← covered call page</a></span></div></div>""",
        kpis(R),
        '<div class="grid g-2-1">',
        panel("Profit and loss", plot("plot-pnl", 400), "dollars, one contract or 100 shares", flush=True),
        '<div class="grid" style="margin:0;align-content:start">',
        panel("Summary", say("summary")),
        panel("Strategy", say("strategy") + rule_card(), "full rules in the tabs below"),
        "</div></div>",
        panel("Books", summary_table(R) + say("result"), "same engine, same data"),
        '<div style="height:12px"></div><div class="grid g-1-1">',
        panel("Reg T available funds", plot("plot-funds", 320), "account holding only the cash to open", flush=True),
        panel("LEAP value and the stock", plot("plot-leap", 320), flush=True),
        "</div>",
        panel("Capital", say("capital")),
        '<div style="height:12px"></div>',
        f'<section class="panel">{tab_html}</section>',
        '<div style="height:12px"></div>',
        panel("Limitations", say("limitations")),
        f"<script>{CHART_JS.replace('/*__DATA__*/null', json.dumps(chart_data(R), separators=(',', ':')))}</script>",
        "</div></body></html>",
    ]
    if missing:
        print(f"  prose sections still empty: {', '.join(missing)}", file=sys.stderr)
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=yaml.safe_load(RULES_FILE.read_text())["underlying"])
    ap.add_argument("--facts", action="store_true", help="print the placeholders and exit")
    ap.add_argument("--draft", action="store_true", help="mark unwritten prose sections on the page")
    ap.add_argument("--blotter", action="store_true",
                    help="include the trade-by-trade blotter (every fill is an LSEG quote)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    R = json.loads((REPO / "cache" / f"pmcc_results_{args.ticker}.json").read_text())
    f = facts(R)
    if args.facts:
        for k, v in f.items():
            print(f"{{{{{k}}}}}  =  {v}")
        return 0
    html = render(R, load_prose(f), args.draft, args.blotter)
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
