"""
docs/assets/nvda_marks.js in a real browser (Playwright's Chromium; skipped when it is not installed), on files built
from the made-up market (nm_synth) and served to a bare page that holds the element inside a section headed h3, as
the vol-arb page's panels are. The end labels never cross a line of the chart and stay inside it, at a wide and a
phone width, open and settled; the title is an h4 unless the element asks for another level, and says "Final mark"
once the cycle is settled; the fallback line names the latest mark, not tonight's; every close's figures are in a
table (not only in the hover), the touch tooltip follows a drag and hides when the finger lifts, the keyboard steps
through the closes; on a phone the P&L table fits its box, the models that chose each side under the side's row.
"""

from __future__ import annotations

import json
import math

import pytest

import nm_synth as S
from trading_app.lib import nvda_marks as NM

JS = S.REPO / "docs" / "assets" / "nvda_marks.js"
GEN = "2026-11-17T03:30:00Z"
BASE = "http://nm.test/"
MINUS, JOIN = "−", "⁠"


def _browser_ready() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return False
    try:
        with sync_playwright() as pw:
            pw.chromium.launch().close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _browser_ready(), reason="playwright chromium not installed")


def swung(doc: dict, peak: bool = False) -> dict:
    """The file with made-up P&L to date (the page reads nothing else of the rows): up and down, then a steep last
    move into each end point from beyond it, the shape that put the old end labels on top of the lines; with `peak`,
    a last close at the top of the chart instead, where an end label meets the report's label at a phone width."""
    out = json.loads(json.dumps(doc))
    put = out["marks"]["put_rich"]["mid"]["daily"]
    n = len(put)
    path = [0.0] + [round(400.0 * k * (1 if k % 2 else -0.5), 2) for k in range(1, n - 2)] + [4200.0, 1434.0]
    if peak:
        assert max(map(abs, path)) < 10000.0
        path[-1] = 10000.0                              # a grid line, so the end point is the chart's very top
    for side, sign in (("put_rich", 1.0), ("call_rich", -1.0)):
        for r, v in zip(out["marks"][side]["mid"]["daily"], path[-n:]):
            r["pnl_to_date"] = sign * v + 0.0
        out["marks"][side]["mid"]["pnl"] = out["pnl"][side]["mid"] = sign * path[-1]
    return out


@pytest.fixture(scope="module")
def docs():
    mk, _ = S.market()
    cfg = S.config(mk)
    spec = NM.cycle_spec(cfg)
    out = {d: NM.document(NM.mark_open_cycle(mk, spec, d), spec, cfg, generated_at=GEN)
           for d in ("2026-10-19", "2026-10-20", "2026-11-05", "2026-11-16", "2026-11-19", "2026-11-20")}
    out["swung-open"], out["swung-settled"] = swung(out["2026-11-16"]), swung(out["2026-11-20"])
    out["peak-early"] = swung(out["2026-11-05"], peak=True)
    return out


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        yield b
        b.close()


def page_html(attrs: str = "") -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>marks</title>
<style>body{{margin:0;background:#F5F6F8}} .wrap{{padding:16px}} .panel{{background:#fff;padding:16px}}</style>
</head><body><div class="wrap"><section class="panel"><h3>The open cycle, entered October 19</h3>
<div data-nvda-mark="vol_arb" data-baked="2026-11-16"{attrs}><p>baked figures</p></div>
</section></div><script src="assets/nvda_marks.js" defer></script></body></html>"""


def open_page(browser, doc, width=1280, attrs="", fail=False, touch=False):
    ctx = browser.new_context(viewport={"width": width, "height": 900}, has_touch=touch)
    pg = ctx.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    body = {"index.html": (page_html(attrs), "text/html"), "assets/nvda_marks.js": (JS.read_text(), "text/javascript"),
            "data/vol_arb_nvda_mark.json": (json.dumps(doc), "application/json")}

    def serve(route):
        path = route.request.url[len(BASE):].split("?")[0]
        if fail and path.startswith("data/"):
            return route.abort()
        if path not in body:
            return route.fulfill(status=404, body="")
        text, ctype = body[path]
        route.fulfill(status=200, content_type=ctype, body=text)
    pg.route(BASE + "**", serve)
    pg.goto(BASE + "index.html")
    pg.wait_for_selector("[data-nm-state]")
    pg.wait_for_timeout(100)
    pg.errors = errors
    return ctx, pg


def money(x: float) -> str:
    """nvda_marks.js's money(): whole dollars, a sign, a word joiner after it."""
    n = int(math.floor(abs(x) + 0.5))
    return "$0" if n == 0 else (MINUS if x < 0 else "+") + JOIN + f"${n:,}"


# --------------------------------------------------------------------------- the chart's end labels
GEOMETRY = """() => {
  const svg = document.querySelector('.nm-chart svg');
  const box = t => { const b = t.getBBox(); return {x0: b.x, x1: b.x + b.width, y0: b.y, y1: b.y + b.height,
                                                     text: t.textContent, end: t.classList.contains('nm-end')}; };
  const lines = [...svg.querySelectorAll('polyline.nm-line')].map(p =>
    p.getAttribute('points').trim().split(/\\s+/).map(xy => xy.split(',').map(Number)));
  const plot = Math.max(...[...svg.querySelectorAll('line')].map(l => l.x2.baseVal.value - l.x1.baseVal.value));
  return {width: svg.viewBox.baseVal.width, height: svg.viewBox.baseVal.height, plot,
          texts: [...svg.querySelectorAll('text')].map(box), lines};
}"""


def crosses(seg, b, pad=1.0) -> bool:
    """Whether the segment ((x1, y1), (x2, y2)) meets the box b grown by pad (Liang-Barsky clipping)."""
    (x1, y1), (x2, y2) = seg
    x0, xa, y0, ya = b["x0"] - pad, b["x1"] + pad, b["y0"] - pad, b["y1"] + pad
    dx, dy, lo, hi = x2 - x1, y2 - y1, 0.0, 1.0
    for p, q in ((-dx, x1 - x0), (dx, xa - x1), (-dy, y1 - y0), (dy, ya - y1)):
        if p == 0:
            if q < 0:
                return False
            continue
        r = q / p
        if p < 0:
            lo = max(lo, r)
        else:
            hi = min(hi, r)
        if lo > hi:
            return False
    return True


def test_crosses_itself():
    b = {"x0": 0, "x1": 10, "y0": 0, "y1": 10}
    assert crosses(((-5, 5), (15, 5)), b) and crosses(((5, -5), (5, 5)), b) and crosses(((2, 2), (3, 3)), b)
    assert not crosses(((-5, 20), (15, 20)), b) and not crosses(((12, -5), (30, 40)), b)


@pytest.mark.parametrize("width", [1280, 390])
@pytest.mark.parametrize("as_of", ["2026-10-19", "2026-10-20", "2026-11-05", "2026-11-16", "2026-11-19", "2026-11-20",
                                   "swung-open", "swung-settled", "peak-early"])
def test_the_end_labels_never_cross_a_line_and_stay_in_the_chart(browser, docs, width, as_of):
    ctx, pg = open_page(browser, docs[as_of], width)
    try:
        g = pg.evaluate(GEOMETRY)
        assert pg.errors == []
    finally:
        ctx.close()
    ends = [t for t in g["texts"] if t["end"]]
    assert len(ends) == 2
    assert g["plot"] >= 0.5 * g["width"]                # the labels' room never takes the chart's place
    want = {money(docs[as_of]["pnl"][d]["mid"]) for d in ("put_rich", "call_rich")}
    assert {t["text"].split(" ")[-1] for t in ends} == want
    if width == 1280:
        assert sorted(t["text"].rsplit(" ", 1)[0] for t in ends) == ["Call side rich", "Put side rich"]
    for t in ends:
        assert 0 <= t["x0"] and t["x1"] <= g["width"] and 0 <= t["y0"] and t["y1"] <= g["height"], t
        for pts in g["lines"]:
            for seg in zip(pts, pts[1:]):
                assert not crosses(seg, t), (t, seg)
        for other in g["texts"]:                         # nor another label: the other side's, a rule's, an axis's
            if other is not t:
                assert not (t["x0"] < other["x1"] and other["x0"] < t["x1"] and t["y0"] < other["y1"]
                            and other["y0"] < t["y1"]), (t, other)


# --------------------------------------------------------------------------- the title and the fallback
def title_of(pg):
    return pg.evaluate("""() => { const t = document.querySelector('.nm-title');
      return {tag: t.tagName, text: t.textContent, asof: document.querySelector('.nm-asof').textContent,
              badge: (document.querySelector('.nm-badge') || {}).textContent || null}; }""")


@pytest.mark.parametrize("attrs, tag", [("", "H4"), (' data-heading-level="3"', "H3"),
                                        (' data-heading-level="9"', "H4"), (' data-heading-level="x"', "H4")])
def test_the_title_is_an_h4_unless_the_element_asks_for_another_level(browser, docs, attrs, tag):
    ctx, pg = open_page(browser, docs["2026-11-16"], attrs=attrs)
    try:
        got = title_of(pg)
    finally:
        ctx.close()
    assert got["tag"] == tag and got["text"] == "Latest mark, updated nightly"
    assert got["asof"] == "as of the November 16 close"


def test_a_settled_cycle_is_titled_the_final_mark(browser, docs):
    ctx, pg = open_page(browser, docs["2026-11-20"])
    try:
        got = title_of(pg)
    finally:
        ctx.close()
    assert got == {"tag": "H4", "text": "Final mark", "asof": "at the November 20 close", "badge": "Settled at expiry"}


def test_the_fallback_line_names_the_latest_mark(browser, docs):
    ctx, pg = open_page(browser, docs["2026-11-16"], fail=True)
    try:
        got = pg.evaluate("""() => ({state: document.querySelector('[data-nvda-mark]').getAttribute('data-nm-state'),
                                    fail: document.querySelector('.nm-fail').textContent,
                                    baked: !!document.querySelector('[data-nvda-mark] > p')})""")
    finally:
        ctx.close()
    assert got == {"state": "fallback", "baked": True,
                   "fail": "Couldn't load the latest mark; showing the figures as of November 16."}


# --------------------------------------------------------------------------- every close's figures, without a pointer
def test_every_closes_figures_are_in_a_table_the_chart_points_to(browser, docs):
    doc = docs["2026-11-16"]
    ctx, pg = open_page(browser, doc)
    try:
        got = pg.evaluate("""() => {
          const d = document.querySelector('details.nm-nums');
          return {summary: d && d.querySelector('summary').textContent,
                  head: d && [...d.querySelectorAll('thead th')].map(e => e.textContent),
                  rows: d && [...d.querySelectorAll('tbody tr')].map(r => [...r.cells].map(c => c.textContent)),
                  desc: document.querySelector('.nm-chart svg desc').textContent}; }""")
    finally:
        ctx.close()
    put, call = doc["marks"]["put_rich"]["mid"]["daily"], doc["marks"]["call_rich"]["mid"]["daily"]
    assert got["summary"] == f"Show the daily figures ({len(put)} closes)"
    assert got["head"] == ["Close", "NVDA", "Put side rich", "Call side rich"]
    assert got["rows"] == [[NM._day(p["date"]), f"${p['close']:,.2f}", money(p["pnl_to_date"]), money(c["pnl_to_date"])]
                           for p, c in zip(put, call)]
    assert "The table above has the figures" not in got["desc"]
    assert "daily table under the chart" in got["desc"]


TOUCH = """([type, x]) => {
  const rect = document.querySelector('.nm-chart svg rect[fill="transparent"]');
  const r = rect.getBoundingClientRect();
  const t = new Touch({identifier: 1, target: rect, clientX: r.left + x * r.width, clientY: r.top + r.height / 2});
  const lifted = type === 'touchend' || type === 'touchcancel';
  rect.dispatchEvent(new TouchEvent(type, {touches: lifted ? [] : [t], changedTouches: [t], bubbles: true,
                                           cancelable: true}));
  const tip = document.querySelector('.nm-tip');
  return {shown: getComputedStyle(tip).display !== 'none', day: (tip.querySelector('b') || {}).textContent || null};
}"""


def test_the_touch_tooltip_follows_a_drag_and_hides_when_the_finger_lifts(browser, docs):
    doc = docs["2026-11-16"]
    ctx, pg = open_page(browser, doc, width=390, touch=True)
    try:
        start = pg.evaluate(TOUCH, ["touchstart", 0.0])
        moved = pg.evaluate(TOUCH, ["touchmove", 0.6])
        lifted = pg.evaluate(TOUCH, ["touchend", 0.6])
        pg.evaluate(TOUCH, ["touchstart", 0.3])
        cancelled = pg.evaluate(TOUCH, ["touchcancel", 0.3])
    finally:
        ctx.close()
    assert start == {"shown": True, "day": "October 19"}
    assert moved["shown"] and moved["day"] not in (None, "October 19")
    assert not lifted["shown"] and not cancelled["shown"]


def test_the_keyboard_steps_through_the_closes(browser, docs):
    doc = docs["2026-11-16"]
    days = [NM._day(r["date"]) for r in doc["marks"]["put_rich"]["mid"]["daily"]]
    ctx, pg = open_page(browser, doc)
    tip = "() => { const t = document.querySelector('.nm-tip'); return getComputedStyle(t).display === 'none' ? " \
          "null : t.querySelector('b').textContent; }"
    try:
        svg = pg.locator(".nm-chart svg")
        assert svg.get_attribute("tabindex") == "0"
        svg.focus()
        seen = [pg.evaluate(tip)]
        for key in ("ArrowLeft", "ArrowLeft", "Home", "ArrowRight", "End", "Escape"):
            pg.keyboard.press(key)
            seen.append(pg.evaluate(tip))
        pg.keyboard.press("ArrowLeft")
        seen.append(pg.evaluate(tip))
        pg.keyboard.press("Tab")                                    # focus leaves: the tooltip goes
        seen.append(pg.evaluate(tip))
    finally:
        ctx.close()
    assert seen == [days[-1], days[-2], days[-3], days[0], days[1], days[-1], None, days[-2], None]


# --------------------------------------------------------------------------- the P&L table on a phone
TABLE = """() => {
  const w = document.querySelector('.nm-tblwrap'), cs = e => getComputedStyle(e).display;
  return {scroll: w.scrollWidth, client: w.clientWidth, page: document.documentElement.scrollWidth,
          inner: window.innerWidth,
          wide: [...w.querySelectorAll('.nm-wide')].map(cs),
          who: [...w.querySelectorAll('tr.nm-who-row')].map(r => cs(r) === 'none' ? null : r.innerText)};
}"""


@pytest.mark.parametrize("as_of", ["2026-11-16", "2026-11-20"])
def test_on_a_phone_the_pnl_table_fits_with_the_choosers_under_each_side(browser, docs, as_of):
    ctx, pg = open_page(browser, docs[as_of], width=390)
    try:
        got = pg.evaluate(TABLE)
    finally:
        ctx.close()
    assert got["scroll"] <= got["client"] and got["page"] <= got["inner"]
    assert set(got["wide"]) == {"none"} and len(got["wide"]) == 2 + 2 * 3       # the header's two and each row's
    put_side, call_side, idle = got["who"]
    assert "CHOSEN, F7 = 0" in put_side.upper() and "Rule 1" in put_side and "Rule 0" in put_side
    assert "Logistic, refit (p 0.44996)" in call_side and "Logistic, fit once (p 0.458)" in idle


def test_on_a_wide_screen_the_choosers_are_columns(browser, docs):
    ctx, pg = open_page(browser, docs["2026-11-16"])
    try:
        got = pg.evaluate(TABLE)
    finally:
        ctx.close()
    assert got["scroll"] <= got["client"]
    assert "none" not in got["wide"] and got["who"] == [None, None, None]
