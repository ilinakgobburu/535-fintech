"""
The nightly mark on the published site: docs/vol_arb.html holds the data-nvda-mark="vol_arb" element (its own
section, near the top) and loads docs/assets/nvda_marks.js; docs/data/vol_arb_nvda_mark.json is committed, so the page
shows a mark before the first nightly run; the element's baked figures are the November 16 close's, to the cent, with
the data notes. In a browser (Playwright's Chromium; skipped when it is not installed) the real page, served as GitHub
Pages serves docs/, shows the committed file's figures, and keeps the baked ones with one line when the file cannot be
loaded, with no sideways scroll on a phone.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser

import pytest
import yaml

import nm_synth as S
from trading_app.lib import nvda_marks as NM

DOCS = S.REPO / "docs"
PAGE = DOCS / "vol_arb.html"
MARK = DOCS / "data" / "vol_arb_nvda_mark.json"
MINUS, JOIN = "−", "⁠"
BAKED = "2026-11-16"
# the November 16 close, to the cent: the figures the element shows until (or unless) the script loads the file
TARGETS = {"put_rich": {"mid": -995.20, "stress": -1274.22}, "call_rich": {"mid": 995.20, "stress": 716.17}}


def money(x: float) -> str:
    c = round(abs(x) * 100)
    return (MINUS if x < 0 else "+") + JOIN + f"${c // 100:,}.{c % 100:02d}"


def section(text: str) -> str:
    a, b = text.index("<!-- nvda-marks:start"), text.index("<!-- nvda-marks:end -->")
    assert a < b
    return text[a:b]


class Elements(HTMLParser):
    """Every element's tag, attributes and the ids of the elements it sits in."""

    def __init__(self):
        super().__init__()
        self.stack, self.found = [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.found.append((tag, a, [x[1].get("id") for x in self.stack if x[1].get("id")],
                           [x[1].get("class") for x in self.stack]))
        if tag not in ("meta", "link", "br", "img", "input", "hr"):
            self.stack.append((tag, a))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break


def elements():
    p = Elements()
    p.feed(PAGE.read_text())
    return p.found


# --------------------------------------------------------------------------- the page holds the element
def test_the_vol_arb_page_holds_the_mark_and_loads_the_script():
    found = elements()
    marks = [(a, ids, cls) for tag, a, ids, cls in found if "data-nvda-mark" in a]
    assert len(marks) == 1
    attrs, ids, classes = marks[0]
    assert attrs["data-nvda-mark"] == "vol_arb" and attrs["data-baked"] == BAKED
    assert ids == ["mark"] and "panel" in classes                 # in its own section, inside a panel headed h3
    scripts = [a for tag, a, _, _ in found if tag == "script" and a.get("src")]
    assert [s["src"] for s in scripts if "nvda_marks" in s["src"]] == ["assets/nvda_marks.js"]
    assert "defer" in next(s for s in scripts if s["src"] == "assets/nvda_marks.js")
    text = PAGE.read_text()
    assert text.index('data-nvda-mark="vol_arb"') < text.index('<script src="assets/nvda_marks.js" defer></script>')
    assert '<a href="#mark">Nightly mark</a>' in text
    # the page and the script agree on where the file is
    assert (DOCS / "assets" / "nvda_marks.js").exists()
    assert 'vol_arb: {src: "data/vol_arb_nvda_mark.json"' in (DOCS / "assets" / "nvda_marks.js").read_text()


def test_the_baked_figures_are_the_november_16_close_to_the_cent_with_the_notes():
    sec = html.unescape(section(PAGE.read_text()))
    flat = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", sec))
    for side in ("put_rich", "call_rich"):
        for fill in ("mid", "stress"):
            assert money(TARGETS[side][fill]) in sec
    assert sec.count("after $279.02 costs") == 2
    assert "$150.00 for the legs' bid and ask at entry and $129.02 for the hedge trades" in flat
    assert "Latest mark, as of the November 16 close" in flat and "NVDA closed at $156.14" in flat
    assert "p 0.44996" in flat and "p 0.45794" in flat and not re.search(r"\(p 0\.\d{3}\)", flat)
    # the data notes: the November 12 open is genuine (never called bad), the report's timing, the put's IV
    assert "November 12's open of $151.005 is genuine" in flat and "Jensen Huang's resignation" in flat
    assert not re.search(r"suspicious|bad print|bad open", flat, re.I)
    assert "the November 17 mark is the last before the report and the November 18 mark the first after it" in flat
    assert "its IV is carried from November 13" in flat
    assert "Not traded, and not a test result." in flat


# --------------------------------------------------------------------------- the file is committed
def test_the_mark_file_is_committed_and_is_the_configured_cycles():
    doc = json.loads(MARK.read_text())
    assert S.schema_problems(doc) == []
    spec = NM.cycle_spec(yaml.safe_load(S.CONFIG.read_text()))
    assert doc["cycle"]["entry"] == str(spec["entry"].date()) and doc["cycle"]["expiry"] == str(spec["expiry"].date())
    for leg, cp in (("put", NM.PUT), ("call", NM.CALL)):
        assert doc["cycle"]["legs"][leg]["ric"] == spec["legs"][cp]["ric"]
    assert doc["as_of"] >= BAKED
    if doc["as_of"] == BAKED:                                    # the file the page was baked from
        assert doc["pnl"] == TARGETS and doc["data_complete"] is True and doc["status"] == "open"
    if shutil.which("git") and (S.REPO / ".git").exists():
        got = subprocess.run(["git", "ls-files", "--error-unmatch", "docs/data/vol_arb_nvda_mark.json"],
                             cwd=S.REPO, capture_output=True, text=True)
        assert got.returncode == 0, "docs/data/vol_arb_nvda_mark.json is not in the repository"


# --------------------------------------------------------------------------- the real page in a browser
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


browser_only = pytest.mark.skipif(not _browser_ready(), reason="playwright chromium not installed")
BASE = "http://site.test/"
TYPES = {".html": "text/html", ".js": "text/javascript", ".json": "application/json", ".css": "text/css"}


def open_site(pw_browser, width, fail=False):
    ctx = pw_browser.new_context(viewport={"width": width, "height": 900})
    pg = ctx.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))

    def serve(route):
        url = route.request.url
        if not url.startswith(BASE):
            return route.abort()                               # the fonts and Plotly: not needed here
        path = DOCS / url[len(BASE):].split("?")[0]
        if (fail and "/data/" in str(path)) or not path.is_file():
            return route.abort() if fail else route.fulfill(status=404, body="")
        route.fulfill(status=200, content_type=TYPES.get(path.suffix, "application/octet-stream"),
                      body=path.read_bytes())
    pg.route("**/*", serve)
    pg.goto(BASE + "vol_arb.html")
    pg.wait_for_selector("[data-nm-state]")
    pg.wait_for_timeout(100)
    return ctx, pg, errors


STATE = """() => { const el = document.querySelector('[data-nvda-mark]');
  return {state: el.getAttribute('data-nm-state'), page: document.documentElement.scrollWidth,
          inner: window.innerWidth, title: (el.querySelector('.nm-title') || {}).textContent || null,
          fail: (el.querySelector('.nm-fail') || {}).textContent || null,
          cells: [...el.querySelectorAll('tbody tr')].slice(0, 4).map(r => [...r.cells].map(c => c.textContent))}; }"""


@browser_only
@pytest.mark.parametrize("width", [1280, 390])
def test_the_real_page_shows_the_committed_mark(width):
    from playwright.sync_api import sync_playwright
    doc = json.loads(MARK.read_text())
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        try:
            ctx, pg, errors = open_site(b, width)
            got = pg.evaluate(STATE)
            ctx.close()
        finally:
            b.close()
    assert errors == []
    assert got["state"] == "live"
    assert got["title"] == ("Final mark" if doc["status"] == "settled" else "Latest mark, updated nightly")
    assert got["page"] <= got["inner"]
    rows = [c for c in got["cells"] if c and c[0].startswith(("Put side rich", "Call side rich"))]
    for side, row in zip(("put_rich", "call_rich"), rows):
        assert row[1] == money(doc["pnl"][side]["mid"]) and row[2].startswith(money(doc["pnl"][side]["stress"]))


@browser_only
@pytest.mark.parametrize("width", [1280, 390])
def test_the_real_page_keeps_the_baked_figures_when_the_file_cannot_be_loaded(width):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        try:
            ctx, pg, errors = open_site(b, width, fail=True)
            got = pg.evaluate(STATE)
            ctx.close()
        finally:
            b.close()
    assert errors == []
    assert got["state"] == "fallback" and got["page"] <= got["inner"]
    assert got["fail"] == "Couldn't load the latest mark; showing the figures as of November 16."
    assert [r[1:] for r in got["cells"][:2]] == [
        [money(-995.20), money(-1274.22) + "after $279.02 costs"],
        [money(995.20), money(716.17) + "after $279.02 costs"]]
