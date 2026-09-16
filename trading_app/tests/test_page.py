"""
Tests for the published artefact: the payload, the formatters, the page's
prose, and the README.

This file exists because of how the mistakes in this project were actually
caught. Not one of them was caught by a test:

  * a double-quoted string spanning two lines in hw2_app.js -- which would
    have shipped a BLANK PAGE -- was caught by running `node --check` by hand;
  * chart titles clipped off the top of the canvas, because layout.title.y is
    normalised to the paper while legend.y is normalised to the plot area,
    were caught by looking at a screenshot;
  * a premium of exactly $3,566.50 printing as 3,566 in the build log and
    3,567 on the page (Python rounds halves to even, JavaScript rounds them
    away from zero) was caught by cross-checking the README by hand;
  * "n = 26361.0" appearing where a count belongs was caught by reading it;
  * prose asserting "Four rules" after a fifth and sixth were added, and a
    heading promising "two things" above seven of them, were caught by reading.

Every one of those is mechanically checkable, and none of it needs an LSEG
session -- the committed cache is enough. The browser-based checks skip
cleanly when Playwright or its browser is absent, so `pytest tests -q` stays
runnable anywhere.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT.parent / "docs"
PAGE = DOCS / "hw2.html"
APP_JS = ROOT / "scripts" / "hw2_app.js"
TEMPLATE = ROOT / "scripts" / "hw2_template.html"
README = ROOT.parent / "README.md"
CACHE = ROOT / "trading_app" / "data" / "covered_call_AAPL.pkl"


def _build_module():
    spec = importlib.util.spec_from_file_location("build_hw2", ROOT / "scripts" / "build_hw2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _build_module()


@pytest.fixture(scope="module")
def payload():
    if not CACHE.exists():
        pytest.skip(f"no cache at {CACHE}")
    return B.build_payload(CACHE)


# --------------------------------------------------------------------------
# formatters
# --------------------------------------------------------------------------

class TestMoneyFormatter:
    def test_rounds_halves_the_way_javascript_does(self):
        """
        The premium is exactly $3,566.50. Python's format() rounds halves to
        EVEN and gives 3,566; JavaScript's toLocaleString rounds them AWAY
        FROM ZERO and gives 3,567. The build log disagreed with the page it
        had just written by a dollar.
        """
        assert B.money(3566.50) == "3,567"
        assert B.money(2.5) == "3"
        assert B.money(3.5) == "4"
        assert f"{3566.50:,.0f}" == "3,566"      # the behaviour being avoided

    def test_matches_python_where_there_is_no_tie(self):
        for v in (0.0, 1.4, 1.6, 50_682.0, -3_144.0, 33_744.49):
            assert B.money(v) == f"{round(v):,}".replace("-", "-")

    def test_keeps_the_thousands_separator_and_decimals(self):
        assert B.money(50682.0) == "50,682"
        assert B.money(1.005, 2) == "1.01"


class TestJsonScalars:
    def test_counts_stay_integers(self):
        """'n = 26361.0' reads as a bug even when the number is right."""
        assert B.jnum(26361) == 26361
        assert isinstance(B.jnum(26361), int)
        assert not isinstance(B.jnum(26361), float)

    def test_booleans_are_not_integers(self):
        assert B.jnum(True) is True
        assert B.jnum(False) is False

    def test_non_finite_becomes_null_rather_than_breaking_the_dump(self):
        assert B.jnum(float("nan")) is None
        assert B.jnum(float("inf")) is None
        assert B.jnum(float("-inf")) is None

    def test_none_survives(self):
        assert B.jnum(None) is None

    def test_dates_become_strings(self):
        import datetime as dt
        assert B.jnum(dt.date(2026, 9, 4)) == "2026-09-04"


class TestSuiteFacts:
    def test_counts_are_derived_from_source_not_typed_in(self):
        f = B.suite_facts()
        assert f["functions"] == sum(f["per_file"].values())
        assert f["hw2_functions"] == sum(
            n for name, n in f["per_file"].items() if name in set(f["hw2_files"]))
        # Counted with a regex that matches BOTH styles in this repo: HW1's
        # files use module-level `def test_x():` at zero indent, HW2's use
        # class methods. A count keyed to one indentation silently returns
        # zero for the other, which is how this assertion first failed.
        for name, n in f["per_file"].items():
            src = (ROOT / "tests" / name).read_text(encoding="utf-8")
            direct = len(re.findall(r"^[ \t]*(?:async +)?def test_", src, re.M))
            assert n == direct, f"{name}: ast counted {n}, regex {direct}"
        assert f["hw2_functions"] > 100
        assert f["mutations"] > 25

    def test_every_test_file_is_discovered(self):
        """
        This function named ONE file, so the page under-reported itself the
        moment a second and third were added -- the same staleness it exists
        to prevent, one level up.
        """
        f = B.suite_facts()
        on_disk = {p.name for p in (ROOT / "tests").glob("test_*.py")}
        assert set(f["per_file"]) == on_disk

    def test_the_claim_is_a_floor_not_an_overstatement(self):
        """
        Parametrised functions expand into several cases, so pytest must
        collect at least as many as the page claims.
        """
        r = subprocess.run(
            [sys.executable, "-m", "pytest", str(ROOT / "tests"),
             "--collect-only", "-q", "--no-header"],
            capture_output=True, text=True, cwd=ROOT)
        m = re.search(r"(\d+) tests? collected", r.stdout)
        assert m, f"could not read the collection count:\n{r.stdout[-400:]}"
        assert int(m.group(1)) >= B.suite_facts()["functions"]

    def test_mutation_count_matches_the_harness(self):
        import ast
        tree = ast.parse((ROOT / "scripts" / "mutation_check.py").read_text(encoding="utf-8"))
        n = next(len(node.value.elts) for node in ast.walk(tree)
                 if isinstance(node, ast.Assign)
                 and any(getattr(t, "id", "") == "MUTATIONS" for t in node.targets))
        assert B.suite_facts()["mutations"] == n


# --------------------------------------------------------------------------
# the payload
# --------------------------------------------------------------------------

class TestPayload:
    def test_serialises_without_nan(self, payload):
        """allow_nan=False is the whole point; a NaN would make invalid JSON."""
        json.dumps(payload, allow_nan=False)

    def test_carries_every_section_the_page_draws(self, payload):
        for key in ("meta", "headline", "blotter", "cycles", "ledger", "buy_hold",
                    "scatter", "fit", "hour_sweep", "rule_sweep", "min_cash",
                    "ohlc", "theme"):
            assert key in payload, f"payload is missing {key}"

    def test_headline_agrees_with_the_last_ledger_row(self, payload):
        """The tiles and the ledger must not be able to disagree."""
        assert payload["headline"]["final_nav"] == pytest.approx(
            payload["ledger"]["nav"][-1])
        assert payload["headline"]["pnl"] == pytest.approx(
            payload["ledger"]["nav"][-1] - payload["meta"]["start_cash"])

    def test_blotter_cash_sums_to_the_ledger_cash(self, payload):
        total = sum(b["cash_delta"] for b in payload["blotter"])
        assert payload["ledger"]["cash"][-1] == pytest.approx(
            payload["meta"]["start_cash"] + total)

    def test_every_blotter_row_names_a_rule(self, payload):
        for b in payload["blotter"]:
            assert b["note"], f"blotter row with no rule: {b}"
            assert b["side"] in ("BUY", "SELL", "EXPIRE", "ASSIGN")

    def test_ledger_series_are_all_the_same_length(self, payload):
        L = payload["ledger"]
        n = len(L["ts"])
        for k, v in L.items():
            assert len(v) == n, f"ledger['{k}'] has {len(v)} of {n}"

    def test_reg_t_identities_hold_at_every_bar(self, payload):
        L, m = payload["ledger"], payload["meta"]
        for i in range(len(L["ts"])):
            lmv = L["lmv"][i]
            assert L["initial_margin"][i] == pytest.approx(m["initial_rate"] * lmv)
            assert L["maintenance_margin"][i] == pytest.approx(m["maint_rate"] * lmv)
            assert L["available_funds"][i] == pytest.approx(
                L["nav"][i] - L["initial_margin"][i])

    def test_no_impossible_positions(self, payload):
        L = payload["ledger"]
        for i in range(len(L["ts"])):
            assert L["shares"][i] >= 0
            if L["call_strike"][i] is not None:
                assert L["shares"][i] == payload["meta"]["shares"], \
                    "a short call with no stock behind it"
                assert L["option_mv"][i] <= 0, "a short call is a liability"

    def test_assignments_credit_the_strike(self, payload):
        """Each ASSIGN is followed by a stock SELL of 100 at that strike."""
        qty = payload["meta"]["shares"]
        rows = payload["blotter"]
        n = 0
        for i, b in enumerate(rows):
            if b["side"] == "ASSIGN":
                nxt = rows[i + 1]
                assert (nxt["kind"], nxt["side"]) == ("stock", "SELL")
                assert nxt["fill"] == pytest.approx(b["strike"])
                assert nxt["cash_delta"] == pytest.approx(qty * b["strike"])
                assert b["cash_delta"] == 0.0
                n += 1
        assert n == payload["headline"]["assignments"]

    def test_counts_are_ints_in_the_json(self, payload):
        assert isinstance(payload["fit"]["pooled"]["n"], int)
        assert isinstance(payload["meta"]["weeks"], int)
        for b in payload["fit"]["buckets"] + payload["fit"]["by_price"]:
            assert isinstance(b["n"], int)

    def test_scatter_draws_no_more_than_it_has(self, payload):
        s = payload["scatter"]
        assert s["drawn"] <= s["total"]
        assert len(s["mid"]) == s["drawn"] == len(s["trd"])

    def test_every_swept_rule_is_labelled(self, payload):
        for r in payload["rule_sweep"]:
            assert r["label"] and r["label"] != r["rule"], \
                f"{r['rule']} would render as a bare function key"

    def test_hour_sweep_covers_the_tradeable_session(self, payload):
        from trading_app.lib.covered_call import TRADEABLE_HOURS
        assert {h["order_hour"] for h in payload["hour_sweep"]} == set(TRADEABLE_HOURS)
        # only the price paid may differ; the calendar must not
        assert len({h["weeks_booked"] for h in payload["hour_sweep"]}) == 1


# --------------------------------------------------------------------------
# the built artefact
# --------------------------------------------------------------------------

NODE = shutil.which("node")


@pytest.fixture(scope="module")
def html():
    if not PAGE.exists():
        pytest.skip(f"no built page at {PAGE}; run scripts/build_hw2.py")
    return PAGE.read_text(encoding="utf-8")


class TestBuiltPage:
    def test_every_placeholder_was_substituted(self, html):
        """A surviving placeholder means a section of the page is simply gone."""
        for token in ("/*__PAYLOAD__*/null", "/*__STYLE__*/", "/*__APP__*/"):
            assert token not in html, f"{token} was never filled in"

    def test_carries_a_payload_a_stylesheet_and_the_app(self, html):
        assert "const DATA = {" in html
        assert "--panel:" in html, "the shared stylesheet did not get inlined"
        assert "Covered Call" in html

    def test_has_a_title(self, html):
        assert re.search(r"<title>[^<]+</title>", html)

    def test_the_stylesheet_has_one_source(self):
        """
        The HW2 template must not carry its own copy of the palette, or the
        two assignments drift into looking like different sites.
        """
        tpl = TEMPLATE.read_text(encoding="utf-8")
        assert "/*__STYLE__*/" in tpl
        assert "--panel:#141A2E" not in tpl, "palette copied instead of inlined"


@pytest.mark.skipif(NODE is None, reason="node not available")
class TestAppJsParses:
    def test_the_app_script_is_syntactically_valid(self):
        """
        This is the cheapest test in the file and it would have caught the
        worst bug: a double-quoted string spanning two lines, which throws at
        parse time and leaves every section of the page empty. A page that
        renders nothing still builds, uploads and returns HTTP 200.
        """
        r = subprocess.run([NODE, "--check", str(APP_JS)],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"hw2_app.js does not parse:\n{r.stderr}"

    def test_the_embedded_script_is_valid_too(self, html, tmp_path):
        """Parse what actually shipped, not just the source it came from."""
        m = re.findall(r"<script>(.*?)</script>", html, re.S)
        assert m, "no inline scripts in the page"
        for i, block in enumerate(m):
            f = tmp_path / f"block{i}.js"
            f.write_text(block, encoding="utf-8")
            r = subprocess.run([NODE, "--check", str(f)], capture_output=True, text=True)
            assert r.returncode == 0, f"inline script {i} does not parse:\n{r.stderr}"


class TestProseIsComputed:
    """
    The page states its own results in prose. HW1 shipped prose hardcoded to
    one ticker's numbers and it silently became false when the cache was
    rebuilt; this project's rule since then is that any figure which would
    change on a re-pull must be interpolated from the payload.

    Enforced by allowlist rather than by banning digits, because some literals
    genuinely are fixed: an expiry moves $0 whatever the data says. Anything
    new has to be justified here, which is the point -- it forces the question
    "would this go stale?" every time.
    """

    ALLOWED = {
        # structural truths, independent of any cache
        "$0",     # "an expiry moves $0"
        "$3", "$6",   # "a $3 option is not a $6 option" -- illustrative
        # axis-scale note inside a code comment, not prose
        "$15", "$50",
    }

    def _literals(self):
        src = APP_JS.read_text(encoding="utf-8")
        # drop ${...} interpolations: those are computed by definition
        stripped = re.sub(r"\$\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "<EXPR>", src)
        pat = r"\$\d[\d,]*(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?%"
        return sorted({m.group(0) for m in re.finditer(pat, stripped)})

    def test_no_unjustified_hardcoded_figures(self):
        found = set(self._literals())
        unexpected = found - self.ALLOWED
        assert not unexpected, (
            f"hardcoded figures in the page's prose: {sorted(unexpected)}. "
            f"If one would change on a re-pull it must come from the payload; "
            f"if it genuinely cannot change, add it to ALLOWED with a reason.")

    def test_the_allowlist_has_not_gone_stale(self):
        """An allowlist nobody prunes is just a disabled test."""
        found = set(self._literals())
        assert not (self.ALLOWED - found), (
            f"ALLOWED lists literals no longer in the source: "
            f"{sorted(self.ALLOWED - found)} -- prune them")

    def test_reg_t_rates_are_read_from_the_payload(self):
        src = APP_JS.read_text(encoding="utf-8")
        assert "M.initial_rate" in src and "M.maint_rate" in src, \
            "the margin legend should not assert 50%/25% independently"

    def test_counts_in_prose_come_from_the_payload(self):
        """
        Prose said "Four rules" after a fifth and sixth were added, and a
        heading promised "two things" above seven of them.
        """
        src = APP_JS.read_text(encoding="utf-8")
        for banned in ("Four rules", "to forty", "eight such mutations"):
            assert banned not in src, f"stale hardcoded count in prose: {banned!r}"


class TestReadmeMatchesThePage:
    """
    The README write-up quotes the page's figures. Nothing stopped the two
    disagreeing, and they did: the README said the minute pull was 591,034
    bars while the page said 589,081, and it said LOW_1 breached on 21.2%
    where the page rendered 21.3% -- 85/400 is exactly 21.25%, and Python
    rounds that tie to even while JavaScript rounds it away from zero.

    Checking the README against the PAYLOAD would have missed the second one
    entirely, because the payload is right and the disagreement is created by
    the formatter. So the comparison is against the RENDERED page, which is
    the thing being graded.
    """

    LIT = r"\$\d[\d,]*(?:\.\d+)?|\d+(?:\.\d+)?%|\b0\.\d{3,}\b"

    # Everything above this heading states results and must match the page.
    # Below it the write-up discusses bugs and history, and legitimately
    # quotes figures that are no longer true.
    AUDIT_HEADING = "### What a coverage audit turned up afterwards"

    # Figures the write-up quotes while DESCRIBING a past discrepancy rather
    # than asserting a current result, exempt by name and with a reason.
    #
    # The exemption is POSITIONAL, and that matters. A first version matched
    # by value alone, so exempting the narrative mention of "21.2%" also
    # exempted it as a claim -- and the mutation that flips the real figure
    # from 21.3% to 21.2% went from caught to missed without anyone touching
    # the code it was meant to protect. An exemption has to be confined to the
    # region that earned it.
    NARRATIVE = {
        "21.2%":  "the wrong value the README itself used to carry",
        "21.25%": "the exact tie, 85/400, that produced that disagreement",
        "21.3%":  "what the page rendered before the after-hours bar was removed",
        "$50,006.50": "Jun 29's close NAV as the off-by-one bug computed it",
        "$50,036": "Jun 29's corrected close NAV at the old $50,000 starting cash",
        "$50,000": "the flat starting cash the book used before it was sized to the first combo",
        "$0.31": "median miss of the after-hours 'close' against the official close",
        "$15.27": "worst miss of the after-hours 'close' against the official close",
        "$320.05": "the after-hours print that wrongly assigned the 2% rule's call",
        "$319.70": "the official close that shows that assignment never happened",
    }

    @staticmethod
    def _readme_section() -> str:
        text = README.read_text(encoding="utf-8")
        assert "## Assignment 1.2" in text
        return text.split("## Assignment 1.2")[1].split("## Assignment 1.1")[0]

    def _regions(self):
        body = self._readme_section()
        assert self.AUDIT_HEADING in body, "the audit section heading moved"
        claims, narrative = body.split(self.AUDIT_HEADING, 1)
        return claims, narrative

    @staticmethod
    def _figures(text):
        out, seen = [], set()
        for m in re.finditer(TestReadmeMatchesThePage.LIT, text):
            s = m.group(0).rstrip(",.")
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def test_every_figure_the_writeup_claims_appears_on_the_page(self, rendered_text):
        """
        No exemptions in this region. A result stated here must be a result
        the page shows.
        """
        claims, _ = self._regions()
        lits = self._figures(claims)
        assert len(lits) > 30, "the extractor found suspiciously few figures"
        missing = [s for s in lits if s not in rendered_text]
        assert not missing, (
            f"the README's results section quotes figures the page does not "
            f"show: {missing}. Either the README is stale or the page stopped "
            f"reporting them.")

    def test_figures_in_the_audit_section_are_shown_or_declared(self, rendered_text):
        _, narrative = self._regions()
        missing = [s for s in self._figures(narrative)
                   if s not in rendered_text and s not in self.NARRATIVE]
        assert not missing, (
            f"the audit section quotes figures that are neither on the page "
            f"nor declared narrative: {missing}. Add each to NARRATIVE with a "
            f"reason, or correct it.")

    def test_a_narrative_figure_may_not_also_be_used_as_a_claim(self):
        """
        The hole the mutation harness found: a value-only exemption disarmed
        the check everywhere in the file, including the sentence the test
        exists to police.
        """
        claims, _ = self._regions()
        for name in self.NARRATIVE:
            assert name not in claims, (
                f"{name!r} is exempt as narrative but appears in the results "
                f"section, where it would go unchecked")

    def test_narrative_exemptions_are_used_justified_and_absent(self, rendered_text):
        """
        Three ways an exemption list rots: an entry for a figure the README no
        longer mentions, an entry with no stated reason, and -- worst -- an
        entry for a figure the page DOES show, which hides a real disagreement
        behind a justification.
        """
        _, body = self._regions()
        for name, why in self.NARRATIVE.items():
            assert why, f"{name} is exempt with no reason given"
            assert name in body, (
                f"{name!r} is exempt but no longer appears in the README -- "
                f"prune it")
            assert name not in rendered_text, (
                f"{name!r} is exempt as narrative but the page does show it; "
                f"the exemption is hiding a real comparison")


# --------------------------------------------------------------------------
# it has to actually render
# --------------------------------------------------------------------------


def _open(pg, target, charts: int = 0):
    """
    Load a page and wait for what the test actually needs.

    These waited for `networkidle`, i.e. for the Plotly CDN to go quiet. On a
    slow connection that timed out at 60s -- a failure about the network, not
    the page, which surfaced as a red suite during the mutation run. A test
    that fails on bad Wi-Fi teaches people to ignore red, so the wait is now on
    the condition itself: the document loaded, and, where charts matter, that
    many Plotly charts drawn.
    """
    pg.goto(f"file://{Path(target).resolve()}", wait_until="load", timeout=90000)
    if charts:
        pg.wait_for_function(
            f"() => document.querySelectorAll('.js-plotly-plot').length >= {charts}"
            f" || window.__plotlyFailed",
            timeout=90000)
    pg.wait_for_timeout(800)          # let Plotly finish laying out titles

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


BROWSER = _browser_ready()
needs_browser = pytest.mark.skipif(
    not BROWSER, reason="playwright chromium not installed (python -m playwright install chromium)")


@pytest.fixture(scope="module")
def rendered():
    if not BROWSER:
        pytest.skip("playwright chromium not installed")
    if not PAGE.exists():
        pytest.skip("no built page")
    from playwright.sync_api import sync_playwright
    errors = []
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 1000})
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errors.append(f"PAGEERROR: {e}"))
        _open(pg, PAGE, charts=4)
        info = pg.evaluate("""() => {
          const ids = ['tiles','tbl-blotter','tbl-ledger','plot-nav','plot-margin',
            'plot-scatter','tbl-price','tbl-buckets','tbl-moves','plot-hours',
            'tbl-hours','tbl-rules','analysis','methods','feas','fit-pooled',
            'fit-note','hours-note','rules-note','blotter-note','data-state'];
          const lens = {};
          ids.forEach(i => { const e = document.getElementById(i);
            lens[i] = e ? e.innerHTML.trim().length : -1; });
          return {
            lens,
            charts: document.querySelectorAll('.js-plotly-plot').length,
            chartTitles: [...document.querySelectorAll('.gtitle')].map(e => e.textContent),
            // Geometry, not presence: a clipped title is still in the DOM.
            titleBoxes: [...document.querySelectorAll('.js-plotly-plot')].flatMap(p => {
              const pr = p.getBoundingClientRect();
              return [...p.querySelectorAll('.gtitle')].map(t => {
                const r = t.getBoundingClientRect();
                return { text: t.textContent, top: r.top, bottom: r.bottom,
                         h: r.height, plotTop: pr.top, plotBottom: pr.bottom };
              });
            }),
            text: document.body.innerText,
            scrollW: document.documentElement.scrollWidth,
            clientW: document.documentElement.clientWidth,
          };
        }""")
        b.close()
    return {"errors": errors, **info}


@pytest.fixture(scope="module")
def rendered_text(rendered):
    return rendered["text"]


@needs_browser
class TestRenders:
    def test_no_console_or_page_errors(self, rendered):
        assert rendered["errors"] == [], f"page threw: {rendered['errors'][:5]}"

    def test_no_section_is_empty(self, rendered):
        empty = {k: v for k, v in rendered["lens"].items() if v <= 0}
        assert not empty, f"sections missing or empty: {empty}"

    def test_all_four_charts_drew(self, rendered):
        assert rendered["charts"] == 4, f"only {rendered['charts']} charts rendered"

    def test_every_chart_has_a_visible_title(self, rendered):
        """
        layout.title.y is normalised to the PAPER while legend.y is normalised
        to the PLOT AREA. Setting both to 1.0 put every title's baseline at the
        very top of the canvas, where it was clipped -- the charts drew
        perfectly and were silently unlabelled.

        Counting <text class="gtitle"> nodes does NOT catch that: Plotly emits
        the element either way and merely positions it outside the drawing
        area. The first version of this test counted, passed against the bug,
        and was only found out by the mutation harness. So it measures
        GEOMETRY: the title box has to have real height and sit inside its own
        chart.
        """
        titles = [t for t in rendered["chartTitles"] if t and t.strip()]
        assert len(titles) == 4, f"expected 4 chart titles, saw {titles}"
        for t in rendered["titleBoxes"]:
            assert t["h"] > 4, f"title {t['text']!r} has no height (clipped)"
            assert t["top"] >= t["plotTop"] - 1, (
                f"title {t['text']!r} sits {t['plotTop'] - t['top']:.0f}px above "
                f"its chart and is clipped off the canvas")
            assert t["bottom"] <= t["plotBottom"] + 1, \
                f"title {t['text']!r} falls outside its chart"

    def test_no_undefined_or_nan_leaked_into_the_text(self, rendered):
        bad = re.findall(r"undefined|NaN|\[object \w+\]", rendered["text"])
        assert not bad, f"template leaked {sorted(set(bad))} into the page"

    def test_the_page_does_not_scroll_sideways(self, rendered):
        assert rendered["scrollW"] <= rendered["clientW"] + 1

    def test_pages_build_declares_no_live_data_connection(self, rendered):
        """Required: on github.io the Data section must say so."""
        assert "Data connection required" in rendered["text"]

    def test_the_headline_numbers_reach_the_page(self, rendered, payload):
        for v in (payload["headline"]["final_nav"], payload["headline"]["bh_final"]):
            assert B.money(v) in rendered["text"], f"{B.money(v)} not on the page"


@needs_browser
class TestRendersNarrow:
    @pytest.mark.parametrize("width", [360, 390, 768])
    def test_no_horizontal_overflow_at_phone_width(self, width):
        """
        A grid item defaults to min-width:auto and refuses to shrink below its
        content, so an overflow-x:auto table inside one pushes the whole PAGE
        sideways instead of scrolling itself. Only visible below ~420px.
        """
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            pg = b.new_page(viewport={"width": width, "height": 900})
            _open(pg, PAGE, charts=4)
            sw, cw = pg.evaluate(
                "() => [document.documentElement.scrollWidth, document.documentElement.clientWidth]")
            b.close()
        assert sw <= cw + 1, f"page scrolls sideways at {width}px ({sw} > {cw})"


class TestCommittedArtefactMatchesSource:
    """
    docs/hw2.html is the graded deliverable and it is committed, which means
    it can silently fall behind the source it was built from -- edit
    hw2_app.js, forget to rebuild, push, and the site shows the old page while
    the repository shows the new code. Nothing else in this suite would
    notice, because every other test reads one side or the other.
    """

    def test_the_build_is_deterministic(self, tmp_path):
        """
        Two builds of the same cache must be byte-identical, or the staleness
        check below is meaningless. The scatter is sampled with a fixed seed
        for exactly this reason.
        """
        a, b = tmp_path / "a.html", tmp_path / "b.html"
        for out in (a, b):
            r = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "build_hw2.py"), "--out", str(out)],
                capture_output=True, text=True, cwd=ROOT)
            assert r.returncode == 0, r.stderr
        assert a.read_bytes() == b.read_bytes(), "the build is not reproducible"

    def test_the_committed_page_is_up_to_date(self, tmp_path):
        if not CACHE.exists():
            pytest.skip("no cache")
        out = tmp_path / "rebuilt.html"
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_hw2.py"), "--out", str(out)],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr
        assert out.read_bytes() == PAGE.read_bytes(), (
            "docs/hw2.html is stale: rebuilding from the current source and "
            "cache produces a different file. Run scripts/build_hw2.py and "
            "commit the result.")


class TestTemplateAgreesWithTheCode:
    def test_the_reg_t_rates_in_the_equation_block_match_the_module(self):
        """
        The Reg T formulas are written out as prose in the template, with the
        rates spelled as literals. They are constants of the regulation rather
        than parameters of this book, so hardcoding them is defensible -- but
        only while they still match the constants the ledger actually applies.
        """
        from trading_app.lib.covered_call import INITIAL_RATE, MAINT_RATE
        tpl = TEMPLATE.read_text(encoding="utf-8")
        assert f"initial   = {INITIAL_RATE * 100:.0f}% × LMV" in tpl
        assert f"maint     = {MAINT_RATE * 100:.0f}% × LMV" in tpl

    def test_the_shares_per_contract_claim_matches_the_module(self):
        from trading_app.lib.covered_call import SHARES_PER_CONTRACT
        tpl = TEMPLATE.read_text(encoding="utf-8")
        assert f"{SHARES_PER_CONTRACT} shares" in tpl


@needs_browser
class TestAssignmentOnePagesStillRender:
    """
    HW2 reads its stylesheet out of HW1's template and added a nav link to it,
    so both HW1 pages were rebuilt. A finished assignment quietly breaking is
    worse than an unfinished one failing loudly.
    """

    @pytest.mark.parametrize("name", ["index.html", "ccj.html"])
    def test_page_renders_without_errors(self, name):
        target = DOCS / name
        if not target.exists():
            pytest.skip(f"no {name}")
        from playwright.sync_api import sync_playwright
        errors = []
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            pg = b.new_page(viewport={"width": 1280, "height": 1000})
            pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            pg.on("pageerror", lambda e: errors.append(f"PAGEERROR: {e}"))
            _open(pg, target, charts=1)
            info = pg.evaluate("""() => ({
              charts: document.querySelectorAll('.js-plotly-plot').length,
              text: document.body.innerText.length,
              junk: (document.body.innerText.match(/undefined|NaN|\\[object \\w+\\]/g) || []).length,
              hw2link: !!document.querySelector('nav a[href="hw2.html"]'),
            })""")
            b.close()
        assert errors == [], f"{name} threw: {errors[:3]}"
        assert info["charts"] > 0, f"{name} drew no charts"
        assert info["text"] > 5000, f"{name} rendered almost no text"
        assert info["junk"] == 0, f"{name} leaked undefined/NaN into the page"
        assert info["hw2link"], f"{name} lost its link to assignment 2"


# --------------------------------------------------------------------------
# requirements read off the professor's assignment page
# --------------------------------------------------------------------------
# Each of these was a gap found by reading assignment.html in full rather than
# the rubric summary: a Data PAGE rather than a section, assignment booked as
# two rows, the OCC symbol, the Reg T columns in the ledger itself.

DATA_PAGE = DOCS / "data.html"


class TestDataPage:
    def test_exists_and_says_data_connection_required(self):
        """'A Data page on github.io must show Data connection required.'"""
        assert DATA_PAGE.exists(), "docs/data.html is missing"
        assert "Data connection required" in DATA_PAGE.read_text(encoding="utf-8")

    def test_links_back_to_the_graded_book(self):
        assert 'href="hw2.html"' in DATA_PAGE.read_text(encoding="utf-8")

    def test_the_book_links_to_it(self):
        assert 'href="data.html"' in TEMPLATE.read_text(encoding="utf-8")

    def test_makes_no_attempt_to_reach_lseg(self):
        """'Live LSEG on Pages is a defect.' No script, so nothing to fetch."""
        html = DATA_PAGE.read_text(encoding="utf-8")
        assert "<script" not in html
        assert "fetch(" not in html

    def test_is_up_to_date_with_the_builder(self, payload):
        css = re.search(r"<style>(.*?)</style>",
                        (ROOT / "scripts" / "page_template.html").read_text(encoding="utf-8"),
                        re.S).group(1)
        assert B.render_data_page(css, payload["meta"]) == DATA_PAGE.read_text(encoding="utf-8"), \
            "docs/data.html is stale; run scripts/build_hw2.py"


class TestAssignmentSpecOnThePage:
    def test_the_blotter_has_a_notes_column(self, html):
        assert "Notes — the rule that fired" in html

    def test_every_option_row_carries_its_occ_symbol(self, payload):
        for b in payload["blotter"]:
            if b["kind"] == "call":
                assert re.fullmatch(r"[A-Z ]{6}\d{6}C\d{8}", b["occ"]), b

    def test_ledger_payload_carries_the_reg_t_columns_the_table_shows(self, payload):
        for k in ("initial_margin", "maintenance_margin", "available_funds"):
            assert k in payload["ledger"]

    def test_the_stock_leg_of_an_assignment_is_a_negative_quantity_on_the_page(self):
        """
        Stock rows store an unsigned 100 and take direction from the side, so
        a naive renderer prints the delivery as +100 -- a sale shown as a buy.
        """
        src = APP_JS.read_text(encoding="utf-8")
        assert 'r.kind === "stock" && r.side === "SELL") ? -r.qty' in src


@needs_browser
class TestNewSectionsRender:
    def test_rationale_contracts_and_reg_t_columns_are_present(self, rendered):
        t = rendered["text"]
        assert "Why AAPL?" in t
        assert "Why wait through expiry instead of buying the call back" in t
        assert "Contracts written" in t
        for col in ("Initial", "Maint", "Available"):
            assert col in t

    def test_the_assignment_delivery_shows_as_minus_one_hundred(self, rendered):
        assert "delivered against assignment" in rendered["text"]
        assert "-100" in rendered["text"]

    def test_the_data_page_renders(self):
        from playwright.sync_api import sync_playwright
        errors = []
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            pg = b.new_page(viewport={"width": 390, "height": 900})
            pg.on("pageerror", lambda e: errors.append(str(e)))
            _open(pg, DATA_PAGE)
            text = pg.evaluate("() => document.body.innerText")
            sw, cw = pg.evaluate("() => [document.documentElement.scrollWidth, document.documentElement.clientWidth]")
            b.close()
        assert not errors
        assert "Data connection required" in text
        assert sw <= cw + 1, "data page scrolls sideways at phone width"


# --------------------------------------------------------------------------
# starting cash: the cost of the first combo, as the professor's figure implies
# --------------------------------------------------------------------------
# The book started with a flat $50,000, which put Jun 29's NAV near $50k where
# the professor expected about $27k. His reference code uses the same NAV,
# initial and available-funds formulas; the gap was the idle cash. Starting
# cash is now derived: exactly enough to put the first covered call on.

class TestStartingCash:
    def test_is_the_cost_of_the_first_combo(self, payload):
        b = payload["blotter"]
        buy = next(e for e in b if e["kind"] == "stock" and e["side"] == "BUY")
        sell = next(e for e in b if e["kind"] == "call" and e["side"] == "SELL"
                    and e["ts"] == buy["ts"])
        assert payload["meta"]["start_cash"] == pytest.approx(
            100 * buy["fill"] - 100 * sell["fill"])
        assert payload["meta"]["start_cash"] != pytest.approx(50_000.0)

    def test_leaves_exactly_zero_cash_after_the_first_trade(self, payload):
        L = payload["ledger"]
        first = next(i for i, sh in enumerate(L["shares"]) if sh > 0)
        assert L["cash"][first] == pytest.approx(0.0, abs=0.005)

    def test_jun_29_close_nav_is_about_27k(self, payload):
        """
        The figure the professor gave. Starting cash 27,850.50, plus the $36
        the position gained by the close: 27,886.50.
        """
        L = payload["ledger"]
        i = L["ts"].index("2026-06-29 20:00:00")
        assert L["nav"][i] == pytest.approx(27_886.50)
        assert L["nav"][i] == pytest.approx(payload["meta"]["start_cash"] + 36.0)

    def test_the_book_never_becomes_unfundable(self, payload):
        """Less cash means margin, and Reg T must still permit every trade."""
        assert payload["headline"]["ever_infeasible"] is False
        assert payload["headline"]["min_available"] > 0
        assert payload["meta"]["start_cash"] > payload["min_cash"]["min_cash"]

    def test_trading_pnl_does_not_depend_on_starting_cash(self, payload):
        """
        Cash is not an input to any TRADE, so trading P&L is the +$681.82 it was
        at $50,000. What the smaller account adds is the cost of borrowing.
        """
        H = payload["headline"]
        assert H["pnl"] + H["margin_interest"] == pytest.approx(681.82, abs=0.01)
        assert H["margin_interest"] > 0, "the book borrows, so it must pay"


@needs_browser
class TestMarginIsDisclosed:
    def test_the_page_says_later_entries_are_bought_on_margin(self, rendered, payload):
        if payload["headline"]["min_cash"] >= 0:
            pytest.skip("the book never borrowed")
        t = rendered["text"]
        assert "bought on margin" in t
        assert "a year" in t and "actual/360" in t and "assumed broker rate" in t
        assert B.money(abs(payload["headline"]["min_cash"])) in t

    def test_the_bid_fill_cost_is_on_the_page(self, rendered, payload):
        t = rendered["text"]
        cost = payload["headline"]["bid_fill_cost"]
        assert f"${cost:,.2f}" in t
        assert "filled at the" in t and "bid" in t



class TestInterestAndFillSensitivity:
    def test_interest_matches_an_independent_recomputation(self, payload):
        """
        Rebuilt from the ledger's own cash column and dates, without calling
        the accrual code: each session's last row accrues its debit balance at
        the stated rate for the calendar days to the next session.
        """
        import datetime as _dt
        L, rate = payload["ledger"], payload["meta"]["margin_rate"]
        last = {}
        for i, ts in enumerate(L["ts"]):
            last[ts[:10]] = i
        dates = sorted(last)
        total = 0.0
        for d, n in zip(dates, dates[1:]):
            cash = L["cash"][last[d]]
            if cash < 0:
                days = (_dt.date.fromisoformat(n) - _dt.date.fromisoformat(d)).days
                total += -cash * rate * days / 360
        assert payload["headline"]["margin_interest"] == pytest.approx(total)
        assert L["accrued_interest"][-1] == pytest.approx(total)

    def test_the_rate_is_stated_not_hidden(self, payload):
        assert payload["meta"]["margin_rate"] == pytest.approx(0.07)

    def test_bid_fill_cost_is_half_the_spread_on_every_call_written(self, payload):
        booked = [c for c in payload["cycles"] if c["status"] in ("assigned", "expired")]
        assert payload["headline"]["bid_fill_cost"] == pytest.approx(
            sum((c["mid"] - c["bid"]) * 100 for c in booked))
        assert 0 < payload["headline"]["bid_fill_cost"] < payload["headline"]["premium"]



class TestBlotterReadsUnambiguously:
    """
    The professor read the blotter's ASSIGN row -- Cash Δ +$0.00 -- as the
    assignment proceeds going missing. They were on the next row, the stock
    SELL of 100 at the strike, immediately below his screenshot's crop. The
    arithmetic was right and the presentation was not, so the blotter now
    carries a running cash balance, tints the two rows of one assignment
    together, and the ASSIGN note points at the stock leg by amount.
    """

    def test_every_assign_note_names_the_credit_and_the_next_row(self, payload):
        qty = payload["meta"]["shares"]
        for b in payload["blotter"]:
            if b["side"] == "ASSIGN":
                assert "next row" in b["note"]
                assert f"{qty * b['strike']:,.2f}" in b["note"]

    @needs_browser
    def test_the_blotter_carries_a_running_cash_balance(self, rendered, payload):
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            br = pw.chromium.launch()
            pg = br.new_page(viewport={"width": 1700, "height": 1000})
            pg.goto(f"file://{PAGE.resolve()}", wait_until="load", timeout=90000)
            pg.wait_for_function(
                "() => document.querySelectorAll('#tbl-blotter tbody tr').length > 4", timeout=90000)
            hdr = pg.evaluate("() => [...document.querySelectorAll('#tbl-blotter thead th')].map(e => e.innerText)")
            cells = pg.evaluate("""() => [...document.querySelectorAll('#tbl-blotter tbody tr')]
                .map(r => ({cls: r.className, after: r.children[7].innerText}))""")
            br.close()
        assert "Cash after" in hdr
        # The running balance must equal start cash plus the deltas so far.
        # Compared as numbers: the page writes "−$300.00" with a typographic
        # minus and a dollar sign, which no Python formatter reproduces.
        def as_number(text):
            t = text.replace("\u2212", "-").replace("$", "").replace(",", "")
            return float(t)

        running = payload["meta"]["start_cash"]
        for row, b in zip(cells, payload["blotter"]):
            running += b["cash_delta"]
            assert as_number(row["after"]) == pytest.approx(running, abs=0.005), b["note"][:50]

    @needs_browser
    def test_the_two_rows_of_an_assignment_are_tinted_together(self, payload):
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            br = pw.chromium.launch()
            pg = br.new_page(viewport={"width": 1700, "height": 1000})
            pg.goto(f"file://{PAGE.resolve()}", wait_until="load", timeout=90000)
            pg.wait_for_function(
                "() => document.querySelectorAll('#tbl-blotter tbody tr').length > 4", timeout=90000)
            cls = pg.evaluate("""() => [...document.querySelectorAll('#tbl-blotter tbody tr')]
                .map(r => r.className)""")
            br.close()
        paired = [i for i, c in enumerate(cls) if "evt-pair" in c]
        assert len(paired) == 2 * payload["headline"]["assignments"]
        # they come in adjacent couples: ASSIGN then the stock SELL
        for a, b_ in zip(paired[::2], paired[1::2]):
            assert b_ == a + 1


class TestBookInvariants:
    """
    Whole-book properties a reader could challenge, asserted on the published
    payload rather than on a fixture. These were run as a one-off audit after
    the professor's blotter note; a one-off audit only protects the day it runs.
    """

    def test_every_written_call_has_exactly_one_terminal_row(self, payload):
        writes = [b for b in payload["blotter"] if b["kind"] == "call" and b["side"] == "SELL"]
        terms = [b for b in payload["blotter"] if b["side"] in ("EXPIRE", "ASSIGN")]
        assert len(writes) == len(terms) == payload["headline"]["weeks_booked"]
        for w, t in zip(writes, terms):
            assert w["instrument"] == t["instrument"], "a call settled against another contract"

    def test_the_short_call_is_always_covered_and_never_doubled(self, payload):
        L = payload["ledger"]
        for i in range(len(L["ts"])):
            if L["call_strike"][i] is not None:
                assert L["shares"][i] == payload["meta"]["shares"]
        assert min(L["shares"]) >= 0
        assert max(L["option_mv"]) <= 0

    def test_initial_is_twice_maintenance_whenever_long(self, payload):
        L = payload["ledger"]
        for i in range(len(L["ts"])):
            assert L["initial_margin"][i] == pytest.approx(2 * L["maintenance_margin"][i])

    def test_every_call_filled_at_the_mid_of_its_own_quote(self, payload):
        for c in payload["cycles"]:
            if c.get("mid") is not None:
                assert c["mid"] == pytest.approx((c["bid"] + c["ask"]) / 2)

    def test_no_strike_was_ever_sold_below_spot(self, payload):
        for c in payload["cycles"]:
            if c.get("strike") is not None:
                assert c["strike"] >= c["spot"] - 1e-9, c["iso"]

    def test_assigned_exactly_when_the_close_is_above_the_strike(self, payload):
        for c in payload["cycles"]:
            if c["status"] == "assigned":
                assert c["settle"] > c["strike"], c["iso"]
            elif c["status"] == "expired":
                assert c["settle"] <= c["strike"], c["iso"]

    def test_entries_are_on_the_booked_hour_and_terminals_at_the_close(self, payload):
        hh = f"{payload['meta']['order_hour']:02d}"
        for b in payload["blotter"]:
            if b["side"] == "BUY":
                assert b["ts"][11:13] == hh
            if b["side"] in ("EXPIRE", "ASSIGN"):
                assert b["ts"][11:16] == "20:00"

    def test_a_write_with_no_stock_leg_explains_itself(self, payload):
        """Weeks after an expiry write against shares already held."""
        by_ts = {}
        for b in payload["blotter"]:
            by_ts.setdefault(b["ts"], []).append(b)
        for b in payload["blotter"]:
            if b["kind"] == "call" and b["side"] == "SELL":
                bought = any(x["kind"] == "stock" and x["side"] == "BUY" for x in by_ts[b["ts"]])
                if not bought:
                    assert "Already long" in b["note"], b["ts"]
