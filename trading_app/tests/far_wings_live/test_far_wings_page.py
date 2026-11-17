"""
The far-wing page as published (docs/far_wings.html), checked against this repository:

  * it cites only what a reader of the repository can open: every commit it names is in this repository's history,
    and every repository path it names in <code> is a file (or folder) here, so no research record kept outside
    the repository is cited by commit, checksum or path;
  * it never calls a stock tracked by the course server when price_history.json's entry for it has no price the
    server wrote: those entries were added in this repository, and the page says they are not yet confirmed;
  * its NVDA section is a dated one-off snapshot, labeled not traded live and not updated nightly (no nightly
    renderer marks the far-wing cycle; tests/nvda_marks/test_nvda_marks_site.py checks every page's mark elements).

No LSEG, no network: the committed files only.
"""

from __future__ import annotations

import fnmatch
import html
import json
import re
import shutil
import subprocess

import pytest
import yaml

from fw_live_synth import CONFIG, REPO

PAGE = REPO / "docs" / "far_wings.html"
PRICE_HISTORY = REPO / "price_history.json"
GITHUB = "https://github.com/ilinakgobburu/535-fintech/blob/main/"
NAMED_ABSENT = {"mkt/fills.json"}           # named only to say it is not there yet
HEXLIKE = re.compile(r"(?<![0-9A-Za-z_])(?=[0-9a-f]*[0-9])(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}(?![0-9A-Za-z_])")
PATHLIKE = re.compile(r"^(?:[\w.*-]+/)+[\w.*-]*$|^[\w*-][\w.*-]*\.(?:py|ya?ml|json|md|csv|html|js|txt|pkl)$")


def shown() -> str:
    """The page without its scripts and styles: what a reader sees, with its tags."""
    return re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", PAGE.read_text(), flags=re.S)


def flat(text: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text)).split())


def coded() -> set[str]:
    return {html.unescape(x).strip() for x in re.findall(r"<code[^>]*>([^<]+)</code>", shown())}


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


needs_git = pytest.mark.skipif(not (shutil.which("git") and (REPO / ".git").exists()), reason="not a git checkout")


# --------------------------------------------------------------------------- citations a reader can open
@needs_git
def test_every_commit_the_page_names_is_in_this_repository():
    hashes = set(HEXLIKE.findall(flat(shown()))) | {c for c in coded() if re.fullmatch(r"[0-9a-f]{7,40}", c)}
    assert hashes, "the page names the commit its nightly-script files are read at"
    missing = [h for h in sorted(hashes)
               if git("cat-file", "-e", f"{h}^{{commit}}").returncode != 0
               or git("merge-base", "--is-ancestor", h, "HEAD").returncode != 0]
    assert missing == [], f"the page names commits not in this repository's history: {missing}"


@needs_git
def test_every_repository_path_the_page_names_is_here():
    tree = git("ls-files").stdout.splitlines()
    files = set(tree)
    dirs = {"/".join(f.split("/")[:i]) + "/" for f in tree for i in range(1, f.count("/") + 1)}
    names = {f.rsplit("/", 1)[-1] for f in tree}
    bad = []
    for c in sorted(coded()):
        if not PATHLIKE.match(c) or c in NAMED_ABSENT:
            continue
        ok = (any(fnmatch.fnmatchcase(f, c) for f in files) if "*" in c else c in dirs if c.endswith("/")
              else c in files if "/" in c else c in names)
        if not ok:
            bad.append(c)
    links = [u[len(GITHUB):] for u in map(html.unescape, re.findall(r'href="([^"]+)"', shown()))
             if u.startswith(GITHUB)]
    bad += [f"link {u}" for u in links if u not in files]
    assert bad == [], f"the page names files a reader of the repository cannot open: {bad}"
    if "mkt/fills.json" in coded() and "mkt/fills.json" not in files:
        assert "there is no mkt/fills.json" in flat(shown())


def test_the_page_cites_no_research_record_by_path():
    """The research behind sections 1 to 4 is described in words: no path to its configs, caches, notes or the
    page's own builder."""
    text = PAGE.read_text()
    for p in ("cache/", "explore/", "notes/", "site/", "config/vol_wings", "config/zone_", "config/vol_payoff",
              "sha256"):
        assert p not in text, p
    assert "kept outside this repository, so they are described here, not linked" in flat(shown())


# --------------------------------------------------------------------------- the stocks the server prices
def entries() -> tuple[list[str], list[str]]:
    """The configured names whose price_history.json entry is tracked, split into those with prices the server
    wrote and those without (entries added in this repository)."""
    ph = json.loads(PRICE_HISTORY.read_text())
    names = yaml.safe_load(CONFIG.read_text())["names"]
    priced, added = [], []
    for n in names:
        e = ph.get(str(n["conid"])) or {}
        if e.get("tracked") is True:
            (priced if e.get("price_history") else added).append(n["ticker"])
    return priced, added


def test_the_page_never_calls_the_added_entries_tracked_by_the_server():
    priced, added = entries()
    text = flat(shown())
    assert "NVDA" in priced
    if added:
        assert len(priced) == 1, "the page's wording names one priced stock; rebuild it from its builder"
        assert f"Only {priced[0]}'s entry there has prices from the server." in text
        assert f"So far the course server has written prices only for {priced[0]}" in text
        assert (f"The {len(added)} basket names' entries were added in this repository, and the server has not "
                "priced them yet, so it is not confirmed that the server will price them, resolve their orders or "
                "keep their entries" in text)
        assert f"{len(added)} not yet confirmed by the server" in text
    # the wording the review found: the server said to price, or to track, all of them
    assert not re.search(r"server prices the stocks \S+ tracks, which is now all", text)
    assert not re.search(r"with all \d+ tracked", text)
    assert not re.search(r"all \d+ stocks tracked", text)


# --------------------------------------------------------------------------- the NVDA section
def test_the_nvda_section_is_a_dated_snapshot_not_a_nightly_mark():
    page = PAGE.read_text()
    assert "data-nvda-mark" not in page and "nvda_marks.js" not in page
    sec = page[page.index('<section class="part" id="nvda">'):]
    sec = sec[:sec.index("</section>")]
    text = flat(sec)
    assert "NVDA's October 19 cycle, as of the November 16 close" in text
    assert "This month" not in text
    assert "Not traded live, and not updated nightly." in text
    assert "first live orders are for November 18, so this cycle was never sent to the course server" in text
    assert "they stay as they are through the November 20 expiry and after it" in text
