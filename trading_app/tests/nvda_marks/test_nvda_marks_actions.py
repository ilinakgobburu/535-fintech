"""
The nightly-mark Action and the page script. The Action runs at 04:45 UTC Tue-Sat and with --retry at 09:15, on main
only, more than an hour from every orders run, in its own concurrency group (in the orders' group it could cancel a
pending orders run), logging in to LSEG without taking over another session (so it never signs out an orders run),
and only after waiting while any other LSEG workflow is queued or running (test_nvda_marks_wait.py),
with the far wings' Python and packages and LSEG secrets passed only as the mark step's environment, never echoed
and never under `set -x`; its commit step stages nothing but docs/data/*.json, refuses if anything else is staged,
and rebases and retries its push.
The commit step's own shell is run against a temporary repository to show that. docs/assets/nvda_marks.js fetches its
file relative to the page without the cache, keeps the baked figures when it cannot, and is plain ES5 (its behavior
in a browser: test_nvda_marks_page.py).
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest
import yaml

import nm_synth as S

WORKFLOWS = S.REPO / ".github" / "workflows"
JS = S.REPO / "docs" / "assets" / "nvda_marks.js"


def workflow(name):
    w = yaml.safe_load((WORKFLOWS / name).read_text())
    return w, w.get("on", w.get(True))           # YAML 1.1 reads the key `on` as True


def steps():
    w, _ = workflow("nvda-marks.yml")
    return w["jobs"]["mark"]["steps"]


def test_the_schedule_the_retry_and_the_run_button():
    w, on = workflow("nvda-marks.yml")
    assert on["schedule"] == [{"cron": "45 4 * * 2-6"}, {"cron": "15 9 * * 2-6"}]
    assert "workflow_dispatch" in on and on["workflow_dispatch"] is None
    mark = next(s for s in steps() if s.get("id") == "mark")
    # only the second schedule passes --retry; the first and the Run button mark afresh
    assert mark["run"] == "python nvda_marks.py ${{ github.event.schedule == '15 9 * * 2-6' && '--retry' || '' }}"
    assert on["schedule"][1]["cron"] in mark["run"]


def utc_minutes(cron: str) -> tuple[int, str]:
    minute, hour, dom, month, dow = cron.split()
    assert (dom, month) == ("*", "*") and minute.isdigit() and hour.isdigit()
    return int(hour) * 60 + int(minute), dow


def test_its_runs_keep_well_clear_of_every_order_run():
    """The marks keep more than an hour from each scheduled orders run, so the two rarely want the one LSEG login at
    once, and off the top of the hour, when GitHub delays most."""
    _, on = workflow("nvda-marks.yml")
    mine = [utc_minutes(s["cron"]) for s in on["schedule"]]
    theirs = []
    for f in sorted(WORKFLOWS.glob("*.yml")):
        w, o = workflow(f.name)
        if f.name != "nvda-marks.yml" and (w.get("concurrency") or {}).get("group") == "mkt-open-orders":
            theirs += [utc_minutes(s["cron"]) for s in (o or {}).get("schedule") or []]
    assert theirs                                          # the far-wing orders' 03:00 and 06:00
    for m, dow in mine:
        assert m % 60 != 0
        for t, their_dow in theirs:
            assert dow == their_dow and min(abs(m - t), 24 * 60 - abs(m - t)) > 60


def test_it_has_its_own_group_so_it_can_never_cancel_an_orders_run():
    """GitHub cancels a group's pending run when another is queued: in the orders' group a marks run queued while an
    orders run waited would cancel it. Its own group, which no other workflow uses, never waits for, cancels or is
    cancelled by an orders run."""
    w, _ = workflow("nvda-marks.yml")
    assert w["concurrency"] == {"group": "nvda-marks", "cancel-in-progress": False}
    for f in sorted(WORKFLOWS.glob("*.yml")):
        if f.name != "nvda-marks.yml":
            assert (workflow(f.name)[0].get("concurrency") or {}).get("group") != "nvda-marks", f.name
    for name in ("far-wings-orders.yml", "vol-arb-orders.yml", "wheel-orders.yml"):
        assert workflow(name)[0]["concurrency"] == {"group": "mkt-open-orders", "cancel-in-progress": False}
    text = (WORKFLOWS / "nvda-marks.yml").read_text()
    assert "open_session(take_over=False)" in text and "A mark never signs out an orders run." in text


def test_the_marks_log_in_without_taking_over_another_session(monkeypatch):
    """nvda_marks.py opens its LSEG session with take_over=False: an orders run holding the login is never signed
    out, the mark's login fails instead."""
    import importlib.util
    import sys
    import types
    calls = []
    fake = types.ModuleType("lseg_login")
    fake.open_session = lambda **kw: calls.append(kw) or "session"
    monkeypatch.setitem(sys.modules, "lseg_login", fake)
    spec = importlib.util.spec_from_file_location("nvda_marks_for_login", S.REPO / "nvda_marks.py")
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    assert mod.open_lseg() == "session" and calls == [{"take_over": False}]


@pytest.mark.parametrize("kwargs, signon", [({}, True), ({"take_over": True}, True), ({"take_over": False}, False)])
def test_the_shared_login_takes_over_only_when_asked(monkeypatch, kwargs, signon):
    """lseg_login.open_session passes take_over to LSEG as signon_control (default True, as the order scripts and the
    plot have always logged in); with False a failed login says another session may hold the login."""
    import importlib.util
    import sys
    import types
    seen = {}

    class Sess:
        open_state = "opened"

        def open(self):
            pass

    class Definition:
        def __init__(self, **kw):
            seen.update(kw)

        def get_session(self):
            return Sess()

    ld = types.ModuleType("lseg.data")
    ld.OpenState = types.SimpleNamespace(Opened="opened")
    ld.session = types.SimpleNamespace(platform=types.SimpleNamespace(Definition=Definition,
                                                                      GrantPassword=lambda **kw: kw),
                                       set_default=lambda s: None)
    lseg = types.ModuleType("lseg")
    lseg.data = ld
    monkeypatch.setitem(sys.modules, "lseg", lseg)
    monkeypatch.setitem(sys.modules, "lseg.data", ld)
    for k in ("LSEG_APP_KEY", "LSEG_USERNAME", "LSEG_PASSWORD"):
        monkeypatch.setenv(k, "x")
    spec = importlib.util.spec_from_file_location("lseg_login_for_test", S.REPO / "lseg_login.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.open_session(**kwargs)
    assert seen["signon_control"] is signon
    Sess.open_state = "closed"
    with pytest.raises(SystemExit) as exc:
        mod.open_session(**kwargs)
    assert ("another session holds this login" in str(exc.value)) is (not signon)


def test_it_uses_the_order_workflows_python_and_packages():
    w, _ = workflow("nvda-marks.yml")
    fw, _ = workflow("far-wings-orders.yml")
    assert w["permissions"] == {"contents": "write", "actions": "read"}      # actions: read for the wait step
    job = w["jobs"]["mark"]
    assert job["if"] == "github.ref == 'refs/heads/main'"
    fw_steps = fw["jobs"]["orders"]["steps"]
    py = next(s for s in job["steps"] if s.get("uses", "").startswith("actions/setup-python"))
    fw_py = next(s for s in fw_steps if s.get("uses", "").startswith("actions/setup-python"))
    assert py == fw_py
    assert next(s for s in job["steps"] if s.get("uses", "").startswith("actions/checkout")) == \
        next(s for s in fw_steps if s.get("uses", "").startswith("actions/checkout"))
    pip = next(s["run"] for s in job["steps"] if "pip install" in s.get("run", ""))
    assert pip == next(s["run"] for s in fw_steps if "pip install" in s.get("run", ""))


def test_secrets_are_only_the_mark_steps_environment_and_never_shown():
    text = (WORKFLOWS / "nvda-marks.yml").read_text()
    mark = next(s for s in steps() if s.get("id") == "mark")
    assert set(mark["env"]) == {"PYTHONWARNINGS", "LSEG_APP_KEY", "LSEG_USERNAME", "LSEG_PASSWORD"}
    for k in ("LSEG_APP_KEY", "LSEG_USERNAME", "LSEG_PASSWORD"):
        assert mark["env"][k] == "${{ secrets." + k + " }}"
    assert len(re.findall(r"secrets\.", text)) == 3
    assert "set -x" not in text and "set -o xtrace" not in text
    runs = "\n".join(s.get("run", "") for s in steps())
    assert not re.search(r"\b(printenv|env)\b\s*($|\|)", runs, re.M)
    assert not re.search(r"echo[^\n]*(LSEG|secrets|\$\{\{)", runs)
    assert "secrets" not in runs


def test_the_commit_step_stages_only_docs_data_and_rebases_its_push():
    commit = next(s for s in steps() if s.get("name") == "Commit docs/data")
    run = commit["run"]
    assert re.findall(r"git add[^;\n]*", run) == ["git add -- 'docs/data/*.json'"]
    assert "mkt/" not in run and "far_wings" not in run and "ledger" not in run
    assert "grep -v -E '^docs/data/[^/]+\\.json$'" in run
    assert "for attempt in 1 2 3; do" in run
    assert "git pull --rebase --autostash origin main && git push origin HEAD:main" in run
    assert commit["env"] == {"AS_OF": "${{ steps.mark.outputs.as_of }}"}
    # the order workflows' check of who writes the orders file must not count this one
    assert "mkt/open_orders.json" not in (WORKFLOWS / "nvda-marks.yml").read_text().split("run:", 1)[-1]


# --------------------------------------------------------------------------- the commit step's shell, for real
def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path):
    if not shutil.which("git") or not shutil.which("bash"):
        pytest.skip("git and bash are needed")
    remote, seed, work = tmp_path / "remote.git", tmp_path / "seed", tmp_path / "work"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    git(tmp_path, "init", "-q", "-b", "main", str(seed))
    for rel, text in (("docs/data/vol_arb_nvda_mark.json", '{"as_of": "2026-11-13"}\n'),
                      ("mkt/open_orders.json", "[]\n"), ("data/far_wings_ledger.json", "{}\n")):
        (seed / rel).parent.mkdir(parents=True, exist_ok=True)
        (seed / rel).write_text(text)
    git(seed, "-c", "user.name=t", "-c", "user.email=t@example.com", "add", "-A")
    git(seed, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "seed")
    git(seed, "remote", "add", "origin", str(remote))
    git(seed, "push", "-q", "origin", "main")
    git(tmp_path, "clone", "-q", str(remote), str(work))
    return remote, seed, work


def run_commit_step(work, as_of="2026-11-16"):
    script = next(s for s in steps() if s.get("name") == "Commit docs/data")["run"]
    script = script.replace("sleep 10", "sleep 0")
    return subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", script], cwd=work,
                          env={"AS_OF": as_of, "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
                               "HOME": str(work), "GIT_CONFIG_NOSYSTEM": "1"},
                          capture_output=True, text=True)


def test_the_commit_step_commits_the_mark_alone_and_rebases_onto_a_newer_main(repo):
    remote, seed, work = repo
    (work / "docs/data/vol_arb_nvda_mark.json").write_text('{"as_of": "2026-11-16"}\n')
    (work / "data/far_wings_ledger.json").write_text('{"stray": 1}\n')     # never committed by this step
    # someone else pushed in the meantime (the orders)
    (seed / "mkt/open_orders.json").write_text('[{"id": "nvda-fw-put"}]\n')
    git(seed, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-am", "orders")
    git(seed, "push", "-q", "origin", "main")
    got = run_commit_step(work)
    assert got.returncode == 0, got.stderr
    log = git(remote, "log", "--format=%s", "main")
    assert log.splitlines()[:2] == ["NVDA vol-arb mark at the 2026-11-16 close", "orders"]
    assert git(remote, "show", "--name-only", "--format=", "main").split() == ["docs/data/vol_arb_nvda_mark.json"]
    assert git(remote, "show", "main:mkt/open_orders.json") == '[{"id": "nvda-fw-put"}]\n'
    assert git(remote, "show", "main:data/far_wings_ledger.json") == "{}\n"
    assert (work / "data/far_wings_ledger.json").read_text() == '{"stray": 1}\n'   # set aside and put back


def test_the_commit_step_does_nothing_when_the_mark_is_unchanged(repo):
    remote, _, work = repo
    head = git(remote, "rev-parse", "main")
    got = run_commit_step(work)
    assert got.returncode == 0 and "docs/data is unchanged" in got.stdout
    assert git(remote, "rev-parse", "main") == head


def test_the_commit_step_refuses_when_anything_else_is_staged(repo):
    remote, _, work = repo
    head = git(remote, "rev-parse", "main")
    (work / "docs/data/vol_arb_nvda_mark.json").write_text('{"as_of": "2026-11-16"}\n')
    (work / "data/far_wings_ledger.json").write_text('{"x": 1}\n')
    git(work, "add", "data/far_wings_ledger.json")
    got = run_commit_step(work)
    assert got.returncode == 1 and "Files outside docs/data/*.json are staged" in got.stdout
    assert git(remote, "rev-parse", "main") == head


# --------------------------------------------------------------------------- the page script
def test_the_page_script_fetches_relative_to_the_page_without_the_cache():
    js = JS.read_text()
    assert 'vol_arb: {src: "data/vol_arb_nvda_mark.json"' in js
    assert 'fetch(r.src, {cache: "no-store"})' in js
    assert re.findall(r"https?://[^\s\"']+", js) == ["http://www.w3.org/2000/svg"]
    assert 'document.querySelectorAll("[data-nvda-mark]")' in js and "var RENDERERS = {" in js


def test_the_page_script_keeps_the_baked_figures_and_says_what_it_shows():
    js = JS.read_text()
    assert "Couldn't load the latest mark; showing the figures as of " in js and "tonight" not in js
    assert '"stale: last mark " + dayName(doc.as_of)' in js and "var STALE_DAYS = 4;" in js
    assert '"Settled at expiry"' in js and '"Latest mark, updated nightly"' in js and '"Final mark"' in js
    # the new content is built whole and checked before the baked content is replaced
    body = js[js.index("var done = function (doc) {"):]
    assert body.index("r.check(doc);") < body.index("r.render(doc, new Date(), headingLevel(el));") < \
        body.index('el.innerHTML = "";')
    # colors only from the page's variables, with fallbacks
    assert not re.search(r"(?<![,(])#[0-9A-Fa-f]{6}\b(?!\))", js.replace("var(--", ""))


def test_the_page_script_is_plain_es5():
    js = JS.read_text()
    code = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)
    assert "=>" not in code and "`" not in code and "?." not in code and "??" not in code
    assert not re.search(r"\b(let|const|class)\s", code)
    if shutil.which("node"):
        got = subprocess.run(["node", "-e", "new (require('vm').Script)(require('fs').readFileSync(process.argv[1],"
                              "'utf8'))", str(JS)], capture_output=True, text=True)
        assert got.returncode == 0, got.stderr
