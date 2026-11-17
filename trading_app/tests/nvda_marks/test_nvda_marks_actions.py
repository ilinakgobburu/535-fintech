"""
The nightly-mark Action and the page script. The Action runs at 03:30 UTC Tue-Sat and with --retry at 07:30, on main
only, in the order workflows' concurrency group (their pushes and its push never race), with the far wings' Python
and packages and LSEG secrets passed only as the mark step's environment, never echoed and never under `set -x`; its
commit step stages nothing but docs/data/*.json, refuses if anything else is staged, and rebases and retries its push.
The commit step's own shell is run against a temporary repository to show that. docs/assets/nvda_marks.js fetches its
file relative to the page without the cache, keeps the baked figures when it cannot, and is plain ES5.
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
    assert on["schedule"] == [{"cron": "30 3 * * 2-6"}, {"cron": "30 7 * * 2-6"}]
    assert "workflow_dispatch" in on and on["workflow_dispatch"] is None
    mark = next(s for s in steps() if s.get("id") == "mark")
    # only the second schedule passes --retry; the first and the Run button mark afresh
    assert mark["run"] == "python nvda_marks.py ${{ github.event.schedule == '30 7 * * 2-6' && '--retry' || '' }}"
    assert on["schedule"][1]["cron"] in mark["run"]


def test_it_shares_the_order_workflows_group_python_and_packages():
    w, _ = workflow("nvda-marks.yml")
    fw, _ = workflow("far-wings-orders.yml")
    assert w["concurrency"] == fw["concurrency"] == {"group": "mkt-open-orders", "cancel-in-progress": False}
    assert w["permissions"] == {"contents": "write"}
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
    assert "Couldn't load tonight's mark; showing the figures as of " in js
    assert '"stale: last mark " + dayName(doc.as_of)' in js and "var STALE_DAYS = 4;" in js
    assert '"Settled at expiry"' in js and '"Latest mark, updated nightly"' in js
    # the new content is built whole and checked before the baked content is replaced
    body = js[js.index("var done = function (doc) {"):]
    assert body.index("r.check(doc);") < body.index("r.render(doc, new Date());") < body.index('el.innerHTML = "";')
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
