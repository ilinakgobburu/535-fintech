"""
The nightly mark never logs in to LSEG while another workflow that logs in to LSEG is queued or running
(lseg_wait.py, the step before the mark in .github/workflows/nvda-marks.yml), however late GitHub starts either run.
The orders and the plot log in taking over any other session; the mark logs in with take_over=False
(test_nvda_marks_actions.py), so even a mark that starts as an orders run starts can only fail, never sign it out.
Here: which workflows count, which runs count, the bounded wait (a fake clock: free at once, free after a while, busy
or unreachable to the bound, and never a sleep past it), GitHub's API as asked (a fake opener: the URL, the token
only in the header and never printed), and the workflow's wiring (the wait right before the login, long enough for a
whole far-wing orders run, inside the job's limit, under GitHub's rate limit).
"""

from __future__ import annotations

import importlib.util
import io
import json
import re

import pytest
import yaml

import nm_synth as S

WORKFLOWS = S.REPO / ".github" / "workflows"
LSEG_WORKFLOWS = {"far-wings-orders.yml": "Far-wing orders", "vol-arb-orders.yml": "Vol-arb orders",
                  "wheel-orders.yml": "Wheel orders", "lseg-plot.yml": "LSEG plot"}


@pytest.fixture(scope="module")
def W():
    spec = importlib.util.spec_from_file_location("lseg_wait_for_test", S.REPO / "lseg_wait.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(rid, file, status="in_progress", name=None):
    return {"id": rid, "path": f".github/workflows/{file}" if file else None, "status": status,
            "name": name or LSEG_WORKFLOWS.get(file, file), "run_started_at": "2026-11-18T03:41:00Z"}


class Clock:
    def __init__(self):
        self.t, self.slept = 0.0, []

    def __call__(self):
        return self.t

    def sleep(self, s):
        assert s > 0
        self.slept.append(s)
        self.t += s


def fetcher(answers):
    """fetch(status) from a list of answers, one per poll: each a list of runs (their own status picks the
    filter they come back under) or an exception."""
    asked = []

    def fetch(status):
        poll = sum(1 for s in asked if s == "requested") - (status != "requested")
        asked.append(status)
        answer = answers[min(poll, len(answers) - 1)]
        if isinstance(answer, Exception):
            raise answer
        return [r for r in answer if r["status"] == status]
    fetch.asked = asked
    return fetch


# --------------------------------------------------------------------------- which workflows and runs count
def test_the_lseg_workflows_are_the_order_workflows_and_the_plot_but_not_the_mark(W):
    assert W.lseg_workflows(WORKFLOWS, "nvda-marks.yml") == LSEG_WORKFLOWS
    assert "nvda-marks.yml" in W.lseg_workflows(WORKFLOWS)        # it passes LSEG secrets too, left out by --self
    # every workflow that passes an LSEG secret is one of them: none can be added without the wait seeing it
    for f in WORKFLOWS.glob("*.y*ml"):
        if "secrets.LSEG_" in f.read_text() and f.name != "nvda-marks.yml":
            assert f.name in LSEG_WORKFLOWS


def test_a_workflow_without_lseg_secrets_does_not_count(W, tmp_path):
    (tmp_path / "a.yml").write_text("name: Prices\njobs: {}\n")
    (tmp_path / "b.yml").write_text('name: "Orders"\nenv:\n  K: ${{ secrets.LSEG_APP_KEY }}\n')
    (tmp_path / "c.yaml").write_text("env:\n  K: ${{ secrets.LSEG_PASSWORD }}\n")
    assert W.lseg_workflows(tmp_path) == {"b.yml": "Orders", "c.yaml": "c.yaml"}
    assert W.lseg_workflows(tmp_path, "b.yml") == {"c.yaml": "c.yaml"}


def test_which_runs_count(W):
    wf = W.lseg_workflows(WORKFLOWS, "nvda-marks.yml")
    assert W.is_lseg_run(run(1, "far-wings-orders.yml"), wf)
    assert W.is_lseg_run({**run(1, None), "path": ".github/workflows/lseg-plot.yml@refs/heads/main"}, wf)
    assert W.is_lseg_run(run(1, None, name="Far-wing orders"), wf)          # no path: matched by the name
    assert not W.is_lseg_run(run(1, "nvda-marks.yml", name="NVDA marks"), wf)
    assert not W.is_lseg_run({**run(1, None), "path": "dynamic/pages/pages-build-deployment",
                              "name": "pages build and deployment"}, wf)
    assert not W.is_lseg_run(run(1, None, name="NVDA marks"), wf)
    assert not W.is_lseg_run(run(7, "far-wings-orders.yml"), wf, own_run_id="7")   # this run itself
    assert W.is_lseg_run(run(7, "far-wings-orders.yml"), wf, own_run_id="8")


def test_every_unfinished_status_is_asked_and_each_run_counted_once(W):
    wf = W.lseg_workflows(WORKFLOWS, "nvda-marks.yml")
    assert set(W.ACTIVE) == {"requested", "queued", "pending", "waiting", "in_progress"}
    runs = [run(1, "far-wings-orders.yml", "queued"), run(2, "wheel-orders.yml", "waiting"),
            run(3, "nvda-marks.yml", "in_progress"), run(4, "lseg-plot.yml", "pending"),
            run(5, "vol-arb-orders.yml", "requested")]
    fetch = fetcher([runs + [run(1, "far-wings-orders.yml", "in_progress")]])
    got = W.active_lseg_runs(fetch, wf)
    assert sorted(r["id"] for r in got) == [1, 2, 4, 5] and len(got) == 4
    assert fetch.asked == list(W.ACTIVE)
    # a status of "completed" in the answer (the run ended between the filter and the answer) never counts
    done = {**run(9, "far-wings-orders.yml"), "status": "completed"}
    assert W.active_lseg_runs(lambda status: [done], wf) == []
    with pytest.raises(OSError):
        W.active_lseg_runs(fetcher([OSError("down")]), wf)


# --------------------------------------------------------------------------- the bounded wait
def wait(W, answers, max_wait=600, poll=60):
    clock, said = Clock(), []
    fetch = fetcher(answers)
    ok = W.wait_until_free(fetch, W.lseg_workflows(WORKFLOWS, "nvda-marks.yml"), max_wait, poll,
                           clock=clock, sleep=clock.sleep, say=said.append)
    return ok, clock, said, fetch


def test_free_at_once_logs_in_without_waiting(W):
    ok, clock, said, _ = wait(W, [[run(3, "nvda-marks.yml")]])
    assert ok and clock.slept == [] and "the mark may log in" in said[-1]


def test_an_orders_run_in_progress_is_waited_for_until_it_ends(W):
    orders = run(11, "far-wings-orders.yml")
    ok, clock, said, fetch = wait(W, [[orders], [orders], [{**orders, "status": "in_progress"}], []])
    assert ok and clock.slept == [60, 60, 60] and clock.t == 180
    assert "Far-wing orders run 11 (in_progress" in said[0] and said[-1].startswith("No other LSEG workflow")
    assert fetch.asked.count("in_progress") == 4


def test_a_queued_orders_run_is_waited_for_too(W):
    ok, clock, _, _ = wait(W, [[run(12, "far-wings-orders.yml", "queued")], []])
    assert ok and clock.slept == [60]


def test_still_running_at_the_bound_never_logs_in_and_never_sleeps_past_it(W):
    ok, clock, said, fetch = wait(W, [[run(13, "far-wings-orders.yml")]], max_wait=150, poll=60)
    assert ok is False
    assert clock.slept == [60, 60, 30] and clock.t == 150         # asked at 0, 60, 120 and at the bound
    assert fetch.asked.count("in_progress") == 4
    assert said[-1].startswith("Still Far-wing orders run 13") and "the mark does not log in" in said[-1]


def test_github_unreachable_fails_closed_and_recovers_when_it_answers(W):
    ok, clock, said, _ = wait(W, [OSError("connection refused")], max_wait=120)
    assert ok is False and clock.t == 120 and "could not ask GitHub" in said[-1]
    ok, clock, _, _ = wait(W, [OSError("rate limited"), []], max_wait=120)
    assert ok and clock.slept == [60]


def test_no_wait_asks_once(W):
    ok, clock, _, fetch = wait(W, [[run(14, "lseg-plot.yml")]], max_wait=0)
    assert ok is False and clock.slept == [] and fetch.asked == list(W.ACTIVE)


def test_a_late_orders_run_after_a_late_mark_is_still_waited_for(W):
    """GitHub starts the mark hours late, just after the orders' retry started: the mark waits for that run."""
    retry = {**run(15, "far-wings-orders.yml"), "run_started_at": "2026-11-18T09:02:00Z", "event": "schedule"}
    ok, clock, said, _ = wait(W, [[retry]] * 30 + [[]], max_wait=100 * 60)
    assert ok and clock.t == 30 * 60 and "Far-wing orders run 15" in said[0]


# --------------------------------------------------------------------------- GitHub's API as asked
class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_the_api_request_and_the_token(W):
    seen = []

    def opener(req, timeout):
        seen.append((req.full_url, dict(req.header_items()), timeout))
        return Resp(json.dumps({"total_count": 1, "workflow_runs": [run(21, "far-wings-orders.yml")]}).encode())
    fetch = W.github_fetch("someone/repo", "tok-123", "https://api.github.com/", opener=opener)
    assert fetch("in_progress") == [run(21, "far-wings-orders.yml")]
    url, headers, timeout = seen[0]
    assert url == "https://api.github.com/repos/someone/repo/actions/runs?status=in_progress&per_page=100"
    assert headers["Authorization"] == "Bearer tok-123" and headers["Accept"] == "application/vnd.github+json"
    assert timeout == 30
    with pytest.raises(ValueError):
        W.github_fetch("o/r", "", opener=lambda req, timeout: Resp(b'{"message": "Not Found"}'))("queued")


def test_main_waits_then_lets_the_mark_log_in_and_never_prints_the_token(W, monkeypatch, capsys):
    answers = iter([[run(31, "far-wings-orders.yml")]] * len(W.ACTIVE) + [[]] * (len(W.ACTIVE) * 2))
    urls = []

    def opener(req, timeout):
        urls.append(req.full_url)
        status = re.search(r"status=(\w+)", req.full_url).group(1)
        runs = [r for r in next(answers) if r["status"] == status]
        return Resp(json.dumps({"workflow_runs": runs}).encode())
    monkeypatch.setattr(W.urllib.request, "urlopen", opener)
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    monkeypatch.setenv("GITHUB_REPOSITORY", "someone/repo")
    monkeypatch.setenv("GH_TOKEN", "secret-token-value")
    monkeypatch.setenv("GITHUB_RUN_ID", "99")
    assert W.main(["--self", "nvda-marks.yml", "--max-wait-minutes", "5", "--poll-seconds", "1"]) == 0
    out = capsys.readouterr().out
    assert "Far-wing orders run 31" in out and "the mark may log in" in out
    assert "secret-token-value" not in out
    assert all(u.startswith("https://api.github.com/repos/someone/repo/actions/runs?status=") for u in urls)


def test_main_refuses_without_the_repository_or_when_busy_at_the_bound(W, monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    with pytest.raises(SystemExit) as exc:
        W.main([])
    assert "the mark does not log in" in str(exc.value)

    def busy(req, timeout):
        status = re.search(r"status=(\w+)", req.full_url).group(1)
        return Resp(json.dumps({"workflow_runs": [run(32, "far-wings-orders.yml", status)]}).encode())
    monkeypatch.setattr(W.urllib.request, "urlopen", busy)
    monkeypatch.setenv("GITHUB_REPOSITORY", "someone/repo")
    assert W.main(["--max-wait-minutes", "0"]) == 1
    assert "the mark does not log in" in capsys.readouterr().out


def test_main_never_waits_for_its_own_run(W, monkeypatch, capsys):
    """This run is left out by its id too, so a mark never waits on itself even under another workflow file name."""
    def own(req, timeout):
        status = re.search(r"status=(\w+)", req.full_url).group(1)
        return Resp(json.dumps({"workflow_runs": [run(77, "nvda-marks.yml", status)]}).encode())
    monkeypatch.setattr(W.urllib.request, "urlopen", own)
    monkeypatch.setenv("GITHUB_REPOSITORY", "someone/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "77")
    assert W.main(["--self", "renamed-marks.yml", "--max-wait-minutes", "0"]) == 0
    monkeypatch.setenv("GITHUB_RUN_ID", "78")
    assert W.main(["--self", "renamed-marks.yml", "--max-wait-minutes", "0"]) == 1
    capsys.readouterr()


# --------------------------------------------------------------------------- the workflow's wiring
def marks_job():
    return yaml.safe_load((WORKFLOWS / "nvda-marks.yml").read_text())


def test_the_wait_is_the_step_right_before_the_login_and_the_mark_runs_only_after_it(W):
    w = marks_job()
    steps = w["jobs"]["mark"]["steps"]
    i = next(k for k, s in enumerate(steps) if s.get("id") == "mark")
    wait_step = steps[i - 1]
    assert wait_step["run"].startswith("python lseg_wait.py --self nvda-marks.yml ")
    assert wait_step["env"] == {"GH_TOKEN": "${{ github.token }}"}
    # a failed wait stops the job: neither step may carry on past it
    for s in (wait_step, steps[i]):
        assert "continue-on-error" not in s and "if" not in s
    assert not any("lseg_wait" in s.get("run", "") or "nvda_marks.py" in s.get("run", "") for s in steps[:i - 1])
    assert w["permissions"] == {"contents": "write", "actions": "read"}


def test_the_wait_covers_a_whole_orders_run_fits_the_job_and_stays_under_the_rate_limit(W):
    w = marks_job()
    steps = w["jobs"]["mark"]["steps"]
    cmd = next(s["run"] for s in steps if "lseg_wait.py" in s.get("run", ""))
    max_wait = float(re.search(r"--max-wait-minutes (\d+(?:\.\d+)?)", cmd).group(1))
    poll = float(re.search(r"--poll-seconds (\d+(?:\.\d+)?)", cmd).group(1))
    fw = yaml.safe_load((WORKFLOWS / "far-wings-orders.yml").read_text())
    assert max_wait >= fw["jobs"]["orders"]["timeout-minutes"]           # a whole orders run, to its limit
    assert w["jobs"]["mark"]["timeout-minutes"] >= max_wait + 15          # and the mark after it
    assert len(W.ACTIVE) * 3600 / poll <= 500                             # GITHUB_TOKEN: 1,000 requests an hour
