#!/usr/bin/env python3
"""
Wait until no other workflow that logs in to LSEG is queued or running, before the nightly mark logs in.

The repo's workflows share one LSEG login, and one login holds one session at a time. The order workflows
(far-wings-orders.yml every weeknight; vol-arb-orders.yml and wheel-orders.yml from their Run buttons) and the plot
(lseg-plot.yml) log in taking over any other session. The nightly mark (nvda-marks.yml, nvda_marks.py) logs in with
lseg_login.open_session(take_over=False), so it can never sign one of them out: while another session holds the login
its login fails and nothing is marked. This script is the step before that login. It asks GitHub for the repository's
runs that are requested, queued, pending, waiting or in progress, and while any of them is a run of a workflow that
logs in to LSEG, it waits and asks again, up to a bound. GitHub can start a scheduled run hours late (2026-11-17:
11:23 and 15:29 UTC for 03:00 and 06:00 runs), so the schedules alone do not keep the two apart.

Exit 0 when no such run is queued or running (the mark may log in). Exit 1 when one still is at the bound, or when
GitHub could not be asked at all by then: the mark does not log in, the old mark stays, and the retry (or the next
night) marks. A workflow logs in to LSEG when its file passes an LSEG_ secret; the mark's own workflow is left out
(--self), and so is this run (GITHUB_RUN_ID).

    python3 lseg_wait.py --self nvda-marks.yml                # in the Action (GITHUB_REPOSITORY, GH_TOKEN)
    python3 lseg_wait.py --self nvda-marks.yml --max-wait-minutes 0   # ask once, never wait

The token is the job's own GITHUB_TOKEN (the workflow grants actions: read); it is sent only to GitHub's API and
never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
WORKFLOWS = ROOT / ".github" / "workflows"
# every status GitHub gives a run that has not finished (the API's status filter takes each of these)
ACTIVE = ("requested", "queued", "pending", "waiting", "in_progress")
LSEG_SECRET = re.compile(r"secrets\.LSEG_[A-Z_]+")


def lseg_workflows(folder: Path = WORKFLOWS, own: str = "") -> dict[str, str]:
    """The workflows that log in to LSEG ({file name: workflow name}): every file that passes an LSEG_ secret,
    except `own`."""
    found = {}
    for f in sorted([*folder.glob("*.yml"), *folder.glob("*.yaml")]):
        if f.name == own:
            continue
        text = f.read_text()
        if LSEG_SECRET.search(text):
            m = re.search(r"^name:\s*(.+?)\s*$", text, re.M)
            found[f.name] = m.group(1).strip("'\"") if m else f.name
    return found


def run_file(run: dict) -> str:
    """The workflow file a run belongs to: its path's last part (".github/workflows/x.yml@refs/heads/main" -> x.yml)."""
    return (run.get("path") or "").split("@", 1)[0].rsplit("/", 1)[-1]


def is_lseg_run(run: dict, workflows: dict[str, str], own_run_id: str = "") -> bool:
    """A run of a workflow that logs in to LSEG, and not this run. A run is matched by its workflow file, or, when
    GitHub leaves the path out, by its workflow's name."""
    if own_run_id and str(run.get("id")) == str(own_run_id):
        return False
    f = run_file(run)
    if f:
        return f in workflows
    return run.get("name") in set(workflows.values())


def active_lseg_runs(fetch: Callable[[str], list], workflows: dict[str, str], own_run_id: str = "") -> list[dict]:
    """Every run of an LSEG workflow that has not finished, once each. fetch(status) gives the runs in that status;
    an error from it is raised (the caller cannot say the login is free)."""
    seen, out = set(), []
    for status in ACTIVE:
        for run in fetch(status):
            if run.get("id") in seen or (run.get("status") or status) == "completed":
                continue
            seen.add(run.get("id"))
            if is_lseg_run(run, workflows, own_run_id):
                out.append(run)
    return out


def describe(run: dict) -> str:
    return (f"{run.get('name') or run_file(run)} run {run.get('id')} ({run.get('status')}, "
            f"started {run.get('run_started_at') or run.get('created_at') or '?'})")


def wait_until_free(fetch: Callable[[str], list], workflows: dict[str, str], max_wait: float, poll: float,
                    own_run_id: str = "", clock: Callable[[], float] = time.monotonic,
                    sleep: Callable[[float], None] = time.sleep, say: Callable[[str], None] = print) -> bool:
    """True as soon as no LSEG run is queued or running; False when one still is (or GitHub could not be asked) at
    max_wait seconds. Asks at once, then every `poll` seconds; never sleeps past the bound."""
    deadline = clock() + max_wait
    while True:
        try:
            busy = active_lseg_runs(fetch, workflows, own_run_id)
            problem = None
        except Exception as exc:                       # GitHub unreachable, rate limited, a bad answer
            busy, problem = None, f"could not ask GitHub for the runs ({type(exc).__name__}: {exc})"
        if busy == []:
            say("No other LSEG workflow is queued or running: the mark may log in.")
            return True
        left = deadline - clock()
        if busy:
            say("Waiting: " + "; ".join(describe(r) for r in busy) + f" ({max(left, 0) / 60:.0f} min left to wait)")
        else:
            say(f"Waiting: {problem} ({max(left, 0) / 60:.0f} min left to wait)")
        if left <= 0:
            say("Still " + ("; ".join(describe(r) for r in busy) if busy else problem)
                + f" after {max_wait / 60:.0f} minutes: the mark does not log in, nothing is marked "
                  "(the retry, or the next night, marks).")
            return False
        sleep(min(poll, left))


def github_fetch(repo: str, token: str, api: str = "https://api.github.com",
                 opener: Callable | None = None) -> Callable[[str], list]:
    """fetch(status): the repository's workflow runs in that status (up to 100, the most recent first)."""
    def fetch(status: str) -> list:
        open_url = opener or urllib.request.urlopen
        req = urllib.request.Request(
            f"{api.rstrip('/')}/repos/{repo}/actions/runs?status={status}&per_page=100",
            headers={"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
                     "User-Agent": "nightly-mark-lseg-wait", **({"Authorization": f"Bearer {token}"} if token else {})})
        with open_url(req, timeout=30) as resp:
            doc = json.loads(resp.read().decode("utf-8"))
        runs = doc.get("workflow_runs")
        if not isinstance(runs, list):
            raise ValueError("no workflow_runs in GitHub's answer")
        return runs
    return fetch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--self", dest="own", default="nvda-marks.yml",
                    help="this workflow's file, left out of the LSEG workflows (default nvda-marks.yml)")
    ap.add_argument("--max-wait-minutes", type=float, default=100.0,
                    help="how long to wait at most (default 100: the far-wing orders' 90-minute limit and a margin)")
    ap.add_argument("--poll-seconds", type=float, default=60.0, help="how often to ask GitHub again (default 60)")
    args = ap.parse_args(argv)

    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        sys.exit("GITHUB_REPOSITORY is not set: cannot ask GitHub which runs are going, so the mark does not log in.")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
    workflows = lseg_workflows(WORKFLOWS, args.own)
    if not workflows:
        sys.exit(f"No workflow in {WORKFLOWS.relative_to(ROOT)} passes an LSEG secret: nothing to wait for is "
                 "unexpected, so the mark does not log in.")
    print("Workflows that log in to LSEG: " + ", ".join(f"{n} ({f})" for f, n in workflows.items()))
    fetch = github_fetch(repo, token, os.environ.get("GITHUB_API_URL", "https://api.github.com"))
    ok = wait_until_free(fetch, workflows, args.max_wait_minutes * 60, args.poll_seconds,
                         own_run_id=os.environ.get("GITHUB_RUN_ID", ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
