"""
The two Actions: the vol-arb one runs every weeknight at 03:00 UTC (22:00 New York in winter) on main, with the
wheel's pinned packages plus scikit-learn, and commits mkt/open_orders.json with the rebase-and-retry step; the
wheel's keeps only its Run button, so one nightly job writes the orders file.
"""

from __future__ import annotations

import yaml

import va_live_synth as S

WORKFLOWS = S.REPO / ".github" / "workflows"


def workflow(name):
    w = yaml.safe_load((WORKFLOWS / name).read_text())
    return w, w.get("on", w.get(True))           # YAML 1.1 reads the key `on` as True


def test_the_vol_arb_action():
    w, on = workflow("vol-arb-orders.yml")
    # 22:00 New York in winter, and the retry at 01:00, before the open
    assert on["schedule"] == [{"cron": "0 3 * * 2-6"}, {"cron": "0 6 * * 2-6"}] and "workflow_dispatch" in on
    assert w["permissions"] == {"contents": "write"}
    job = w["jobs"]["orders"]
    assert job["if"] == "github.ref == 'refs/heads/main'"
    steps = {s["name"]: s for s in job["steps"]}
    py = next(s for s in job["steps"] if s.get("uses", "").startswith("actions/setup-python"))
    assert py["with"]["python-version"] == "3.11"
    pip = next(s["run"] for s in job["steps"] if "pip install" in s.get("run", ""))
    wheel_pip = next(s["run"] for s in workflow("wheel-orders.yml")[0]["jobs"]["orders"]["steps"]
                     if "pip install" in s.get("run", ""))
    assert set(pip.split()) == set(wheel_pip.split()) | {"scikit-learn==1.5.1"}
    run = next(s for s in job["steps"] if s.get("id") == "orders")
    # only the second schedule passes --retry; the first and the Run button decide afresh
    assert run["run"] == "python vol_arb_orders.py ${{ github.event.schedule == '0 6 * * 2-6' && '--retry' || '' }}"
    assert set(run["env"]) == {"PYTHONWARNINGS", "LSEG_APP_KEY", "LSEG_USERNAME", "LSEG_PASSWORD"}
    commit = steps["Commit mkt/open_orders.json and the ledger"]["run"]
    assert "git add mkt/open_orders.json data/vol_arb_ledger.json" in commit       # the ledger the book check reads
    assert "git pull --rebase origin main" in commit
    assert "git push origin HEAD:main" in commit and "steps.orders.outputs.for_day" in \
        steps["Commit mkt/open_orders.json and the ledger"]["env"]["FOR_DAY"]


def test_the_wheel_action_keeps_only_its_run_button():
    w, on = workflow("wheel-orders.yml")
    assert "schedule" not in on and "workflow_dispatch" in on
    scheduled = [n for n in ("vol-arb-orders.yml", "wheel-orders.yml") if "schedule" in workflow(n)[1]]
    assert scheduled == ["vol-arb-orders.yml"]          # one nightly writer of mkt/open_orders.json
