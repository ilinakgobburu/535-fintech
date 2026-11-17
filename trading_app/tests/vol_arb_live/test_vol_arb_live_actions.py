"""
The order Actions: the vol-arb one keeps only its Run button (the far wings' Action writes mkt/open_orders.json every
night now), which does nothing unless its overwrite box is ticked; it runs on main with the wheel's pinned packages
plus scikit-learn and commits mkt/open_orders.json with the rebase-and-retry step. The wheel's keeps only its Run
button too, so one nightly job writes the orders file.
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
    # retired from the schedule; its Run button needs its box, which passes the flag past the far wings' claim
    assert "schedule" not in on and "workflow_dispatch" in on
    box = on["workflow_dispatch"]["inputs"]["overwrite_far_wings"]
    assert box["type"] == "boolean" and box["default"] is False
    assert w["permissions"] == {"contents": "write"}
    job = w["jobs"]["orders"]
    assert job["if"] == "github.ref == 'refs/heads/main' && inputs.overwrite_far_wings"
    steps = {s["name"]: s for s in job["steps"]}
    py = next(s for s in job["steps"] if s.get("uses", "").startswith("actions/setup-python"))
    assert py["with"]["python-version"] == "3.11"
    pip = next(s["run"] for s in job["steps"] if "pip install" in s.get("run", ""))
    wheel_pip = next(s["run"] for s in workflow("wheel-orders.yml")[0]["jobs"]["orders"]["steps"]
                     if "pip install" in s.get("run", ""))
    assert set(pip.split()) == set(wheel_pip.split()) | {"scikit-learn==1.5.1"}
    run = next(s for s in job["steps"] if s.get("id") == "orders")
    assert run["run"] == "python vol_arb_orders.py --overwrite-far-wings"
    assert set(run["env"]) == {"PYTHONWARNINGS", "LSEG_APP_KEY", "LSEG_USERNAME", "LSEG_PASSWORD"}
    commit = steps["Commit mkt/open_orders.json and the ledger"]["run"]
    assert "git add mkt/open_orders.json data/vol_arb_ledger.json" in commit       # the ledger the book check reads
    assert "git pull --rebase origin main" in commit
    assert "git push origin HEAD:main" in commit and "steps.orders.outputs.for_day" in \
        steps["Commit mkt/open_orders.json and the ledger"]["env"]["FOR_DAY"]


def test_the_wheel_action_keeps_only_its_run_button():
    w, on = workflow("wheel-orders.yml")
    assert "schedule" not in on and "workflow_dispatch" in on
    names = ("far-wings-orders.yml", "vol-arb-orders.yml", "wheel-orders.yml")
    scheduled = [n for n in names if "schedule" in workflow(n)[1]]
    assert scheduled == ["far-wings-orders.yml"]        # one nightly writer of mkt/open_orders.json


def test_the_wheels_run_button_does_nothing_unless_its_box_is_ticked():
    w, on = workflow("wheel-orders.yml")
    box = on["workflow_dispatch"]["inputs"]["overwrite_vol_arb"]
    assert box["type"] == "boolean" and box["default"] is False
    job = w["jobs"]["orders"]
    assert job["if"] == "github.ref == 'refs/heads/main' && inputs.overwrite_vol_arb"
    run = next(s for s in job["steps"] if s.get("id") == "orders")
    assert run["run"] == "python wheel_orders.py --overwrite-vol-arb"
    # one concurrency group for every writer of mkt/open_orders.json, never cancelled mid-run
    groups = {n: workflow(n)[0]["concurrency"] for n in ("far-wings-orders.yml", "vol-arb-orders.yml",
                                                         "wheel-orders.yml")}
    assert all(g == {"group": "mkt-open-orders", "cancel-in-progress": False} for g in groups.values())
