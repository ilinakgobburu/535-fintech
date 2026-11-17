"""
The far-wing Action is the one nightly writer of mkt/open_orders.json: 03:00 UTC Tue-Sat (22:00 New York in winter)
and the 06:00 retry, on main only, with the wheel's pinned packages plus scikit-learn (lib/vol_arb_live imports
lib/vol_arb_wf), committing mkt/open_orders.json and its ledger with the rebase-and-retry step, in the concurrency
group the vol-arb and wheel Actions share. Neither of those has a schedule any more.
"""

from __future__ import annotations

import yaml

import fw_live_synth as S

WORKFLOWS = S.REPO / ".github" / "workflows"


def workflow(name):
    w = yaml.safe_load((WORKFLOWS / name).read_text())
    return w, w.get("on", w.get(True))           # YAML 1.1 reads the key `on` as True


def test_the_far_wing_action():
    w, on = workflow("far-wings-orders.yml")
    assert on["schedule"] == [{"cron": "0 3 * * 2-6"}, {"cron": "0 6 * * 2-6"}] and "workflow_dispatch" in on
    assert on["workflow_dispatch"] is None                       # the Run button needs no box: it is the owner
    assert w["permissions"] == {"contents": "write"}
    assert w["concurrency"] == {"group": "mkt-open-orders", "cancel-in-progress": False}
    job = w["jobs"]["orders"]
    assert job["if"] == "github.ref == 'refs/heads/main'"
    py = next(s for s in job["steps"] if s.get("uses", "").startswith("actions/setup-python"))
    assert py["with"]["python-version"] == "3.11"
    pip = next(s["run"] for s in job["steps"] if "pip install" in s.get("run", ""))
    wheel_pip = next(s["run"] for s in workflow("wheel-orders.yml")[0]["jobs"]["orders"]["steps"]
                     if "pip install" in s.get("run", ""))
    assert set(pip.split()) == set(wheel_pip.split()) | {"scikit-learn==1.5.1"}
    run = next(s for s in job["steps"] if s.get("id") == "orders")
    # only the second schedule passes --retry; the first and the Run button decide afresh
    assert run["run"] == "python far_wings_orders.py ${{ github.event.schedule == '0 6 * * 2-6' && '--retry' || '' }}"
    assert set(run["env"]) == {"PYTHONWARNINGS", "LSEG_APP_KEY", "LSEG_USERNAME", "LSEG_PASSWORD"}
    commit = next(s for s in job["steps"] if s.get("name") == "Commit mkt/open_orders.json and the ledger")
    assert "git add mkt/open_orders.json\n" in commit["run"]
    assert "if [ -f data/far_wings_ledger.json ]; then git add data/far_wings_ledger.json; fi" in commit["run"]
    assert "git pull --rebase origin main" in commit["run"] and "git push origin HEAD:main" in commit["run"]
    assert commit["env"]["FOR_DAY"] == "${{ steps.orders.outputs.for_day }}"


def test_it_is_the_only_scheduled_writer_and_every_writer_shares_the_group():
    names = sorted(p.name for p in WORKFLOWS.glob("*.yml"))
    writers = [n for n in names if "mkt/open_orders.json" in (WORKFLOWS / n).read_text().split("run:", 1)[-1]]
    assert set(writers) >= {"far-wings-orders.yml", "vol-arb-orders.yml", "wheel-orders.yml"}
    assert [n for n in writers if "schedule" in workflow(n)[1]] == ["far-wings-orders.yml"]
    assert all(workflow(n)[0]["concurrency"] == {"group": "mkt-open-orders", "cancel-in-progress": False}
               for n in writers)


def test_the_vol_arb_action_keeps_only_a_run_button_with_a_box():
    w, on = workflow("vol-arb-orders.yml")
    assert "schedule" not in on
    box = on["workflow_dispatch"]["inputs"]["overwrite_far_wings"]
    assert box == {"description": box["description"], "type": "boolean", "default": False}
    job = w["jobs"]["orders"]
    assert job["if"] == "github.ref == 'refs/heads/main' && inputs.overwrite_far_wings"
    run = next(s for s in job["steps"] if s.get("id") == "orders")
    assert run["run"] == "python vol_arb_orders.py --overwrite-far-wings"
