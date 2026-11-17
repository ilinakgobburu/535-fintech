"""
vol_arb_orders.py refuses outright while config/far_wings_live.yaml says the far wings own mkt/open_orders.json (her
choice of 2026-11-16, "Yes, switch to far wings"), as wheel_orders.py does: no flag gets past it. An override would
not last: the far wings' next nightly run replaces every order for its session (it would drop the vol-arb's legs),
and a vol-arb run the far wings' wings or hedge, both green. Going back to the vol-arb means committing
owns_open_orders: false there first. A --dry-run writes nothing and still runs. LSEG and Cboe are the fakes of
test_vol_arb_orders.py (no network); its scratch repo has no far-wing config unless a test writes one.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

import va_live_synth as S
from test_vol_arb_orders import ENTRY, OLD_ORDERS, FakeLSEG, O, env, orders_file  # noqa: F401

FAR_WINGS = [{"id": f"nvda-fw-{w}-2026-11-23", "for": "2026-11-24", "order_type": "lmt", "action": "SHT", "qty": q,
              "underlying": 4815747, "contract": w, "expiry": "2026-12-18", "strike": k, "ref": "open",
              "lmt_prc": p} for w, q, k, p in (("put", 41, "145", "0.53"), ("call", 59, "220", "0.24"))]


def claim(env, text):
    path = env["root"] / "config" / "far_wings_live.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)


def test_the_repo_says_the_far_wings_own_the_file():
    assert O.ROOT == S.REPO and O.FAR_WINGS_CONFIG.as_posix() == "config/far_wings_live.yaml"
    assert O.far_wings_own_orders() is True


@pytest.mark.parametrize("args", [[], ["--retry"]])
@pytest.mark.parametrize("before", ["wheel", "far wings"])
def test_the_vol_arb_refuses_while_the_far_wings_own_the_file(env, args, before, capsys):
    claim(env, "owns_open_orders: true\n")
    if before == "far wings":                      # the far wings' entry for Nov 24, written at 22:00 on the 23rd
        (env["root"] / "mkt" / "open_orders.json").write_text(json.dumps(FAR_WINGS, indent=2) + "\n")
    want = orders_file(env)
    env["use"](FakeLSEG(ENTRY))
    with pytest.raises(SystemExit, match="The far-wing trade owns mkt/open_orders.json.*commit owns_open_orders: "
                                         "false there first"):
        O.main(args)
    assert orders_file(env) == want and env["sessions"] == [] and env["vix_reads"] == 0
    assert not (env["root"] / "data" / "vol_arb_ledger.json").exists() and not env["output"].exists()


@pytest.mark.parametrize("args", [["--overwrite-far-wings"], ["--overwrite-far-wings", "--retry"]])
def test_no_flag_gets_past_the_claim(env, args, capsys):
    # the override is gone: argparse stops on it (status 2) before anything is read, asked or written
    claim(env, "owns_open_orders: true\n")
    env["use"](FakeLSEG(ENTRY))
    with pytest.raises(SystemExit) as stop:
        O.main(args)
    assert stop.value.code == 2 and "unrecognized arguments: --overwrite-far-wings" in capsys.readouterr().err
    assert orders_file(env) == OLD_ORDERS and env["sessions"] == [] and env["vix_reads"] == 0
    assert not (env["root"] / "data" / "vol_arb_ledger.json").exists() and not env["output"].exists()


def test_the_script_has_no_override_left():
    assert "overwrite" not in (S.REPO / "vol_arb_orders.py").read_text()
    assert "overwrite" not in (S.REPO / ".github" / "workflows" / "vol-arb-orders.yml").read_text()


def test_a_dry_run_still_runs_and_writes_nothing(env, capsys):
    claim(env, "owns_open_orders: true\n")
    env["use"](FakeLSEG(ENTRY))
    assert O.main(["--dry-run", "--as-of", "2026-11-23"]) == 0
    assert orders_file(env) == OLD_ORDERS and '"id": "nvda-va-put-2026-11-23"' in capsys.readouterr().out


@pytest.mark.parametrize("text", ["owns_open_orders: false\n", "owns_open_orders: 'true'\n", "{}\n", None])
def test_without_the_far_wings_claim_the_vol_arb_writes(env, text):
    if text is not None:
        claim(env, text)
    assert O.far_wings_own_orders() is False
    env["use"](FakeLSEG(dt.date(2026, 11, 24)), now=dt.datetime(2026, 11, 24, 22, 0))
    assert O.main([]) == 0 and orders_file(env) == []


def test_once_the_claim_is_given_up_the_vol_arb_writes_its_entry(env):
    # going back to the vol-arb: owns_open_orders false committed in the far wings' config, then a plain run
    claim(env, "owns_open_orders: false\n")
    env["use"](FakeLSEG(ENTRY))
    assert O.main([]) == 0
    assert [o["id"] for o in orders_file(env)] == ["nvda-va-put-2026-11-23", "nvda-va-call-2026-11-23"]
