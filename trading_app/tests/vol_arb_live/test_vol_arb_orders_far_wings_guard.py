"""
vol_arb_orders.py refuses to write while config/far_wings_live.yaml says the far wings own mkt/open_orders.json
(her choice of 2026-11-16, "Yes, switch to far wings"), unless given --overwrite-far-wings, as wheel_orders.py
refused for the vol-arb. A --dry-run writes nothing and still runs. LSEG and Cboe are the fakes of
test_vol_arb_orders.py (no network); its scratch repo has no far-wing config unless a test writes one.
"""

from __future__ import annotations

import datetime as dt

import pytest

import va_live_synth as S
from test_vol_arb_orders import ENTRY, OLD_ORDERS, FakeLSEG, O, env, orders_file  # noqa: F401


def claim(env, text):
    path = env["root"] / "config" / "far_wings_live.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)


def test_the_repo_says_the_far_wings_own_the_file():
    assert O.ROOT == S.REPO and O.FAR_WINGS_CONFIG.as_posix() == "config/far_wings_live.yaml"
    assert O.far_wings_own_orders() is True


@pytest.mark.parametrize("args", [[], ["--retry"]])
def test_the_vol_arb_refuses_while_the_far_wings_own_the_file(env, args):
    claim(env, "owns_open_orders: true\n")
    env["use"](FakeLSEG(ENTRY))
    with pytest.raises(SystemExit, match="The far-wing trade owns mkt/open_orders.json"):
        O.main(args)
    assert orders_file(env) == OLD_ORDERS and env["sessions"] == [] and env["vix_reads"] == 0
    assert not (env["root"] / "data" / "vol_arb_ledger.json").exists() and not env["output"].exists()


def test_the_override_writes(env):
    claim(env, "owns_open_orders: true\n")
    env["use"](FakeLSEG(ENTRY))
    assert O.main(["--overwrite-far-wings"]) == 0
    assert [o["id"] for o in orders_file(env)] == ["nvda-va-put-2026-11-23", "nvda-va-call-2026-11-23"]


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
