"""
The stale-book guards on mkt/open_orders.json as it stands (guard_problems): only this strategy's rejected or
unanswered orders stop the run, another's are noted, and an id listed in config's `acknowledged` is noted with her
reason, so one bad result cannot keep every later night red. Made-up orders and fills.
"""

from __future__ import annotations

import datetime as dt

import pytest

import va_live_synth as S
from trading_app.lib import vol_arb_live as VL

D = dt.date(2026, 11, 24)
PUT = {"id": "nvda-va-put-2026-11-23", "for": "2026-11-24"}
CALL = {"id": "nvda-va-call-2026-11-23", "for": "2026-11-24"}
WHEEL = {"id": "nvda-csp-2026-11-16", "for": "2026-11-16"}
LATER = {"id": "nvda-va-hedge-2026-11-24", "for": "2026-11-25"}


def guard(written, fills, ack=None):
    return VL.guard_problems(written, fills, D, "NVDA", ack or {})


def test_nothing_to_wait_for_before_sanji_keeps_fills():
    assert guard([PUT, CALL, WHEEL], None) == ([], [])


def test_this_strategys_rejected_order_stops():
    stops, notes = guard([PUT, CALL], [{"id": PUT["id"], "status": "rejected"}, {"id": CALL["id"], "status": "filled"}])
    assert len(stops) == 1 and stops[0].startswith("Sanji rejected nvda-va-put-2026-11-23")
    assert "add nvda-va-put-2026-11-23 to config/vol_arb_live.yaml's acknowledged" in stops[0] and notes == []


def test_this_strategys_unanswered_order_stops_but_a_later_one_is_not_waited_for():
    stops, _ = guard([PUT, CALL, LATER], [{"id": CALL["id"], "status": "filled"}])
    assert stops == ["mkt/fills.json has no result yet for nvda-va-put-2026-11-23: the book may be stale. If Sanji "
                     "has resolved the session and will not answer, add the id to config/vol_arb_live.yaml's "
                     "acknowledged with the reason"]


def test_another_strategys_orders_are_only_noted():
    stops, notes = guard([WHEEL, PUT], [{"id": PUT["id"], "status": "filled"}])
    assert stops == [] and notes == ["mkt/fills.json has no result for nvda-csp-2026-11-16, which is not this "
                                     "strategy's"]
    stops, notes = guard([WHEEL], [{"id": WHEEL["id"], "status": "rejected"}])
    assert stops == [] and notes[0].startswith("Sanji rejected nvda-csp-2026-11-16, which is not this strategy's")


def test_an_acknowledged_id_is_noted_with_its_reason():
    ack = {PUT["id"]: "strike token fixed; the call filled alone"}
    stops, notes = guard([PUT, CALL], [{"id": PUT["id"], "status": "rejected"}, {"id": CALL["id"], "status": "filled"}],
                         ack)
    assert stops == [] and notes == ["Sanji rejected nvda-va-put-2026-11-23; acknowledged in config/vol_arb_live.yaml: "
                                     "strike token fixed; the call filled alone"]
    stops, notes = guard([PUT, CALL], [{"id": CALL["id"], "status": "filled"}], ack)
    assert stops == [] and notes == ["mkt/fills.json has no result for nvda-va-put-2026-11-23; acknowledged in "
                                     "config/vol_arb_live.yaml: strike token fixed; the call filled alone"]
    # acknowledging one does not wave through another
    stops, _ = guard([PUT, CALL], [], ack)
    assert stops and "nvda-va-call-2026-11-23" in stops[0] and "nvda-va-put" not in stops[0]


@pytest.mark.parametrize("ack", [[], "x", {"nvda-va-put-2026-11-23": ""}, {"nvda-va-put-2026-11-23": None},
                                 {1: "reason"}])
def test_the_config_needs_ids_and_reasons(ack):
    with pytest.raises(ValueError, match="acknowledged"):
        VL.check_config({**S.config(), "acknowledged": ack})


def test_the_repo_config_acknowledges_nothing():
    assert S.config()["acknowledged"] == {}
    VL.check_config({**S.config(), "acknowledged": {"nvda-va-put-2026-11-23": "made up"}})
