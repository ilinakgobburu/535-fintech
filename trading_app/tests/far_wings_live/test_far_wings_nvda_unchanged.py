"""
NVDA's orders are unchanged by the basket: on each of fw_nvda_scenarios' 53 made-up nights (entries on 12 smiles and
levels, 4 spots and a leftover hedge; the band on 32 books and closes, a leg with no recent mid, one wing assigned;
the expiry night's leftover shares; a quiet night), far_wings_live.plan_night for the basket's NVDA writes exactly
the orders, and the notes, that the single-name code wrote (recorded at commit b5f201e in nvda_before_basket.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import fw_live_synth as S
import fw_nvda_scenarios as N
from trading_app.lib import far_wings_live as FW

BEFORE = json.loads((Path(__file__).parent / "nvda_before_basket.json").read_text())
SCENARIOS = N.scenarios()


def test_every_scenario_was_recorded():
    assert BEFORE["recorded_with"].startswith("b5f201e")
    assert [label for label, _ in SCENARIOS] == list(BEFORE["scenarios"])
    assert len(SCENARIOS) == 53 and sum(len(v["orders"]) for v in BEFORE["scenarios"].values()) == 81
    kinds = {(o["contract"], o["action"]) for v in BEFORE["scenarios"].values() for o in v["orders"]}
    assert kinds >= {("put", "SHT"), ("call", "SHT"), ("underlying", "BUY"), ("underlying", "SEL"),
                     ("underlying", "SHT"), ("underlying", "BTC")}


@pytest.mark.parametrize("label, kw", SCENARIOS, ids=[label for label, _ in SCENARIOS])
def test_nvdas_orders_and_notes_are_the_single_name_codes(label, kw):
    got = FW.plan_night(S.config(), name=S.NVDA_NAME, **kw)
    want = BEFORE["scenarios"][label]
    assert got["orders"] == want["orders"]
    assert got["notes"] == want["notes"]
