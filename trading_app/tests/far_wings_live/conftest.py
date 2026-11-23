"""
The far wings' live tests. Since 2026-11-23 the live script does not hedge (lib/far_wings_live.HEDGE = False, config
`hedge: false`). The hedge path is still in the code, behind HEDGE, so the tests written for it (the band, the hedge
limit, crossing zero, NVDA's single-name scenarios) run with it switched back on: HEDGE True and every config the
helpers build carrying hedge: true. test_far_wings_no_hedge.py covers the default.
"""
from __future__ import annotations

import pytest

import fw_live_synth as S
from trading_app.lib import far_wings_live as FW

HEDGE_TESTS = {
    "test_each_name_is_banded_on_its_own_book_never_pooled",
    "test_the_hedge_limit_is_each_names_own",
    "test_the_dec_18_cycle_from_the_first_entry_to_the_next_regular_entry",
    "test_a_leg_with_no_mid_tonight_keeps_its_last_iv",
    "test_every_order_is_legal_by_the_order_page",
    "test_the_band_at_its_boundary",
    "test_the_band_from_real_deltas_after_the_entry",
    "test_the_hedge_limit_is_40_times_the_book",
    "test_the_shares_count_in_the_band",
    "test_the_target_is_the_options_delta_rounded",
    "test_two_orders_when_the_hedge_crosses_zero_from_short_to_long",
    "test_nvdas_orders_and_notes_are_the_single_name_codes",
    "test_a_move_past_the_band_writes_the_hedge",
    "test_a_rejected_wing_stops_until_acknowledged",
    "test_the_night_after_the_fill_checks_the_band_and_trades_nothing_inside_it",
}


@pytest.fixture(autouse=True)
def _registered_hedge(request, monkeypatch):
    """The registered hedge, on, for the tests written for it (HEDGE_TESTS); the default (no hedge) otherwise."""
    if request.node.originalname in HEDGE_TESTS:
        monkeypatch.setattr(FW, "HEDGE", True)
        config = S.config
        monkeypatch.setattr(S, "config", lambda **over: config(**{"hedge": True, **over}))
        check = FW.check_config                    # configs some modules built at import, before this fixture

        def check_hedged(cfg):
            if isinstance(cfg, dict) and cfg.get("hedge") is False:
                cfg["hedge"] = True
            return check(cfg)
        monkeypatch.setattr(FW, "check_config", check_hedged)
    yield
