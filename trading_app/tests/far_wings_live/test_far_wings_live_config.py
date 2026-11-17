"""
config/far_wings_live.yaml: its registered numbers are the module's (and the choices' as she gave them), the lib
functions it pins hash to the files as they stand, and each malformed setting is refused.
"""

from __future__ import annotations

import copy

import pytest

import fw_live_synth as S
from trading_app.lib import far_wings_live as FW
from trading_app.lib import vol_arb as VA
from trading_app.lib import vol_arb_live as VL

LIB = S.REPO / "trading_app" / "trading_app" / "lib"


def read(module):
    return (LIB / f"{module}.py").read_text()


def test_the_repo_config_is_well_formed_and_owns_the_orders():
    cfg = S.config()
    FW.check_config(cfg)
    assert cfg["owns_open_orders"] is True and cfg["acknowledged"] == {}
    assert cfg["underlying"] == {"root": "NVDA", "ric": "NVDA.O", "conid": S.NVDA}
    assert cfg["registered"] == {"put_max_abs_delta": 0.10, "put_target_abs_delta": 0.05, "call_max_abs_delta": 0.05,
                                 "call_target_abs_delta": 0.03, "vega_dollars": 200, "capital": 100000,
                                 "band_share_of_book": 0.05, "lot": 100}
    # the strikes asked for are the backtest caches' bands and grid, as vol-arb's live config has them
    va = VL_config()
    assert cfg["chain"]["bands"] == va["chain"]["bands"] and cfg["chain"]["grid"] == va["chain"]["grid"]
    assert set(cfg["technical_choices"]) >= {"calendar", "wings", "size", "fills", "hedge"}


def VL_config():
    import yaml
    return yaml.safe_load((S.REPO / "config" / "vol_arb_live.yaml").read_text())


def test_the_module_numbers():
    assert FW.CAP == {"P": 0.10, "C": 0.05} and FW.TARGET == {"P": 0.05, "C": 0.03}
    assert (FW.VEGA_DOLLARS, FW.CAPITAL, FW.BAND_SHARE, FW.LOT) == (200.0, 100_000.0, 0.05, 100)
    assert FW.TAG == "fw" and FW.own_prefix("NVDA") == "nvda-fw-" and VL.own_prefix("NVDA") != FW.own_prefix("NVDA")
    assert VA.CAPITAL == FW.CAPITAL


@pytest.mark.parametrize("key, value", [("put_max_abs_delta", 0.15), ("put_target_abs_delta", 0.04),
                                        ("call_max_abs_delta", 0.10), ("call_target_abs_delta", "0.03"),
                                        ("vega_dollars", 100), ("capital", None), ("band_share_of_book", 0.1),
                                        ("lot", 10)])
def test_a_registered_number_that_differs_stops(key, value):
    cfg = copy.deepcopy(S.config())
    cfg["registered"][key] = value
    with pytest.raises(ValueError, match=f"disagrees with lib/far_wings_live.py: {key} is"):
        FW.check_config(cfg)


@pytest.mark.parametrize("path, value, match", [
    (("underlying", "conid"), "4815747", "integer conid"),
    (("underlying", "ric"), "", "integer conid"),
    (("owns_open_orders",), "true", "owns_open_orders"),
    (("acknowledged",), {"nvda-fw-put-2026-11-23": ""}, "acknowledged"),
    (("acknowledged",), ["nvda-fw-put-2026-11-23"], "acknowledged"),
    (("chain", "min_dated_share"), 0, "min_dated_share"),
    (("chain", "min_dated_share"), 1.5, "min_dated_share"),
    (("limits", "max_hedge_x_capital"), 0, "max_hedge_x_capital"),
    (("limits", "max_hedge_x_capital"), "40", "max_hedge_x_capital"),
])
def test_a_malformed_setting_stops(path, value, match):
    cfg = copy.deepcopy(S.config())
    d = cfg
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        FW.check_config(cfg)


# --------------------------------------------------------------------------- the pinned lib functions
def test_the_recorded_lib_functions_are_the_files():
    got = FW.lib_functions(read)
    assert S.config()["lib_functions"] == got
    # what it reaches: the wing rule's helpers, the band's delta, the calendar, the book and the order checks
    for k in ("wheel.option_greeks", "vol_arb.two_sided", "vol_arb.leg_vega", "vol_arb.contracts_for",
              "vol_arb.greeks_at", "vol_arb_live.leg_delta", "vol_arb_live.transition", "vol_arb_live.entry_sessions",
              "vol_arb_live.read_book", "wheel_live.validate_order", "wheel_live.number_token",
              "wheel_live.merge_orders", "wheel_live.decision_dates", "pmcc_daily.carry", "vol.implied_vol",
              "ric.build_option_ric"):
        assert k in got, k
    assert not any(k.startswith("far_wings_live.") for k in got)
    # the vol-arb ML fit is not reached
    assert not any(k.startswith("vol_arb_wf.") for k in got)


@pytest.mark.parametrize("module, old, new", [
    ("wheel", "    F, D = spot * np.exp((r - q) * T), np.exp(-r * T)\n    iv = implied_vol(",
     "    F, D = spot * np.exp((r - q) * T), np.exp(-r * T)\n    iv  = implied_vol("),
    ("vol_arb_live", "def transition(held: int, target: int)", "def transition(held: int, target: int = 0)"),
    ("wheel_live", 'return f"{round(float(x), 2):.2f}".rstrip("0").rstrip(".")',
     'return f"{round(float(x), 3):.3f}".rstrip("0").rstrip(".")'),
    ("vol_arb", "per_contract = vega_per_share / 100.0 * lot", "per_contract = vega_per_share / 100 * lot"),
])
def test_an_edit_to_a_used_lib_function_changes_the_hashes(module, old, new):
    text = read(module)
    assert text.count(old) == 1
    got = FW.lib_functions(lambda m: text.replace(old, new) if m == module else read(m))
    assert FW.hash_differences(S.config()["lib_functions"], got) != []


def test_an_edit_elsewhere_does_not():
    text = read("vol_arb_live")
    old = "def panel_pin_problems("
    assert text.count(old) == 1
    got = FW.lib_functions(lambda m: text.replace(old, "def panel_pin_problems( ") if m == "vol_arb_live" else read(m))
    assert FW.hash_differences(S.config()["lib_functions"], got) == []
