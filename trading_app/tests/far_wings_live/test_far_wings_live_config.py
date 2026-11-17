"""
config/far_wings_live.yaml: its registered numbers are the module's (and the choices' as she gave them), the basket's
31 names as she chose them (NVDA's entry as the single-name config had it), the lib functions it pins hash to the
files as they stand, and each malformed setting is refused.
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


# The basket as she chose it on 2026-11-17: NVDA, then the long/short vol book's registered draw (ranks 1-30), each
# ticker's RIC as LSEG returns it and its IB conid, all 31 confirmed through her TWS by matching IB's ISIN to LSEG's
# (V 49462172 and T 37018770 too, and XOM's new 895178251 after its holding-company reorganization).
BASKET = [("NVDA", "NVDA.O", 4815747), ("MSFT", "MSFT.OQ", 272093), ("AAPL", "AAPL.OQ", 265598),
          ("AMZN", "AMZN.OQ", 3691937), ("GOOG", "GOOG.OQ", 208813720), ("BRK B", "BRKb.N", 72063691),
          ("META", "META.OQ", 107113386), ("JNJ", "JNJ.N", 8719), ("JPM", "JPM.N", 1520593), ("V", "V.N", 49462172),
          ("XOM", "XOM.N", 895178251), ("WMT", "WMT.OQ", 13824), ("PFE", "PFE.N", 11031), ("UNH", "UNH.N", 13272),
          ("BAC", "BAC.N", 10098), ("VZ", "VZ.N", 4901), ("PG", "PG.N", 11054), ("WFC", "WFC.N", 10375),
          ("INTC", "INTC.OQ", 270639), ("CVX", "CVX.N", 5684), ("T", "T.N", 37018770), ("KO", "KO.N", 8894),
          ("MRK", "MRK.N", 70101545), ("MA", "MA.N", 38685693), ("CSCO", "CSCO.OQ", 268084), ("HD", "HD.N", 7930),
          ("BA", "BA.N", 4762), ("DIS", "DIS.N", 6459), ("ORCL", "ORCL.N", 272800), ("PEP", "PEP.OQ", 11017),
          ("CMCSA", "CMCSA.OQ", 267748)]


def test_the_repo_config_is_well_formed_and_owns_the_orders():
    cfg = S.config()
    FW.check_config(cfg)
    assert cfg["owns_open_orders"] is True and cfg["acknowledged"] == {}
    assert "underlying" not in cfg
    assert cfg["registered"] == {"put_max_abs_delta": 0.10, "put_target_abs_delta": 0.05, "call_max_abs_delta": 0.05,
                                 "call_target_abs_delta": 0.03, "vega_dollars": 200, "capital": 100000,
                                 "band_share_of_book": 0.05, "lot": 100}
    # the strikes asked for are the backtest caches' bands, as vol-arb's live config has them, and NVDA's grid is
    # still the backtest's (NVDA_2024's); the single-name config's NVDA is the basket's first name, unchanged
    va = VL_config()
    assert cfg["chain"]["bands"] == va["chain"]["bands"] and "grid" not in cfg["chain"]
    nvda = cfg["names"][0]
    assert {k: nvda[k] for k in ("ticker", "root", "ric", "conid")} == {"ticker": "NVDA", "root": "NVDA",
                                                                         "ric": "NVDA.O", "conid": S.NVDA}
    assert nvda["grid"] == va["chain"]["grid"] and nvda["lot"] == 100
    assert set(cfg["technical_choices"]) >= {"calendar", "wings", "size", "fills", "hedge", "basket_grids",
                                             "one_file", "name_by_name"}


def test_the_basket_is_the_31_names_she_chose():
    names = S.config()["names"]
    assert [(n["ticker"], n["ric"], n["conid"]) for n in names] == BASKET
    assert all(type(n["conid"]) is int for n in names)              # no name waits for a conid
    for t, isin in (("V", "US92826C8394"), ("T", "US00206R1023")):
        note = " ".join(S.name(t)["probe"].split())
        assert f"ISIN {isin} matches LSEG's" in note and "not yet confirmed" not in note
    assert all(n["lot"] == 100 and n["root"] == n["ticker"].replace(" ", "") for n in names)
    # BRK B's option root as LSEG answered it; its id root has no space; every probe note is dated
    brk = S.name("BRK B")
    assert brk["root"] == "BRKB" and FW.own_prefix("BRK B") == "brkb-fw-" and "root BRKB confirmed" in brk["probe"]
    assert all(n["probe"].startswith("2026-11-17 probe, Dec 18 2026 monthlies") for n in names[1:])
    assert all(len(n["grid"]) == 1 and n["grid"][0][0] is None for n in names[1:])
    steps = {n["ticker"]: n["grid"][0][1] for n in names[1:]}
    assert {t for t, k in steps.items() if k == 10.0} == {"JNJ", "UNH"}
    # $0.50 where $1 and $2.50 strikes are both listed (BAC, VZ) or may be: PFE, T and CMCSA (the probe never asked
    # a $2.50-multiple x.50 strike), INTC (its put wing, near 41, is below the $50 the probe rounded to); $2.50 for
    # ORCL and DIS, whose put wings fall below the $100 the probe rounded to
    assert {t for t, k in steps.items() if k == 0.5} == {"BAC", "VZ", "PFE", "T", "CMCSA", "INTC"}
    assert {t for t, k in steps.items() if k == 1.0} == set()
    assert {t for t, k in steps.items() if k == 2.5} == {"WFC", "KO", "CSCO", "DIS", "ORCL"}
    assert {t for t, k in steps.items() if k == 5.0} == set(steps) - {"JNJ", "UNH", "BAC", "VZ", "PFE", "T", "CMCSA",
                                                                      "INTC", "WFC", "KO", "CSCO", "DIS", "ORCL"}
    for t in ("PFE", "T", "CMCSA", "INTC", "ORCL", "DIS"):
        assert "costs only never-listed requests" in " ".join(S.name(t)["probe"].split()), t
    header = " ".join((S.REPO / "config" / "far_wings_live.yaml").read_text().split("\nnames:")[0]
                      .replace("#", " ").split())
    assert "These 30 names were NOT in the far-wing backtest, which used 11 stocks" in header
    assert '"why cant you do more than nvda?"' in header and '"The 30-stock basket" and "$100k book per stock"' in header


def test_every_id_root_and_prefix_is_distinct_and_none_starts_another():
    tickers = [n["ticker"] for n in S.config()["names"]]
    prefixes = [FW.own_prefix(t) for t in tickers]
    assert len(set(prefixes)) == 31 and all("-" not in FW.id_root(t) and FW.id_root(t).isalnum() for t in tickers)
    for a in prefixes:
        assert [b for b in prefixes if b.startswith(a)] == [a]
    for t in tickers:
        assert FW.owner(f"{FW.own_prefix(t)}put-2026-11-23", tickers) == t
    assert FW.owner("nvda-va-put-2026-11-23", tickers) is None and FW.owner("t-csp-2026-11-16", tickers) is None
    assert FW.owner("nvda-fw-put-2026-11-23", tickers) == "NVDA" and FW.owner(None, tickers) is None
    # only the start of an id says whose it is: her own order naming another id inside is not T's (t-fw-)
    assert FW.owner("msft-close-t-fw-put-2026-11-23", tickers) is None
    assert FW.owner("hand-brkb-fw-call-2026-11-23", tickers) is None


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
    (("names", 0, "conid"), "4815747", "conid must be an integer, or null"),
    (("names", 0, "conid"), True, "conid must be an integer, or null"),
    (("names", 0, "ric"), "", "ric must be the stock's LSEG RIC"),
    (("names", 0, "root"), "nvda", "upper-case option root"),
    (("names", 0, "root"), "BRK B", "upper-case option root"),
    (("names", 0, "ticker"), "", "ticker must be a non-empty string"),
    (("names", 0, "ticker"), " NVDA", "ticker must be a non-empty string"),
    (("names", 0, "ticker"), "NVDA\nstopped=x", "ticker must be a non-empty string"),
    (("names", 0, "ticker"), "nvda", "ticker must be a non-empty string"),
    (("names", 0, "lot"), 10, "every name is sized on the registered lot 100"),
    (("names", 0, "grid"), [], "grid is not a list"),
    (("names", 0, "grid"), [[150, 1.0], [300, 2.5]], "null \\(any price above\\) last"),
    (("names", 0, "grid"), [[300, 2.5], [150, 1.0], [None, 10.0]], "rising bounds"),
    (("names", 0, "grid"), [[None, 0.0]], "positive increment"),
    (("names", 0, "grid"), [[None, "5"]], "positive increment"),
    (("names", 0, "probe"), "", "probe, when given"),
    (("names", 1, "ticker"), "NVDA", "two names have the same ticker"),
    (("names", 1, "ticker"), "NV DA", "two names have the same id root"),
    (("names", 1, "ticker"), "N.V.D.A", "two names have the same id root"),
    (("names", 1, "conid"), 4815747, "two names have the same conid"),
    (("names", 0), {"ticker": "NVDA", "ric": "NVDA.O", "root": "NVDA", "conid": 4815747, "lot": 100},
     "each name needs exactly ticker, ric, root, conid, grid, lot"),
    (("names", 0), {**S.NVDA_NAME, "exchange": "NASDAQ"}, "each name needs exactly"),
    (("names",), [], "names must be a list"),
    (("names",), {"NVDA": {}}, "names must be a list"),
    (("underlying",), {"root": "NVDA", "ric": "NVDA.O", "conid": 4815747}, "underlying is gone"),
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
