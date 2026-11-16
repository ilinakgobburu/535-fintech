"""
The live config and the frozen code: lib/vol_arb.py and lib/vol_arb_wf.py hash to what config/vol_arb_live.yaml
records (copied byte for byte from the backtest's commit), so does every lib/wheel.py, lib/vol.py and
lib/pmcc_daily.py function they use, and the config's registered numbers are the library's.
"""

from __future__ import annotations

import ast
import hashlib
import textwrap

import pytest
import yaml

import va_live_synth as S
from trading_app.lib import vol_arb as VA
from trading_app.lib import vol_arb_live as VL
from trading_app.lib import vol_arb_wf as WF

def test_the_backtest_code_is_the_recorded_copy():
    cfg = S.config()
    recorded = cfg["lib"]["sha256"]
    assert set(recorded) == {"trading_app/trading_app/lib/vol_arb.py", "trading_app/trading_app/lib/vol_arb_wf.py"}
    for rel, digest in recorded.items():
        assert hashlib.sha256((S.REPO / rel).read_bytes()).hexdigest() == digest, rel
    assert recorded["trading_app/trading_app/lib/vol_arb.py"] == \
        "5b6363607c7edf418d55bef10c266fb764d711da00475ebeb0084f5923676c8e"
    assert recorded["trading_app/trading_app/lib/vol_arb_wf.py"] == \
        "c288253f52cb0186373a1b7a42dc50ae6869fb88b2c95ac7b761d98f5b020355"
    assert "51cab41d41df3208074ec78aec8cecdf2a08ddc8" in cfg["lib"]["source"]


LIB = S.REPO / "trading_app" / "trading_app" / "lib"


def lib_source(module):
    return (LIB / f"{module}.py").read_text()


def test_the_lib_functions_the_frozen_code_uses_are_the_recorded_ones():
    pinned = S.config()["lib"]["functions"]
    assert VL.dependency_hashes(lib_source) == pinned
    assert set(pinned) == {
        "pmcc_daily.carry", "pmcc_daily.monthly_expiries", "pmcc_daily.third_friday",
        "vol.DAYS_PER_YEAR", "vol.GREEKS", "vol.MIN_T", "vol.VOL_HI", "vol.VOL_LO", "vol.bs_greeks", "vol.bs_price",
        "vol.implied_vol", "wheel.CALL", "wheel.CONTRACT", "wheel.DAYS_PER_YEAR", "wheel.PUT", "wheel._num",
        "wheel.option_greeks", "wheel.select"}
    # each is the sha256 of the lines that define it, as they are on disk
    text = VL._module_index(lib_source("pmcc_daily"))[0]["carry"][1]
    assert text.startswith("def carry(market: dict, day: pd.Timestamp)") and "pd.Timedelta(days=365)" in text
    assert pinned["pmcc_daily.carry"] == hashlib.sha256(text.encode()).hexdigest()


def _external_uses(path):
    """(module, name) for each W.<name> and each `from .vol / .pmcc_daily / .wheel import name` in one file."""
    tree = ast.parse(path.read_text())
    aliases, out = {}, set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            for a in node.names:
                if node.module is None:
                    aliases[a.asname or a.name] = a.name
                elif node.module in ("wheel", "vol", "pmcc_daily"):
                    out.add((node.module, a.name))
    for sub in ast.walk(tree):
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and aliases.get(sub.value.id) == "wheel":
            out.add(("wheel", sub.attr))
    return out


def test_the_roots_are_everything_the_live_path_takes_from_the_public_lib():
    used = _external_uses(LIB / "vol_arb.py") | _external_uses(LIB / "vol_arb_wf.py") | _external_uses(LIB / "vol_arb_live.py")
    # entry_sessions is called only by vol_arb.stock_cycles, the backtest loop, which the live script never runs
    assert used - {("wheel", "entry_sessions")} == set(VL.BACKTEST_ROOTS)
    assert ("wheel", "entry_sessions") in used


FAKE = {
    "a": textwrap.dedent('''\
        from .b import helper as aid, LIMIT
        from . import c as C
        from .unused import nothing
        import numpy as np
        SCALE = 2.0
        OTHER = 9.0


        def top(x):
            return aid(x) * SCALE + C.leaf(x) + np.sqrt(LIMIT)


        def lonely(x):
            return OTHER
        '''),
    "b": textwrap.dedent('''\
        LIMIT = 4.0


        def helper(x):
            return deeper(x) + LIMIT


        def deeper(x):
            return -x
        '''),
    "c": "def leaf(x):\n    return x\n",
}


def test_dependency_hashes_follows_calls_imports_aliases_and_constants():
    got = VL.dependency_hashes(FAKE.__getitem__, roots=(("a", "top"),))
    assert set(got) == {"a.top", "a.SCALE", "b.helper", "b.deeper", "b.LIMIT", "c.leaf"}   # not a.lonely, a.OTHER
    assert got["c.leaf"] == hashlib.sha256(FAKE["c"].rstrip("\n").encode()).hexdigest()
    # a change two calls down, a changed constant, a new callee and a comment inside each change the result
    for module, old, new in [("b", "return -x", "return -2 * x"), ("b", "LIMIT = 4.0", "LIMIT = 5.0"),
                             ("b", "return deeper(x) + LIMIT", "return deeper(x) + LIMIT + top2(x)\n\n\ndef top2(x):\n    return 0"),
                             ("c", "    return x", "    # note\n    return x")]:
        edited = {**FAKE, module: FAKE[module].replace(old, new)}
        assert edited[module] != FAKE[module]
        assert VL.dependency_hashes(edited.__getitem__, roots=(("a", "top"),)) != got, (module, old)
    # an edit to a function the roots do not reach changes nothing
    edited = {**FAKE, "a": FAKE["a"].replace("return OTHER", "return OTHER + 1")}
    assert VL.dependency_hashes(edited.__getitem__, roots=(("a", "top"),)) == got
    with pytest.raises(KeyError, match="defines no missing"):
        VL.dependency_hashes(FAKE.__getitem__, roots=(("a", "missing"),))


def test_hash_differences_names_each_kind():
    assert VL.hash_differences({"a": "1", "b": "2"}, {"a": "1", "b": "2"}) == []
    assert VL.hash_differences({"a": "1", "b": "2"}, {"a": "9", "c": "3"}) == [
        "a changed", "c is used but not recorded", "b is recorded but no longer used"]


def test_the_imported_modules_are_those_files():
    assert VL.sha256_hex(open(VA.__file__, "rb").read()) == S.config()["lib"]["sha256"]["trading_app/trading_app/lib/vol_arb.py"]
    assert VL.sha256_hex(open(WF.__file__, "rb").read()) == S.config()["lib"]["sha256"]["trading_app/trading_app/lib/vol_arb_wf.py"]


def test_the_config():
    cfg = S.config()
    VL.check_config(cfg)
    assert cfg["underlying"] == {"root": "NVDA", "ric": "NVDA.O", "conid": 4815747}
    assert cfg["rung"] in ("rule_1", "logistic", "forest")
    assert cfg["panel"] == "data/vol_arb_dev_panel.json"
    assert cfg["development"]["stocks"] == ["NVDA", "AMD", "DIS", "COST", "DAL"]
    assert [str(d) for d in cfg["development"]["window"]] == ["2019-01-02", "2023-12-29"]
    assert cfg["registered"] == {"corridor_sd": 1.5, "band_put_rich_above": 0.55, "band_call_rich_below": 0.45,
                                 "target_delta": 0.25, "vega_dollars": 200, "capital": 100000, "warm_months": 24,
                                 "lot": 100}
    assert cfg["chain"]["bands"] == {"P": [0.70, 1.02], "C": [0.97, 1.40]}
    assert cfg["data"]["vix_csv"] == "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


@pytest.mark.parametrize("key, value", [
    ("corridor_sd", 2.0), ("band_put_rich_above", 0.6), ("band_call_rich_below", 0.4), ("target_delta", 0.3),
    ("vega_dollars", 100), ("capital", 50000), ("warm_months", 12), ("lot", 10), ("lot", "100"),
])
def test_a_number_that_is_not_the_librarys_is_refused(key, value):
    cfg = S.config()
    cfg["registered"][key] = value
    with pytest.raises(ValueError, match=key):
        VL.check_config(cfg)


@pytest.mark.parametrize("change", [{"rung": "rule_0"}, {"rung": "tree"},
                                    {"underlying": {"root": "NVDA", "ric": "NVDA.O", "conid": "4815747"}}])
def test_a_bad_rung_or_conid_is_refused(change):
    with pytest.raises(ValueError):
        VL.check_config(S.config(**change))
