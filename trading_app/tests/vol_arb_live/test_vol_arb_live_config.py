"""
The live config and the frozen code: lib/vol_arb.py and lib/vol_arb_wf.py hash to what config/vol_arb_live.yaml
records (copied byte for byte from the backtest's commit), and the config's registered numbers are the library's.
"""

from __future__ import annotations

import hashlib

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
