"""
The development panel: its format (vol_arb_dev_panel/1), the export from a development output, and the loader's
refusals. The development output here is made up in the runner's shape (cache/vol_arb_ml_dev.json: stage, commit,
choice_for_test, cycles with status, features, flat_baseline, raw_premiums and label), written and read back as
JSON the way the runner writes it.
"""

from __future__ import annotations

import copy
import json
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

import va_live_synth as S
from trading_app.lib import vol_arb as VA
from trading_app.lib import vol_arb_live as VL
from trading_app.lib import vol_arb_wf as WF

sys.path.insert(0, str(S.REPO))
import export_vol_arb_panel as EXPORT  # noqa: E402

sys.path.remove(str(S.REPO))


def dev_output(chosen="logistic"):
    """A made-up development output: the synthetic panel's rows as measured cycles, plus one skipped, one with a
    feature that is not finite and one with a leg missing (no label)."""
    cycles = []
    for r in S.panel_obj()["rows"]:
        cycles.append({"stock": r["stock"], "entry": r["entry"], "expiry": r["expiry"], "status": "measured",
                       "measure": {}, "features": {f: r[f] for f in VA.FEATURES}, "flat_baseline": 0.02,
                       "raw_premiums": {"f1_put_premium_raw": r["f1_put_premium"] + 0.01,
                                        "f2_call_premium_raw": r["f2_call_premium"] - 0.01},
                       "label": r["label"], "no_trade": None, "trades": {}})
    cycles[3]["label"], cycles[3]["no_trade"] = None, VA.LEG_MISSING
    skipped = {**copy.deepcopy(cycles[0]), "entry": "2019-01-22", "status": "few_strikes", "features": None,
               "flat_baseline": None, "raw_premiums": None, "label": None, "stock": "ZZZ"}
    nan_row = {**copy.deepcopy(cycles[1]), "status": VA.NOT_FINITE, "stock": "YYY"}
    nan_row["features"]["f8_vix"] = float("nan")
    out = {"stage": "dev", "config": "config/vol_arb_ml.yaml", "registered": "2026-11-11", "commit": "abc123",
           "environment": {"sklearn": "1.5.1"}, "stocks": S.DEV_STOCKS, "window": ["2019-01-02", "2023-12-29"],
           "choice_for_test": {"chosen": chosen, "why": "made up", "mid_sharpe": {}},
           "cycles": cycles + [skipped, nan_row]}
    return json.loads(json.dumps(out))          # through JSON, as the runner writes it (NaN round-trips)


def test_export_is_vol_arb_wf_panels_rows():
    dev = dev_output()
    panel = VL.export_panel(dev, source="vol_arb_ml_dev.json", sha256="f" * 64)
    assert panel["format"] == VL.PANEL_FORMAT and panel["features"] == list(VA.FEATURES)
    assert panel["made_from"] == {"file": "vol_arb_ml_dev.json", "sha256": "f" * 64, "commit": "abc123",
                                  "config": "config/vol_arb_ml.yaml", "registered": "2026-11-11"}
    assert panel["choice_for_test"]["chosen"] == "logistic"
    want = WF.panel(dev["cycles"], predict=False)
    assert len(panel["rows"]) == len(want) == 295                 # the skipped and the not-finite have no row
    got = pd.DataFrame(panel["rows"])
    assert list(got["id"]) == list(want["id"])
    assert np.array_equal(got[list(VA.FEATURES)].to_numpy(float), want[list(VA.FEATURES)].to_numpy(float))
    assert panel["rows"][3]["label"] is None                                  # the leg-missing cycle
    assert sum(r["label"] is None for r in panel["rows"]) == 1
    assert all(type(r["label"]) is int for i, r in enumerate(panel["rows"]) if i != 3)
    assert set(panel["rows"][0]) == {"id", "stock", "entry", "expiry", *VA.FEATURES, "label"}


def test_the_exported_panel_loads_and_keeps_every_row():
    dev = dev_output()
    obj = json.loads(json.dumps(VL.export_panel(dev, source="x", sha256="0" * 64)))
    df = VL.load_panel(obj, S.config(rung="logistic"))
    assert len(df) == 295 and int(df["label"].isna().sum()) == 1 and not df["predict"].any()
    assert list(df.columns) == ["id", "stock", "entry", "expiry", *VA.FEATURES, "label", "predict"]
    assert df["entry"].is_monotonic_increasing                    # the runner's cycles come stock by stock
    assert dev["cycles"][0]["stock"] == dev["cycles"][1]["stock"] and df["stock"].iloc[0] != df["stock"].iloc[1]
    assert pd.api.types.is_datetime64_any_dtype(df["entry"]) and pd.api.types.is_datetime64_any_dtype(df["expiry"])


def test_a_row_with_no_label_is_never_trained_on():
    obj = S.panel_obj(chosen="logistic")
    obj["rows"][0]["label"] = None
    df = VL.load_panel(obj, S.config(rung="logistic"))
    p_with = VL.decide("logistic", {f: 0.0 for f in VA.FEATURES}, df, S.ENTRY)
    assert p_with["train_rows"] == 294


@pytest.mark.parametrize("bad", [{"stage": "test"}, {"choice_for_test": {}}, {"cycles": []}])
def test_export_refuses(bad):
    with pytest.raises(VL.PanelError):
        VL.export_panel({**dev_output(), **bad}, source="x", sha256="0")


def test_the_export_script(tmp_path, capsys):
    src = S.write_json(tmp_path / "vol_arb_ml_dev.json", dev_output("forest"))
    out = tmp_path / "data" / "vol_arb_dev_panel.json"
    assert EXPORT.main(["--dev", str(src), "--out", str(out)]) == 0
    obj = json.loads(out.read_text())
    assert obj["made_from"]["sha256"] == VL.sha256_hex(src.read_bytes()) and obj["made_from"]["file"] == src.name
    assert len(VL.load_panel(obj, S.config(rung="forest"))) == 295
    # the lines it prints for config/vol_arb_live.yaml pin exactly the file it wrote
    printed = capsys.readouterr().out.split("config/vol_arb_live.yaml:\n", 1)[1]
    cfg = yaml.safe_load(printed)
    assert cfg["rung"] == "forest"
    assert cfg["panel_pin"] == {"sha256": VL.sha256_hex(out.read_bytes()),
                                "dev_output_sha256": VL.sha256_hex(src.read_bytes()), "dev_output_commit": "abc123"}
    assert VL.panel_pin_problems(out.read_bytes(), obj, {**S.config(), **cfg}) == []
    with pytest.raises(SystemExit):
        EXPORT.main(["--dev", str(S.write_json(tmp_path / "t.json", {**dev_output(), "stage": "test"})),
                     "--out", str(out)])


def test_panel_pin_problems():
    raw = json.dumps(S.panel_obj()).encode()
    obj = json.loads(raw)
    pin = {"sha256": VL.sha256_hex(raw), "dev_output_sha256": "0" * 64, "dev_output_commit": None}
    assert VL.panel_pin_problems(raw, obj, {"panel_pin": pin}) == []
    assert VL.panel_pin_problems(raw, obj, {"panel_pin": {**pin, "sha256": None}})[0].startswith(
        "config/vol_arb_live.yaml pins no panel yet")
    assert VL.panel_pin_problems(raw, obj, {}) == VL.panel_pin_problems(raw, obj, {"panel_pin": None})
    assert len(VL.panel_pin_problems(raw, obj, {})) == 1
    other = raw.replace(b'"label": 1', b'"label": 0', 1)
    assert VL.panel_pin_problems(other, json.loads(other), {"panel_pin": pin}) == [
        f"its sha256 is {VL.sha256_hex(other)}, panel_pin records {pin['sha256']}"]
    assert len(VL.panel_pin_problems(raw, obj, {"panel_pin": {**pin, "dev_output_sha256": "1" * 64,
                                                              "dev_output_commit": "x"}})) == 2
    assert VL.panel_pin_problems(raw, obj, {"panel_pin": {**pin, "sha256": "A" * 64}}) == [
        f"panel_pin.sha256 {'A' * 64!r} is not a sha256"]


def test_the_repo_config_pins_no_panel_until_the_export():
    assert S.config()["panel_pin"] == {"sha256": None, "dev_output_sha256": None, "dev_output_commit": None}


def mutate(change):
    obj = S.panel_obj()
    change(obj)
    return obj


def _set(row, key, value):
    def f(obj):
        obj["rows"][row][key] = value
    return f


REFUSED = {
    "format": lambda o: o.update(format="vol_arb_dev_panel/0"),
    "features out of order": lambda o: o.update(features=list(reversed(VA.FEATURES))),
    "rung disagrees": lambda o: o.update(choice_for_test={"chosen": "forest"}),
    "no recorded choice": lambda o: o.pop("choice_for_test"),
    "a test-period row": lambda o: o["rows"].append({**o["rows"][0], "id": "NVDA|2024-02-20", "entry": "2024-02-20",
                                                     "expiry": "2024-03-15"}),
    "a test stock": lambda o: o["rows"].append({**o["rows"][0], "id": "MA|2019-01-22", "stock": "MA"}),
    "expiry not after entry": _set(0, "expiry", "2019-01-22"),
    "id not the row's": _set(0, "id", "NVDA|2019-01-23"),
    "NaN feature": _set(0, "f8_vix", float("nan")),
    "string feature": _set(0, "f1_put_premium", "0.1"),
    "bool feature": _set(0, "f7_earnings_in_cycle", True),
    "missing feature": lambda o: o["rows"][0].pop("f9_ret21"),
    "label 2": _set(0, "label", 2),
    "label True": _set(0, "label", True),
    "label missing": lambda o: o["rows"][0].pop("label"),
    "duplicate id": lambda o: o["rows"].append(dict(o["rows"][0])),
    "no rows": lambda o: o.update(rows=[]),
    "no labelled row": lambda o: [r.update(label=None) for r in o["rows"]],
    "entry not a date": _set(0, "entry", "soon"),
}


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_the_loader_refuses(name):
    with pytest.raises(VL.PanelError):
        VL.load_panel(mutate(REFUSED[name]), S.config())


def test_the_loader_refuses_anything_but_an_object():
    for obj in (None, [], "panel"):
        with pytest.raises(VL.PanelError):
            VL.load_panel(obj, S.config())


def test_fewer_than_24_entry_months_is_refused():
    obj = S.panel_obj()
    obj["rows"] = [r for r in obj["rows"] if r["entry"] < "2020-12-01"]     # 23 entry months, Jan 2019-Nov 2020
    df = VL.load_panel(obj, S.config())
    assert df["entry"].dt.to_period("M").nunique() == 23
    for rung in VL.RUNGS:
        with pytest.raises(VL.PanelError):
            VL.decide(rung, {f: 0.0 for f in VA.FEATURES}, df, S.ENTRY)
    obj["rows"] = [r for r in S.panel_obj()["rows"] if r["entry"] < "2021-01-01"]   # 24
    VL.decide("rule_1", {f: 0.0 for f in VA.FEATURES}, VL.load_panel(obj, S.config()), S.ENTRY)


def test_only_rows_finished_before_d_are_fit_on():
    df = VL.load_panel(S.panel_obj(chosen="logistic"), S.config(rung="logistic"))
    on = pd.Timestamp("2022-01-10")             # mid-development: rows whose expiry is after this are not finished
    d = VL.decide("logistic", {f: 0.0 for f in VA.FEATURES}, df, on)
    assert d["train_rows"] == int(((df["expiry"] < on) & df["label"].notna()).sum()) < len(df)


def test_decide_refuses_an_unknown_rung():
    df = VL.load_panel(S.panel_obj(), S.config())
    with pytest.raises(ValueError):
        VL.decide("rule_0", {f: 0.0 for f in VA.FEATURES}, df, S.ENTRY)
