"""
wheel_orders.py refuses to run while config/vol_arb_live.yaml says the vol-arb owns mkt/open_orders.json, unless
given --overwrite-vol-arb, so the wheel's Run button cannot replace the vol-arb's legs or hedge; and it refuses
outright while config/far_wings_live.yaml says the far wings own it (no flag gets past that). The repo's vol-arb
config no longer claims the file (the far wings do), so the vol-arb guard is checked on a made-up claim. Nothing here
reaches LSEG: the guards stop before the wheel reads its config, and the session opener is replaced.
"""

from __future__ import annotations

import json
import sys

import pytest

import va_live_synth as S

_path = list(sys.path)
sys.path.insert(0, str(S.REPO))
import wheel_orders as WO  # noqa: E402

sys.path[:] = _path


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "mkt").mkdir()
    orders = [{"id": "nvda-va-put-2026-11-23", "for": "2026-11-24"}, {"id": "nvda-va-call-2026-11-23", "for": "2026-11-24"}]
    S.write_json(tmp_path / "mkt" / "open_orders.json", orders)
    monkeypatch.setattr(WO, "MKT", tmp_path / "mkt")
    monkeypatch.setattr(WO, "CONFIG", tmp_path / "no_wheel_config_here.yaml")     # past the guards, this fails
    claim = tmp_path / "vol_arb_live.yaml"
    claim.write_text("owns_open_orders: true\n")
    monkeypatch.setattr(WO, "VOL_ARB_CONFIG", claim)
    monkeypatch.setattr(WO, "FAR_WINGS_CONFIG", tmp_path / "no_far_wings_config_here.yaml")
    sessions = []
    monkeypatch.setattr(WO, "open_session", lambda: sessions.append(1))
    return {"root": tmp_path, "orders": orders, "sessions": sessions}


def test_the_repo_configs_say_the_far_wings_own_the_orders_not_the_vol_arb():
    assert S.config()["owns_open_orders"] is False
    assert WO.VOL_ARB_CONFIG == S.CONFIG and WO.vol_arb_owns_orders() is False
    assert WO.FAR_WINGS_CONFIG == S.REPO / "config" / "far_wings_live.yaml" and WO.far_wings_own_orders() is True


def test_the_wheel_refuses_while_the_vol_arb_owns_the_file(repo):
    with pytest.raises(SystemExit, match="The two-sided variance trade owns mkt/open_orders.json"):
        WO.main([])
    assert json.loads((repo["root"] / "mkt" / "open_orders.json").read_text()) == repo["orders"]
    assert repo["sessions"] == []


def test_the_override_gets_past_the_guard(repo):
    with pytest.raises(FileNotFoundError, match="no_wheel_config_here"):          # it went on to read its config
        WO.main(["--overwrite-vol-arb"])


@pytest.mark.parametrize("text", ["owns_open_orders: false\n", "rung: rule_1\n", "owns_open_orders: 'true'\n", None])
def test_without_the_vol_arbs_claim_the_wheel_runs(repo, monkeypatch, tmp_path, text):
    path = tmp_path / "vol_arb_live_other.yaml"
    if text is not None:
        path.write_text(text)
    monkeypatch.setattr(WO, "VOL_ARB_CONFIG", path)
    assert WO.vol_arb_owns_orders() is False
    with pytest.raises(FileNotFoundError):
        WO.main([])


@pytest.mark.parametrize("args", [[], ["--overwrite-vol-arb"]])
def test_the_wheel_refuses_outright_while_the_far_wings_own_the_file(repo, monkeypatch, args):
    monkeypatch.setattr(WO, "FAR_WINGS_CONFIG", S.REPO / "config" / "far_wings_live.yaml")      # the repo's claim
    with pytest.raises(SystemExit, match="The far-wing trade owns mkt/open_orders.json"):
        WO.main(args)
    assert json.loads((repo["root"] / "mkt" / "open_orders.json").read_text()) == repo["orders"]
    assert repo["sessions"] == []


@pytest.mark.parametrize("text", ["owns_open_orders: false\n", "owns_open_orders: 'true'\n", "{}\n"])
def test_without_the_far_wings_claim_the_wheel_goes_on(repo, monkeypatch, tmp_path, text):
    path = tmp_path / "far_wings_live.yaml"
    path.write_text(text)
    monkeypatch.setattr(WO, "FAR_WINGS_CONFIG", path)
    assert WO.far_wings_own_orders() is False
    with pytest.raises(FileNotFoundError, match="no_wheel_config_here"):
        WO.main(["--overwrite-vol-arb"])
