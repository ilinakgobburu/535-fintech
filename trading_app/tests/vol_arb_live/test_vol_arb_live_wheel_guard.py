"""
wheel_orders.py refuses to run while config/vol_arb_live.yaml says the vol-arb owns mkt/open_orders.json, unless
given --overwrite-vol-arb, so the wheel's Run button cannot replace the vol-arb's legs or hedge. Nothing here
reaches LSEG: the guard stops before the wheel reads its config, and the session opener is replaced.
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
    monkeypatch.setattr(WO, "CONFIG", tmp_path / "no_wheel_config_here.yaml")     # past the guard, this fails
    sessions = []
    monkeypatch.setattr(WO, "open_session", lambda: sessions.append(1))
    return {"root": tmp_path, "orders": orders, "sessions": sessions}


def test_the_repo_config_says_the_vol_arb_owns_the_orders():
    assert S.config()["owns_open_orders"] is True
    assert WO.VOL_ARB_CONFIG == S.CONFIG and WO.vol_arb_owns_orders() is True


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
    path = tmp_path / "vol_arb_live.yaml"
    if text is not None:
        path.write_text(text)
    monkeypatch.setattr(WO, "VOL_ARB_CONFIG", path)
    assert WO.vol_arb_owns_orders() is False
    with pytest.raises(FileNotFoundError):
        WO.main([])
