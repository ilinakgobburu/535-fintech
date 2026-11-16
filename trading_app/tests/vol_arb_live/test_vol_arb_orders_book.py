"""
vol_arb_orders.py end to end on the book: the switch night with the wheel's put filled, a book Sanji wrote without
mkt/fills.json, contracts read as shares, an assignment, and the ledger the script keeps of what it wrote. LSEG and
Cboe are the fakes of test_vol_arb_orders.py (no network); the books and fills are made up.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

import va_live_synth as S
from trading_app.lib import vol_arb_live as VL
from test_vol_arb_orders import (ENTRY, FOR, OLD_ORDERS, FakeLSEG, O, by_hand_target, entry_orders, env,  # noqa: F401
                                 hedge_orders, leg_frames, option, orders_file, sanji, shares, write_mkt)

WHEEL_PUT = OLD_ORDERS[0]                       # nvda-csp-2026-11-16: SHT 6 NVDA 2026-12-18 155 put, at 7.78


def ledger(env):
    path = env["root"] / "data" / "vol_arb_ledger.json"
    return json.loads(path.read_text()) if path.exists() else None


def test_the_wheels_filled_put_stops_the_first_vol_arb_night(env, capsys):
    env["use"](FakeLSEG(dt.date(2026, 11, 17), chain=False), now=dt.datetime(2026, 11, 17, 22, 0))
    write_mkt(env, "fills.json", [{"id": WHEEL_PUT["id"], "status": "filled"}])
    write_mkt(env, "positions.json", {"cash": 0, "positions": [option("P", 155, -6)]})
    with pytest.raises(SystemExit, match="the book holds -6 2026-12-18 155 put, which no filled nvda-va-\\* order "
                                         "accounts for.*close it by hand"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS and env["sessions"] == [] and ledger(env) is None


def test_the_wheels_put_stops_the_entry_night_too(env):
    # the case the review found: the entry would net into the same Dec 18 put and lose the $200 sizing
    env["use"](FakeLSEG(ENTRY))
    write_mkt(env, "fills.json", [{"id": WHEEL_PUT["id"], "status": "filled"}])
    write_mkt(env, "positions.json", {"cash": 0, "positions": [option("P", 165, -6)]})
    with pytest.raises(SystemExit, match="which no filled nvda-va-\\* order accounts for"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS and env["sessions"] == [] and env["vix_reads"] == 0


def test_contracts_read_as_shares_stop(env):
    day = dt.date(2026, 11, 25)
    env["use"](FakeLSEG(day, chain=False, legs=leg_frames(day, [("P", 165.0, 0.5), ("C", 195.0, 0.4)])),
               now=dt.datetime(2026, 11, 25, 22, 0))
    entry = entry_orders([("P", 165.0, -10), ("C", 195.0, 12)])
    sanji(env, entry, [option("P", 165, -1000), option("C", 195, 1200)])
    with pytest.raises(SystemExit, match="holds -1000 2026-12-18 165 put, but this strategy's fills add up to -10"):
        O.main([])
    assert orders_file(env) == entry and env["sessions"] == []


def test_huge_shares_held_stop_before_any_order(env):
    # shares are not reconciled with the fills (an assignment delivers them), so the limit is their guard
    day = dt.date(2026, 11, 25)
    env["use"](FakeLSEG(day, chain=False, legs=leg_frames(day, [("P", 165.0, 0.5), ("C", 195.0, 0.4)])),
               now=dt.datetime(2026, 11, 25, 22, 0))
    entry = entry_orders([("P", 165.0, -10), ("C", 195.0, 12)])
    sanji(env, entry + hedge_orders("2026-11-24", day, 0, -400),
          [option("P", 165, -10), option("C", 195, 12), shares(-40000)])
    with pytest.raises(SystemExit, match="shares held, -40000 NVDA shares.*more than 5 x"):
        O.main([])
    assert orders_file(env) == entry + hedge_orders("2026-11-24", day, 0, -400)


def test_a_book_without_a_fills_file_stops_every_night(env):
    # Sanji wrote positions.json but no fills.json: the same stale book must not get a second market hedge
    legs = [("P", 165.0, 0.5), ("C", 195.0, 0.4)]
    for day, now in ((dt.date(2026, 11, 25), dt.datetime(2026, 11, 25, 22, 0)),
                     (dt.date(2026, 11, 27), dt.datetime(2026, 11, 27, 22, 0))):
        env["use"](FakeLSEG(day, chain=False, legs=leg_frames(day, legs)), now=now)
        write_mkt(env, "positions.json", {"cash": 0, "positions": [option("P", 165, -10), option("C", 195, 12)]})
        with pytest.raises(SystemExit, match="there is no mkt/fills.json"):
            O.main([])
        assert orders_file(env) == OLD_ORDERS and env["sessions"] == []


def test_after_the_short_put_is_assigned_the_shares_are_sold_down_to_the_calls_hedge(env):
    # put_rich: SHT 14 165P, BUY 14 195C, hedged short 672. The close went through 165, Sanji assigned the put
    # (gone, +1400 shares), so the book is 14 long calls and +728 shares: SEL 728, then SHT the calls' hedge
    day = dt.date(2026, 12, 2)
    env["use"](FakeLSEG(day, chain=False, legs=leg_frames(day, [("C", 195.0, 0.4)])),
               now=dt.datetime(2026, 12, 2, 22, 0))
    sanji(env, entry_orders([("P", 165.0, -14), ("C", 195.0, 14)]) + hedge_orders("2026-11-24", "2026-11-25", 0, -672),
          [option("C", 195, 14), shares(728)], written=hedge_orders("2026-12-01", "2026-12-02", -672, -672))
    assert O.main([]) == 0
    target = by_hand_target(day, [("C", 195.0, 14, 0.4)])
    assert target < 0
    assert [(o["id"], o["action"], o["qty"], o["for"]) for o in orders_file(env)] == [
        ("nvda-va-hedge-sel-2026-12-02", "SEL", 728, "2026-12-03"),
        ("nvda-va-hedge-sht-2026-12-02", "SHT", -target, "2026-12-03")]


def test_the_ledger_keeps_what_was_written_and_the_next_night_checks_against_it(env, capsys):
    env["use"](FakeLSEG(ENTRY))
    assert O.main([]) == 0
    entry = orders_file(env)
    assert [o["id"] for o in entry] == ["nvda-va-put-2026-11-23", "nvda-va-call-2026-11-23"]
    assert ledger(env) == {"format": "vol_arb_ledger/1", "orders": {o["id"]: o for o in entry},
                           "nights": {"2026-11-23": {"for": "2026-11-24", "ids": [o["id"] for o in entry]}}}
    out = capsys.readouterr().out
    assert "Dropped nvda-csp-2026-11-16 (for 2026-11-16, not this strategy's) from mkt/open_orders.json" in out
    # Sanji fills both legs on the 24th (fills appended, positions rewritten); the 24th's run hedges them
    put, call = entry
    write_mkt(env, "fills.json", [{"id": put["id"], "status": "filled"}, {"id": call["id"], "status": "filled"}])
    sign = {"SHT": -1, "BUY": 1}
    write_mkt(env, "positions.json", {"cash": 0, "positions": [
        option("P", float(put["strike"]), sign[put["action"]] * put["qty"]),
        option("C", float(call["strike"]), sign[call["action"]] * call["qty"])]})
    day = dt.date(2026, 11, 24)
    legs = [("P", float(put["strike"]), 0.5), ("C", float(call["strike"]), 0.4)]
    env["use"](FakeLSEG(day, chain=False, legs=leg_frames(day, legs)), now=dt.datetime(2026, 11, 24, 22, 0))
    assert O.main([]) == 0
    target = by_hand_target(day, [("P", legs[0][1], sign[put["action"]] * put["qty"], 0.5),
                                  ("C", legs[1][1], sign[call["action"]] * call["qty"], 0.4)])
    (hedge,) = orders_file(env)
    assert (hedge["id"], hedge["action"], hedge["qty"]) == ("nvda-va-hedge-2026-11-24", "SHT", -target)
    got = ledger(env)
    assert set(got["orders"]) == {put["id"], call["id"], hedge["id"]}
    assert got["nights"]["2026-11-24"] == {"for": "2026-11-25", "ids": [hedge["id"]]}
    # a fill the ledger does not hold stops the run
    write_mkt(env, "fills.json", [{"id": put["id"], "status": "filled"}, {"id": call["id"], "status": "filled"},
                                  {"id": "nvda-va-hedge-2026-11-20", "status": "filled"}])
    with pytest.raises(SystemExit, match="nvda-va-hedge-2026-11-20 filled, but data/vol_arb_ledger.json has no"):
        O.main([])


def test_a_dry_run_leaves_the_ledger_alone(env):
    env["use"](FakeLSEG(ENTRY))
    assert O.main(["--dry-run"]) == 0
    assert ledger(env) is None
    assert O.main([]) == 0
    before = ledger(env)
    assert before["nights"]["2026-11-23"]["ids"] == ["nvda-va-put-2026-11-23", "nvda-va-call-2026-11-23"]
    assert O.main(["--dry-run", "--as-of", "2026-11-23"]) == 0
    assert ledger(env) == before


def test_a_bad_ledger_stops(env):
    env["use"](FakeLSEG(ENTRY))
    S.write_json(env["root"] / "data" / "vol_arb_ledger.json", {"orders": []})
    with pytest.raises(SystemExit, match="not a vol_arb_ledger/1 file"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS


def test_a_run_while_the_session_is_open_records_nothing(env, capsys):
    env["use"](FakeLSEG(ENTRY))
    assert O.main([]) == 0
    before, entry = ledger(env), orders_file(env)
    # Tuesday 10:00: the entry's session (the 24th) is under way, so its orders are kept and nothing is added
    env["use"](FakeLSEG(ENTRY), now=dt.datetime(2026, 11, 24, 10, 0))
    assert O.main([]) == 0
    assert "its session has not been resolved yet, so no new order" in capsys.readouterr().out
    assert orders_file(env) == entry and ledger(env) == before


def ack(env, **ids):
    cfg = json.loads(env["config"].read_text())
    cfg["acknowledged"] = {k.replace("_", "-"): v for k, v in ids.items()}
    S.write_json(env["config"], cfg)


def test_a_rejected_leg_stops_every_night_until_acknowledged(env, capsys):
    # the entry's put was rejected and its call filled: the book holds the call alone
    put, call = entry_orders([("P", 165.0, -14), ("C", 195.0, 14)])
    write_mkt(env, "open_orders.json", [put, call])
    path = env["root"] / "data" / "vol_arb_ledger.json"
    S.write_json(path, {"format": "vol_arb_ledger/1", "orders": {put["id"]: put, call["id"]: call}, "nights": {}})
    write_mkt(env, "fills.json", [{"id": put["id"], "status": "rejected"}, {"id": call["id"], "status": "filled"}])
    write_mkt(env, "positions.json", {"cash": 0, "positions": [option("C", 195, 14)]})
    for day in (dt.date(2026, 11, 24), dt.date(2026, 11, 25)):
        env["use"](FakeLSEG(day, chain=False, legs=leg_frames(day, [("C", 195.0, 0.4)])),
                   now=dt.datetime.combine(day, dt.time(22, 0)))
        with pytest.raises(SystemExit, match="Sanji rejected nvda-va-put-2026-11-23"):
            O.main([])
        assert orders_file(env) == [put, call]
    ack(env, nvda_va_put_2026_11_23="rejected; the call filled alone and is hedged as it stands")
    assert O.main([]) == 0
    out = capsys.readouterr().out
    assert "Note: Sanji rejected nvda-va-put-2026-11-23; acknowledged" in out
    (hedge,) = orders_file(env)                                            # the rejected order is gone from the file
    assert hedge["id"] == "nvda-va-hedge-2026-11-25" and hedge["action"] == "SHT"
    assert hedge["qty"] == -by_hand_target(dt.date(2026, 11, 25), [("C", 195.0, 14, 0.4)])


def test_the_wheels_unanswered_put_does_not_stop_the_vol_arb(env, capsys):
    # Sanji starts keeping fills.json without a result for the wheel's last order: noted, not red
    env["use"](FakeLSEG(dt.date(2026, 11, 17), chain=False), now=dt.datetime(2026, 11, 17, 22, 0))
    write_mkt(env, "fills.json", [])
    write_mkt(env, "positions.json", {"cash": 100000, "positions": []})
    assert O.main([]) == 0
    out = capsys.readouterr().out
    assert "Note: mkt/fills.json has no result for nvda-csp-2026-11-16, which is not this strategy's" in out
    assert "Dropped nvda-csp-2026-11-16 (for 2026-11-16, not this strategy's)" in out and orders_file(env) == []
