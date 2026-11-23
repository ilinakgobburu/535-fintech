"""
The one-off first entry (her decision of 2026-11-17, asked "When should the live far-wing script place its first
entry ...? The tested rule enters Mon Nov 23": "Tomorrow (Wednesday)"), end to end with LSEG faked (no network):
tonight's 22:00 run (2026-11-17, 03:00 UTC on the 18th) writes every tracked name's Dec 18 wings for 2026-11-18,
NVDA alone when Sanji tracks NVDA alone (the repo's price_history.json cut to NVDA's entry, as it stood before the
basket's 30 conids were added) and all 31 with the repo's price_history.json as it stands now (NVDA's entry and the 30
additions, price_history_additions.json, for Sanji to track); the 01:00 retry and a run GitHub
delays to 06:23 write the same; one delayed to 10:29 writes nothing, goes red, and the cycle falls back to the
regular 2026-11-23 entry. Then the cycle: 11-18 and 11-19 hedge only, the 11-20 expiry night leaves the Dec 18 legs
and their shares alone, 11-23 hedges only and does not enter again while a name untracked on 11-17 and tracked by
then enters, Thanksgiving and the 11-27 early close, the 12-18 expiry closes the shares, and 12-21 is the regular
entry into 2027-01-15. A name tracked on 11-17 that got no first-entry order does not enter on 11-23 either.

Made-up data only (test_far_wings_basket's names, closes and smiles); nothing here is market data.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fw_live_synth as S
from test_far_wings_basket import (CFG, NAMES, SKEW, TICKERS, VOL, FakeBasket, O, assert_legal, by_name,  # noqa: F401
                                   chain_rows_of, closes_of, env, exact_delta, ledger_file, orders_file, output,
                                   step_at, write_mkt)
from trading_app.lib import far_wings_live as FW
from trading_app.lib import wheel_live as L
from trading_app.lib.ric import build_option_ric

HERE = Path(__file__).parent
FIRST, FIRST_FOR, DEC, JAN = dt.date(2026, 11, 17), dt.date(2026, 11, 18), dt.date(2026, 12, 18), dt.date(2027, 1, 15)
TONIGHT = dt.datetime(2026, 11, 17, 22, 0)          # 03:00 UTC on 2026-11-18
LATE_LINE = ("the session for 2026-11-17's orders (2026-11-18) is already open: nothing written (a late run must not "
             "fill a day late)")
QUESTION = ("When should the live far-wing script place its first entry (selling the Dec 18 put and call wings on "
            "each tracked stock)? The tested rule enters Mon Nov 23; starting earlier is a one-off early first entry, "
            "recorded as such, then the normal monthly calendar (next entry Dec 21)")


def real_price_history() -> dict:
    return json.loads((S.REPO / "price_history.json").read_text())


def nvda_alone() -> dict:
    """The repo's price_history.json with NVDA's entry alone: Sanji pricing NVDA only, as before the basket's 30
    conids were added (and as when he has not yet taken them up)."""
    return {k: v for k, v in real_price_history().items() if int(k) == S.NVDA}


def additions() -> dict:
    return json.loads((HERE / "price_history_additions.json").read_text())


def track(env, *tickers, untracked=()):
    S.write_json(env["root"] / "price_history.json",
                 S.price_history(*(NAMES[t]["conid"] for t in tickers),
                                 untracked=[NAMES[t]["conid"] for t in untracked]))


def market_of(ticker, day, expiry=DEC):
    """The night's market as the fake serves it: the name's closes, the T-bill, dividends and `expiry`'s chain."""
    c = closes_of(ticker, day)
    rows = chain_rows_for(ticker, day, expiry)
    return FW.market(c, None, pd.Series(S.RATE_PCT, index=c.index), S.DIVS, (), FW.quote_frame(rows))


def chain_rows_for(ticker, day, expiry):
    """The name's made-up chain of `expiry` on D, as test_far_wings_basket's fake serves Dec 18's."""
    if expiry == DEC:
        return chain_rows_of(ticker, day)
    n = NAMES[ticker]
    spot = float(closes_of(ticker, day).iloc[-1])
    return S.chain_rows(spot, day=day, expiry=expiry, root=n["root"], step=step_at(n, spot), s0=VOL[ticker],
                        b=SKEW[ticker])


def chain_frames(tickers, day, expiry):
    """(ric, BID/ASK frame) for each contract of each name's `expiry` chain on D, to add to a FakeBasket."""
    return [(r["ric"], pd.DataFrame({"BID": [r["bid"]], "ASK": [r["ask"]]}, index=pd.DatetimeIndex([day])))
            for t in tickers for r in chain_rows_for(t, day, expiry)]


def plan(ticker, day=FIRST, for_day=FIRST_FOR, expiry=DEC):
    """The pure plan of one name's entry on the fake's data: what the script must write for it."""
    return FW.plan_night(CFG, day, for_day, FW.Book(), name=NAMES[ticker], entry_expiry=expiry,
                         mkt=market_of(ticker, day, expiry))


def by_hand_strikes(ticker, grid):
    """The strikes asked for, worked out here: the bands times the closes of 2026-11-10 to 2026-11-17, the 7 calendar
    days up to D, on the grid tier of their mean."""
    c = closes_of(ticker, FIRST)
    w = c[(c.index >= pd.Timestamp("2026-11-10")) & (c.index <= pd.Timestamp("2026-11-17"))]
    assert [d.date().isoformat() for d in w.index] == ["2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13",
                                                       "2026-11-16", "2026-11-17"]
    step = next(float(s) for b, s in grid if b is None or w.mean() < float(b))
    out = {}
    for cp, (lo, hi) in CFG["chain"]["bands"].items():
        k_lo, k_hi = math.floor(w.min() * lo / step) * step, math.ceil(w.max() * hi / step) * step
        out[cp] = [round(float(k), 4) for k in np.arange(k_lo, k_hi + step / 2, step)]
    return out


def files_now(env):
    root = env["root"]
    return {p.relative_to(root).as_posix(): p.read_bytes() for d in ("mkt", "data") if (root / d).exists()
            for p in sorted((root / d).iterdir())}


def red(excinfo) -> str:
    assert excinfo.value.code not in (0, None)
    return str(excinfo.value.code)


# --------------------------------------------------------------------------- the config
def test_the_config_records_her_decision_as_a_one_off():
    fe = CFG["first_entry"]
    assert fe["question"] == QUESTION and fe["answer"] == "Tomorrow (Wednesday)" and fe["one_off"] is True
    assert fe["asked"] == "why cant this start running, say tomorrow or thursday?"
    assert FW.first_entry(CFG) == {"day": FIRST, "for": FIRST_FOR, "expiry": DEC, "replaces": dt.date(2026, 11, 23),
                                   "answer": "Tomorrow (Wednesday)"}
    assert fe["band_window"] == ["2026-11-10", "2026-11-17"]
    # what differs from the tested calendar, as the config says it: 4 sessions early, 31 days to expiry, not 25
    sessions = [d.date() for d in L.sessions(FIRST, dt.date(2026, 11, 23))]
    assert len(sessions) - 1 == 4 and (DEC - FIRST).days == 31 and (DEC - dt.date(2026, 11, 23)).days == 25
    assert "entered 4 sessions early" in fe["differs"] and "31 calendar days to expiry instead of 25" in fe["differs"]
    assert "2026-11-10 to 2026-11-17, the nearest analogue" in fe["differs"]
    assert "falls back to the regular entry on 2026-11-23" in " ".join(fe["late"].split())
    assert "enters at the regular entry on 2026-11-23" in fe["rule"]


def _set(cfg, key, value):
    out = copy.deepcopy(cfg)
    if value is None:
        del out["first_entry"][key]
    else:
        out["first_entry"][key] = value
    return out


@pytest.mark.parametrize("key, value, match", [
    ("day", "2026-11-21", "must be an NYSE session that is not a registered entry"),       # a Saturday
    ("day", "2026-11-23", "must be an NYSE session that is not a registered entry"),       # the regular entry itself
    ("day", "2026-11-16", "first_entry.for 2026-11-18 must be the session after 2026-11-16"),
    ("for", "2026-11-19", "must be the session after 2026-11-17, 2026-11-18"),
    ("expiry", "2027-01-15", "must be the next registered entry after 2026-11-17 and its expiry"),
    ("replaces", "2026-12-21", "must be the next registered entry after 2026-11-17 and its expiry"),
    ("band_window", ["2026-11-09", "2026-11-17"], "band_window must be the 7 calendar days up to 2026-11-17"),
    ("band_window", ["2026-11-10"], "band_window must be the 7 calendar days"),
    ("day", dt.date(2026, 11, 17), "first_entry.day must be a YYYY-MM-DD string"),
    ("one_off", False, "one_off must be true"),
    ("one_off", "true", "one_off must be true"),
    ("answer", " ", "first_entry.answer must say it in words"),
    ("why", None, "first_entry needs exactly"),
    ("names", ["NVDA"], "first_entry needs exactly"),
])
def test_a_malformed_first_entry_stops(key, value, match):
    with pytest.raises(ValueError, match=match):
        FW.check_config(_set(CFG, key, value))


def test_without_the_block_the_registered_calendar_alone():
    cfg = {k: v for k, v in CFG.items() if k != "first_entry"}
    FW.check_config(cfg)
    assert FW.first_entry(cfg) is None
    assert FW.night_of(FIRST, first=None)["entry"] is False


def test_night_of_is_the_registered_calendar_plus_the_one_first_entry():
    first = FW.first_entry(CFG)
    days = [d.date() for d in L.sessions("2026-11-01", "2027-02-28")]
    got = {d: FW.night_of(d, first=first) for d in days}
    entries = {d: (n["expiry"], n["first"], n["band_ref"]) for d, n in got.items() if n["entry"]}
    assert entries == {FIRST: (DEC, True, FIRST),
                       dt.date(2026, 11, 23): (DEC, False, dt.date(2026, 11, 20)),
                       dt.date(2026, 12, 21): (JAN, False, DEC),
                       dt.date(2027, 1, 19): (dt.date(2027, 2, 19), False, JAN),
                       dt.date(2027, 2, 22): (dt.date(2027, 3, 19), False, dt.date(2027, 2, 19))}
    # every other night is vol_arb_live.night's, with no band reference
    for d, n in got.items():
        if d != FIRST:
            assert {k: n[k] for k in ("entry", "expiry", "forced", "late")} == FW.night(d)
            assert n["first"] is False and (n["band_ref"] is None) == (not n["entry"])
    # strike_plan's window for the first entry is the 7 calendar days up to D, 2026-11-10 to 2026-11-17
    hi = closes_of("NVDA", FIRST).copy()
    hi[(hi.index >= pd.Timestamp("2026-11-10")) & (hi.index <= pd.Timestamp("2026-11-17"))] = 200.0
    band = {"P": [1.0, 1.0], "C": [1.0, 1.0]}
    assert FW.strike_plan(hi, got[FIRST]["band_ref"], FIRST, band, [[None, 10.0]])["P"] == [200.0]
    outside = hi.copy()
    outside[outside.index < pd.Timestamp("2026-11-10")] = 50.0           # closes before the window do not count
    assert FW.strike_plan(outside, got[FIRST]["band_ref"], FIRST, band, [[None, 10.0]])["P"] == [200.0]
    # a forced dry-run entry keeps its own expiry and band reference
    forced = FW.night_of(dt.date(2026, 11, 16), True, first)
    assert forced["entry"] and forced["forced"] and forced["expiry"] == DEC and forced["band_ref"] == dt.date(2026, 11,
                                                                                                             20)
    assert FW.night_of(FIRST, True, first)["first"] is True and FW.night_of(FIRST, True, first)["forced"] is False


# --------------------------------------------------------------------------- each name's entry: entry_for
def _order(t, cp, day, expiry=DEC):
    return FW.option_orders(root=t, conid=NAMES[t]["conid"], day=day, for_day=FW.next_session(day), cp=cp,
                            expiry=expiry, strike=150.0, limit="0.5", held=0, target=-3)[0]


def test_entry_for_one_entry_a_cycle():
    first = FW.first_entry(CFG)
    regular, on_first = FW.night_of(dt.date(2026, 11, 23), first=first), FW.night_of(FIRST, first=first)
    quiet = FW.night_of(dt.date(2026, 11, 24), first=first)
    empty, book = FW.read_ledger(None), FW.Book()
    rec = FW.record_first_entry(empty, FIRST, FIRST_FOR, DEC, ["NVDA", "AAPL"])
    mine = FW.name_ledger(rec, "NVDA")
    day = dt.date(2026, 11, 23)
    assert FW.entry_for(quiet, "NVDA", mine, book, rec, dt.date(2026, 11, 24)) == (None, None)
    # tracked by the first entry's run: no entry on the 23rd, whatever it got then (AAPL got nothing)
    for t in ("NVDA", "AAPL"):
        x, why = FW.entry_for(regular, t, FW.name_ledger(rec, t), book, rec, day)
        assert x is None and why == ("its 2026-12-18 cycle began at the one-off first entry on 2026-11-17 (tracked "
                                     "that night, data/far_wings_ledger.json's first_entry), so no entry tonight: "
                                     "the band check only")
    # not tracked then: the regular entry
    assert FW.entry_for(regular, "BRK B", FW.name_ledger(rec, "BRK B"), book, rec, day) == (DEC, None)
    assert FW.entry_for(regular, "NVDA", mine, book, empty, day) == (DEC, None)
    # the first entry's night itself (its retry): the names it recorded still enter
    assert FW.entry_for(on_first, "AAPL", FW.name_ledger(rec, "AAPL"), book, rec, FIRST) == (DEC, None)
    # a record of another expiry does not stop the 23rd's entry, nor the 21st's
    other = {**rec, "first_entry": {**rec["first_entry"], "expiry": "2027-01-15"}}
    assert FW.entry_for(regular, "NVDA", mine, book, other, day) == (DEC, None)
    dec21 = FW.night_of(dt.date(2026, 12, 21), first=first)
    assert FW.entry_for(dec21, "NVDA", mine, book, rec, dt.date(2026, 12, 21)) == (JAN, None)


def test_entry_for_never_enters_an_expiry_already_held_or_ordered():
    first = FW.first_entry(CFG)
    regular, day = FW.night_of(dt.date(2026, 11, 23), first=first), dt.date(2026, 11, 23)
    empty = FW.read_ledger(None)
    held = FW.Book(cash=0, options=[{"cp": "C", "expiry": DEC, "strike": 220.0, "qty": -5}])
    x, why = FW.entry_for(regular, "NVDA", FW.name_ledger(empty, "NVDA"), held, empty, day)
    assert x is None and why == ("it already holds 2026-12-18 220 call, so no second 2026-12-18 entry: the band check "
                                 "only")
    jan = FW.Book(cash=0, options=[{"cp": "C", "expiry": JAN, "strike": 220.0, "qty": -5}])
    assert FW.entry_for(regular, "NVDA", FW.name_ledger(empty, "NVDA"), jan, empty, day) == (DEC, None)
    # its wings ordered on an earlier night (even unfilled), not on this night (a rerun decides again)
    put = _order("NVDA", "P", FIRST)
    led = FW.record_night(empty, "NVDA", [put], FIRST, FIRST_FOR)
    x, why = FW.entry_for(regular, "NVDA", FW.name_ledger(led, "NVDA"), FW.Book(), led, day)
    assert x is None and why == ("its 2026-12-18 wings were ordered on an earlier night (nvda-fw-put-2026-11-17), so "
                                 "no second 2026-12-18 entry: the band check only")
    same = FW.record_night(empty, "NVDA", [_order("NVDA", "P", day)], day, dt.date(2026, 11, 24))
    assert FW.entry_for(regular, "NVDA", FW.name_ledger(same, "NVDA"), FW.Book(), same, day) == (DEC, None)
    # a hedge, or another expiry's wing, on an earlier night is not this cycle's entry
    hedge = FW.stock_orders(root="NVDA", conid=S.NVDA, day=FIRST, for_day=FIRST_FOR, held=0, target=-100)
    jan_put = _order("NVDA", "P", dt.date(2026, 11, 2), expiry=dt.date(2026, 11, 20))
    led = FW.record_night(FW.record_night(empty, "NVDA", hedge, FIRST, FIRST_FOR), "NVDA", [jan_put],
                          dt.date(2026, 11, 2), dt.date(2026, 11, 3))
    assert FW.entry_for(regular, "NVDA", FW.name_ledger(led, "NVDA"), FW.Book(), led, day) == (DEC, None)


def test_the_ledger_keeps_the_first_entry_record():
    empty = FW.read_ledger(None)
    a = FW.record_first_entry(empty, FIRST, FIRST_FOR, DEC, ["NVDA", "AAPL"])
    assert a["first_entry"] == {"day": "2026-11-17", "for": "2026-11-18", "expiry": "2026-12-18",
                                "names": ["NVDA", "AAPL"]}
    # a retry's names join the 22:00 run's, each once, in the order they came
    b = FW.record_first_entry(a, FIRST, FIRST_FOR, DEC, ["AAPL", "BRK B"])
    assert b["first_entry"]["names"] == ["NVDA", "AAPL", "BRK B"] and "first_entry" not in empty
    # every later night keeps it, and the file reads back
    c = FW.record_night(b, "NVDA", [], dt.date(2026, 11, 18), dt.date(2026, 11, 19))
    assert c["first_entry"] == b["first_entry"] and FW.read_ledger(json.loads(json.dumps(c))) == c
    for bad in ({"day": "2026-11-17"}, {**b["first_entry"], "names": "NVDA"},
                {**b["first_entry"], "expiry": "Dec 18"}, {**b["first_entry"], "extra": 1}, [],
                {**b["first_entry"], "names": [1]}):
        with pytest.raises(FW.LedgerError, match="first_entry is not"):
            FW.read_ledger({**c, "first_entry": bad})


# --------------------------------------------------------------------------- tonight, 2026-11-17 22:00
def test_tonights_first_entry_with_nvda_alone_tracked(env, capsys):
    real = nvda_alone()
    assert FW.tracked_conids(real) == {S.NVDA}                     # Sanji tracks NVDA alone
    S.write_json(env["root"] / "price_history.json", real)
    fake = env["use"](FakeBasket(FIRST, ["NVDA"]), now=TONIGHT)
    assert O.main([]) == 0
    got = orders_file(env)
    assert [(o["id"], o["for"], o["action"], o["contract"], o["expiry"], o["order_type"], o["ref"]) for o in got] == [
        ("nvda-fw-put-2026-11-17", "2026-11-18", "SHT", "put", "2026-12-18", "lmt", "open"),
        ("nvda-fw-call-2026-11-17", "2026-11-18", "SHT", "call", "2026-12-18", "lmt", "open")]
    assert got == plan("NVDA")["orders"]
    assert_legal(got)
    # the chain asked for is the Dec 18 chain banded on the closes of 2026-11-10 to 2026-11-17
    strikes = by_hand_strikes("NVDA", NAMES["NVDA"]["grid"])
    assert [r[1] for r in fake.asked("NVDA") if r[1] != "NVDA.O"] == [
        build_option_ric("NVDA", DEC, k, cp, expired=False) for cp in ("P", "C") for k in strikes[cp]]
    led = ledger_file(env)
    assert led["first_entry"] == {"day": "2026-11-17", "for": "2026-11-18", "expiry": "2026-12-18", "names": ["NVDA"]}
    assert led["names"]["NVDA"]["nights"] == {"2026-11-17": {"for": "2026-11-18", "ids": [o["id"] for o in got]}}
    assert output(env) == "for_day=2026-11-18\n"
    out = capsys.readouterr().out
    assert ('Quotes from the close of 2026-11-17, orders for 2026-11-18: entry night, expiry 2026-12-18 (the one-off '
            'first entry, her decision of 2026-11-17: "Tomorrow (Wednesday)"; the regular entry of this cycle is '
            '2026-11-23; chain banded on the closes of 2026-11-10 to 2026-11-17)') in out
    assert "Entry 2026-11-17 -> 2026-12-18: spot" in out and "T 31 days" in out
    assert out.count(": no orders tonight") == 30                  # the 30 names Sanji does not track
    assert "Dropped nvda-csp-2026-11-16 (for 2026-11-16, not this strategy's)" in out
    assert ("Recorded the one-off first entry in data/far_wings_ledger.json: the 2026-12-18 cycle of NVDA began "
            "tonight") in out


def test_tonights_first_entry_with_all_31_tracked(env, capsys):
    adds = additions()
    assert {int(k) for k in adds} == {n["conid"] for n in CFG["names"]} - {S.NVDA}
    assert all(v["tracked"] is True for v in adds.values())
    # the repo's price_history.json as it stands: NVDA's entry, then the 30 additions in config order
    real = real_price_history()
    assert real == {**nvda_alone(), **adds} and list(real) == [str(S.NVDA), *adds]
    assert FW.tracked_conids(real) == {n["conid"] for n in CFG["names"]}
    S.write_json(env["root"] / "price_history.json", real)
    fake = env["use"](FakeBasket(FIRST, TICKERS), now=TONIGHT)
    assert O.main([]) == 0
    got = orders_file(env)
    assert list(by_name(got)) == TICKERS and len(got) == 62
    for t in TICKERS:
        mine = by_name(got)[t]
        assert mine == plan(t)["orders"], t
        assert [(o["id"], o["for"], o["expiry"]) for o in mine] == [
            (f"{FW.id_root(t)}-fw-put-2026-11-17", "2026-11-18", "2026-12-18"),
            (f"{FW.id_root(t)}-fw-call-2026-11-17", "2026-11-18", "2026-12-18")], t
    assert_legal(got)
    for t in ("NVDA", "BRK B", "PFE", "MA"):
        strikes = by_hand_strikes(t, NAMES[t]["grid"])
        assert [r[1] for r in fake.asked(t) if r[1] != NAMES[t]["ric"]] == [
            build_option_ric(NAMES[t]["root"], DEC, k, cp, expired=False) for cp in ("P", "C") for k in strikes[cp]]
    led = ledger_file(env)
    assert led["first_entry"]["names"] == TICKERS and list(led["names"]) == TICKERS
    assert output(env) == "for_day=2026-11-18\n" and "no orders tonight" not in capsys.readouterr().out


@pytest.mark.parametrize("now, args", [
    (dt.datetime(2026, 11, 18, 1, 0), ["--retry"]),     # the 01:00 retry, the 22:00 run having failed
    (dt.datetime(2026, 11, 18, 6, 23), []),            # the 22:00 run as GitHub started it on 2026-11-17 (11:23 UTC)
    (dt.datetime(2026, 11, 18, 9, 29), ["--retry"]),    # the retry, a minute before Wednesday's open
])
def test_the_first_entry_from_the_retry_or_a_run_delayed_to_before_the_open(env, now, args):
    S.write_json(env["root"] / "price_history.json", nvda_alone())
    env["use"](FakeBasket(FIRST, ["NVDA"]), now=now)
    assert O.main(args) == 0
    assert orders_file(env) == plan("NVDA")["orders"]
    assert [o["for"] for o in orders_file(env)] == ["2026-11-18", "2026-11-18"]
    assert ledger_file(env)["first_entry"]["names"] == ["NVDA"] and output(env) == "for_day=2026-11-18\n"


@pytest.mark.parametrize("args", [[], ["--retry"]])
def test_the_first_entry_refused_at_10_29_writes_nothing_and_falls_back_to_the_23rd(env, args):
    S.write_json(env["root"] / "price_history.json", nvda_alone())
    before = files_now(env)
    fake = env["use"](FakeBasket(FIRST, ["NVDA"]), now=dt.datetime(2026, 11, 18, 10, 29))   # 15:29 UTC
    with pytest.raises(SystemExit) as e:
        O.main(args)
    msg = red(e)
    assert LATE_LINE in msg
    assert ("2026-11-17 was the one-off first entry's night, so the early first entry is lost: no wing is placed from "
            "2026-11-17's close for 2026-11-18, and the 2026-12-18 cycle falls back to the regular entry on 2026-11-23 "
            "(orders for 2026-11-24)") in msg
    assert files_now(env) == before and fake.requests == [] and env["sessions"] == [] and output(env) == ""
    # nothing recorded: Monday's regular entry trades the Dec 18 cycle
    env["use"](FakeBasket(S.ENTRY, ["NVDA"]), now=dt.datetime(2026, 11, 23, 22, 0))
    assert O.main([]) == 0
    assert [(o["id"], o["for"], o["expiry"]) for o in orders_file(env)] == [
        ("nvda-fw-put-2026-11-23", "2026-11-24", "2026-12-18"), ("nvda-fw-call-2026-11-23", "2026-11-24", "2026-12-18")]
    assert "first_entry" not in ledger_file(env)


@pytest.mark.parametrize("args", [["--retry"], []])
def test_a_late_run_after_the_22_run_wrote_leaves_its_names_and_their_cycle(env, args):
    # AAPL not final at 22:00 (one listed put with no D row), NVDA written; the retry (or the Run button) starts at
    # 10:29, by when Sanji also tracks BRK B: NVDA's orders stand, AAPL began its cycle at 22:00 (tracked then) and
    # gets no wings, not on the 23rd either, and BRK B, not tracked by the run that wrote, enters on the 23rd. Before,
    # a plain run here said the first entry was lost and the cycle fell back to the 23rd for every name
    track(env, "NVDA", "AAPL")
    fake = FakeBasket(FIRST, ["NVDA", "AAPL", "BRK B"])
    ric = next(r["ric"] for r in chain_rows_of("AAPL", FIRST) if r["cp"] == "P")
    fake.options[ric] = fake.options[ric].iloc[:1]
    env["use"](fake, now=TONIGHT)
    assert O.main([]) == 0
    assert list(by_name(orders_file(env))) == ["NVDA"]
    assert ledger_file(env)["first_entry"]["names"] == ["NVDA", "AAPL"]
    track(env, "NVDA", "AAPL", "BRK B")
    before = files_now(env)
    fake.requests.clear()
    env["use"](fake, now=dt.datetime(2026, 11, 18, 10, 29))
    with pytest.raises(SystemExit) as e:
        O.main(args)
    msg = red(e)
    assert LATE_LINE in msg
    assert ("the orders already written from 2026-11-17's close for NVDA stand as written "
            "(data/far_wings_ledger.json)") in msg
    assert ("the one-off first entry was written and stands (data/far_wings_ledger.json's first_entry: NVDA, AAPL): "
            "the 2026-12-18 cycle of each of these began on 2026-11-17's close, so none enters it again on "
            "2026-11-23") in msg
    assert ("AAPL, found tracked by the run that wrote it but not decided, get no wings this cycle, as a name not "
            "decided at a regular entry's retry gets none") in msg
    assert ("BRK B, not tracked by the run that wrote it, enter the 2026-12-18 cycle at the regular entry on "
            "2026-11-23 (orders for 2026-11-24)") in msg
    assert "is lost" not in msg and "falls back" not in msg
    assert files_now(env) == before and fake.requests == [] and env["sessions"][1:] == []


def test_the_first_entry_stands_when_the_name_it_wrote_is_no_longer_tracked(env):
    # NVDA written at 22:00, AAPL not final; by 10:29 Sanji no longer tracks NVDA. No tradable name has a night from
    # 2026-11-17, but the ledger's first_entry shows the night was written: the first entry stands, it is not lost
    track(env, "NVDA", "AAPL")
    fake = FakeBasket(FIRST, ["NVDA", "AAPL"])
    ric = next(r["ric"] for r in chain_rows_of("AAPL", FIRST) if r["cp"] == "P")
    fake.options[ric] = fake.options[ric].iloc[:1]
    env["use"](fake, now=TONIGHT)
    assert O.main([]) == 0
    track(env, "AAPL", untracked=["NVDA"])
    before = files_now(env)
    fake.requests.clear()
    env["use"](fake, now=dt.datetime(2026, 11, 18, 10, 29))
    with pytest.raises(SystemExit) as e:
        O.main([])
    msg = red(e)
    assert "the one-off first entry was written and stands (data/far_wings_ledger.json's first_entry: NVDA, AAPL)" \
        in msg
    assert "AAPL, found tracked by the run that wrote it but not decided, get no wings this cycle" in msg
    assert "is lost" not in msg and "falls back" not in msg and "stand as written" not in msg
    assert files_now(env) == before and fake.requests == []


@pytest.mark.parametrize("first_args, first_now", [
    (["--retry"], dt.datetime(2026, 11, 18, 1, 0)),     # the 22:00 run failed (or had not started); the retry wrote
    ([], TONIGHT),                                    # the 22:00 run wrote
])
@pytest.mark.parametrize("late_now", [dt.datetime(2026, 11, 18, 10, 29), dt.datetime(2026, 11, 18, 15, 59)])
def test_a_plain_run_after_the_first_entry_was_written_says_it_stands_and_stays_green(env, capsys, first_args,
                                                                                      first_now, late_now):
    # GitHub starts the 03:00 UTC run after the 06:00 retry has written, or she presses the Run button on Wednesday
    # morning after a good night: the first entry was written, so the run has nothing to do. Before, it went red and
    # said the early first entry was lost and the cycle fell back to 2026-11-23, while the Dec 18 wings were in the
    # file and the ledger's first_entry kept NVDA from entering on the 23rd
    S.write_json(env["root"] / "price_history.json", nvda_alone())
    env["use"](FakeBasket(FIRST, ["NVDA"]), now=first_now)
    assert O.main(first_args) == 0
    assert ledger_file(env)["first_entry"]["names"] == ["NVDA"]
    before = files_now(env)
    capsys.readouterr()
    fake = env["use"](FakeBasket(FIRST, ["NVDA"]), now=late_now)
    sessions = len(env["sessions"])
    assert O.main([]) == 0
    out = capsys.readouterr().out
    assert (f"Late run ({late_now:%Y-%m-%d %H:%M} New York time): the orders for 2026-11-18 from 2026-11-17's close "
            "were already written (nvda-fw-put-2026-11-17, nvda-fw-call-2026-11-17); nothing to do. The one-off first "
            "entry stands (data/far_wings_ledger.json's first_entry: NVDA): none of these names enters the 2026-12-18 "
            "cycle again on 2026-11-23.") in out
    assert "lost" not in out and "falls back" not in out
    assert files_now(env) == before and fake.requests == [] and len(env["sessions"]) == sessions
    # and the retry, run after it, says the same
    assert O.main(["--retry"]) == 0
    assert "Retry: the orders for 2026-11-18 from 2026-11-17's close were already written" in capsys.readouterr().out
    assert files_now(env) == before


def test_a_plain_rerun_that_reaches_the_open_after_the_night_was_written_stays_green(env, capsys, monkeypatch):
    # the retry wrote at 01:00; a plain run starts at 09:29 and decides again (a rerun decides afresh), but the open
    # comes while it asks LSEG: nothing is written, the retry's orders stand, and the run is green
    S.write_json(env["root"] / "price_history.json", nvda_alone())
    env["use"](FakeBasket(FIRST, ["NVDA"]), now=dt.datetime(2026, 11, 18, 1, 0))
    assert O.main(["--retry"]) == 0
    before = files_now(env)
    fake = env["use"](FakeBasket(FIRST, ["NVDA"]))
    clock = iter([dt.datetime(2026, 11, 18, 9, 29), dt.datetime(2026, 11, 18, 9, 30)])
    monkeypatch.setattr(O, "now_ny", lambda: next(clock))
    capsys.readouterr()
    assert O.main([]) == 0
    out = capsys.readouterr().out
    assert fake.requests                                            # it asked LSEG before the open came
    assert ("Late run (2026-11-18 09:30 New York time): the orders for 2026-11-18 from 2026-11-17's close were "
            "already written (nvda-fw-put-2026-11-17, nvda-fw-call-2026-11-17); nothing to do. The one-off first "
            "entry stands") in out
    assert files_now(env) == before


@pytest.mark.parametrize("nvda_written", [False, True])
@pytest.mark.parametrize("tracked", [("NVDA", "AAPL"), ("NVDA",)])
def test_a_late_regular_entry_night_loses_no_entry_for_a_name_whose_cycle_began_at_the_first_entry(env, tracked,
                                                                                                   nvda_written):
    # 2026-11-23's runs late (the Run button Tuesday 10:29), NVDA holding the Dec 18 put of its first entry: its cycle
    # began on 11-17, so it had no entry to lose, only its band check (unless the 22:00 run wrote its night, which then
    # stands); AAPL, not tracked on 11-17, loses its entry. Before, the log said every name's entry was lost
    track(env, *tracked)
    put = _order("NVDA", "P", FIRST)
    led = FW.record_first_entry(FW.record_night(FW.read_ledger(None), "NVDA", [put], FIRST, FIRST_FOR), FIRST,
                                FIRST_FOR, DEC, ["NVDA"])
    if nvda_written:
        led = FW.record_night(led, "NVDA", [], S.ENTRY, S.FOR)
    S.write_json(env["root"] / "data" / "far_wings_ledger.json", led)
    write_mkt(env, "open_orders.json", [])
    write_mkt(env, "fills.json", [{"id": put["id"], "status": "filled"}])
    write_mkt(env, "positions.json", {"cash": 100_000, "positions": legs_of([put])})
    before = files_now(env)
    fake = env["use"](FakeBasket(S.ENTRY, list(tracked)), now=dt.datetime(2026, 11, 24, 10, 29))
    if nvda_written and tracked == ("NVDA",):
        assert O.main([]) == 0                                     # every name written: nothing to do
        assert files_now(env) == before and fake.requests == []
        return
    with pytest.raises(SystemExit) as e:
        O.main([])
    msg = red(e)
    assert "the session for 2026-11-23's orders (2026-11-24) is already open: nothing written" in msg
    assert ("2026-11-23 was an entry night, so this cycle's entry is lost for AAPL: its wings" in msg) == \
        ("AAPL" in tracked)
    assert "lost for NVDA" not in msg and "NVDA, AAPL" not in msg and "first entry" not in msg
    band = ("the night's check from 2026-11-23's close for NVDA (the close of leftover shares) is not "
            "written either")
    assert (band in msg) is not nvda_written
    assert ("the orders already written from 2026-11-23's close for NVDA stand as written" in msg) is nvda_written
    if not nvda_written and tracked == ("NVDA",):
        assert msg.count(". ") == 1                                # the late line and the band check, nothing lost
    assert files_now(env) == before and fake.requests == []


def test_the_first_entry_is_named_only_on_its_own_night():
    rec = FW.record_first_entry(FW.read_ledger(None), FIRST, FIRST_FOR, DEC, ["NVDA", "AAPL"])
    assert O.first_of(rec, FIRST) == rec["first_entry"] and O.first_of(rec, S.ENTRY) is None
    assert O.first_of(FW.read_ledger(None), FIRST) is None
    done = {"NVDA": "nvda-fw-put-2026-11-17", "AAPL": "no order"}
    assert O.already_written("Retry", FIRST, FIRST_FOR, done, rec) == (
        "Retry: the orders for 2026-11-18 from 2026-11-17's close were already written (NVDA: nvda-fw-put-2026-11-17; "
        "AAPL: no order); nothing to do. The one-off first entry stands (data/far_wings_ledger.json's first_entry: "
        "NVDA, AAPL): none of these names enters the 2026-12-18 cycle again on 2026-11-23.")
    assert O.already_written("Retry", S.ENTRY, S.FOR, {"NVDA": "no order"}, rec) == (
        "Retry: the orders for 2026-11-24 from 2026-11-23's close were already written (no order); nothing to do.")


# --------------------------------------------------------------------------- the cycle, 11-17 to 12-21
def fill_all(env, orders, positions):
    """Sanji filled every one of `orders` (and every earlier one): mkt/fills.json and the book after them."""
    path = env["root"] / "mkt" / "fills.json"
    fills = json.loads(path.read_text()) if path.exists() else []
    fills += [{"id": o["id"], "status": "filled"} for o in orders if o["id"] not in {f["id"] for f in fills}]
    write_mkt(env, "fills.json", fills)
    write_mkt(env, "positions.json", {"cash": 100_000, "positions": positions})


def legs_of(orders):
    return [{"underlying": o["underlying"], "contract": o["contract"], "expiry": o["expiry"],
             "strike": float(o["strike"]), "qty": -o["qty"]} for o in orders if o["contract"] != "underlying"]


def leg_quotes(day, orders):
    """Each held wing's daily quotes up to D (one vol), as LSEG would serve them."""
    out = []
    for o in orders:
        t = FW.owner(o["id"], TICKERS)
        c = closes_of(t, day)
        h = S.leg_history(c, o["contract"][0].upper(), DEC, float(o["strike"]), 0.4)
        out.append((build_option_ric(NAMES[t]["root"], DEC, float(o["strike"]), o["contract"][0].upper(),
                                     expired=False), pd.DataFrame({"BID": h["bid"], "ASK": h["ask"]})))
    return out


def option_rics_asked(fake, t):
    return {r[1] for r in fake.asked(t) if r[1] != NAMES[t]["ric"]}


def ric_of(o):
    t = FW.owner(o["id"], TICKERS)
    return build_option_ric(NAMES[t]["root"], dt.date.fromisoformat(o["expiry"]), float(o["strike"]),
                            o["contract"][0].upper(), expired=False)


def test_the_dec_18_cycle_from_the_first_entry_to_the_next_regular_entry(env, exact_delta, capsys):
    # 2026-11-17 22:00: NVDA and BRK B tracked, AAPL not yet
    track(env, "NVDA", "BRK B", untracked=["AAPL"])
    env["use"](FakeBasket(FIRST, ["NVDA", "BRK B"]), now=TONIGHT)
    assert O.main([]) == 0
    wings = orders_file(env)
    assert [o["id"] for o in wings] == ["nvda-fw-put-2026-11-17", "nvda-fw-call-2026-11-17", "brkb-fw-put-2026-11-17",
                                        "brkb-fw-call-2026-11-17"]
    assert by_name(wings) == {"NVDA": plan("NVDA")["orders"], "BRK B": plan("BRK B")["orders"]}
    assert ledger_file(env)["first_entry"]["names"] == ["NVDA", "BRK B"]
    assert "AAPL: conid 265598 is not in price_history.json with tracked: true" in capsys.readouterr().out
    fill_all(env, wings, legs_of(wings))
    for o in wings:
        exact_delta[(o["contract"][0].upper(), float(o["strike"]))] = 0.0
    nv_put = next(o for o in wings if o["id"] == "nvda-fw-put-2026-11-17")

    def night(day, now=None, tickers=("NVDA", "BRK B"), chain=False, extra=()):
        fake = FakeBasket(day, list(tickers), chain=chain, legs=leg_quotes(day, wings) + list(extra))
        env["use"](fake, now=now or dt.datetime.combine(day, dt.time(22, 0)))
        assert O.main([]) == 0
        return fake, orders_file(env), capsys.readouterr().out

    # 11-18: the band check only, inside it; no chain asked for, only the held wings
    fake, got, out = night(dt.date(2026, 11, 18))
    assert got == [] and output(env).endswith("for_day=2026-11-19\n")
    assert "not an entry night" in out and out.count("is inside the $5,000 band: no hedge trade (holding 0)") == 2
    assert option_rics_asked(fake, "NVDA") == {ric_of(o) for o in by_name(wings)["NVDA"]}
    # 11-19: NVDA's put now -0.05 delta: outside its band, a hedge for Friday's open
    exact_delta[("P", float(nv_put["strike"]))] = -0.05
    shares = round(nv_put["qty"] * 100 * 0.05)
    _, got, out = night(dt.date(2026, 11, 19))
    assert [(o["id"], o["action"], o["qty"], o["for"]) for o in got] == [
        ("nvda-fw-hedge-2026-11-19", "SHT", shares, "2026-11-20")]
    assert_legal(got)
    hedged = legs_of(wings) + [{"underlying": S.NVDA, "contract": "underlying", "qty": -shares}]
    fill_all(env, got, hedged)
    # 11-20, the Nov expiry's night: the Dec 18 legs are held and their hedge stays, a band check like any other
    fake, got, out = night(dt.date(2026, 11, 20))
    assert got == [] and output(env).endswith("for_day=2026-11-23\n")
    assert f"is inside the $5,000 band: no hedge trade (holding {-shares})" in out
    assert "close the" not in out and "leftover" not in out
    assert option_rics_asked(fake, "NVDA") == {ric_of(o) for o in by_name(wings)["NVDA"]}
    # 11-23, the regular entry: NVDA and BRK B hedge only; AAPL, tracked now, enters the Dec 18 cycle
    track(env, "NVDA", "BRK B", "AAPL")
    fake, got, out = night(dt.date(2026, 11, 23), tickers=("NVDA", "BRK B", "AAPL"), chain=True)
    assert "Quotes from the close of 2026-11-23, orders for 2026-11-24: entry night, expiry 2026-12-18\n" in out
    for t in ("NVDA", "BRK B"):
        assert (f"  {t}: its 2026-12-18 cycle began at the one-off first entry on 2026-11-17 (tracked that night, "
                f"data/far_wings_ledger.json's first_entry), so no entry tonight: the band check only") in out
        assert option_rics_asked(fake, t) == {ric_of(o) for o in by_name(wings)[t]}          # no chain asked for
    assert by_name(got) == {"AAPL": plan("AAPL", S.ENTRY, S.FOR)["orders"]}
    assert [o["id"] for o in got] == ["aapl-fw-put-2026-11-23", "aapl-fw-call-2026-11-23"]
    assert_legal(got)
    assert ledger_file(env)["first_entry"]["names"] == ["NVDA", "BRK B"]
    aapl = got
    wings = wings + aapl
    fill_all(env, aapl, hedged + legs_of(aapl))
    for o in aapl:
        exact_delta[(o["contract"][0].upper(), float(o["strike"]))] = 0.0
    # Thanksgiving week: Wednesday night for Friday (its 13:00 early close), then Friday after it for Monday
    _, got, out = night(dt.date(2026, 11, 25), tickers=("NVDA", "BRK B", "AAPL"))
    assert got == [] and output(env).endswith("for_day=2026-11-27\n")
    _, got, out = night(dt.date(2026, 11, 27), now=dt.datetime(2026, 11, 27, 13, 5), tickers=("NVDA", "BRK B", "AAPL"))
    assert got == [] and output(env).endswith("for_day=2026-11-30\n")
    # 12-18, the Dec expiry: Sanji has settled the wings; the shares left are closed at Monday's open
    fill_all(env, [], [{"underlying": S.NVDA, "contract": "underlying", "qty": -shares}])
    fake = FakeBasket(DEC, ["NVDA", "BRK B", "AAPL"], chain=False)
    env["use"](fake, now=dt.datetime(2026, 12, 18, 22, 0))
    assert O.main([]) == 0
    got = orders_file(env)
    assert [(o["id"], o["action"], o["qty"], o["for"]) for o in got] == [
        ("nvda-fw-hedge-2026-12-18", "BTC", shares, "2026-12-21")]
    assert "no options held: close the" in capsys.readouterr().out
    fill_all(env, got, [])
    # 12-21: the registered calendar, every name into 2027-01-15
    day = dt.date(2026, 12, 21)
    fake = FakeBasket(day, ["NVDA", "BRK B", "AAPL"], chain=False,
                      legs=chain_frames(["NVDA", "BRK B", "AAPL"], day, JAN))
    env["use"](fake, now=dt.datetime(2026, 12, 21, 22, 0))
    assert O.main([]) == 0
    got = orders_file(env)
    assert [(o["id"], o["for"], o["expiry"]) for o in got] == [
        (f"{FW.id_root(t)}-fw-{w}-2026-12-21", "2026-12-22", "2027-01-15") for t in ("NVDA", "AAPL", "BRK B")
        for w in ("put", "call")]                                            # config order
    assert "T 25 days" in capsys.readouterr().out
    for t in ("NVDA", "BRK B", "AAPL"):
        assert by_name(got)[t] == plan(t, day, dt.date(2026, 12, 22), JAN)["orders"], t
    assert_legal(got)


@pytest.mark.parametrize("how", ["not_final_at_the_retry", "no_eligible_strike"])
def test_a_name_tracked_at_the_first_entry_that_got_no_order_does_not_enter_on_the_23rd(env, capsys, how):
    track(env, "NVDA", "AAPL")
    fake = FakeBasket(FIRST, ["NVDA", "AAPL"])
    aapl = [r for r in fake.options if r.startswith("AAPL")]
    if how == "not_final_at_the_retry":
        for ric in aapl:                            # no D row on any AAPL option, at 22:00 or at 01:00
            fake.options[ric] = fake.options[ric].iloc[:1]
    else:
        spot = float(closes_of("AAPL", FIRST).iloc[-1])
        for r in chain_rows_of("AAPL", FIRST):      # only strikes near the money listed: no |delta| under the caps
            if abs(r["strike"] / spot - 1) > 0.04:
                del fake.options[r["ric"]]
    env["use"](fake, now=TONIGHT)
    assert O.main([]) == 0
    if how == "not_final_at_the_retry":
        env["use"](fake, now=dt.datetime(2026, 11, 18, 1, 0))
        with pytest.raises(SystemExit, match="The retry decided nothing more, and AAPL: still not final"):
            O.main(["--retry"])
    else:
        assert ledger_file(env)["names"]["AAPL"]["nights"]["2026-11-17"] == {"for": "2026-11-18", "ids": []}
        assert "no put wing" in capsys.readouterr().out
    assert list(by_name(orders_file(env))) == ["NVDA"]
    assert ledger_file(env)["first_entry"]["names"] == ["NVDA", "AAPL"]
    capsys.readouterr()
    # Monday: neither enters; only NVDA's unfilled first-entry orders are gone from the file
    write_mkt(env, "fills.json", [{"id": o["id"], "status": "unfilled"} for o in orders_file(env)])
    write_mkt(env, "positions.json", {"cash": 100_000, "positions": []})
    fake = FakeBasket(S.ENTRY, ["NVDA", "AAPL"])
    env["use"](fake, now=dt.datetime(2026, 11, 23, 22, 0))
    assert O.main([]) == 0
    assert orders_file(env) == [] and fake.asked("AAPL") == [] and fake.asked("NVDA") == []
    out = capsys.readouterr().out
    assert "  AAPL: its 2026-12-18 cycle began at the one-off first entry on 2026-11-17" in out
    assert "  NVDA: its 2026-12-18 cycle began at the one-off first entry on 2026-11-17" in out
