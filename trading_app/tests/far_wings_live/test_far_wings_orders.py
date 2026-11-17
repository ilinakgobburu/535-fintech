"""
far_wings_orders.py end to end with LSEG replaced by a fake (no network), with NVDA the one name Sanji tracks (the
basket's other 30 names each get one "no orders tonight" line): what it asks for, what it writes, and that every
failure stops it before mkt/open_orders.json is touched, as the single-name script did. The fake serves the synthetic
NVDA closes, the T-bill, dividends and a Black-76 chain (fw_live_synth); a RIC it does not hold answers as LSEG does
for an unlisted contract ("The universe is not found"). The basket's own nights are test_far_wings_basket.py.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

import fw_live_synth as S
from trading_app.lib import far_wings_live as FW
from trading_app.lib.ric import build_option_ric

_path = list(sys.path)
sys.path.insert(0, str(S.REPO))
import far_wings_orders as O  # noqa: E402

sys.path[:] = _path

ENTRY, FOR, EXP = S.ENTRY, S.FOR, S.EXPIRY
OLD_ORDERS = [{"id": "nvda-csp-2026-11-16", "for": "2026-11-16", "order_type": "lmt", "action": "SHT", "qty": 6,
               "underlying": S.NVDA, "contract": "put", "expiry": "2026-12-18", "strike": "155", "ref": "open",
               "lmt_prc": "7.78"}]


class FakeLSEG:
    """lseg.data's get_history and get_data, from made-up frames. Records every request."""

    def __init__(self, day, chain=True, last=None, missing_close=False, legs=(), chain_day=None):
        self.day = pd.Timestamp(day)
        self.closes = S.closes(end=day, last=last)
        if missing_close:                              # LSEG has not posted D's bar yet
            self.closes = self.closes[self.closes.index < self.day]
        self.requests, self.errors, self.options = [], {}, {}
        if chain:
            cday = pd.Timestamp(chain_day or day)
            for r in S.chain_rows(float(S.closes(end=day, last=last).iloc[-1]), day=day):
                prev = cday - pd.Timedelta(days=3)
                self.options[r["ric"]] = pd.DataFrame({"BID": [r["bid"] * 3, r["bid"]], "ASK": [r["ask"] * 3, r["ask"]]},
                                                      index=pd.DatetimeIndex([prev, cday]))
        for ric, frame in legs:
            self.options[ric] = frame

    @staticmethod
    def _window(df, start, end):
        return df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]

    def get_history(self, universe, fields, interval="daily", start=None, end=None, adjustments=None, count=None):
        self.requests.append(("history", universe, tuple(fields), adjustments))
        if universe in self.errors:
            raise self.errors[universe]
        if universe == "NVDA.O":
            return self._window(pd.DataFrame({"TRDPRC_1": self.closes}), start, end)
        if universe == "US3MT=RR":
            return self._window(pd.DataFrame({"MID_YLD_1": S.RATE_PCT}, index=self.closes.index), start, end)
        if universe in self.options:
            return self._window(self.options[universe], start, end)
        raise Exception("No data to return, please check errors: ERROR: No successful response.\n"
                        "(TS.Interday.UserRequestError.70005, The universe is not found)")

    def get_data(self, universe, fields, parameters=None):
        self.requests.append(("data", universe, tuple(fields), parameters))
        if fields[0] == "TR.DivExDate":
            d = S.DIVS
            return pd.DataFrame({"Instrument": universe, "Dividend Ex Date": [str(x.date()) for x in d.index],
                                 "Gross Dividend Amount": d.to_numpy(), "Dividend Pay Date": None})
        raise AssertionError(fields)

    def option_requests(self):
        return [r for r in self.requests if r[0] == "history" and r[1] not in ("NVDA.O", "US3MT=RR")]

    def lseg_requests(self):
        return [r for r in self.requests if r[0] != "sleep"]


class Session:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A scratch repo root: mkt/ with the wheel's last order, the repo's config, a fake LSEG."""
    state = {"sessions": [], "sleeps": []}
    (tmp_path / "mkt").mkdir()
    S.write_json(tmp_path / "mkt" / "open_orders.json", OLD_ORDERS)
    monkeypatch.setattr(O, "ROOT", tmp_path)
    monkeypatch.setattr(O, "MKT", tmp_path / "mkt")
    state["config"] = S.write_json(tmp_path / "config.yaml", S.config())          # JSON is YAML
    monkeypatch.setattr(O, "CONFIG", state["config"])
    S.write_json(tmp_path / "price_history.json", S.price_history(S.NVDA))       # Sanji tracks NVDA alone here

    def open_lseg():
        s = Session()
        state["sessions"].append(s)
        return s

    monkeypatch.setattr(O, "open_lseg", open_lseg)

    def sleep(s):
        state["sleeps"].append(s)
        if state.get("fake") is not None:
            state["fake"].requests.append(("sleep", s))

    monkeypatch.setattr(O.time, "sleep", sleep)
    out = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    state["root"], state["output"] = tmp_path, out

    def use(fake, now=None):
        state["fake"] = fake
        monkeypatch.setattr(O, "ld", fake)
        monkeypatch.setattr(O, "now_ny", lambda: now or dt.datetime(2026, 11, 23, 22, 0))
        return fake

    state["use"] = use
    return state


def orders_file(env):
    return json.loads((env["root"] / "mkt" / "open_orders.json").read_text())


def ledger_file(env):
    path = env["root"] / "data" / "far_wings_ledger.json"
    return json.loads(path.read_text()) if path.exists() else None


def write_mkt(env, name, obj):
    S.write_json(env["root"] / "mkt" / name, obj)


def option(cp, strike, qty, expiry="2026-12-18"):
    return {"underlying": S.NVDA, "contract": "put" if cp == "P" else "call", "expiry": expiry, "strike": strike,
            "qty": qty}


def set_cfg(env, **over):
    cfg = json.loads(env["config"].read_text())
    cfg.update(over)
    S.write_json(env["config"], cfg)


def entry_night(env):
    """The Nov 23 entry, written by the script; then Sanji fills both wings for Nov 24."""
    env["use"](FakeLSEG(ENTRY))
    assert O.main([]) == 0
    written = orders_file(env)
    write_mkt(env, "fills.json", [{"id": o["id"], "status": "filled"} for o in written])
    write_mkt(env, "positions.json", {"cash": 100_000, "positions": [
        option(o["contract"][0].upper(), float(o["strike"]), -o["qty"]) for o in written]})
    return written


def leg_frames(day, legs, last=None):
    c = S.closes(end=day, last=last)
    out = []
    for cp, k, vol in legs:
        h = S.leg_history(c, cp, EXP, k, vol)
        out.append((build_option_ric("NVDA", EXP, k, cp, expired=False),
                    pd.DataFrame({"BID": h["bid"], "ASK": h["ask"]})))
    return out


# --------------------------------------------------------------------------- the entry night
def test_the_entry_night_writes_the_two_wings(env, capsys):
    fake = env["use"](FakeLSEG(ENTRY))
    assert O.main([]) == 0
    got = orders_file(env)
    assert [(o["id"], o["action"], o["contract"], o["strike"], o["for"], o["order_type"]) for o in got] == [
        ("nvda-fw-put-2026-11-23", "SHT", "put", "145", "2026-11-24", "lmt"),
        ("nvda-fw-call-2026-11-23", "SHT", "call", "220", "2026-11-24", "lmt")]
    assert all(S.order_problems(o) == [] for o in got)
    # the same orders the pure plan gives on the same data (the wheel's old order for Nov 16 is dropped)
    want = FW.plan_night(S.config(), ENTRY, FOR, FW.Book(), name=S.NVDA_NAME, entry_expiry=EXP,
                         mkt=S.market())["orders"]
    assert got == want
    assert env["output"].read_text() == "for_day=2026-11-24\n"
    assert len(env["sessions"]) == 1 and env["sessions"][0].closed
    asked = fake.option_requests()
    assert all(r[2] == ("BID", "ASK") for r in asked)
    plan = FW.strike_plan(fake.closes, dt.date(2026, 11, 20), ENTRY, S.config()["chain"]["bands"],
                          S.NVDA_NAME["grid"])
    assert [r[1] for r in asked] == [build_option_ric("NVDA", EXP, k, cp, expired=False)
                                     for cp in ("P", "C") for k in plan[cp]]
    stock = [r for r in fake.requests if r[1] == "NVDA.O" and r[0] == "history"]
    assert stock == [("history", "NVDA.O", ("TRDPRC_1",), ["exchangeCorrection"])]
    led = ledger_file(env)
    assert led["format"] == "far_wings_ledger/2" and list(led["names"]) == ["NVDA"]
    assert led["names"]["NVDA"]["nights"]["2026-11-23"] == {"for": "2026-11-24", "ids": [o["id"] for o in got]}
    assert led["names"]["NVDA"]["orders"] == {o["id"]: o for o in got}
    out = capsys.readouterr().out
    assert "Dropped nvda-csp-2026-11-16 (for 2026-11-16, not this strategy's)" in out
    assert "sell 41 145 puts at 0.53" in out and "sell 59 220 calls at 0.24" in out


def test_every_lseg_request_is_followed_by_the_pause(env):
    fake = env["use"](FakeLSEG(ENTRY))
    assert O.main([]) == 0
    log = [r[0] for r in fake.requests]
    assert len(log) > 20 and log[0] != "sleep" and log[-1] == "sleep"
    assert all((a == "sleep") != (b == "sleep") for a, b in zip(log, log[1:]))
    assert env["sleeps"] == [O.PAUSE] * len(fake.lseg_requests())


def test_the_entrys_limits_are_ds_mids_not_earlier_quotes(env):
    # the fake's earlier row is three times D's quotes
    env["use"](FakeLSEG(ENTRY))
    O.main([])
    spot = float(S.closes().iloc[-1])
    rows = {(r["cp"], r["strike"]): r for r in S.chain_rows(spot)}
    for o in orders_file(env):
        r = rows[(o["contract"][0].upper(), float(o["strike"]))]
        assert float(o["lmt_prc"]) == pytest.approx((r["bid"] + r["ask"]) / 2, abs=0.005)


# 2026-11-17 is the one-off first entry's night (test_far_wings_first_entry.py); 2026-11-20 the Nov expiry's
@pytest.mark.parametrize("day", [dt.date(2026, 11, 18), dt.date(2026, 11, 20), dt.date(2026, 11, 24),
                                 dt.date(2026, 12, 2), dt.date(2026, 12, 18)])
def test_a_night_that_is_not_an_entry_with_an_empty_book_writes_nothing_new_and_skips_lseg(env, day):
    env["use"](FakeLSEG(day, chain=False), now=dt.datetime.combine(day, dt.time(22, 0)))
    assert O.main([]) == 0
    assert orders_file(env) == [] and env["sessions"] == []
    nxt = FW.next_session(day)
    assert env["output"].read_text() == f"for_day={nxt.isoformat()}\n"
    assert ledger_file(env)["names"] == {"NVDA": {"orders": {}, "nights": {day.isoformat(): {"for": nxt.isoformat(),
                                                                                         "ids": []}}}}


def test_the_dry_run_prints_and_writes_nothing(env, capsys):
    env["use"](FakeLSEG(ENTRY))
    before = (env["root"] / "mkt" / "open_orders.json").read_bytes()
    assert O.main(["--dry-run", "--as-of", "2026-11-23"]) == 0
    assert (env["root"] / "mkt" / "open_orders.json").read_bytes() == before
    assert not env["output"].exists() and ledger_file(env) is None
    out = capsys.readouterr().out
    assert "Dry run, nothing written" in out and '"id": "nvda-fw-put-2026-11-23"' in out


def test_force_entry_checks_the_next_entrys_chain_on_the_latest_close(env, capsys):
    day = dt.date(2026, 11, 16)
    fake = FakeLSEG(day, chain=False)
    for r in S.chain_rows(float(S.closes(end=day).iloc[-1]), day=day):
        fake.options[r["ric"]] = pd.DataFrame({"BID": [r["bid"]], "ASK": [r["ask"]]}, index=pd.DatetimeIndex([day]))
    env["use"](fake, now=dt.datetime(2026, 11, 16, 22, 0))
    assert O.main(["--dry-run", "--as-of", "2026-11-16", "--force-entry"]) == 0
    out = capsys.readouterr().out
    assert "expiry 2026-12-18 (forced)" in out and '"id": "nvda-fw-put-2026-11-16"' in out
    assert all("L1826" in r[1] or "X1826" in r[1] for r in fake.option_requests())     # the Dec 18 chain


@pytest.mark.parametrize("args", [["--force-entry"], ["--as-of", "2026-11-23"], ["--dry-run", "--as-of", "2026-11-22"]])
def test_one_off_flags_need_a_dry_run_on_a_session(env, args):
    env["use"](FakeLSEG(ENTRY))
    with pytest.raises(SystemExit):
        O.main(args)
    assert orders_file(env) == OLD_ORDERS


LATE_LINE = "is already open: nothing written (a late run must not fill a day late)"


def red(excinfo) -> str:
    """The message of a sys.exit(message): the run ended non-zero (exit status 1) with it."""
    assert excinfo.value.code not in (0, None)
    return str(excinfo.value.code)


@pytest.mark.parametrize("now, args", [
    (dt.datetime(2026, 11, 23, 22, 0), []),           # the 22:00 run
    (dt.datetime(2026, 11, 24, 1, 0), ["--retry"]),    # the 01:00 retry, nothing written yet
    (dt.datetime(2026, 11, 24, 6, 23), []),           # the 22:00 run as GitHub started it on 2026-11-17 (11:23 UTC)
    (dt.datetime(2026, 11, 24, 9, 29), ["--retry"]),   # the retry a minute before the open
])
def test_a_run_up_to_the_open_writes_ds_orders_for_the_session_after_d(env, now, args):
    env["use"](FakeLSEG(ENTRY), now=now)
    assert O.main(args) == 0
    assert [(o["id"], o["for"]) for o in orders_file(env)] == [("nvda-fw-put-2026-11-23", "2026-11-24"),
                                                               ("nvda-fw-call-2026-11-23", "2026-11-24")]
    assert env["output"].read_text() == "for_day=2026-11-24\n"


@pytest.mark.parametrize("now, args", [
    (dt.datetime(2026, 11, 24, 9, 30), []),            # the open itself
    (dt.datetime(2026, 11, 24, 10, 29), ["--retry"]),  # the retry as GitHub started it on 2026-11-17 (15:29 UTC)
    (dt.datetime(2026, 11, 24, 15, 59), []),           # Tuesday's close not in yet: D is still Monday
])
def test_a_late_run_writes_nothing_and_goes_red(env, now, args):
    # the vol-arb's retry, started at 15:29 UTC on 2026-11-17, wrote "Quotes from the close of 2026-11-16, orders for
    # 2026-11-18": 'for' from the clock. Here D is Monday's close and its orders could only be for Tuesday, which
    # has opened: nothing is written to mkt/ or the ledger, LSEG is not asked, and the run ends red
    before = files_now(env)
    fake = env["use"](FakeLSEG(ENTRY), now=now)
    with pytest.raises(SystemExit) as e:
        O.main(args)
    msg = red(e)
    assert f"the session for 2026-11-23's orders (2026-11-24) {LATE_LINE}" in msg
    assert ("2026-11-23 was an entry night, so this cycle's entry is lost for NVDA: its wings are placed only from "
            "2026-11-23's close, for 2026-11-24, and not again this cycle (the next entry is 2026-12-21, for the "
            "2027-01-15 expiry)") in msg
    assert "stand as written" not in msg                           # nothing was written from Monday's close
    assert "band check" not in msg                                 # an empty book: no hedge waits
    assert files_now(env) == before and ledger_file(env) is None and not env["output"].exists()
    assert fake.requests == [] and env["sessions"] == []


def test_a_late_run_with_leftover_shares_leaves_them_for_the_next_night(env, capsys):
    # the expiry's leftover -350 shares were not closed; both Monday-night runs failed; the Run button Tuesday 10:00.
    # Before, it wrote the BTC for Wednesday from Monday's close; now nothing, and Tuesday night closes them
    write_mkt(env, "open_orders.json", [])
    hedge = {"id": "nvda-fw-hedge-2026-11-19", "for": "2026-11-20", "order_type": "mkt", "action": "SHT", "qty": 350,
             "underlying": S.NVDA, "contract": "underlying", "ref": "open"}
    S.write_json(env["root"] / "data" / "far_wings_ledger.json",
                 FW.record_night(FW.read_ledger(None), "NVDA", [hedge], dt.date(2026, 11, 19), dt.date(2026, 11, 20)))
    write_mkt(env, "fills.json", [{"id": hedge["id"], "status": "filled"}])
    write_mkt(env, "positions.json", {"cash": 0, "positions": [
        {"underlying": S.NVDA, "contract": "underlying", "qty": -350}]})
    before = files_now(env)
    fake = env["use"](FakeLSEG(ENTRY), now=dt.datetime(2026, 11, 24, 10, 0))
    with pytest.raises(SystemExit) as e:
        O.main([])
    msg = red(e)
    assert ("the band check from 2026-11-23's close for NVDA (any hedge, or the close of leftover shares) is not "
            "written either: it waits for the next night's run, on 2026-11-24's close") in msg
    assert files_now(env) == before and fake.requests == []
    # Tuesday 22:00: D is Tuesday, the shares are closed at Wednesday's open
    day = dt.date(2026, 11, 24)
    env["use"](FakeLSEG(day, chain=False), now=dt.datetime(2026, 11, 24, 22, 0))
    assert O.main([]) == 0
    assert [(o["id"], o["action"], o["qty"], o["for"]) for o in orders_file(env)] == [
        ("nvda-fw-hedge-2026-11-24", "BTC", 350, "2026-11-25")]


def test_a_run_that_reaches_the_open_while_asking_lseg_writes_nothing(env, monkeypatch):
    # 09:29 when it starts (D Monday, for Tuesday), 09:30 by the time it would write: what it decided is a day late
    before = files_now(env)
    fake = env["use"](FakeLSEG(ENTRY))
    clock = iter([dt.datetime(2026, 11, 24, 9, 29), dt.datetime(2026, 11, 24, 9, 30)])
    monkeypatch.setattr(O, "now_ny", lambda: next(clock))
    with pytest.raises(SystemExit) as e:
        O.main(["--retry"])
    assert f"the session for 2026-11-23's orders (2026-11-24) {LATE_LINE}" in red(e)
    assert files_now(env) == before and not env["output"].exists()
    assert fake.option_requests() and env["sessions"][0].closed       # LSEG was asked, and the session closed


def test_a_dry_run_after_the_open_still_shows_and_says_a_real_run_would_be_red(env, capsys):
    env["use"](FakeLSEG(ENTRY), now=dt.datetime(2026, 11, 24, 10, 29))
    assert O.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert ("Note (dry run): the session for 2026-11-23's orders (2026-11-24) is already open, so a run now would "
            "write nothing and end red") in out
    assert '"for": "2026-11-24"' in out and orders_file(env) == OLD_ORDERS


@pytest.mark.parametrize("now, d, for_day, refused", [
    (dt.datetime(2026, 11, 20, 22, 0), "2026-11-20", "2026-11-23", False),     # Friday night: Monday
    (dt.datetime(2026, 11, 22, 23, 0), "2026-11-20", "2026-11-23", False),     # Sunday night
    (dt.datetime(2026, 11, 23, 9, 30), "2026-11-20", "2026-11-23", True),      # Monday's open
    (dt.datetime(2027, 1, 15, 22, 0), "2027-01-15", "2027-01-19", False),      # before the MLK Monday: Tuesday
    (dt.datetime(2027, 1, 18, 12, 0), "2027-01-15", "2027-01-19", False),      # on the holiday itself
    (dt.datetime(2026, 11, 25, 22, 0), "2026-11-25", "2026-11-27", False),     # Thanksgiving eve: Friday
    (dt.datetime(2026, 11, 27, 12, 0), "2026-11-25", "2026-11-27", True),      # Friday, before its 13:00 close
    (dt.datetime(2026, 11, 27, 13, 0), "2026-11-27", "2026-11-30", False),     # its early close: Monday
])
def test_weekends_holidays_and_the_early_close_end_to_end(env, now, d, for_day, refused):
    write_mkt(env, "open_orders.json", [])
    env["use"](FakeLSEG(dt.date.fromisoformat(d), chain=False), now=now)
    if refused:
        with pytest.raises(SystemExit) as e:
            O.main([])
        assert f"the session for {d}'s orders ({for_day}) {LATE_LINE}" in red(e)
        assert ledger_file(env) is None
        return
    assert O.main([]) == 0
    assert env["output"].read_text() == f"for_day={for_day}\n"
    assert ledger_file(env)["names"]["NVDA"]["nights"] == {d: {"for": for_day, "ids": []}}


EDT = dt.timezone(dt.timedelta(hours=-4))
DELAYED_RETRIES = [
    # (the night's 22:00 run, the retry GitHub delayed past the next day's close, the retry's D, its for, the new close)
    (dt.datetime(2026, 11, 26, 22, 0), dt.datetime(2026, 11, 27, 13, 5), "2026-11-25", "2026-11-27", "2026-11-27"),
    (dt.datetime(2026, 12, 23, 22, 0), dt.datetime(2026, 12, 24, 13, 5), "2026-12-23", "2026-12-24", "2026-12-24"),
    # summer: 06:00 UTC is 02:00 EDT, and 14 hours later is 16:00, the ordinary close
    (dt.datetime(2027, 7, 13, 23, 0), dt.datetime(2027, 7, 14, 20, 0, tzinfo=dt.timezone.utc).astimezone(EDT)
     .replace(tzinfo=None), "2027-07-13", "2027-07-14", "2027-07-14"),
]


@pytest.mark.parametrize("night, retry, d, for_day, new_close", DELAYED_RETRIES)
def test_a_retry_delayed_past_the_next_close_after_a_written_night_has_nothing_to_do(env, capsys, night, retry, d,
                                                                                     for_day, new_close):
    # Before, the retry took the close of minutes before as its D, found no night for it in the ledger and decided it
    # in its lenient mode (orders for the session after it); now it retries its own night, already written: green
    write_mkt(env, "open_orders.json", [])
    env["use"](FakeLSEG(dt.date.fromisoformat(d), chain=False), now=night)
    assert O.main([]) == 0
    assert ledger_file(env)["names"]["NVDA"]["nights"] == {d: {"for": for_day, "ids": []}}
    before = files_now(env)
    fake = env["use"](FakeLSEG(dt.date.fromisoformat(new_close), chain=False), now=retry)
    capsys.readouterr()
    assert O.main(["--retry"]) == 0
    out = capsys.readouterr().out
    assert f"Retry: the orders for {for_day} from {d}'s close were already written (no order); nothing to do." in out
    assert files_now(env) == before and fake.requests == []
    assert retry.date() == dt.date.fromisoformat(new_close) and FW.run_dates(retry)[0].isoformat() == new_close


@pytest.mark.parametrize("night, retry, d, for_day, new_close", DELAYED_RETRIES)
def test_a_retry_delayed_past_the_next_close_after_a_failed_night_writes_nothing_and_goes_red(env, night, retry, d,
                                                                                              for_day, new_close):
    # nothing written from D's close (both runs failed or late): the retry is late for its own night, and leaves the
    # new close to that night's 22:00 run (strict) instead of deciding it from quotes minutes old
    write_mkt(env, "open_orders.json", [])
    before = files_now(env)
    fake = env["use"](FakeLSEG(dt.date.fromisoformat(new_close), chain=False), now=retry)
    with pytest.raises(SystemExit) as e:
        O.main(["--retry"])
    assert f"the session for {d}'s orders ({for_day}) {LATE_LINE}" in red(e)
    assert files_now(env) == before and ledger_file(env) is None and not env["output"].exists()
    assert fake.requests == [] and env["sessions"] == []
    # the 22:00 run that night decides the new close, as on any night
    env["use"](FakeLSEG(dt.date.fromisoformat(new_close), chain=False), now=retry.replace(hour=22, minute=0))
    assert O.main([]) == 0
    assert list(ledger_file(env)["names"]["NVDA"]["nights"]) == [new_close]


# --------------------------------------------------------------------------- the nights after: the band
def test_the_night_after_the_fill_checks_the_band_and_trades_nothing_inside_it(env, capsys):
    written = entry_night(env)
    day = dt.date(2026, 11, 24)
    fake = env["use"](FakeLSEG(day, chain=False, legs=leg_frames(day, [("P", 145.0, 0.51), ("C", 220.0, 0.42)])),
                      now=dt.datetime(2026, 11, 24, 22, 0))
    assert O.main([]) == 0
    out = capsys.readouterr().out
    assert "is inside the $5,000 band: no hedge trade (holding 0)" in out
    assert orders_file(env) == []
    assert {r[1] for r in fake.option_requests()} == {build_option_ric("NVDA", EXP, float(o["strike"]),
                                                                       o["contract"][0].upper(), expired=False)
                                                      for o in written}
    assert len(env["sessions"]) == 2 and env["sessions"][-1].closed


def test_a_move_past_the_band_writes_the_hedge(env, capsys):
    entry_night(env)
    day = dt.date(2026, 11, 25)
    legs = [("P", 145.0, 0.51), ("C", 220.0, 0.42)]
    env["use"](FakeLSEG(day, chain=False, last=205.0, legs=leg_frames(day, legs, last=205.0)),
               now=dt.datetime(2026, 11, 25, 22, 0))
    assert O.main([]) == 0
    got = orders_file(env)
    assert [(o["id"], o["action"], o["order_type"], o["for"]) for o in got] == [
        ("nvda-fw-hedge-2026-11-25", "BUY", "mkt", "2026-11-27")]
    # the same order the pure plan gives on the same book and quotes
    c = S.closes(end=day, last=205.0)
    hist = {(cp, EXP, k): S.leg_history(c, cp, EXP, k, v) for cp, k, v in legs}
    book = FW.Book(cash=100_000, options=[{"cp": "P", "expiry": EXP, "strike": 145.0, "qty": -41},
                                          {"cp": "C", "expiry": EXP, "strike": 220.0, "qty": -59}])
    want = FW.plan_night(S.config(), day, dt.date(2026, 11, 27), book, name=S.NVDA_NAME,
                         mkt=S.market(day, c=c, chain=False), leg_history=hist)
    assert got == want["orders"] and want["hedge"]["breached"] is True
    assert S.order_problems(got[0]) == []
    assert "is outside the $5,000 band: hedge to" in capsys.readouterr().out


def test_the_night_of_the_expiry_closes_the_leftover_shares_from_the_close_alone(env):
    write_mkt(env, "open_orders.json", [])
    hedge = {"id": "nvda-fw-hedge-2026-12-10", "for": "2026-12-11", "order_type": "mkt", "action": "SHT", "qty": 350,
             "underlying": S.NVDA, "contract": "underlying", "ref": "open"}
    S.write_json(env["root"] / "data" / "far_wings_ledger.json",
                 FW.record_night(FW.read_ledger(None), "NVDA", [hedge], dt.date(2026, 12, 10), dt.date(2026, 12, 11)))
    write_mkt(env, "fills.json", [{"id": hedge["id"], "status": "filled"}])
    write_mkt(env, "positions.json", {"cash": 0, "positions": [
        {"underlying": S.NVDA, "contract": "underlying", "qty": -350}]})
    fake = env["use"](FakeLSEG(EXP, chain=False), now=dt.datetime(2026, 12, 18, 22, 0))
    assert O.main([]) == 0
    assert [(o["id"], o["action"], o["qty"], o["for"]) for o in orders_file(env)] == [
        ("nvda-fw-hedge-2026-12-18", "BTC", 350, "2026-12-21")]
    assert [r[1] for r in fake.lseg_requests()] == ["NVDA.O"]              # only the close, for the shares' limit


def test_an_expired_option_still_in_the_book_stops(env):
    written = entry_night(env)
    env["use"](FakeLSEG(EXP, chain=False), now=dt.datetime(2026, 12, 18, 22, 0))
    with pytest.raises(SystemExit, match="still lists options that expired by 2026-12-18"):
        O.main([])
    assert orders_file(env) == written


def test_huge_shares_held_stop_before_any_order(env):
    hedge = {"id": "nvda-fw-hedge-2026-12-10", "for": "2026-12-11", "order_type": "mkt", "action": "BUY", "qty": 120,
             "underlying": S.NVDA, "contract": "underlying", "ref": "open"}
    S.write_json(env["root"] / "data" / "far_wings_ledger.json",
                 FW.record_night(FW.read_ledger(None), "NVDA", [hedge], dt.date(2026, 12, 10), dt.date(2026, 12, 11)))
    write_mkt(env, "fills.json", [{"id": hedge["id"], "status": "filled"}])
    write_mkt(env, "positions.json", {"cash": 0, "positions": [
        {"underlying": S.NVDA, "contract": "underlying", "qty": 120 * 100}]})           # read a hundred times over
    write_mkt(env, "open_orders.json", [])
    day = dt.date(2026, 12, 14)
    env["use"](FakeLSEG(day, chain=False, last=400.0), now=dt.datetime(2026, 12, 14, 22, 0))
    with pytest.raises(SystemExit, match="the shares held, \\+12000 NVDA shares .* more than 40 x"):
        O.main([])
    assert orders_file(env) == []


# --------------------------------------------------------------------------- the guards and the book
def test_the_wheels_filled_put_stops_the_first_far_wing_night(env):
    write_mkt(env, "fills.json", [{"id": OLD_ORDERS[0]["id"], "status": "filled"}])
    write_mkt(env, "positions.json", {"cash": 0, "positions": [option("P", 155, -6)]})
    env["use"](FakeLSEG(ENTRY))
    with pytest.raises(SystemExit, match="the book holds -6 2026-12-18 155 put, which no filled nvda-fw-\\* order"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS and env["sessions"] == []


def test_a_rejected_wing_stops_until_acknowledged(env, capsys):
    written = entry_night(env)
    write_mkt(env, "fills.json", [{"id": written[0]["id"], "status": "rejected"},
                                  {"id": written[1]["id"], "status": "filled"}])
    write_mkt(env, "positions.json", {"cash": 0, "positions": [option("C", 220.0, -written[1]["qty"])]})
    day = dt.date(2026, 11, 24)
    env["use"](FakeLSEG(day, chain=False, legs=leg_frames(day, [("C", 220.0, 0.42)])),
               now=dt.datetime(2026, 11, 24, 22, 0))
    for _ in range(2):
        with pytest.raises(SystemExit, match="Sanji rejected nvda-fw-put-2026-11-23"):
            O.main([])
        assert orders_file(env) == written
    set_cfg(env, acknowledged={written[0]["id"]: "rejected; the call filled alone"})
    assert O.main([]) == 0
    assert "acknowledged in config/far_wings_live.yaml: rejected; the call filled alone" in capsys.readouterr().out
    # the call alone is far outside the band (some -195 shares of delta): hedged as it stands
    assert [(o["id"], o["action"]) for o in orders_file(env)] == [("nvda-fw-hedge-2026-11-24", "BUY")]


def test_an_unanswered_wing_stops(env):
    written = entry_night(env)
    write_mkt(env, "fills.json", [{"id": written[1]["id"], "status": "filled"}])
    env["use"](FakeLSEG(dt.date(2026, 11, 24), chain=False), now=dt.datetime(2026, 11, 24, 22, 0))
    with pytest.raises(SystemExit, match="no result yet for nvda-fw-put-2026-11-23"):
        O.main([])


def test_a_book_without_a_fills_file_stops(env):
    write_mkt(env, "positions.json", {"cash": 0, "positions": [option("P", 145.0, -41)]})
    env["use"](FakeLSEG(dt.date(2026, 11, 24), chain=False), now=dt.datetime(2026, 11, 24, 22, 0))
    with pytest.raises(SystemExit, match="Can't read the book|there is no mkt/fills.json"):
        O.main([])


def test_a_bad_ledger_stops(env):
    S.write_json(env["root"] / "data" / "far_wings_ledger.json", {"format": "vol_arb_ledger/1", "orders": {},
                                                                  "nights": {}})
    env["use"](FakeLSEG(ENTRY))
    with pytest.raises(SystemExit, match="is not a far_wings_ledger/2 file"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS


@pytest.mark.parametrize("args", [[], ["--retry"]])
def test_a_run_while_the_session_is_open_writes_nothing_and_keeps_the_written_orders(env, capsys, args):
    written = entry_night(env)                                     # for Nov 24
    write_mkt(env, "fills.json", [])
    write_mkt(env, "positions.json", {"cash": 100_000, "positions": []})
    before = files_now(env)
    # Tuesday noon, Sanji not yet through Tuesday: D is Monday, whose orders are for Tuesday, which is open. Before
    # 2026-11-17 this run took Wednesday from the clock and only merge_orders' note kept Tuesday's orders; then it
    # went red as late although Monday's orders were written. Now it finds them in the ledger: nothing to do, green
    fake = env["use"](FakeLSEG(dt.date(2026, 11, 23), chain=False), now=dt.datetime(2026, 11, 24, 12, 0))
    capsys.readouterr()
    assert O.main(args) == 0
    who = "Retry" if args else "Late run (2026-11-24 12:00 New York time)"
    assert (f"{who}: the orders for 2026-11-24 from 2026-11-23's close were already written (nvda-fw-put-2026-11-23, "
            "nvda-fw-call-2026-11-23); nothing to do.") in capsys.readouterr().out
    assert files_now(env) == before and orders_file(env) == written and fake.requests == []


# --------------------------------------------------------------------------- the retry, the chain, LSEG
def test_the_retry_does_nothing_once_tonights_orders_are_written(env, capsys):
    env["use"](FakeLSEG(ENTRY))
    assert O.main([]) == 0
    before = (env["root"] / "mkt" / "open_orders.json").read_bytes()
    fake = env["use"](FakeLSEG(ENTRY), now=dt.datetime(2026, 11, 24, 1, 0))       # 01:00: the same D and for
    env["sessions"].clear()
    assert O.main(["--retry"]) == 0
    assert "already written (nvda-fw-put-2026-11-23, nvda-fw-call-2026-11-23); nothing to do" in \
        capsys.readouterr().out
    assert (env["root"] / "mkt" / "open_orders.json").read_bytes() == before
    assert fake.requests == [] and env["sessions"] == []


def test_the_retry_chooses_without_a_hole_the_first_run_stopped_on(env, capsys):
    fake = FakeLSEG(ENTRY)
    ric = build_option_ric("NVDA", EXP, 150.0, "P", expired=False)
    fake.options[ric] = fake.options[ric].iloc[:1]                 # no D quote for the 150 put
    env["use"](fake)
    with pytest.raises(SystemExit, match="1 listed 2026-12-18 option\\(s\\) have earlier rows but none for "
                                         "2026-11-23"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS and not env["output"].exists()
    env["use"](fake, now=dt.datetime(2026, 11, 24, 1, 0))
    assert O.main(["--retry"]) == 0
    assert f"Note: chose without 1 listed contract(s) with no 2026-11-23 quote ({ric})" in capsys.readouterr().out
    assert [o["id"] for o in orders_file(env)] == ["nvda-fw-put-2026-11-23", "nvda-fw-call-2026-11-23"]


def no_bid_on_d(fake, keep=lambda row: True):
    """D's row of each chain contract `keep` picks, left with no bid (LSEG gives it as missing): not two-sided."""
    picked = [r for r in S.chain_rows(float(fake.closes.iloc[-1])) if keep(r)]
    for r in picked:
        f = fake.options[r["ric"]].copy()
        f.loc[f.index[-1], "BID"] = float("nan")
        fake.options[r["ric"]] = f
    return picked


@pytest.mark.parametrize("rights, named", [(("P", "C"), "put and no call"), (("C",), "call"), (("P",), "put")])
def test_a_right_with_d_rows_but_no_two_sided_quote_stops_and_the_retry_decides(env, capsys, rights, named):
    fake = FakeLSEG(ENTRY)
    assert no_bid_on_d(fake, lambda r: r["cp"] in rights)
    env["use"](fake)
    with pytest.raises(SystemExit, match=f"^No 2026-12-18 {named} has a two-sided 2026-11-23 quote on LSEG .* "
                                         f"LSEG's 2026-11-23 data is not final: no order written"):
        O.main([])
    # red, before anything is written: the night is not in the ledger, so the 01:00 retry decides
    assert orders_file(env) == OLD_ORDERS and ledger_file(env) is None and not env["output"].exists()
    first = capsys.readouterr().out
    assert ("two-sided: 0 puts" in first) == ("P" in rights) and (", 0 calls)" in first) == ("C" in rights)
    env["use"](fake, now=dt.datetime(2026, 11, 24, 1, 0))
    assert O.main(["--retry"]) == 0
    out = capsys.readouterr().out
    assert "Retry: nothing was written for 2026-11-24 from 2026-11-23's close yet, so this run decides." in out
    assert f"Note: the retry still finds that no 2026-12-18 {named} has a two-sided 2026-11-23 quote" in out
    got = [o["id"] for o in orders_file(env)]
    assert got == [i for cp, i in (("P", "nvda-fw-put-2026-11-23"), ("C", "nvda-fw-call-2026-11-23"))
                   if cp not in rights]
    assert ledger_file(env)["names"]["NVDA"]["nights"]["2026-11-23"] == {"for": "2026-11-24", "ids": got}


def test_a_few_one_sided_strikes_are_not_a_partial_chain(env, capsys):
    # only a right with no two-sided quote at all stops: a few strikes with no bid are a healthy chain's holes
    fake = FakeLSEG(ENTRY)
    picked = no_bid_on_d(fake, lambda r: (r["cp"], r["strike"]) in {("P", 130.0), ("P", 135.0), ("C", 240.0)})
    assert len(picked) == 3
    env["use"](fake)
    assert O.main([]) == 0
    assert [o["id"] for o in orders_file(env)] == ["nvda-fw-put-2026-11-23", "nvda-fw-call-2026-11-23"]
    out = capsys.readouterr().out
    assert "Note:" not in out and "two-sided: " in out


def test_a_chain_with_no_row_for_d_stops(env):
    env["use"](FakeLSEG(ENTRY, chain_day="2026-11-20"))
    with pytest.raises(SystemExit, match="data not final"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS


def test_unlisted_strikes_are_skipped_but_any_other_error_stops(env):
    fake = FakeLSEG(ENTRY)
    bad = build_option_ric("NVDA", EXP, 165.0, "P", expired=False)
    fake.errors[bad] = Exception("Error code 404 | Not Found")
    env["use"](fake)
    with pytest.raises(SystemExit, match="LSEG error on NVDA.*404 \\| Not Found"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS and env["sessions"][0].closed
    assert [r[1] for r in fake.option_requests()][-1] == bad


def test_stale_stock_data_stops(env):
    env["use"](FakeLSEG(ENTRY, missing_close=True))
    with pytest.raises(SystemExit, match="LSEG's last close for NVDA.O is 2026-11-20, not 2026-11-23: data not final"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS and env["sessions"][0].closed


def test_a_held_leg_lseg_does_not_know_stops(env):
    entry_night(env)
    env["use"](FakeLSEG(dt.date(2026, 11, 24), chain=False), now=dt.datetime(2026, 11, 24, 22, 0))
    with pytest.raises(SystemExit, match="LSEG does not know NVDA[LX]1826\\d{5}.U, a contract the book holds"):
        O.main([])


def test_an_entry_night_failure_says_the_hedge_was_not_written_either(env):
    hedge = {"id": "nvda-fw-hedge-2026-11-19", "for": "2026-11-20", "order_type": "mkt", "action": "SHT", "qty": 350,
             "underlying": S.NVDA, "contract": "underlying", "ref": "open"}
    S.write_json(env["root"] / "data" / "far_wings_ledger.json",
                 FW.record_night(FW.read_ledger(None), "NVDA", [hedge], dt.date(2026, 11, 19), dt.date(2026, 11, 20)))
    write_mkt(env, "fills.json", [{"id": hedge["id"], "status": "filled"}, {"id": OLD_ORDERS[0]["id"],
                                                                             "status": "filled"}])
    write_mkt(env, "positions.json", {"cash": 0, "positions": [
        {"underlying": S.NVDA, "contract": "underlying", "qty": -350}]})
    env["use"](FakeLSEG(ENTRY, chain_day="2026-11-20"))
    with pytest.raises(SystemExit, match="data not final.*Tonight's hedge was not written either: the book's -350"):
        O.main([])


# --------------------------------------------------------------------------- writing the two files
def files_now(env):
    root = env["root"]
    return {p.relative_to(root).as_posix(): p.read_bytes() for d in ("mkt", "data") if (root / d).exists()
            for p in sorted((root / d).iterdir())}


def the_next_night(env):
    """After the Nov 23 entry was written and filled: the Nov 24 run, which writes [] and records its night."""
    written = entry_night(env)
    day = dt.date(2026, 11, 24)
    env["use"](FakeLSEG(day, chain=False, legs=leg_frames(day, [("P", 145.0, 0.51), ("C", 220.0, 0.42)])),
               now=dt.datetime(2026, 11, 24, 22, 0))
    return written


def test_the_ledger_is_renamed_into_place_before_the_orders_file(env, monkeypatch):
    the_next_night(env)
    before = files_now(env)
    moves, replace = [], O.os.replace

    def recorded(src, dst):
        moves.append((Path(src).parent.name, Path(src).name, Path(dst).name))
        return replace(src, dst)

    monkeypatch.setattr(O.os, "replace", recorded)
    assert O.main([]) == 0
    assert [(d, n) for d, _, n in moves] == [("data", "far_wings_ledger.json"), ("mkt", "open_orders.json")]
    assert all(s.startswith("." + n + ".") and s.endswith(".tmp") for _, s, n in moves)   # each beside its file
    now = files_now(env)
    assert set(now) == set(before)                                         # no temporary file left behind
    assert json.loads(now["mkt/open_orders.json"]) == []
    assert json.loads(now["data/far_wings_ledger.json"])["names"]["NVDA"]["nights"]["2026-11-24"] == {
        "for": "2026-11-25", "ids": []}
    assert all((env["root"] / f).stat().st_mode & 0o777 == 0o644 for f in now)


def _fail_record_night(monkeypatch):
    def boom(*a, **k):
        raise OSError("No space left on device")
    monkeypatch.setattr(O.FW, "record_night", boom)


def _fail_staging_the_orders_file(monkeypatch):
    staged = O._staged

    def stage(path, text):
        if path.name == "open_orders.json":
            raise OSError("No space left on device")
        return staged(path, text)
    monkeypatch.setattr(O, "_staged", stage)


def _fail_the_first_rename(monkeypatch):
    def replace(src, dst):
        raise OSError("Read-only file system")
    monkeypatch.setattr(O.os, "replace", replace)


@pytest.mark.parametrize("fail", [_fail_record_night, _fail_staging_the_orders_file, _fail_the_first_rename])
def test_a_failure_while_writing_leaves_both_files_as_they_were(env, monkeypatch, fail):
    # before, mkt/open_orders.json was written first: a failing ledger write left orders the ledger did not know,
    # and every later night stopped with a LedgerError once Sanji recorded their fills
    the_next_night(env)
    before, output = files_now(env), env["output"].read_text()
    fail(monkeypatch)
    with pytest.raises(OSError):
        O.main([])
    assert files_now(env) == before                      # both files byte for byte, and no temporary file
    assert env["output"].read_text() == output           # no for_day for the commit step


def test_a_failure_on_the_first_night_leaves_no_ledger_and_the_old_file(env, monkeypatch):
    env["use"](FakeLSEG(ENTRY))
    _fail_record_night(monkeypatch)
    with pytest.raises(OSError):
        O.main([])
    assert orders_file(env) == OLD_ORDERS and ledger_file(env) is None
    assert files_now(env) == {"mkt/open_orders.json": (env["root"] / "mkt" / "open_orders.json").read_bytes()}


# --------------------------------------------------------------------------- the config and the lib
@pytest.mark.parametrize("args", [[], ["--retry"]])
def test_a_retired_far_wings_writes_nothing_and_ends_green(env, capsys, args):
    # retired on purpose: exit 0, so the two scheduled runs a night do not go red (the commit step finds nothing)
    set_cfg(env, owns_open_orders=False)
    fake = env["use"](FakeLSEG(ENTRY))
    assert O.main(args) == 0
    out = capsys.readouterr().out
    assert "says owns_open_orders: false" in out and "nothing written" in out
    assert "Delete the schedule in .github/workflows/far-wings-orders.yml" in out
    assert orders_file(env) == OLD_ORDERS and env["sessions"] == [] and fake.requests == []
    assert ledger_file(env) is None and not env["output"].exists()
    assert O.main(["--dry-run"]) == 0                              # a dry run still shows what it would do
    assert orders_file(env) == OLD_ORDERS and "Dry run, nothing written" in capsys.readouterr().out


def test_a_retired_far_wings_still_stops_on_a_malformed_config(env):
    set_cfg(env, owns_open_orders="false")
    env["use"](FakeLSEG(ENTRY))
    with pytest.raises(SystemExit, match="owns_open_orders must be true or false"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS


def test_the_config_says_how_to_retire_and_what_the_schedule_does_then():
    text = (S.REPO / "config" / "far_wings_live.yaml").read_text().split("registered:")[0]
    assert "ends green (exit 0)" in text and "delete the schedule in far-wings-orders.yml" in text


def test_a_config_that_disagrees_stops_before_anything(env):
    cfg = json.loads(env["config"].read_text())
    cfg["registered"]["band_share_of_book"] = 0.10
    S.write_json(env["config"], cfg)
    env["use"](FakeLSEG(ENTRY))
    with pytest.raises(SystemExit, match="disagrees with lib/far_wings_live.py: band_share_of_book"):
        O.main([])
    assert env["sessions"] == [] and orders_file(env) == OLD_ORDERS


@pytest.mark.parametrize("module, old, new", [
    ("wheel", "    F, D = spot * np.exp((r - q) * T), np.exp(-r * T)\n    iv = implied_vol(",
     "    F, D = spot * np.exp((r - q) * T), np.exp(-r * T)\n    _ = read_splits\n    iv = implied_vol("),
    ("vol_arb_live", "FALLBACK_SESSIONS = 30", "FALLBACK_SESSIONS = 5"),
    ("vol_arb", "c = c[c[\"mid\"].notna() & (c[\"mid\"] > 0)]", "c = c[c[\"mid\"].notna() & (c[\"mid\"] >= 0)]"),
])
def test_a_changed_lib_function_stops(env, tmp_path, monkeypatch, module, old, new):
    lib = tmp_path / "lib"
    lib.mkdir()
    for f in O.LIB.glob("*.py"):
        (lib / f.name).write_text(f.read_text())
    text = (lib / f"{module}.py").read_text()
    assert text.count(old) == 1
    (lib / f"{module}.py").write_text(text.replace(old, new))
    monkeypatch.setattr(O, "LIB", lib)
    env["use"](FakeLSEG(ENTRY))
    with pytest.raises(SystemExit, match="lib functions lib/far_wings_live.py uses are not the recorded ones"):
        O.main([])
    assert env["sessions"] == [] and orders_file(env) == OLD_ORDERS


def test_print_lib_functions_prints_the_recorded_lines(capsys):
    assert O.main(["--print-lib-functions"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "lib_functions:"
    assert dict(line.strip().split(": ") for line in out[1:]) == S.config()["lib_functions"]
