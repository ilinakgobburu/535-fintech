"""
vol_arb_orders.py end to end with LSEG and Cboe replaced by fakes (no network): what it asks for, what it writes,
and that every failure stops it before mkt/open_orders.json is touched. The fake LSEG serves the synthetic NVDA
closes, the T-bill, dividends, report dates and a Black-76 chain (va_live_synth); a RIC it does not hold answers
as LSEG does for an unlisted contract ("The universe is not found").
"""

from __future__ import annotations

import datetime as dt
import io
import json
import sys
import urllib.request

import numpy as np
import pandas as pd
import pytest

import va_live_synth as S
from trading_app.lib import vol_arb_live as VL
from trading_app.lib.ric import build_option_ric
from trading_app.lib.vol import bs_greeks, bs_price

_path = list(sys.path)
sys.path.insert(0, str(S.REPO))
import vol_arb_orders as O  # noqa: E402

sys.path[:] = _path

ENTRY, FOR = S.ENTRY, dt.date(2026, 11, 24)
OLD_ORDERS = [{"id": "nvda-csp-2026-11-16", "for": "2026-11-16", "order_type": "lmt", "action": "SHT", "qty": 6,
               "underlying": S.NVDA, "contract": "put", "expiry": "2026-12-18", "strike": "155", "ref": "open",
               "lmt_prc": "7.78"}]


class FakeLSEG:
    """lseg.data's get_history and get_data, from made-up frames. Records every request."""

    def __init__(self, day, b=-0.06, chain=True, errors=None, last_close=None, expected=("2027-02-25",),
                 actual=("2026-08-26", "2026-11-19"), legs=(), chain_day=None, vix=None):
        self.day = pd.Timestamp(day)
        self.closes = S.closes(end=day)
        if last_close is not None:                     # LSEG has not posted D's bar yet
            self.closes = self.closes[self.closes.index < self.day]
        self.opens = self.closes.shift(1).fillna(self.closes.iloc[0]) * 1.001
        self.requests, self.errors, self.expected, self.actual = [], dict(errors or {}), list(expected), list(actual)
        self.options = {}
        spot = float(S.closes(end=day).iloc[-1])
        if chain:
            q = S.trailing_yield(self.dividends_series(), spot, day)
            cday = pd.Timestamp(chain_day or day)
            for r in S.chain_rows(spot, day=day, b=b, q=q):
                prev = cday - pd.Timedelta(days=3)
                self.options[r["ric"]] = pd.DataFrame({"BID": [r["bid"] * 3, r["bid"]], "ASK": [r["ask"] * 3, r["ask"]]},
                                                      index=pd.DatetimeIndex([prev, cday]))
        for ric, frame in legs:
            self.options[ric] = frame
        self.vix = vix                                  # {date: close} LSEG serves for .VIX, or None (unknown RIC)

    def dividends_series(self):
        return pd.Series([0.01, 0.01], index=pd.DatetimeIndex(["2026-06-11", "2026-09-04"]))

    @staticmethod
    def _window(df, start, end):
        return df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]

    def get_history(self, universe, fields, interval="daily", start=None, end=None, adjustments=None, count=None):
        self.requests.append(("history", universe, tuple(fields), adjustments))
        if universe in self.errors:
            raise self.errors[universe]
        if universe == "NVDA.O":
            return self._window(pd.DataFrame({"TRDPRC_1": self.closes, "OPEN_PRC": self.opens}), start, end)
        if universe == "US3MT=RR":
            return self._window(pd.DataFrame({"MID_YLD_1": S.RATE_PCT}, index=self.closes.index), start, end)
        if universe in self.options:
            return self._window(self.options[universe], start, end)
        if universe == ".VIX" and self.vix is not None:
            v = pd.Series(self.vix, dtype=float)
            return self._window(pd.DataFrame({"TRDPRC_1": v.to_numpy()}, index=pd.DatetimeIndex(v.index)), start, end)
        raise Exception(f"Error code -1 | The universe is not found: {universe}")

    def get_data(self, universe, fields, parameters=None):
        self.requests.append(("data", universe, tuple(fields), parameters))
        if fields[0] == "TR.DivExDate":
            d = self.dividends_series()
            return pd.DataFrame({"Instrument": universe, "Dividend Ex Date": [str(x.date()) for x in d.index],
                                 "Gross Dividend Amount": d.to_numpy(), "Dividend Pay Date": None})
        if fields[0] == "TR.ExpectedReportDate":
            return pd.DataFrame({"Instrument": [universe] * max(1, len(self.expected)),
                                 "Expected Report Date": self.expected or [None]})
        if fields[0] == "TR.EPSActReportDate":
            return pd.DataFrame({"Instrument": universe, "Report Date": self.actual})
        raise AssertionError(fields)

    def option_requests(self):
        return [r for r in self.requests if r[0] == "history" and r[1] not in ("NVDA.O", "US3MT=RR", ".VIX")]

    def lseg_requests(self):
        return [r for r in self.requests if r[0] != "sleep"]


class Session:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


VIX_CSV = "DATE,OPEN,HIGH,LOW,CLOSE\n11/20/2026,19,20,18,18.4\n11/23/2026,18,19,17,18.0\n11/24/2026,18,19,17,17.0\n"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A scratch repo root: mkt/ with the wheel's last order, the synthetic panel, fakes for LSEG and Cboe."""
    state = {"sessions": [], "sleeps": [], "vix_reads": 0, "vix": VIX_CSV}
    (tmp_path / "mkt").mkdir()
    S.write_json(tmp_path / "mkt" / "open_orders.json", OLD_ORDERS)
    panel = S.write_json(tmp_path / "data" / "vol_arb_dev_panel.json", S.panel_obj())
    monkeypatch.setattr(O, "ROOT", tmp_path)
    monkeypatch.setattr(O, "MKT", tmp_path / "mkt")
    # the repo's config with the synthetic panel pinned, as export_vol_arb_panel.py's lines would pin the real one
    state["config"] = S.write_json(tmp_path / "config.yaml", S.pin_panel(S.config(), panel))    # JSON is YAML
    monkeypatch.setattr(O, "CONFIG", state["config"])

    def open_lseg():
        s = Session()
        state["sessions"].append(s)
        return s

    def urlopen(req, timeout=None):
        state["vix_reads"] += 1
        assert req.full_url == "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
        return io.BytesIO(state["vix"].encode())

    monkeypatch.setattr(O, "open_lseg", open_lseg)
    def sleep(s):
        state["sleeps"].append(s)
        if state.get("fake") is not None:
            state["fake"].requests.append(("sleep", s))

    monkeypatch.setattr(O.time, "sleep", sleep)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
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


# --------------------------------------------------------------------------- the entry night
def test_the_entry_night_writes_the_two_legs(env, capsys):
    fake = env["use"](FakeLSEG(ENTRY))
    assert O.main([]) == 0
    got = orders_file(env)
    assert [(o["id"], o["action"], o["contract"], o["for"]) for o in got] == [
        ("nvda-va-put-2026-11-23", "SHT", "put", "2026-11-24"), ("nvda-va-call-2026-11-23", "BUY", "call", "2026-11-24")]
    # the same orders the pure plan gives on the same data (the old wheel order for Nov 16 is dropped)
    c, o, rate, div = S.market_parts(ENTRY)
    want = VL.plan_night(S.config(), ENTRY, FOR, VL.Book(), entry_expiry=S.EXPIRY, mkt=S.entry_market(-0.06),
                         vix_close=18.0, panel=VL.load_panel(S.panel_obj(), S.config()))["orders"]
    assert got == want
    assert env["output"].read_text() == "for_day=2026-11-24\n"
    assert len(env["sessions"]) == 1 and env["sessions"][0].closed
    asked = fake.option_requests()
    assert all(r[2] == ("BID", "ASK") for r in asked) and env["sleeps"] == [O.PAUSE] * len(fake.lseg_requests())
    plan = VL.strike_plan(fake.closes, dt.date(2026, 11, 20), ENTRY, S.config()["chain"]["bands"],
                          S.config()["chain"]["grid"])
    assert [r[1] for r in asked] == [build_option_ric("NVDA", S.EXPIRY, k, cp, expired=False)
                                     for cp in ("P", "C") for k in plan[cp]]
    stock = [r for r in fake.requests if r[1] == "NVDA.O" and r[0] == "history"]
    assert stock[0][2] == ("TRDPRC_1", "OPEN_PRC") and stock[0][3] == ["exchangeCorrection"]
    out = capsys.readouterr().out
    assert "not listed" in out and "put_rich (rule_1)" in out


@pytest.mark.parametrize("night", ["entry", "hedge"])
def test_every_lseg_request_is_followed_by_the_pause(env, night):
    if night == "entry":
        fake = env["use"](FakeLSEG(ENTRY))
    else:
        day = dt.date(2026, 11, 25)
        fake = env["use"](FakeLSEG(day, chain=False, legs=leg_frames(day, [("P", 165.0, 0.5), ("C", 195.0, 0.4)])),
                          now=dt.datetime(2026, 11, 25, 22, 0))
        sanji(env, entry_orders([("P", 165.0, -10), ("C", 195.0, 12)]), [option("P", 165, -10), option("C", 195, 12)])
    assert O.main([]) == 0
    log = [r[0] for r in fake.requests]
    kinds = {r[0] for r in fake.lseg_requests()}
    assert len(log) >= 8 and kinds <= {"history", "data"}
    # request, pause, request, pause, ...: never two requests without the pause between them
    assert all((a == "sleep") != (b == "sleep") for a, b in zip(log, log[1:]))
    assert log[0] != "sleep" and log[-1] == "sleep" and log.count("sleep") == len(fake.lseg_requests())
    assert all(r[1] == O.PAUSE for r in fake.requests if r[0] == "sleep")


def test_the_dry_run_prints_and_writes_nothing(env, capsys):
    env["use"](FakeLSEG(ENTRY))
    before = (env["root"] / "mkt" / "open_orders.json").read_bytes()
    assert O.main(["--dry-run", "--as-of", "2026-11-23"]) == 0
    assert (env["root"] / "mkt" / "open_orders.json").read_bytes() == before
    assert not env["output"].exists()
    out = capsys.readouterr().out
    assert "Dry run, nothing written" in out and '"id": "nvda-va-put-2026-11-23"' in out


def test_force_entry_checks_the_next_entrys_chain_on_the_latest_close(env, capsys):
    day = dt.date(2026, 11, 16)
    fake = env["use"](FakeLSEG(day), now=dt.datetime(2026, 11, 16, 22, 0))
    env["vix"] = VIX_CSV + "11/16/2026,17,18,16,16.5\n"
    assert O.main(["--dry-run", "--as-of", "2026-11-16", "--force-entry"]) == 0
    out = capsys.readouterr().out
    assert "expiry 2026-12-18 (forced)" in out and '"id": "nvda-va-put-2026-11-16"' in out
    assert all("L1826" in r[1] or "X1826" in r[1] for r in fake.option_requests())     # the Dec 18 chain


@pytest.mark.parametrize("args", [["--force-entry"], ["--as-of", "2026-11-23"], ["--dry-run", "--as-of", "2026-11-22"]])
def test_one_off_flags_need_a_dry_run_on_a_session(env, args):
    env["use"](FakeLSEG(ENTRY))
    with pytest.raises(SystemExit):
        O.main(args)
    assert orders_file(env) == OLD_ORDERS


def test_a_non_entry_night_with_an_empty_book_writes_nothing_new_and_skips_lseg(env):
    env["use"](FakeLSEG(FOR), now=dt.datetime(2026, 11, 24, 22, 0))
    assert O.main([]) == 0
    assert orders_file(env) == [] and env["sessions"] == [] and env["vix_reads"] == 0
    assert env["output"].read_text() == "for_day=2026-11-25\n"


def test_unlisted_strikes_are_skipped_but_any_other_error_stops(env):
    fake = FakeLSEG(ENTRY)
    bad = build_option_ric("NVDA", S.EXPIRY, 165.0, "P", expired=False)
    fake.errors[bad] = Exception("Error code 503 | Service unavailable")
    env["use"](fake)
    with pytest.raises(SystemExit, match="LSEG error"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS and not env["output"].exists()
    assert env["sessions"][0].closed
    asked = [r[1] for r in fake.option_requests()]
    assert asked[-1] == bad                                      # it stopped at the failure, nothing after


def test_a_chain_with_no_row_for_d_stops(env):
    env["use"](FakeLSEG(ENTRY, chain_day="2026-11-20"))         # LSEG's option bars end the session before
    with pytest.raises(SystemExit, match="data not final"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS


def test_a_chain_missing_d_for_two_listed_contracts_stops(env):
    # LSEG part-way through publishing D: the 165 put and 195 call have their earlier row but not D's (this traded
    # 167.5 / 192.5 at 13 contracts before)
    fake = FakeLSEG(ENTRY)
    for cp, k in (("P", 165.0), ("C", 195.0)):
        ric = build_option_ric("NVDA", S.EXPIRY, k, cp, expired=False)
        fake.options[ric] = fake.options[ric].iloc[:1]
    env["use"](fake)
    with pytest.raises(SystemExit, match="2 listed 2026-12-18 option\\(s\\) have earlier rows but none for "
                                         "2026-11-23 \\(NVDA.*\\): LSEG's 2026-11-23 data is not final"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS and not env["output"].exists() and env["sessions"][0].closed


def test_a_chain_with_too_few_d_rows_stops(env):
    # every other listed contract with no row at all in the window: under 90% of the chain has D
    fake = FakeLSEG(ENTRY)
    for i, ric in enumerate(sorted(fake.options)):
        if i % 2:
            fake.options[ric] = fake.options[ric].iloc[:0]
    env["use"](fake)
    with pytest.raises(SystemExit, match="Only \\d+ of the \\d+ listed 2026-12-18 options have a 2026-11-23 row "
                                         "\\(under 90%\\): the chain looks partial"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS


def test_a_few_contracts_never_quoted_are_no_problem(env):
    # two far strikes LSEG lists but never quoted in the window: 90% or more of the chain still has D
    fake = FakeLSEG(ENTRY)
    for cp, k in (("P", 122.5), ("C", 252.5)):
        ric = build_option_ric("NVDA", S.EXPIRY, k, cp, expired=False)
        if ric in fake.options:
            fake.options[ric] = fake.options[ric].iloc[:0]
    env["use"](fake)
    assert O.main([]) == 0
    assert [o["id"] for o in orders_file(env)] == ["nvda-va-put-2026-11-23", "nvda-va-call-2026-11-23"]


def test_a_transient_not_found_is_an_error_not_an_unlisted_strike(env):
    fake = FakeLSEG(ENTRY)
    for cp, k in (("P", 165.0), ("C", 195.0)):
        fake.errors[build_option_ric("NVDA", S.EXPIRY, k, cp, expired=False)] = Exception("Error code 404 | Not Found")
    env["use"](fake)
    with pytest.raises(SystemExit, match="LSEG error on NVDA.*404 \\| Not Found"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS


def test_chain_problem():
    E, D = S.EXPIRY, ENTRY
    assert VL.chain_problem(E, D, 50, 50, [], 0.9) is None
    assert VL.chain_problem(E, D, 50, 45, [], 0.9) is None                       # 90% exactly
    assert VL.chain_problem(E, D, 50, 44, [], 0.9).startswith("Only 44 of the 50 listed 2026-12-18 options")
    assert VL.chain_problem(E, D, 50, 0, [], 0.9).startswith("No 2026-12-18 option has a 2026-11-23 row")
    assert "1 listed 2026-12-18 option(s) have earlier rows" in VL.chain_problem(E, D, 50, 49, ["X"], 0.9)
    many = VL.chain_problem(E, D, 50, 44, list("ABCDEF"), 0.9)
    assert "(A, B, C, D, ...)" in many


@pytest.mark.parametrize("share", [0, 1.2, "0.9", None])
def test_the_config_needs_a_share(share):
    cfg = S.config()
    cfg["chain"]["min_dated_share"] = share
    with pytest.raises(ValueError, match="min_dated_share"):
        VL.check_config(cfg)


def test_the_chain_uses_only_ds_quotes(env):
    # the fake's earlier row is three times D's quotes; the orders are those of D's chain
    env["use"](FakeLSEG(ENTRY))
    O.main([])
    put = orders_file(env)[0]
    k, mid = float(put["strike"]), float(put["lmt_prc"])
    row = next(r for r in S.chain_rows(float(S.closes().iloc[-1]), b=-0.06,
                                       q=S.trailing_yield(FakeLSEG(ENTRY).dividends_series(), float(S.closes().iloc[-1]),
                                                          ENTRY)) if r["cp"] == "P" and r["strike"] == k)
    assert mid == pytest.approx((row["bid"] + row["ask"]) / 2, abs=0.006)


def test_stale_stock_data_stops(env):
    env["use"](FakeLSEG(ENTRY, last_close=True))
    with pytest.raises(SystemExit, match="data not final"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS and env["sessions"][0].closed


def test_no_vix_close_for_d_anywhere_stops(env):
    env["vix"] = "DATE,OPEN,HIGH,LOW,CLOSE\n11/20/2026,19,20,18,18.4\n"
    env["use"](FakeLSEG(ENTRY))                                  # LSEG does not know .VIX either
    with pytest.raises(SystemExit, match="Cboe's VIX file ends 2026-11-20, without 2026-11-23, and LSEG's .VIX has "
                                         "no 2026-11-23 close either"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS and env["sessions"][0].closed
    env["use"](FakeLSEG(ENTRY, vix={"2026-11-20": 18.4}))           # LSEG knows .VIX, but not D's close
    with pytest.raises(SystemExit, match="has no 2026-11-23 close either"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS


@pytest.mark.parametrize("cboe", ["without D", "a block page", "unreachable"])
def test_lseg_vix_stands_in_when_cboe_lacks_d(env, capsys, monkeypatch, cboe):
    if cboe == "without D":
        env["vix"] = "DATE,OPEN,HIGH,LOW,CLOSE\n11/20/2026,19,20,18,18.4\n"
    elif cboe == "a block page":
        env["vix"] = "<!DOCTYPE html><html><body>Access denied</body></html>\n"
    else:
        def down(req, timeout=None):
            raise OSError("timed out")
        monkeypatch.setattr(urllib.request, "urlopen", down)
    fake = env["use"](FakeLSEG(ENTRY, vix={"2026-11-20": 18.4, "2026-11-23": 18.0}))
    assert O.main([]) == 0
    out = capsys.readouterr().out
    assert "LSEG's .VIX close will be used" in out and "VIX closed at 18.00 on 2026-11-23 (LSEG .VIX" in out
    assert {"without D": "ends 2026-11-20", "a block page": "it starts '<!DOCTYPE html>",
            "unreachable": "could not be read (timed out)"}[cboe] in out
    # the same orders as with Cboe's 18.0
    assert orders_file(env) == VL.plan_night(S.config(), ENTRY, FOR, VL.Book(), entry_expiry=S.EXPIRY,
                                             mkt=S.entry_market(-0.06), vix_close=18.0,
                                             panel=VL.load_panel(S.panel_obj(), S.config()))["orders"]
    vix_req = [r for r in fake.requests if r[1] == ".VIX"]
    assert vix_req == [("history", ".VIX", ("TRDPRC_1",), None)]


def test_cboes_close_is_used_when_it_has_d(env, capsys):
    fake = env["use"](FakeLSEG(ENTRY, vix={"2026-11-23": 99.0}))
    assert O.main([]) == 0
    assert "VIX closed at 18.00 on 2026-11-23 (Cboe)" in capsys.readouterr().out
    assert not any(r[1] == ".VIX" for r in fake.requests)


def test_no_fallback_configured_stops_before_lseg(env):
    cfg = json.loads(env["config"].read_text())
    cfg["data"]["vix_fallback_ric"] = None
    S.write_json(env["config"], cfg)
    env["vix"] = "DATE,OPEN,HIGH,LOW,CLOSE\n11/20/2026,19,20,18,18.4\n"
    env["use"](FakeLSEG(ENTRY, vix={"2026-11-23": 18.0}))
    with pytest.raises(SystemExit, match="without 2026-11-23: the VIX feature can't be set"):
        O.main([])
    assert env["sessions"] == [] and orders_file(env) == OLD_ORDERS


def test_an_entry_night_failure_says_the_hedge_was_not_written_either(env):
    # the expiry's leftover -350 shares were not closed (say Sanji had not filled the close); the panel is missing
    write_mkt(env, "fills.json", [{"id": "nvda-va-hedge-2026-11-19", "status": "filled"},
                                  {"id": "nvda-csp-2026-11-16", "status": "filled"}])
    write_mkt(env, "positions.json", {"cash": 0, "positions": [
        {"underlying": S.NVDA, "contract": "underlying", "qty": -350}]})
    S.write_json(env["root"] / "data" / "vol_arb_ledger.json", {"format": "vol_arb_ledger/1", "orders": {
        "nvda-va-hedge-2026-11-19": {"id": "nvda-va-hedge-2026-11-19", "for": "2026-11-20", "order_type": "mkt",
                                     "action": "SHT", "qty": 350, "underlying": S.NVDA, "contract": "underlying",
                                     "ref": "open"}}, "nights": {}})
    (env["root"] / "data" / "vol_arb_dev_panel.json").unlink()
    env["use"](FakeLSEG(ENTRY))
    with pytest.raises(SystemExit, match="export the development panel.*Tonight's hedge was not written either: the "
                                         "book's -350 NVDA shares stay"):
        O.main([])
    env["vix"] = "<html>blocked</html>"
    S.write_json(env["root"] / "data" / "vol_arb_dev_panel.json", S.panel_obj())
    with pytest.raises(SystemExit, match="the VIX feature can't be set.*Tonight's hedge was not written either"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS


def test_a_failure_with_no_shares_says_nothing_of_a_hedge(env):
    (env["root"] / "data" / "vol_arb_dev_panel.json").unlink()
    env["use"](FakeLSEG(ENTRY))
    with pytest.raises(SystemExit) as exc:
        O.main([])
    assert "hedge" not in str(exc.value.code)


# --------------------------------------------------------------------------- a run after the open
def test_a_run_after_the_next_open_places_no_entry_only_the_hedge(env, capsys):
    # the 22:00 and 01:00 runs failed; the Run button on Tuesday 10:00 must not enter from Monday's quotes for Wednesday
    write_mkt(env, "open_orders.json", [])
    env["use"](FakeLSEG(ENTRY), now=dt.datetime(2026, 11, 24, 10, 0))
    assert O.main([]) == 0
    out = capsys.readouterr().out
    assert "2026-11-23 was an entry night, but 2026-11-24 has opened since" in out and "only the hedge" in out
    assert orders_file(env) == [] and env["sessions"] == [] and env["vix_reads"] == 0
    # with leftover shares the hedge is still written, for Wednesday
    write_mkt(env, "fills.json", [{"id": "nvda-va-hedge-2026-11-19", "status": "filled"}])
    write_mkt(env, "positions.json", {"cash": 0, "positions": [
        {"underlying": S.NVDA, "contract": "underlying", "qty": -350}]})
    S.write_json(env["root"] / "data" / "vol_arb_ledger.json", {"format": "vol_arb_ledger/1", "orders": {
        "nvda-va-hedge-2026-11-19": {"id": "nvda-va-hedge-2026-11-19", "for": "2026-11-20", "order_type": "mkt",
                                     "action": "SHT", "qty": 350, "underlying": S.NVDA, "contract": "underlying",
                                     "ref": "open"}}, "nights": {}})
    assert O.main([]) == 0
    assert [(o["id"], o["action"], o["qty"], o["for"]) for o in orders_file(env)] == [
        ("nvda-va-hedge-2026-11-23", "BTC", 350, "2026-11-25")]


def test_a_dry_run_as_of_the_entry_still_enters(env, capsys):
    env["use"](FakeLSEG(ENTRY), now=dt.datetime(2026, 11, 24, 10, 0))
    assert O.main(["--dry-run", "--as-of", "2026-11-23"]) == 0
    assert '"id": "nvda-va-put-2026-11-23"' in capsys.readouterr().out


# --------------------------------------------------------------------------- the retry
def test_the_retry_does_nothing_once_tonights_orders_are_written(env, capsys):
    env["use"](FakeLSEG(ENTRY))
    assert O.main([]) == 0
    before = (env["root"] / "mkt" / "open_orders.json").read_bytes()
    ledger = (env["root"] / "data" / "vol_arb_ledger.json").read_bytes()
    fake = env["use"](FakeLSEG(ENTRY), now=dt.datetime(2026, 11, 24, 1, 0))       # 01:00: the same D and for
    env["vix_reads"], env["sessions"] = 0, []
    assert O.main(["--retry"]) == 0
    assert "already written (nvda-va-put-2026-11-23, nvda-va-call-2026-11-23); nothing to do" in capsys.readouterr().out
    assert (env["root"] / "mkt" / "open_orders.json").read_bytes() == before
    assert (env["root"] / "data" / "vol_arb_ledger.json").read_bytes() == ledger
    assert fake.requests == [] and env["sessions"] == [] and env["vix_reads"] == 0


def test_the_retry_decides_when_the_first_run_failed(env, capsys):
    env["vix"] = "DATE,OPEN,HIGH,LOW,CLOSE\n11/20/2026,19,20,18,18.4\n"      # 22:00: Cboe not updated yet
    env["use"](FakeLSEG(ENTRY))
    with pytest.raises(SystemExit):
        O.main([])
    env["vix"] = VIX_CSV                                                          # 01:00: it is
    env["use"](FakeLSEG(ENTRY), now=dt.datetime(2026, 11, 24, 1, 0))
    assert O.main(["--retry"]) == 0
    assert "nothing was written for 2026-11-24 from 2026-11-23's close yet, so this run decides" in \
        capsys.readouterr().out
    assert [o["id"] for o in orders_file(env)] == ["nvda-va-put-2026-11-23", "nvda-va-call-2026-11-23"]
    assert env["output"].read_text() == "for_day=2026-11-24\n"


def test_a_night_with_no_order_counts_as_decided(env, capsys):
    env["use"](FakeLSEG(FOR), now=dt.datetime(2026, 11, 24, 22, 0))
    assert O.main([]) == 0 and orders_file(env) == []
    env["use"](FakeLSEG(FOR), now=dt.datetime(2026, 11, 25, 1, 0))
    assert O.main(["--retry"]) == 0
    assert "already written (no order); nothing to do" in capsys.readouterr().out
    # a different night is not decided by it
    env["use"](FakeLSEG(dt.date(2026, 11, 25)), now=dt.datetime(2026, 11, 26, 1, 0))
    assert O.main(["--retry"]) == 0
    assert "so this run decides" in capsys.readouterr().out


def test_decided():
    led = VL.record_night(VL.read_ledger(None), [], dt.date(2026, 11, 24), dt.date(2026, 11, 25))
    assert VL.decided(led, dt.date(2026, 11, 24), dt.date(2026, 11, 25)) == "no order"
    assert VL.decided(led, dt.date(2026, 11, 24), dt.date(2026, 11, 26)) is None      # a run for another session
    assert VL.decided(led, dt.date(2026, 11, 23), dt.date(2026, 11, 24)) is None


def test_a_next_report_date_not_yet_rolled_forward(env):
    # no expected date after D, but the last report (Nov 19) rules out another before Dec 18: it goes ahead
    env["use"](FakeLSEG(ENTRY, expected=("2026-11-19",)))
    assert O.main(["--dry-run"]) == 0
    # no expected date and the last report in August: whether one falls in the cycle is unknown, so it stops
    env["use"](FakeLSEG(ENTRY, expected=(), actual=("2026-08-26",)))
    with pytest.raises(SystemExit, match="earnings feature cannot be set"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS and env["sessions"][-1].closed


def test_a_missing_or_mismatched_panel_stops_before_lseg(env):
    env["use"](FakeLSEG(ENTRY))
    (env["root"] / "data" / "vol_arb_dev_panel.json").unlink()
    with pytest.raises(SystemExit, match="export the development panel"):
        O.main([])
    S.write_json(env["root"] / "data" / "vol_arb_dev_panel.json", S.panel_obj(chosen="forest"))
    with pytest.raises(SystemExit, match="chose 'forest'"):
        O.main([])
    assert env["sessions"] == [] and orders_file(env) == OLD_ORDERS
    other = S.write_json(env["root"] / "elsewhere.json", S.panel_obj())
    assert O.main(["--dry-run", "--panel", str(other)]) == 0


def set_pin(env, **pin):
    cfg = json.loads(env["config"].read_text())
    cfg["panel_pin"] = {**cfg["panel_pin"], **pin}
    S.write_json(env["config"], cfg)


def test_an_unpinned_panel_stops_an_entry_but_a_dry_run_goes_on(env, capsys):
    set_pin(env, sha256=None, dev_output_sha256=None, dev_output_commit=None)
    env["use"](FakeLSEG(ENTRY))
    with pytest.raises(SystemExit, match="pins no panel yet"):
        O.main([])
    assert env["sessions"] == [] and orders_file(env) == OLD_ORDERS
    assert O.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "a real run would stop here" in out and '"id": "nvda-va-put-2026-11-23"' in out


@pytest.mark.parametrize("change", [
    "another dev output",          # a second development run: other labels, other made_from, the same rung
    "a hand-edited row",           # one finite feature changed
    "a row removed",
])
def test_a_panel_that_is_not_the_pinned_one_stops(env, change):
    path = env["root"] / "data" / "vol_arb_dev_panel.json"
    if change == "another dev output":
        obj = S.panel_obj(seed=11)
        obj["made_from"] = {"file": "vol_arb_ml_dev.json", "sha256": "e" * 64, "commit": "deadbeef"}
    else:
        obj = json.loads(path.read_text())
        if change == "a hand-edited row":
            obj["rows"][7]["f4_tilt"] += 0.01
        else:
            del obj["rows"][7]
    VL.load_panel(obj, S.config())                    # valid in every other way
    S.write_json(path, obj)
    env["use"](FakeLSEG(ENTRY))
    with pytest.raises(SystemExit, match="not the pinned one: its sha256 is"):
        O.main([])
    assert env["sessions"] == [] and orders_file(env) == OLD_ORDERS


def test_the_pin_must_agree_with_what_the_panel_says_it_was_made_from(env):
    set_pin(env, dev_output_commit="0123abc")
    env["use"](FakeLSEG(ENTRY))
    with pytest.raises(SystemExit, match="commit None, panel_pin records '0123abc'"):
        O.main([])
    set_pin(env, dev_output_commit=None, dev_output_sha256="1" * 64)
    with pytest.raises(SystemExit, match="sha256 '0000"):
        O.main([])
    set_pin(env, dev_output_sha256="0" * 64, sha256="not a hash")
    with pytest.raises(SystemExit, match="is not a sha256"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS


def test_the_pin_is_not_needed_on_a_night_without_an_entry(env):
    set_pin(env, sha256=None)
    env["use"](FakeLSEG(FOR), now=dt.datetime(2026, 11, 24, 22, 0))
    assert O.main([]) == 0 and orders_file(env) == []


def test_changed_backtest_code_stops(env, tmp_path, monkeypatch):
    cfg = S.config()
    cfg["lib"]["sha256"]["trading_app/trading_app/lib/vol_arb.py"] = "0" * 64
    path = tmp_path / "cfg.yaml"
    path.write_text(json.dumps(cfg, default=str))                # JSON is YAML
    monkeypatch.setattr(O, "CONFIG", path)
    env["use"](FakeLSEG(ENTRY))
    with pytest.raises(SystemExit, match="not the frozen backtest code"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS


@pytest.mark.parametrize("module, old, new", [
    # select gaining vol-arb-ml's cap (default none), the dividend window, an IV solve's bracket, a new helper
    ("wheel", "target: float, floor: float | None = None) -> pd.Series | None:",
     "target: float, floor: float | None = None, cap: float | None = None) -> pd.Series | None:"),
    ("pmcc_daily", "pd.Timedelta(days=365)) & (d.index <= day)].sum())", "pd.Timedelta(days=360)) & (d.index <= day)].sum())"),
    ("vol", "VOL_LO, VOL_HI = ", "VOL_LO, VOL_HI =  "),
    ("wheel", "    F, D = spot * np.exp((r - q) * T), np.exp(-r * T)\n    iv = implied_vol(",
     "    F, D = spot * np.exp((r - q) * T), np.exp(-r * T)\n    _ = read_splits\n    iv = implied_vol("),
])
def test_a_changed_lib_function_stops(env, tmp_path, monkeypatch, module, old, new):
    lib = tmp_path / "lib"
    lib.mkdir()
    for f in ("wheel", "vol", "pmcc_daily"):
        (lib / f"{f}.py").write_text((O.LIB / f"{f}.py").read_text())
    text = (lib / f"{module}.py").read_text()
    assert text.count(old) == 1
    (lib / f"{module}.py").write_text(text.replace(old, new))
    monkeypatch.setattr(O, "LIB", lib)
    env["use"](FakeLSEG(ENTRY))
    with pytest.raises(SystemExit, match="lib functions are not the recorded ones"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS and env["sessions"] == []


def test_the_lseg_frames_are_read_as_the_backtests_cache(env):
    fake = env["use"](FakeLSEG(ENTRY))
    div = O.dividends("NVDA.O", ENTRY, 400)
    assert list(div.index) == list(pd.DatetimeIndex(["2026-06-11", "2026-09-04"])) and list(div) == [0.01, 0.01]
    params = next(r[3] for r in fake.requests if r[2][0] == "TR.DivExDate")
    assert params == {"SDate": "2025-10-19", "EDate": "2027-01-22"}
    rate = O.tbill("US3MT=RR", ENTRY)
    assert rate.index[-1] == pd.Timestamp(ENTRY) and rate.iloc[-1] == S.RATE_PCT
    closes, opens = O.stock_bars("NVDA.O", ENTRY, 200)
    assert closes.index[-1] == pd.Timestamp(ENTRY) and closes.index[0] >= pd.Timestamp("2026-05-07")
    assert np.array_equal(closes.to_numpy(float), fake.closes[closes.index].to_numpy(float))
    assert closes.index[0] == fake.closes.index[fake.closes.index >= pd.Timestamp("2026-05-07")][0]
    expected, actual = O.report_dates("NVDA.O", ENTRY)
    assert expected == [pd.Timestamp("2027-02-25")] and actual == [pd.Timestamp("2026-08-26"), pd.Timestamp("2026-11-19")]


# --------------------------------------------------------------------------- guards
def write_mkt(env, name, obj):
    S.write_json(env["root"] / "mkt" / name, obj)


def test_an_order_with_no_result_stops_once_sanji_keeps_fills(env):
    env["use"](FakeLSEG(ENTRY))
    write_mkt(env, "open_orders.json", [{**OLD_ORDERS[0], "id": "nvda-va-hedge-2026-11-20", "for": "2026-11-23"}])
    write_mkt(env, "fills.json", [])
    with pytest.raises(SystemExit, match="no result yet"):
        O.main([])
    assert env["sessions"] == []


def test_a_rejected_order_stops(env):
    env["use"](FakeLSEG(ENTRY))
    hedge = {"id": "nvda-va-hedge-2026-11-20", "for": "2026-11-23", "order_type": "mkt", "action": "BTC", "qty": 300,
             "underlying": S.NVDA, "contract": "underlying", "ref": "open"}
    write_mkt(env, "open_orders.json", [hedge])
    write_mkt(env, "fills.json", [{"id": hedge["id"], "status": "rejected"}])
    with pytest.raises(SystemExit, match="Sanji rejected nvda-va-hedge-2026-11-20"):
        O.main([])
    assert orders_file(env) == [hedge] and env["sessions"] == []


def test_an_option_listed_past_its_expiry_stops(env):
    env["use"](FakeLSEG(ENTRY))
    write_mkt(env, "positions.json", {"cash": 0, "positions": [
        {"underlying": S.NVDA, "contract": "put", "expiry": "2026-11-20", "strike": 165, "qty": -10}]})
    with pytest.raises(SystemExit, match="expired by 2026-11-23"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS


def test_a_book_in_another_shape_stops(env):
    env["use"](FakeLSEG(ENTRY))
    write_mkt(env, "positions.json", {"positions": []})
    with pytest.raises(SystemExit, match="Can't read the book"):
        O.main([])


def test_open_orders_must_be_a_list(env):
    env["use"](FakeLSEG(ENTRY))
    write_mkt(env, "open_orders.json", {"id": "x"})
    with pytest.raises(SystemExit):
        O.main([])


# --------------------------------------------------------------------------- the hedge night
def priced(closes, cp, expiry, strike, sigma, r=S.RATE_PCT / 100, q_of=None):
    rows = {}
    for d, s in closes.tail(10).items():
        T = (pd.Timestamp(expiry) - d).days / 365.0
        q = q_of(s, d)
        mid = float(bs_price(s * np.exp((r - q) * T), strike, T, sigma, np.exp(-r * T), cp))
        rows[d] = {"BID": mid * 0.99, "ASK": mid * 1.01}
    return pd.DataFrame.from_dict(rows, orient="index")


def entry_orders(legs, day=ENTRY, for_day=FOR, expiry=S.EXPIRY):
    """This strategy's entry orders for [(cp, strike, signed contracts)], as plan_night writes them."""
    out = []
    for cp, k, n in legs:
        out += VL.option_orders(root="NVDA", conid=S.NVDA, day=day, for_day=for_day, cp=cp, expiry=expiry, strike=k,
                                limit=3.0, held=0, target=n)
    return out


def hedge_orders(day, for_day, held, target):
    return VL.stock_orders(root="NVDA", conid=S.NVDA, day=day, for_day=for_day, held=held, target=target)


def option(cp, k, n, expiry="2026-12-18"):
    return {"underlying": S.NVDA, "contract": "put" if cp == "P" else "call", "expiry": expiry, "strike": k, "qty": n}


def shares(n):
    return {"underlying": S.NVDA, "contract": "underlying", "qty": n}


def sanji(env, filled, positions, other_fills=(), unfilled=(), written=None):
    """
    The repo after Sanji resolved this strategy's orders: the ledger holds every one of them (as the script records
    what it writes), mkt/fills.json gives `filled` as filled and `unfilled` as unfilled (plus `other_fills`), and
    mkt/positions.json is `positions`. mkt/open_orders.json is `written` (by default the orders just resolved).
    """
    path = env["root"] / "data" / "vol_arb_ledger.json"
    ledger = VL.read_ledger(json.loads(path.read_text()) if path.exists() else None)
    for o in [*filled, *unfilled]:
        ledger["orders"][o["id"]] = o
    S.write_json(path, ledger)
    write_mkt(env, "fills.json", [{"id": o["id"], "status": "filled"} for o in filled]
              + [{"id": o["id"], "status": "unfilled"} for o in unfilled] + list(other_fills))
    write_mkt(env, "positions.json", {"cash": 0, "positions": positions})
    write_mkt(env, "open_orders.json", [*filled, *unfilled] if written is None else written)


def leg_frames(day, legs):
    """Each held leg's daily BID/ASK at one vol, as LSEG would serve it: [(ric, frame)] for [(cp, strike, sigma)]."""
    closes = S.closes(end=day)
    divs = FakeLSEG(day).dividends_series()
    return [(build_option_ric("NVDA", S.EXPIRY, k, cp, expired=False),
             priced(closes, cp, S.EXPIRY, k, s, q_of=lambda x, d: S.trailing_yield(divs, x, d))) for cp, k, s in legs]


def by_hand_target(day, legs):
    """-round(net delta) by hand: Black-76 delta to spot at each leg's vol on D's close, for [(cp, k, n, sigma)]."""
    closes = S.closes(end=day)
    divs = FakeLSEG(day).dividends_series()
    spot, r = float(closes.iloc[-1]), S.RATE_PCT / 100
    T = (pd.Timestamp(S.EXPIRY) - pd.Timestamp(day)).days / 365.0
    F = spot * np.exp((r - S.trailing_yield(divs, spot, day)) * T)
    return -int(round(sum(n * 100 * bs_greeks(F, k, T, s, np.exp(-r * T), cp)["delta"] * F / spot
                          for cp, k, n, s in legs)))


LEGS = [("P", 165.0, -10, 0.50), ("C", 195.0, 12, 0.40)]


def test_a_hedge_night_writes_the_stock_order(env):
    day, for_day = dt.date(2026, 11, 25), dt.date(2026, 11, 27)
    fake = env["use"](FakeLSEG(day, chain=False, legs=leg_frames(day, [(cp, k, s) for cp, k, _, s in LEGS])),
                      now=dt.datetime(2026, 11, 25, 22, 0))
    sanji(env, entry_orders([(cp, k, n) for cp, k, n, _ in LEGS]) + hedge_orders("2026-11-24", day, 0, -400),
          [option(cp, k, n) for cp, k, n, _ in LEGS] + [shares(-400)])
    assert O.main([]) == 0
    target = by_hand_target(day, LEGS)
    (o,) = orders_file(env)
    assert (o["id"], o["for"], o["order_type"], o["contract"]) == ("nvda-va-hedge-2026-11-25", "2026-11-27", "mkt",
                                                                   "underlying")
    assert target < -400 and (o["action"], o["qty"]) == ("SHT", -400 - target)
    assert env["vix_reads"] == 0 and env["sessions"][0].closed
    assert not any(r[1] == "NVDA.O" and r[0] == "data" and r[2] == ("TR.ExpectedReportDate",) for r in fake.requests)
    assert len(fake.option_requests()) == 2


def test_a_held_contract_lseg_does_not_know_stops(env):
    day = dt.date(2026, 11, 25)
    env["use"](FakeLSEG(day, chain=False), now=dt.datetime(2026, 11, 25, 22, 0))
    sanji(env, entry_orders([("P", 165.0, -10)]), [option("P", 165, -10)])
    with pytest.raises(SystemExit, match="does not know"):
        O.main([])
    assert orders_file(env) == entry_orders([("P", 165.0, -10)])


def test_after_the_expiry_the_leftover_shares_are_closed_without_lseg(env):
    env["use"](FakeLSEG(dt.date(2026, 12, 18), chain=False), now=dt.datetime(2026, 12, 18, 22, 0))
    # the legs expired today (Sanji removed them); the hedge sold short 350 shares on the 17th
    sanji(env, entry_orders([("P", 165.0, -10), ("C", 195.0, 12)]) + hedge_orders("2026-12-16", "2026-12-17", 0, -350),
          [shares(-350)], written=hedge_orders("2026-12-17", "2026-12-18", -350, -350))
    assert O.main([]) == 0
    assert [(o["action"], o["qty"], o["for"]) for o in orders_file(env)] == [("BTC", 350, "2026-12-21")]
    assert env["sessions"] == []
