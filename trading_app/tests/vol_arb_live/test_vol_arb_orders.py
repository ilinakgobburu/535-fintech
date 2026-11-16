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
                 actual=("2026-08-26", "2026-11-19"), legs=(), chain_day=None):
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
        return [r for r in self.requests if r[0] == "history" and r[1] not in ("NVDA.O", "US3MT=RR")]


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
    S.write_json(tmp_path / "data" / "vol_arb_dev_panel.json", S.panel_obj())
    monkeypatch.setattr(O, "ROOT", tmp_path)
    monkeypatch.setattr(O, "MKT", tmp_path / "mkt")
    monkeypatch.setattr(O, "CONFIG", S.CONFIG)

    def open_lseg():
        s = Session()
        state["sessions"].append(s)
        return s

    def urlopen(req, timeout=None):
        state["vix_reads"] += 1
        assert req.full_url == "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
        return io.BytesIO(state["vix"].encode())

    monkeypatch.setattr(O, "open_lseg", open_lseg)
    monkeypatch.setattr(O.time, "sleep", lambda s: state["sleeps"].append(s))
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    out = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    state["root"], state["output"] = tmp_path, out

    def use(fake, now=None):
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
    assert all(r[2] == ("BID", "ASK") for r in asked) and env["sleeps"] == [O.PAUSE] * len(asked)
    plan = VL.strike_plan(fake.closes, dt.date(2026, 11, 20), ENTRY, S.config()["chain"]["bands"],
                          S.config()["chain"]["grid"])
    assert [r[1] for r in asked] == [build_option_ric("NVDA", S.EXPIRY, k, cp, expired=False)
                                     for cp in ("P", "C") for k in plan[cp]]
    stock = [r for r in fake.requests if r[1] == "NVDA.O" and r[0] == "history"]
    assert stock[0][2] == ("TRDPRC_1", "OPEN_PRC") and stock[0][3] == ["exchangeCorrection"]
    out = capsys.readouterr().out
    assert "not listed" in out and "put_rich (rule_1)" in out


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


def test_no_vix_close_for_d_stops_before_lseg(env):
    env["vix"] = "DATE,OPEN,HIGH,LOW,CLOSE\n11/20/2026,19,20,18,18.4\n"
    env["use"](FakeLSEG(ENTRY))
    with pytest.raises(SystemExit, match="VIX"):
        O.main([])
    assert env["sessions"] == [] and orders_file(env) == OLD_ORDERS


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
    write_mkt(env, "fills.json", [{"id": "nvda-csp-2026-11-16", "status": "rejected"}])
    with pytest.raises(SystemExit, match="rejected"):
        O.main([])


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


def test_a_hedge_night_writes_the_stock_order(env):
    day, for_day = dt.date(2026, 11, 25), dt.date(2026, 11, 27)
    closes = S.closes(end=day)
    divs = FakeLSEG(day).dividends_series()
    q_of = lambda s, d: S.trailing_yield(divs, s, d)                    # noqa: E731
    legs = [("P", 165.0, -10, 0.50), ("C", 195.0, 12, 0.40)]
    frames = [(build_option_ric("NVDA", S.EXPIRY, k, cp, expired=False), priced(closes, cp, S.EXPIRY, k, s, q_of=q_of))
              for cp, k, _, s in legs]
    fake = env["use"](FakeLSEG(day, chain=False, legs=frames), now=dt.datetime(2026, 11, 25, 22, 0))
    write_mkt(env, "positions.json", {"cash": 0, "positions": [
        {"underlying": S.NVDA, "contract": "put" if cp == "P" else "call", "expiry": "2026-12-18", "strike": k, "qty": n}
        for cp, k, n, _ in legs] + [{"underlying": S.NVDA, "contract": "underlying", "qty": -400}]})
    assert O.main([]) == 0
    # by hand: Black-76 delta to spot at each leg's vol on D's close
    spot, r = float(closes.iloc[-1]), S.RATE_PCT / 100
    T = (pd.Timestamp(S.EXPIRY) - pd.Timestamp(day)).days / 365.0
    F = spot * np.exp((r - q_of(spot, pd.Timestamp(day))) * T)
    net = sum(n * 100 * bs_greeks(F, k, T, s, np.exp(-r * T), cp)["delta"] * F / spot for cp, k, n, s in legs)
    target = -int(round(net))
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
    write_mkt(env, "positions.json", {"cash": 0, "positions": [
        {"underlying": S.NVDA, "contract": "put", "expiry": "2026-12-18", "strike": 165, "qty": -10}]})
    with pytest.raises(SystemExit, match="does not know"):
        O.main([])
    assert orders_file(env) == OLD_ORDERS


def test_after_the_expiry_the_leftover_shares_are_closed_without_lseg(env):
    env["use"](FakeLSEG(dt.date(2026, 12, 18), chain=False), now=dt.datetime(2026, 12, 18, 22, 0))
    write_mkt(env, "positions.json", {"cash": 0, "positions": [
        {"underlying": S.NVDA, "contract": "underlying", "qty": -350}]})
    assert O.main([]) == 0
    assert [(o["action"], o["qty"], o["for"]) for o in orders_file(env)] == [("BTC", 350, "2026-12-21")]
    assert env["sessions"] == []
