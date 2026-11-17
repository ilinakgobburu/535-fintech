"""
far_wings_orders.py on the basket, end to end with LSEG replaced by a fake (no network): two names, all 31, a name
whose chain is not final at 22:00 and recovers at the 01:00 retry while the others order, or does not (skipped, and
the retry red), a transient LSEG error asked once more, an unexpected error on one name stopping that name only,
names with no conid or one Sanji does not track, each name's own band, a name stopped by its own book while the
others order (and the rejected order kept in the file so the same check stops it the next night), the checks that
stop every name (the T-bill, the login), and every order legal by the course's order page field by field with
unique ids. NVDA's orders in a basket night are the ones the NVDA-only night writes from the same data.

Made-up data only: each name's closes are fw_live_synth's alternating +-2% series from a level near its 2026-11-17
probe close, and its chain a Black-76 smile on its own grid's strikes under its own LSEG option root.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import sys

import pandas as pd
import pytest

import fw_live_synth as S
from trading_app.lib import far_wings_live as FW
from trading_app.lib import wheel_live as L
from trading_app.lib.ric import build_option_ric

_path = list(sys.path)
sys.path.insert(0, str(S.REPO))
import far_wings_orders as O  # noqa: E402

sys.path[:] = _path

ENTRY, FOR, EXP = S.ENTRY, S.FOR, S.EXPIRY
CFG = S.config()
NAMES = {n["ticker"]: n for n in CFG["names"]}
TICKERS = list(NAMES)
LEVEL = {"NVDA": 175.0, "MSFT": 518.0, "AAPL": 353.0, "AMZN": 258.0, "GOOG": 281.0, "BRK B": 593.0, "META": 626.0,
         "JNJ": 283.0, "JPM": 338.0, "V": 426.0, "XOM": 167.0, "WMT": 128.0, "PFE": 28.9, "UNH": 412.0, "BAC": 50.1,
         "VZ": 46.3, "PG": 173.0, "WFC": 84.8, "INTC": 55.5, "CVX": 217.0, "T": 22.4, "KO": 99.5, "MRK": 144.0,
         "MA": 662.0, "CSCO": 94.5, "HD": 347.0, "BA": 165.0, "DIS": 119.0, "ORCL": 119.0, "PEP": 146.0,
         "CMCSA": 25.6}
VOL = {t: (0.45 if t == "NVDA" else 0.25 + 0.30 * (i % 7) / 6) for i, t in enumerate(TICKERS)}
SKEW = {t: (-0.06 if t == "NVDA" else -0.10 + 0.02 * (i % 6)) for i, t in enumerate(TICKERS)}
WHEEL = {"id": "nvda-csp-2026-11-16", "for": "2026-11-16", "order_type": "lmt", "action": "SHT", "qty": 6,
         "underlying": S.NVDA, "contract": "put", "expiry": "2026-12-18", "strike": "155", "ref": "open",
         "lmt_prc": "7.78"}
UNLISTED = Exception("No data to return, please check errors: ERROR: No successful response.\n"
                     "(TS.Interday.UserRequestError.70005, The universe is not found)")
V_NULL = {**NAMES["V"], "conid": None}          # made up: V as it would be with its IB conid not confirmed


def step_at(name, price):
    return next(float(s) for b, s in name["grid"] if b is None or price < float(b))


def closes_of(ticker, day=ENTRY, last=None):
    return S.closes(end=day, start_px=LEVEL[ticker], last=last)


def chain_rows_of(ticker, day=ENTRY, last=None):
    n = NAMES[ticker]
    spot = float(closes_of(ticker, day, last).iloc[-1])
    return S.chain_rows(spot, day=day, root=n["root"], step=step_at(n, spot), s0=VOL[ticker], b=SKEW[ticker])


class FakeBasket:
    """lseg.data's get_history and get_data for several names, from made-up frames. Records every request."""

    def __init__(self, day, tickers, chain=True, last=None, chain_day=None, legs=()):
        self.day = pd.Timestamp(day)
        self.requests, self.errors, self.options, self.stocks = [], {}, {}, {}
        for t in tickers:
            c = closes_of(t, day, (last or {}).get(t))
            self.stocks[NAMES[t]["ric"]] = c
            if chain:
                cday = pd.Timestamp((chain_day or {}).get(t, day))
                for r in chain_rows_of(t, day, (last or {}).get(t)):
                    prev = cday - pd.Timedelta(days=3)
                    self.options[r["ric"]] = pd.DataFrame({"BID": [r["bid"] * 3, r["bid"]],
                                                           "ASK": [r["ask"] * 3, r["ask"]]},
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
        if universe in self.stocks:
            return self._window(pd.DataFrame({"TRDPRC_1": self.stocks[universe]}), start, end)
        if universe == "US3MT=RR":
            return self._window(pd.DataFrame({"MID_YLD_1": S.RATE_PCT}, index=S.closes(end=self.day.date()).index),
                                start, end)
        if universe in self.options:
            return self._window(self.options[universe], start, end)
        raise UNLISTED

    def get_data(self, universe, fields, parameters=None):
        self.requests.append(("data", universe, tuple(fields), parameters))
        d = S.DIVS
        return pd.DataFrame({"Instrument": universe, "Dividend Ex Date": [str(x.date()) for x in d.index],
                             "Gross Dividend Amount": d.to_numpy(), "Dividend Pay Date": None})

    def asked(self, ticker):
        """Every request about one name: its stock RIC, and its options (its root then a month code and digits)."""
        n = NAMES[ticker]
        root = n["root"]
        return [r for r in self.requests if r[1] == n["ric"] or (r[1].startswith(root) and r[1][len(root)].isalpha()
                                                                  and r[1][len(root) + 1:len(root) + 3].isdigit())]


class Session:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A scratch repo root: mkt/ with the wheel's last order, the repo's config, price_history.json, a fake LSEG."""
    state = {"sessions": []}
    (tmp_path / "mkt").mkdir()
    S.write_json(tmp_path / "mkt" / "open_orders.json", [WHEEL])
    monkeypatch.setattr(O, "ROOT", tmp_path)
    monkeypatch.setattr(O, "MKT", tmp_path / "mkt")
    state["config"] = S.write_json(tmp_path / "config.yaml", CFG)
    monkeypatch.setattr(O, "CONFIG", state["config"])

    def open_lseg():
        s = Session()
        state["sessions"].append(s)
        return s

    monkeypatch.setattr(O, "open_lseg", open_lseg)
    monkeypatch.setattr(O.time, "sleep", lambda s: None)
    out = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    state["root"], state["output"] = tmp_path, out

    def track(*tickers, untracked=()):
        S.write_json(tmp_path / "price_history.json",
                     S.price_history(*(NAMES[t]["conid"] for t in tickers),
                                     untracked=[NAMES[t]["conid"] for t in untracked]))

    def use(fake, now=None):
        monkeypatch.setattr(O, "ld", fake)
        monkeypatch.setattr(O, "now_ny", lambda: now or dt.datetime(2026, 11, 23, 22, 0))
        return fake

    state["track"], state["use"] = track, use
    return state


def orders_file(env):
    return json.loads((env["root"] / "mkt" / "open_orders.json").read_text())


def ledger_file(env):
    path = env["root"] / "data" / "far_wings_ledger.json"
    return json.loads(path.read_text()) if path.exists() else None


def output(env):
    return env["output"].read_text() if env["output"].exists() else ""


def write_mkt(env, name, obj):
    S.write_json(env["root"] / "mkt" / name, obj)


def set_names(env, names):
    cfg = json.loads(env["config"].read_text())
    cfg["names"] = names
    S.write_json(env["config"], cfg)


def by_name(orders):
    out = {}
    for o in orders:
        out.setdefault(FW.owner(o["id"], TICKERS), []).append(o)
    return out


def plan_for(ticker, day=ENTRY, for_day=FOR, book=None, name=None, **kw):
    """The pure plan of one name on the same made-up data as the fake: what the script must write for it."""
    c = closes_of(ticker, day)
    rows = chain_rows_of(ticker, day)
    mkt = FW.market(c, None, pd.Series(S.RATE_PCT, index=c.index), S.DIVS, (), FW.quote_frame(rows))
    return FW.plan_night(CFG, day, for_day, book or FW.Book(), name=name or NAMES[ticker], entry_expiry=EXP, mkt=mkt,
                         **kw)


def assert_legal(orders):
    """order.txt's checks field by field, wheel_live's too, ids unique, each order on its own name's conid."""
    ids = [o["id"] for o in orders]
    assert len(set(ids)) == len(ids), ids
    for o in orders:
        assert S.order_problems(o) == [], o
        L.validate_order(o)
        t = FW.owner(o["id"], TICKERS)
        assert t is not None and o["underlying"] == NAMES[t]["conid"] and o["ref"] == "open", o
        if o["contract"] == "underlying":
            assert o["order_type"] == "mkt" and set(o) == {"id", "for", "order_type", "action", "qty", "underlying",
                                                           "contract", "ref"}
        else:
            assert o["order_type"] == "lmt" and set(o) == L.ORDER_FIELDS
            assert float(o["strike"]) % step_at(NAMES[t], float(o["strike"])) == 0 or NAMES[t]["ticker"] == "NVDA"


# --------------------------------------------------------------------------- entry nights
def test_a_two_name_entry_night_writes_both_names_wings_in_one_file(env, capsys):
    env["track"]("NVDA", "BRK B")
    fake = env["use"](FakeBasket(ENTRY, ["NVDA", "BRK B"]))
    assert O.main([]) == 0
    got = orders_file(env)
    assert [o["id"] for o in got] == ["nvda-fw-put-2026-11-23", "nvda-fw-call-2026-11-23",
                                      "brkb-fw-put-2026-11-23", "brkb-fw-call-2026-11-23"]
    assert by_name(got) == {"NVDA": plan_for("NVDA")["orders"], "BRK B": plan_for("BRK B")["orders"]}
    assert {o["underlying"] for o in by_name(got)["BRK B"]} == {72063691}
    assert all(float(o["strike"]) % 5 == 0 for o in by_name(got)["BRK B"])
    assert_legal(got)
    # the ledger keeps each name apart; one LSEG session; the T-bill asked for once; nothing stopped
    led = ledger_file(env)
    assert set(led["names"]) == {"NVDA", "BRK B"}
    for t in ("NVDA", "BRK B"):
        mine = by_name(got)[t]
        assert led["names"][t]["nights"]["2026-11-23"] == {"for": "2026-11-24", "ids": [o["id"] for o in mine]}
        assert led["names"][t]["orders"] == {o["id"]: o for o in mine}
    assert len(env["sessions"]) == 1 and env["sessions"][0].closed
    assert [r[1] for r in fake.requests].count("US3MT=RR") == 1
    assert output(env) == "for_day=2026-11-24\n"
    # BRK B's chain is asked for under LSEG's root BRKB, its strikes on its own grid
    plan = FW.strike_plan(fake.stocks["BRKb.N"], dt.date(2026, 11, 20), ENTRY, CFG["chain"]["bands"],
                          NAMES["BRK B"]["grid"])
    assert [r[1] for r in fake.asked("BRK B") if r[1] != "BRKb.N"] == [
        build_option_ric("BRKB", EXP, k, cp, expired=False) for cp in ("P", "C") for k in plan[cp]]
    out = capsys.readouterr().out
    assert "Dropped nvda-csp-2026-11-16 (for 2026-11-16, not this strategy's)" in out
    assert "Tonight by name: 2 decided (NVDA 2, BRK B 2)" in out


@pytest.mark.parametrize("all_31", [True, False])
def test_the_31_name_entry_night(env, capsys, all_31):
    # the real config, every conid confirmed by ISIN: all 31 names order; with V's conid made null (made up: a name
    # whose conid is not confirmed), the other 30 do and V gets one line
    names = copy.deepcopy(CFG["names"])
    if not all_31:
        next(n for n in names if n["ticker"] == "V")["conid"] = None
        set_names(env, names)
    NAMES_ = {n["ticker"]: n for n in names}
    S.write_json(env["root"] / "price_history.json",
                 S.price_history(*(n["conid"] for n in names if n["conid"] is not None)))
    fake = env["use"](FakeBasket(ENTRY, TICKERS))
    assert O.main([]) == 0
    got = orders_file(env)
    ordering = [t for t in TICKERS if NAMES_[t]["conid"] is not None]
    assert len(ordering) == (31 if all_31 else 30)
    assert list(by_name(got)) == ordering                          # config order, every tradable name
    assert all([o["contract"] for o in by_name(got)[t]] == ["put", "call"] for t in ordering)
    assert len(got) == 2 * len(ordering) and len({o["id"] for o in got}) == len(got)
    for o in got:
        t = FW.owner(o["id"], TICKERS)
        assert S.order_problems(o) == [] and o["underlying"] == NAMES_[t]["conid"]
        assert o["id"] in (f"{FW.id_root(t)}-fw-put-2026-11-23", f"{FW.id_root(t)}-fw-call-2026-11-23")
        L.validate_order(o)
    assert any(o["id"].startswith("brkb-fw-") for o in got) and not any(" " in o["id"] for o in got)
    # NVDA's orders are the single-name night's, and each name's are its own plan on its own data
    assert by_name(got)["NVDA"] == plan_for("NVDA")["orders"]
    for t in ("MSFT", "BRK B", "JNJ", "BAC", "PFE", "T", "INTC", "ORCL", "V" if all_31 else "DIS"):
        assert by_name(got)[t] == plan_for(t, name=NAMES_[t])["orders"], t
    assert {o["underlying"] for o in got if o["id"].startswith(("v-fw-", "t-fw-"))} == (
        {49462172, 37018770} if all_31 else {37018770})
    led = ledger_file(env)
    assert list(led["names"]) == ordering
    # within the request estimate: each name's closes, dividends and chain, and one T-bill
    n_requests = len(fake.requests)
    assert [r[1] for r in fake.requests].count("US3MT=RR") == 1
    assert n_requests == 1 + sum(len(fake.asked(t)) for t in ordering)
    assert all([r[0] for r in fake.asked(t)][:2] == ["history", "data"] for t in ordering)    # its close, dividends
    assert n_requests < 2500
    out = capsys.readouterr().out
    assert ("no Interactive Brokers conid confirmed yet" in out) is (not all_31)
    if not all_31:
        assert "V: no Interactive Brokers conid confirmed yet (conid: null in config/far_wings_live.yaml): no orders " \
               "tonight" in out
        assert fake.asked("V") == []
    assert "STOPPED" not in out and output(env) == "for_day=2026-11-24\n"


def test_nvdas_orders_in_a_basket_night_are_the_nvda_only_nights(env, tmp_path):
    env["track"]("NVDA")
    env["use"](FakeBasket(ENTRY, ["NVDA"]))
    assert O.main([]) == 0
    alone = orders_file(env)
    write_mkt(env, "open_orders.json", [WHEEL])
    (env["root"] / "data" / "far_wings_ledger.json").unlink()
    env["track"]("NVDA", "MSFT", "BRK B", "PFE", "BAC")
    env["use"](FakeBasket(ENTRY, ["NVDA", "MSFT", "BRK B", "PFE", "BAC"]))
    assert O.main([]) == 0
    assert by_name(orders_file(env))["NVDA"] == alone
    assert [o["id"] for o in alone] == ["nvda-fw-put-2026-11-23", "nvda-fw-call-2026-11-23"]
    assert [(o["strike"], o["qty"], o["lmt_prc"]) for o in alone] == [("145", 41, "0.53"), ("220", 59, "0.24")]


# --------------------------------------------------------------------------- one name not final
def stale_strike(fake, ticker):
    """One listed contract of the name's chain with earlier rows but no D row: LSEG part-way through D."""
    ric = next(r["ric"] for r in chain_rows_of(ticker) if r["cp"] == "P")
    fake.options[ric] = fake.options[ric].iloc[:1]
    return ric


def test_one_names_chain_not_final_at_22_is_decided_at_the_retry_while_the_others_order(env, capsys):
    env["track"]("NVDA", "AAPL", "BRK B")
    fake = FakeBasket(ENTRY, ["NVDA", "AAPL", "BRK B"])
    hole = stale_strike(fake, "AAPL")
    env["use"](fake)
    assert O.main([]) == 0
    first = orders_file(env)
    assert list(by_name(first)) == ["NVDA", "BRK B"]
    assert set(ledger_file(env)["names"]) == {"NVDA", "BRK B"}
    assert output(env) == "for_day=2026-11-24\n"                    # green: the retry has AAPL
    out = capsys.readouterr().out
    assert "AAPL not final: 1 listed 2026-12-18 option(s) have earlier rows but none for 2026-11-23" in out
    assert "left for the 01:00 retry" in out and "left for the 01:00 retry: AAPL" in out
    # 01:00: only AAPL is decided, choosing without the hole; the other names' orders stay as written
    retry = env["use"](fake, now=dt.datetime(2026, 11, 24, 1, 0))
    fake.requests.clear()
    assert O.main(["--retry"]) == 0
    out = capsys.readouterr().out
    assert "Retry: NVDA, BRK B already written for 2026-11-24 from 2026-11-23's close; this run decides AAPL." in out
    assert f"Note: chose without 1 listed contract(s) with no 2026-11-23 quote ({hole})" in out
    assert retry.asked("NVDA") == [] and retry.asked("BRK B") == [] and retry.asked("AAPL")
    got = orders_file(env)
    assert list(by_name(got)) == ["NVDA", "AAPL", "BRK B"]              # config order
    assert by_name(got)["NVDA"] == by_name(first)["NVDA"] and by_name(got)["BRK B"] == by_name(first)["BRK B"]
    assert [o["id"] for o in by_name(got)["AAPL"]] == ["aapl-fw-put-2026-11-23", "aapl-fw-call-2026-11-23"]
    assert_legal(got)
    led = ledger_file(env)
    assert led["names"]["AAPL"]["nights"]["2026-11-23"]["ids"] == [o["id"] for o in by_name(got)["AAPL"]]
    assert led["names"]["NVDA"]["nights"]["2026-11-23"]["ids"] == [o["id"] for o in by_name(first)["NVDA"]]
    # a second retry finds every name written
    fake.requests.clear()
    assert O.main(["--retry"]) == 0
    assert "were already written (NVDA: nvda-fw-put-2026-11-23, nvda-fw-call-2026-11-23; AAPL: " in \
        capsys.readouterr().out
    assert fake.requests == []


@pytest.mark.parametrize("holds", [False, True])
def test_a_name_still_not_final_at_the_retry_is_skipped_and_the_retry_goes_red(env, capsys, holds):
    # whether or not AAPL holds positions, the retry is the last run before the open: its entry (or its hedge) is
    # lost for the night, so the run is red, though the 22:00 run's orders for NVDA stand
    env["track"]("NVDA", "AAPL")
    if holds:
        # AAPL holds 120 shares from a hedge of its own (the leftover the expiry night's order did not close)
        hedge = {"id": "aapl-fw-hedge-2026-11-19", "for": "2026-11-20", "order_type": "mkt", "action": "BUY",
                 "qty": 120, "underlying": NAMES["AAPL"]["conid"], "contract": "underlying", "ref": "open"}
        S.write_json(env["root"] / "data" / "far_wings_ledger.json",
                     FW.record_night(FW.read_ledger(None), "AAPL", [hedge], dt.date(2026, 11, 19),
                                     dt.date(2026, 11, 20)))
        write_mkt(env, "fills.json", [{"id": hedge["id"], "status": "filled"}, {"id": WHEEL["id"], "status": "unfilled"}])
        write_mkt(env, "positions.json", {"cash": 0, "positions": [
            {"underlying": NAMES["AAPL"]["conid"], "contract": "underlying", "qty": 120}]})
    fake = env["use"](FakeBasket(ENTRY, ["NVDA", "AAPL"], chain_day={"AAPL": "2026-11-20"}))   # no D row at all
    assert O.main([]) == 0
    assert list(by_name(orders_file(env))) == ["NVDA"]
    capsys.readouterr()
    first = (env["root"] / "mkt" / "open_orders.json").read_bytes()
    env["use"](fake, now=dt.datetime(2026, 11, 24, 1, 0))
    hedge = (" Tonight's hedge was not written either: the book's \\+120 AAPL shares stay as they are until a run "
             "succeeds \\(the Run button\\)." if holds else "")
    with pytest.raises(SystemExit, match="^The retry decided nothing more, and AAPL: still not final at the retry "
                                         "\\(No 2026-12-18 option has a 2026-11-23 row on LSEG yet: data not "
                                         f"final\\): no order written.{hedge}$"):
        O.main(["--retry"])
    out = capsys.readouterr().out
    assert "AAPL skipped: still not final at the retry (No 2026-12-18 option has a 2026-11-23 row on LSEG yet: " \
           "data not final), so no order for it from 2026-11-23's close." in out
    assert "Tonight by name: 0 decided; written by the 22:00 run: NVDA; SKIPPED at the retry (red): AAPL" in out
    # the 22:00 run's orders stand as written; AAPL has no night in the ledger, so a Run button can still decide it
    assert (env["root"] / "mkt" / "open_orders.json").read_bytes() == first
    assert list(by_name(orders_file(env))) == ["NVDA"]
    assert "2026-11-23" not in ledger_file(env)["names"].get("AAPL", {"nights": {}})["nights"]
    assert output(env) == "for_day=2026-11-24\n"
    assert ("Tonight's hedge was not written either: the book's +120 AAPL shares stay as they are" in out) is holds
    assert "nothing more could be decided" not in out


def test_a_retry_that_decides_one_name_and_skips_another_is_red_after_the_commit(env, capsys):
    # 22:00: NVDA ordered, BRK B's chain part-way through D, AAPL's with no D row at all. 01:00: BRK B decided
    # (choosing without its hole) and written; AAPL still not final, so skipped= turns the run red after the commit
    env["track"]("NVDA", "AAPL", "BRK B")
    fake = FakeBasket(ENTRY, ["NVDA", "AAPL", "BRK B"], chain_day={"AAPL": "2026-11-20"})
    stale_strike(fake, "BRK B")
    env["use"](fake)
    assert O.main([]) == 0
    assert list(by_name(orders_file(env))) == ["NVDA"] and output(env) == "for_day=2026-11-24\n"
    env["output"].unlink()
    capsys.readouterr()
    env["use"](fake, now=dt.datetime(2026, 11, 24, 1, 0))
    assert O.main(["--retry"]) == 0
    assert list(by_name(orders_file(env))) == ["NVDA", "BRK B"]
    assert set(ledger_file(env)["names"]) == {"NVDA", "BRK B"}
    assert output(env) == "for_day=2026-11-24\nskipped=AAPL\n"
    out = capsys.readouterr().out
    assert "Tonight by name: 1 decided (BRK B 2); written by the 22:00 run: NVDA; SKIPPED at the retry (the Action " \
           "goes red after the commit): AAPL" in out


def stale_share(fake, ticker, share):
    """More than `share` of the listed contracts the night asks for with earlier rows but no D row."""
    n_ = NAMES[ticker]
    plan = FW.strike_plan(fake.stocks[n_["ric"]], dt.date(2026, 11, 20), ENTRY, CFG["chain"]["bands"], n_["grid"])
    rics = [r for r in (build_option_ric(n_["root"], EXP, k, cp, expired=False) for cp in ("P", "C") for k in plan[cp])
            if r in fake.options]
    n = int(len(rics) * share) + 1
    for ric in rics[::max(1, len(rics) // n)][:n]:
        fake.options[ric] = fake.options[ric].iloc[:1]
    return n, len(rics)


def test_a_chain_under_90_percent_dated_at_the_retry_is_skipped_not_chosen_from(env, capsys):
    # the retry takes a few holes as holes (min_dated_share 0.9), but a chain with more than a tenth of its listed
    # contracts missing D is partial: still not final, so AAPL is skipped and the retry is red
    env["track"]("NVDA", "AAPL")
    fake = FakeBasket(ENTRY, ["NVDA", "AAPL"])
    n, listed = stale_share(fake, "AAPL", 0.12)
    assert n / listed > 0.1
    env["use"](fake)
    assert O.main([]) == 0
    capsys.readouterr()
    env["use"](fake, now=dt.datetime(2026, 11, 24, 1, 0))
    with pytest.raises(SystemExit, match=f"^The retry decided nothing more, and AAPL: still not final at the retry "
                                         f"\\(Only {listed - n} of the {listed} listed 2026-12-18 options have a "
                                         f"2026-11-23 row \\(under 90%\\): the chain looks partial"):
        O.main(["--retry"])
    assert list(by_name(orders_file(env))) == ["NVDA"]
    assert "2026-11-23" not in ledger_file(env)["names"].get("AAPL", {"nights": {}})["nights"]


def test_every_name_not_final_at_22_stops_red_with_nothing_written(env):
    env["track"]("NVDA", "AAPL")
    env["use"](FakeBasket(ENTRY, ["NVDA", "AAPL"], chain_day={"NVDA": "2026-11-20", "AAPL": "2026-11-20"}))
    with pytest.raises(SystemExit, match="^No name could be decided tonight, so no order was written. NVDA: No "
                                         "2026-12-18 option has a 2026-11-23 row .* AAPL: No 2026-12-18 option"):
        O.main([])
    assert orders_file(env) == [WHEEL] and ledger_file(env) is None and output(env) == ""


def test_one_names_lseg_error_is_that_names_not_final(env, capsys):
    env["track"]("NVDA", "AAPL")
    fake = FakeBasket(ENTRY, ["NVDA", "AAPL"])
    fake.errors["AAPL.OQ"] = Exception("Error code 404 | Not Found")
    env["use"](fake)
    assert O.main([]) == 0
    assert list(by_name(orders_file(env))) == ["NVDA"]
    assert "AAPL not final: LSEG error on AAPL.OQ: Error code 404 | Not Found; left for the 01:00 retry" in \
        capsys.readouterr().out


class Flaky(FakeBasket):
    """LSEG failing once on each RIC in `once` (a transient error), then answering as before."""

    def __init__(self, *args, once=(), **kw):
        super().__init__(*args, **kw)
        self.once = set(once)

    def get_history(self, universe, fields, **kw):
        if universe in self.once:
            self.once.discard(universe)
            self.requests.append(("history", universe, tuple(fields), kw.get("adjustments")))
            raise Exception("Error code 404 | Not Found\nretry later")
        return super().get_history(universe, fields, **kw)


def test_a_transient_lseg_error_is_asked_once_more_and_the_name_is_decided(env, capsys, monkeypatch):
    # one failure each on AAPL's close, one of its options and the T-bill: each is asked once more 5 s later and
    # every name is decided as without the errors; a RIC LSEG does not know (70005) is asked once only
    env["track"]("NVDA", "AAPL")
    put = next(r["ric"] for r in chain_rows_of("AAPL") if r["cp"] == "P")
    fake = env["use"](Flaky(ENTRY, ["NVDA", "AAPL"], once={"AAPL.OQ", put, "US3MT=RR"}))
    sleeps = []
    monkeypatch.setattr(O.time, "sleep", sleeps.append)
    assert O.main([]) == 0
    got = orders_file(env)
    assert by_name(got) == {"NVDA": plan_for("NVDA")["orders"], "AAPL": plan_for("AAPL")["orders"]}
    asked = [r[1] for r in fake.requests if r[0] == "history"]
    assert asked.count("AAPL.OQ") == asked.count(put) == asked.count("US3MT=RR") == 2
    unlisted = [u for u in set(asked) if u not in fake.stocks and u not in fake.options and u != "US3MT=RR"]
    assert unlisted and all(asked.count(u) == 1 for u in unlisted)
    assert sleeps.count(O.RETRY_PAUSE) == 3 and O.RETRY_PAUSE == 5.0
    assert len(sleeps) == len(fake.requests) + 3 and set(sleeps) == {O.PAUSE, O.RETRY_PAUSE}
    out = capsys.readouterr().out
    assert "LSEG error on AAPL.OQ: Error code 404 | Not Found; asking once more in 5 s" in out
    assert output(env) == "for_day=2026-11-24\n"


def test_a_t_bill_error_stops_every_name_with_a_clear_exit(env):
    env["track"]("NVDA", "AAPL")
    fake = FakeBasket(ENTRY, ["NVDA", "AAPL"])
    fake.errors["US3MT=RR"] = Exception("Error code 500 | Internal Server Error")
    env["use"](fake)
    with pytest.raises(SystemExit, match="^LSEG error on US3MT=RR \\(every name's rate\\): Error code 500 \\| "
                                         "Internal Server Error: no order written.$"):
        O.main([])
    assert [r[1] for r in fake.requests].count("US3MT=RR") == 2
    assert orders_file(env) == [WHEEL] and ledger_file(env) is None and env["sessions"][0].closed


def test_a_failed_lseg_login_stops_the_night_once(env, monkeypatch):
    env["track"]("NVDA", "AAPL")
    env["use"](FakeBasket(ENTRY, ["NVDA", "AAPL"]))
    tries = []

    def refused():
        tries.append(1)
        raise ConnectionError("connection reset")

    monkeypatch.setattr(O, "open_lseg", refused)
    with pytest.raises(SystemExit, match="^Can't open the LSEG session \\(ConnectionError\\): no order written.$"):
        O.main([])
    assert tries == [1] and orders_file(env) == [WHEEL] and ledger_file(env) is None


class OddDividends(FakeBasket):
    """LSEG answering AAPL's dividends with something that is not a frame."""

    def get_data(self, universe, fields, parameters=None):
        if universe == "AAPL.OQ":
            self.requests.append(("data", universe, tuple(fields), parameters))
            return "not a frame"
        return super().get_data(universe, fields, parameters)


def test_an_unexpected_error_on_one_name_stops_that_name_only(env, capsys):
    env["track"]("NVDA", "AAPL", "BRK B")
    env["use"](OddDividends(ENTRY, ["NVDA", "AAPL", "BRK B"]))
    assert O.main([]) == 0
    got = orders_file(env)
    assert by_name(got) == {"NVDA": plan_for("NVDA")["orders"], "BRK B": plan_for("BRK B")["orders"]}
    assert set(ledger_file(env)["names"]) == {"NVDA", "BRK B"}
    assert output(env) == "for_day=2026-11-24\nstopped=AAPL\n"
    out = capsys.readouterr().out
    assert "AAPL stopped, no orders tonight: unexpected AttributeError on its data or plan:" in out
    assert "STOPPED (the Action goes red after the commit): AAPL" in out


def test_write_orders_checks_every_order_of_this_strategys_before_writing(env):
    # plan_night checks each name's orders; write_orders checks the whole file again before touching it
    bad = {**plan_for("NVDA")["orders"][0], "strike": 145.0}
    before = (env["root"] / "mkt" / "open_orders.json").read_bytes()
    with pytest.raises(ValueError, match="strike must be a string token"):
        O.write_orders([WHEEL], {"NVDA": [bad]}, TICKERS, ENTRY, FOR, False, FW.read_ledger(None), {})
    assert (env["root"] / "mkt" / "open_orders.json").read_bytes() == before and ledger_file(env) is None


def test_the_chain_and_held_legs_are_asked_under_the_names_option_root(env, monkeypatch):
    # every real name's root is its ticker without spaces (BRKB for BRK B); a made-up root BRKZ shows the chain and
    # the held legs are asked under the root, not the ticker
    monkeypatch.setitem(NAMES, "BRK B", {**NAMES["BRK B"], "root": "BRKZ"})
    set_names(env, [NAMES["NVDA"], NAMES["BRK B"]])
    env["track"]("NVDA", "BRK B")
    fake = env["use"](FakeBasket(ENTRY, ["NVDA", "BRK B"]))
    assert O.main([]) == 0
    assert by_name(orders_file(env))["BRK B"] == plan_for("BRK B")["orders"]
    opts = [r[1] for r in fake.asked("BRK B") if r[1] != "BRKb.N"]
    assert opts and all(r.startswith("BRKZ") for r in opts)
    day = dt.date(2026, 11, 30)
    legs = [("BRK B", "P", 500.0, 3)]
    held_books(env, legs)
    fake = env["use"](FakeBasket(day, ["NVDA", "BRK B"], chain=False, legs=leg_frames(day, legs)),
                      now=dt.datetime(2026, 11, 30, 22, 0))
    assert O.main([]) == 0
    assert build_option_ric("BRKZ", EXP, 500.0, "P", expired=False) in [r[1] for r in fake.requests]
    assert "2026-11-30" in ledger_file(env)["names"]["BRK B"]["nights"]


class NoRate(FakeBasket):
    """LSEG with no T-bill print in the window."""

    def get_history(self, universe, fields, **kw):
        if universe == "US3MT=RR":
            self.requests.append(("history", universe, tuple(fields), None))
            return pd.DataFrame()
        return super().get_history(universe, fields, **kw)


def test_the_t_bill_missing_stops_every_name(env):
    # every name's rate: without it none can be decided, so the run stops before writing, as before
    env["track"]("NVDA", "AAPL")
    env["use"](NoRate(ENTRY, ["NVDA", "AAPL"]))
    with pytest.raises(SystemExit, match="LSEG returned no US3MT=RR yield up to 2026-11-23: no order written"):
        O.main([])
    assert orders_file(env) == [WHEEL] and env["sessions"][0].closed


# --------------------------------------------------------------------------- names that cannot trade
def test_a_null_conid_and_an_untracked_conid_get_no_orders_and_never_stop_the_others(env, capsys):
    names = [NAMES["NVDA"], V_NULL, NAMES["AAPL"], NAMES["MSFT"]]
    set_names(env, names)
    env["track"]("NVDA", untracked=["AAPL"])                     # AAPL listed tracked: false; MSFT not there at all
    fake = env["use"](FakeBasket(ENTRY, ["NVDA", "V", "AAPL", "MSFT"]))
    assert O.main([]) == 0
    assert list(by_name(orders_file(env))) == ["NVDA"]
    out = capsys.readouterr().out
    assert "V: no Interactive Brokers conid confirmed yet (conid: null in config/far_wings_live.yaml): no orders " \
           "tonight" in out
    for t in ("AAPL", "MSFT"):
        assert f"{t}: conid {NAMES[t]['conid']} is not in price_history.json with tracked: true, so Sanji does not " \
               f"price it and could not resolve its orders: no orders tonight" in out
    assert fake.asked("V") == fake.asked("AAPL") == fake.asked("MSFT") == []
    assert set(ledger_file(env)["names"]) == {"NVDA"} and "stopped" not in output(env)
    # --include-untracked (a dry run) decides AAPL and MSFT too; V still has no conid to write
    capsys.readouterr()
    assert O.main(["--dry-run", "--as-of", "2026-11-23", "--include-untracked"]) == 0
    out = capsys.readouterr().out
    assert "decided anyway (--include-untracked): a live run gives it no orders" in out
    assert '"id": "aapl-fw-put-2026-11-23"' in out and '"id": "msft-fw-call-2026-11-23"' in out
    assert '"id": "v-fw-' not in out


def test_an_untracked_name_holding_positions_turns_the_run_red(env, capsys):
    set_names(env, [NAMES["NVDA"], NAMES["AAPL"]])
    env["track"]("NVDA")
    write_mkt(env, "fills.json", [{"id": WHEEL["id"], "status": "unfilled"}])
    write_mkt(env, "positions.json", {"cash": 0, "positions": [
        {"underlying": NAMES["AAPL"]["conid"], "contract": "underlying", "qty": 50}]})
    env["use"](FakeBasket(ENTRY, ["NVDA"]))
    assert O.main([]) == 0
    assert list(by_name(orders_file(env))) == ["NVDA"]
    assert output(env).endswith("stopped=AAPL\n")
    assert "STOPPED (the Action goes red after the commit): AAPL" in capsys.readouterr().out


def test_no_tracked_name_writes_nothing_and_a_missing_price_history_stops(env, capsys):
    env["track"]()
    fake = env["use"](FakeBasket(ENTRY, ["NVDA"]))
    assert O.main([]) == 0
    assert "No name can trade tonight (see above): nothing written." in capsys.readouterr().out
    assert orders_file(env) == [WHEEL] and fake.requests == [] and output(env) == ""
    (env["root"] / "price_history.json").unlink()
    with pytest.raises(SystemExit, match="Can't tell which conids Sanji tracks"):
        O.main([])
    S.write_json(env["root"] / "price_history.json", [4815747])
    with pytest.raises(SystemExit, match="Can't tell which conids Sanji tracks"):
        O.main([])


def test_tracked_conids_reads_only_tracked_true():
    got = FW.tracked_conids({"4815747": {"tracked": True}, "265598": {"tracked": False}, "1": {"tracked": "true"},
                             "x": {"tracked": True}, "8719": [], "72063691": {"tracked": True, "price_history": {}}})
    assert got == {4815747, 72063691}
    assert FW.untradable(NAMES["NVDA"], got) is None and FW.untradable(NAMES["AAPL"], got).startswith("conid 265598")
    assert FW.untradable(V_NULL, got).startswith("no Interactive Brokers conid")
    assert FW.untradable(NAMES["V"], {49462172}) is None and FW.untradable(NAMES["T"], {37018770}) is None


# --------------------------------------------------------------------------- each name its own book
@pytest.fixture
def exact_delta(monkeypatch):
    deltas = {}

    def fake(mkt, cp, expiry, strike, history, day):
        return {"iv": 0.5, "iv_day": pd.Timestamp(day), "mid": 1.0, "delta": deltas[(cp, strike)]}

    monkeypatch.setattr(FW, "leg_delta", fake)
    return deltas


def held_books(env, legs):
    """Each name's entry filled as `legs` [(ticker, cp, strike, contracts)], the ledger, fills and positions to match."""
    led, fills, positions, written = FW.read_ledger(None), [], [], []
    for t, cp, k, qty in legs:
        o = FW.option_orders(root=t, conid=NAMES[t]["conid"], day=ENTRY, for_day=FOR, cp=cp, expiry=EXP, strike=k,
                             limit="0.5", held=0, target=-qty)
        led = FW.record_night(led, t, o, ENTRY, FOR)
        fills += [{"id": x["id"], "status": "filled"} for x in o]
        written += o
        positions.append({"underlying": NAMES[t]["conid"], "contract": "put" if cp == "P" else "call",
                          "expiry": EXP.isoformat(), "strike": k, "qty": -qty})
    S.write_json(env["root"] / "data" / "far_wings_ledger.json", led)
    write_mkt(env, "open_orders.json", written)
    write_mkt(env, "fills.json", fills)
    write_mkt(env, "positions.json", {"cash": 0, "positions": positions})


def leg_frames(day, legs):
    out = []
    for t, cp, k, _ in legs:
        c = closes_of(t, day)
        h = S.leg_history(c, cp, EXP, k, 0.4)
        out.append((build_option_ric(NAMES[t]["root"], EXP, k, cp, expired=False),
                    pd.DataFrame({"BID": h["bid"], "ASK": h["ask"]})))
    return out


@pytest.mark.parametrize("case", ["opposite", "same_sign"])
def test_each_name_is_banded_on_its_own_book_never_pooled(env, exact_delta, case):
    day = dt.date(2026, 11, 30)
    env["track"]("NVDA", "PFE")
    legs = [("NVDA", "P", 180.0, 32), ("PFE", "C", 30.0, 32)]
    held_books(env, legs)
    if case == "opposite":
        # NVDA +200 shares at $25.00 is $5,000 (inside, exactly); PFE -200 at $25.01 is $5,002 (outside): only PFE
        # hedges. Pooled, the two would net to $2 and neither would.
        exact_delta[("P", 180.0)], exact_delta[("C", 30.0)] = -0.0625, 0.0625
        last = {"NVDA": 25.0, "PFE": 25.01}
    else:
        # each +160 shares ($4,000 at $25): inside each name's own band, though pooled ($8,000) they would breach
        exact_delta[("P", 180.0)], exact_delta[("C", 30.0)] = -0.05, -0.05
        last = {"NVDA": 25.0, "PFE": 25.0}
    fake = FakeBasket(day, ["NVDA", "PFE"], chain=False, last=last, legs=leg_frames(day, legs))
    env["use"](fake, now=dt.datetime(2026, 11, 30, 22, 0))
    assert O.main([]) == 0
    got = orders_file(env)
    if case == "opposite":
        assert [(o["id"], o["action"], o["qty"], o["underlying"]) for o in got] == [
            ("pfe-fw-hedge-2026-11-30", "BUY", 200, NAMES["PFE"]["conid"])]
    else:
        assert got == []
    assert_legal(got)
    led = ledger_file(env)
    assert led["names"]["NVDA"]["nights"]["2026-11-30"] == {"for": "2026-12-01", "ids": []}
    assert led["names"]["PFE"]["nights"]["2026-11-30"]["ids"] == [o["id"] for o in got]


def test_the_hedge_limit_is_each_names_own(env, exact_delta, capsys):
    # PFE's short calls deep in the money at $2,500 need +3,200 shares, $8,000,000: past 40 x ITS $100,000 book, so
    # PFE stops (its book is probably misread); NVDA, inside its own limit, hedges on
    day = dt.date(2026, 11, 30)
    env["track"]("NVDA", "PFE")
    legs = [("NVDA", "P", 180.0, 32), ("PFE", "C", 30.0, 32)]
    held_books(env, legs)
    exact_delta[("P", 180.0)], exact_delta[("C", 30.0)] = -0.0625, 1.0
    fake = FakeBasket(day, ["NVDA", "PFE"], chain=False, last={"NVDA": 25.01, "PFE": 2500.0},
                      legs=leg_frames(day, legs))
    env["use"](fake, now=dt.datetime(2026, 11, 30, 22, 0))
    assert O.main([]) == 0
    # NVDA's hedge, and PFE's orders kept as the file had them (a stopped name's are not rewritten)
    assert [(o["id"], o["action"], o["qty"]) for o in orders_file(env)] == [("nvda-fw-hedge-2026-11-30", "SHT", 200),
                                                                         ("pfe-fw-call-2026-11-23", "SHT", 32)]
    assert output(env).endswith("stopped=PFE\n")
    assert "PFE stopped, no orders tonight: the hedge target, +3200 PFE shares ($8,000,000 at 2500.00), is more " \
           "than 40 x the $100,000 book" in capsys.readouterr().out
    assert "2026-11-30" not in ledger_file(env)["names"]["PFE"]["nights"]


# --------------------------------------------------------------------------- a name stopped by its own book
def test_a_name_stopped_by_its_own_book_gets_no_orders_while_the_others_do(env, capsys):
    env["track"]("NVDA", "BRK B")
    write_mkt(env, "fills.json", [{"id": WHEEL["id"], "status": "filled"}])
    write_mkt(env, "positions.json", {"cash": 0, "positions": [
        {"underlying": S.NVDA, "contract": "put", "expiry": "2026-12-18", "strike": 155, "qty": -6}]})
    fake = env["use"](FakeBasket(ENTRY, ["NVDA", "BRK B"]))
    assert O.main([]) == 0
    got = orders_file(env)
    assert list(by_name(got)) == ["BRK B"] and by_name(got)["BRK B"] == plan_for("BRK B")["orders"]
    assert fake.asked("NVDA") == []
    assert output(env) == "for_day=2026-11-24\nstopped=NVDA\n"
    out = capsys.readouterr().out
    assert "NVDA stopped, no orders tonight: The book is not this strategy's alone" in out
    assert "the book holds -6 2026-12-18 155 put, which no filled nvda-fw-* order accounts for" in out
    assert set(ledger_file(env)["names"]) == {"BRK B"}


def test_a_rejected_order_stops_its_name_every_night_until_acknowledged(env, capsys):
    env["track"]("NVDA", "BRK B")
    env["use"](FakeBasket(ENTRY, ["NVDA", "BRK B"]))
    assert O.main([]) == 0
    entry = orders_file(env)
    mine = by_name(entry)
    # Sanji rejects NVDA's put, fills the rest
    fills = [{"id": o["id"], "status": "rejected" if o["id"] == "nvda-fw-put-2026-11-23" else "filled"} for o in entry]
    write_mkt(env, "fills.json", fills)
    write_mkt(env, "positions.json", {"cash": 0, "positions": [
        {"underlying": NAMES[FW.owner(o["id"], TICKERS)]["conid"], "contract": o["contract"], "expiry": o["expiry"],
         "strike": float(o["strike"]), "qty": -o["qty"]} for o in entry if o["id"] != "nvda-fw-put-2026-11-23"]})
    brk_legs = [("BRK B", o["contract"][0].upper(), float(o["strike"]), o["qty"]) for o in mine["BRK B"]]
    for night in (dt.date(2026, 11, 24), dt.date(2026, 11, 25)):
        fake = FakeBasket(night, ["NVDA", "BRK B"], chain=False, legs=leg_frames(night, brk_legs))
        env["use"](fake, now=dt.datetime.combine(night, dt.time(22, 0)))
        assert O.main([]) == 0
        got = orders_file(env)
        # NVDA's orders stay in the file as they were, so the same check stops it again the next night
        assert by_name(got)["NVDA"] == mine["NVDA"] and fake.asked("NVDA") == []
        assert output(env).endswith("stopped=NVDA\n")
        assert "Sanji rejected nvda-fw-put-2026-11-23" in capsys.readouterr().out
        # BRK B is banded on its own; its hedge (if any) is answered by Sanji before the next night
        new = by_name(got).get("BRK B", [])
        assert all(o["for"] == FW.next_session(night).isoformat() for o in new)
        write_mkt(env, "fills.json", json.loads((env["root"] / "mkt" / "fills.json").read_text())
                  + [{"id": o["id"], "status": "filled"} for o in new])
        pos = json.loads((env["root"] / "mkt" / "positions.json").read_text())
        for o in new:
            pos["positions"].append({"underlying": NAMES["BRK B"]["conid"], "contract": "underlying",
                                     "qty": (1 if o["action"] in ("BUY", "BTC") else -1) * o["qty"]})
        write_mkt(env, "positions.json", pos)
        env["output"].unlink()
    cfg = json.loads(env["config"].read_text())
    cfg["acknowledged"] = {"nvda-fw-put-2026-11-23": "rejected in a test; the call filled alone"}
    S.write_json(env["config"], cfg)
    night = dt.date(2026, 11, 26)
    nvda_legs = [("NVDA", "C", float(mine["NVDA"][1]["strike"]), mine["NVDA"][1]["qty"])]
    env["use"](FakeBasket(night, ["NVDA", "BRK B"], chain=False, legs=leg_frames(night, brk_legs + nvda_legs)),
               now=dt.datetime(2026, 11, 26, 22, 0))
    assert O.main([]) == 0
    got = orders_file(env)
    assert not any(o["id"] in {"nvda-fw-put-2026-11-23", "nvda-fw-call-2026-11-23"} for o in got)
    assert "stopped" not in output(env)
    assert "acknowledged in config/far_wings_live.yaml: rejected in a test" in capsys.readouterr().out


# --------------------------------------------------------------------------- what stops every name
def test_an_unanswered_order_of_one_name_stops_every_name(env):
    env["track"]("NVDA", "BRK B")
    env["use"](FakeBasket(ENTRY, ["NVDA", "BRK B"]))
    assert O.main([]) == 0
    entry = orders_file(env)
    write_mkt(env, "fills.json", [{"id": o["id"], "status": "filled"} for o in entry if o["id"] != "brkb-fw-call-2026-11-23"])
    day = dt.date(2026, 11, 24)
    env["use"](FakeBasket(day, ["NVDA", "BRK B"], chain=False), now=dt.datetime(2026, 11, 24, 22, 0))
    with pytest.raises(SystemExit, match="no result yet for brkb-fw-call-2026-11-23: the book may be stale"):
        O.main([])
    assert orders_file(env) == entry


def test_an_option_listed_past_its_expiry_on_one_name_stops_every_name(env):
    env["track"]("NVDA", "BRK B")
    held_books(env, [("BRK B", "P", 500.0, 3)])
    env["use"](FakeBasket(EXP, ["NVDA", "BRK B"], chain=False), now=dt.datetime(2026, 12, 18, 22, 0))
    with pytest.raises(SystemExit, match="still lists options that expired by 2026-12-18: .*'strike': 500.0"):
        O.main([])


def test_a_run_during_an_open_session_keeps_every_names_orders(env, capsys):
    env["track"]("NVDA", "BRK B")
    env["use"](FakeBasket(ENTRY, ["NVDA", "BRK B"]))
    assert O.main([]) == 0
    written, led = orders_file(env), ledger_file(env)
    write_mkt(env, "fills.json", [])
    write_mkt(env, "positions.json", {"cash": 0, "positions": []})
    env["use"](FakeBasket(ENTRY, ["NVDA", "BRK B"], chain=False), now=dt.datetime(2026, 11, 24, 12, 0))
    assert O.main([]) == 0
    assert orders_file(env) == written and ledger_file(env) == led
    assert "its session has not been resolved yet, so no new order" in capsys.readouterr().out


# --------------------------------------------------------------------------- the one-off flags
def test_only_is_a_dry_run_of_those_names(env, capsys):
    env["track"]("NVDA", "AAPL", "BRK B", "META")
    fake = env["use"](FakeBasket(ENTRY, ["NVDA", "AAPL", "BRK B", "META"]))
    with pytest.raises(SystemExit, match="add --dry-run"):
        O.main(["--only", "AAPL,BRK B"])
    with pytest.raises(SystemExit, match="--only names \\['BRKB'\\] not in"):
        O.main(["--dry-run", "--only", "AAPL,BRKB"])
    before = (env["root"] / "mkt" / "open_orders.json").read_bytes()
    assert O.main(["--dry-run", "--as-of", "2026-11-23", "--only", "AAPL, BRK B"]) == 0
    assert (env["root"] / "mkt" / "open_orders.json").read_bytes() == before and ledger_file(env) is None
    assert fake.asked("NVDA") == [] and fake.asked("META") == [] and fake.asked("AAPL") and fake.asked("BRK B")
    out = capsys.readouterr().out
    assert '"id": "aapl-fw-put-2026-11-23"' in out and '"id": "brkb-fw-call-2026-11-23"' in out


# --------------------------------------------------------------------------- the pure composition
def test_compose_orders_keeps_each_undecided_names_orders_and_drops_another_strategys():
    t = ["NVDA", "AAPL", "BRK B"]
    a = {"id": "aapl-fw-put-2026-11-23", "for": "2026-11-24"}
    b = {"id": "brkb-fw-call-2026-11-23", "for": "2026-11-24"}
    n_old = {"id": "nvda-fw-hedge-2026-11-20", "for": "2026-11-23"}
    n_new = {"id": "nvda-fw-put-2026-11-23", "for": "2026-11-24"}
    written = [WHEEL, n_old, b, a]
    got, note = FW.compose_orders(written, {"NVDA": [n_new]}, t, ENTRY, FOR)
    assert (got, note) == ([n_new, a, b], "")
    got, _ = FW.compose_orders(written, {"NVDA": [], "AAPL": []}, t, ENTRY, FOR)
    assert got == [b]
    # an order for the session in progress: kept alone, nothing added
    live = {"id": "brkb-fw-hedge-2026-11-23", "for": "2026-11-24"}
    got, note = FW.compose_orders([live, a], {"NVDA": [n_new]}, t, ENTRY, dt.date(2026, 11, 25))
    assert got == [live, a] and "its session has not been resolved yet" in note
    # her own order for a session after D stays; one for a session up to D leaves, and so do the retired
    # strategies' own orders whatever their session (the vol-arb's and the wheel's ids), as before the basket
    hand = {**WHEEL, "id": "nvda-close-wheel-2026-11-23", "for": "2026-11-24", "action": "BTC"}
    va = {**WHEEL, "id": "nvda-va-put-2026-11-23", "for": "2026-11-24"}
    csp = {**WHEEL, "id": "nvda-csp-2026-11-24", "for": "2026-11-24"}
    cc = {**WHEEL, "id": "nvda-cc-2026-11-24", "for": "2026-11-24", "contract": "call"}
    done = {**hand, "id": "nvda-close-wheel-2026-11-20", "for": "2026-11-23"}        # her order for D: resolved
    got, _ = FW.compose_orders([WHEEL, va, done, hand, csp, a, cc], {"NVDA": [n_new]}, t, ENTRY, FOR)
    assert got == [n_new, a, hand]
    assert [bool(FW.RETIRED_IDS.match(i)) for i in ("nvda-va-hedge-2026-11-23", "nvda-csp-2026-11-05",
                                                    "tsla-cc-2026-11-05", "nvda-close-wheel-2026-11-23",
                                                    "nvda-fw-put-2026-11-23", "my-cc-note")] == [True, True, True,
                                                                                                False, False, True]


def test_her_hand_order_closing_a_stopped_names_position_survives_the_others_being_written(env, capsys):
    # NVDA holds the wheel's last put, so it is stopped; she writes a BTC for it by hand. BRK B's entry is written,
    # and her order stays in the file for Sanji (before the basket, the stopped run left the file alone)
    env["track"]("NVDA", "BRK B")
    hand = {**WHEEL, "id": "nvda-close-wheel-put-2026-11-23", "for": "2026-11-24", "action": "BTC", "order_type": "mkt"}
    del hand["lmt_prc"]
    write_mkt(env, "open_orders.json", [WHEEL, hand])
    write_mkt(env, "fills.json", [{"id": WHEEL["id"], "status": "filled"}])
    write_mkt(env, "positions.json", {"cash": 0, "positions": [
        {"underlying": S.NVDA, "contract": "put", "expiry": "2026-12-18", "strike": 155, "qty": -6}]})
    env["use"](FakeBasket(ENTRY, ["NVDA", "BRK B"]))
    assert O.main([]) == 0
    got = orders_file(env)
    assert [o["id"] for o in got] == ["brkb-fw-put-2026-11-23", "brkb-fw-call-2026-11-23", hand["id"]]
    assert got[-1] == hand and output(env).endswith("stopped=NVDA\n")
    out = capsys.readouterr().out
    assert f"Kept {hand['id']} (for 2026-11-24, not this strategy's)" in out
    assert "Dropped nvda-csp-2026-11-16 (for 2026-11-16, not this strategy's)" in out
