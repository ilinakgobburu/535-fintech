"""
lib/nvda_marks.py on made-up data (nm_synth): the nightly mark of an open cycle is the frozen backtest's own trade,
row for row. For every close before the expiry, mark_open_cycle's rows (P&L to date, the day's P&L, shares, hedge
trades, the split to date, each leg's mark and IV) equal VA.run_trade's on the full cycle up to that close, exactly,
for both sides at both fills; from the expiry on, they are run_trade's own rows with the settlement. The placeholder
session never reaches the output, and nothing dated after the mark changes it. The cycle is read from the config and
checked, never decided again; the notes say what the data did (carried marks and IVs, the report's timing).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import nm_synth as S
from trading_app.lib import nvda_marks as NM
from trading_app.lib import vol_arb as VA

GEN = "2026-11-17T03:30:00Z"


@pytest.fixture(scope="module")
def full():
    """The made-up market, the cycle, and VA.run_trade on the full cycle (to the expiry) for every side and fill."""
    mk, frames = S.market()
    spec = S.spec(mk)
    q = mk["quotes"][mk["quotes"]["expiry"] == S.EXPIRY]
    path = VA.cycle_path(mk, q, NM.legs_for_path(spec), S.ENTRY, S.EXPIRY)
    n = {cp: spec["legs"][cp]["contracts"] for cp in ("P", "C")}
    trades = {(d, f): VA.run_trade(path, VA.DIRECTIONS[d], n, NM.FILLS[f]) for d in NM.DIRECTIONS for f in NM.FILLS}
    return mk, spec, path, trades


def doc_of(mk, spec, as_of):
    return NM.document(NM.mark_open_cycle(mk, spec, as_of), spec, S.config(mk), generated_at=GEN)


# --------------------------------------------------------------------------- the frozen trade, row for row
def test_every_close_before_the_expiry_is_run_trades_own_rows(full):
    mk, spec, path, trades = full
    days = path["days"]
    assert days[0] == S.ENTRY and days[-1] == S.EXPIRY and len(days) == 25
    for k, as_of in enumerate(days[:-1]):
        mark = NM.mark_open_cycle(mk, spec, as_of)
        assert mark["status"] == "open" and mark["as_of"] == as_of and mark["settlement"] is None
        for cp in ("P", "C"):
            assert [r["mark"] for r in mark["legs"][cp]] == list(path["legs"][cp]["mark"][:k + 1])
            assert [r["iv"] for r in mark["legs"][cp]] == list(path["legs"][cp]["iv"][:k + 1])
        for (d, f), tr in trades.items():
            want = tr["ledger"].iloc[:k + 1]
            got = mark["sides"][d][f]
            rows = got["rows"]
            assert [r["date"] for r in rows] == list(days[:k + 1])
            assert [r["pnl_to_date"] for r in rows] == want["nav"].tolist()
            assert [r["day_pnl"] for r in rows] == want["pnl"].tolist()
            assert [r["shares"] for r in rows] == want["shares"].tolist()
            assert [r["hedge_trade"] for r in rows] == want["hedge_trade"].tolist()
            assert [r["cash"] for r in rows] == want["cash"].tolist()
            assert [r["option_mv"] for r in rows] == want["option_mv"].tolist()
            assert [r["dividend"] for r in rows] == want["dividend"].tolist()
            assert [r["close"] for r in rows] == want["close"].tolist()
            assert [r["open"] for r in rows] == list(path["open"][:k + 1])
            assert [r["put_mark"] for r in rows] == list(path["legs"]["P"]["mark"][:k + 1])
            assert [r["call_iv"] for r in rows] == list(path["legs"]["C"]["iv"][:k + 1])
            for s in VA.SPLIT:
                assert [r["split_to_date"][s] for r in rows] == want[s].cumsum().tolist()
                assert got["split"][s] == float(want[s].sum())
            assert got["pnl"] == float(want["nav"].iloc[-1])
            assert got["hedge_trades"] == int((want["hedge_trade"] != 0).sum())
            assert got["shares_held"] == int(want["shares"].iloc[-1])
            assert got["entry_shares"] == tr["entry_shares"]
            assert got["monthly"] == VA.monthly_pnl(want)
            assert got["hedge_costs"] == pytest.approx(-float(want["costs"].sum()) - tr["spread_cost"], abs=1e-9)


def test_the_stress_fill_costs_the_entry_spread_and_5bp_of_each_hedge_trade(full):
    mk, spec, path, _ = full
    as_of = pd.Timestamp("2026-11-10")
    mark = NM.mark_open_cycle(mk, spec, as_of)
    for d in NM.DIRECTIONS:
        mid, stress = mark["sides"][d]["mid"], mark["sides"][d]["stress"]
        legs = spec["legs"]
        spread = sum(legs[cp]["contracts"] * spec["lot"] * (legs[cp]["ask"] - legs[cp]["bid"]) / 2 for cp in "PC")
        assert stress["spread_cost"] == pytest.approx(spread, abs=1e-9)
        k = list(path["days"]).index(as_of)
        px = [path["close"][0]] + list(path["open"][1:k + 1])
        hedge = sum(5e-4 * abs(r["hedge_trade"]) * p for r, p in zip(stress["rows"], px))
        assert stress["hedge_costs"] == pytest.approx(hedge, rel=1e-12)
        assert mid["pnl"] - stress["pnl"] == pytest.approx(spread + hedge, abs=1e-6)
        assert mid["spread_cost"] == 0 and mid["hedge_costs"] == 0
    # the sides are mirror images at mid: each leg's sign flips, so do the hedge and the P&L
    assert mark["sides"]["put_rich"]["mid"]["pnl"] == pytest.approx(-mark["sides"]["call_rich"]["mid"]["pnl"], abs=1e-9)


@pytest.mark.parametrize("after", [0, 1, 4])
def test_on_and_after_the_expiry_the_cycle_is_settled_by_run_trade(full, after):
    mk, spec, path, trades = full
    sessions = mk["stock"].index[mk["stock"].index >= S.EXPIRY]
    as_of = sessions[after]
    mark = NM.mark_open_cycle(mk, spec, as_of)
    assert mark["status"] == "settled" and mark["as_of"] == S.EXPIRY
    for (d, f), tr in trades.items():
        got = mark["sides"][d][f]
        assert [r["pnl_to_date"] for r in got["rows"]] == tr["ledger"]["nav"].tolist()
        assert [r["shares"] for r in got["rows"]] == tr["ledger"]["shares"].tolist()
        assert got["pnl"] == tr["pnl"] and got["hedge_trades"] == tr["hedge_trades"]
        assert got["split"] == tr["split"] and got["shares_held"] == 0
        assert got["rows"][-1]["option_mv"] == 0.0
    K = {cp: spec["legs"][cp]["strike"] for cp in "PC"}
    S_T = float(mk["stock"].at[S.EXPIRY, "close"])
    assert mark["settlement"] == {"date": S.EXPIRY, "close": S_T,
                                  "intrinsic": {"P": max(K["P"] - S_T, 0.0), "C": max(S_T - K["C"], 0.0)}}
    doc = NM.document(mark, spec, S.config(mk), generated_at=GEN)
    assert doc["status"] == "settled" and doc["as_of"] == "2026-11-20" and S.schema_problems(doc) == []
    # the expiry's close settles at intrinsic: nothing about its quotes is noted
    assert not any("November 20" in n for n in NM.carried_notes(mark))


def test_after_the_expiry_nothing_later_is_read(full):
    mk, spec, _, _ = full
    later = NM.mark_open_cycle(mk, spec, "2026-11-27")
    m2 = dict(mk)
    st = mk["stock"].copy()
    st.loc[st.index > S.EXPIRY, ["close", "open"]] *= 3.0
    m2["stock"] = st
    assert S.dumps(NM.document(NM.mark_open_cycle(m2, spec, "2026-11-27"), spec, S.config(mk), generated_at=GEN)) \
        == S.dumps(NM.document(later, spec, S.config(mk), generated_at=GEN))


# --------------------------------------------------------------------------- the placeholder, and the future
@pytest.mark.parametrize("as_of", ["2026-10-19", "2026-10-29", "2026-11-05", "2026-11-16", "2026-11-19"])
def test_the_placeholder_never_reaches_the_output(full, as_of):
    mk, spec, _, _ = full
    doc = doc_of(mk, spec, as_of)
    assert doc["status"] == "open" and doc["settlement"] is None and doc["as_of"] == as_of
    text = json.dumps({k: v for k, v in doc.items() if k not in ("cycle", "events", "notes")})
    assert "2026-11-20" not in text
    for side in doc["marks"].values():
        for m in side.values():
            assert all(r["date"] <= as_of for r in m["daily"]) and m["daily"][-1]["date"] == as_of
            assert m["daily"][-1]["option_mv"] != 0.0
    assert S.schema_problems(doc) == []


def perturbed(mk: dict, after) -> dict:
    """Every input dated after `after` changed: closes and opens, quotes (moved, removed, added), the T-bill, and
    dividends going ex after it (one on the expiry)."""
    after = pd.Timestamp(after)
    m = dict(mk)
    st = mk["stock"].copy()
    st.loc[st.index > after, "close"] *= 1.37
    st.loc[st.index > after, "open"] *= 0.71
    m["stock"] = st
    q = mk["quotes"].copy()
    late = q["date"] > after
    q.loc[late, ["bid", "ask", "mid"]] *= 2.5
    q = q[~(late & (q["cp"] == "C"))]
    extra = q[q["date"] == q["date"].max()].assign(date=pd.Timestamp("2026-11-30"))
    m["quotes"] = pd.concat([q, extra], ignore_index=True)
    tb = mk["tbill"].copy()
    tb[tb.index > after] = 0.09
    m["tbill"] = tb
    m["dividends"] = pd.concat([mk["dividends"], pd.Series([5.0, 7.0], index=pd.DatetimeIndex(
        [after + pd.Timedelta(days=1), S.EXPIRY]))]).sort_index()
    return m


@pytest.mark.parametrize("as_of", ["2026-10-19", "2026-11-04", "2026-11-12", "2026-11-19"])
def test_nothing_dated_after_the_mark_changes_it(full, as_of):
    mk, spec, _, _ = full
    assert S.dumps(doc_of(perturbed(mk, as_of), spec, as_of)) == S.dumps(doc_of(mk, spec, as_of))


def test_a_market_that_ends_at_the_mark_gives_the_same_mark(full):
    mk, spec, _, _ = full
    short, _ = S.market(upto="2026-11-13")
    assert S.dumps(doc_of(short, spec, "2026-11-13")) == S.dumps(doc_of(mk, spec, "2026-11-13"))


def test_open_cycle_path_guards(full):
    mk, spec, _, _ = full
    q = mk["quotes"]
    legs = NM.legs_for_path(spec)
    with pytest.raises(ValueError):
        NM.open_cycle_path(mk, q, legs, S.ENTRY, S.EXPIRY, S.EXPIRY)             # not open
    with pytest.raises(ValueError):
        NM.open_cycle_path(mk, q, legs, S.ENTRY, S.EXPIRY, "2026-11-14")         # a Saturday
    with pytest.raises(ValueError):
        NM.mark_open_cycle(mk, spec, "2026-10-16")                              # before the entry
    with pytest.raises(ValueError):
        NM.mark_open_cycle(mk, spec, "2026-11-15")                              # a Sunday
    path = NM.open_cycle_path(mk, q, legs, S.ENTRY, S.EXPIRY, "2026-11-10")
    with pytest.raises(ValueError):
        NM.mark_to_market(path, VA.DIRECTIONS["put_rich"], {"P": 1, "C": 1}, "mid", "2026-11-09")


# --------------------------------------------------------------------------- the cycle, as decided at entry
def test_the_committed_config_is_the_entry_nights_cycle():
    import yaml
    spec = NM.cycle_spec(yaml.safe_load(S.CONFIG.read_text()))
    assert (spec["entry"], spec["expiry"], spec["lot"], spec["spot_at_entry"]) == (S.ENTRY, S.EXPIRY, 100, 217.09)
    assert {cp: (v["ric"], v["strike"], v["contracts"], v["mid"]) for cp, v in spec["legs"].items()} == {
        "P": ("NVDAW202620250.U", 202.5, 10, 4.375), "C": ("NVDAK202623500.U", 235.0, 10, 3.575)}
    d0, d1 = spec["decisions"]["f7_0"], spec["decisions"]["f7_1"]
    assert {m: v["direction"] for m, v in d0.items()} == {
        "logistic_refit": "call_rich", "logistic_fit_once": None, "forest_refit": None, "rule_1": "put_rich",
        "rule_0": "put_rich"}
    assert {m: v["direction"] for m, v in d1.items()} == {m: "put_rich" for m in NM.MODELS}
    assert d0["logistic_refit"]["p"] == 0.4499606129063129 and d1["logistic_fit_once"]["p"] == 0.559579272597368


def broken(path: str, value):
    cfg = S.config()
    node = cfg
    keys = path.split(".")
    for k in keys[:-1]:
        node = node[k]
    if value is KeyError:
        del node[keys[-1]]
    else:
        node[keys[-1]] = value
    return cfg


@pytest.mark.parametrize("path, value", [
    ("cycle.entry", "2026-10-20"),                                  # not the calendar's entry for Nov 20
    ("cycle.expiry", "2026-12-18"),
    ("cycle.lot", 0),
    ("cycle.spot_at_entry", float("nan")),
    ("cycle.legs.P.contracts", 0),
    ("cycle.legs.C.contracts", 2.5),
    ("cycle.legs.P.mid", 99.0),                                     # not the middle of its bid and ask
    ("cycle.legs.P.delta", 0.25),                                   # a put's delta is negative
    ("cycle.legs.C.iv", KeyError),
    ("cycle.legs.C.ric", "NVDAK182623500.U"),                       # another expiry's call
    ("cycle.legs.P.ric", "NVDAK202620250.U"),                       # a call's RIC for the put
    ("cycle.legs.P.ric", "NVDAW202620250.U^K26"),                   # the expired name
    ("cycle.legs.P.strike", 205.0),                                 # not the RIC's strike
    ("cycle.decisions.f7_0.logistic_refit.direction", "put_rich"),  # p 0.44996 is call_rich
    ("cycle.decisions.f7_1.forest_refit.direction", None),          # p 0.55018 is put_rich
    ("cycle.decisions.f7_0.logistic_fit_once.p", 1.5),
    ("cycle.decisions.f7_0.rule_1.p", 0.6),                         # a rule has no p
    ("cycle.decisions.f7_1.rule_0.direction", "call_rich"),
    ("cycle.decisions.f7_1.rule_1.direction", "both"),
    ("cycle.decisions.f7_0.forest_refit", KeyError),
])
def test_a_cycle_that_does_not_hang_together_is_refused(path, value):
    with pytest.raises(NM.SpecError):
        NM.cycle_spec(broken(path, value))


def test_nothing_is_decided_again():
    """The models' choices in the file are the config's, whatever the market does after the entry."""
    mk, _ = S.market()
    cfg = S.config(mk)
    spec = NM.cycle_spec(cfg)
    doc = NM.document(NM.mark_open_cycle(mk, spec, "2026-11-16"), spec, cfg, generated_at=GEN)
    for reading in NM.READINGS:
        for m in NM.MODELS:
            want = cfg["cycle"]["decisions"][reading][m]
            got = doc["cycle"]["decisions"][reading][m]
            assert got["direction"] == want["direction"]
            assert got["p"] == (None if want["p"] is None else round(want["p"], 5))
    assert {k: doc["cycle"]["legs"]["put"][k] for k in ("ric", "strike", "contracts", "mid", "bid", "ask")} == {
        k: cfg["cycle"]["legs"]["P"][k] for k in ("ric", "strike", "contracts", "mid", "bid", "ask")}


# --------------------------------------------------------------------------- what the data did
def test_carried_marks_and_ivs_are_the_paths_and_the_notes_say_so(full):
    mk, spec, path, _ = full
    mark = NM.mark_open_cycle(mk, spec, "2026-11-17")
    days = list(path["days"])
    for cp in ("P", "C"):
        rows = mark["legs"][cp]
        for i, r in enumerate(rows[1:], start=1):
            if not r["own_mid"]:
                assert r["mark"] == rows[i - 1]["mark"] and r["iv"] == rows[i - 1]["iv"]
            if not r["own_iv"]:
                assert r["iv"] == rows[i - 1]["iv"]
            if r["own_iv"]:
                assert r["iv_from"] == r["date"]
    call_gap = mark["legs"]["C"][days.index(pd.Timestamp("2026-10-29"))]
    assert not call_gap["own_mid"] and call_gap["mark_from"] == pd.Timestamp("2026-10-28")
    no_bid = mark["legs"]["P"][days.index(pd.Timestamp("2026-11-02"))]
    assert not no_bid["own_mid"] and no_bid["mid"] is None
    under = mark["legs"]["P"][days.index(S.UNDER)]
    assert under["own_mid"] and not under["own_iv"] and under["mid"] < under["intrinsic"]
    notes = NM.carried_notes(mark)
    assert any(n.startswith("The call had no two-sided closing quote on October 29, so its mark and IV there are "
                            "carried from October 28") for n in notes)
    assert any("The put had no two-sided closing quote on November 2" in n for n in notes)
    assert any("November 16" in n and "under its intrinsic value" in n and "no implied vol" in n for n in notes)
    # the entry close is the spec's own and never noted
    assert all("October 19" not in n for n in notes)


def test_the_real_cycles_note_on_november_16_reads_as_it_should():
    """The wording for the real case: one close, a put mid under intrinsic, the IV carried from Nov 13."""
    T = pd.Timestamp
    rows = [{"date": T("2026-10-19"), "own_mid": True, "own_iv": True, "mark_from": T("2026-10-19"),
             "iv_from": T("2026-10-19"), "mid": 4.375, "intrinsic": 0.0},
            {"date": T("2026-11-13"), "own_mid": True, "own_iv": True, "mark_from": T("2026-11-13"),
             "iv_from": T("2026-11-13"), "mid": 39.8, "intrinsic": 39.65},
            {"date": T("2026-11-16"), "own_mid": True, "own_iv": False, "mark_from": T("2026-11-16"),
             "iv_from": T("2026-11-13"), "mid": 45.975, "intrinsic": 46.36}]
    mark = {"expiry": S.EXPIRY, "legs": {"P": rows, "C": rows[:1]}}
    assert NM.carried_notes(mark) == [
        "November 16's closing mid for the put ($45.975) is under its intrinsic value ($46.36), so it has no implied "
        "vol; its IV is carried from November 13 (the registered rule) and its delta recomputed from it at that close."]


@pytest.mark.parametrize("as_of, want", [
    ("2026-11-16", "the November 17 mark is the last before them and the November 18 mark the first after"),
    ("2026-11-17", "This mark (November 17 close) is before NVIDIA's fiscal Q3 2027 results, reported November 17 "
                   "after the close; the November 18 mark is the first after them."),
    ("2026-11-18", "This mark (November 18 close) is the first after NVIDIA's fiscal Q3 2027 results, reported "
                   "November 17 after the close."),
    ("2026-11-19", "came on November 17 after the close: the November 18 mark was the first after them."),
])
def test_the_report_notes_say_which_mark_is_before_and_after(full, as_of, want):
    mk, spec, _, _ = full
    doc = doc_of(mk, spec, as_of)
    assert sum(want in n for n in doc["notes"]) == 1


def test_dated_notes_show_from_their_date(full):
    mk, spec, _, _ = full
    assert not any("151.005" in n for n in doc_of(mk, spec, "2026-11-11")["notes"])
    assert sum("151.005" in n for n in doc_of(mk, spec, "2026-11-12")["notes"]) == 1


# --------------------------------------------------------------------------- the night's data
def test_data_problems_and_the_last_complete_session(full):
    mk, spec, _, _ = full
    d = pd.Timestamp("2026-11-16")
    assert NM.data_problems(mk, spec, d) == []
    m = dict(mk)
    m["stock"] = mk["stock"].drop(d)
    assert NM.data_problems(m, spec, d) == ["LSEG has no NVDA close for 2026-11-16"]
    assert NM.last_complete_session(m, S.ENTRY, d) == pd.Timestamp("2026-11-13")
    st = mk["stock"].copy()
    st.loc[d, "open"] = np.nan
    m["stock"] = st
    assert NM.data_problems(m, spec, d) == ["LSEG has no NVDA open for 2026-11-16"]
    assert NM.last_complete_session(m, S.ENTRY, d) == pd.Timestamp("2026-11-13")
    st = mk["stock"].copy()
    st.loc[S.ENTRY, "open"] = np.nan                       # the entry's open is never used
    m["stock"] = st
    assert NM.data_problems(m, spec, S.ENTRY) == [] and NM.last_complete_session(m, S.ENTRY, S.ENTRY) == S.ENTRY
    q = mk["quotes"]
    m = dict(mk)
    m["quotes"] = q[~((q["ric"] == spec["legs"]["C"]["ric"]) & (q["date"] == d))]
    assert NM.data_problems(m, spec, d) == [f"LSEG has no 2026-11-16 quote for the call ({spec['legs']['C']['ric']}) yet"]
    # a row with no bid is final data (carried, with a note), and at the expiry the legs settle at intrinsic
    assert NM.data_problems(mk, spec, "2026-11-02") == []
    m["quotes"] = q[q["date"] != S.EXPIRY]
    assert NM.data_problems(m, spec, S.EXPIRY) == []
    assert NM.last_complete_session(mk, S.ENTRY, "2026-10-16") is None


def without_quotes(mk: dict, keep) -> dict:
    """The market with only the quote rows `keep(row ric, row date)` says to keep."""
    m = dict(mk)
    q = mk["quotes"]
    m["quotes"] = q[[keep(r, pd.Timestamp(d)) for r, d in zip(q["ric"], q["date"])]].reset_index(drop=True)
    return m


def test_at_the_expiry_a_leg_needs_its_row_on_the_session_before(full):
    mk, spec, _, _ = full
    put, call = spec["legs"]["P"]["ric"], spec["legs"]["C"]["ric"]
    nov19 = pd.Timestamp("2026-11-19")
    assert list(NM.quoted_closes(mk, spec, S.EXPIRY))[-1] == nov19
    assert list(NM.quoted_closes(mk, spec, "2026-11-23")) == list(NM.quoted_closes(mk, spec, S.EXPIRY))
    assert NM.quoted_closes(mk, spec, "2026-11-16")[-1] == pd.Timestamp("2026-11-16")
    assert NM.quoted_closes(mk, spec, S.ENTRY).tolist() == [S.ENTRY]
    assert NM.data_problems(mk, spec, S.EXPIRY) == [] and NM.history_problems(mk, spec, S.EXPIRY) == []
    m = without_quotes(mk, lambda r, d: not (r == put and d == nov19))
    assert NM.data_problems(m, spec, S.EXPIRY) == [
        f"LSEG has no 2026-11-19 quote for the put ({put}), the last close before the expiry"]
    assert NM.history_problems(m, spec, S.EXPIRY) == []                 # one close: the retry may carry it
    # the expiry's own rows are not read (the legs settle at intrinsic)
    assert NM.data_problems(without_quotes(mk, lambda r, d: d != S.EXPIRY), spec, S.EXPIRY) == []
    # the review's case: every leg quote before the expiry gone, which settled as if marked from the entry
    gone = without_quotes(mk, lambda r, d: d >= S.EXPIRY)
    assert NM.history_problems(gone, spec, S.EXPIRY) == [
        f"LSEG gave no quotes for the put ({put}) from 2026-10-19 to 2026-11-19",
        f"LSEG gave no quotes for the call ({call}) from 2026-10-19 to 2026-11-19"]
    assert len(NM.data_problems(gone, spec, S.EXPIRY)) == 2


@pytest.mark.parametrize("day", ["2026-11-16", "2026-11-20"])
def test_a_history_that_is_empty_starts_late_or_stops_early_is_never_marked(full, day):
    mk, spec, _, _ = full
    put, call = spec["legs"]["P"]["ric"], spec["legs"]["C"]["ric"]
    reads = NM.quoted_closes(mk, spec, day)
    last, before = reads[-1], reads[-2]
    assert NM.history_problems(mk, spec, day) == []
    assert NM.history_problems(without_quotes(mk, lambda r, d: r != call), spec, day) == [
        f"LSEG gave no quotes for the call ({call}) from 2026-10-19 to {last.date()}"]
    late = without_quotes(mk, lambda r, d: r != put or d > S.ENTRY)
    assert NM.history_problems(late, spec, day) == [
        f"LSEG has no quote for the put ({put}) on the entry close, 2026-10-19: its history starts 2026-10-20"]
    cut = without_quotes(mk, lambda r, d: r != put or d <= pd.Timestamp("2026-11-12"))
    assert NM.history_problems(cut, spec, day) == [
        f"LSEG's last quote for the put ({put}) is from 2026-11-12, before both {before.date()} and {last.date()}"]
    # the last close alone missing is not final data (data_problems), which the retry may carry; not a broken history
    one = without_quotes(mk, lambda r, d: not (r == call and d == last))
    assert NM.history_problems(one, spec, day) == [] and len(NM.data_problems(one, spec, day)) == 1
    two = without_quotes(mk, lambda r, d: not (r == call and d in (last, before)))
    assert len(NM.history_problems(two, spec, day)) == 1
    # a hole further back is LSEG's data as it is, carried with a note (the made-up call has none on Oct 29)
    assert pd.Timestamp("2026-10-29") not in set(mk["quotes"].loc[mk["quotes"]["ric"] == call, "date"])
    assert NM.history_problems(mk, spec, "2026-10-29") == [] and NM.history_problems(mk, spec, "2026-10-30") == []
    assert NM.history_problems(mk, spec, "2026-10-16") == []                 # before the entry: nothing read


def test_legs_now_is_each_legs_mark_and_iv_at_the_mark(full):
    mk, spec, path, _ = full
    days = list(path["days"])
    # (the mark, then each leg's mark_from and iv_from there), from the made-up quotes: the put's mid is under its
    # intrinsic value on Nov 5, 12, 13, 16 and 19 (no IV), it has no bid on Nov 2; the call has no row on Oct 29 and
    # no bid from Nov 2 on
    for as_of, froms in (("2026-11-05", ("2026-11-05", "2026-11-04", "2026-10-30", "2026-10-30")),
                         ("2026-11-19", ("2026-11-19", "2026-11-18", "2026-10-30", "2026-10-30")),
                         ("2026-11-16", ("2026-11-16", "2026-11-11", "2026-10-30", "2026-10-30")),
                         ("2026-11-02", ("2026-10-30", "2026-10-30", "2026-10-30", "2026-10-30")),
                         ("2026-10-29", ("2026-10-29", "2026-10-29", "2026-10-28", "2026-10-28")),
                         ("2026-11-10", ("2026-11-10", "2026-11-10", "2026-10-30", "2026-10-30"))):
        doc = doc_of(mk, spec, as_of)
        k = days.index(pd.Timestamp(as_of))
        spot = float(mk["stock"].at[pd.Timestamp(as_of), "close"])
        for cp, leg in (("P", "put"), ("C", "call")):
            now = doc["legs_now"][leg]
            assert now["mark"] == round(float(path["legs"][cp]["mark"][k]), 4)
            assert now["iv"] == round(float(path["legs"][cp]["iv"][k]), 6)
            assert now["iv_at_entry"] == round(spec["legs"][cp]["iv"], 6)
            K = spec["legs"][cp]["strike"]
            assert now["intrinsic"] == round(max(K - spot, 0.0) if cp == "P" else max(spot - K, 0.0), 4)
        now = doc["legs_now"]
        assert (now["put"]["mark_from"], now["put"]["iv_from"], now["call"]["mark_from"], now["call"]["iv_from"]) == froms
        # a carried IV is the one of the close it is carried from
        assert now["put"]["iv"] == round(float(path["legs"]["P"]["iv"][days.index(pd.Timestamp(froms[1]))]), 6)
    # the entry close: the spec's own mid and IV
    doc = doc_of(mk, spec, S.ENTRY)
    assert doc["legs_now"]["put"] == {"mark": spec["legs"]["P"]["mid"], "iv": round(spec["legs"]["P"]["iv"], 6),
                                      "iv_from": "2026-10-19", "mark_from": "2026-10-19",
                                      "intrinsic": round(max(spec["legs"]["P"]["strike"] - spec["spot_at_entry"], 0), 4),
                                      "iv_at_entry": round(spec["legs"]["P"]["iv"], 6)}


def test_a_revised_entry_close_or_quote_is_noted(full):
    mk, spec, _, _ = full
    assert NM.entry_differences(mk, spec) == []
    m = dict(mk)
    st = mk["stock"].copy()
    st.loc[S.ENTRY, "close"] += 0.5
    m["stock"] = st
    q = mk["quotes"].copy()
    hit = (q["ric"] == spec["legs"]["P"]["ric"]) & (q["date"] == S.ENTRY)
    q.loc[hit, "bid"] += 0.05
    m["quotes"] = q
    notes = NM.entry_differences(m, spec)
    assert len(notes) == 2
    assert notes[0].startswith("LSEG's October 19 NVDA close is now")
    assert notes[1].startswith("LSEG's October 19 quote for the put is now") and "which the mark keeps" in notes[1]


@pytest.mark.parametrize("side, change", [("ask", 0.05), ("ask", -0.01), ("bid", float("nan")), ("ask", float("nan"))])
def test_a_revision_of_either_side_of_an_entry_quote_is_noted(full, side, change):
    mk, spec, _, _ = full
    call = spec["legs"]["C"]
    q = mk["quotes"].copy()
    hit = (q["ric"] == call["ric"]) & (q["date"] == S.ENTRY)
    q.loc[hit, side] = q.loc[hit, side] + change
    m = dict(mk, quotes=q)
    b, a = (call["bid"] + (change if side == "bid" else 0.0)), (call["ask"] + (change if side == "ask" else 0.0))
    assert NM.entry_differences(m, spec) == [
        f"LSEG's October 19 quote for the call is now {NM._px(b)} / {NM._px(a)}; the cycle was entered at "
        f"{NM._px(call['bid'])} / {NM._px(call['ask'])} (mid {NM._px(call['mid'])}), which the mark keeps"]


# --------------------------------------------------------------------------- the file
def test_the_document_is_well_formed_rounded_and_repeatable(full):
    mk, spec, _, _ = full
    mark = NM.mark_open_cycle(mk, spec, "2026-11-16")
    doc = NM.document(mark, spec, S.config(mk), generated_at=GEN)
    assert S.schema_problems(doc) == []
    assert doc["generated_at"] == GEN and doc["as_of"] == "2026-11-16" and doc["status"] == "open"
    for d in NM.DIRECTIONS:
        for f in NM.FILLS:
            assert doc["marks"][d][f]["pnl"] == round(mark["sides"][d][f]["pnl"], 2)
    assert doc["vol"]["realized_since_entry"] == round(mark["vol"]["realized_since_entry"], 6)
    closes = mk["stock"]["close"].loc[S.ENTRY:"2026-11-16"]
    r = np.log(closes / closes.shift(1)).dropna()
    assert mark["vol"]["realized_since_entry"] == pytest.approx(float(np.sqrt(252 / len(r) * (r ** 2).sum())), rel=1e-15)
    again = NM.document(NM.mark_open_cycle(mk, spec, "2026-11-16"), spec, S.config(mk), generated_at=GEN)
    assert S.dumps(again) == S.dumps(doc)
    assert NM._r(-1e-9, 2) == 0.0 and str(NM._r(-1e-9, 2)) == "0.0" and NM._r(float("nan"), 2) is None
    assert doc["data_complete"] is True
    partial = NM.document(mark, spec, S.config(mk), generated_at=GEN, data_complete=False)
    assert partial["data_complete"] is False and S.schema_problems(partial) == []
    assert {k: v for k, v in partial.items() if k != "data_complete"} == \
        {k: v for k, v in doc.items() if k != "data_complete"}
    with pytest.raises(TypeError):
        NM.document(mark, spec, S.config(mk), generated_at=GEN, data_complete="yes")


def test_the_schema_check_itself_catches_what_it_should(full):
    mk, spec, _, _ = full
    doc = doc_of(mk, spec, "2026-11-16")
    bad = json.loads(json.dumps(doc))
    bad["marks"]["put_rich"]["mid"]["pnl"] += 1
    assert S.schema_problems(bad)
    bad = json.loads(json.dumps(doc))
    bad["settlement"] = {"date": "2026-11-20"}
    assert S.schema_problems(bad)
    bad = json.loads(json.dumps(doc))
    del bad["notes"]
    assert S.schema_problems(bad)
    bad = json.loads(json.dumps(doc))
    bad["legs_now"]["put"]["iv_from"] = "2026-11-17"                  # after the mark
    assert S.schema_problems(bad) == ["legs_now.put dates on or before as_of"]
    bad = json.loads(json.dumps(doc))
    bad["data_complete"] = 1
    assert S.schema_problems(bad) == ["data_complete is true or false"]
