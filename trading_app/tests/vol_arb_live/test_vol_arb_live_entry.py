"""
The entry night: the measurement and features (the backtest's own functions), the configured rung, the legs and
their sizes, and the two option orders. The chains are Black-76 prices from a known smile (va_live_synth), so the
legs and the contracts are known before the code runs: the strike whose |delta| at its own vol is nearest 0.25,
and round(200 / its vega per vol point per contract). The models' p must be what the registered walk-forward
would give the same row, fit on the same panel.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import va_live_synth as S
from trading_app.lib import vol_arb as VA
from trading_app.lib import vol_arb_live as VL
from trading_app.lib import vol_arb_wf as WF
from trading_app.lib import wheel_live as L
from trading_app.lib.vol import bs_greeks

FOR = dt.date(2026, 11, 24)
PUT_SKEW, CALL_SKEW = -0.06, 0.06


def plan(rung="rule_1", b=PUT_SKEW, panel_rule="tilt", constant=False, vix=18.0, book=None, mkt=None, history=None):
    cfg = S.config(rung=rung)
    panel = VL.load_panel(S.panel_obj(rule=panel_rule, chosen=rung, constant=constant), cfg)
    return VL.plan_night(cfg, S.ENTRY, FOR, book or VL.Book(), entry_expiry=S.EXPIRY,
                         mkt=mkt if mkt is not None else S.entry_market(b), leg_history=history,
                         vix_close=vix, panel=panel)


def expected_leg(b, cp):
    """The strike nearest 0.25 |delta| at the smile's own vol, and round(200 / vega) contracts, by hand."""
    mkt = S.entry_market(b)
    spot = float(mkt["stock"]["close"].iloc[-1])
    r = S.RATE_PCT / 100
    q = 0.02 / spot                                   # the two $0.01 dividends of the trailing year
    assert q == S.trailing_yield(mkt["dividends"], spot, S.ENTRY)
    T = (pd.Timestamp(S.EXPIRY) - pd.Timestamp(S.ENTRY)).days / 365.0
    F, Dc = spot * np.exp((r - q) * T), np.exp(-r * T)
    best = None
    for row in S.chain_rows(spot, b=b, q=q):
        if row["cp"] != cp:
            continue
        sigma = float(S.smile(row["strike"], F, T, b))
        g = bs_greeks(F, row["strike"], T, sigma, Dc, cp)
        miss = abs(abs(g["delta"] * F / spot) - 0.25)
        if best is None or miss < best[0] - 1e-6:
            best = (miss, row["strike"], g["vega"], (row["bid"] + row["ask"]) / 2)
    _, k, vega, mid = best
    return k, int(round(200.0 / vega)), mid


DIRECTIONS = [("rule_1", PUT_SKEW, "put_rich"), ("rule_1", CALL_SKEW, "call_rich"),
              ("logistic", PUT_SKEW, "put_rich"), ("logistic", CALL_SKEW, "call_rich"),
              ("forest", PUT_SKEW, "put_rich"), ("forest", CALL_SKEW, "call_rich")]


@pytest.mark.parametrize("rung, b, direction", DIRECTIONS)
def test_entry_orders_for_each_rung_and_direction(rung, b, direction):
    p = plan(rung, b)
    assert p["decision"]["direction"] == direction and p["decision"]["rung"] == rung
    put, call = p["orders"]
    sells, buys = (put, call) if direction == "put_rich" else (call, put)
    assert (sells["action"], buys["action"]) == ("SHT", "BUY")
    assert (put["contract"], call["contract"]) == ("put", "call")
    assert (put["id"], call["id"]) == ("nvda-va-put-2026-11-23", "nvda-va-call-2026-11-23")
    for o, cp in ((put, "P"), (call, "C")):
        k, n, mid = expected_leg(b, cp)
        assert o["strike"] == L.number_token(k) and o["qty"] == n and o["lmt_prc"] == L.number_token(mid)
        assert o["for"] == "2026-11-24" and o["expiry"] == "2026-12-18" and o["order_type"] == "lmt"
        assert o["underlying"] == S.NVDA and o["ref"] == "open"
        assert isinstance(o["strike"], str) and isinstance(o["lmt_prc"], str)     # tokens are strings
        assert type(o["qty"]) is int and type(o["underlying"]) is int
        assert set(o) <= L.ORDER_FIELDS
        L.validate_order(o)
    assert p["hedge"]["legs"] == [] and p["hedge"]["target"] == 0        # the new legs are hedged once held


def test_rule_1_is_put_rich_exactly_when_f1_is_above_f2():
    for b in (PUT_SKEW, CALL_SKEW, -0.01, 0.01):
        p = plan("rule_1", b)
        f = p["entry"]["features"]
        want = "put_rich" if f["f1_put_premium"] > f["f2_call_premium"] else "call_rich"
        assert p["decision"]["direction"] == want


@pytest.mark.parametrize("rung", ["logistic", "forest"])
@pytest.mark.parametrize("b", [PUT_SKEW, CALL_SKEW])
def test_the_live_p_is_the_walk_forwards_p_for_the_same_row(rung, b):
    """The registered walk-forward, given the panel as finished history and tonight's row to predict, gives the
    same p: the live fit is the backtest's fit, on the development rows only."""
    p = plan(rung, b)
    cfg = S.config(rung=rung)
    panel = VL.load_panel(S.panel_obj(chosen=rung), cfg)
    row = {"id": "NVDA|2026-11-23", "stock": "NVDA", "entry": pd.Timestamp(S.ENTRY), "expiry": pd.Timestamp(S.EXPIRY),
           **p["entry"]["features"], "label": np.nan, "predict": True}
    pool = pd.concat([panel, pd.DataFrame([row])], ignore_index=True)
    wf = WF.walk_forward(pool, [pd.Timestamp(S.ENTRY)], models=(rung,))
    assert wf["p"][rung]["NVDA|2026-11-23"] == p["decision"]["p"]
    assert p["decision"]["train_rows"] == len(wf["train"][pd.Timestamp(S.ENTRY)]) == 295


@pytest.mark.parametrize("rung", ["logistic", "forest"])
def test_no_trade_inside_the_band(rung):
    # nothing to learn (every row's features the same, labels alternating): p is about 0.5
    p = plan(rung, PUT_SKEW, panel_rule="alternating", constant=True)
    assert WF.P_LO <= p["decision"]["p"] <= WF.P_HI and p["decision"]["direction"] is None
    assert p["orders"] == [] and "inside the band" in p["notes"][0]


def test_the_band_edges_are_the_registered_ones():
    assert (WF.decide(0.5501), WF.decide(0.55), WF.decide(0.45), WF.decide(0.4499)) == \
        ("put_rich", None, None, "call_rich")


def test_sizing_is_200_dollars_a_vol_point_per_leg():
    p = plan("rule_1", PUT_SKEW)
    for cp in ("P", "C"):
        leg = p["entry"]["legs"][cp]
        vega_per_contract = leg["vega"] / 100.0 * 100        # per vol point, 100 shares
        assert leg["contracts"] == max(1, round(200.0 / vega_per_contract))
        assert abs(leg["contracts"] * vega_per_contract - 200.0) <= vega_per_contract / 2
    assert VA.contracts_for(14.344, 100) == 14 and VA.contracts_for(500.0, 100) == 1     # at least one


# --------------------------------------------------------------------------- nights with no trade
def drop_strikes(spot, keep_puts, keep_calls):
    rows = S.chain_rows(spot)
    return [r for r in rows if (r["cp"] == "P" and r["strike"] in keep_puts) or (r["cp"] == "C" and r["strike"] in keep_calls)]


def market_with(rows):
    c, o, rate, div = S.market_parts()
    return VL.market(c, o, rate, div, earnings=[], quotes=VL.quote_frame(rows))


def test_skip_rule_nights_give_no_orders():
    spot = float(S.closes().iloc[-1])
    full = S.chain_rows(spot)
    cases = {
        "no_forward": [r for r in full if r["cp"] == "P"],                                   # no calls at all
        # two puts below K0 = 177.5 (172.5, 175); the puts above K0 keep the ATM bracket
        "few_strikes": drop_strikes(spot, {172.5, 175.0, 177.5, 180.0, 182.5, 185.0}, set(np.arange(170.0, 252.51, 2.5))),
        "short_reach": [r for r in full if not (r["cp"] == "P" and r["strike"] < 165.0)],     # puts stop short
    }
    for reason, rows in cases.items():
        p = plan("rule_1", mkt=market_with(rows))
        assert p["entry"]["status"] == reason, reason
        assert p["orders"] == [] and p["decision"] is None
        assert f"no trade ({reason})" in p["notes"][0]


def test_the_short_reach_case_reaches_with_its_full_chain():
    # the same chain with every strike: measured, so the skip above is the missing strikes', not the smile's
    assert plan("rule_1")["entry"]["status"] == "measured"


def test_a_feature_that_is_not_finite_means_no_trade():
    p = plan("rule_1", vix=np.nan)                  # no VIX close: f8 is NaN
    assert p["entry"]["status"] == VA.NOT_FINITE and p["orders"] == []


def test_an_entry_needs_the_panel():
    cfg = S.config()
    with pytest.raises(VL.PanelError):
        VL.plan_night(cfg, S.ENTRY, FOR, VL.Book(), entry_expiry=S.EXPIRY, mkt=S.entry_market(), vix_close=18.0)


def test_the_entry_also_closes_a_leftover_hedge():
    p = plan("rule_1", PUT_SKEW, book=VL.Book(cash=0.0, shares=-300))
    assert [o["id"] for o in p["orders"]] == ["nvda-va-put-2026-11-23", "nvda-va-call-2026-11-23",
                                              "nvda-va-hedge-2026-11-23"]
    hedge = p["orders"][-1]
    assert (hedge["action"], hedge["qty"], hedge["order_type"], hedge["contract"]) == ("BTC", 300, "mkt", "underlying")
    assert "lmt_prc" not in hedge and "strike" not in hedge and "expiry" not in hedge


def test_a_held_opposite_position_in_the_same_contract_is_closed_then_opened():
    # the book holds +5 of the very put the put-rich trade sells: one order cannot flip it, so SEL 5 then SHT
    k, n, mid = expected_leg(PUT_SKEW, "P")
    book = VL.Book(cash=0.0, options=[{"cp": "P", "expiry": S.EXPIRY, "strike": k, "qty": 5}])
    mkt = S.entry_market(PUT_SKEW)
    hist = {("P", S.EXPIRY, k): S.leg_history(mkt["stock"]["close"].tail(3), "P", S.EXPIRY, k, 0.47)}
    p = plan("rule_1", PUT_SKEW, book=book, mkt=mkt, history=hist)
    puts = [o for o in p["orders"] if o["contract"] == "put"]
    assert [(o["id"], o["action"], o["qty"]) for o in puts] == [
        ("nvda-va-put-sel-2026-11-23", "SEL", 5), ("nvda-va-put-sht-2026-11-23", "SHT", n - 5)]
    ids = [o["id"] for o in p["orders"]]
    assert len(ids) == len(set(ids))


def test_measure_entry_uses_only_the_entry_days_quotes_of_its_expiry():
    spot = float(S.closes().iloc[-1])
    rows = S.chain_rows(spot)
    other_day = [{**r, "date": pd.Timestamp("2026-11-20"), "bid": r["bid"] * 3, "ask": r["ask"] * 3} for r in rows]
    other_exp = [{**r, "expiry": pd.Timestamp("2027-01-15"), "bid": r["bid"] * 3, "ask": r["ask"] * 3} for r in rows]
    a = VL.measure_entry(market_with(rows), S.ENTRY, S.EXPIRY, 18.0)
    b = VL.measure_entry(market_with(rows + other_day + other_exp), S.ENTRY, S.EXPIRY, 18.0)
    assert a["features"] == b["features"] and a["legs"][VL.PUT]["ric"] == b["legs"][VL.PUT]["ric"]


def test_the_features_come_from_the_backtests_functions():
    mkt = S.entry_market(PUT_SKEW)
    rec = VL.measure_entry(mkt, S.ENTRY, S.EXPIRY, 18.0)
    entry, expiry = pd.Timestamp(S.ENTRY), pd.Timestamp(S.EXPIRY)
    spot = float(mkt["stock"].at[entry, "close"])
    r, q = VA.W.carry(mkt, entry)
    m = VA.measure(VA.two_sided(mkt["quotes"]), spot, r, q, (expiry - entry).days / 365.0)
    real = VA.realized(mkt["stock"]["close"], mkt["dividends"]).loc[entry]
    assert rec["features"] == VA.features(m, real, 18.0, False)
    assert rec["features"]["f8_vix"] == 0.18 and rec["features"]["f7_earnings_in_cycle"] == 0.0
    # a report inside the cycle sets f7
    mkt2 = {**mkt, "earnings": pd.DatetimeIndex(["2026-12-01"])}
    assert VL.measure_entry(mkt2, S.ENTRY, S.EXPIRY, 18.0)["features"]["f7_earnings_in_cycle"] == 1.0


@pytest.mark.parametrize("rung", ["logistic", "forest"])
def test_the_fit_does_not_depend_on_the_panels_row_order(rung):
    """decide() fits in entry then stock order, as the walk-forward does, whatever order it is handed."""
    cfg = S.config(rung=rung)
    panel = VL.load_panel(S.panel_obj(chosen=rung), cfg)
    features = plan(rung, PUT_SKEW)["entry"]["features"]
    shuffled = panel.sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert VL.decide(rung, features, shuffled, S.ENTRY)["p"] == VL.decide(rung, features, panel, S.ENTRY)["p"]


def test_a_missing_leg_means_no_trade(monkeypatch):
    real = VA.legs_at_entry

    def no_call(*a, **k):
        legs = real(*a, **k)
        return {**legs, "C": None}

    monkeypatch.setattr(VL.VA, "legs_at_entry", no_call)
    p = plan("rule_1", PUT_SKEW)
    assert p["entry"]["no_trade"] == VA.LEG_MISSING and p["orders"] == [] and p["decision"] is None


def test_the_tbill_is_carried_forward_onto_sessions_without_a_print():
    c, o, rate, div = S.market_parts()
    sparse = pd.Series([5.0, 3.0], index=[c.index[-10], c.index[-3]])         # no print for the last two sessions
    mkt = VL.market(c, o, sparse, div)
    assert mkt["tbill"].iloc[-1] == 0.03 and mkt["tbill"].iloc[-5] == 0.05 and np.isnan(mkt["tbill"].iloc[0])
    assert VA.W.carry(mkt, c.index[-1])[0] == 0.03


def test_quote_frame_mids_are_load_markets():
    q = VL.quote_frame([
        {"date": "2026-11-23", "ric": "a", "cp": "P", "strike": 100, "expiry": "2026-12-18", "bid": 1.0, "ask": 1.2},
        {"date": "2026-11-23", "ric": "b", "cp": "P", "strike": 105, "expiry": "2026-12-18", "bid": np.nan, "ask": 1.2},
        {"date": "2026-11-23", "ric": "c", "cp": "P", "strike": 110, "expiry": "2026-12-18", "bid": 1.3, "ask": 1.2},
        {"date": "2026-11-23", "ric": "d", "cp": "P", "strike": 115, "expiry": "2026-12-18", "bid": 0.0, "ask": 0.4},
        {"date": "2026-11-23", "ric": "e", "cp": "P", "strike": 120, "expiry": "2026-12-18", "bid": 0.7, "ask": 0.7},
    ])
    assert q["mid"].iloc[4] == pytest.approx(0.7)                          # a locked quote is two-sided
    assert q["mid"].iloc[0] == pytest.approx(1.1)
    assert np.isnan(q["mid"].iloc[1]) and np.isnan(q["mid"].iloc[2])       # one-sided, crossed
    assert q["mid"].iloc[3] == pytest.approx(0.2)                          # a zero bid LSEG did send
    assert list(VA.two_sided(q)["ric"]) == ["a", "d", "e"]


def test_the_strike_plan_is_the_backtests_bands_and_grid():
    cfg = S.config()
    closes = pd.Series([170.0, 180.0, 175.0], index=pd.DatetimeIndex(["2026-11-13", "2026-11-18", "2026-11-23"]))
    plan_ = VL.strike_plan(closes, dt.date(2026, 11, 20), S.ENTRY, cfg["chain"]["bands"], cfg["chain"]["grid"])
    # mean 175 -> the $2.50 tier; puts floor(170 x 0.70 / 2.5) x 2.5 = 117.5 to ceil(180 x 1.02 / 2.5) x 2.5 = 185
    assert plan_["P"][0] == 117.5 and plan_["P"][-1] == 185.0 and np.allclose(np.diff(plan_["P"]), 2.5)
    # calls floor(170 x 0.97 / 2.5) x 2.5 = 162.5 to ceil(180 x 1.40 / 2.5) x 2.5 = 252.5
    assert plan_["C"][0] == 162.5 and plan_["C"][-1] == 252.5
    low = VL.strike_plan(closes / 2, dt.date(2026, 11, 20), S.ENTRY, cfg["chain"]["bands"], cfg["chain"]["grid"])
    assert np.allclose(np.diff(low["P"]), 1.0)                             # below $150: $1 strikes
    high = VL.strike_plan(closes * 2, dt.date(2026, 11, 20), S.ENTRY, cfg["chain"]["bands"], cfg["chain"]["grid"])
    assert np.allclose(np.diff(high["C"]), 10.0)                           # above $300: $10
    # closes after D are not known yet, and those before the 7-day window are not used
    later = pd.concat([closes, pd.Series([900.0], index=pd.DatetimeIndex(["2026-11-24"])),
                       pd.Series([10.0], index=pd.DatetimeIndex(["2026-11-12"]))])
    assert VL.strike_plan(later, dt.date(2026, 11, 20), S.ENTRY, cfg["chain"]["bands"], cfg["chain"]["grid"]) == plan_


def test_earnings_dates():
    exp = [pd.Timestamp("2027-02-25")]
    act = [pd.Timestamp("2026-11-19"), pd.Timestamp("2026-08-26")]
    dates, why = VL.earnings_dates(exp, act, S.ENTRY, S.EXPIRY, 80)
    assert pd.Timestamp("2027-02-25") in dates and "2027-02-25" in why
    # not rolled forward yet, but the last report was 4 days ago: none can fall before Dec 18
    dates, why = VL.earnings_dates([pd.Timestamp("2026-11-19")], act, S.ENTRY, S.EXPIRY, 80)
    assert not VA.earnings_between(dates, S.ENTRY, S.EXPIRY)
    # not rolled forward and the last report long ago: unknown, so it stops
    with pytest.raises(ValueError):
        VL.earnings_dates([], [pd.Timestamp("2026-08-26")], S.ENTRY, S.EXPIRY, 80)
    with pytest.raises(ValueError):
        VL.earnings_dates([pd.NaT], [], S.ENTRY, S.EXPIRY, 80)


def test_vix_file_parse():
    text = "DATE,OPEN,HIGH,LOW,CLOSE\n11/20/2026,19,20,18,18.5\n11/23/2026,18,19,17,17.25\n11/23/2026,18,19,17,17.5\n"
    s = VL.parse_vix(text)
    assert s.loc[pd.Timestamp("2026-11-23")] == 17.5 and list(s.index) == sorted(s.index)
