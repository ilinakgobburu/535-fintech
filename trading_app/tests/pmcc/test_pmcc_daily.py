"""
Tests for the registered PMCC engine, on a market small enough to check by
hand: a flat $50 stock, options priced from one known volatility with a 10
cent quote around each price, no dividends and a zero rate unless a test says
otherwise. Every expected number below is arithmetic on those inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_app.lib import pmcc_daily as E
from trading_app.lib.ric import build_option_ric
from trading_app.lib.vol import bs_greeks, bs_price

VOL = 0.30
HALF = 0.05          # half the quoted spread
START, END = "2019-01-02", "2019-04-30"
LEAP_EXPIRY = E.third_friday(2020, 1)


def make_market(path: dict | None = None, opens: dict | None = None,
                dividends: dict | None = None, earnings=(), drop=()) -> dict:
    """`path` overrides closes from a date on; `drop` removes (expiry, strike) quotes."""
    days = pd.bdate_range("2018-12-03", "2019-05-31")
    days = days[~days.isin(pd.to_datetime(["2019-01-21", "2019-02-18", "2019-04-19"]))]
    close = pd.Series(50.0, index=days)
    for d, v in (path or {}).items():
        close[close.index >= pd.Timestamp(d)] = v
    stock = pd.DataFrame({"close": close, "open": close.shift(1).fillna(50.0)})
    for d, v in (opens or {}).items():
        stock.at[pd.Timestamp(d), "open"] = v

    rows = []
    monthly = E.monthly_expiries(days, "2019-01-01", "2019-05-31")
    listed = [(e, k) for e in monthly for k in (45.0, 50.0, 52.5, 55.0, 57.5, 60.0)]
    listed += [(LEAP_EXPIRY, k) for k in (35.0, 40.0, 45.0, 50.0)]
    for expiry, k in listed:
        if (expiry, k) in drop:
            continue
        ric = build_option_ric("DAL", expiry.date(), k, "C")
        for d in days[days <= expiry]:
            T = (expiry - d).days / 365.0
            fair = bs_price(close[d], k, T, VOL, 1.0, "C") if T > 0 else max(close[d] - k, 0.0)
            bid = round(fair - HALF, 4)
            rows.append({"date": d, "ric": ric, "strike": k, "expiry": expiry,
                         "bid": bid if bid > 0 else np.nan, "ask": round(fair + HALF, 4)})
    quotes = pd.DataFrame(rows)
    ok = quotes["bid"].notna()
    quotes["mid"] = np.where(ok, (quotes["bid"] + quotes["ask"]) / 2.0, np.nan)
    div = pd.Series(dividends or {}, dtype=float)
    div.index = pd.DatetimeIndex(pd.to_datetime(div.index))
    return {"quotes": quotes, "stock": stock, "tbill": pd.Series(0.0, index=days),
            "dividends": div, "earnings": pd.DatetimeIndex(pd.to_datetime(list(earnings)))}


def delta(spot, k, days):
    return bs_greeks(spot, k, days / 365.0, VOL, 1.0, "C")["delta"]


def price(spot, k, days):
    return bs_price(spot, k, days / 365.0, VOL, 1.0, "C")


ENTRY = pd.Timestamp("2019-01-22")      # session after the 18 Jan expiry; the 21st is a holiday
FEB = pd.Timestamp("2019-02-15")


class TestCalendar:
    def test_good_friday_expiry_moves_to_thursday(self):
        m = make_market()
        assert pd.Timestamp("2019-04-18") in E.monthly_expiries(m["stock"].index, START, END)

    def test_first_entry_is_the_session_after_the_first_expiry(self):
        res = E.run(make_market(), start=START, end=END)
        assert res["cycles"][0]["entry"] == ENTRY
        assert res["cycles"][0]["expiry"] == FEB
        assert res["cycles"][0]["dte"] == 24


class TestSelection:
    def test_leap_is_the_strike_whose_delta_is_nearest_080(self):
        days = (LEAP_EXPIRY - ENTRY).days
        want = min((35.0, 40.0, 45.0, 50.0), key=lambda k: abs(delta(50, k, days) - 0.80))
        res = E.run(make_market(), start=START, end=END)
        assert res["cycles"][0]["long_strike"] == want == 40.0

    def test_short_is_the_strike_whose_delta_is_nearest_025(self):
        want = min((50.0, 52.5, 55.0, 57.5, 60.0), key=lambda k: abs(delta(50, k, 24) - 0.25))
        res = E.run(make_market(), start=START, end=END)
        assert res["cycles"][0]["strike"] == want == 52.5

    def test_a_strike_with_no_bid_cannot_be_chosen(self):
        m = make_market(drop={(FEB, 52.5)})
        res = E.run(m, start=START, end=END)
        assert res["cycles"][0]["strike"] != 52.5

    def test_short_strike_is_never_below_the_leap_strike(self):
        m = make_market()
        day = m["quotes"][m["quotes"]["date"] == ENTRY]
        pick = E.select_short(day, FEB, 50.0, ENTRY, 0.0, 0.0, floor=55.0)
        assert pick["strike"] == 55.0

    def test_selection_reads_only_the_entry_day(self):
        """Changing every later price must not change what is chosen or paid."""
        a = E.run(make_market(), start=START, end=END)
        b = E.run(make_market(path={"2019-01-23": 70.0}), start=START, end=END)
        first = lambda r: [(e["instrument"], e["price"]) for e in r["blotter"] if e["date"] == ENTRY]
        assert first(a) == first(b)


class TestFills:
    def test_cross_buys_at_the_ask_and_sells_at_the_bid(self):
        res = E.run(make_market(), start=START, end=END, fill="cross")
        buy, sell = res["blotter"][0], res["blotter"][1]
        leap_days = (LEAP_EXPIRY - ENTRY).days
        assert buy["price"] == pytest.approx(price(50, 40, leap_days) + HALF, abs=1e-4)
        assert sell["price"] == pytest.approx(price(50, 52.5, 24) - HALF, abs=1e-4)

    def test_opening_capital_is_the_leap_less_the_first_premium(self):
        res = E.run(make_market(), start=START, end=END)
        leap_days = (LEAP_EXPIRY - ENTRY).days
        want = 100 * (price(50, 40, leap_days) + HALF) - 100 * (price(50, 52.5, 24) - HALF)
        assert E.opening_capital(res) == pytest.approx(want, abs=0.02)

    def test_mid_fill_is_cheaper_by_both_half_spreads(self):
        c = E.opening_capital(E.run(make_market(), start=START, end=END, fill="cross"))
        m = E.opening_capital(E.run(make_market(), start=START, end=END, fill="mid"))
        assert c - m == pytest.approx(100 * 2 * HALF, abs=0.02)


class TestExpiry:
    def test_out_of_the_money_expires_and_keeps_the_premium(self):
        res = E.run(make_market(), start=START, end=END)
        assert res["cycles"][0]["status"] == "expired"
        assert not res["assignments"]

    def test_close_equal_to_strike_expires(self):
        res = E.run(make_market(path={"2019-02-15": 52.5}), start=START, end=END)
        assert res["cycles"][0]["status"] == "expired"

    def test_assignment_sells_100_short_at_the_strike(self):
        res = E.run(make_market(path={"2019-02-15": 56.0}), start=START, end=END)
        rows = [e for e in res["blotter"] if e["date"] == FEB]
        assert [e["side"] for e in rows] == ["ASSIGN", "SELL"]
        assert rows[1]["cash_delta"] == pytest.approx(5250.0)

    def test_cover_is_the_next_open_plus_a_cent(self):
        m = make_market(path={"2019-02-15": 56.0}, opens={"2019-02-19": 57.0})
        res = E.run(m, start=START, end=END)
        cover = next(e for e in res["blotter"] if e["when"] == "open")
        assert cover["date"] == pd.Timestamp("2019-02-19")       # the 18th is a holiday
        assert cover["cash_delta"] == pytest.approx(-5701.0)
        a = res["assignments"][0]
        assert a["gap"] == pytest.approx(1.0)
        assert a["cover_cost"] == pytest.approx(100 * (57.01 - 52.5))

    def test_the_leap_survives_an_assignment(self):
        res = E.run(make_market(path={"2019-02-15": 56.0}), start=START, end=END)
        assert sum(e["leg"] == "long" for e in res["blotter"]) == 1


class TestLedger:
    def book(self, **kw):
        m = make_market(**kw)
        res = E.run(m, start=START, end=END)
        cap = E.opening_capital(res)
        return m, res, cap, E.build_ledger(res, m, cap)

    def test_cash_is_zero_after_the_opening_trade(self):
        _, _, _, led = self.book()
        assert led.iloc[0]["cash"] == pytest.approx(0.0, abs=1e-6)

    def test_nav_is_cash_plus_marks(self):
        _, _, _, led = self.book()
        r = led.iloc[5]
        assert r["nav"] == pytest.approx(r["cash"] + r["long_mv"] + r["short_mv"] + r["stock_mv"])
        assert r["short_mv"] < 0 < r["long_mv"]

    def test_available_funds_is_cash_while_no_stock_is_held(self):
        _, _, _, led = self.book()
        flat = led[led["shares"] == 0]
        assert np.allclose(flat["available_funds"], flat["cash"])

    def test_assignment_night_carries_the_short_stock_requirement(self):
        _, _, cap, led = self.book(path={"2019-02-15": 56.0})
        r = led[led["date"] == FEB].iloc[0]
        assert r["shares"] == -100
        cash = 0.0 + 5250.0
        assert r["cash"] == pytest.approx(cash)
        assert r["available_funds"] == pytest.approx(cash - 5600.0 - 0.5 * 5600.0)
        assert E.min_start_cash(led, cap)["min_cash"] == pytest.approx(cap + 3150.0)

    def test_an_assigned_call_is_gone_from_the_book(self):
        """It became short stock. Marking it as well would count the loss twice."""
        _, _, cap, led = self.book(path={"2019-02-15": 56.0}, opens={"2019-02-19": 56.0})
        r = led[led["date"] == FEB].iloc[0]
        assert r["short_mv"] == 0.0 and np.isnan(r["short_strike"])
        assert r["nav"] == pytest.approx(r["cash"] + r["long_mv"] - 5600.0)

    def test_an_expired_call_is_gone_from_the_book(self):
        _, _, _, led = self.book()
        r = led[led["date"] == FEB].iloc[0]
        assert r["short_mv"] == 0.0 and np.isnan(r["short_strike"])

    def test_pnl_by_leg_sums_to_the_change_in_nav(self):
        _, res, cap, led = self.book(path={"2019-02-15": 56.0})
        legs = E.pnl_by_leg(res, led)
        assert sum(legs.values()) == pytest.approx(led.iloc[-1]["nav"] - cap)

    def test_net_delta_is_long_but_under_one_share_equivalent(self):
        _, _, _, led = self.book()
        r = led.iloc[0]
        days = (LEAP_EXPIRY - ENTRY).days
        assert r["delta"] == pytest.approx(100 * (delta(50, 40, days) - delta(50, 52.5, 24)), abs=1.0)


class TestBenchmarks:
    def test_covered_call_writes_the_same_strike(self):
        m = make_market()
        a = E.run(m, start=START, end=END)
        b = E.run(m, start=START, end=END, long_leg="stock")
        assert [c["strike"] for c in a["cycles"]] == [c["strike"] for c in b["cycles"]]

    def test_covered_call_assignment_goes_flat_then_rebuys_at_the_open(self):
        m = make_market(path={"2019-02-15": 56.0}, opens={"2019-02-19": 57.0})
        res = E.run(m, start=START, end=END, long_leg="stock")
        led = E.build_ledger(res, m, E.opening_capital(res))
        assert led[led["date"] == FEB].iloc[0]["shares"] == 0
        assert led[led["date"] == pd.Timestamp("2019-02-19")].iloc[0]["shares"] == 100

    def test_buy_and_hold_collects_the_dividend_on_the_ex_date(self):
        m = make_market(dividends={"2019-03-01": 0.35})
        res = E.run(m, start=START, end=END, long_leg="stock", write_calls=False)
        led = E.build_ledger(res, m, E.opening_capital(res))
        assert led["dividend"].sum() == pytest.approx(35.0)
        assert led.iloc[-1]["nav"] == pytest.approx(5000.0 + 35.0)

    def test_the_leap_book_gets_no_dividend(self):
        m = make_market(dividends={"2019-03-01": 0.35})
        res = E.run(m, start=START, end=END)
        assert E.build_ledger(res, m, E.opening_capital(res))["dividend"].sum() == 0.0


class TestEarningsRule:
    def test_skip_rule_writes_no_call_over_an_earnings_date(self):
        m = make_market(earnings=["2019-02-07"])
        res = E.run(m, start=START, end=END, skip_earnings=True)
        assert res["cycles"][0]["status"] == "skipped: earnings in cycle"
        assert res["cycles"][1]["status"] == "expired"
        assert sum(e["leg"] == "long" for e in res["blotter"]) == 1    # the LEAP is still held

    def test_primary_rule_sells_through_it(self):
        m = make_market(earnings=["2019-02-07"])
        assert E.run(m, start=START, end=END)["cycles"][0]["status"] == "expired"


class TestPerformance:
    def test_monthly_returns_compound_to_the_total(self):
        m = make_market(path={"2019-03-01": 54.0})
        res = E.run(m, start=START, end=END)
        cap = E.opening_capital(res)
        perf = E.performance(E.build_ledger(res, m, cap), cap, m["tbill"])
        total = np.prod([1 + v for v in perf["monthly_returns"].values()]) - 1
        assert 100 * total == pytest.approx(perf["return_pct"])
        assert perf["max_drawdown_pct"] <= 0


class TestSpinOffRoot:
    def test_a_digit_after_the_root_is_read_as_the_same_underlying(self, tmp_path):
        import pickle
        idx = pd.to_datetime(["2022-10-24"])
        q = pd.DataFrame({"BID": [11.0], "ASK": [15.5]}, index=idx)
        (tmp_path / "options.pkl").write_bytes(pickle.dumps({"NVSA202308000.U^A23": q}))
        (tmp_path / "options_adjusted.pkl").write_bytes(pickle.dumps({"NVS1A192407000.U^A24": q}))
        pd.DataFrame({"TRDPRC_1": [80.0], "OPEN_PRC": [79.0]}, index=idx).to_pickle(
            tmp_path / "stock_unadjusted_ohlc.pkl")
        aux = {"tbill": pd.DataFrame({"MID_YLD_1": [4.0]}, index=idx),
               "dividends": pd.DataFrame({"i": ["NVS"], "ex": [pd.NaT], "amt": [None], "pay": [pd.NaT]}),
               "earnings": pd.DataFrame({"i": ["NVS"], "d": [pd.Timestamp("2022-10-25")]})}
        (tmp_path / "aux.pkl").write_bytes(pickle.dumps(aux))
        m = E.load_market(tmp_path)
        got = m["quotes"].set_index("ric")
        assert got.at["NVS1A192407000.U^A24", "strike"] == 70.0
        assert got.at["NVS1A192407000.U^A24", "expiry"] == pd.Timestamp("2024-01-19")
        assert len(m["dividends"]) == 0


class TestVersionTwoRules:
    """The LEAP is also replaced on delta, and the short strike respects break-even."""

    def test_a_leap_that_has_lost_its_delta_is_replaced(self):
        m = make_market(path={"2019-02-01": 38.0})          # the 40 LEAP is now out of the money
        res = E.run(m, start=START, end=END, roll_delta=0.50)
        rolls = [e for e in res["blotter"] if e["leg"] == "long" and e["date"] == pd.Timestamp("2019-02-19")]
        assert [e["side"] for e in rolls] == ["SELL", "BUY"]
        assert rolls[0]["strike"] == 40.0 and rolls[1]["strike"] == 35.0
        assert "delta" in rolls[0]["note"]

    def test_version_one_keeps_it(self):
        m = make_market(path={"2019-02-01": 38.0})
        res = E.run(m, start=START, end=END)
        assert sum(e["leg"] == "long" for e in res["blotter"]) == 1

    def test_a_healthy_leap_is_not_replaced(self):
        res = E.run(make_market(), start=START, end=END, roll_delta=0.50)
        assert sum(e["leg"] == "long" for e in res["blotter"]) == 1

    def test_short_strike_is_at_or_above_break_even(self):
        res = E.run(make_market(), start=START, end=END, breakeven_floor=True, fill="mid")
        c = res["cycles"][0]
        leap_cost = price(50, 40, (LEAP_EXPIRY - ENTRY).days)
        assert c["floor"] == pytest.approx(40.0 + leap_cost, abs=1e-3)
        # 40 + 11.74 = 51.74, so 50 is ruled out and 52.5, the 0.25-delta strike, stands
        assert 50.0 < c["floor"] <= c["strike"] == 52.5

    def test_no_strike_at_break_even_skips_the_cycle(self):
        m = make_market(path={"2019-02-01": 38.0})
        res = E.run(m, start=START, end=END, breakeven_floor=True, fill="mid")
        # break-even is above every listed strike with a bid once the stock is at 38
        assert res["cycles"][1]["status"] == "skipped: no qualifying strike"
