"""
Tests for the covered-call engine.

Two classes of bug are pinned here, and they are pinned differently.

The RIC and calendar tests pin failures that produce SILENCE -- a strike that
never resolves, a week that never happens. Those are the ones that cost you
cycles without an error message, which is the same failure mode HW1's put-wing
tests exist for.

The blotter/ledger/Reg T tests pin failures that produce a PLAUSIBLE WRONG
NUMBER: a NAV that drifts from cash, an assignment that credits the settle
instead of the strike, a margin figure computed off the wrong side of the
book. Nothing about the page looks broken when one of those is wrong, which is
why each has a case whose answer is known by construction rather than by
running the code and blessing the output.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from trading_app.lib.covered_call import (
    ASSIGN, BUY, EXPIRE, INITIAL_RATE, MAINT_RATE, SELL, SHARES_PER_CONTRACT,
    build_ledger, min_start_cash, nearest_otm, run_backtest, trading_weeks,
)
from trading_app.lib.ric import build_option_ric, parse_option_ric


# --------------------------------------------------------------------------
# fixtures: a two-week universe with prices chosen so every answer is known
# --------------------------------------------------------------------------

HOURS = list(range(13, 21))


def make_stock(day_prices: dict[dt.date, float]) -> pd.DataFrame:
    rows, idx = [], []
    for day, px in day_prices.items():
        for h in HOURS:
            idx.append(pd.Timestamp(dt.datetime.combine(day, dt.time(h))))
            rows.append({"TRDPRC_1": px, "OPEN_PRC": px, "HIGH_1": px, "LOW_1": px})
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx)).sort_index()
    df["date"] = [t.date() for t in df.index]
    return df


def make_options(expiry: dt.date, days: list[dt.date], strikes, quote=(0.90, 1.10),
                 root="AAPL") -> pd.DataFrame:
    rows = []
    for day in days:
        for h in HOURS:
            ts = pd.Timestamp(dt.datetime.combine(day, dt.time(h)))
            for k in strikes:
                bid, ask = quote
                rows.append({
                    "ts": ts, "ric": build_option_ric(root, expiry, float(k), "C"),
                    "strike": float(k), "expiry": expiry,
                    "bid": bid, "ask": ask, "mid": (bid + ask) / 2,
                    "spread": ask - bid, "trdprc_1": (bid + ask) / 2,
                    "num_moves": 10.0, "acvol_uns": 100.0, "date": day,
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 1. the RIC day rule -- the handout is wrong and the failure is silent
# --------------------------------------------------------------------------

class TestRicDayPadding:
    def test_single_digit_day_is_zero_padded(self):
        """
        The handout says "DAY not zero-padded" and prints AAPLH72620500.U^H26
        for the 7-Aug-2026 205 call. Probed against LSEG, that form returns
        nothing and the padded form returns 478 observations. Aug 7 and Sep 4
        are both single-digit Fridays in the backtest window, so an unpadded
        builder loses 2 of 10 cycles with no error at all.
        """
        got = build_option_ric("AAPL", dt.date(2026, 8, 7), 205.0, "C")
        assert got == "AAPLH072620500.U^H26"
        assert got != "AAPLH72620500.U^H26", "the handout's form does not resolve"

    def test_body_is_always_nine_digits(self):
        """DD + YY + SSSSS. A single-digit day must not shorten it to eight."""
        for day in (2, 7, 17, 28):
            ric = build_option_ric("AAPL", dt.date(2026, 9, day), 315.0, "C")
            body = ric.split(".")[0]
            digits = "".join(c for c in body if c.isdigit())
            assert len(digits) == 9, f"{ric} has {len(digits)} body digits"

    def test_round_trips(self):
        for day, k in ((2, 300.0), (7, 205.5), (17, 320.0), (31, 275.0)):
            e = dt.date(2026, 7, day) if day != 31 else dt.date(2026, 7, 31)
            ric = build_option_ric("AAPL", e, k, "C")
            back = parse_option_ric(ric)
            assert back is not None, f"{ric} did not parse"
            assert back["expiry"] == e and back["strike"] == pytest.approx(k)
            assert back["cp"] == "C"


# --------------------------------------------------------------------------
# 2. the calendar -- "Friday" is a description, not a rule
# --------------------------------------------------------------------------

class TestCalendar:
    def test_holiday_week_expires_thursday(self):
        """
        Jul 3 2026 is closed (Jul 4 falls on a Saturday). The week must expire
        on Thursday Jul 2, because that is the RIC that resolves against LSEG.
        A hardcoded W-FRI drops this cycle entirely.
        """
        days = {dt.date(2026, 6, 29): 300.0, dt.date(2026, 6, 30): 300.0,
                dt.date(2026, 7, 1): 300.0, dt.date(2026, 7, 2): 300.0}
        weeks = trading_weeks(make_stock(days))
        assert len(weeks) == 1
        assert weeks[0]["entry_date"] == dt.date(2026, 6, 29)
        assert weeks[0]["expiry_date"] == dt.date(2026, 7, 2)
        assert weeks[0]["short_week"] is True

    def test_monday_holiday_shifts_entry(self):
        """Labor Day: the entry is Tuesday, not a skipped week."""
        days = {dt.date(2026, 9, 8): 300.0, dt.date(2026, 9, 9): 300.0,
                dt.date(2026, 9, 10): 300.0, dt.date(2026, 9, 11): 300.0}
        weeks = trading_weeks(make_stock(days))
        assert weeks[0]["entry_date"].strftime("%a") == "Tue"
        assert weeks[0]["expiry_date"] == dt.date(2026, 9, 11)

    def test_single_session_week_is_not_a_cycle(self):
        days = {dt.date(2026, 7, 6): 300.0}
        assert trading_weeks(make_stock(days)) == []


# --------------------------------------------------------------------------
# 3. the strike rule
# --------------------------------------------------------------------------

class TestStrikeRule:
    CHAIN = pd.DataFrame({"strike": [295.0, 297.5, 300.0, 302.5, 305.0],
                          "mid": [8.0, 6.0, 4.0, 2.5, 1.5]})

    def test_spot_between_strikes_takes_the_next_one_up(self):
        assert nearest_otm(self.CHAIN, 301.2) == 302.5

    def test_spot_exactly_on_a_strike_takes_that_strike(self):
        """
        The ATM clause. 'Strictly above' silently skips to the next strike and
        sells an extra $2.50 of upside for no reason -- a plausible wrong
        number, not a crash.
        """
        assert nearest_otm(self.CHAIN, 300.0) == 300.0

    def test_spot_above_the_chain_returns_none(self):
        assert nearest_otm(self.CHAIN, 400.0) is None

    def test_never_selects_an_itm_strike(self):
        for spot in np.arange(294.0, 306.0, 0.37):
            k = nearest_otm(self.CHAIN, float(spot))
            if k is not None:
                assert k >= spot - 1e-9, f"picked ITM strike {k} at spot {spot}"


# --------------------------------------------------------------------------
# 4. the blotter and the ledger
# --------------------------------------------------------------------------

MON, TUE, WED, THU, FRI = (dt.date(2026, 7, 6), dt.date(2026, 7, 7),
                           dt.date(2026, 7, 8), dt.date(2026, 7, 9),
                           dt.date(2026, 7, 10))
STRIKES = [295.0, 297.5, 300.0, 302.5, 305.0]


def one_week(settle: float, spot: float = 300.0, quote=(0.90, 1.10),
             start_cash: float = 50_000.0):
    """
    One cycle with every number chosen so the answer is arithmetic.

    spot 300 sits exactly on a listed strike, so the ATM clause fires and the
    strike is 300. The quote is 0.90/1.10, so the mid is exactly 1.00 and the
    premium is exactly $100.
    """
    prices = {MON: spot, TUE: spot, WED: spot, THU: spot, FRI: settle}
    stock = make_stock(prices)
    opts = make_options(FRI, [MON, TUE, WED, THU, FRI], STRIKES, quote=quote)
    weeks = trading_weeks(stock)
    run = run_backtest(stock, opts, weeks, order_hour=15, start_cash=start_cash)
    return run, build_ledger(run, stock, opts), stock, opts


class TestBlotter:
    def test_entry_books_stock_then_call_at_the_same_bar(self):
        run, _, _, _ = one_week(settle=295.0)
        b = run["blotter"]
        assert [e["side"] for e in b] == [BUY, SELL, EXPIRE]
        assert b[0]["ts"] == b[1]["ts"], "both legs must fill at one moment"
        assert b[0]["qty"] == SHARES_PER_CONTRACT
        assert b[1]["qty"] == -1

    def test_premium_is_100_times_the_mid_not_the_bid_or_ask(self):
        run, _, _, _ = one_week(settle=295.0, quote=(0.90, 1.10))
        sell = [e for e in run["blotter"] if e["side"] == SELL][0]
        assert sell["fill"] == pytest.approx(1.00)
        assert sell["cash_delta"] == pytest.approx(100.0)

    def test_atm_clause_picks_the_strike_spot_sits_on(self):
        run, _, _, _ = one_week(settle=295.0, spot=300.0)
        assert run["cycles"][0]["strike"] == pytest.approx(300.0)

    def test_no_quote_skips_the_week_and_books_nothing(self):
        """
        'No bid/ask -> no fill.' The combo is one decision, so an unpriceable
        call means we do not buy the stock either. A version that bought the
        shares anyway would leave a naked long the write-up never claimed.
        """
        stock = make_stock({MON: 300.0, TUE: 300.0, WED: 300.0, THU: 300.0,
                            FRI: 295.0})
        opts = make_options(FRI, [MON, TUE, WED, THU, FRI], STRIKES)
        opts.loc[:, ["bid", "ask", "mid"]] = np.nan
        run = run_backtest(stock, opts, trading_weeks(stock), order_hour=15)
        assert run["blotter"] == []
        assert run["cycles"][0]["status"].startswith("skipped")

    def test_assignment_credits_the_strike_not_the_settle(self):
        """
        The classic plausible-wrong-number. Crediting the settle turns a capped
        strategy into an uncapped one and quietly deletes the entire point of
        the assignment.
        """
        run, _, _, _ = one_week(settle=310.0)
        a = [e for e in run["blotter"] if e["side"] == ASSIGN][0]
        assert a["cash_delta"] == pytest.approx(100 * 300.0)
        assert a["cash_delta"] != pytest.approx(100 * 310.0)

    def test_settle_equal_to_strike_expires(self):
        """An ATM call has no intrinsic value; the tie has to be decided."""
        run, _, _, _ = one_week(settle=300.0)
        assert [e["side"] for e in run["blotter"]][-1] == EXPIRE

    def test_a_penny_over_the_strike_assigns(self):
        run, _, _, _ = one_week(settle=300.01)
        assert [e["side"] for e in run["blotter"]][-1] == ASSIGN

    def test_no_rolls_and_no_buy_to_close(self):
        run, _, _, _ = one_week(settle=310.0)
        sides = [e["side"] for e in run["blotter"]]
        assert sides.count(SELL) == 1
        assert BUY not in [e["side"] for e in run["blotter"] if e["kind"] == "call"]


class TestLedger:
    def test_cash_moves_only_on_blotter_events(self):
        """
        The ledger is derived from the blotter, so at every bar its cash must
        equal start_cash plus the deltas of the events that have happened. Any
        drift means marks are leaking into cash.
        """
        run, led, _, _ = one_week(settle=295.0)
        for _, row in led.iterrows():
            due = run["start_cash"] + sum(
                e["cash_delta"] for e in run["blotter"] if e["ts"] <= row["ts"])
            assert row["cash"] == pytest.approx(due), f"cash drift at {row['ts']}"

    def test_expiry_worthless_keeps_shares_and_premium(self):
        run, led, _, _ = one_week(settle=295.0)
        last = led.iloc[-1]
        assert last["shares"] == SHARES_PER_CONTRACT
        # 50,000 - 30,000 stock + 100 premium
        assert last["cash"] == pytest.approx(20_100.0)
        assert last["nav"] == pytest.approx(20_100.0 + 100 * 295.0)

    def test_assignment_leaves_the_book_flat(self):
        run, led, _, _ = one_week(settle=310.0)
        last = led.iloc[-1]
        assert last["shares"] == 0
        assert last["option_mv"] == 0.0
        # bought at 300, delivered at 300, kept the 100 premium
        assert last["cash"] == pytest.approx(50_100.0)
        assert last["nav"] == pytest.approx(50_100.0)

    def test_upside_is_capped_at_the_strike(self):
        """The whole strategy in one assertion: a bigger rally pays no more."""
        _, led_a, _, _ = one_week(settle=310.0)
        _, led_b, _, _ = one_week(settle=360.0)
        assert led_a["nav"].iloc[-1] == pytest.approx(led_b["nav"].iloc[-1])

    def test_shares_are_never_negative_and_the_call_is_always_covered(self):
        run, led, _, _ = one_week(settle=310.0)
        assert (led["shares"] >= 0).all()
        open_call = led["call_ric"].notna()
        assert (led.loc[open_call, "shares"] == SHARES_PER_CONTRACT).all(), \
            "a short call must never be naked"


class TestRegT:
    def test_the_four_figures_are_the_stated_formulas(self):
        run, led, _, _ = one_week(settle=295.0)
        row = led[led["shares"] > 0].iloc[0]
        lmv = row["shares"] * row["stock_mark"]
        assert row["lmv"] == pytest.approx(lmv)
        assert row["initial_margin"] == pytest.approx(INITIAL_RATE * lmv)
        assert row["maintenance_margin"] == pytest.approx(MAINT_RATE * lmv)
        assert row["available_funds"] == pytest.approx(row["nav"] - row["initial_margin"])
        assert row["excess_liquidity"] == pytest.approx(row["nav"] - row["maintenance_margin"])

    def test_covered_short_call_adds_nothing_to_initial_margin(self):
        """Reg T, not portfolio margin: the covered call's requirement is $0."""
        run, led, _, _ = one_week(settle=295.0)
        row = led[led["call_ric"].notna()].iloc[0]
        assert row["initial_margin"] == pytest.approx(INITIAL_RATE * row["lmv"])

    def test_nav_is_cash_plus_stock_plus_a_negative_option(self):
        run, led, _, _ = one_week(settle=295.0)
        row = led[led["call_ric"].notna()].iloc[0]
        assert row["option_mv"] < 0, "a short call is a liability"
        assert row["nav"] == pytest.approx(row["cash"] + row["stock_mv"] + row["option_mv"])

    def test_flat_book_has_no_margin_requirement(self):
        run, led, _, _ = one_week(settle=310.0)
        last = led.iloc[-1]
        assert last["lmv"] == 0
        assert last["initial_margin"] == 0
        assert last["available_funds"] == pytest.approx(last["nav"])

    def test_thin_account_is_flagged_infeasible(self):
        """
        100 shares at $300 needs $15,000 of initial. An account holding
        $12,000 could not have put this trade on, and the ledger has to say so
        rather than quietly running a book that never existed.
        """
        _, led, _, _ = one_week(settle=295.0, start_cash=12_000.0)
        assert not led["feasible"].all()
        assert led["available_funds"].min() < 0

    def test_min_start_cash_finds_the_edge(self):
        run, _, stock, opts = one_week(settle=295.0, start_cash=12_000.0)
        got = min_start_cash(run, stock, opts, lo=0.0, hi=100_000.0, tol=1.0)
        assert got["binds"] is True
        led = build_ledger({**run, "start_cash": got["min_cash"]}, stock, opts)
        assert led["available_funds"].min() >= -1e-6
        led_less = build_ledger({**run, "start_cash": got["min_cash"] - 50.0},
                                stock, opts)
        assert led_less["available_funds"].min() < 0


# --------------------------------------------------------------------------
# 5. settlement reads the last trade, never the bar extremes
# --------------------------------------------------------------------------

class TestSettlementField:
    def test_a_spiking_high_does_not_cause_assignment(self):
        """
        On the real AAPL pull, HIGH_1 runs more than 1% above the bar's own
        open/close body on 12.8% of bars and LOW_1 more than 1% below on 21.2%,
        with excursions past 10% -- one hour that opened and closed near $301
        reports a high of $333. Those are odd-lot and out-of-sequence prints
        surviving into the extremes; TRDPRC_1 shows nothing of the kind.

        So a rule phrased as "did the stock touch the strike" would book
        assignments against trades that never happened. This pins the choice:
        a bar whose HIGH_1 is far through the strike but whose closing print
        is below it must EXPIRE.
        """
        stock = make_stock({MON: 300.0, TUE: 300.0, WED: 300.0, THU: 300.0,
                            FRI: 299.0})
        stock["HIGH_1"] = 340.0          # the bad print, well through any strike
        opts = make_options(FRI, [MON, TUE, WED, THU, FRI], STRIKES)
        run = run_backtest(stock, opts, trading_weeks(stock), order_hour=15)
        assert [e["side"] for e in run["blotter"]][-1] == EXPIRE
        assert run["cycles"][0]["settle"] == pytest.approx(299.0)

    def test_a_collapsing_low_does_not_change_the_entry_price(self):
        """The entry is the stock print, not the bar's low."""
        stock = make_stock({MON: 300.0, TUE: 300.0, WED: 300.0, THU: 300.0,
                            FRI: 295.0})
        stock["LOW_1"] = 255.0
        opts = make_options(FRI, [MON, TUE, WED, THU, FRI], STRIKES)
        run = run_backtest(stock, opts, trading_weeks(stock), order_hour=15)
        buy = [e for e in run["blotter"] if e["side"] == BUY][0]
        assert buy["fill"] == pytest.approx(300.0)


class TestOhlcIntegrity:
    def test_flags_extremes_outside_the_body(self):
        from trading_app.lib.cc_analysis import ohlc_integrity
        stock = make_stock({MON: 300.0, TUE: 300.0})
        stock["HIGH_1"] = 300.0
        stock["LOW_1"] = 300.0
        clean = ohlc_integrity(stock)
        assert clean["high"]["over_1pct"] == 0
        assert clean["low"]["over_1pct"] == 0

        stock.loc[stock.index[0], "HIGH_1"] = 340.0     # +13%
        dirty = ohlc_integrity(stock)
        assert dirty["high"]["over_1pct"] == 1
        assert dirty["high"]["max_pct"] == pytest.approx(100 * (340 / 300 - 1))


# --------------------------------------------------------------------------
# 6. the sweeps must differ only in the argument that was swept
# --------------------------------------------------------------------------

class TestSweeps:
    def test_hour_sweep_changes_the_premium_but_not_the_calendar(self):
        """
        Every hour runs the identical engine, so the number of cycles booked
        must not depend on the hour -- only what was paid for them. If the
        cycle count moved, the comparison would be between different books.
        """
        from trading_app.lib.cc_analysis import fill_hour_sweep
        stock = make_stock({MON: 300.0, TUE: 300.0, WED: 300.0, THU: 300.0,
                            FRI: 295.0})
        opts = make_options(FRI, [MON, TUE, WED, THU, FRI], STRIKES)
        rows = fill_hour_sweep(stock, opts, trading_weeks(stock))
        assert len({r["weeks_booked"] for r in rows}) == 1
        assert {r["order_hour"] for r in rows} == set(range(13, 20))

    def test_ols_r2_is_one_on_a_perfect_line(self):
        from trading_app.lib.cc_analysis import ols
        x = np.linspace(1, 10, 50)
        f = ols(x, 3.0 + 2.0 * x)
        assert f["r2"] == pytest.approx(1.0)
        assert f["slope"] == pytest.approx(2.0)
        assert f["intercept"] == pytest.approx(3.0)

    def test_ols_is_degenerate_safe(self):
        from trading_app.lib.cc_analysis import ols
        f = ols(np.array([1.0, 1.0, 1.0]), np.array([2.0, 3.0, 4.0]))
        assert np.isnan(f["r2"])


# --------------------------------------------------------------------------
# 7. the volatility-aware strike rule
# --------------------------------------------------------------------------

class TestImpliedVolRule:
    @staticmethod
    def chain_at_vol(spot: float, T: float, sigma: float,
                     strikes=np.arange(280.0, 361.0, 2.5)) -> pd.DataFrame:
        """A chain whose mids are Black-76 prices at a KNOWN sigma."""
        from trading_app.lib.vol import bs_price
        return pd.DataFrame({
            "strike": strikes,
            "mid": [bs_price(spot, float(k), T, sigma, 1.0, "C") for k in strikes],
        })

    def test_atm_iv_recovers_the_vol_it_was_priced_at(self):
        from trading_app.lib.covered_call import atm_iv
        T = 4 / 365.25
        got = atm_iv(self.chain_at_vol(315.0, T, 0.32), 315.0, T)
        assert got == pytest.approx(0.32, abs=1e-3)

    def test_higher_vol_sells_a_further_strike(self):
        """
        The entire motivation for the rule: a fixed-distance rule sells the
        same cap in a calm week and a violent one, and this must not.
        """
        from trading_app.lib.covered_call import by_assignment_prob
        T = 4 / 365.25
        calm = by_assignment_prob(self.chain_at_vol(315.0, T, 0.20), 315.0,
                                  target=0.25, T=T)
        wild = by_assignment_prob(self.chain_at_vol(315.0, T, 0.60), 315.0,
                                  target=0.25, T=T)
        assert wild > calm, f"vol 60% sold {wild}, vol 20% sold {calm}"

    def test_a_lower_breach_target_sells_a_further_strike(self):
        from trading_app.lib.covered_call import by_assignment_prob
        T = 4 / 365.25
        ch = self.chain_at_vol(315.0, T, 0.30)
        k25 = by_assignment_prob(ch, 315.0, target=0.25, T=T)
        k15 = by_assignment_prob(ch, 315.0, target=0.15, T=T)
        assert k15 > k25

    def test_the_strike_is_never_below_spot(self):
        from trading_app.lib.covered_call import by_assignment_prob
        T = 4 / 365.25
        for sigma in (0.15, 0.30, 0.75):
            for target in (0.10, 0.25, 0.45):
                k = by_assignment_prob(self.chain_at_vol(315.0, T, sigma), 315.0,
                                       target=target, T=T)
                assert k is None or k >= 315.0 - 1e-9

    def test_no_horizon_means_no_strike(self):
        """Without T there is no distribution, so the rule must decline."""
        from trading_app.lib.covered_call import by_assignment_prob
        ch = self.chain_at_vol(315.0, 4 / 365.25, 0.30)
        assert by_assignment_prob(ch, 315.0, target=0.25, T=None) is None
        assert by_assignment_prob(ch, 315.0, target=0.25, T=0.0) is None

    def test_unquotable_chain_means_no_strike(self):
        from trading_app.lib.covered_call import by_assignment_prob
        ch = pd.DataFrame({"strike": [300.0, 310.0], "mid": [np.nan, np.nan]})
        assert by_assignment_prob(ch, 305.0, target=0.25, T=4 / 365.25) is None

    def test_years_to_expiry_counts_down_to_the_close(self):
        from trading_app.lib.covered_call import SESSION_END_UTC, years_to_expiry
        exp = dt.date(2026, 7, 10)
        mon = pd.Timestamp(dt.datetime(2026, 7, 6, 15))
        thu = pd.Timestamp(dt.datetime(2026, 7, 9, 15))
        assert years_to_expiry(mon, exp) > years_to_expiry(thu, exp) > 0
        at_close = pd.Timestamp(dt.datetime.combine(exp, dt.time(SESSION_END_UTC)))
        assert years_to_expiry(at_close, exp) == 0.0
        assert years_to_expiry(at_close + pd.Timedelta(hours=5), exp) == 0.0

    def test_every_registered_rule_has_a_label(self):
        """A rule added in Python must not reach the page as a bare key."""
        from trading_app.lib.covered_call import STRIKE_RULE_META, STRIKE_RULES
        assert set(STRIKE_RULES) == set(STRIKE_RULE_META)
        for name, meta in STRIKE_RULE_META.items():
            assert meta["label"] and meta["label"] != name


# --------------------------------------------------------------------------
# 8. the hourly-vs-minute study
# --------------------------------------------------------------------------

class TestBarSizeStudy:
    @staticmethod
    def minute_panel(n_hours=2, per_hour=60):
        rows = []
        base = pd.Timestamp("2026-08-31 13:00:00")
        rng = np.random.default_rng(0)
        for hh in range(n_hours):
            for i in range(per_hour):
                ts = base + pd.Timedelta(hours=hh, minutes=i)
                bid = 3.00 + 0.01 * rng.integers(-5, 6)
                rows.append({"ts": ts, "ric": "AAPLI042632000.U^I26", "strike": 320.0,
                             "expiry": dt.date(2026, 9, 4), "bid": bid, "ask": bid + 0.10,
                             "mid": bid + 0.05, "spread": 0.10,
                             "trdprc_1": bid + 0.05, "num_moves": 5.0,
                             "acvol_uns": 50.0, "date": ts.date()})
        return pd.DataFrame(rows)

    def test_detects_that_the_hourly_quote_is_the_last_minute(self):
        """
        The check that licenses 'mid at the order bar'. If an hourly BID/ASK
        were a min-bid/max-ask envelope, every mid in the book would be the
        midpoint of an hour of quote range rather than a tradeable price.
        """
        from trading_app.lib.cc_analysis import bar_size_study
        m = self.minute_panel()
        m["hour"] = m["ts"].dt.floor("h")
        last = m.groupby("hour").last().reset_index()
        h = pd.DataFrame({
            "ts": last["hour"], "ric": last["ric"], "strike": last["strike"],
            "expiry": last["expiry"], "bid": last["bid"], "ask": last["ask"],
            "mid": last["mid"], "spread": last["spread"],
            "trdprc_1": last["trdprc_1"], "num_moves": last["num_moves"],
            "acvol_uns": last["acvol_uns"], "date": last["hour"].dt.date,
        })
        out = bar_size_study(h, m.drop(columns=["hour"]))
        assert out["snapshot"]["bid_is_last_pct"] == pytest.approx(100.0)
        assert out["snapshot"]["ask_is_last_pct"] == pytest.approx(100.0)

    def test_spots_an_envelope_instead_of_a_snapshot(self):
        """The same check must FAIL loudly if the hourly quote is aggregated."""
        from trading_app.lib.cc_analysis import bar_size_study
        m = self.minute_panel()
        m["hour"] = m["ts"].dt.floor("h")
        g = m.groupby("hour").agg(lo=("bid", "min"), hi=("ask", "max")).reset_index()
        h = pd.DataFrame({
            "ts": g["hour"], "ric": "AAPLI042632000.U^I26", "strike": 320.0,
            "expiry": dt.date(2026, 9, 4), "bid": g["lo"], "ask": g["hi"],
            "mid": (g["lo"] + g["hi"]) / 2, "spread": g["hi"] - g["lo"],
            "trdprc_1": (g["lo"] + g["hi"]) / 2, "num_moves": 5.0, "acvol_uns": 50.0,
            "date": g["hour"].dt.date,
        })
        out = bar_size_study(h, m.drop(columns=["hour"]))
        assert out["snapshot"]["bid_is_last_pct"] < 100.0
        assert out["snapshot"]["bid_is_min_pct"] == pytest.approx(100.0)

    def test_compares_only_cells_present_in_both_panels(self):
        """
        The comparison is only meaningful on IDENTICAL contracts and days.
        The minute pull used a narrower strike band than the hourly one, so an
        unmatched comparison would be reading a difference in which contracts
        were sampled as though it were a difference in bar size -- which is the
        exact mistake the study exists to avoid making.
        """
        from trading_app.lib.cc_analysis import bar_size_study
        m = self.minute_panel()
        extra = m.copy()
        extra["ric"] = "AAPLI042633000.U^I26"      # a contract minute never saw
        extra["strike"] = 330.0
        extra["spread"] = 2.00                     # and a wildly different quote
        extra["bid"] = 0.10
        extra["ask"] = 2.10
        extra["mid"] = 1.10
        h = pd.concat([m, extra], ignore_index=True)

        out = bar_size_study(h, m)
        assert out["hourly"]["contracts"] == 1, "the unmatched contract leaked in"
        assert out["hourly"]["median_spread_all"] == pytest.approx(0.10)
        assert out["matched_cells"] == 1

    def test_an_empty_hourly_panel_does_not_explode(self):
        """df[[]] is COLUMN selection, so an empty mask must still be boolean."""
        from trading_app.lib.cc_analysis import bar_size_study
        m = self.minute_panel()
        out = bar_size_study(m.iloc[:0].copy(), m)
        assert out["matched_cells"] == 0
        assert out["hourly"]["quoted_bars"] == 0
