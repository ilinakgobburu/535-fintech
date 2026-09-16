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
from pathlib import Path

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
        delivered = [e for e in run["blotter"]
                     if e["kind"] == "stock" and e["side"] == SELL][0]
        assert delivered["fill"] == pytest.approx(300.0)
        assert delivered["cash_delta"] == pytest.approx(100 * 300.0)
        assert delivered["cash_delta"] != pytest.approx(100 * 310.0)

    def test_assignment_is_two_rows_as_the_assignment_specifies(self):
        """
        "Friday ITM is ASSIGN on the call and a stock SELL at the strike."
        It used to be one ASSIGN row carrying the cash: arithmetically right,
        and still a blotter in which the shares never visibly left.
        """
        run, _, _, _ = one_week(settle=310.0)
        tail = run["blotter"][-2:]
        assert [(e["kind"], e["side"]) for e in tail] == [("call", ASSIGN),
                                                         ("stock", SELL)]
        assert tail[0]["ts"] == tail[1]["ts"], "both legs settle at one moment"
        assert tail[0]["cash_delta"] == 0.0, "the cash belongs to the stock leg"
        assert tail[1]["qty"] == SHARES_PER_CONTRACT

    def test_the_assign_row_points_at_the_stock_leg(self):
        """
        An ASSIGN row moving $0 was read by a grader as the assignment
        proceeds never arriving -- the next row, the stock sale at the strike,
        was just below the crop of his screenshot. The note now names the
        credit so the row cannot be read alone.
        """
        run, _, _, _ = one_week(settle=310.0)
        assign = [e for e in run["blotter"] if e["side"] == ASSIGN][0]
        assert "next row" in assign["note"]
        assert f"{100 * 300.0:,.2f}" in assign["note"]

    def test_assignment_moves_the_shares_exactly_once(self):
        """Decrementing on both rows would leave the book at -100 shares."""
        run, led, _, _ = one_week(settle=310.0)
        assert led["shares"].iloc[-1] == 0
        assert (led["shares"] >= 0).all()

    def test_option_rows_carry_an_occ_symbol(self):
        run, _, _, _ = one_week(settle=310.0)
        for e in run["blotter"]:
            if e["kind"] == "call":
                assert e["occ"] == "AAPL  260710C00300000"

    def test_settle_equal_to_strike_expires(self):
        """An ATM call has no intrinsic value; the tie has to be decided."""
        run, _, _, _ = one_week(settle=300.0)
        assert [e["side"] for e in run["blotter"]][-1] == EXPIRE

    def test_a_penny_over_the_strike_assigns(self):
        run, _, _, _ = one_week(settle=300.01)
        calls = [e["side"] for e in run["blotter"] if e["kind"] == "call"]
        assert calls[-1] == ASSIGN

    def test_no_rolls_and_no_buy_to_close(self):
        run, _, _, _ = one_week(settle=310.0)
        call_sides = [e["side"] for e in run["blotter"] if e["kind"] == "call"]
        assert call_sides.count(SELL) == 1, "one call written per cycle, never rolled"
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
        assert {r["order_hour"] for r in rows} == set(range(14, 21)), "as-of bars 14:00-20:00"

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
        base = pd.Timestamp("2026-08-31 13:01:00")   # as-of stamps: 13:01 .. 15:00
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
        m["hour"] = m["ts"].dt.ceil("h")   # as-of: (H-1h, H]
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
        m["hour"] = m["ts"].dt.ceil("h")   # as-of: (H-1h, H]
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


# --------------------------------------------------------------------------
# 9. the loaders
# --------------------------------------------------------------------------
# HW1's lesson was that loader bugs produce ZERO ROWS AND NO ERROR MESSAGE,
# which is the worst outcome because an empty panel looks like a quiet market.
# Every case below has a known answer by construction.

OPT_FIELDS = ["BID", "ASK", "TRDPRC_1", "OPEN_PRC", "HIGH_1", "LOW_1",
              "ACVOL_UNS", "NUM_MOVES"]
CALL_RIC = "AAPLI042632000.U^I26"      # AAPL 4-Sep-26 320 CALL
PUT_RIC = "AAPLU042632000.U^I26"       # the same contract's PUT


def opt_frame(ric=CALL_RIC, bid=(1.0, 1.1), ask=(1.2, 1.3),
              hours=("15:00", "16:00"), trd=None):
    idx = pd.DatetimeIndex([f"2026-08-31 {h}" for h in hours])
    cols = pd.MultiIndex.from_product([[ric], OPT_FIELDS], names=["RIC", "Field"])
    df = pd.DataFrame(np.nan, index=idx, columns=cols)
    df.loc[:, (ric, "BID")] = list(bid)
    df.loc[:, (ric, "ASK")] = list(ask)
    if trd is not None:
        df.loc[:, (ric, "TRDPRC_1")] = list(trd)
    return df


class TestStockPanel:
    def _raw(self, cols, rows, hours=("15:00", "16:00")):
        from trading_app.lib.covered_call import stock_panel
        idx = pd.DatetimeIndex([f"2026-08-31 {h}" for h in hours])
        return stock_panel({"stock": pd.DataFrame(rows, index=idx, columns=cols)})

    def test_relabels_to_bar_end_and_drops_premarket_and_after_hours(self):
        """
        LSEG stamps a bar with its START. Raw 12:00 is premarket (ends 13:00,
        before the 13:30 open); raw 13:00 holds the open and becomes 14:00;
        raw 19:00 is the closing hour and becomes 20:00; raw 20:00 is 4-5pm
        ET after-hours trading and must not survive at all.
        """
        from trading_app.lib.covered_call import stock_panel
        hours = [8, 12, 13, 15, 19, 20, 21, 23]
        idx = pd.DatetimeIndex([f"2026-08-31 {h:02d}:00" for h in hours])
        raw = pd.DataFrame({"TRDPRC_1": [float(h) for h in hours]}, index=idx)
        got = stock_panel({"stock": raw})
        assert list(got.index.hour) == [14, 16, 20]
        # the price moves with its bar: raw 19:00's print is now stamped 20:00
        assert list(got["TRDPRC_1"]) == [13.0, 15.0, 19.0]

    def test_resolves_a_ric_field_multiindex(self):
        got = self._raw(pd.MultiIndex.from_product([["AAPL.O"], ["TRDPRC_1", "HIGH_1"]]),
                        [[300.0, 301.0], [302.0, 303.0]])
        assert set(got.columns) >= {"TRDPRC_1", "HIGH_1"}
        assert got["TRDPRC_1"].iloc[0] == pytest.approx(300.0)

    def test_resolves_a_field_ric_multiindex_too(self):
        """
        Taking the LAST level assumes (RIC, Field). Handed (Field, RIC) it used
        to name every column after the RIC, producing duplicates and then
        dying inside pd.to_numeric with a TypeError about its argument -- a
        failure a long way from its cause. Membership decides, not position.
        """
        got = self._raw(pd.MultiIndex.from_product([["TRDPRC_1", "HIGH_1"], ["AAPL.O"]]),
                        [[300.0, 301.0], [302.0, 303.0]])
        assert set(got.columns) >= {"TRDPRC_1", "HIGH_1"}
        assert got["TRDPRC_1"].iloc[0] == pytest.approx(300.0)
        assert got["HIGH_1"].iloc[0] == pytest.approx(301.0)

    def test_refuses_a_multiindex_it_cannot_resolve(self):
        with pytest.raises(ValueError, match="recognisable field"):
            self._raw(pd.MultiIndex.from_product([["a", "b"], ["c", "d"]]),
                      [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])

    def test_refuses_duplicate_columns_rather_than_silently_picking_one(self):
        with pytest.raises(ValueError, match="duplicate stock columns"):
            self._raw(["TRDPRC_1", "TRDPRC_1"], [[300.0, 301.0], [302.0, 303.0]])

    def test_deduplicates_repeated_timestamps_keeping_the_last(self):
        """A frame stitched from weekly chunks repeats a bar at every seam."""
        got = self._raw(["TRDPRC_1"], [[300.0], [999.0]], hours=("15:00", "15:00"))
        assert len(got) == 1
        assert got["TRDPRC_1"].iloc[0] == pytest.approx(999.0)

    def test_coerces_text_to_numbers(self):
        got = self._raw(["TRDPRC_1"], [["300.5"], ["n/a"]])
        assert got["TRDPRC_1"].iloc[0] == pytest.approx(300.5)
        assert np.isnan(got["TRDPRC_1"].iloc[1])

    def test_adds_a_date_column_the_calendar_depends_on(self):
        got = self._raw(["TRDPRC_1"], [[300.0], [301.0]])
        assert list(got["date"]) == [dt.date(2026, 8, 31)] * 2


class TestOptionPanel:
    def test_mid_is_the_quote_midpoint(self):
        from trading_app.lib.covered_call import option_panel
        p = option_panel({"options": opt_frame(bid=(1.0, 1.1), ask=(1.2, 1.3))})
        assert list(p["mid"]) == pytest.approx([1.1, 1.2])
        assert list(p["spread"]) == pytest.approx([0.2, 0.2])

    def test_a_one_sided_quote_has_no_mid(self):
        """'No bid/ask -> no fill', expressed once in the data."""
        from trading_app.lib.covered_call import option_panel
        f = opt_frame()
        f.loc[f.index[0], (CALL_RIC, "ASK")] = np.nan
        p = option_panel({"options": f}).sort_values("ts")
        assert np.isnan(p["mid"].iloc[0])
        assert not np.isnan(p["mid"].iloc[1])

    def test_a_crossed_quote_has_no_mid(self):
        """ask < bid is bad data, and its midpoint is not a price."""
        from trading_app.lib.covered_call import option_panel
        p = option_panel({"options": opt_frame(bid=(1.0, 1.1), ask=(0.5, 1.3))})
        p = p.sort_values("ts")
        assert np.isnan(p["mid"].iloc[0])
        assert p["mid"].iloc[1] == pytest.approx(1.2)

    def test_puts_are_dropped_because_this_book_is_calls_only(self):
        from trading_app.lib.covered_call import option_panel
        assert len(option_panel({"options": opt_frame(ric=PUT_RIC)})) == 0

    def test_unparseable_rics_are_skipped_not_fatal(self):
        from trading_app.lib.covered_call import option_panel
        good = opt_frame()
        bad = opt_frame(ric="NOT-A-RIC")
        p = option_panel({"options": pd.concat([good, bad], axis=1)})
        assert set(p["ric"]) == {CALL_RIC}

    def test_strike_and_expiry_come_off_the_ric(self):
        from trading_app.lib.covered_call import option_panel
        p = option_panel({"options": opt_frame()})
        assert p["strike"].iloc[0] == pytest.approx(320.0)
        assert p["expiry"].iloc[0] == dt.date(2026, 9, 4)

    def test_an_all_nan_field_does_not_delete_the_contract(self):
        """TRDPRC_1 is absent whenever nothing traded, which is most bars."""
        from trading_app.lib.covered_call import option_panel
        p = option_panel({"options": opt_frame()})
        assert len(p) == 2
        assert p["trdprc_1"].isna().all()

    def test_keeps_only_session_hours(self):
        from trading_app.lib.covered_call import option_panel
        p = option_panel({"options": opt_frame(bid=(1.0, 1.1), ask=(1.2, 1.3),
                                               hours=("09:00", "15:00"))})
        assert list(p["ts"].dt.hour) == [16], "raw 15:00 is the bar ending 16:00"

    def test_requires_a_multiindex_rather_than_guessing(self):
        from trading_app.lib.covered_call import option_panel
        flat = pd.DataFrame({CALL_RIC: [1.0]},
                            index=pd.DatetimeIndex(["2026-08-31 15:00"]))
        with pytest.raises(ValueError, match="MultiIndex"):
            option_panel({"options": flat})


class TestMidVsPrint:
    def test_survives_duplicate_stock_timestamps(self):
        """
        reindex refuses to work against duplicate labels. stock_panel already
        de-duplicates so the build never hit this, but a public function should
        not crash on a raw frame with a ValueError naming neither cause nor
        caller.
        """
        from trading_app.lib.cc_analysis import mid_vs_print
        from trading_app.lib.covered_call import option_panel
        # as-of stamps, to line up with option_panel's relabelled bars
        idx = pd.DatetimeIndex(["2026-08-31 16:00", "2026-08-31 16:00",
                                "2026-08-31 17:00"])
        stock = pd.DataFrame({"TRDPRC_1": [300.0, 301.0, 302.0]}, index=idx)
        opts = option_panel({"options": opt_frame(trd=(1.15, 1.25))})
        out = mid_vs_print(opts, stock)
        assert out["resid"]["n"] == 2

    def test_needs_both_a_quote_and_a_print(self):
        from trading_app.lib.cc_analysis import mid_vs_print
        from trading_app.lib.covered_call import option_panel
        stock = pd.DataFrame({"TRDPRC_1": [300.0, 300.0]},
                             index=pd.DatetimeIndex(["2026-08-31 16:00",
                                                     "2026-08-31 17:00"]))
        opts = option_panel({"options": opt_frame()})       # never printed
        assert mid_vs_print(opts, stock)["resid"] == {}

    def test_in_spread_locates_the_print_inside_the_quote(self):
        from trading_app.lib.cc_analysis import mid_vs_print
        from trading_app.lib.covered_call import option_panel
        stock = pd.DataFrame({"TRDPRC_1": [300.0, 300.0]},
                             index=pd.DatetimeIndex(["2026-08-31 16:00",
                                                     "2026-08-31 17:00"]))
        # bid 1.00/ask 1.20: one print at the bid, one at the ask
        opts = option_panel({"options": opt_frame(bid=(1.0, 1.0), ask=(1.2, 1.2),
                                                  trd=(1.0, 1.2))})
        r = mid_vs_print(opts, stock)["resid"]
        assert r["at_bid_pct"] == pytest.approx(50.0)
        assert r["at_ask_pct"] == pytest.approx(50.0)
        assert r["at_mid_pct"] == pytest.approx(0.0)
        assert r["outside_pct"] == pytest.approx(0.0)


class TestOtherStrikeRules:
    CHAIN = pd.DataFrame({"strike": [295.0, 297.5, 300.0, 302.5, 305.0, 310.0],
                          "mid": [8.0, 6.0, 4.0, 2.5, 1.5, 0.30]})

    def test_offset_rule_clears_the_offset(self):
        from trading_app.lib.covered_call import otm_by_offset
        assert otm_by_offset(self.CHAIN, 300.0, offset_pct=0.01) == 305.0
        assert otm_by_offset(self.CHAIN, 300.0, offset_pct=0.0) == 300.0

    def test_premium_floor_takes_the_furthest_strike_still_paid_for(self):
        from trading_app.lib.covered_call import by_premium_floor
        assert by_premium_floor(self.CHAIN, 300.0, floor=0.50) == 305.0
        assert by_premium_floor(self.CHAIN, 300.0, floor=2.00) == 302.5

    def test_premium_floor_never_goes_below_spot(self):
        from trading_app.lib.covered_call import by_premium_floor
        assert by_premium_floor(self.CHAIN, 300.0, floor=0.01) >= 300.0

    def test_premium_floor_declines_when_nothing_pays_enough(self):
        from trading_app.lib.covered_call import by_premium_floor
        assert by_premium_floor(self.CHAIN, 300.0, floor=99.0) is None


class TestSweepsAndBenchmark:
    def test_rule_sweep_labels_and_calibrates_every_rule(self):
        from trading_app.lib.cc_analysis import rule_sweep
        from trading_app.lib.covered_call import STRIKE_RULE_META
        stock = make_stock({MON: 300.0, TUE: 300.0, WED: 300.0, THU: 300.0,
                            FRI: 295.0})
        opts = make_options(FRI, [MON, TUE, WED, THU, FRI], STRIKES)
        rows = rule_sweep(stock, opts, trading_weeks(stock), order_hour=15)
        assert {r["rule"] for r in rows} == set(STRIKE_RULE_META)
        for r in rows:
            assert r["label"] and r["label"] != r["rule"]
            if r["target_prob"] is None:
                assert r["realised_prob"] is None
            else:
                assert r["prob_gap"] == pytest.approx(
                    r["realised_prob"] - r["target_prob"])

    def test_buy_and_hold_starts_where_the_book_started(self):
        from trading_app.lib.cc_analysis import buy_and_hold
        run, led, stock, opts = one_week(settle=295.0)
        bh = buy_and_hold(stock, led, run["start_cash"])
        first = led.index[led["shares"] > 0][0]
        px0 = float(led.at[first, "stock_mark"])
        # same entry price, no premium, no cap
        assert bh["nav"].iloc[0] == pytest.approx(run["start_cash"])
        assert bh["nav"].iloc[-1] == pytest.approx(
            run["start_cash"] - 100 * px0 + 100 * 295.0)

    def test_buy_and_hold_is_uncapped_where_the_book_is_not(self):
        from trading_app.lib.cc_analysis import buy_and_hold
        lo, hi = [], []
        for settle in (310.0, 360.0):
            run, led, stock, opts = one_week(settle=settle)
            lo.append(led["nav"].iloc[-1])
            hi.append(buy_and_hold(stock, led, run["start_cash"])["nav"].iloc[-1])
        assert lo[0] == pytest.approx(lo[1]), "the covered call must be capped"
        assert hi[1] > hi[0], "buy and hold must not be"


# --------------------------------------------------------------------------
# 10. picking the order bar, and refusing to guess when the cache is absent
# --------------------------------------------------------------------------

class TestOrderBarSelection:
    @staticmethod
    def day_frame(hours):
        idx = pd.DatetimeIndex([f"2026-08-31 {h:02d}:00" for h in hours])
        df = pd.DataFrame({"TRDPRC_1": range(len(hours))}, index=idx)
        df["date"] = [t.date() for t in df.index]
        return df

    def test_takes_the_bar_stamped_at_the_requested_hour(self):
        from trading_app.lib.covered_call import _bar_at
        f = self.day_frame([13, 14, 15, 16])
        assert _bar_at(f, dt.date(2026, 8, 31), 15).hour == 15

    def test_falls_back_to_the_last_bar_before_the_requested_hour(self):
        from trading_app.lib.covered_call import _bar_at
        f = self.day_frame([13, 14, 16])          # no 15:00 bar
        assert _bar_at(f, dt.date(2026, 8, 31), 15).hour == 14

    def test_when_the_day_starts_late_it_takes_the_first_bar_it_has(self):
        """
        Documented rather than assumed: asking for 10:00 on a day whose first
        bar is 13:00 returns the 13:00 bar, which is AFTER the hour requested.
        That is the only sensible answer, but it means the order timestamp is
        not always the hour asked for, and the blotter records the bar it
        actually used rather than the one requested.
        """
        from trading_app.lib.covered_call import _bar_at
        f = self.day_frame([13, 14, 15])
        assert _bar_at(f, dt.date(2026, 8, 31), 10).hour == 13

    def test_a_day_with_no_bars_has_no_order_bar(self):
        from trading_app.lib.covered_call import _bar_at, _last_bar
        f = self.day_frame([13, 14])
        assert _bar_at(f, dt.date(2026, 9, 1), 15) is None
        assert _last_bar(f, dt.date(2026, 9, 1)) is None

    def test_settlement_uses_the_last_bar_of_the_day(self):
        from trading_app.lib.covered_call import _last_bar
        f = self.day_frame([13, 14, 15, 16, 20])
        assert _last_bar(f, dt.date(2026, 8, 31)).hour == 20


class TestOccSymbol:
    def test_matches_the_osi_layout(self):
        from trading_app.lib.ric import occ_symbol
        assert occ_symbol("AAPL", dt.date(2026, 9, 4), 320.0, "C") == "AAPL  260904C00320000"
        assert occ_symbol("AAPL", dt.date(2026, 8, 7), 312.5, "C") == "AAPL  260807C00312500"
        assert len(occ_symbol("SPY", dt.date(2026, 8, 7), 600.0, "P")) == 21

    def test_rejects_a_bad_right(self):
        from trading_app.lib.ric import occ_symbol
        with pytest.raises(ValueError):
            occ_symbol("AAPL", dt.date(2026, 9, 4), 320.0, "X")


class TestLoadCache:
    def test_a_missing_cache_says_how_to_build_it(self):
        """
        The failure mode worth avoiding is a bare FileNotFoundError on a
        pickle path, which tells a reader nothing about needing LSEG.
        """
        from trading_app.lib.covered_call import load_cache
        with pytest.raises(FileNotFoundError, match="fetch_hw2.py"):
            load_cache("/nonexistent/covered_call_NOPE.pkl")


# --------------------------------------------------------------------------
# 11. bar timestamps and the close -- the off-by-one found in class
# --------------------------------------------------------------------------
# LSEG stamps an intraday bar with its START; its prices are as of its END.
# The book used LSEG's label as the observation time, which (a) stamped every
# trade one period early and (b) treated LSEG's "20:00" bar -- 4 to 5pm ET,
# after-hours -- as the session close. Every end-of-day NAV and every expiry
# settlement came from after-hours prints. Jun 29 read $50,006.50; the true
# close NAV is $50,036.00.

class TestAsOfStamps:
    def test_an_hourly_bar_is_restamped_to_its_end(self):
        from trading_app.lib.covered_call import stock_panel
        raw = pd.DataFrame({"TRDPRC_1": [281.505]},
                           index=pd.DatetimeIndex(["2026-06-29 15:00"]))
        got = stock_panel({"stock": raw, "interval": "1h"})
        assert list(got.index) == [pd.Timestamp("2026-06-29 16:00")]

    def test_the_after_hours_bar_is_dropped(self):
        """LSEG's 20:00 bar is 4-5pm ET. It is not the close; it is gone."""
        from trading_app.lib.covered_call import stock_panel
        raw = pd.DataFrame({"TRDPRC_1": [281.63, 281.55]},
                           index=pd.DatetimeIndex(["2026-06-29 19:00", "2026-06-29 20:00"]))
        got = stock_panel({"stock": raw})
        assert list(got.index) == [pd.Timestamp("2026-06-29 20:00")]
        assert got["TRDPRC_1"].iloc[0] == pytest.approx(281.63), \
            "the bar stamped 20:00 must carry the closing hour's print, not after-hours"

    def test_minute_bars_restamp_by_one_minute(self):
        from trading_app.lib.covered_call import stock_panel
        t = ["2026-06-29 13:29", "2026-06-29 13:30", "2026-06-29 19:59", "2026-06-29 20:00"]
        raw = pd.DataFrame({"TRDPRC_1": [1.0, 2.0, 3.0, 4.0]}, index=pd.DatetimeIndex(t))
        got = stock_panel({"stock": raw, "interval": "1min"})
        assert [str(x.time()) for x in got.index] == ["13:31:00", "20:00:00"]
        assert list(got["TRDPRC_1"]) == [2.0, 3.0]

    def test_option_bars_are_restamped_the_same_way(self):
        """Stock and options must agree, or every combo is priced across two moments."""
        from trading_app.lib.covered_call import option_panel, stock_panel
        o = option_panel({"options": opt_frame(hours=("15:00", "19:00"), bid=(1.0, 1.1),
                                               ask=(1.2, 1.3))})
        s = stock_panel({"stock": pd.DataFrame(
            {"TRDPRC_1": [1.0, 2.0]}, index=pd.DatetimeIndex(["2026-08-31 15:00", "2026-08-31 19:00"]))})
        assert list(o["ts"]) == list(s.index)

    def test_an_unknown_interval_is_refused(self):
        from trading_app.lib.covered_call import stock_panel
        raw = pd.DataFrame({"TRDPRC_1": [1.0]}, index=pd.DatetimeIndex(["2026-06-29 15:00"]))
        with pytest.raises(ValueError, match="interval"):
            stock_panel({"stock": raw, "interval": "7min"})

    def test_tradeable_hours_are_bars_that_close_inside_the_session(self):
        from trading_app.lib.covered_call import TRADEABLE_HOURS
        assert TRADEABLE_HOURS == tuple(range(14, 21))


class TestTheOfficialClose:
    @staticmethod
    def expiry_week(last_trade: float, official: float | None, strike_spot=300.0):
        stock = make_stock({MON: strike_spot, TUE: strike_spot, WED: strike_spot,
                            THU: strike_spot, FRI: last_trade})
        stock["OFFICIAL_CLOSE"] = np.nan
        if official is not None:
            stock.loc[[t for t in stock.index if t.date() == FRI], "OFFICIAL_CLOSE"] = official
        opts = make_options(FRI, [MON, TUE, WED, THU, FRI], STRIKES)
        run = run_backtest(stock, opts, trading_weeks(stock), order_hour=15)
        return run, build_ledger(run, stock, opts), stock

    def test_the_panel_attaches_the_official_close_by_date(self):
        from trading_app.lib.covered_call import stock_panel
        raw = pd.DataFrame({"TRDPRC_1": [281.63, 289.09]},
                           index=pd.DatetimeIndex(["2026-06-29 19:00", "2026-06-30 19:00"]))
        got = stock_panel({"stock": raw, "stock_official_close": {"2026-06-29": 281.74}})
        assert got["OFFICIAL_CLOSE"].iloc[0] == pytest.approx(281.74)
        assert np.isnan(got["OFFICIAL_CLOSE"].iloc[1]), "no close known -> NaN, not a guess"

    def test_assignment_is_decided_by_the_official_close(self):
        """
        The last trade before the bell was above the strike; the closing
        auction was not. The call expires. Deciding on the last trade would
        book an assignment that did not happen.
        """
        run, _, _ = self.expiry_week(last_trade=300.40, official=299.90)
        assert [e["side"] for e in run["blotter"] if e["kind"] == "call"][-1] == EXPIRE
        assert run["cycles"][0]["settle"] == pytest.approx(299.90)
        assert run["cycles"][0]["settle_source"] == "official close"

    def test_and_the_other_way_round(self):
        run, _, _ = self.expiry_week(last_trade=299.90, official=300.40)
        assert [e["side"] for e in run["blotter"] if e["kind"] == "call"][-1] == ASSIGN

    def test_without_an_official_close_it_says_it_used_the_last_trade(self):
        run, _, _ = self.expiry_week(last_trade=295.0, official=None)
        assert run["cycles"][0]["settle_source"] == "last trade"
        assert "last trade" in run["blotter"][-1]["note"]

    def test_the_close_bar_is_marked_at_the_official_close(self):
        _, led, _ = self.expiry_week(last_trade=295.0, official=296.25)
        close_row = led[led["ts"] == pd.Timestamp(dt.datetime.combine(FRI, dt.time(20)))].iloc[0]
        assert close_row["stock_mark"] == pytest.approx(296.25)
        assert close_row["nav"] == pytest.approx(
            close_row["cash"] + 100 * 296.25 + close_row["option_mv"])

    def test_bars_before_the_close_keep_their_own_last_trade(self):
        from trading_app.lib.covered_call import stock_marks
        _, _, stock = self.expiry_week(last_trade=295.0, official=296.25)
        marks = stock_marks(stock)
        before = [t for t in stock.index if t.date() == FRI and t.hour < 20]
        assert all(marks[t] == pytest.approx(295.0) for t in before)


# ---------------------------------------------------------------------------
# the real book: pin the numbers that were wrong, computed a second way
# ---------------------------------------------------------------------------

REAL_CACHE = Path(__file__).resolve().parents[1] / "trading_app" / "data" / "covered_call_AAPL.pkl"
MINUTE_CACHE = REAL_CACHE.with_name("covered_call_AAPL_1min.pkl")


@pytest.fixture(scope="module")
def real_book():
    if not REAL_CACHE.exists():
        pytest.skip("no committed cache")
    from trading_app.lib.covered_call import load_cache, option_panel, stock_panel
    P = load_cache(REAL_CACHE)
    if not P.get("stock_official_close"):
        pytest.skip("cache predates official closes; run fetch_hw2.py --backfill-closes")
    st, op = stock_panel(P), option_panel(P)
    run = run_backtest(st, op, trading_weeks(st), order_hour=16, start_cash=50_000.0)
    return P, st, op, run, build_ledger(run, st, op)


class TestRealBookAtTheClose:
    def test_jun_29_close_nav_matches_a_hand_calculation_from_raw_lseg_data(self, real_book):
        """
        The row flagged in class. Rebuilt WITHOUT the pipeline, straight from
        LSEG's raw start-stamped frames, so a bug shared by the loaders and the
        ledger cannot make both sides agree:

            bought 100 at raw-15:00 print 281.505, sold the call at raw-15:00
            mid 3.00  ->  cash 50,000 - 28,150.50 + 300 = 22,149.50
            close: official 281.74, option mid in raw-19:00 bar 2.875
            NAV = 22,149.50 + 28,174 - 287.50 = 50,036.00

        It used to read 50,006.50, from after-hours prices.
        """
        P, _, _, _, led = real_book
        raw_s = P["stock"].copy()
        raw_s.index = pd.DatetimeIndex(raw_s.index)
        ric = "AAPLG022628250.U^G26"
        q_open = P["options"][ric].loc[pd.Timestamp("2026-06-29 15:00")]
        q_close = P["options"][ric].loc[pd.Timestamp("2026-06-29 19:00")]
        cash = (50_000 - 100 * float(raw_s.at[pd.Timestamp("2026-06-29 15:00"), "TRDPRC_1"])
                + 100 * (float(q_open["BID"]) + float(q_open["ASK"])) / 2)
        nav = (cash + 100 * P["stock_official_close"]["2026-06-29"]
               - 100 * (float(q_close["BID"]) + float(q_close["ASK"])) / 2)

        row = led[led["ts"] == pd.Timestamp("2026-06-29 20:00")].iloc[0]
        assert row["nav"] == pytest.approx(nav)
        assert row["nav"] == pytest.approx(50_036.00)
        assert row["nav"] != pytest.approx(50_006.50), "the after-hours value is back"

    def test_every_close_bar_is_marked_at_the_official_close(self, real_book):
        P, _, _, _, led = real_book
        closes = P["stock_official_close"]
        n = 0
        for _, row in led[led["ts"].dt.time == dt.time(20)].iterrows():
            oc = closes.get(str(row["ts"].date()))
            if oc is not None:
                assert row["stock_mark"] == pytest.approx(oc), row["ts"]
                n += 1
        assert n >= 45

    def test_every_expiry_settles_on_the_official_close(self, real_book):
        P, _, _, run, _ = real_book
        for c in run["cycles"]:
            if c["status"] in ("assigned", "expired"):
                assert c["settle_source"] == "official close", c["iso"]
                assert c["settle"] == pytest.approx(P["stock_official_close"][str(c["expiry_date"])])

    def test_no_bar_is_stamped_outside_the_session(self, real_book):
        _, st, op, _, led = real_book
        for stamps in (st.index, pd.DatetimeIndex(op["ts"]), pd.DatetimeIndex(led["ts"])):
            t = stamps.time
            assert all(dt.time(13, 30) < x <= dt.time(20, 0) for x in t), "a bar outside 13:30-20:00"

    def test_trades_are_stamped_when_their_prices_were_observed(self, real_book):
        """The entry fills on prices as of 16:00 UTC, so it is stamped 16:00, not 15:00."""
        _, _, _, run, _ = real_book
        assert run["blotter"][0]["ts"] == pd.Timestamp("2026-06-29 16:00")

    def test_final_nav_did_not_move(self, real_book):
        """Assignment pays the strike, so fixing the close cannot change where the book ends."""
        _, _, _, _, led = real_book
        assert led["nav"].iloc[-1] == pytest.approx(50_681.82, abs=0.01)


class TestLsegStampsTheStartOfTheBar:
    """
    A contract with the data vendor, checked rather than assumed. If LSEG ever
    starts stamping bar ends, every relabel above becomes the off-by-one.
    Needs the one-minute cache, which is not committed, so it skips cleanly.
    """

    def test_hourly_price_is_the_last_minute_of_the_hour_it_is_labelled_with(self):
        if not (REAL_CACHE.exists() and MINUTE_CACHE.exists()):
            pytest.skip("needs both the hourly and the one-minute cache")
        from trading_app.lib.covered_call import load_cache
        h = load_cache(REAL_CACHE)["stock"]
        m = load_cache(MINUTE_CACHE)["stock"]
        h.index, m.index = pd.DatetimeIndex(h.index), pd.DatetimeIndex(m.index)
        hp = pd.to_numeric(h["TRDPRC_1"], errors="coerce")
        mp = pd.to_numeric(m["TRDPRC_1"], errors="coerce").dropna()
        start_hits = end_hits = n = 0
        for d in sorted(set(m.index.date))[:15]:
            for H in range(14, 20):
                t = pd.Timestamp(f"{d} {H:02d}:00")
                if t not in hp.index or pd.isna(hp[t]):
                    continue
                inside = mp[(mp.index >= t) & (mp.index < t + pd.Timedelta(hours=1))]
                before = mp[(mp.index >= t - pd.Timedelta(hours=1)) & (mp.index < t)]
                if len(inside) and len(before):
                    n += 1
                    start_hits += abs(inside.iloc[-1] - hp[t]) < 1e-6
                    end_hits += abs(before.iloc[-1] - hp[t]) < 1e-6
        assert n > 50
        assert start_hits / n > 0.95, f"start-stamped on only {start_hits}/{n} bars"
        assert end_hits / n < 0.05


# --------------------------------------------------------------------------
# 12. margin interest: an accrued liability, never a cash movement
# --------------------------------------------------------------------------

class TestMarginInterest:
    @staticmethod
    def two_weeks(start_cash, rate):
        """Mon-Fri, then the following Monday, so a weekend sits inside."""
        days = {MON: 300.0, TUE: 300.0, WED: 300.0, THU: 300.0, FRI: 295.0,
                dt.date(2026, 7, 13): 295.0, dt.date(2026, 7, 14): 295.0}
        stock = make_stock(days)
        opts = make_options(FRI, list(days)[:5], STRIKES)
        run = run_backtest(stock, opts, trading_weeks(stock)[:1], order_hour=15,
                           start_cash=start_cash, margin_rate=rate)
        return run, build_ledger(run, stock, opts)

    def test_no_borrowing_means_no_interest(self):
        _, led = self.two_weeks(50_000.0, 0.07)
        assert (led["accrued_interest"] == 0).all()

    def test_a_zero_rate_charges_nothing_even_when_borrowing(self):
        _, led = self.two_weeks(20_000.0, 0.0)
        assert (led["cash"] < 0).any()
        assert (led["accrued_interest"] == 0).all()

    def test_accrues_actual_over_360_on_the_debit_balance(self):
        """
        Start 20,000; buy 100 at 300 (-30,000), collect 100 at the 1.00 mid:
        cash -9,900 from Monday on. Each close accrues 9,900 x 7% x days/360.
        Mon->Tue, Tue->Wed, Wed->Thu, Thu->Fri are one day each; Fri->Mon is
        three. The row on Monday the 13th has seen all five closes: 7 days.
        """
        _, led = self.two_weeks(20_000.0, 0.07)
        mon13 = led[led["ts"] == pd.Timestamp("2026-07-13 20:00")].iloc[0]
        assert mon13["cash"] == pytest.approx(-9_900.0)
        assert mon13["accrued_interest"] == pytest.approx(9_900 * 0.07 * 7 / 360)

    def test_the_weekend_counts_three_days(self):
        _, led = self.two_weeks(20_000.0, 0.07)
        fri = led[led["ts"] == pd.Timestamp("2026-07-10 20:00")].iloc[0]["accrued_interest"]
        mon = led[led["ts"] == pd.Timestamp("2026-07-13 14:00")].iloc[0]["accrued_interest"]
        assert mon - fri == pytest.approx(9_900 * 0.07 * 3 / 360)

    def test_interest_never_touches_cash(self):
        """Cash moves only on blotter events; interest is a liability in NAV."""
        run, led = self.two_weeks(20_000.0, 0.07)
        for _, row in led.iterrows():
            due = run["start_cash"] + sum(e["cash_delta"] for e in run["blotter"]
                                          if e["ts"] <= row["ts"])
            assert row["cash"] == pytest.approx(due)

    def test_interest_comes_out_of_nav_and_available_funds(self):
        _, with_rate = self.two_weeks(20_000.0, 0.07)
        _, without = self.two_weeks(20_000.0, 0.0)
        last_a, last_b = with_rate.iloc[-1], without.iloc[-1]
        assert last_a["nav"] == pytest.approx(last_b["nav"] - last_a["accrued_interest"])
        assert last_a["available_funds"] == pytest.approx(
            last_b["available_funds"] - last_a["accrued_interest"])

    def test_buy_and_hold_borrows_on_the_same_terms(self):
        from trading_app.lib.cc_analysis import buy_and_hold
        days = {MON: 300.0, TUE: 300.0, WED: 300.0, THU: 300.0, FRI: 300.0}
        stock = make_stock(days)
        opts = make_options(FRI, list(days), STRIKES)
        run = run_backtest(stock, opts, trading_weeks(stock), order_hour=15, start_cash=29_000.0)
        led = build_ledger(run, stock, opts)
        free = buy_and_hold(stock, led, 29_000.0, margin_rate=0.0)
        paid = buy_and_hold(stock, led, 29_000.0, margin_rate=0.07)
        # 100 x 300 against 29,000 cash: a 1,000 loan for Mon->Thu closes, 4 days
        assert free["nav"].iloc[-1] - paid["nav"].iloc[-1] == pytest.approx(1_000 * 0.07 * 4 / 360)
