"""
The two wings on the entry close: the strike rule (|delta| under the cap, closest to the target, the tie further out
of the money), compared with the rule as the technical choices write it (fw_live_synth.rule_as_written, a plain loop)
on the same synthetic chains; what is not two-sided or has no implied vol; and the size, round(200 / vega per vol
point per contract), at least 1. The backtest's lib/vol_wings.py does not exist yet, so the comparison is with the
rule as written; once it does, the same chains should be run through its wing choice too.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

import fw_live_synth as S
from trading_app.lib import far_wings_live as FW
from trading_app.lib import wheel as W
from trading_app.lib.vol import bs_price


def wings(mkt, day=S.ENTRY, expiry=S.EXPIRY):
    return FW.wings_at_entry(mkt, day, expiry)


def carry(mkt, day=S.ENTRY):
    return W.carry(mkt, pd.Timestamp(day))


# --------------------------------------------------------------------------- the rule, against the rule as written
CHAINS = list(itertools.product([-0.10, -0.04, 0.0, 0.05], [0.25, 0.45, 0.90], [60.0, 178.5, 900.0]))


@pytest.mark.parametrize("b, s0, spot", CHAINS)
def test_the_strike_choice_is_the_rule_as_written(b, s0, spot):
    c = S.closes(last=spot)
    rows = S.chain_rows(spot, b=b, s0=s0, strikes=S.strikes_around(spot, step=spot / 70))
    mkt = S.market(rows=rows, c=c)
    r, q = carry(mkt)
    got = wings(mkt)
    for cp in ("P", "C"):
        want = S.rule_as_written(rows, cp, spot, S.ENTRY, S.EXPIRY, r, q)
        assert (None if got[cp] is None else (got[cp]["strike"], abs(got[cp]["delta"]))) == \
            (None if want is None else (want[0], pytest.approx(want[1], abs=1e-12)))


@pytest.mark.parametrize("seed", range(12))
def test_the_rule_as_written_on_ragged_chains(seed):
    # noisy mids, strikes missing, a few one-sided quotes: the far mids need not give a |delta| monotone in strike
    rng = np.random.default_rng(seed)
    spot = float(rng.uniform(40, 400))
    ks = S.strikes_around(spot, step=round(spot / 60, 1))
    drop = {(cp, float(k)) for cp in "PC" for k in ks if rng.random() < 0.15}
    one = {(cp, float(k)) for cp in "PC" for k in ks if rng.random() < 0.1}
    rows = S.chain_rows(spot, b=float(rng.uniform(-0.12, 0.08)), s0=float(rng.uniform(0.2, 1.0)), strikes=ks,
                        drop=drop, one_sided=one)
    for row in rows:
        f = float(np.exp(rng.normal(0, 0.15)))
        row["bid"], row["ask"] = row["bid"] * f, row["ask"] * f
    mkt = S.market(rows=rows, c=S.closes(last=spot))
    r, q = carry(mkt)
    got = wings(mkt)
    for cp in ("P", "C"):
        want = S.rule_as_written(rows, cp, spot, S.ENTRY, S.EXPIRY, r, q)
        assert (None if got[cp] is None else got[cp]["strike"]) == (None if want is None else want[0])


def test_the_wings_on_the_base_chain():
    mkt = S.market()
    got = wings(mkt)
    put, call = got["P"], got["C"]
    assert (put["strike"], call["strike"]) == (145.0, 220.0)
    assert -0.10 < put["delta"] < 0 and 0 < call["delta"] < 0.05
    # every other eligible two-sided strike is further from the target
    r, q = carry(mkt)
    T = (S.EXPIRY - S.ENTRY).days / 365
    spot = float(mkt["stock"]["close"].iloc[-1])
    day = mkt["quotes"]
    for cp, leg in (("P", put), ("C", call)):
        for _, row in day[(day["cp"] == cp) & (day["mid"] > 0)].iterrows():
            d = abs(W.option_greeks(row["mid"], spot, row["strike"], T, r, q, cp)["delta"])
            if d < FW.CAP[cp] and row["strike"] != leg["strike"]:
                assert abs(d - FW.TARGET[cp]) > abs(abs(leg["delta"]) - FW.TARGET[cp])
    assert got["T"] == 25 / 365 and got["lot"] == 100 and got["spot"] == spot


# --------------------------------------------------------------------------- choose(): the cap, the target, the ties
def cands(*pairs):
    return pd.DataFrame([{"strike": k, "delta": d} for k, d in pairs])


def test_a_tie_goes_to_the_lower_strike_for_a_put_and_the_higher_for_a_call():
    assert FW.choose(cands((150.0, -0.05), (145.0, -0.05), (155.0, -0.05)), "P")["strike"] == 145.0
    assert FW.choose(cands((215.0, 0.03), (225.0, 0.03), (220.0, 0.03)), "C")["strike"] == 225.0


def test_the_closest_to_the_target_wins_whatever_its_strike():
    assert FW.choose(cands((140.0, -0.02), (150.0, -0.06), (145.0, -0.045)), "P")["strike"] == 145.0
    assert FW.choose(cands((230.0, 0.01), (215.0, 0.049), (220.0, 0.032)), "C")["strike"] == 220.0


def test_the_cap_is_strict():
    # 0.10 itself is not a far put, nor 0.05 a far call, however close to nothing else
    assert FW.choose(cands((160.0, -0.10), (100.0, -0.001)), "P")["strike"] == 100.0
    assert FW.choose(cands((160.0, -0.10)), "P") is None
    assert FW.choose(cands((205.0, 0.05)), "C") is None
    assert FW.choose(cands((205.0, 0.0499999)), "C")["strike"] == 205.0
    assert FW.choose(cands((160.0, -0.0999999)), "P")["strike"] == 160.0


def test_a_missing_delta_is_never_chosen():
    assert FW.choose(cands((145.0, np.nan), (150.0, -0.09)), "P")["strike"] == 150.0
    assert FW.choose(cands((145.0, np.nan)), "P") is None


def test_no_eligible_strike_is_no_wing():
    # a chain that stops near the money: every put above 0.10 |delta|, every call above 0.05
    spot = float(S.closes().iloc[-1])
    rows = S.chain_rows(spot, strikes=np.arange(170.0, 190.01, 2.5))
    got = wings(S.market(rows=rows))
    assert got["P"] is None and got["C"] is None
    assert wings(S.market(rows=[]))["P"] is None


def test_only_two_sided_quotes_with_an_implied_vol_count():
    spot = float(S.closes().iloc[-1])
    base = wings(S.market())
    k_put, k_call = base["P"]["strike"], base["C"]["strike"]
    # the chosen strikes one-sided, or with the ask under the bid: the next best is chosen instead
    got = wings(S.market(rows=S.chain_rows(spot, one_sided={("P", k_put), ("C", k_call)})))
    assert got["P"]["strike"] != k_put and got["C"]["strike"] != k_call
    rows = S.chain_rows(spot)
    for row in rows:
        if (row["cp"], row["strike"]) in {("P", k_put), ("C", k_call)}:
            row["bid"], row["ask"] = row["ask"], row["bid"]
    got = wings(S.market(rows=rows))
    assert got["P"]["strike"] != k_put and got["C"]["strike"] != k_call
    # a mid above the strike (more than any put can be worth) has no implied vol
    rows = S.chain_rows(spot)
    for row in rows:
        if row["cp"] == "P" and row["strike"] == k_put:
            row["bid"], row["ask"] = 900.0, 1000.0
    T = (S.EXPIRY - S.ENTRY).days / 365
    assert not np.isfinite(W.option_greeks(950.0, spot, k_put, T, 0.04, 0.0, "P")["iv"])
    assert wings(S.market(rows=rows))["P"]["strike"] != k_put


def test_another_expirys_quotes_and_another_days_are_left_out():
    spot = float(S.closes().iloc[-1])
    base = wings(S.market())
    other = S.chain_rows(spot, expiry=pd.Timestamp("2027-01-15").date())
    early = [{**r, "date": pd.Timestamp("2026-11-20")} for r in S.chain_rows(spot, b=-0.2)]
    # the earlier day's rows first: the same RICs, so only the date keeps them out
    got = wings(S.market(rows=early + other + S.chain_rows(spot)))
    assert got["P"]["strike"] == base["P"]["strike"] and got["C"]["strike"] == base["C"]["strike"]
    assert got["P"]["mid"] == base["P"]["mid"] and got["C"]["mid"] == base["C"]["mid"]


# --------------------------------------------------------------------------- a right with no two-sided quote
def day_frame(rows):
    return FW.quote_frame(rows)


def test_a_healthy_chain_has_no_one_sided_problem():
    spot = float(S.closes().iloc[-1])
    assert FW.one_sided_problem(day_frame(S.chain_rows(spot)), S.EXPIRY, S.ENTRY) is None
    # all but one quote on each right with no bid: one two-sided quote a right is enough
    rows = S.chain_rows(spot)
    keep = {cp: next(r["strike"] for r in rows if r["cp"] == cp) for cp in "PC"}
    one = {(r["cp"], r["strike"]) for r in rows if r["strike"] != keep[r["cp"]]}
    assert FW.one_sided_problem(day_frame(S.chain_rows(spot, one_sided=one)), S.EXPIRY, S.ENTRY) is None


@pytest.mark.parametrize("rights, named", [("PC", "put and no call"), ("P", "put"), ("C", "call")])
def test_a_right_with_rows_but_no_two_sided_quote_is_named(rights, named):
    spot = float(S.closes().iloc[-1])
    rows = S.chain_rows(spot)
    one = {(r["cp"], r["strike"]) for r in rows if r["cp"] in rights}
    got = FW.one_sided_problem(day_frame(S.chain_rows(spot, one_sided=one)), S.EXPIRY, S.ENTRY)
    assert got == (f"no 2026-12-18 {named} has a two-sided 2026-11-23 quote on LSEG (bid and ask both there), "
                   f"though the chain has 2026-11-23 rows")
    # and then that wing is none: the problem is exactly the case the wing rule would read as no eligible strike
    w = wings(S.market(rows=S.chain_rows(spot, one_sided=one)))
    assert all((w[cp] is None) == (cp in rights) for cp in "PC")


@pytest.mark.parametrize("spoil", ["ask under bid", "zero mid", "no ask", "another expiry"])
def test_what_is_not_two_sided_is_vol_arbs_rule(spoil):
    spot = float(S.closes().iloc[-1])
    calls = [r for r in S.chain_rows(spot) if r["cp"] == "C"]
    puts = [r for r in S.chain_rows(spot) if r["cp"] == "P"]
    for r in calls:
        if spoil == "ask under bid":
            r["bid"], r["ask"] = r["ask"], r["bid"]
        elif spoil == "zero mid":
            r["bid"], r["ask"] = 0.0, 0.0
        elif spoil == "no ask":
            r["ask"] = np.nan
        else:
            r["expiry"] = pd.Timestamp("2027-01-15")
    got = FW.one_sided_problem(day_frame(puts + calls), S.EXPIRY, S.ENTRY)
    assert got is not None and got.startswith("no 2026-12-18 call has a two-sided")
    # a zero bid with an ask is two-sided (its mid is above zero), as vol_arb.two_sided has it
    for r in calls:
        r.update(expiry=pd.Timestamp(S.EXPIRY), bid=0.0, ask=0.05)
    assert FW.one_sided_problem(day_frame(puts + calls), S.EXPIRY, S.ENTRY) is None


# --------------------------------------------------------------------------- the size
def test_each_wing_is_sized_to_about_200_dollars_a_vol_point():
    mkt = S.market()
    got = wings(mkt)
    r, q = got["r"], got["q"]
    T, spot = got["T"], got["spot"]
    F, D = spot * np.exp((r - q) * T), np.exp(-r * T)
    for cp in ("P", "C"):
        leg = got[cp]
        h = 1e-5
        fd = (bs_price(F, leg["strike"], T, leg["iv"] + h, D, cp) - bs_price(F, leg["strike"], T, leg["iv"] - h, D, cp)) / (2 * h)
        assert leg["vega"] == pytest.approx(fd, rel=1e-6)                  # Black-76 vega per share per 1.00
        per_point = leg["vega"] / 100 * 100                                  # per vol point per 100-share contract
        assert leg["contracts"] == max(1, round(200 / per_point))
        assert abs(leg["contracts"] * per_point - 200) <= per_point / 2
    assert (got["P"]["contracts"], got["C"]["contracts"]) == (41, 59)


def test_contracts_for():
    assert FW.contracts_for(4.0) == 50                 # $4 a vol point per contract: 50 contracts
    assert FW.contracts_for(4.0, lot=50) == 100        # half the lot, twice the contracts
    assert FW.contracts_for(3.0) == 67                 # 66.7 rounds to 67
    assert FW.contracts_for(1e6) == 1                  # at least one
    assert FW.contracts_for(80.0) == 2                 # 2.5 rounds to the even 2, as round() does
    assert FW.contracts_for(160.0) == 1                # 1.25 rounds to 1


def test_the_entrys_lot_sizes_the_wings():
    mkt = S.market()
    mkt["lot"] = pd.Series(50.0, index=mkt["stock"].index)
    got = wings(mkt)
    base = wings(S.market())
    assert got["lot"] == 50
    for cp in ("P", "C"):
        assert got[cp]["contracts"] == FW.contracts_for(base[cp]["vega"], 50)
        assert got[cp]["contracts"] in (2 * base[cp]["contracts"] - 1, 2 * base[cp]["contracts"],
                                        2 * base[cp]["contracts"] + 1)
