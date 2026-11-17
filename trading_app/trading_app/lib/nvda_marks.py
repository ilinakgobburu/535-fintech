"""
The two-sided variance trade's open NVDA cycle, marked to market each night. No I/O: nvda_marks.py at the repo root
pulls LSEG, builds the market with lib/vol_arb_live.market (the live readers, whose parity with the backtest's cache
readers is tested in tests/vol_arb_live/test_vol_arb_live_backtest_parity.py), calls mark_open_cycle() and
document(), and writes docs/data/vol_arb_nvda_mark.json; .github/workflows/nvda-marks.yml runs it every weeknight and
docs/assets/nvda_marks.js shows the file on the vol-arb page.

NOTHING IS RE-DECIDED. The cycle (entry, expiry, legs, contracts, each model's p and direction) is data in
config/nvda_marks.yaml, decided on the entry night; cycle_spec() only reads and checks it. What is computed each night
is the mark, with the frozen backtest code of lib/vol_arb.py, unchanged:

    open (entry <= as_of < expiry)
        open_cycle_path   VA.cycle_path needs the expiry's session. The market's sessions and quotes after as_of are
                          dropped and one placeholder session is put on the expiry's date (close and open = as_of's
                          close), so cycle_path runs unchanged and every leg's T, greeks and IV count to the real
                          expiry.
        mark_to_market    VA.run_trade on that path, keeping the ledger rows up to as_of (the placeholder's
                          settlement row is dropped). run_trade's row i reads day i's close, open, dividend and marks
                          and day i-1's greeks; the settlement (intrinsic, shares closed) is on the last day alone, so
                          the kept rows use nothing of the placeholder.
    settled (as_of on or after the expiry)
        VA.cycle_path to the expiry and VA.run_trade's own rows, its settlement row included; nothing after the
        expiry is read.

Both sides are marked, put_rich (sell the put, buy the call) and call_rich (buy the put, sell the call), at both
fills: mid, and stress (VA's "cross": the put and call sold at the bid and bought at the ask at entry, and 5 bp of
each hedge trade's value), whichever side a model chose, so the page can show every model's trade.

A leg with no two-sided quote at a close keeps its last mark and IV, and one whose mid has no implied vol (a put mid
under intrinsic) keeps its last IV, with the delta recomputed at that close's spot and T: VA.leg_path's rule, which
leg_days() reports so the notes can say so.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import vol_arb as VA
from . import vol_arb_live as VL
from . import vol_arb_wf as WF
from . import wheel as W
from .ric import parse_option_ric

PUT, CALL = VA.PUT, VA.CALL
LEG = {PUT: "put", CALL: "call"}
FILLS = {"mid": "mid", "stress": "cross"}           # the page's name: VA.run_trade's
DIRECTIONS = ("put_rich", "call_rich")
MODELS = ("logistic_refit", "logistic_fit_once", "forest_refit", "rule_1", "rule_0")
FITTED = ("logistic_refit", "logistic_fit_once", "forest_refit")      # the models with a p
READINGS = ("f7_0", "f7_1")
MODEL_NAMES = {"logistic_refit": "Logistic, refit", "logistic_fit_once": "Logistic, fit once",
               "forest_refit": "Forest, refit", "rule_1": "Rule 1", "rule_0": "Rule 0"}
SIDE_TEXT = {"put_rich": "sell the put, buy the call", "call_rich": "buy the put, sell the call"}
FILL_TEXT = {"mid": "the legs at their mids at entry, the hedge at each open with no cost",
             "stress": "the leg sold at the bid and the leg bought at the ask at entry, and 5 bp of each hedge "
                       "trade's value"}
SCHEMA_VERSION = 1
STRATEGY = "vol_arb"
LEG_KEYS = ("ric", "strike", "contracts", "mid", "bid", "ask", "iv", "delta")


class SpecError(ValueError):
    """config/nvda_marks.yaml's cycle is missing a value, or its values do not hang together."""


# --------------------------------------------------------------------------- the cycle, as decided at entry
def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def cycle_spec(cfg: dict) -> dict:
    """
    The cycle in config/nvda_marks.yaml, read and checked, never recomputed: the entry and expiry (the expiry the
    vol-arb calendar gives the entry), the lot, NVDA's entry close and ATM IV, both legs as VA.legs_at_entry gave them
    (RIC, strike, contracts, mid, bid, ask, IV, delta), and each model's p and direction under both earnings-flag
    readings, each direction the one the 0.45-0.55 band gives its p (vol_arb_wf.decide).
    """
    if not isinstance(cfg, dict) or not isinstance(cfg.get("cycle"), dict):
        raise SpecError("config/nvda_marks.yaml has no cycle")
    c = cfg["cycle"]
    try:
        entry, expiry = pd.Timestamp(c["entry"]).normalize(), pd.Timestamp(c["expiry"]).normalize()
    except (KeyError, TypeError, ValueError) as exc:
        raise SpecError(f"the cycle's entry or expiry is missing or not a date ({exc})") from None
    if not entry < expiry:
        raise SpecError(f"the cycle's entry {entry.date()} is not before its expiry {expiry.date()}")
    if VL.entry_expiry(entry) != expiry.date():
        raise SpecError(f"{entry.date()} is not the vol-arb's entry for the {expiry.date()} expiry")
    lot = c.get("lot")
    if not isinstance(lot, int) or isinstance(lot, bool) or lot <= 0:
        raise SpecError(f"the cycle's lot is not a whole number of shares: {lot!r}")
    for k in ("spot_at_entry", "atm_iv_at_entry"):
        if not _finite(c.get(k)) or float(c[k]) <= 0:
            raise SpecError(f"the cycle's {k} is not a number above 0: {c.get(k)!r}")
    legs = {}
    for cp in (PUT, CALL):
        leg = (c.get("legs") or {}).get(cp)
        if not isinstance(leg, dict) or any(k not in leg for k in LEG_KEYS):
            raise SpecError(f"the cycle's {LEG[cp]} leg needs {', '.join(LEG_KEYS)}")
        if not isinstance(leg["contracts"], int) or isinstance(leg["contracts"], bool) or leg["contracts"] < 1:
            raise SpecError(f"the {LEG[cp]} leg's contracts is not a whole number of at least 1")
        if any(not _finite(leg[k]) for k in ("strike", "mid", "bid", "ask", "iv", "delta")):
            raise SpecError(f"the {LEG[cp]} leg's strike, mid, bid, ask, IV and delta must be numbers")
        if not 0 < leg["bid"] <= leg["mid"] <= leg["ask"] or abs(leg["mid"] - (leg["bid"] + leg["ask"]) / 2) > 1e-9:
            raise SpecError(f"the {LEG[cp]} leg's mid is not the middle of its bid and ask")
        if (leg["delta"] < 0) != (cp == PUT) or not leg["iv"] > 0:
            raise SpecError(f"the {LEG[cp]} leg's delta or IV has the wrong sign")
        meta = parse_option_ric(leg["ric"])
        if (meta is None or meta["cp"] != cp or meta["expiry"] != expiry.date() or meta["expired"]
                or abs(meta["strike"] - float(leg["strike"])) > 1e-9):
            raise SpecError(f"the {LEG[cp]} leg's RIC {leg['ric']!r} is not the live {expiry.date()} "
                            f"{leg['strike']:g} {LEG[cp]}")
        legs[cp] = {"ric": str(leg["ric"]), "strike": float(leg["strike"]), "contracts": int(leg["contracts"]),
                    **{k: float(leg[k]) for k in ("mid", "bid", "ask", "iv", "delta")}}
    decisions = {}
    for reading in READINGS:
        got = (c.get("decisions") or {}).get(reading)
        if not isinstance(got, dict) or set(got) != set(MODELS):
            raise SpecError(f"the cycle's decisions.{reading} must name exactly {', '.join(MODELS)}")
        decisions[reading] = {}
        for m in MODELS:
            d = got[m] if isinstance(got[m], dict) else {}
            direction, p = d.get("direction"), d.get("p")
            if direction not in (*DIRECTIONS, None):
                raise SpecError(f"decisions.{reading}.{m}: direction {direction!r} is not put_rich, call_rich or null")
            if m in FITTED:
                if not _finite(p) or not 0.0 <= float(p) <= 1.0:
                    raise SpecError(f"decisions.{reading}.{m}: p {p!r} is not a probability")
                if WF.decide(float(p)) != direction:
                    raise SpecError(f"decisions.{reading}.{m}: p {p} gives {WF.decide(float(p))}, not {direction}")
                p = float(p)
            elif p is not None:
                raise SpecError(f"decisions.{reading}.{m} is a rule: it has no p")
            decisions[reading][m] = {"p": p, "direction": direction}
        if decisions[reading]["rule_0"]["direction"] != "put_rich":
            raise SpecError("rule 0 trades the put side as rich, always")
    return {"entry": entry, "expiry": expiry, "lot": lot, "spot_at_entry": float(c["spot_at_entry"]),
            "atm_iv_at_entry": float(c["atm_iv_at_entry"]), "legs": legs, "decisions": decisions}


def legs_for_path(spec: dict) -> dict:
    """The legs as VA.legs_at_entry returns them, which VA.cycle_path reads."""
    return {"lot": spec["lot"], **{cp: dict(spec["legs"][cp]) for cp in (PUT, CALL)}}


# --------------------------------------------------------------------------- the mark of a cycle not yet expired
def open_cycle_path(market: dict, quotes_of_expiry: pd.DataFrame, legs: dict, entry, expiry, asof) -> dict:
    """
    VA.cycle_path for a cycle still open at `asof` (entry <= asof < expiry). The market's sessions after `asof` and
    the quotes after it are dropped, and one placeholder session is put on the expiry's date with close and open
    equal to asof's close, so that VA.cycle_path runs unchanged (it needs the expiry's session) and each leg's T and
    greeks count to the real expiry. The path's last day is the placeholder; mark_to_market() keeps only the rows
    up to `asof`, which do not read it.
    """
    entry, expiry, asof = (pd.Timestamp(x) for x in (entry, expiry, asof))
    stock = market["stock"]
    if not entry <= asof < expiry:
        raise ValueError(f"asof {asof.date()} must be on or after the entry {entry.date()} and before the expiry "
                         f"{expiry.date()}")
    if asof not in stock.index:
        raise ValueError(f"the stock has no session on {asof.date()}")
    kept = stock[stock.index <= asof]
    last = float(kept.at[asof, "close"])
    placeholder = pd.DataFrame({"close": [last], "open": [last]}, index=pd.DatetimeIndex([expiry]))
    m = dict(market)
    m["stock"] = pd.concat([kept, placeholder])
    path = VA.cycle_path(m, quotes_of_expiry[quotes_of_expiry["date"] <= asof], legs, entry, expiry)
    path["placeholder"] = expiry
    return path


def mark_to_market(path: dict, signs: dict, contracts: dict, fill: str, asof) -> dict:
    """
    One open cycle's trade at `asof`: VA.run_trade on open_cycle_path()'s path, keeping the ledger rows up to
    `asof` (the placeholder's settlement row is dropped). P&L to date = NAV at asof's close (cash + option marks at
    that close's mids + shares x close); the split, hedge trades, costs and dividends are summed over the kept rows.
    """
    asof = pd.Timestamp(asof)
    days = path["days"]
    if len(days) < 2 or days[-2] != asof or not days[-1] > asof:
        raise ValueError("the path must end with asof and then one placeholder session after it (open_cycle_path)")
    tr = VA.run_trade(path, signs, contracts, fill)
    led = tr["ledger"].iloc[:-1].reset_index(drop=True)
    if pd.Timestamp(led["date"].iloc[-1]) != asof:
        raise ValueError("the kept ledger does not end at asof")
    spread = -tr["spread_cost"]                 # the entry legs' distance from mid (<= 0), in the entry row's costs
    return {"ledger": led, "pnl": float(led["nav"].iloc[-1]),
            "split": {k: float(led[k].sum()) for k in VA.SPLIT},
            "contracts": tr["contracts"], "units": tr["units"], "entry_shares": tr["entry_shares"],
            "hedge_trades": int((led["hedge_trade"] != 0).sum()),
            "hedge_costs": float(spread - led["costs"].sum()), "spread_cost": float(tr["spread_cost"]),
            "dividends": float(led["dividend"].sum()), "shares_held": int(led["shares"].iloc[-1]),
            "option_mv": float(led["option_mv"].iloc[-1]), "cash": float(led["cash"].iloc[-1]),
            "monthly": VA.monthly_pnl(led), "fill": fill, "asof": asof}


def settled_trade(path: dict, signs: dict, contracts: dict, fill: str) -> dict:
    """A cycle run to its expiry: VA.run_trade's own ledger and totals, the settlement row included."""
    tr = VA.run_trade(path, signs, contracts, fill)
    led = tr["ledger"]
    return {"ledger": led, "pnl": tr["pnl"], "split": tr["split"], "contracts": tr["contracts"],
            "units": tr["units"], "entry_shares": tr["entry_shares"], "hedge_trades": int(tr["hedge_trades"]),
            "hedge_costs": tr["hedge_costs"], "spread_cost": tr["spread_cost"], "dividends": tr["dividends"],
            "shares_held": int(led["shares"].iloc[-1]), "option_mv": float(led["option_mv"].iloc[-1]),
            "cash": float(led["cash"].iloc[-1]), "monthly": VA.monthly_pnl(led), "fill": fill,
            "asof": pd.Timestamp(led["date"].iloc[-1])}


# --------------------------------------------------------------------------- what the legs' quotes gave each close
def leg_days(market: dict, quotes_of_expiry: pd.DataFrame, spec: dict, path: dict, days) -> dict:
    """
    For each leg and each close in `days` (the path's own, up to the mark), what VA.leg_path had to work with: the
    close's own two-sided mid or none (then the last mark is carried), and whether that mid has an implied vol
    (W.option_greeks, as leg_path computes it; without one the last IV is carried), the day each carried value is
    from, and the leg's intrinsic value at that close. The entry close is the spec's (select()'s mid and IV).
    Reporting only: the marks themselves are the path's.
    """
    expiry = spec["expiry"]
    out = {}
    for cp in (PUT, CALL):
        leg, p = spec["legs"][cp], path["legs"][cp]
        mids = quotes_of_expiry[quotes_of_expiry["ric"] == leg["ric"]].groupby("date")["mid"].first()
        K, rows = leg["strike"], []
        mark_from = iv_from = pd.Timestamp(days[0])
        for i, d in enumerate(days):
            d = pd.Timestamp(d)
            spot = float(market["stock"].at[d, "close"])
            T = (expiry - d).days / VA.DAYS_PER_YEAR
            if i == 0:
                mid, own_mid, own_iv = leg["mid"], True, True
            else:
                mid = float(mids.get(d, np.nan))
                own_mid = bool(np.isfinite(mid) and mid > 0)
                own_iv = False
                if own_mid:
                    r, q = W.carry(market, d)
                    own_iv = bool(np.isfinite(W.option_greeks(mid, spot, K, T, r, q, cp)["iv"]))
            if own_mid:
                mark_from = d
            if own_iv:
                iv_from = d
            rows.append({"date": d, "mid": mid if own_mid else None, "mark": float(p["mark"][i]),
                         "iv": float(p["iv"][i]), "own_mid": own_mid, "own_iv": own_iv,
                         "mark_from": mark_from, "iv_from": iv_from,
                         "intrinsic": max(K - spot, 0.0) if cp == PUT else max(spot - K, 0.0)})
        out[cp] = rows
    return out


# --------------------------------------------------------------------------- the night's mark
def mark_open_cycle(market: dict, spec: dict, as_of) -> dict:
    """
    The cycle marked at `as_of`'s close, for each side (put_rich, call_rich) and fill (mid, stress): the daily rows
    from the entry to the mark (NVDA's close, each leg's mark and IV, the hedge shares and trade, the P&L to date and
    the day's, and the P&L split to date, VA.run_trade's), the totals (P&L, split, hedge trades, costs, dividends),
    plus NVDA's move and realized vol since the entry and each leg's IV at the mark. Open (entry <= as_of < expiry):
    marked at as_of's mids, not settled (open_cycle_path, mark_to_market). Settled once as_of is on or after the
    expiry: run_trade's own rows to the expiry, settlement included. `market` is lib/vol_arb_live.market's dict (or
    wheel.load_market's); nothing in it dated after the mark is read.
    """
    entry, expiry = spec["entry"], spec["expiry"]
    as_of = pd.Timestamp(as_of).normalize()
    if as_of < entry:
        raise ValueError(f"{as_of.date()} is before the cycle's entry {entry.date()}")
    if entry not in market["stock"].index:
        raise ValueError(f"the stock has no session on the entry {entry.date()}")
    quotes = market["quotes"]
    q_exp = quotes[quotes["expiry"] == expiry]
    legs = legs_for_path(spec)
    n = {cp: spec["legs"][cp]["contracts"] for cp in (PUT, CALL)}
    settled = as_of >= expiry
    if settled:
        path = VA.cycle_path(market, q_exp, legs, entry, expiry)
        days = path["days"]
        trades = {d: {f: settled_trade(path, VA.DIRECTIONS[d], n, FILLS[f]) for f in FILLS} for d in DIRECTIONS}
    else:
        path = open_cycle_path(market, q_exp, legs, entry, expiry, as_of)
        days = path["days"][:-1]
        trades = {d: {f: mark_to_market(path, VA.DIRECTIONS[d], n, FILLS[f], as_of) for f in FILLS}
                  for d in DIRECTIONS}
    mark_day = pd.Timestamp(days[-1])
    lt = leg_days(market, q_exp, spec, path, days)
    closes = pd.Series(path["close"][:len(days)], index=days)
    sides = {}
    for d in DIRECTIONS:
        sides[d] = {}
        for f in FILLS:
            t = trades[d][f]
            led = t["ledger"]
            if len(led) != len(days) or list(pd.DatetimeIndex(led["date"])) != list(days):
                raise ValueError("the ledger and the path's days disagree")
            cum = {s: led[s].cumsum().to_numpy(float) for s in VA.SPLIT}
            rows = []
            for k in range(len(led)):
                r = led.iloc[k]
                rows.append({"date": pd.Timestamp(r["date"]), "close": float(r["close"]),
                             "open": float(path["open"][k]),
                             "put_mark": lt[PUT][k]["mark"], "put_iv": lt[PUT][k]["iv"],
                             "call_mark": lt[CALL][k]["mark"], "call_iv": lt[CALL][k]["iv"],
                             "hedge_trade": int(r["hedge_trade"]), "shares": int(r["shares"]),
                             "dividend": float(r["dividend"]), "option_mv": float(r["option_mv"]),
                             "cash": float(r["cash"]), "pnl_to_date": float(r["nav"]), "day_pnl": float(r["pnl"]),
                             "split_to_date": {s: float(cum[s][k]) for s in VA.SPLIT}})
            sides[d][f] = {"pnl": t["pnl"], "split": t["split"], "hedge_trades": t["hedge_trades"],
                           "entry_shares": t["entry_shares"], "shares_held": t["shares_held"],
                           "spread_cost": t["spread_cost"], "hedge_costs": t["hedge_costs"],
                           "dividends": t["dividends"], "units": t["units"], "monthly": t["monthly"], "rows": rows}
    rets = np.log(closes / closes.shift(1)).dropna()
    realized = float(np.sqrt(VA.SESSIONS_PER_YEAR / len(rets) * float((rets ** 2).sum()))) if len(rets) else None
    settlement = None
    if settled:
        settlement = {"date": expiry, "close": float(path["close"][-1]),
                      "intrinsic": {cp: float(path["legs"][cp]["intrinsic"]) for cp in (PUT, CALL)}}
    return {"status": "settled" if settled else "open", "as_of": mark_day, "entry": entry, "expiry": expiry,
            "sides": sides, "legs": lt, "settlement": settlement,
            "nvda": {"entry_close": float(closes.iloc[0]), "close": float(closes.iloc[-1]),
                     "change": float(closes.iloc[-1] - closes.iloc[0]),
                     "change_pct": 100.0 * float(closes.iloc[-1] / closes.iloc[0] - 1.0),
                     "high_close": float(closes.max()), "low_close": float(closes.min()),
                     "sessions": len(days)},
            "vol": {"realized_since_entry": realized, "returns": int(len(rets)),
                    "atm_iv_at_entry": spec["atm_iv_at_entry"]}}


# --------------------------------------------------------------------------- the night's data, and what to say
def last_complete_session(market: dict, entry, day) -> pd.Timestamp | None:
    """The last session from the entry to `day` with NVDA's close and (after the entry) its open, or None."""
    st = market["stock"]
    entry, day = pd.Timestamp(entry), pd.Timestamp(day)
    w = st[(st.index >= entry) & (st.index <= day)]
    ok = w["close"].notna() & ((w.index == entry) | w["open"].notna())
    return pd.Timestamp(w.index[ok][-1]) if ok.any() else None


def quoted_closes(market: dict, spec: dict, day) -> pd.DatetimeIndex:
    """
    The closes whose leg quotes a mark at `day` reads: the stock's sessions from the entry to `day` while the cycle
    is open, and from the entry to the session before the expiry once it is settled (the expiry's close settles at
    intrinsic, so its quotes are not read). The first is the entry close, whose mark and IV are the spec's.
    """
    day = pd.Timestamp(day).normalize()
    idx = market["stock"].index
    days = idx[(idx >= spec["entry"]) & (idx <= day)]
    return days[days < spec["expiry"]]


def _leg_dates(market: dict, ric: str) -> set:
    q = market["quotes"]
    return set(pd.DatetimeIndex(q.loc[q["ric"] == ric, "date"]))


def data_problems(market: dict, spec: dict, day) -> list[str]:
    """
    Why `day`'s data is not final for the mark (empty when it is): NVDA has no close that session, or (after the
    entry) no open, or LSEG has no row at all for a leg on the last close the mark reads (quoted_closes): `day`
    itself while the cycle is open (not published yet), the session before the expiry once it is settled. A leg with
    a row but no two-sided quote (no bid: nobody bidding for a far wing) is final data: VA.leg_path carries its last
    mark and IV, the registered rule, and the notes say so (carried_notes). The expiry's own quotes are not needed:
    the legs settle at intrinsic there. The retry may mark over these, the missing close carried; history_problems()
    lists what it may not.
    """
    day = pd.Timestamp(day).normalize()
    st, out = market["stock"], []
    if day not in st.index or not np.isfinite(st.at[day, "close"]):
        return [f"LSEG has no NVDA close for {day.date()}"]
    if day > spec["entry"] and not np.isfinite(st.at[day, "open"]):
        out.append(f"LSEG has no NVDA open for {day.date()}")
    reads = quoted_closes(market, spec, day)
    if len(reads):
        last = reads[-1]
        for cp in (PUT, CALL):
            ric = spec["legs"][cp]["ric"]
            if last not in _leg_dates(market, ric):
                out.append(f"LSEG has no {last.date()} quote for the {LEG[cp]} ({ric}) yet" if day < spec["expiry"]
                           else f"LSEG has no {last.date()} quote for the {LEG[cp]} ({ric}), the last close before "
                                f"the expiry")
    return out


def history_problems(market: dict, spec: dict, day) -> list[str]:
    """
    Why the legs' quote history cannot be marked at `day` at all, not even by the retry (empty when it can): for a
    leg, LSEG gave no quote on any close the mark reads (quoted_closes; an empty answer), or none on the entry close
    (the history starts late), or none on either of the last two closes it reads (the history stops early: the retry
    carries one missing close, never more). Marked over, each would carry the leg's mark and IV from wherever its
    quotes stop, and change every hedge trade after it. Any other missing close is LSEG's data as it is, carried
    with a note (carried_notes).
    """
    reads = quoted_closes(market, spec, day)
    out = []
    if not len(reads):
        return out
    entry = spec["entry"]
    for cp in (PUT, CALL):
        ric, name = spec["legs"][cp]["ric"], LEG[cp]
        got = _leg_dates(market, ric) & set(reads)
        if not got:
            out.append(f"LSEG gave no quotes for the {name} ({ric}) from {entry.date()} to {reads[-1].date()}")
        elif entry not in got:
            out.append(f"LSEG has no quote for the {name} ({ric}) on the entry close, {entry.date()}: its history "
                       f"starts {min(got).date()}")
        elif len(reads) > 1 and reads[-1] not in got and reads[-2] not in got:
            out.append(f"LSEG's last quote for the {name} ({ric}) is from {max(got).date()}, before both "
                       f"{reads[-2].date()} and {reads[-1].date()}")
    return out


def entry_differences(market: dict, spec: dict) -> list[str]:
    """Where LSEG's data for the entry close now differs from the values the cycle was entered with (a revision)."""
    out, entry = [], spec["entry"]
    st = market["stock"]
    if entry in st.index and abs(float(st.at[entry, "close"]) - spec["spot_at_entry"]) > 1e-9:
        out.append(f"LSEG's {_day(entry)} NVDA close is now ${float(st.at[entry, 'close']):,.2f}; the cycle was "
                   f"entered at ${spec['spot_at_entry']:,.2f}, and the hedge's first shares are marked at LSEG's "
                   f"close")
    q = market["quotes"]
    for cp in (PUT, CALL):
        leg = spec["legs"][cp]
        at = q[(q["ric"] == leg["ric"]) & (q["date"] == entry)]
        if at.empty:
            continue
        b, a = float(at["bid"].iloc[0]), float(at["ask"].iloc[0])
        same = all(np.isfinite(x) and abs(x - y) <= 1e-9 for x, y in ((b, leg["bid"]), (a, leg["ask"])))
        if not same:
            out.append(f"LSEG's {_day(entry)} quote for the {LEG[cp]} is now {_px(b)} / {_px(a)}; the cycle was "
                       f"entered at {_px(leg['bid'])} / {_px(leg['ask'])} (mid {_px(leg['mid'])}), which the mark keeps")
    return out


def _day(d) -> str:
    d = pd.Timestamp(d)
    return f"{d:%B} {d.day}"


def _px(x) -> str:
    return f"${x:,.2f}" if x is not None and np.isfinite(x) else "none"


def _trim(x: float) -> str:
    """A price as the page writes it: $45.975, $46.36, $4.30 (at least two decimals)."""
    s = f"{x:,.4f}".rstrip("0")
    whole, _, frac = s.partition(".")
    return f"${whole}.{frac.ljust(2, '0')}"


def carried_notes(mark: dict) -> list[str]:
    """
    One note per run of closes on which a leg's mark or IV was carried (VA.leg_path's registered rule), grouped by
    the close the value is carried from. The expiry's close is left out: the legs settle at intrinsic there.
    """
    notes = []
    for cp in (PUT, CALL):
        rows = [r for r in mark["legs"][cp][1:] if r["date"] < mark["expiry"]]
        no_mid = [r for r in rows if not r["own_mid"]]
        no_iv = [r for r in rows if r["own_mid"] and not r["own_iv"]]
        for src, iv_src in sorted({(r["mark_from"], r["iv_from"]) for r in no_mid}):
            got = [r["date"] for r in no_mid if (r["mark_from"], r["iv_from"]) == (src, iv_src)]
            carried = (f"its mark and IV there are carried from {_day(src)}" if iv_src == src else
                       f"its mark there is carried from {_day(src)} and its IV from {_day(iv_src)}")
            notes.append(f"The {LEG[cp]} had no two-sided closing quote on {_dates(got)}, so {carried} (the "
                         f"registered rule).")
        for src in sorted({r["iv_from"] for r in no_iv}):
            got = [r for r in no_iv if r["iv_from"] == src]
            under = all(r["mid"] < r["intrinsic"] for r in got)
            if len(got) == 1 and under:
                r = got[0]
                why = (f"{_day(r['date'])}'s closing mid for the {LEG[cp]} ({_trim(r['mid'])}) is under its "
                       f"intrinsic value ({_trim(r['intrinsic'])}), so it has no implied vol")
            elif under:
                why = (f"the {LEG[cp]}'s closing mid is under its intrinsic value on {_dates([r['date'] for r in got])}, "
                       f"so it has no implied vol there")
            else:
                why = (f"the {LEG[cp]}'s closing mid has no implied vol on {_dates([r['date'] for r in got])}")
            notes.append(f"{why[0].upper()}{why[1:]}; its IV is carried from {_day(src)} (the registered rule) and its "
                         f"delta recomputed from it at {'that close' if len(got) == 1 else 'each of those closes'}.")
    return notes


def _dates(days) -> str:
    names = [_day(d) for d in sorted(days)]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def event_notes(events: list, mark: dict) -> list[str]:
    """Each event inside the cycle (an earnings report after a close): which mark is the last before it, which the
    first after."""
    out = []
    for e in events or []:
        day = pd.Timestamp(e["date"]).normalize()
        if not mark["entry"] <= day <= mark["expiry"]:
            continue
        after = e.get("timing") == "after_close"
        first = pd.Timestamp(VL.next_session(day)) if after else day
        last = day if after else pd.Timestamp(VL.L.sessions(day - pd.Timedelta(days=10), day - pd.Timedelta(days=1))[-1])
        when = f"{_day(day)} after the close" if after else f"{_day(day)} before the open"
        a = mark["as_of"]
        if a < last:
            out.append(f"{e['what']} come on {when}, inside this cycle (expiry {_day(mark['expiry'])}): the "
                       f"{_day(last)} mark is the last before them and the {_day(first)} mark the first after.")
        elif a == last:
            out.append(f"This mark ({_day(a)} close) is before {e['what']}, reported {when}; the {_day(first)} "
                       f"mark is the first after them.")
        elif a == first:
            out.append(f"This mark ({_day(a)} close) is the first after {e['what']}, reported {when}.")
        else:
            out.append(f"{e['what']} came on {when}: the {_day(first)} mark was the first after them.")
    return out


# --------------------------------------------------------------------------- the file
def _r(x, nd: int):
    """x rounded for the file (-0.0 written as 0.0); None for a missing or non-finite value."""
    if x is None:
        return None
    x = float(x)
    return round(x, nd) + 0.0 if math.isfinite(x) else None


def _iso(d) -> str:
    return pd.Timestamp(d).date().isoformat()


def document(mark: dict, spec: dict, cfg: dict, *, extra_notes=(), generated_at: str,
             data_complete: bool = True) -> dict:
    """
    docs/data/vol_arb_nvda_mark.json from mark_open_cycle()'s mark (schema_version 1): dollars to the cent, prices
    to 4 decimals, vols to 6. Everything but generated_at is a function of the mark and the config, so the same
    inputs give the same file. data_complete is False when the retry marked over data that was not final (a leg's
    missing close carried): a settled file is left alone by later runs only when it is True.
    """
    if not isinstance(data_complete, bool):
        raise TypeError("data_complete must be True or False")
    put, call = spec["legs"][PUT], spec["legs"][CALL]
    legs_now = {}
    for cp in (PUT, CALL):
        last = mark["legs"][cp][-1]
        legs_now[LEG[cp]] = {"mark": _r(last["mark"], 4), "iv": _r(last["iv"], 6), "iv_from": _iso(last["iv_from"]),
                             "mark_from": _iso(last["mark_from"]), "intrinsic": _r(last["intrinsic"], 4),
                             "iv_at_entry": _r(spec["legs"][cp]["iv"], 6)}
    marks = {}
    for d in DIRECTIONS:
        marks[d] = {}
        for f in FILLS:
            s = mark["sides"][d][f]
            marks[d][f] = {
                "pnl": _r(s["pnl"], 2), "split": {k: _r(v, 2) for k, v in s["split"].items()},
                "hedge_trades": s["hedge_trades"], "entry_shares": s["entry_shares"], "shares": s["shares_held"],
                "spread_cost": _r(s["spread_cost"], 2), "hedge_costs": _r(s["hedge_costs"], 2),
                "dividends": _r(s["dividends"], 2),
                "units": {LEG[cp]: int(v) for cp, v in s["units"].items()},
                "daily": [{"date": _iso(r["date"]), "close": _r(r["close"], 4), "open": _r(r["open"], 4),
                           "put_mark": _r(r["put_mark"], 4), "put_iv": _r(r["put_iv"], 6),
                           "call_mark": _r(r["call_mark"], 4), "call_iv": _r(r["call_iv"], 6),
                           "hedge_trade": r["hedge_trade"], "shares": r["shares"], "dividend": _r(r["dividend"], 2),
                           "option_mv": _r(r["option_mv"], 2), "cash": _r(r["cash"], 2),
                           "pnl_to_date": _r(r["pnl_to_date"], 2), "day_pnl": _r(r["day_pnl"], 2),
                           "split_to_date": {k: _r(v, 2) for k, v in r["split_to_date"].items()}}
                          for r in s["rows"]]}
    shown = [n["text"] for n in (cfg.get("notes") or []) if mark["as_of"] >= pd.Timestamp(n["from"]).normalize()]
    notes = [*shown, *event_notes(cfg.get("events") or [], mark), *carried_notes(mark), *extra_notes]
    settlement = None
    if mark["settlement"] is not None:
        st = mark["settlement"]
        settlement = {"date": _iso(st["date"]), "close": _r(st["close"], 4),
                      "intrinsic": {LEG[cp]: _r(v, 4) for cp, v in st["intrinsic"].items()}}
    nv, vol = mark["nvda"], mark["vol"]
    return {
        "schema_version": SCHEMA_VERSION,
        "strategy": STRATEGY,
        "underlying": cfg["underlying"]["name"],
        "as_of": _iso(mark["as_of"]),
        "generated_at": generated_at,
        "status": mark["status"],
        "data_complete": data_complete,
        "cycle": {
            "entry": _iso(spec["entry"]), "expiry": _iso(spec["expiry"]), "lot": spec["lot"],
            "spot_at_entry": spec["spot_at_entry"], "atm_iv_at_entry": _r(spec["atm_iv_at_entry"], 6),
            "legs": {LEG[cp]: {"ric": leg["ric"], "strike": leg["strike"], "contracts": leg["contracts"],
                               "mid": leg["mid"], "bid": leg["bid"], "ask": leg["ask"], "iv": _r(leg["iv"], 6),
                               "delta": _r(leg["delta"], 6)}
                     for cp, leg in ((PUT, put), (CALL, call))},
            "decisions": {reading: {m: {"p": _r(v["p"], 5), "direction": v["direction"]}
                                    for m, v in spec["decisions"][reading].items()}
                          for reading in READINGS},
            "models": dict(MODEL_NAMES),
            "source": ("the vol-arb rules' NVDA continuation, computed 2026-11-17: decided on the entry night and "
                       "never recomputed; only the legs are marked each night"),
        },
        "sides": {d: SIDE_TEXT[d] for d in DIRECTIONS},
        "fills": dict(FILL_TEXT),
        "nvda": {"entry_close": _r(nv["entry_close"], 4), "close": _r(nv["close"], 4), "change": _r(nv["change"], 4),
                 "change_pct": _r(nv["change_pct"], 4), "high_close": _r(nv["high_close"], 4),
                 "low_close": _r(nv["low_close"], 4), "sessions": nv["sessions"]},
        "vol": {"realized_since_entry": _r(vol["realized_since_entry"], 6), "returns": vol["returns"],
                "atm_iv_at_entry": _r(vol["atm_iv_at_entry"], 6),
                "how": ("realized: zero-mean, annualized (x 252 / returns) over NVDA's daily log returns from the "
                        "entry close to the mark; implied: the measured at-the-money vol of the cycle's chain at "
                        "entry")},
        "legs_now": legs_now,
        "pnl": {d: {f: marks[d][f]["pnl"] for f in FILLS} for d in DIRECTIONS},
        "marks": marks,
        "settlement": settlement,
        "events": [{"date": _iso(e["date"]), "timing": e.get("timing"), "what": e["what"]}
                   for e in (cfg.get("events") or [])],
        "notes": notes,
    }
