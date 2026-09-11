"""
Analysis over a covered-call run: the fill assumption, the counterfactuals,
and the benchmark.

Everything here calls covered_call.run_backtest / build_ledger rather than
re-deriving anything, so a comparison is always a comparison of decisions and
never of two implementations that drifted apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .covered_call import (
    SHARES_PER_CONTRACT, STRIKE_RULE_META, STRIKE_RULES, TRADEABLE_HOURS,
    build_ledger, run_backtest,
)


# --------------------------------------------------------------------------
# the fill assumption: does the mid predict the trade?
# --------------------------------------------------------------------------

def ols(x: np.ndarray, y: np.ndarray) -> dict:
    """Least squares y = a + b x, with R^2. Returns NaNs rather than raising."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3 or np.ptp(x) == 0:
        return {"n": int(n), "slope": np.nan, "intercept": np.nan,
                "r2": np.nan, "rmse": np.nan}
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(n), "slope": float(b), "intercept": float(a),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "rmse": float(np.sqrt(ss_res / n)),
    }


def mid_vs_print(options: pd.DataFrame, stock: pd.DataFrame) -> dict:
    """
    TRDPRC_1 against mid, pooled and then conditioned.

    Pooled R^2 on option prices needs a companion, because a chain spanning
    deep-ITM contracts worth $40 and wing contracts worth $0.05 can produce a
    spectacular R^2 from the price range alone -- such a fit is graded on its
    ability to know that a $40 option is not a $0.05 option, which nobody
    doubted. HW1 hit the same trap from the other side, where pooling
    INVERTED the spread conclusion until it was conditioned on moneyness.

    So we condition, and on this data the conditioning does NOT destroy the
    fit: R^2 stays between roughly 0.95 and 0.99 inside narrow price bands.
    That is worth stating plainly rather than quietly dropping, because it
    was not the expected result. The mid really does track the print.

    What it does not establish is that the mid was FILLABLE, and those are
    different claims. Predicting the level of a price and being able to
    transact at it are separated by the spread, so we also report:

      - within moneyness buckets, where the price range is narrow
      - by how many times the contract traded in the bar (NUM_MOVES), because
        a "last trade" from a single lot is a different object from a mid
      - the residual in DOLLARS and as a fraction of the quoted spread, which
        is the only scale on which "close to the mid" means anything
    """
    px = pd.to_numeric(stock["TRDPRC_1"], errors="coerce")
    spot = px.reindex(options["ts"].values).to_numpy(dtype=float)

    d = options.copy()
    d["spot"] = spot
    d = d[d["mid"].notna() & d["trdprc_1"].notna() & np.isfinite(d["spot"])]
    d = d[d["spread"].notna() & (d["spread"] > 0)]
    if d.empty:
        return {"pooled": ols(np.array([]), np.array([])), "buckets": [],
                "by_moves": [], "points": [], "resid": {}}

    d["moneyness"] = d["strike"] / d["spot"]
    d["resid"] = d["trdprc_1"] - d["mid"]
    # Where in the quote the print landed: 0 = bid, 0.5 = mid, 1 = ask.
    d["in_spread"] = (d["trdprc_1"] - d["bid"]) / d["spread"]

    pooled = ols(d["mid"].to_numpy(float), d["trdprc_1"].to_numpy(float))

    # Price bands. This is the decisive conditioning: R^2 across the whole
    # chain is inflated by range, so the question is whether it survives when
    # the range is squeezed. Reported whatever the answer turns out to be.
    by_price = []
    for lo, hi, lab in [(0.0, 0.5, "under $0.50"), (0.5, 1.0, "$0.50-$1"),
                        (1.0, 2.0, "$1-$2"), (2.0, 4.0, "$2-$4"),
                        (4.0, 7.0, "$4-$7"), (7.0, 12.0, "$7-$12"),
                        (12.0, np.inf, "over $12")]:
        sp = d[(d["mid"] >= lo) & (d["mid"] < hi)]
        if len(sp) < 30:
            continue
        f = ols(sp["mid"].to_numpy(float), sp["trdprc_1"].to_numpy(float))
        f.update({
            "label": lab,
            "median_abs_resid": float(sp["resid"].abs().median()),
            "median_spread": float(sp["spread"].median()),
            "resid_over_spread": float((sp["resid"].abs() / sp["spread"]).median()),
            "at_mid_pct": float(100.0 * ((sp["in_spread"] - 0.5).abs() <= 0.125).mean()),
            "outside_pct": float(100.0 * ((sp["in_spread"] < 0) | (sp["in_spread"] > 1)).mean()),
        })
        by_price.append(f)

    edges = [0.0, 0.90, 0.97, 1.00, 1.03, 1.10, np.inf]
    labels = ["deep ITM <0.90", "ITM 0.90-0.97", "near ATM 0.97-1.00",
              "near ATM 1.00-1.03", "OTM 1.03-1.10", "deep OTM >1.10"]
    buckets = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        s = d[(d["moneyness"] >= lo) & (d["moneyness"] < hi)]
        if len(s) < 10:
            continue
        f = ols(s["mid"].to_numpy(float), s["trdprc_1"].to_numpy(float))
        f.update({
            "label": lab,
            "median_abs_resid": float(s["resid"].abs().median()),
            "median_spread": float(s["spread"].median()),
            "resid_over_spread": float((s["resid"].abs() / s["spread"]).median()),
            "at_mid_pct": float(100.0 * ((s["in_spread"] - 0.5).abs() <= 0.125).mean()),
            "price_range": [float(s["mid"].min()), float(s["mid"].max())],
        })
        buckets.append(f)

    by_moves = []
    for lo, hi, lab in [(1, 2, "1 trade"), (2, 6, "2-5 trades"),
                        (6, 21, "6-20 trades"), (21, np.inf, "21+ trades")]:
        s = d[(d["num_moves"] >= lo) & (d["num_moves"] < hi)]
        if len(s) < 10:
            continue
        f = ols(s["mid"].to_numpy(float), s["trdprc_1"].to_numpy(float))
        f.update({"label": lab,
                  "median_abs_resid": float(s["resid"].abs().median()),
                  "resid_over_spread": float((s["resid"].abs() / s["spread"]).median()),
                  "at_mid_pct": float(100.0 * ((s["in_spread"] - 0.5).abs() <= 0.125).mean())})
        by_moves.append(f)

    # The contracts the strategy actually wrote: just OTM, a few dollars of
    # premium. A fill assumption only has to hold where it is being used.
    w = d[(d["moneyness"] >= 1.00) & (d["moneyness"] < 1.02)
          & (d["mid"] >= 1.0) & (d["mid"] < 8.0)]
    written = ols(w["mid"].to_numpy(float), w["trdprc_1"].to_numpy(float)) if len(w) > 30 else {}
    if written:
        written.update({
            "label": "K/S in [1.00, 1.02), mid $1-$8",
            "median_abs_resid": float(w["resid"].abs().median()),
            "median_spread": float(w["spread"].median()),
            "resid_over_spread": float((w["resid"].abs() / w["spread"]).median()),
            "at_mid_pct": float(100.0 * ((w["in_spread"] - 0.5).abs() <= 0.125).mean()),
            "outside_pct": float(100.0 * ((w["in_spread"] < 0) | (w["in_spread"] > 1)).mean()),
        })

    return {
        "pooled": pooled,
        "buckets": buckets,
        "by_price": by_price,
        "written": written,
        "by_moves": by_moves,
        "resid": {
            "median_abs": float(d["resid"].abs().median()),
            "median_signed": float(d["resid"].median()),
            "median_spread": float(d["spread"].median()),
            "median_resid_over_spread": float((d["resid"].abs() / d["spread"]).median()),
            "at_mid_pct": float(100.0 * ((d["in_spread"] - 0.5).abs() <= 0.125).mean()),
            "at_bid_pct": float(100.0 * (d["in_spread"] <= 0.125).mean()),
            "at_ask_pct": float(100.0 * (d["in_spread"] >= 0.875).mean()),
            "outside_pct": float(100.0 * ((d["in_spread"] < 0) | (d["in_spread"] > 1)).mean()),
            "n": int(len(d)),
        },
        "frame": d,
    }


# --------------------------------------------------------------------------
# sweeps
# --------------------------------------------------------------------------

def _headline(run: dict, ledger: pd.DataFrame) -> dict:
    cyc = pd.DataFrame(run["cycles"])
    booked = cyc[cyc["status"].isin(["assigned", "expired"])] if len(cyc) else cyc
    prem = float(booked["premium"].sum()) if len(booked) else 0.0
    return {
        "rule": run["rule"],
        "order_hour": run["order_hour"],
        "final_nav": float(ledger["nav"].iloc[-1]),
        "pnl": float(ledger["nav"].iloc[-1] - run["start_cash"]),
        "premium_collected": prem,
        "weeks_booked": int(len(booked)),
        "weeks_skipped": int(len(cyc) - len(booked)) if len(cyc) else 0,
        "assignments": int((booked["status"] == "assigned").sum()) if len(booked) else 0,
        "min_available": float(ledger["available_funds"].min()),
        "max_nav": float(ledger["nav"].max()),
        "min_nav": float(ledger["nav"].min()),
    }


def fill_hour_sweep(stock, options, weeks, *, rule="nearest_otm",
                    start_cash=50_000.0) -> list[dict]:
    """
    The same book written at each tradeable hour of the entry day.

    This is the evidence behind the fill write-up. On Mon 2026-08-31 the mid of
    the AAPL 4-Sep 320 call went 2.35 -> 1.17 -> 1.94 inside one session. If
    the hour you happen to send the order at moves the premium by that much,
    then "fill at mid" is only half an assumption; the other half is "at which
    mid", and it deserves to be measured rather than asserted.
    """
    out = []
    for h in TRADEABLE_HOURS:
        run = run_backtest(stock, options, weeks, rule=rule,
                           order_hour=h, start_cash=start_cash)
        led = build_ledger(run, stock, options)
        out.append(_headline(run, led))
    return out


def rule_sweep(stock, options, weeks, *, order_hour=15,
               start_cash=50_000.0) -> list[dict]:
    """Every strike rule through the identical engine."""
    out = []
    for name in STRIKE_RULES:
        run = run_backtest(stock, options, weeks, rule=name,
                           order_hour=order_hour, start_cash=start_cash)
        led = build_ledger(run, stock, options)
        h = _headline(run, led)
        cyc = pd.DataFrame(run["cycles"])
        booked = cyc[cyc["status"].isin(["assigned", "expired"])] if len(cyc) else cyc
        h["median_otm_pct"] = float(booked["otm_pct"].median()) if len(booked) else np.nan
        h["median_premium"] = float(booked["mid"].median()) if len(booked) else np.nan

        meta = STRIKE_RULE_META.get(name, {})
        h["label"] = meta.get("label", name)
        h["target_prob"] = meta.get("target_prob")

        # Calibration, for the rules that claim a probability. The target is
        # RISK-NEUTRAL, so a gap against the realised rate is expected rather
        # than a defect -- the risk-neutral measure has zero drift and the tape
        # did not. Reporting both is the only way that distinction survives.
        if h["target_prob"] is not None and len(booked):
            h["realised_prob"] = float((booked["status"] == "assigned").mean())
            h["prob_gap"] = h["realised_prob"] - h["target_prob"]
        else:
            h["realised_prob"] = None
            h["prob_gap"] = None

        h["median_atm_iv"] = (float(booked["atm_iv"].median())
                              if len(booked) and "atm_iv" in booked else np.nan)
        h["iv_range"] = ([float(booked["atm_iv"].min()), float(booked["atm_iv"].max())]
                         if len(booked) and "atm_iv" in booked else None)
        out.append(h)
    return out


# --------------------------------------------------------------------------
# the benchmark
# --------------------------------------------------------------------------

def buy_and_hold(stock: pd.DataFrame, ledger: pd.DataFrame,
                 start_cash: float, shares: int = SHARES_PER_CONTRACT) -> pd.DataFrame:
    """
    100 shares bought at the first entry bar and simply held.

    The honest benchmark for a covered call is not cash -- it is the same
    stock without the cap, because the cap is the only thing the strategy
    actually did. Selling calls into a name that ran from $294 to $328 is
    where theory meets tape.
    """
    first = ledger.index[ledger["shares"] > 0]
    if len(first) == 0:
        return pd.DataFrame(columns=["ts", "nav"])
    t0 = ledger.at[first[0], "ts"]
    px0 = float(ledger.at[first[0], "stock_mark"])
    cash = start_cash - shares * px0
    sub = ledger[ledger["ts"] >= t0]
    return pd.DataFrame({
        "ts": sub["ts"].to_numpy(),
        "nav": cash + shares * sub["stock_mark"].to_numpy(float),
    })


# --------------------------------------------------------------------------
# data quality: which underlying field is safe to settle on
# --------------------------------------------------------------------------

def ohlc_integrity(stock: pd.DataFrame) -> dict:
    """
    How far the bar extremes stray outside the bar's own open/close body.

    This started as a sanity check and turned into a reason the backtest is
    built the way it is. A bar's open and close are both real, sequenced
    trades, so anything the bar genuinely traded through should sit close to
    that body. On this AAPL pull it does not: HIGH_1 runs more than 1% above
    the body on a large minority of bars and LOW_1 more than 1% below on more,
    with excursions past 10% -- odd-lot, out-of-sequence and cross prints
    surviving into the extremes.

    TRDPRC_1 shows nothing of the kind. So the last-trade series is clean and
    the extremes are not, which is exactly why entry and settlement here read
    TRDPRC_1 and never HIGH_1/LOW_1. A rule phrased as "did the stock touch
    the strike" would have booked assignments that never happened.

    HIGH_1/LOW_1 are still used in the fetcher, where they only widen the
    strike band and cost nothing but a few dead RICs.
    """
    c = pd.to_numeric(stock["TRDPRC_1"], errors="coerce").astype(float)
    o = pd.to_numeric(stock["OPEN_PRC"], errors="coerce").astype(float)
    hi = pd.to_numeric(stock["HIGH_1"], errors="coerce").astype(float)
    lo = pd.to_numeric(stock["LOW_1"], errors="coerce").astype(float)

    body_hi, body_lo = np.maximum(o, c), np.minimum(o, c)
    hi_ex = ((hi - body_hi) / body_hi).replace([np.inf, -np.inf], np.nan)
    lo_ex = ((body_lo - lo) / body_lo).replace([np.inf, -np.inf], np.nan)
    ret = c.pct_change().abs()

    def band(ex):
        return {
            "median_pct": float(100 * ex.median()),
            "p95_pct": float(100 * ex.quantile(0.95)),
            "max_pct": float(100 * ex.max()),
            "over_1pct": int((ex > 0.01).sum()),
            "over_1pct_share": float(100 * (ex > 0.01).mean()),
            "over_3pct": int((ex > 0.03).sum()),
            "over_3pct_share": float(100 * (ex > 0.03).mean()),
        }

    return {
        "bars": int(len(stock)),
        "high": band(hi_ex),
        "low": band(lo_ex),
        "close_move": {
            "median_pct": float(100 * ret.median()),
            "p99_pct": float(100 * ret.quantile(0.99)),
            "max_pct": float(100 * ret.max()),
            "over_3pct": int((ret > 0.03).sum()),
        },
    }


# --------------------------------------------------------------------------
# does the bar size change the answer?
# --------------------------------------------------------------------------

def bar_size_study(opt_h: pd.DataFrame, opt_m: pd.DataFrame) -> dict:
    """
    The hourly panel against a 1-minute pull of the same contracts.

    Three questions, and the third is the one that nearly produced a false
    claim.

    1. WHAT IS AN HOURLY BID/ASK? The whole book fills at (BID+ASK)/2, and the
       page asserts that an hourly bar's quote is the one standing at the END
       of the hour. That was a convention, not a measurement. Against the
       minute data it is exactly the last minute's quote -- checked on every
       matched contract-hour -- so the assumption is now verified rather than
       declared, and "mid at the order bar" means what it claims to mean.

    2. HOW MUCH OF THE IMPOSSIBLE PRINTING IS THE BAR? At hourly, a large
       minority of prints land outside their own bar's quote, which is not an
       arbitrage but an artefact: the quote is end-of-hour and the print is
       somewhere inside it. Re-measured on identical (contract, day) cells at
       one minute, that fraction collapses. What remains is the real rate.

    3. ARE MINUTE SPREADS TIGHTER? This looks true and is not. Conditioning on
       "this bar also printed" -- which the mid-vs-print comparison must do --
       is far more selective at one minute than at one hour, and it selects the
       liquid, tight-spread moments. Compared unconditionally on the same
       contracts and days the two distributions agree. HW1 found pooling could
       invert a spread conclusion; this is the same hazard wearing a different
       hat, and it is reported because it was nearly missed.
    """
    def _prep(o):
        d = o[o["mid"].notna() & o["spread"].notna() & (o["spread"] > 0)].copy()
        d["in_spread"] = (d["trdprc_1"] - d["bid"]) / d["spread"]
        return d

    keys = set(zip(opt_m["ric"], opt_m["date"]))
    # An explicitly-typed boolean Series, not a bare list: df[[]] is COLUMN
    # selection, so an empty mask silently returns a frame with no columns at
    # all rather than no rows, and the next line dies on a missing "mid".
    mask = pd.Series([(r, d) in keys for r, d in zip(opt_h["ric"], opt_h["date"])],
                     index=opt_h.index, dtype=bool)
    h = _prep(opt_h[mask])
    m = _prep(opt_m)

    def _side(d):
        traded = d[d["trdprc_1"].notna()]
        ins = traded["in_spread"]
        res = (traded["trdprc_1"] - traded["mid"]).abs()
        return {
            "contracts": int(d["ric"].nunique()),
            "quoted_bars": int(len(d)),
            "printed_bars": int(len(traded)),
            "printed_share": float(100 * len(traded) / max(1, len(d))),
            "median_spread_all": float(d["spread"].median()),
            "median_spread_printed": float(traded["spread"].median()),
            "median_abs_resid": float(res.median()),
            "resid_over_spread": float((res / traded["spread"]).median()),
            "at_mid_pct": float(100 * ((ins - 0.5).abs() <= 0.125).mean()),
            "outside_pct": float(100 * ((ins < 0) | (ins > 1)).mean()),
        }

    # (1) is the hourly quote the last minute's quote?
    mm = opt_m.copy()
    mm["hour"] = mm["ts"].dt.floor("h")
    agg = (mm.groupby(["ric", "hour"])
             .agg(last_bid=("bid", "last"), last_ask=("ask", "last"),
                  min_bid=("bid", "min"), max_ask=("ask", "max"),
                  n=("bid", "size")).reset_index())
    hh = opt_h[["ric", "ts", "bid", "ask"]].rename(
        columns={"ts": "hour", "bid": "h_bid", "ask": "h_ask"})
    j = agg.merge(hh, on=["ric", "hour"], how="inner").dropna(subset=["h_bid", "h_ask"])
    j = j[j["n"] >= 30]
    snapshot = {
        "matched_hours": int(len(j)),
        "bid_is_last_pct": float(100 * np.isclose(j["h_bid"], j["last_bid"], atol=1e-9).mean()),
        "ask_is_last_pct": float(100 * np.isclose(j["h_ask"], j["last_ask"], atol=1e-9).mean()),
        "bid_is_min_pct": float(100 * np.isclose(j["h_bid"], j["min_bid"], atol=1e-9).mean()),
        "ask_is_max_pct": float(100 * np.isclose(j["h_ask"], j["max_ask"], atol=1e-9).mean()),
    } if len(j) else {}

    return {"hourly": _side(h), "minute": _side(m), "snapshot": snapshot,
            "matched_cells": int(len(set(zip(h["ric"], h["date"])) & keys))}
