"""
The two-sided variance trade (config/vol_arb_ml.yaml, part_2 and extras_2026_11_15). PRIVATE: local branch
vol-arb-ml only. No I/O: every function takes load_market's market dict (or pieces of it) and returns numbers.

MEASUREMENT, from the entry session's closing quotes of the cycle's own expiry only:

    two-sided   load_market's mid present and above zero
    forward     F = K* + e^(rT) (C(K*) - P(K*)), K* the strike with both rights two-sided and the smallest |C - P|
    ATM vol     each two-sided option's IV from its own mid as wheel.option_greeks computes it (spot, r, q, T);
                at the highest strike <= F and the lowest strike > F that each have a put IV and a call IV, the
                mean of the two; sigma_ATM linear in strike between them, at F
    corridor    |ln(K/F)| <= 1.5 sigma_ATM sqrt(T)
    K0          the highest strike <= F with both rights two-sided; Q(K0) the mean of the two mids
    grid        inside the corridor: two-sided puts below K0, K0, two-sided calls above K0;
                dK = (K[i+1] - K[i-1]) / 2, the one gap at the ends
    put_var     (2/T) sum_puts dK/K^2 e^(rT) P(K) + half of K0's term
    call_var    (2/T) sum_calls dK/K^2 e^(rT) C(K) + half of K0's term - (1/T)(F/K0 - 1)^2
    skip rule   (i) >= 3 puts and >= 3 calls in the grid's wings (inside the corridor only); (ii) on each side the
                outermost two-sided strike INSIDE the corridor lies within one step of the corridor's end, the step
                its gap to the next two-sided strike inward (clarified_2026_11_15 skip_rule_ii and step: a strike
                beyond the end does not by itself satisfy it)
    parabola    sigma(x) = a + b x + c x^2, x = ln(K/F)/sqrt(T), vega-weighted least squares on the grid
    protection  (put_var + call_var - sigma_ATM^2) / (put_var + call_var)
    flat        amended_2026_11_15 rule_1: put_flat and call_flat by the same formula on the same grid (strikes,
                dK, F, K0, T, r), every option priced by Black-76 at the parity forward F, D = e^(-rT), vol
                sigma_ATM (K0's term the mean of its flat put and call); flat_baseline = ln(put_flat / call_flat),
                stored on the cycle for reporting (not a feature)

FEATURES f1 and f2 (amended_2026_11_15b), each netted by its own side's flat share:

    f1 = ln(put_var / down_semi_63) - ln(2 put_flat / (put_flat + call_flat))
    f2 = ln(call_var / up_semi_63) - ln(2 call_flat / (put_flat + call_flat))

so f1 - f2 is the raw difference less ln(put_flat / call_flat), and rule 1 is simply put_rich when f1 > f2. On a
flat smile put_var = put_flat and call_var = call_flat, so f1 = ln(m / down_semi_63) and f2 = ln(m / up_semi_63)
with m = (put_flat + call_flat) / 2, whichever side of the strip the stretch from K0 to F falls in. The raw log
ratios (un-netted) are stored on each cycle and panel row for reporting.

variance() and fit_parabola() are config/vix_split.yaml's (sb-index-etfs: lib/vix_split.py), copied, not merged;
the ATM vol here is linear in strike (vix_split's chain_atm is linear in log-moneyness) and the strip is the
corridor's, to the cycle's own expiry, with no 30-day interpolation and no zero-bid walk.

THE TRADE, the same construction for every rung: wheel.select()'s put and call closest to 0.25 |delta| (all listed
strikes), round(200 / vega per vol point per contract) contracts each (at least 1, units = the entry's lot),
filled at the entry close (mid, or the stress fill: sell at the bid, buy at the ask), hedged to -round(net delta) at
the entry close and then at each later open up to and including the expiry's, from the PREVIOUS close's deltas
(a leg with no usable mid that close keeps its last IV). Stress costs 5 bp of each hedge trade's value. Dividends on
the shares held at the prior close. Options marked at the close's mid (a missing mid carries the last), settled at
intrinsic on the expiry close, where the shares are closed too. The P&L is an overlay: cash starts at zero and earns
nothing.

    NAV = cash + option marks + shares x close
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import wheel as W
from .vol import bs_greeks, bs_price

PUT, CALL = W.PUT, W.CALL
DAYS_PER_YEAR = 365.0
SESSIONS_PER_YEAR = 252
CORRIDOR_SD = 1.5            # her choice of 2026-11-11
MIN_SIDE = 3                 # skip rule (i): two-sided strikes a side inside the corridor
STRIKE_TOL = 1e-9            # dollars: the reach rule's "at" and "no more than"
LOG_TOL = 1e-12              # log-moneyness: a strike on the corridor's edge is inside it
TARGET_DELTA = 0.25
VEGA_DOLLARS = 200.0         # per vol point per leg
CAPITAL = 100_000.0          # one book per stock
STRESS_COST = 5e-4           # 5 bp of each hedge trade's value
SEMI_N, RV_N, RET_N = 63, 21, 21
FLOOR = 1e-10                # a semivariance or rv21 below this is set to it before a ratio or log
VIX_REF = 0.20               # extras: contracts x min(1, 0.20 / VIX at entry)
FILLS = ("mid", "cross")
DIRECTIONS = {"put_rich": {PUT: -1, CALL: +1},      # sell the put, buy the call: what the label scores
              "call_rich": {PUT: +1, CALL: -1},     # buy the put, sell the call
              "put_only": {PUT: -1}}                # extras: the hedged short put alone
FEATURES = ("f1_put_premium", "f2_call_premium", "f3_protection_share", "f4_tilt", "f5_bend",
            "f6_iv_over_rv", "f7_earnings_in_cycle", "f8_vix", "f9_ret21")
RAW = ("f1_put_premium_raw", "f2_call_premium_raw")   # amended_2026_11_15b: f1 and f2 before netting, reported only
SKIPS = ("no_forward", "no_atm", "few_strikes", "short_reach")
NOT_FINITE = "features_not_finite"     # clarified not_finite: skipped for every rung, rule 0 and the extras too
LEG_MISSING = "leg_missing"            # part_2 no_trade: a leg missing at entry
SPLIT = ("gamma", "theta", "vega", "delta_hedge", "costs", "residual")


# --------------------------------------------------------------------------
# the chain
# --------------------------------------------------------------------------

def two_sided(quotes: pd.DataFrame, expiry=None) -> pd.DataFrame:
    """
    The two-sided quotes (mid present and above zero) of one session, of `expiry` if given. A strike quoted
    twice on one right (two RICs) keeps the alphabetically first RIC's quote, so the result never depends on
    row order.
    """
    c = quotes
    if expiry is not None:
        c = c[c["expiry"] == pd.Timestamp(expiry)]
    c = c[c["mid"].notna() & (c["mid"] > 0)]
    return c.sort_values(["cp", "strike", "ric"]).drop_duplicates(["cp", "strike"], keep="first")


def sides(chain: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(put mids, call mids), each indexed by strike, ascending."""
    put = chain[chain["cp"] == PUT].set_index("strike")["mid"].astype(float).sort_index()
    call = chain[chain["cp"] == CALL].set_index("strike")["mid"].astype(float).sort_index()
    return put, call


def parity_forward(put: pd.Series, call: pd.Series, r: float, T: float) -> tuple[float, float] | None:
    """F = K* + e^(rT)(C(K*) - P(K*)) and K*; None when no strike has both rights. A tie in |C - P| goes to
    the lower strike."""
    both = put.index.intersection(call.index).sort_values()
    if len(both) == 0:
        return None
    diff = call.loc[both] - put.loc[both]
    kstar = float(diff.abs().idxmin())
    return float(kstar + np.exp(r * T) * float(diff.loc[kstar])), kstar


def chain_ivs(mids: pd.Series, cp: str, spot: float, r: float, q: float, T: float) -> pd.Series:
    """Each option's implied vol from its own mid, as wheel.option_greeks computes it (NaN when it has none)."""
    return pd.Series([W.option_greeks(float(m), spot, float(k), T, r, q, cp)["iv"] for k, m in mids.items()],
                     index=mids.index, dtype=float)


def atm_vol(put_iv: pd.Series, call_iv: pd.Series, F: float) -> dict | None:
    """
    sigma_ATM: at the highest strike at or below F and the lowest above F that each have both a put and a call
    IV, the mean of the two; linear in strike between them, at F. None when F is not bracketed.
    """
    both = [k for k in put_iv.index.intersection(call_iv.index)
            if np.isfinite(put_iv.loc[k]) and np.isfinite(call_iv.loc[k])]
    lo = [k for k in both if k <= F]
    hi = [k for k in both if k > F]
    if not lo or not hi:
        return None
    k_lo, k_hi = max(lo), min(hi)
    v_lo = (put_iv.loc[k_lo] + call_iv.loc[k_lo]) / 2.0
    v_hi = (put_iv.loc[k_hi] + call_iv.loc[k_hi]) / 2.0
    sigma = v_lo + (v_hi - v_lo) * (F - k_lo) / (k_hi - k_lo)
    return {"sigma": float(sigma), "k_lo": float(k_lo), "k_hi": float(k_hi), "iv_lo": float(v_lo),
            "iv_hi": float(v_hi)}


def corridor(F: float, sigma: float, T: float, width: float = CORRIDOR_SD) -> dict:
    """|ln(K/F)| <= width x sigma x sqrt(T): the half-width w in log-moneyness and the two ends in dollars."""
    w = width * sigma * np.sqrt(T)
    return {"w": float(w), "lo": float(F * np.exp(-w)), "hi": float(F * np.exp(w))}


def inside(K, F: float, w: float) -> np.ndarray:
    return np.abs(np.log(np.asarray(K, float) / F)) <= w + LOG_TOL


def strike_k0(put: pd.Series, call: pd.Series, F: float) -> float | None:
    """The highest strike at or below F with both a put and a call two-sided."""
    both = [k for k in put.index.intersection(call.index) if k <= F]
    return float(max(both)) if both else None


def reaches(strikes, end: float, side: str) -> dict:
    """
    Skip rule (ii) on one side (clarified_2026_11_15 skip_rule_ii and step): `strikes` are that side's two-sided
    strikes INSIDE the corridor (the strip itself; sorted or not); the outermost must lie within one step of the
    corridor's `end`, the step its gap to the next of them inward. measure() passes the in-corridor strikes only,
    so a two-sided strike beyond the end never counts. Puts reach down (side "P"), calls up ("C").
    """
    ks = np.sort(np.asarray(strikes, float))
    if len(ks) == 0:
        return {"ok": False, "outer": np.nan, "short": np.nan, "step": np.nan}
    if side == PUT:
        outer, nxt = ks[0], (ks[1] if len(ks) > 1 else np.nan)
        short, step = outer - end, nxt - outer
    else:
        outer, nxt = ks[-1], (ks[-2] if len(ks) > 1 else np.nan)
        short, step = end - outer, outer - nxt
    ok = bool(short <= STRIKE_TOL or (np.isfinite(step) and short <= step + STRIKE_TOL))
    return {"ok": ok, "outer": float(outer), "short": float(short), "step": float(step)}


def variance(K, Q, side, F: float, K0: float, T: float, R: float) -> dict:
    """
    lib/vix_split.py's variance(), unchanged: sigma^2 = (2/T) sum(dK / K^2 e^(RT) Q(K)) - (1/T)(F/K0 - 1)^2 over
    the strip, and its put and call halves: K0's term split evenly between them, the correction to the call half.
    """
    K, Q, side = np.asarray(K, float), np.asarray(Q, float), np.asarray(side)
    dK = np.empty_like(K)
    dK[1:-1] = (K[2:] - K[:-2]) / 2.0
    dK[0], dK[-1] = K[1] - K[0], K[-1] - K[-2]
    contrib = 2.0 / T * dK / K ** 2 * np.exp(R * T) * Q
    correction = (F / K0 - 1.0) ** 2 / T
    at_k0 = contrib[side == "K0"].sum() / 2.0
    return {"sigma2": float(contrib.sum() - correction),
            "put_var": float(contrib[side == "P"].sum() + at_k0),
            "call_var": float(contrib[side == "C"].sum() + at_k0 - correction),
            "correction": float(correction), "dK": dK}


def leg_vega(F: float, K: float, T: float, iv: float, D: float) -> float:
    """Black-76 vega per share per 1.00 of vol (the same for a put and a call)."""
    return bs_greeks(F, K, T, iv, D, CALL)["vega"] if np.isfinite(iv) else np.nan


def flat_prices(K, side, F: float, T: float, r: float, sigma: float) -> np.ndarray:
    """
    amended rule_1 and clarified rule_1_flat_prices: each grid strike's option priced by Black-76 at the parity
    forward F, D = e^(-rT), at the flat vol sigma_ATM; puts below K0, calls above, K0 the mean of its put and call.
    """
    D = np.exp(-r * T)
    out = []
    for k, s in zip(np.asarray(K, float), np.asarray(side)):
        put, call = bs_price(F, k, T, sigma, D, PUT), bs_price(F, k, T, sigma, D, CALL)
        out.append(put if s == "P" else call if s == "C" else (put + call) / 2.0)
    return np.array(out, float)


def flat_baseline(K, side, F: float, K0: float, T: float, r: float, sigma: float) -> dict:
    """put_flat and call_flat: variance() on the measured grid at the flat prices; baseline ln(put_flat / call_flat)."""
    v = variance(K, flat_prices(K, side, F, T, r, sigma), side, F, K0, T, r)
    with np.errstate(divide="ignore", invalid="ignore"):
        base = float(np.log(v["put_var"] / v["call_var"])) if v["call_var"] > 0 and v["put_var"] > 0 else np.nan
    return {"put_flat": v["put_var"], "call_flat": v["call_var"], "baseline": base}


def fit_parabola(K, iv, F: float, T: float, vega) -> dict | None:
    """
    lib/vix_split.py's fit_parabola() on given strikes: sigma(x) = a + b x + c x^2, x = ln(K/F)/sqrt(T), weighted
    least squares with Black-76 vega as the weight (np.polyfit's w is sqrt(vega), so the squared residuals are
    weighted by vega). None with fewer than 3 usable points.
    """
    K, iv, vega = (np.asarray(a, float) for a in (K, iv, vega))
    x = np.log(K / F) / np.sqrt(T)
    ok = np.isfinite(iv) & np.isfinite(vega) & (vega > 0)
    if ok.sum() < 3:
        return None
    c, b, a = np.polyfit(x[ok], iv[ok], 2, w=np.sqrt(vega[ok]))
    resid = iv[ok] - (a + b * x[ok] + c * x[ok] ** 2)
    return {"a": float(a), "b": float(b), "c": float(c), "n_fit": int(ok.sum()),
            "rmse": float(np.sqrt(np.mean(resid ** 2)))}


def protection_share(put_var: float, call_var: float, sigma_atm: float) -> float:
    """Method 1 on the corridor: the part of the strip's variance above sigma_ATM^2, as a share of it."""
    total = put_var + call_var
    return float((total - sigma_atm ** 2) / total) if total != 0 else np.nan


def measure(chain: pd.DataFrame, spot: float, r: float, q: float, T: float,
            width: float = CORRIDOR_SD) -> dict:
    """
    Everything part_2's measurement asks of one stock-cycle, from its expiry's two-sided quotes at the entry
    close (two_sided()'s output: cp, strike, mid). `status` is "measured" or the first skip reason that applies,
    in the order no_forward, no_atm, few_strikes, short_reach. The parabola is fitted whenever the grid exists
    (also on a skipped cycle, for the smile charts); the side variances and the flat baseline only on a measured one.
    """
    out = {"status": None, "T": float(T), "r": float(r), "q": float(q), "spot": float(spot)}
    put, call = sides(chain)
    fwd = parity_forward(put, call, r, T)
    if fwd is None:
        return {**out, "status": "no_forward"}
    F, kstar = fwd
    out.update(F=F, kstar=kstar)
    piv, civ = chain_ivs(put, PUT, spot, r, q, T), chain_ivs(call, CALL, spot, r, q, T)
    atm = atm_vol(piv, civ, F)
    if atm is None:
        return {**out, "status": "no_atm"}
    sigma = atm["sigma"]
    cor = corridor(F, sigma, T, width)
    K0 = strike_k0(put, call, F)           # exists: the ATM bracket's lower strike has both rights two-sided
    out.update(sigma_atm=sigma, atm=atm, corridor=cor, K0=K0)
    pk = [float(k) for k in put.index if k < K0 and inside(k, F, cor["w"])]
    ck = [float(k) for k in call.index if k > K0 and inside(k, F, cor["w"])]
    K = np.array(pk + [K0] + ck)
    side = np.array(["P"] * len(pk) + ["K0"] + ["C"] * len(ck))
    Q = np.array([put.loc[k] for k in pk] + [(put.loc[K0] + call.loc[K0]) / 2.0] + [call.loc[k] for k in ck])
    iv = np.array([piv.loc[k] for k in pk] + [np.nanmean([piv.loc[K0], civ.loc[K0]])
                                               if np.isfinite([piv.loc[K0], civ.loc[K0]]).any() else np.nan]
                  + [civ.loc[k] for k in ck])
    Fc, D = spot * np.exp((r - q) * T), np.exp(-r * T)      # the forward and discount option_greeks priced at
    vega = np.array([leg_vega(Fc, k, T, v, D) for k, v in zip(K, iv)])
    out.update(n_put=len(pk), n_call=len(ck), grid={"K": K, "Q": Q, "side": side, "iv": iv, "vega": vega})
    out["parabola"] = fit_parabola(K, iv, F, T, vega)
    put_reach = reaches(pk, cor["lo"], PUT)            # the in-corridor strip only (clarified skip_rule_ii)
    call_reach = reaches(ck, cor["hi"], CALL)
    out.update(put_reach=put_reach, call_reach=call_reach)
    if len(pk) < MIN_SIDE or len(ck) < MIN_SIDE:
        return {**out, "status": "few_strikes"}
    if not (put_reach["ok"] and call_reach["ok"]):
        return {**out, "status": "short_reach"}
    v = variance(K, Q, side, F, K0, T, r)
    flat = flat_baseline(K, side, F, K0, T, r, sigma)
    out.update(put_var=v["put_var"], call_var=v["call_var"], correction=v["correction"], dK=v["dK"],
               protection_share=protection_share(v["put_var"], v["call_var"], sigma), put_flat=flat["put_flat"],
               call_flat=flat["call_flat"], flat_baseline=flat["baseline"], status="measured")
    return out


# --------------------------------------------------------------------------
# realized, trailing
# --------------------------------------------------------------------------

def session_dividends(index: pd.DatetimeIndex, dividends: pd.Series) -> pd.Series:
    """Each session's dividend per share: the dividends whose ex-date is that session, summed (as wheel.run)."""
    d = dividends.groupby(level=0).sum() if len(dividends) else pd.Series(dtype=float)
    return d.reindex(index).fillna(0.0).astype(float)


def realized(closes: pd.Series, dividends: pd.Series) -> pd.DataFrame:
    """
    Per session, trailing: ret = ln((P_t + D_t) / P_t-1) (D_t the dividend going ex on t), zero mean;
    down_semi_63 / up_semi_63 the sums of the squared returns below / above zero over the 63 returns up to and
    including the session's, x 252/63; rv21 the sum of the 21 squared returns up to it, x 252/21; ret21 the log
    change of the close over 21 sessions (no dividend). Decimal units.
    """
    c = pd.to_numeric(closes, errors="coerce").astype(float)
    d = session_dividends(c.index, dividends)
    ret = np.log((c + d) / c.shift(1))
    sq = ret ** 2
    down = sq.where(ret < 0, 0.0).where(ret.notna())
    up = sq.where(ret > 0, 0.0).where(ret.notna())
    return pd.DataFrame({
        "ret": ret,
        "down_semi_63": down.rolling(SEMI_N, min_periods=SEMI_N).sum() * SESSIONS_PER_YEAR / SEMI_N,
        "up_semi_63": up.rolling(SEMI_N, min_periods=SEMI_N).sum() * SESSIONS_PER_YEAR / SEMI_N,
        "rv_63": sq.rolling(SEMI_N, min_periods=SEMI_N).sum() * SESSIONS_PER_YEAR / SEMI_N,
        "rv21": sq.rolling(RV_N, min_periods=RV_N).sum() * SESSIONS_PER_YEAR / RV_N,
        "ret21": np.log(c / c.shift(RET_N)),
    }, index=c.index)


def _floor(x) -> float:
    """part_2 floors: a semivariance or rv21 below 1e-10 is set to 1e-10 (NaN stays NaN)."""
    return max(float(x), FLOOR) if np.isfinite(x) else np.nan


def premiums(m: dict, real: pd.Series) -> dict:
    """
    amended_2026_11_15b's f1 and f2 and their raw (un-netted) log ratios, for a measured cycle `m` (put_var,
    call_var, put_flat, call_flat) and realized()'s row at the entry:

        f1_put_premium_raw  = ln(put_var / down_semi_63)
        f2_call_premium_raw = ln(call_var / up_semi_63)
        f1_put_premium      = f1_put_premium_raw - ln(2 put_flat / (put_flat + call_flat))
        f2_call_premium     = f2_call_premium_raw - ln(2 call_flat / (put_flat + call_flat))

    A flat share that is not a positive number gives a NaN feature (then the cycle is features_not_finite).
    """
    pf, cf = np.float64(m["put_flat"]), np.float64(m["call_flat"])
    with np.errstate(divide="ignore", invalid="ignore"):
        raw1 = float(np.log(m["put_var"] / _floor(real["down_semi_63"])))
        raw2 = float(np.log(m["call_var"] / _floor(real["up_semi_63"])))
        net1 = float(np.log(2.0 * pf / (pf + cf)))
        net2 = float(np.log(2.0 * cf / (pf + cf)))
    return {"f1_put_premium_raw": raw1, "f2_call_premium_raw": raw2,
            "f1_put_premium": raw1 - net1, "f2_call_premium": raw2 - net2}


def features(m: dict, real: pd.Series, vix_close: float, earnings_in_cycle: bool) -> dict:
    """The nine features of a measured cycle; `real` is realized()'s row at the entry, `vix_close` Cboe's close.
    f1 and f2 are premiums()'s, netted by each side's flat share (amended_2026_11_15b)."""
    par = m.get("parabola") or {}
    prem = premiums(m, real)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "f1_put_premium": prem["f1_put_premium"],
            "f2_call_premium": prem["f2_call_premium"],
            "f3_protection_share": float(m["protection_share"]),
            "f4_tilt": float(par.get("b", np.nan)),
            "f5_bend": float(par.get("c", np.nan)),
            "f6_iv_over_rv": float(m["sigma_atm"] / np.sqrt(_floor(real["rv21"]))),
            "f7_earnings_in_cycle": 1.0 if earnings_in_cycle else 0.0,
            "f8_vix": float(vix_close) / 100.0,
            "f9_ret21": float(real["ret21"]),
        }


def earnings_between(earnings, entry, expiry) -> bool:
    e = pd.DatetimeIndex(earnings)
    return bool(((e > pd.Timestamp(entry)) & (e <= pd.Timestamp(expiry))).any())


# --------------------------------------------------------------------------
# the trade
# --------------------------------------------------------------------------

def contracts_for(vega_per_share: float, lot: int, dollars: float = VEGA_DOLLARS) -> int:
    """round(dollars / (vega per 1 vol point per contract)), at least 1; a contract is `lot` units."""
    per_contract = vega_per_share / 100.0 * lot
    return max(1, int(round(dollars / per_contract)))


def vix_scaled(contracts: int, vix: float, ref: float = VIX_REF) -> int:
    """extras vix_scaled_size: round(contracts x min(1, ref / VIX)), at least 1; `vix` in decimal (f8)."""
    return max(1, int(round(contracts * min(1.0, ref / vix))))


def greeks_at(spot: float, K: float, T: float, iv: float, r: float, q: float, cp: str) -> dict:
    """
    One leg's Black-76 greeks per share at a close, in spot terms as wheel.option_greeks turns its delta:
    F = S e^((r-q)T), D = e^(-rT); delta and gamma to spot, theta per year (F and D fixed), vega per 1.00.
    """
    nan = {"delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan}
    if not (np.isfinite(iv) and T > 0 and spot > 0):
        return nan
    F, D = spot * np.exp((r - q) * T), np.exp(-r * T)
    g = bs_greeks(F, K, T, iv, D, cp)
    return {"delta": g["delta"] * F / spot, "gamma": g["gamma"] * (F / spot) ** 2, "theta": g["theta"],
            "vega": g["vega"]}


def legs_at_entry(market: dict, day_quotes: pd.DataFrame, entry, expiry) -> dict:
    """wheel.select()'s put and call closest to 0.25 |delta| on the entry close, each with its contracts."""
    entry, expiry = pd.Timestamp(entry), pd.Timestamp(expiry)
    spot = float(market["stock"].at[entry, "close"])
    r, q = W.carry(market, entry)
    T = (expiry - entry).days / DAYS_PER_YEAR
    lot = int(market["lot"].get(entry, W.CONTRACT)) if "lot" in market else W.CONTRACT
    out = {"lot": lot}
    for cp in (PUT, CALL):
        pick = W.select(day_quotes, cp, expiry, spot, entry, r, q, TARGET_DELTA)
        if pick is None:
            out[cp] = None
            continue
        F, D = spot * np.exp((r - q) * T), np.exp(-r * T)
        vega = leg_vega(F, float(pick["strike"]), T, float(pick["iv"]), D)
        out[cp] = {"ric": pick["ric"], "strike": float(pick["strike"]), "mid": float(pick["mid"]),
                   "bid": W._num(pick["bid"]), "ask": W._num(pick["ask"]), "iv": float(pick["iv"]),
                   "delta": float(pick["delta"]), "vega": float(vega), "contracts": contracts_for(vega, lot)}
    return out


def leg_path(market: dict, mids: pd.Series, leg: dict, cp: str, days: pd.DatetimeIndex, expiry) -> dict:
    """
    One leg over the cycle's sessions: the mark (the close's mid, a missing one carrying the last), the IV (its
    own mid's, or the last when that close has no usable mid), and the greeks at each close before expiry from
    that IV at that close's spot and T. On the entry close the IV and delta are select()'s, from option_greeks.
    """
    n = len(days)
    mark, iv = np.full(n, np.nan), np.full(n, np.nan)
    g = {k: np.full(n, np.nan) for k in ("delta", "gamma", "theta", "vega")}
    K = leg["strike"]
    last_mark, last_iv = leg["mid"], leg["iv"]
    for i, d in enumerate(days):
        spot = float(market["stock"].at[d, "close"])
        r, q = W.carry(market, d)
        T = (pd.Timestamp(expiry) - d).days / DAYS_PER_YEAR
        m = float(mids.get(d, np.nan)) if i else leg["mid"]
        if np.isfinite(m) and m > 0:
            last_mark = m
            own = W.option_greeks(m, spot, K, T, r, q, cp)["iv"] if i else leg["iv"]
            if np.isfinite(own):
                last_iv = own
        mark[i], iv[i] = last_mark, last_iv
        if T > 0:
            for k, v in greeks_at(spot, K, T, last_iv, r, q, cp).items():
                g[k][i] = v
    if n:
        g["delta"][0] = leg["delta"]           # select()'s, from wheel.option_greeks on the entry mid
    return {"mark": mark, "iv": iv, **g}


def cycle_path(market: dict, quotes_of_expiry: pd.DataFrame, legs: dict, entry, expiry) -> dict:
    """The sessions entry..expiry with closes, opens, dividends and each present leg's path."""
    entry, expiry = pd.Timestamp(entry), pd.Timestamp(expiry)
    idx = market["stock"].index
    days = idx[(idx >= entry) & (idx <= expiry)]
    if len(days) == 0 or days[0] != entry or days[-1] != expiry:
        raise ValueError(f"the stock has no session on {entry.date()} or {expiry.date()}")
    st = market["stock"].loc[days]
    if st["open"].iloc[1:].isna().any():
        raise ValueError(f"an open is missing between {entry.date()} and {expiry.date()}")
    out = {"days": days, "close": st["close"].to_numpy(float), "open": st["open"].to_numpy(float),
           "div": session_dividends(days, market["dividends"]).to_numpy(float), "lot": legs["lot"],
           "elapsed": np.r_[0, np.diff(days.values).astype("timedelta64[D]").astype(int)], "legs": {}}
    for cp in (PUT, CALL):
        leg = legs.get(cp)
        if leg is None:
            continue
        rows = quotes_of_expiry[quotes_of_expiry["ric"] == leg["ric"]]
        mids = rows.groupby("date")["mid"].first()
        p = leg_path(market, mids, leg, cp, days, expiry)
        K, S_T = leg["strike"], out["close"][-1]
        p.update(strike=K, intrinsic=max(K - S_T, 0.0) if cp == PUT else max(S_T - K, 0.0),
                 mid0=leg["mid"], bid0=leg["bid"], ask0=leg["ask"])
        out["legs"][cp] = p
    return out


def run_trade(path: dict, signs: dict, contracts: dict, fill: str = "mid") -> dict:
    """
    Book one cycle's trade on a cycle_path(): `signs` {cp: +1 long / -1 short} for the legs traded, `contracts`
    {cp: n}. Returns the daily ledger (cash, option marks, shares, close, NAV, the day's P&L and its split) and
    the totals. The split (extras pnl_split), each day from the previous close's greeks:

        gamma        sum_legs 1/2 Gamma (S_t - S_t-1)^2
        theta        sum_legs Theta x calendar days elapsed / 365
        vega         sum_legs Vega x (IV_t - IV_t-1), each leg's own IV (carried when it has none that close)
        delta_hedge  sum_legs Delta (S_t - S_t-1) + the shares' P&L that day (trades at the open, dividends)
        costs        the stress fill's: entry legs' distance from mid, and 5 bp of each hedge trade
        residual     the day's P&L less the five above
    """
    if fill not in FILLS:
        raise ValueError(f"fill is 'mid' or 'cross', not {fill!r}")
    stress = fill == "cross"
    lot, days = int(path["lot"]), path["days"]
    close, open_, div, elapsed = path["close"], path["open"], path["div"], path["elapsed"]
    missing = [cp for cp in signs if cp not in path["legs"]]
    if missing:
        raise ValueError(f"traded legs {missing} have no path")
    legs = [(cp, path["legs"][cp], int(signs[cp]) * int(contracts[cp]) * lot) for cp in (PUT, CALL) if cp in signs]
    last = len(days) - 1
    rows, cash, shares, trades, cost_total = [], 0.0, 0, 0, 0.0

    def net_delta(i):
        return sum(n * p["delta"][i] for _, p, n in legs)

    # ---- the entry close
    spread = 0.0
    for cp, p, n in legs:
        px = p["mid0"] if not stress else (p["bid0"] if n < 0 else p["ask0"])
        cash -= n * px
        spread += n * (p["mid0"] - px)
    shares = -int(round(net_delta(0)))
    cash -= shares * close[0]
    cost = STRESS_COST * abs(shares) * close[0] if stress else 0.0
    cash -= cost
    cost_total += cost
    trades += int(shares != 0)
    option_mv = sum(n * p["mark"][0] for _, p, n in legs)
    nav = cash + option_mv + shares * close[0]
    costs0 = spread - cost
    rows.append({"date": days[0], "cash": cash, "option_mv": option_mv, "shares": shares, "close": close[0],
                 "nav": nav, "pnl": nav, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "delta_hedge": 0.0,
                 "costs": costs0, "residual": nav - costs0, "dividend": 0.0, "hedge_trade": shares})
    spread_total = spread

    # ---- each later session: dividend on the prior close's shares, rebalance at the open, mark or settle
    for i in range(1, last + 1):
        prev_nav, held = nav, shares
        dividend = held * div[i]
        cash += dividend
        target = -int(round(net_delta(i - 1)))
        move = target - shares
        cash -= move * open_[i]
        cost = STRESS_COST * abs(move) * open_[i] if stress else 0.0
        trades += int(move != 0)
        shares = target
        if i == last:                                    # expiry: intrinsic, and the shares closed at the close
            cash += sum(n * p["intrinsic"] for _, p, n in legs)
            option_mv = 0.0
            cash += shares * close[i]
            out_cost = STRESS_COST * abs(shares) * close[i] if stress else 0.0
            cost += out_cost
            trades += int(shares != 0)
            closed = shares
            shares = 0
        else:
            option_mv = sum(n * p["mark"][i] for _, p, n in legs)
            closed = 0
        cash -= cost
        cost_total += cost
        nav = cash + option_mv + shares * close[i]
        pnl = nav - prev_nav
        dS = close[i] - close[i - 1]
        share_pnl = held * (open_[i] - close[i - 1]) + (closed if i == last else shares) * (close[i] - open_[i]) \
            + dividend
        part = {
            "gamma": sum(n * 0.5 * p["gamma"][i - 1] * dS * dS for _, p, n in legs),
            "theta": sum(n * p["theta"][i - 1] * elapsed[i] / DAYS_PER_YEAR for _, p, n in legs),
            "vega": sum(n * p["vega"][i - 1] * ((p["iv"][i] if i < last else p["iv"][i - 1]) - p["iv"][i - 1])
                        for _, p, n in legs),
            "delta_hedge": sum(n * p["delta"][i - 1] * dS for _, p, n in legs) + share_pnl,
            "costs": -cost,
        }
        part["residual"] = pnl - sum(part.values())
        rows.append({"date": days[i], "cash": cash, "option_mv": option_mv, "shares": shares, "close": close[i],
                     "nav": nav, "pnl": pnl, **part, "dividend": dividend, "hedge_trade": move})
    ledger = pd.DataFrame(rows)
    return {"ledger": ledger, "pnl": float(nav), "split": {k: float(ledger[k].sum()) for k in SPLIT},
            "contracts": {cp: int(contracts[cp]) for cp in signs}, "units": {cp: n for cp, _, n in legs},
            "entry_shares": int(rows[0]["shares"]), "hedge_trades": trades, "hedge_costs": float(cost_total),
            "spread_cost": float(-spread_total), "dividends": float(ledger["dividend"].sum()),
            "fill": fill}


def monthly_pnl(ledger: pd.DataFrame) -> dict:
    """Each calendar month's P&L from month-end marks: the month's last NAV less the last before it (0 before
    entry)."""
    nav = ledger.set_index("date")["nav"]
    me = nav.groupby(nav.index.to_period("M")).last()
    pnl = me.diff()
    pnl.iloc[0] = me.iloc[0]
    return {str(k): float(v) for k, v in pnl.items()}


def summarise(trade: dict) -> dict:
    """What a cycle record keeps of one trade."""
    return {"pnl": trade["pnl"], "monthly": monthly_pnl(trade["ledger"]), "split": trade["split"],
            "contracts": trade["contracts"], "entry_shares": trade["entry_shares"],
            "hedge_trades": trade["hedge_trades"], "hedge_costs": trade["hedge_costs"],
            "spread_cost": trade["spread_cost"], "dividends": trade["dividends"]}


# --------------------------------------------------------------------------
# one stock over a window
# --------------------------------------------------------------------------

def _scalars(m: dict) -> dict:
    """The measurement's figures for the output (no arrays)."""
    keep = ("F", "kstar", "sigma_atm", "K0", "n_put", "n_call", "put_var", "call_var", "protection_share", "put_flat",
            "call_flat", "flat_baseline", "T", "r", "q", "spot")
    out = {k: m[k] for k in keep if k in m}
    if "corridor" in m:
        out["corridor_lo"], out["corridor_hi"] = m["corridor"]["lo"], m["corridor"]["hi"]
    for side in ("put_reach", "call_reach"):
        if side in m:
            out[side] = m[side]
    if m.get("parabola"):
        out.update({k: m["parabola"][k] for k in ("a", "b", "c", "n_fit", "rmse")})
    return out


def stock_cycles(market: dict, start, end, vix: pd.Series, ticker: str = "") -> list[dict]:
    """
    Every cycle of [start, end] (wheel.entry_sessions): its measurement, status, features (f1 and f2 netted), flat
    baseline and raw f1 and f2 (both reported only) and label, and each trade the stages and the extras score, at
    both fills: put_rich, call_rich and put_only at the registered contracts, and put_rich and call_rich VIX-scaled.
    A measured cycle whose nine
    features are not all finite becomes features_not_finite: no legs, no trades, no label, in no rung (clarified
    not_finite). put_only is booked whenever the put leg exists, also with the call missing (clarified put_leg_only).
    """
    entries = W.entry_sessions(market["stock"].index, start, end)
    quotes = market["quotes"]
    by_exp = {pd.Timestamp(e): g for e, g in quotes[quotes["expiry"].isin(list(entries.values()))].groupby("expiry")}
    real = realized(market["stock"]["close"], market["dividends"])
    vix = pd.Series(vix, dtype=float)
    out = []
    for entry, expiry in entries.items():
        q_exp = by_exp.get(pd.Timestamp(expiry), quotes.iloc[:0])
        day_q = q_exp[q_exp["date"] == entry]
        spot = float(market["stock"].at[entry, "close"])
        r, q = W.carry(market, entry)
        T = (expiry - entry).days / DAYS_PER_YEAR
        m = measure(two_sided(day_q), spot, r, q, T)
        rec = {"stock": ticker, "entry": entry, "expiry": expiry, "status": m["status"], "measure": _scalars(m),
               "features": None, "flat_baseline": None, "raw_premiums": None, "label": None, "no_trade": None,
               "legs": None, "missing_legs": None, "trades": {}}
        if m["status"] == "measured":
            f = features(m, real.loc[entry], float(vix.get(entry, np.nan)),
                         earnings_between(market["earnings"], entry, expiry))
            prem = premiums(m, real.loc[entry])
            rec["features"], rec["flat_baseline"] = f, m["flat_baseline"]
            rec["raw_premiums"] = {k: prem[k] for k in RAW}
            if not all(np.isfinite(list(f.values()))):
                rec["status"] = NOT_FINITE
            elif not np.isfinite(m["flat_baseline"]):
                # finite netted f1 and f2 make put_flat and call_flat of one sign, so ln(put_flat / call_flat) is
                # finite too: a NaN here means the stored baseline disagrees with the flat prices f1 and f2 used
                raise ValueError(f"{ticker} {pd.Timestamp(entry).date()}: the nine features are finite but the "
                                 "flat baseline ln(put_flat / call_flat) is not; the stored record would contradict "
                                 "the netted f1 and f2")
        if rec["status"] == "measured":
            legs = legs_at_entry(market, day_q, entry, expiry)
            rec["legs"] = {cp: (None if legs[cp] is None else {k: v for k, v in legs[cp].items()})
                           for cp in (PUT, CALL)}
            rec["lot"] = legs["lot"]
            rec["missing_legs"] = [cp for cp in (PUT, CALL) if legs[cp] is None]
            if legs[PUT] is None and legs[CALL] is None:
                rec["no_trade"] = LEG_MISSING
            else:
                path = cycle_path(market, q_exp, legs, entry, expiry)
                base = {cp: legs[cp]["contracts"] for cp in (PUT, CALL) if legs[cp] is not None}
                vix_dec = rec["features"]["f8_vix"]
                scaled = {cp: vix_scaled(n, vix_dec) for cp, n in base.items()}
                variants = {}
                if legs[PUT] is not None:
                    variants["put_only"] = (DIRECTIONS["put_only"], base)
                if legs[PUT] is not None and legs[CALL] is not None:
                    for d in ("put_rich", "call_rich"):
                        variants[d] = (DIRECTIONS[d], base)
                        variants[d + "_vix"] = (DIRECTIONS[d], scaled)
                else:
                    rec["no_trade"] = LEG_MISSING
                for name, (signs, n) in variants.items():
                    rec["trades"][name] = {fill: summarise(run_trade(path, signs, n, fill)) for fill in FILLS}
                if "put_rich" in rec["trades"]:
                    rec["label"] = int(rec["trades"]["put_rich"]["mid"]["pnl"] > 0)
        out.append(rec)
    return out


def smile_chart(cycles: list[dict]) -> list[dict]:
    """extras smile_charts: per stock and entry, the parabola's a, b, c and sigma_ATM (NaN where not reached)."""
    return [{"stock": c["stock"], "entry": c["entry"], "expiry": c["expiry"], "status": c["status"],
             **{k: c["measure"].get(k, np.nan) for k in ("a", "b", "c", "sigma_atm")}} for c in cycles]
