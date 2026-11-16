"""
The ladder, the purged walk-forward and the bar of config/vol_arb_ml.yaml's part_2, on lib/vol_arb.py's cycle
records. PRIVATE: local branch vol-arb-ml only. No I/O.

    rule_0     put_rich every measured cycle (the baseline; never the model sent to the test)
    rule_1     put_rich when f1 > f2, call_rich otherwise (a tie too), on f1 and f2 netted by each side's flat share
               (amended_2026_11_15b): the same decisions as amended_2026_11_15's f1 - f2 > ln(put_flat / call_flat) on
               the raw log ratios, since the netted f1 - f2 is the raw difference less that baseline
    logistic   LogisticRegression(l2, C 1.0, lbfgs, max_iter 1000) on the nine features standardised by a
               StandardScaler fit on the training rows only
    forest     RandomForestClassifier(500 trees, max_depth 3, min_samples_leaf 20, max_features sqrt,
               random_state 2026, n_jobs 1)
    a model trades put_rich when p = P(label 1) > 0.55, call_rich when p < 0.45, nothing between; a training set
    with one label gives p = that label's share

At each entry date t, in order, a model is fit on every row whose expiry is before t (strictly) and predicts every
row entered at t; predictions start at the first entry date with at least 24 earlier entry months of finished
rows, and every rung is scored over the same calendar months from then. A measured row with no label (a leg
missing) is predicted and decided but never trained on, and counts toward the warm-up's 24 entry months. A book's
month is its P&L over (stocks x $100,000) from month-end marks, 0 with no trade; the Sharpe ratio is mean / sd
(ddof 1) x sqrt(12), NaN with a zero sd.

The models and agreement() read the netted f1 and f2 too; the raw (un-netted) f1 and f2 and the flat baseline ride
on each panel row for reporting and feed nothing.

The extras of extras_2026_11_15 (extras()) are reported only: they read copies of what the stage decided and
nothing they compute goes back into the walk-forward, the choice for the test or the pass bar.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import vol_arb as VA

RUNGS = ("rule_0", "rule_1", "logistic", "forest")
MODELS = ("logistic", "forest")
CANDIDATES = ("rule_1", "logistic", "forest")      # rule 0 is never the model sent to the test
PUT_RICH, CALL_RICH = "put_rich", "call_rich"
P_HI, P_LO = 0.55, 0.45
WARM_MONTHS = 24
MID_REPORT = 1.2                                   # reported, not judged: the chosen rung's test Sharpe at mid
FILLS = VA.FILLS
CAPITAL = VA.CAPITAL
FEATURES = list(VA.FEATURES)


# --------------------------------------------------------------------------
# the panel
# --------------------------------------------------------------------------

def row_id(stock: str, entry) -> str:
    return f"{stock}|{pd.Timestamp(entry).date()}"


def panel(cycles, predict: bool = True) -> pd.DataFrame:
    """One row per measured stock-cycle: id, stock, entry, expiry, the nine features (f1 and f2 netted), the flat
    baseline ln(put_flat / call_flat) and the raw f1 and f2 (reported only, not features) and the label (NaN with no
    trade). features_not_finite and skipped cycles have no row."""
    rows = []
    for c in cycles:
        if c["status"] != "measured":
            continue
        rows.append({"id": row_id(c["stock"], c["entry"]), "stock": c["stock"], "entry": pd.Timestamp(c["entry"]),
                     "expiry": pd.Timestamp(c["expiry"]), **{f: float(c["features"][f]) for f in FEATURES},
                     "flat_baseline": float(c["flat_baseline"]),
                     **{k: float(c["raw_premiums"][k]) for k in VA.RAW},
                     "label": np.nan if c["label"] is None else float(c["label"]), "predict": predict})
    cols = ["id", "stock", "entry", "expiry", *FEATURES, "flat_baseline", *VA.RAW, "label", "predict"]
    out = pd.DataFrame(rows, columns=cols)
    out["entry"] = pd.to_datetime(out["entry"])
    out["expiry"] = pd.to_datetime(out["expiry"])
    out["predict"] = out["predict"].astype(bool)
    return out


# --------------------------------------------------------------------------
# the rungs
# --------------------------------------------------------------------------

def rule_0(rows: pd.DataFrame) -> dict:
    return {i: PUT_RICH for i in rows["id"]}


def rule_1(rows: pd.DataFrame) -> dict:
    """put_rich when the netted f1 is above the netted f2, call_rich otherwise (a tie is call_rich)."""
    return {i: (PUT_RICH if f1 > f2 else CALL_RICH)
            for i, f1, f2 in zip(rows["id"], rows["f1_put_premium"], rows["f2_call_premium"])}


def make_model(name: str):
    if name == "logistic":
        return LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000)
    if name == "forest":
        return RandomForestClassifier(n_estimators=500, max_depth=3, min_samples_leaf=20, max_features="sqrt",
                                      random_state=2026, n_jobs=1)
    raise ValueError(f"no model {name!r}")


def fit_predict(name: str, X: np.ndarray, y: np.ndarray, X_new: np.ndarray) -> np.ndarray:
    """p = P(label 1) for each row of X_new, from a fit on (X, y) only. One label: p is its share, 1 or 0."""
    X, X_new = np.asarray(X, float), np.asarray(X_new, float)
    y = np.asarray(y).astype(int)
    if len(y) == 0:
        return np.full(len(X_new), np.nan)
    if len(np.unique(y)) == 1:
        return np.full(len(X_new), float(y.mean()))
    model = make_model(name)
    if name == "logistic":
        scaler = StandardScaler().fit(X)
        model.fit(scaler.transform(X), y)
        prob = model.predict_proba(scaler.transform(X_new))
    else:
        model.fit(X, y)
        prob = model.predict_proba(X_new)
    return prob[:, list(model.classes_).index(1)]


def decide(p: float):
    """put_rich above 0.55, call_rich below 0.45, no trade between (or with no p)."""
    if not np.isfinite(p):
        return None
    return PUT_RICH if p > P_HI else CALL_RICH if p < P_LO else None


# --------------------------------------------------------------------------
# the walk-forward
# --------------------------------------------------------------------------

def finished(pool: pd.DataFrame, t) -> pd.DataFrame:
    """The rows whose cycle ended before t: expiry strictly before the entry date."""
    return pool[pool["expiry"] < pd.Timestamp(t)]


def warm_up_start(pool: pd.DataFrame, dates, months: int = WARM_MONTHS):
    """The first of `dates` with at least `months` distinct entry months among the rows finished before it."""
    for t in sorted(pd.Timestamp(d) for d in dates):
        if finished(pool, t)["entry"].dt.to_period("M").nunique() >= months:
            return t
    return None


def walk_forward(pool: pd.DataFrame, dates, months: int = WARM_MONTHS, models=MODELS) -> dict:
    """
    `pool` is every row (a stage's own with predict True, earlier stages' with False), `dates` the stage's entry
    dates. At each date t from the warm-up start: fit each model on the finished labelled rows (in entry then
    stock order), predict the rows entered at t. Returns the start, each model's p by row and each date's
    training ids.
    """
    start = warm_up_start(pool, dates, months)
    out = {"start": start, "p": {m: {} for m in models}, "train": {}, "one_class": {}}
    if start is None:
        return out
    for t in sorted(pd.Timestamp(d) for d in dates if pd.Timestamp(d) >= start):
        new = pool[pool["predict"] & (pool["entry"] == t)]
        if new.empty:
            continue
        train = finished(pool, t)
        train = train[train["label"].notna()].sort_values(["entry", "stock"])
        out["train"][t] = list(train["id"])
        if len(train) and train["label"].nunique() == 1:
            out["one_class"][t] = float(train["label"].iloc[0])
        for m in models:
            p = fit_predict(m, train[FEATURES].to_numpy(), train["label"].to_numpy(), new[FEATURES].to_numpy())
            out["p"][m].update({i: float(v) for i, v in zip(new["id"], p)})
    return out


def decisions(pool: pd.DataFrame, wf: dict) -> dict:
    """Each rung's direction (or None) for every stage row entered on or after the warm-up start."""
    if wf["start"] is None:
        return {r: {} for r in RUNGS}
    scored = pool[pool["predict"] & (pool["entry"] >= wf["start"])]
    out = {"rule_0": rule_0(scored), "rule_1": rule_1(scored)}
    for m in MODELS:
        out[m] = {i: decide(wf["p"][m].get(i, np.nan)) for i in scored["id"]}
    return out


# --------------------------------------------------------------------------
# books and figures
# --------------------------------------------------------------------------

def month_grid(start, last_expiry) -> list[str]:
    """Every calendar month from the warm-up start's to the stage's last expiry's."""
    if start is None:
        return []
    return [str(p) for p in pd.period_range(pd.Timestamp(start).to_period("M"),
                                            pd.Timestamp(last_expiry).to_period("M"), freq="M")]


def book(records: dict, decision: dict, fill: str, months: list[str], stocks, suffix: str = "",
         capital: float = CAPITAL) -> dict:
    """
    The rung's books at one fill: each stock's month P&L / capital and the pooled sum / (stocks x capital), 0 in a
    month with no trade. `records` maps row id to its cycle record, `decision` row id to a direction or None;
    `suffix` picks a variant of the trade ("_vix"). A decided row whose trade does not exist (a leg missing) is
    no trade.
    """
    per = {s: pd.Series(0.0, index=months) for s in stocks}
    for i, d in decision.items():
        if d is None:
            continue
        tr = records[i]["trades"].get(d + suffix)
        if tr is None:
            continue
        s = records[i]["stock"]
        for mo, v in tr[fill]["monthly"].items():
            if mo not in per[s].index:
                raise ValueError(f"{i}'s {d} P&L in {mo} falls outside the scored months")
            per[s][mo] += v
    pooled = sum(per.values()) / (len(stocks) * capital) if stocks else pd.Series(0.0, index=months)
    return {"pooled": pooled, "by_stock": {s: v / capital for s, v in per.items()}}


def sharpe(ret) -> float:
    """mean / sd (ddof 1) x sqrt(12); NaN with fewer than two months, a non-finite value or a zero sd."""
    r = np.asarray(ret, float)
    if len(r) < 2 or not np.isfinite(r).all():
        return np.nan
    sd = float(np.std(r, ddof=1))
    return float(np.mean(r) / sd * np.sqrt(12.0)) if sd > 0 else np.nan


def figures(ret: pd.Series) -> dict:
    r = pd.Series(ret, dtype=float)
    return {"sharpe": sharpe(r), "months": int(len(r)), "mean_pct": 100.0 * float(r.mean()) if len(r) else np.nan,
            "sd_pct": 100.0 * float(r.std(ddof=1)) if len(r) > 1 else np.nan,
            "sum_pct": 100.0 * float(r.sum()),
            "worst_month_pct": 100.0 * float(r.min()) if len(r) else np.nan,
            "monthly": {k: float(v) for k, v in r.items()}}


def per_year(ret: pd.Series) -> dict:
    r = pd.Series(ret, dtype=float)
    years = pd.PeriodIndex(r.index, freq="M").year
    return {str(y): {"sharpe": sharpe(g), "sum_pct": 100.0 * float(g.sum()), "months": int(len(g))}
            for y, g in r.groupby(years)}


def trades_of(records: dict, decision: dict, suffix: str = "") -> list[tuple[str, str]]:
    return [(i, d) for i, d in decision.items() if d is not None and (d + suffix) in records[i]["trades"]]


def hit_rate(records: dict, decision: dict, fill: str, suffix: str = "") -> float:
    """The share of the rung's trades whose cycle P&L at this fill was above zero."""
    t = trades_of(records, decision, suffix)
    if not t:
        return np.nan
    return float(np.mean([records[i]["trades"][d + suffix][fill]["pnl"] > 0 for i, d in t]))


def counts(records: dict, decision: dict) -> dict:
    vals = list(decision.values())
    traded = trades_of(records, decision)
    return {"rows": len(vals), "trades": len(traded), "put_rich": sum(d == PUT_RICH for _, d in traded),
            "call_rich": sum(d == CALL_RICH for _, d in traded), "no_trade_band": sum(d is None for d in vals),
            "no_trade_leg_missing": sum(d is not None for d in vals) - len(traded)}


def score(records: dict, decision: dict, months: list[str], stocks, suffix: str = "") -> dict:
    """A rung's figures at both fills: pooled, per stock, per year, hit rate, and its counts."""
    out = {"counts": counts(records, decision)}
    for fill in FILLS:
        b = book(records, decision, fill, months, stocks, suffix)
        out[fill] = {"pooled": figures(b["pooled"]), "per_year": per_year(b["pooled"]),
                     "by_stock": {s: figures(v) for s, v in b["by_stock"].items()},
                     "hit_rate": hit_rate(records, decision, fill, suffix)}
    return out


def status_counts(cycles) -> dict:
    """Every stock-cycle by status (measured or its skip reason) and the measured ones with no trade, by stock."""
    out = {}
    for c in cycles:
        s = out.setdefault(c["stock"], {})
        s[c["status"]] = s.get(c["status"], 0) + 1
        if c["status"] == "measured" and c.get("no_trade"):
            key = f"measured_no_trade_{c['no_trade']}"
            s[key] = s.get(key, 0) + 1
    total = {}
    for s in out.values():
        for k, v in s.items():
            total[k] = total.get(k, 0) + v
    return {"by_stock": out, "total": total}


def agreement(rows: pd.DataFrame) -> dict:
    """Methods 1-3 as cross-checks: Pearson correlations of f3 (method 1), f4 (method 3's tilt) and f1 - f2 (2), the
    netted f1 and f2 (amended_2026_11_15b)."""
    if len(rows) < 3:
        return {"rows": int(len(rows)), "f3_f4": np.nan, "f3_m2": np.nan, "f4_m2": np.nan}
    m2 = rows["f1_put_premium"] - rows["f2_call_premium"]
    c = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    return {"rows": int(len(rows)), "f3_f4": c(rows["f3_protection_share"], rows["f4_tilt"]),
            "f3_m2": c(rows["f3_protection_share"], m2), "f4_m2": c(rows["f4_tilt"], m2)}


# --------------------------------------------------------------------------
# a stage, the choice and the bar
# --------------------------------------------------------------------------

def run_stage(cycles: list[dict], stocks, history: list[dict] = (), months: int = WARM_MONTHS) -> dict:
    """
    One stage: the stage's rows predicted (history's only trained on), every rung scored over the same months.
    `cycles` are the stage's stock_cycles() records, `history` an earlier stage's.
    """
    parts = [p for p in (panel(history, predict=False), panel(cycles, predict=True)) if len(p)]
    pool = pd.concat(parts, ignore_index=True) if parts else panel([])
    if pool["id"].duplicated().any():
        raise ValueError("a row id repeats between the stage and its history")
    dates = sorted({pd.Timestamp(c["entry"]) for c in cycles})
    wf = walk_forward(pool, dates, months)
    dec = decisions(pool, wf)
    grid = month_grid(wf["start"], max(pd.Timestamp(c["expiry"]) for c in cycles)) if cycles else []
    records = {row_id(c["stock"], c["entry"]): c for c in cycles}
    stage_rows = pool[pool["predict"]]
    return {"start": wf["start"], "months": grid, "walk_forward": wf, "decisions": dec,
            "scores": {r: score(records, dec[r], grid, stocks) for r in RUNGS},
            "status": status_counts(cycles), "agreement": agreement(stage_rows),
            "rows": int(len(stage_rows)), "history_rows": int((~pool["predict"]).sum())}


def _rank(x: float) -> float:
    return x if np.isfinite(x) else -np.inf


def choice_for_test(sharpes: dict) -> dict:
    """
    From development's pooled mid Sharpe ratios of rule 1, the logistic and the forest: the forest only if it is
    above both others; otherwise the higher of rule 1 and the logistic, a tie to rule 1. A NaN Sharpe ratio is
    below every number.
    """
    r1, lg, fo = (_rank(float(sharpes[k])) for k in CANDIDATES)
    if np.isfinite(fo) and fo > r1 and fo > lg:
        chosen, why = "forest", "the forest's Sharpe ratio is above both rule 1's and the logistic's"
    elif lg > r1:
        chosen, why = "logistic", "the forest is not above both; the logistic is above rule 1"
    else:
        chosen, why = "rule_1", "the forest is not above both; the logistic is not above rule 1 (a tie goes to rule 1)"
    return {"chosen": chosen, "why": why, "mid_sharpe": {k: float(sharpes[k]) for k in CANDIDATES}}


def pass_bar(chosen: dict, rule0: dict) -> dict:
    """
    The final test: the chosen rung's pooled Sharpe at the stress fill above 0, and above rule 0's at mid (mid
    against mid) and at the stress fill (stress against stress). A NaN anywhere is not a pass.
    """
    c_mid, c_x, r_mid, r_x = (float(v) for v in (chosen["mid"], chosen["cross"], rule0["mid"], rule0["cross"]))
    finite = bool(np.isfinite([c_mid, c_x, r_mid, r_x]).all())
    cond = {"stress_above_zero": bool(c_x > 0), "mid_above_rule_0": bool(c_mid > r_mid),
            "stress_above_rule_0": bool(c_x > r_x), "all_finite": finite}
    return {"passes": finite and cond["stress_above_zero"] and cond["mid_above_rule_0"] and cond["stress_above_rule_0"],
            "conditions": cond, "sharpe": {"chosen_mid": c_mid, "chosen_stress": c_x, "rule_0_mid": r_mid,
                                           "rule_0_stress": r_x}}


def sharpes_of(result: dict, rung: str) -> dict:
    return {fill: result["scores"][rung][fill]["pooled"]["sharpe"] for fill in FILLS}


def reported(result: dict, chosen: str, stock: str = "NVDA") -> dict:
    """The test's reported-only figures: mid Sharpe >= 1.2, the one stock alone, every rung's pooled Sharpe."""
    mid = result["scores"][chosen]["mid"]["pooled"]["sharpe"]
    return {"chosen_mid_sharpe_at_least_1_2": bool(np.isfinite(mid) and mid >= MID_REPORT),
            "alone": {"stock": stock, **{r: {fill: result["scores"][r][fill]["by_stock"].get(stock, {}).get("sharpe",
                                                                                                         np.nan)
                                             for fill in FILLS} for r in RUNGS}},
            "every_rung": {r: sharpes_of(result, r) for r in RUNGS}}


# --------------------------------------------------------------------------
# extras_2026_11_15, reported only
# --------------------------------------------------------------------------

def split_sums(records: dict, decision: dict, fill: str, stocks, suffix: str = "") -> dict:
    """extras pnl_split: the six parts of each trade's P&L, summed per stock and pooled."""
    per_trade = [{"id": i, "direction": d, "pnl": records[i]["trades"][d + suffix][fill]["pnl"],
                  **records[i]["trades"][d + suffix][fill]["split"]} for i, d in trades_of(records, decision, suffix)]
    zero = {k: 0.0 for k in (*VA.SPLIT, "pnl")}
    by_stock = {s: dict(zero) for s in stocks}
    for t in per_trade:
        s = records[t["id"]]["stock"]
        for k in zero:
            by_stock[s][k] += t[k]
    pooled = {k: float(sum(v[k] for v in by_stock.values())) for k in zero}
    return {"per_trade": per_trade, "by_stock": by_stock, "pooled": pooled}


def extras(result: dict, cycles: list[dict], stocks, chosen: str | None) -> dict:
    """
    extras_2026_11_15, reported only, on deep copies: the P&L split for every rung and for the put leg alone;
    put_leg_only (the hedged short put every measured cycle scored from the warm-up start, also one whose call leg
    is missing, counted as call_leg_missing: clarified put_leg_only) beside rule 0; rule 0's and the chosen rung's
    trades VIX-scaled beside unscaled; the smile chart rows.
    """
    result, cycles = copy.deepcopy(result), copy.deepcopy(cycles)
    records = {row_id(c["stock"], c["entry"]): c for c in cycles}
    months, dec = result["months"], result["decisions"]
    put_only = {i: "put_only" for i in dec["rule_0"]}
    po = score(records, put_only, months, stocks)
    vix = {}
    for rung in dict.fromkeys(r for r in ("rule_0", chosen) if r):
        scaled = score(records, dec[rung], months, stocks, suffix="_vix")
        vix[rung] = {fill: {"scaled": scaled[fill]["pooled"]["sharpe"],
                            "unscaled": result["scores"][rung][fill]["pooled"]["sharpe"],
                            "scaled_figures": scaled[fill]["pooled"]} for fill in FILLS}
        vix[rung]["counts"] = scaled["counts"]
    split = {rung: {fill: split_sums(records, dec[rung], fill, stocks) for fill in FILLS} for rung in RUNGS}
    split["put_leg_only"] = {fill: split_sums(records, put_only, fill, stocks) for fill in FILLS}
    call_missing = sum("put_rich" not in records[i]["trades"] for i, _ in trades_of(records, put_only))
    return {"put_leg_only": {fill: {"sharpe": po[fill]["pooled"]["sharpe"],
                                    "rule_0_sharpe": result["scores"]["rule_0"][fill]["pooled"]["sharpe"],
                                    "figures": po[fill]} for fill in FILLS}
            | {"counts": {**po["counts"], "call_leg_missing": int(call_missing)}},
            "vix_scaled_size": {"reference_vix": VA.VIX_REF, "rungs": vix},
            "pnl_split": split,
            "smile_charts": VA.smile_chart(cycles)}
