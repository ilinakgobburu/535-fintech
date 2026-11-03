"""
CURIOSITY RUN, NOT USED ON THE SITE: config/vol_screen_2023.yaml, registered 2026-11-03.

Which stock would config/premium_screen.yaml's volatility-premium measure have picked from the S&P 500
of 2023-12-29, over 2014-2023, in place of config/mr_screen.yaml's mean-reversion pick (MA)?

    python3 scripts/vol_screen_2023.py           # pull what is missing (resuming), rank, feasibility, pick, then
                                                 # the wheel (pick already cached) or a request estimate (not cached)
    python3 scripts/vol_screen_2023.py --plan    # what is cached and what is left to pull; no session

Reused, not copied: premium_screen.measure / eligible / history (the measure, the coverage rule, the
RIC fallback); mr_screen.feasible / pull_feasibility / frame_to_series (feasibility (a)-(c) and its
pull), with mr_screen's cached closes and feasibility pulls; fetch_wheel_raw.open_session / plan;
run_wheel.book / cell_row on config/wheel_mr.yaml. Writes only cache/vol_screen_2023*.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mr_screen as MR  # noqa: E402
import premium_screen as PS  # noqa: E402
from fetch_wheel_raw import open_session  # noqa: E402

CFG_PATH = REPO / "config" / "vol_screen_2023.yaml"
MR_RAW = MR.RAW                                   # read only
RAW = REPO / "cache" / "vol_screen_2023_raw.pkl"
OUT = REPO / "cache" / "vol_screen_2023.json"
LOG = REPO / "cache" / "vol_screen_2023.log"
WHEEL_OUT = REPO / "cache" / "vol_screen_2023_wheel.json"
WHEEL_MR = REPO / "config" / "wheel_mr.yaml"
CACHED_WHEELS = ("MA", "APA", "V", "FISV", "ADI")   # options at cache/wheel_raw_{TICKER}_2024
MIN_MONTHS = 111                                  # of 120; coverage >= 95% is premium_screen.eligible's
EXT_SPAN = ("2023-12-15", "2024-02-15")           # closes for December 2023's 21 returns, overlapping the cache
JOIN_TOLERANCE = 0.001
BATCH = 50
NEED = 3                                          # the pick and two follow-ups


class CapReached(BaseException):
    """
    LSEG refused for its request cap. A BaseException, so premium_screen.history's `except Exception`
    cannot swallow it; the run stops with the cache saved and the next run resumes.
    """


def log(msg: str = "") -> None:
    print(msg, flush=True)
    with LOG.open("a") as fh:
        fh.write(msg + "\n")


def root_of(ric: str) -> str:
    return ric.split(".")[0]


def iv_ric(ric: str) -> str:
    return f"{root_of(ric)}ATMIV.U"


def as_float(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").dropna().astype(float).sort_index()
    s.index = pd.DatetimeIndex(s.index)
    return s


# --------------------------------------------------------------------------
# LSEG, counted
# --------------------------------------------------------------------------

class Counted:
    """lseg.data behind one get_history, opened on demand, with every call recorded in the raw cache."""

    def __init__(self, raw: dict):
        self.raw, self.ld, self.what = raw, None, ""

    def open(self) -> None:
        if self.ld is not None:
            return
        import lseg.data as ld
        warnings.filterwarnings("ignore")
        open_session(ld)
        state = ld.session.get_default().open_state
        if state != ld.OpenState.Opened:
            raise ConnectionError(f"LSEG not reachable: no Workspace proxy ready on ports 9000-9010 or 9060 "
                                  f"(session {state.name}); start Workspace and run again")
        self.ld = ld

    def get_history(self, **kw):
        self.open()
        uni = kw.get("universe")
        rec = {"at": str(pd.Timestamp.now().floor("s")), "what": self.what,
               "rics": 1 if isinstance(uni, str) else len(uni), "first": uni if isinstance(uni, str) else uni[0]}
        self.raw["requests"].append(rec)
        try:
            return self.ld.get_history(**kw)
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}".replace("\n", " ")
            if "429" in str(exc) or "Too many" in str(exc):
                raise CapReached(rec["error"]) from exc
            raise

    def close(self) -> None:
        if self.ld is not None:
            self.ld.close_session()


def iv_frames(got: pd.DataFrame, rics: list[str]) -> dict:
    """One frame of the two iv fields per RIC from a get_history frame, whatever its column layout."""
    out = {}
    if got is None or got.empty:
        return out
    if isinstance(got.columns, pd.MultiIndex):
        for r in rics:
            if r in got.columns.get_level_values(0):
                out[r] = got[r]
    elif len(rics) == 1 or got.columns.name in rics:
        out[got.columns.name if got.columns.name in rics else rics[0]] = got
    clean = {}
    for r, f in out.items():
        f = f.reindex(columns=PS.IV_FIELDS).apply(pd.to_numeric, errors="coerce").astype(float).dropna(how="all")
        f.index = pd.DatetimeIndex(f.index)
        if len(f):
            clean[r] = f.sort_index()
    return clean


def pull_batched(api: Counted, raw: dict, todo: list[str], parse, store: dict, missing: dict, label: str, **kw):
    """
    BATCH RICs a call. A batch that raises is asked one by one; a RIC a batch returned nothing for is
    asked alone once, so a partial batch failure is not mistaken for no data.
    """
    api.what = label
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        try:
            got = parse(api.get_history(universe=batch, interval="daily", **kw), batch)
        except Exception as exc:
            log(f"  {label}: batch from {batch[0]} raised {type(exc).__name__}; asking one by one")
            got = {}
        alone = [r for r in batch if r not in got]
        for r in alone:
            try:
                got.update(parse(api.get_history(universe=r, interval="daily", **kw), [r]))
            except Exception as exc:
                missing[r] = f"{type(exc).__name__}: {str(exc)[:120]}".replace("\n", " ")
            time.sleep(0.2)
        for r in alone:
            if r not in got:
                missing.setdefault(r, "no rows")
        store.update(got)
        pd.to_pickle(raw, RAW)
        log(f"  {label} {min(i + BATCH, len(todo))}/{len(todo)}: {sum(r in got for r in batch)} of {len(batch)} returned"
            + (f" ({len(alone)} asked alone)" if alone else ""))


# --------------------------------------------------------------------------
# the pulls
# --------------------------------------------------------------------------

def todo_lists(mr: dict, raw: dict) -> dict:
    ivs = sorted({iv_ric(r) for r in mr["constituents"]})
    iv_todo = [r for r in ivs if r not in raw["iv"] and r not in raw["iv_missing"]]
    have_iv = {r for r in mr["constituents"] if iv_ric(r) in raw["iv"]}
    ext_todo = [r for r in mr["constituents"] if r in mr["px"] and r in have_iv
                and r not in raw["px_ext"] and r not in raw["px_ext_missing"]]
    full_todo = [r for r in mr["missing"] if r not in raw["px_full"] and r not in raw["px_full_missing"]]
    return {"iv": iv_todo, "ext": ext_todo, "full": full_todo, "iv_total": len(ivs), "iv_pending": bool(iv_todo)}


def pull_all(api: Counted, mr: dict, raw: dict, start: str, end: str) -> None:
    todo = todo_lists(mr, raw)
    if not (todo["iv"] or todo["ext"] or todo["full"]):
        return
    api.open()
    if todo["iv"]:
        log(f"iv30: {len(todo['iv'])} ATMIV RICs to pull, {start} to {end}")
        pull_batched(api, raw, todo["iv"], iv_frames, raw["iv"], raw["iv_missing"], "iv30",
                     fields=PS.IV_FIELDS, start=start, end=end)
    todo = todo_lists(mr, raw)
    if todo["ext"]:
        log(f"closes {EXT_SPAN[0]} to {EXT_SPAN[1]}: {len(todo['ext'])} RICs")
        pull_batched(api, raw, todo["ext"], MR.frame_to_series, raw["px_ext"], raw["px_ext_missing"], "closes_ext",
                     fields=["TRDPRC_1"], start=EXT_SPAN[0], end=EXT_SPAN[1])
    if todo["full"]:
        api.what = "closes_full"
        lo = str((pd.Timestamp(start) - pd.Timedelta(days=40)).date())
        for ric in todo["full"]:
            tries = [ric, f"{root_of(ric)}.OQ", f"{root_of(ric)}.O"]
            used, got = PS.history(api, tries, fields=["TRDPRC_1"], start=lo, end=EXT_SPAN[1])
            if used is None:
                raw["px_full_missing"][ric] = f"no rows on {', '.join(tries)}"
            else:
                raw["px_full"][ric] = {"ric": used, "px": as_float(got["TRDPRC_1"])}
            pd.to_pickle(raw, RAW)
            log(f"  closes for {ric}: {'from ' + used if used else 'none on ' + ', '.join(tries)}")


# --------------------------------------------------------------------------
# the screen
# --------------------------------------------------------------------------

def closes(ric: str, mr: dict, raw: dict) -> tuple[pd.Series | None, str | None, dict]:
    """A constituent's split-adjusted closes, the RIC they came from, and how the extension was joined."""
    if ric in raw["px_full"]:
        return raw["px_full"][ric]["px"], raw["px_full"][ric]["ric"], {"closes": "full pull (no cached closes)"}
    if ric not in mr["px"]:
        return None, None, {}
    base = as_float(mr["px"][ric])
    if ric not in raw["px_ext"]:
        return base, ric, {"closes": "cache only: no closes after 2023-12-29"}
    ext = as_float(raw["px_ext"][ric])
    common = base.index.intersection(ext.index)
    if not len(common):
        return base, ric, {"closes": "cache only: extension does not overlap the cache"}
    day = common.max()
    ratio = float(ext.loc[day] / base.loc[day])
    note = {"closes": f"cache + extension joined at {day.date()}"}
    if abs(ratio - 1.0) > JOIN_TOLERANCE:
        ext = ext / ratio
        note["rescaled"] = ratio
    return pd.concat([base, ext[ext.index > base.index.max()]]), ric, note


def months_detail(iv: pd.DataFrame, px: pd.Series, entries: list[str]) -> pd.DataFrame:
    """Per scored month, iv30 and realised, rebuilt from measure()'s entry dates: a diagnostic only."""
    iv30 = iv[PS.IV_FIELDS].mean(axis=1, skipna=False)
    lr = np.log(px).diff()
    pos = {t: i for i, t in enumerate(px.index)}
    rows = []
    for e in map(pd.Timestamp, entries):
        rv = float(lr.iloc[pos[e] + 1: pos[e] + 22].std(ddof=1) * np.sqrt(252) * 100)
        rows.append({"entry": e, "iv30": float(iv30.loc[e]), "rv": rv})
    d = pd.DataFrame(rows)
    d["ratio"] = (d["iv30"] - d["rv"]) / d["iv30"]
    return d


def diagnostics(iv: pd.DataFrame, px: pd.Series, m: dict, start, end) -> dict:
    d = months_detail(iv, px, m["entries"])
    w = iv[(iv.index >= pd.Timestamp(start)) & (iv.index <= pd.Timestamp(end))]
    iv30 = w[PS.IV_FIELDS].mean(axis=1, skipna=False).dropna()
    worst = d.loc[d["ratio"].idxmin()]
    return {"check_mean_matches": bool(abs(d["ratio"].mean() - m["mean_ratio"]) < 1e-9),
            "median_ratio": float(d["ratio"].median()),
            "ratio_without_worst_5pct": float(d["ratio"][d["ratio"] > d["ratio"].quantile(0.05)].mean()),
            "worst_month": f"{worst['entry'].date()} iv30 {worst['iv30']:.1f} rv {worst['rv']:.1f}",
            "iv30_min_median_max": [float(iv30.min()), float(iv30.median()), float(iv30.max())],
            "days_call_put_gap_over_10pts": int(((w["30D_A_IM_C"] - w["30D_A_IM_P"]).abs() > 10).sum()),
            "mean_iv30_entries": float(d["iv30"].mean()), "mean_rv": float(d["rv"].mean())}


def score(mr: dict, raw: dict, start: str, end: str) -> dict:
    table = {}
    for ric in mr["constituents"]:
        iv = raw["iv"].get(iv_ric(ric))
        px, used, join = closes(ric, mr, raw)
        row = {"name": mr["names"].get(ric), "iv_ric": iv_ric(ric), "px_ric": used, **join}
        if iv is None or px is None:
            row.update(coverage=0.0, months=0, mean_ratio=np.nan, mean_points=np.nan, share_positive=np.nan,
                       eligible=False, why_not="no iv30 from LSEG" if iv is None else "no closes")
        else:
            m = PS.measure(iv, px, start, end)
            row.update({k: v for k, v in m.items() if k != "entries"})
            row["eligible"] = PS.eligible(m, months_needed=MIN_MONTHS)
            why = ([f"iv30 on {100 * m['coverage']:.1f}% of days"] if m["coverage"] < 0.95 else []) + \
                  ([f"{m['months']} months scored"] if m["months"] < MIN_MONTHS else [])
            if why:
                row["why_not"] = "; ".join(why)
            row["_m"] = m
        table[ric] = row
    return table


def feasibility(api: Counted, mr: dict, raw: dict, table: dict, ranked: list[str]) -> tuple[list, list]:
    """config/mr_screen.yaml's (a)-(c) down the ranking until NEED stocks pass."""
    chosen, skipped = [], []
    for rank, ric in enumerate(ranked, 1):
        if len(chosen) == NEED:
            break
        px_ric = table[ric]["px_ric"]
        if px_ric in mr["feasibility"]:
            f, source = mr["feasibility"][px_ric], "cache/mr_screen_raw.pkl"
        elif px_ric in raw["feasibility"]:
            f, source = raw["feasibility"][px_ric], "cache/vol_screen_2023_raw.pkl"
        else:
            api.open()
            api.what = "feasibility"
            f = MR.pull_feasibility(api, px_ric)
            raw["feasibility"][px_ric] = f
            pd.to_pickle(raw, RAW)
            source = "pulled"
        check = MR.feasible(f["default"], f["unadjusted"])
        table[ric]["feasibility"] = {**check, "source": source}
        if check["feasible"]:
            chosen.append(ric)
        else:
            skipped.append({"rank": rank, "ric": ric, "name": table[ric]["name"], "why": check["why"],
                            "ric_retired": "^" in px_ric})
    return chosen, skipped


# --------------------------------------------------------------------------
# after the screen
# --------------------------------------------------------------------------

def wheel_on(ticker: str) -> dict:
    """config/wheel_mr.yaml's plain wheel on a stock whose options are cached: train pick, then the test once."""
    import run_wheel as RW
    RW.use_config(WHEEL_MR)
    W = RW.W
    market = W.load_market(REPO / "cache" / f"wheel_raw_{ticker}_2024")
    g = RW.CFG["grid"]
    cells = W.grid_cells(g["put_delta"], g["call_delta"], g["floor"])
    train_grid = [RW.cell_row(c, RW.book(market, "train", "mid", **c)[1]) for c in cells]
    chosen = W.pick(train_grid, RW.CFG["selection"]["score"].split()[0].lower())
    setting = {k: chosen[k] for k in ("put_delta", "call_delta", "floor")}
    registered = REPO / "config" / "wheel_mr_picks" / f"{ticker}.json"     # read only, for the cross-check
    reg_pick = json.loads(registered.read_text())["pick"] if registered.exists() else None
    keep = ("return_pct", "sharpe", "max_drawdown_pct", "pnl", "premium", "interest", "shares_pnl",
            "puts", "assigned", "calls", "called_away", "share_days_pct")
    periods = {}
    for name in ("train", "test"):
        rows = {}
        for label, fill, kw in (("wheel_mid", "mid", setting), ("wheel_bid", "cross", setting),
                                ("buy_and_hold", "mid", {"mode": "buy_and_hold"}), ("t_bills", "mid", {"mode": "cash"})):
            rows[label] = {k: v for k, v in RW.cell_row({}, RW.book(market, name, fill, **kw)[1]).items() if k in keep}
        periods[name] = {"window": list(RW.window(name)), **rows}
    return {"ticker": ticker, "config": "config/wheel_mr.yaml", "setting": setting,
            "train_sharpe_mid": chosen["sharpe"], "same_as_config_wheel_mr_picks": reg_pick == setting,
            "registered_pick": reg_pick, "periods": periods, "train_grid_mid": train_grid}


def request_estimate(ric: str, f: dict, splits: list) -> dict:
    """What scripts/fetch_wheel_raw.py would ask for over 2024-01 to 2026-10: config/wheel_mr.yaml's bands and grid."""
    import fetch_wheel_raw as FW
    key = f"{root_of(ric)}_2024"
    FW.UNDERLYINGS.setdefault(key, {"stock": ric, "root": root_of(ric), "roots_from": [], "units_after": None,
                                    "seed": None, "window": FW.CLASS_RUN, "C": FW.GENERIC, "P": FW.GENERIC,
                                    "splits": [(pd.Timestamp(d).date(), r) for d, r in splits]})
    jobs = FW.plan(as_float(f["unadjusted"]), key)
    scale = {}
    for t in CACHED_WHEELS:
        p = REPO / "cache" / f"wheel_raw_{t}_2024" / "requests.jsonl"
        if p.exists():
            scale[t] = sum(1 for _ in p.open())
    return {"contracts": len(jobs), "puts": sum(j["cp"] == "P" for j in jobs), "calls": sum(j["cp"] == "C" for j in jobs),
            "stock_and_aux": 4, "total_requests": len(jobs) + 4,
            "note": "one request per contract, more where a dropped connection is retried; stock closes, "
                    "dividends, earnings and the T-bill are the other four",
            "for_scale_cached_runs": scale}


# --------------------------------------------------------------------------

def new_raw() -> dict:
    return {"registered": "2026-11-03", "iv": {}, "iv_missing": {}, "px_ext": {}, "px_ext_missing": {},
            "px_full": {}, "px_full_missing": {}, "feasibility": {}, "requests": []}


def request_counts(raw: dict, since: int = 0) -> dict:
    reqs = raw["requests"][since:]
    by = {}
    for r in reqs:
        b = by.setdefault(r["what"], {"calls": 0, "rics": 0, "errors": 0})
        b["calls"] += 1
        b["rics"] += r["rics"]
        b["errors"] += "error" in r
    return {"calls": len(reqs), "rics": sum(r["rics"] for r in reqs), "by_kind": by}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="show what is cached and left to pull; no session")
    ap.add_argument("--retry-missing", action="store_true", help="ask again for RICs recorded as returning nothing")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    cfg = yaml.safe_load(CFG_PATH.read_text())
    start, end = (str(x) for x in cfg["window"])
    mr = pd.read_pickle(MR_RAW)
    raw = pd.read_pickle(RAW) if RAW.exists() else new_raw()
    if args.retry_missing:
        for k in ("iv_missing", "px_ext_missing", "px_full_missing"):
            raw[k] = {}

    todo = todo_lists(mr, raw)
    if args.plan:
        print(f"universe {len(mr['constituents'])} ({mr['as_of']}); ATMIV RICs {todo['iv_total']}: "
              f"{len(raw['iv'])} cached, {len(raw['iv_missing'])} returned nothing, {len(todo['iv'])} to pull "
              f"(~{math.ceil(len(todo['iv']) / BATCH)} calls)")
        print(f"closes after 2023-12-29: {len(raw['px_ext'])} cached, {len(todo['ext'])} to pull"
              + (" (known once iv30 is in)" if todo["iv_pending"] else "")
              + f"; full series for {len(todo['full'])} with no cached closes ({', '.join(todo['full'])})")
        print(f"feasibility pulls cached: {len(mr['feasibility'])} in mr_screen, {len(raw['feasibility'])} here")
        print(f"requests so far: {request_counts(raw)}")
        return 0

    LOG.parent.mkdir(exist_ok=True)
    log(f"\n=== {pd.Timestamp.now():%Y-%m-%d %H:%M} vol_screen_2023 (curiosity run, not used on the site) ===")
    first_request = len(raw["requests"])
    api = Counted(raw)
    try:
        pull_all(api, mr, raw, start, end)
        if todo_lists(mr, raw)["iv"]:
            log("iv30 pull incomplete; run again to resume")
            return 3
        table = score(mr, raw, start, end)
        ranked = sorted((r for r in table if table[r]["eligible"]), key=lambda r: (-table[r]["mean_ratio"], r))
        chosen, skipped = feasibility(api, mr, raw, table, ranked)
    except ConnectionError as exc:
        log(str(exc))
        return 2
    except CapReached as exc:
        log(f"stopped, LSEG request cap: {exc}; the cache is saved and the next run resumes")
        return 3
    finally:
        api.close()
        pd.to_pickle(raw, RAW)
    log(f"LSEG this run: {request_counts(raw, first_request)}")

    # diagnostics on the top of the table and on MA
    look = ranked[:10] + [r for r in ("MA.N",) if r in table and r not in ranked[:10] and "_m" in table[r]]
    for ric in look:
        px, _, _ = closes(ric, mr, raw)
        table[ric]["diagnostics"] = diagnostics(raw["iv"][iv_ric(ric)], px, table[ric]["_m"], start, end)
    for row in table.values():
        row.pop("_m", None)

    scored = [r for r in table if table[r]["months"] > 0]
    counts = {"universe": len(mr["constituents"]), "atmiv_rics": len({iv_ric(r) for r in mr["constituents"]}),
              "iv_returned": sum(iv_ric(r) in raw["iv"] for r in mr["constituents"]),
              "iv_missing": sorted(raw["iv_missing"]), "with_closes": sum(table[r]["px_ric"] is not None for r in table),
              "scored": len(scored), "coverage_ok": sum(table[r]["coverage"] >= 0.95 for r in scored),
              "months_ok": sum(table[r]["months"] >= MIN_MONTHS for r in scored), "eligible": len(ranked)}
    rank_of = {r: i for i, r in enumerate(ranked, 1)}
    mr_picks = {r: rank_of.get(r) for r in ("MA.N", "APA.OQ", "V.N", "FISV.OQ", "ADI.OQ")}
    pick = chosen[0] if chosen else None
    result = {"status": "curiosity run, not used on the site", "registered": str(cfg["registered"]),
              "config": "config/vol_screen_2023.yaml", "window": [start, end], "counts": counts,
              "pick": pick, "follow_ups": chosen[1:], "skipped": skipped,
              "ma": {"rank": rank_of.get("MA.N"), "of": len(ranked), **{k: table.get("MA.N", {}).get(k) for k in
                     ("mean_ratio", "mean_points", "share_positive", "months", "coverage", "why_not")}},
              "mr_screen_picks_rank_here": mr_picks, "rescaled_joins": {r: t["rescaled"] for r, t in table.items() if "rescaled" in t},
              "requests_total": request_counts(raw), "ranked": ranked, "table": table}

    log(f"universe {counts['universe']}; iv30 returned for {counts['iv_returned']}; scored {counts['scored']}; "
        f"coverage >= 95% {counts['coverage_ok']}; >= {MIN_MONTHS} months {counts['months_ok']}; eligible {counts['eligible']}")
    if counts["iv_missing"]:
        miss = counts["iv_missing"]
        log(f"no iv30 for {len(miss)}: {', '.join(miss[:40])}{' ...' if len(miss) > 40 else ''}")
    if result["rescaled_joins"]:
        log(f"extension rescaled at the join: {result['rescaled_joins']}")
    log(f"\n{'rank':>4} {'RIC':12} {'name':30} {'score':>7} {'gap pts':>8} {'IV>RV':>6} {'months':>6} {'cover':>6}")
    for i, r in enumerate(ranked[:25], 1):
        t = table[r]
        f = t.get("feasibility")
        tag = "" if f is None else ("  PICK" if r == pick else "  follow-up" if r in chosen else "  skipped: " + f["why"])
        log(f"{i:4} {r:12} {str(t['name'])[:30]:30} {t['mean_ratio']:7.3f} {t['mean_points']:8.2f} "
            f"{100 * t['share_positive']:5.0f}% {t['months']:6} {100 * t['coverage']:5.1f}%{tag}")
    log(f"\npick: {pick}; follow-ups: {chosen[1:]}")
    log(f"MA: rank {result['ma']['rank']} of {len(ranked)}; mr_screen's picks here: {mr_picks}")
    for r in look:
        log(f"  {r:12} {json.dumps(table[r]['diagnostics'])}")
    OUT.write_text(json.dumps(result, indent=1, default=str))
    log(f"wrote {OUT}")

    if pick is None:
        return 0
    ticker = root_of(pick)
    if ticker in CACHED_WHEELS and (REPO / "cache" / f"wheel_raw_{ticker}_2024").exists():
        w = wheel_on(ticker)
        WHEEL_OUT.write_text(json.dumps({"status": "curiosity run, not used on the site", **w}, indent=1, default=float))
        log(f"\nwheel on {ticker} (config/wheel_mr.yaml): setting {w['setting']}, training Sharpe {w['train_sharpe_mid']:.4f}"
            f"{'' if w['same_as_config_wheel_mr_picks'] else '  (differs from config/wheel_mr_picks!)'}")
        for name, p in w["periods"].items():
            log(f"  {name} {p['window']}")
            for label in ("wheel_mid", "wheel_bid", "buy_and_hold", "t_bills"):
                x = p[label]
                log(f"    {label:13} return {x['return_pct']:6.1f}%  Sharpe {x['sharpe']:5.2f}  max DD {x['max_drawdown_pct']:6.1f}%")
        log(f"wrote {WHEEL_OUT}")
    else:
        px_ric = table[pick]["px_ric"]
        f = mr["feasibility"].get(px_ric) or raw["feasibility"][px_ric]
        est = request_estimate(px_ric, f, table[pick]["feasibility"]["splits"])
        result["option_pull_estimate"] = est
        OUT.write_text(json.dumps(result, indent=1, default=str))
        log(f"\n{ticker}: options not cached, none pulled. fetch_wheel_raw.py would ask for about "
            f"{est['total_requests']} requests ({est['puts']} puts, {est['calls']} calls, 4 stock/aux); "
            f"cached runs for scale: {est['for_scale_cached_runs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
