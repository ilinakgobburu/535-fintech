"""The wheel pull's plan and seed, on made-up closes: no LSEG session."""

from __future__ import annotations

import datetime as dt
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import fetch_wheel_raw as F  # noqa: E402

from trading_app.lib.ric import parse_option_ric  # noqa: E402


def closes(level: float = 40.0, split: tuple | None = None) -> pd.Series:
    days = pd.bdate_range("2018-06-01", "2024-02-01")
    px = pd.Series(level, index=days)
    if split:
        px[px.index < pd.Timestamp(split[0])] *= split[1]
    return px


def test_the_tier_is_chosen_by_price():
    tiers = [(25.0, (0.5,)), (100.0, (1.0,)), (float("inf"), (5.0,))]
    assert F.steps_at(tiers, 20.0) == (0.5,)
    assert F.steps_at(tiers, 25.0) == (1.0,)
    assert F.steps_at(tiers, 400.0) == (5.0,)


def test_puts_sit_below_spot_and_calls_above_on_the_tier_grid():
    jobs = F.plan(closes(40.0), "AMD")
    puts = sorted({j["strike"] for j in jobs if j["cp"] == "P"})
    calls = sorted({j["strike"] for j in jobs if j["cp"] == "C"})
    assert puts[0] == np.floor(40 * 0.70) and puts[-1] == np.ceil(40 * 1.02)
    assert calls[0] == np.floor(40 * 0.97) and calls[-1] == np.ceil(40 * 1.40)
    assert all(k == int(k) for k in puts + calls)          # the $1 tier


def test_every_ric_parses_back_to_its_job():
    for j in F.plan(closes(40.0), "AMD")[:200]:
        meta = parse_option_ric(j["ric"])
        assert (meta["cp"], meta["expiry"], meta["strike"]) == (j["cp"], j["expiry"], j["strike"])


def test_each_contract_is_requested_over_its_last_fifty_days():
    for j in F.plan(closes(40.0), "AMD")[:50]:
        assert j["end"] == j["expiry"] and (j["end"] - j["start"]).days == 50


def test_a_contract_open_across_the_split_carries_its_post_split_strike():
    # NVDA at 200 after 2021-07-20 and 800 before it, in the dollars of the day.
    jobs = F.plan(closes(200.0, split=("2021-07-20", 4.0)), "NVDA")
    aug = sorted(j["strike"] for j in jobs if j["cp"] == "P" and j["expiry"] == dt.date(2021, 8, 20))
    jul = sorted(j["strike"] for j in jobs if j["cp"] == "P" and j["expiry"] == dt.date(2021, 7, 16))
    # Banded on sessions around the July expiry, all pre-split at $800, on the
    # $5 tier, then divided by 4: 0.70 x 800 = 560 -> 140, and 1.02 x 800 =
    # 816, up to the grid's 820 -> 205.
    assert aug[0] == 140.0 and aug[-1] == 205.0
    # A July contract expired before the split and keeps pre-split dollars.
    assert jul[0] == np.floor(800 * 0.70 / 5) * 5


def test_the_seed_is_copied_once_and_failed_records_are_not(tmp_path):
    jobs = F.plan(closes(40.0), "AMD")[:3]
    seed_dir, cache = tmp_path / "seed", tmp_path / "cache"
    seed_dir.mkdir(), cache.mkdir()
    states = ["returned", "never_listed", "failed"]
    with (seed_dir / "requests.jsonl").open("w") as log:
        for j, s in zip(jobs, states):
            log.write(json.dumps({"ric": j["ric"], "kind": "monthly", "expiry": str(j["expiry"]),
                                  "strike": j["strike"], "status": s, "detail": "", "rows": 1}) + "\n")
    frame = pd.DataFrame({"BID": [1.0], "ASK": [1.1]}, index=pd.to_datetime(["2019-01-02"]))
    (seed_dir / "options.pkl").write_bytes(pickle.dumps({jobs[0]["ric"]: frame}))
    assert F.seed(cache, seed_dir, jobs) == 2
    assert F.seed(cache, seed_dir, jobs) == 0
    logged = F.read_log(cache / "requests.jsonl")
    assert set(logged) == {jobs[0]["ric"], jobs[1]["ric"]}
    assert logged[jobs[0]["ric"]]["seeded_from"] == "seed"
    assert list(pickle.loads((cache / "options.pkl").read_bytes())) == [jobs[0]["ric"]]


def test_fiserv_contracts_carry_fi_until_october_2025_and_fisv_after():
    days = pd.bdate_range("2023-06-01", "2026-10-23")
    jobs = F.plan(pd.Series(150.0, index=days), "FISV_2024")
    roots = {j["expiry"]: j["ric"][:4] for j in jobs}
    assert roots[dt.date(2025, 10, 17)].startswith("FI") and not roots[dt.date(2025, 10, 17)].startswith("FISV")
    assert roots[dt.date(2025, 11, 21)] == "FISV"
    assert min(roots) == dt.date(2024, 2, 16) and max(roots) == dt.date(2026, 10, 16)
