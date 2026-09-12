"""
Tests for the fetcher: response-shape resolution, the candidate universe, and
the dry-run path.

None of this had a test, and one of these functions is the one that actually
shipped a bug. `_as_ric_field` resolves an LSEG response into a
(RIC, Field) MultiIndex, and its first version mislabelled field names as RICs
for every bisected batch -- caught only because a batch of 20 reported 26 live
series. A cache written that way is not corrupt in any way that looks like
corruption: it is a frame full of columns called BID and ASK where RICs should
be, which then silently yields no parseable contracts at all.

So the shapes below are not invented. They are the three shapes LSEG was
observed to return, probed directly:

    1 RIC,  N fields -> columns are FIELDS, columns.name = the RIC
    1 RIC,  1 field  -> columns are FIELDS, columns.name = the RIC
    2+ RICs, 1 field -> columns are RICS,   columns.name = the field

Flat columns carry whichever axis has more than one member, and when both are
singletons the columns are fields. Bisection -- a pure speed optimisation --
drives batches down to one RIC and therefore flips the meaning of the response
underneath itself.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_fetcher():
    spec = importlib.util.spec_from_file_location("fetch_hw2", ROOT / "scripts" / "fetch_hw2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F = _load_fetcher()

RIC_A = "AAPLI042632000.U^I26"
RIC_B = "AAPLI042631500.U^I26"
FIELDS = ["BID", "ASK", "TRDPRC_1"]
IDX = pd.DatetimeIndex(["2026-08-31 15:00", "2026-08-31 16:00"])


class TestAsRicField:
    def test_one_ric_many_fields_columns_are_fields(self):
        """The bisected case. Columns are FIELDS; the RIC is on columns.name."""
        got = pd.DataFrame([[1.0, 1.2, 1.1], [1.3, 1.4, 1.35]], index=IDX, columns=FIELDS)
        got.columns.name = RIC_A
        out = F._as_ric_field(got, [RIC_A], FIELDS)
        assert out is not None
        assert list(out.columns.get_level_values(0).unique()) == [RIC_A]
        assert set(out.columns.get_level_values(1)) == set(FIELDS)

    def test_one_ric_one_field(self):
        got = pd.DataFrame([[1.0], [1.3]], index=IDX, columns=["BID"])
        got.columns.name = RIC_A
        out = F._as_ric_field(got, [RIC_A], ["BID"])
        assert list(out.columns) == [(RIC_A, "BID")]

    def test_many_rics_one_field_columns_are_rics(self):
        got = pd.DataFrame([[1.0, 2.0], [1.3, 2.3]], index=IDX, columns=[RIC_A, RIC_B])
        got.columns.name = "BID"
        out = F._as_ric_field(got, [RIC_A, RIC_B], ["BID"])
        assert set(out.columns) == {(RIC_A, "BID"), (RIC_B, "BID")}

    def test_never_puts_a_field_name_on_the_ric_level(self):
        """
        The shipped bug, stated as an assertion. Every label on the RIC level
        must be a RIC we asked for -- never 'BID'.
        """
        got = pd.DataFrame([[1.0, 1.2, 1.1], [1.3, 1.4, 1.35]], index=IDX, columns=FIELDS)
        got.columns.name = RIC_A
        out = F._as_ric_field(got, [RIC_A], FIELDS)
        rics = set(map(str, out.columns.get_level_values(0)))
        assert rics <= {RIC_A}
        assert not (rics & set(FIELDS)), f"field names leaked onto the RIC level: {rics}"

    def test_never_returns_more_series_than_were_requested(self):
        """
        The symptom that exposed the bug: 20 RICs in, 26 'series' out. A
        response can carry fewer than requested; never more.
        """
        got = pd.DataFrame([[1.0, 1.2, 1.1], [1.3, 1.4, 1.35]], index=IDX, columns=FIELDS)
        got.columns.name = RIC_A
        out = F._as_ric_field(got, [RIC_A], FIELDS)
        assert out.columns.get_level_values(0).nunique() <= 1

    def test_accepts_a_ric_field_multiindex_unchanged(self):
        cols = pd.MultiIndex.from_product([[RIC_A, RIC_B], FIELDS])
        got = pd.DataFrame(np.arange(12.0).reshape(2, 6), index=IDX, columns=cols)
        out = F._as_ric_field(got, [RIC_A, RIC_B], FIELDS)
        assert out.columns.names == ["RIC", "Field"]
        assert set(out.columns.get_level_values(0)) == {RIC_A, RIC_B}

    def test_swaps_a_field_ric_multiindex(self):
        """LSEG also emits (Field, RIC). Membership decides, not position."""
        cols = pd.MultiIndex.from_product([FIELDS, [RIC_A, RIC_B]])
        got = pd.DataFrame(np.arange(12.0).reshape(2, 6), index=IDX, columns=cols)
        out = F._as_ric_field(got, [RIC_A, RIC_B], FIELDS)
        assert set(out.columns.get_level_values(0)) == {RIC_A, RIC_B}
        assert set(out.columns.get_level_values(1)) == set(FIELDS)

    def test_refuses_to_guess_an_unrecognisable_shape(self):
        """
        Returning None sends the caller to the per-field fallback. Guessing
        here is what writes a mislabelled cache.
        """
        got = pd.DataFrame([[1.0, 2.0]], index=IDX[:1], columns=["mystery", "columns"])
        assert F._as_ric_field(got, [RIC_A, RIC_B], FIELDS) is None

    def test_refuses_to_guess_even_for_a_single_ric(self):
        """
        Bisection ends at batches of one, so the single-RIC path runs far more
        often than any other. If it relabels whatever it is handed, a response
        of unrecognisable columns becomes a confidently mislabelled frame.
        """
        got = pd.DataFrame([[1.0, 2.0]], index=IDX[:1], columns=["mystery", "columns"])
        assert F._as_ric_field(got, [RIC_A], FIELDS) is None

    def test_a_single_ric_response_keeps_the_ric_on_the_ric_level(self):
        got = pd.DataFrame([[1.0, 1.2, 1.1]], index=IDX[:1], columns=FIELDS)
        got.columns.name = RIC_A
        out = F._as_ric_field(got, [RIC_A], FIELDS)
        assert list(out.columns.get_level_values(0).unique()) == [RIC_A]
        assert set(out.columns.get_level_values(1)) == set(FIELDS)
        # and the levels are not the other way round
        assert out.columns.names == ["RIC", "Field"]

    def test_empty_and_none_are_not_shapes(self):
        assert F._as_ric_field(None, [RIC_A], FIELDS) is None
        assert F._as_ric_field(pd.DataFrame(), [RIC_A], FIELDS) is None


class TestChainRics:
    def test_bands_around_the_observed_range(self):
        rics = F.chain_rics("AAPL", dt.date(2026, 9, 4), 300.0, 310.0,
                            step=2.50, band=5.0)
        from trading_app.lib.ric import parse_option_ric
        ks = sorted(parse_option_ric(r)["strike"] for r in rics)
        assert min(ks) == pytest.approx(295.0)
        assert max(ks) == pytest.approx(315.0)
        assert all(abs((b - a) - 2.50) < 1e-9 for a, b in zip(ks, ks[1:]))

    def test_every_generated_ric_parses_and_is_a_call(self):
        from trading_app.lib.ric import parse_option_ric
        for expiry in (dt.date(2026, 8, 7), dt.date(2026, 9, 4), dt.date(2026, 7, 17)):
            for r in F.chain_rics("AAPL", expiry, 300.0, 310.0, 2.50, 5.0):
                got = parse_option_ric(r)
                assert got is not None, f"{r} does not parse"
                assert got["cp"] == "C"
                assert got["expiry"] == expiry

    def test_single_digit_expiry_days_are_padded(self):
        """Aug 7 and Sep 4 are single-digit Fridays in the backtest window."""
        for expiry in (dt.date(2026, 8, 7), dt.date(2026, 9, 4)):
            for r in F.chain_rics("AAPL", expiry, 300.0, 305.0, 2.50, 0.0):
                body = r.split(".")[0]
                assert len("".join(c for c in body if c.isdigit())) == 9, r


class TestCalendarIsNotDuplicated:
    def test_the_fetcher_does_not_define_its_own_week_calendar(self):
        """
        It used to, with the guard spelled differently. The two agreed, which
        is how a duplicate survives long enough to drift apart.
        """
        src = (ROOT / "scripts" / "fetch_hw2.py").read_text(encoding="utf-8")
        assert "def trading_weeks(" not in src

    def test_both_call_shapes_give_the_same_weeks(self):
        from trading_app.lib.covered_call import trading_weeks
        days = pd.date_range("2026-06-29", "2026-07-10", freq="B")
        frame = pd.DataFrame({"date": [d.date() for d in days]})
        a = trading_weeks(frame)
        b = trading_weeks(pd.DatetimeIndex(days))
        assert a == b


class TestDryRun:
    def test_dry_run_needs_no_session_and_lists_the_weeks(self):
        """
        This path broke once and nothing noticed, because nothing ran it. It
        takes no credentials, so there is no excuse for not running it.
        """
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "fetch_hw2.py"), "--dry-run"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, f"dry run failed:\n{r.stdout}\n{r.stderr}"
        assert "10 weeks" in r.stdout
        assert "2026-W27" in r.stdout
        assert "Traceback" not in r.stderr

    def test_dry_run_accepts_the_minute_interval(self):
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "fetch_hw2.py"),
             "--dry-run", "--interval", "1min"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"


class _StubLseg:
    """
    Stands in for lseg.data. `dead` RICs raise, as a real one does for a
    contract that never listed; everything else returns a healthy
    (RIC, Field) frame. Records every universe it was asked for, so the
    bisection can be counted rather than assumed.
    """

    def __init__(self, dead=(), shape="multi"):
        self.dead = set(dead)
        self.shape = shape
        self.calls = []

    def get_history(self, universe, fields, start, end, interval):
        self.calls.append(tuple(universe))
        if self.dead & set(universe):
            raise RuntimeError("LDError: no data for one or more instruments")
        idx = pd.DatetimeIndex(["2026-08-31 15:00", "2026-08-31 16:00"])
        if self.shape == "flat_one_field" and len(fields) == 1 and len(universe) > 1:
            df = pd.DataFrame(1.0, index=idx, columns=list(universe))
            df.columns.name = fields[0]
            return df
        if len(universe) == 1:
            df = pd.DataFrame(1.0, index=idx, columns=list(fields))
            df.columns.name = universe[0]
            return df
        cols = pd.MultiIndex.from_product([list(universe), list(fields)])
        return pd.DataFrame(1.0, index=idx, columns=cols)


def _stats():
    return {"threw": 0, "collapsed": 0, "unresolved": 0, "dead_rics": []}


class TestPullBatchBisection:
    RICS = [f"AAPLI0426{k:05d}.U^I26" for k in range(30000, 30800, 100)]   # 8 RICs

    def test_a_clean_batch_is_one_request(self):
        ld = _StubLseg()
        st = _stats()
        out = F.pull_batch(ld, self.RICS, ["BID", "ASK"], "2026-08-31", "2026-09-01", st)
        assert len(ld.calls) == 1
        assert out.columns.get_level_values(0).nunique() == len(self.RICS)
        assert st["threw"] == 0

    def test_one_dead_ric_is_bisected_not_exploded(self):
        """
        The point of bisecting: N dead RICs cost O(N log B) requests. The
        first version of this fetcher fell back to one request per
        (RIC, field) and spent 160 requests on a single bad strike.
        """
        ld = _StubLseg(dead={self.RICS[3]})
        st = _stats()
        out = F.pull_batch(ld, self.RICS, ["BID", "ASK"], "2026-08-31", "2026-09-01", st)
        assert len(ld.calls) < 16, f"took {len(ld.calls)} requests; that is not bisection"
        got = set(map(str, out.columns.get_level_values(0)))
        assert self.RICS[3] not in got
        assert got == set(self.RICS) - {self.RICS[3]}
        assert st["dead_rics"] == [self.RICS[3]]

    def test_a_batch_of_one_dead_ric_returns_nothing_and_is_recorded(self):
        ld = _StubLseg(dead={self.RICS[0]})
        st = _stats()
        assert F.pull_batch(ld, [self.RICS[0]], ["BID"], "2026-08-31", "2026-09-01", st) is None
        assert st["dead_rics"] == [self.RICS[0]]

    def test_an_all_dead_batch_returns_nothing(self):
        ld = _StubLseg(dead=set(self.RICS))
        st = _stats()
        assert F.pull_batch(ld, self.RICS, ["BID"], "2026-08-31", "2026-09-01", st) is None
        assert set(st["dead_rics"]) == set(self.RICS)

    def test_never_returns_a_label_outside_the_batch(self):
        """The invariant the fetcher refuses to write a cache without."""
        for dead in ((), {RIC_A}, {self.RICS[0], self.RICS[7]}):
            ld = _StubLseg(dead=dead)
            st = _stats()
            out = F.pull_batch(ld, self.RICS, ["BID", "ASK", "TRDPRC_1"],
                               "2026-08-31", "2026-09-01", st)
            if out is None:
                continue
            assert set(map(str, out.columns.get_level_values(0))) <= set(self.RICS)

    def test_a_collapsed_multi_ric_response_is_re_pulled_per_field(self):
        """
        A flat multi-RIC frame means fields were lost and the response cannot
        say which. HW1's fix, applied only where it is needed.
        """
        ld = _StubLseg(shape="flat_one_field")
        st = _stats()

        real = ld.get_history

        def collapsing(universe, fields, start, end, interval):
            if len(fields) > 1 and len(universe) > 1:      # lose all but one
                idx = pd.DatetimeIndex(["2026-08-31 15:00"])
                df = pd.DataFrame(1.0, index=idx, columns=list(universe))
                df.columns.name = fields[0]
                ld.calls.append(tuple(universe))
                return df
            return real(universe, fields, start, end, interval)

        ld.get_history = collapsing
        out = F.pull_batch(ld, self.RICS[:4], ["BID", "ASK", "TRDPRC_1"],
                           "2026-08-31", "2026-09-01", st)
        assert st["collapsed"] == 1
        assert out is not None
        assert set(out.columns.get_level_values(1)) == {"BID", "ASK", "TRDPRC_1"}
        assert set(map(str, out.columns.get_level_values(0))) <= set(self.RICS[:4])
