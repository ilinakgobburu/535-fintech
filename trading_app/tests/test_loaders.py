"""
Column-shape tests for the LSEG frame reader.

Every bug pinned here produced ZERO rows and no error message. That is the
failure mode worth testing: a raised exception is visible, an empty panel
looks like a quiet day.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_app.lib.loaders import flatten_options  # noqa: E402

DATES = pd.to_datetime(["2026-08-03", "2026-08-04", "2026-08-05"])
CALL = "UUUUH212601200.U^H26"
PUT = "UUUUT212601200.U^H26"          # put body T, expiry-month suffix H


def _multi(cols):
    return pd.DataFrame(
        np.arange(len(DATES) * len(cols), dtype=float).reshape(len(DATES), -1),
        index=DATES, columns=pd.MultiIndex.from_tuples(cols, names=["RIC", "Field"]),
    )


def test_ric_field_multiindex_reads():
    df = _multi([(CALL, "MID_PRICE"), (CALL, "BID"), (PUT, "MID_PRICE")])
    tidy = flatten_options(df)
    assert set(tidy["ric"]) == {CALL, PUT}
    assert set(tidy["field"]) == {"MID_PRICE", "BID"}
    assert set(tidy["cp"]) == {"C", "P"}


def test_field_ric_multiindex_reads_the_other_way_round():
    """LSEG returns either ordering depending on the call."""
    df = _multi([("MID_PRICE", CALL), ("BID", CALL)])
    df.columns.names = ["Field", "RIC"]
    tidy = flatten_options(df)
    assert set(tidy["ric"]) == {CALL}
    assert set(tidy["field"]) == {"MID_PRICE", "BID"}


def test_one_stray_field_label_on_the_ric_level_does_not_swap_the_levels():
    """
    The regression that emptied the whole panel.

    Merging a pull produced a single rogue ('BID','BID') column. The level
    resolver decided which level held the field by asking whether that level
    contained any known field NAME, so one stray 'BID' on the RIC level was
    enough to swap RIC and field. Every RIC then failed to parse and the app
    silently rendered nothing. Levels are now chosen by counting actual RIC
    parses, so hundreds of real RICs outvote one bad label.
    """
    df = _multi([("BID", "BID")] + [(CALL, f) for f in ("ASK", "BID", "MID_PRICE")]
                + [(PUT, f) for f in ("ASK", "BID", "MID_PRICE")])
    tidy = flatten_options(df)
    assert not tidy.empty, "one bad column label emptied the entire panel"
    assert set(tidy["ric"]) == {CALL, PUT}
    assert "BID" not in set(tidy["ric"])


def test_flat_columns_take_the_field_from_columns_name():
    """A single surviving field arrives as flat RIC columns + columns.name."""
    df = pd.DataFrame({CALL: [1.0, 2.0, 3.0]}, index=DATES)
    df.columns.name = "MID_PRICE"
    tidy = flatten_options(df)
    assert set(tidy["field"]) == {"MID_PRICE"}


def test_quotes_dated_after_expiry_are_dropped():
    late = pd.to_datetime(["2026-08-20", "2026-08-21", "2026-09-01"])
    df = pd.DataFrame([[1.0], [2.0], [3.0]], index=late,
                      columns=pd.MultiIndex.from_tuples([(CALL, "MID_PRICE")]))
    tidy = flatten_options(df)
    assert (tidy["dte"] >= 0).all()
    assert len(tidy) == 2          # the 01-Sep row is after the 21-Aug expiry


def test_empty_input_gives_an_empty_frame_not_an_exception():
    assert flatten_options(pd.DataFrame()).empty
    assert flatten_options(None).empty
