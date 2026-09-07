"""
RIC round-trip and rejection tests.

The expensive bug this course hands you is a silent one: a synthetic universe
generates identifiers that never existed, so "no data came back" and "I asked
for the wrong string" look identical. These tests pin the encoding so that a
whole wing cannot go missing again without something turning red.

    python -m pytest tests -q
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_app.lib.ric import (  # noqa: E402
    build_option_ric, expired_suffix_code, parse_option_ric,
)


def test_spec_example_parses():
    """
    The worked example in Appendix A, corrected.

    The assignment prints `UUUUA1502601250.U^A26` for the UUUU 15-Jan-2026 call
    struck at $12.50, but that body carries TEN digits where the scheme it
    states on the same page specifies nine: DD(2) + YY(2) + SSSSS(5). The
    self-consistent identifier is `UUUUA152601250.U^A26` -- 15, 26, 01250 --
    and that is the form that resolves. Second documentation error found in
    this scheme; the expired put suffix is the first, and the costly one.
    """
    assert parse_option_ric("UUUUA1502601250.U^A26") is None  # as printed
    got = parse_option_ric("UUUUA152601250.U^A26")            # as intended
    assert got is not None
    assert got["underlying"] == "UUUU"
    assert got["cp"] == "C"
    assert got["expiry"] == dt.date(2026, 1, 15)
    assert got["strike"] == 12.50
    assert got["expired"] is True


@pytest.mark.parametrize("month,letter", [
    (1, "A"), (6, "F"), (8, "H"), (12, "L"),
])
def test_expired_suffix_is_always_the_call_letter(month, letter):
    """
    The correction to the published scheme. LSEG keys the ^ suffix off the
    expiry month's CALL letter for BOTH rights; the documented "repeat the
    body letter" rule produces put RICs that return no data.
    """
    assert expired_suffix_code(month) == letter


def test_put_ric_uses_call_letter_in_suffix():
    ric = build_option_ric("UUUU", dt.date(2026, 8, 21), 12.0, "P")
    assert ric == "UUUUT212601200.U^H26"      # body T (Aug put), suffix H (Aug)
    assert ric != "UUUUT212601200.U^T26"      # the documented form: no data


def test_call_ric_body_and_suffix_agree():
    assert build_option_ric("UUUU", dt.date(2026, 8, 21), 12.0, "C") == \
        "UUUUH212601200.U^H26"


@pytest.mark.parametrize("cp", ["C", "P"])
@pytest.mark.parametrize("strike", [0.50, 7.0, 12.50, 13.0, 108.0, 999.50])
@pytest.mark.parametrize("expiry", [
    dt.date(2026, 1, 2), dt.date(2026, 6, 26),
    dt.date(2026, 8, 21), dt.date(2026, 12, 18),
])
def test_build_parse_round_trip(cp, strike, expiry):
    """Anything we generate must survive being parsed back, exactly."""
    ric = build_option_ric("UUUU", expiry, strike, cp)
    got = parse_option_ric(ric)
    assert got is not None, f"generated an unparseable RIC: {ric}"
    assert got["cp"] == cp
    assert got["expiry"] == expiry
    assert got["strike"] == pytest.approx(strike)
    assert got["underlying"] == "UUUU"


def test_legacy_put_suffix_still_parses():
    """
    A cache built before the fix carries ^{put letter}. Accept it so old
    pickles degrade to 'parsed' rather than 'silently dropped'.
    """
    assert parse_option_ric("UUUUT212601200.U^T26") is not None


@pytest.mark.parametrize("bad", [
    "UUUUT212601200.U^B26",     # suffix is neither the body nor the call letter
    "UUUUZ212601200.U^H26",     # Z is not a month code at all
    "UUUUB302601200.U^B26",     # 30 February
    "NOTARIC",
    "",
    "UUUU.K",                   # the underlying, not an option
])
def test_malformed_rics_return_none_not_raise(bad):
    """A synthetic universe is full of junk; callers skip, they do not catch."""
    assert parse_option_ric(bad) is None


def test_strike_encoding_is_hundredths():
    # UUUU | A (Jan call) | 15 (day) | 26 (year) | 01250 ($12.50)
    assert build_option_ric("UUUU", dt.date(2026, 1, 15), 12.50, "C") == \
        "UUUUA152601250.U^A26"
    assert parse_option_ric("UUUUA152600050.U^A26")["strike"] == 0.50
