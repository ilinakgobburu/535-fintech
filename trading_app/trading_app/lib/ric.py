"""
OPRA-style RIC parsing and construction.

    {ROOT}{M}{DD}{YY}{SSSSS}.U^{M}{YY}

    ROOT    underlying root, uppercase          UUUU
    M       month letter, A-L calls / M-X puts  A = Jan call, M = Jan put
    DD      two-digit expiration day            15
    YY      two-digit year                      26
    SSSSS   strike x 100, zero-padded to 5      $12.50 -> 01250
    .U      exchange / venue qualifier
    ^{M}{YY} expired-contract suffix -- see the correction below

    UUUUA1502601250.U^A26  =  UUUU 15-Jan-2026 call struck at $12.50

CORRECTION TO THE PUBLISHED SCHEME. The assignment states that the expired
suffix "repeats the month letter". That is true for calls and WRONG for puts.
LSEG keys the ^ suffix off the expiry month's CALL letter for both rights, so
an August put carries body letter T and suffix ^H26, not ^T26:

    UUUUT212601200.U^H26   UUUU 21-Aug-2026 PUT  struck at $12.00   (39 obs)
    UUUUT212601200.U^T26   the documented form                      (no data)

Probed against LSEG across six expiry/strike pairs: the ^CALL form returned
data on five, the ^PUT form on none. Generating the documented form is why the
first pull came back calls-only, and validating against it is why the parser
would have discarded the puts even if they had arrived.
"""

from __future__ import annotations

import datetime as dt
import re

# A-L are calls Jan..Dec, M-X are puts Jan..Dec
CALL_CODES = {chr(ord("A") + i): i + 1 for i in range(12)}
PUT_CODES = {chr(ord("M") + i): i + 1 for i in range(12)}

CODE_TO_MONTH = {**CALL_CODES, **PUT_CODES}
CODE_TO_CP = {**{c: "C" for c in CALL_CODES}, **{c: "P" for c in PUT_CODES}}
MONTH_TO_CODE = {
    "C": {v: k for k, v in CALL_CODES.items()},
    "P": {v: k for k, v in PUT_CODES.items()},
}

# The .U and the ^MYY suffix are both optional so this also parses live RICs.
RIC_RE = re.compile(
    r"^(?P<root>[A-Za-z]+)"
    r"(?P<code>[A-Xa-x])"
    r"(?P<day>\d{2})"
    r"(?P<year>\d{2})"
    r"(?P<strike>\d{5})"
    r"(?:\.(?P<venue>[A-Za-z]+))?"
    r"(?:\^(?P<sfx_code>[A-Xa-x])(?P<sfx_year>\d{2}))?$"
)


def expired_suffix_code(month: int) -> str:
    """
    The letter LSEG puts after the ^ on an expired option RIC.

    It is the expiry month's CALL letter for calls AND puts. The assignment
    says the suffix repeats the body's own month letter, which silently
    produces put RICs that do not resolve.
    """
    return MONTH_TO_CODE["C"][month]


def parse_option_ric(ric: str) -> dict | None:
    """
    Explode one RIC into {underlying, expiry, put/call, strike}.

    Returns None -- never raises -- when the string is not a parseable option
    RIC, because a synthetic universe is full of identifiers that never
    existed and callers want to skip them, not handle exceptions.
    """
    text = str(ric).strip()
    m = RIC_RE.match(text)
    if not m:
        return None

    code = m.group("code").upper()
    month = CODE_TO_MONTH.get(code)
    cp = CODE_TO_CP.get(code)
    if month is None or cp is None:
        return None

    year = 2000 + int(m.group("year"))
    day = int(m.group("day"))
    try:
        expiry = dt.date(year, month, day)
    except ValueError:
        # e.g. a generated "Feb 30" candidate
        return None

    # Validate the expired suffix. The correct LSEG form is the expiry month's
    # CALL letter regardless of right; the form the assignment documents (the
    # body letter repeated) is accepted too, so caches built against the old
    # convention still parse instead of silently vanishing.
    sfx = m.group("sfx_code")
    suffix_code = None
    if sfx:
        sfx = sfx.upper()
        suffix_code = sfx
        if sfx not in (expired_suffix_code(month), code):
            return None

    return {
        "ric": text,
        "underlying": m.group("root").upper(),
        "cp": cp,
        "expiry": expiry,
        "strike": int(m.group("strike")) / 100.0,
        "month_code": code,
        "venue": (m.group("venue") or "").upper() or None,
        "expired": bool(sfx),
        "suffix_code": suffix_code,
    }


def build_option_ric(
    root: str,
    expiry: dt.date,
    strike: float,
    cp: str,
    venue: str = "U",
    expired: bool = True,
) -> str:
    """Inverse of parse_option_ric. Used to generate the candidate universe."""
    cp = cp.upper()
    if cp not in ("C", "P"):
        raise ValueError(f"cp must be 'C' or 'P', got {cp!r}")
    code = MONTH_TO_CODE[cp][expiry.month]
    body = (
        f"{root.upper()}{code}"
        f"{expiry.strftime('%d')}{expiry.strftime('%y')}"
        f"{int(round(strike * 100)):05d}"
    )
    ric = f"{body}.{venue}"
    if expired:
        # The suffix keys off the expiry month's CALL letter for BOTH rights.
        # See the module docstring: the published scheme is wrong here, and
        # emitting ^{put letter} returns no data at all.
        ric = f"{ric}^{expired_suffix_code(expiry.month)}{expiry.strftime('%y')}"
    return ric
