"""
OPRA-style RIC parsing and construction.

    {ROOT}{M}{DD}{YY}{SSSSS}.U^{M}{YY}

    ROOT    underlying root, uppercase          UUUU
    M       month letter, A-L calls / M-X puts  A = Jan call, M = Jan put
    DD      two-digit expiration day            15
    YY      two-digit year                      26
    SSSSS   strike x 100, zero-padded to 5      $12.50 -> 01250
    .U      exchange / venue qualifier
    ^{M}{YY} expired-contract suffix, repeats month letter and year

    UUUUA1502601250.U^A26  =  UUUU 15-Jan-2026 call struck at $12.50
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

    # If the expired suffix disagrees with the body, the RIC is malformed.
    sfx = m.group("sfx_code")
    if sfx and sfx.upper() != code:
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
        ric = f"{ric}^{code}{expiry.strftime('%y')}"
    return ric
