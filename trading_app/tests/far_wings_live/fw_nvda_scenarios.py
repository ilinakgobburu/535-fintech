"""
NVDA nights whose orders were recorded with the single-name code (commit b5f201e, before the basket) in
nvda_before_basket.json, so the basket's code can be held to them (not collected: no test_ prefix). Each scenario is
the inputs of one far_wings_live.plan_night call on made-up data (fw_live_synth): entry nights on several smiles,
levels and spots, the band on several books and closes, a leg with no recent mid, and the expiry night's leftover
shares. Only made-up data; nothing here is market data.
"""

from __future__ import annotations

import datetime as dt

import fw_live_synth as S
from trading_app.lib import far_wings_live as FW

EXP = S.EXPIRY


def _book(options=(), shares=0):
    return FW.Book(cash=100_000.0, shares=shares,
                   options=[{"cp": cp, "expiry": EXP, "strike": float(k), "qty": q} for cp, k, q in options])


def scenarios():
    """[(label, plan_night's inputs)], the same on every call."""
    out = []
    for b in (-0.15, -0.06, 0.0, 0.05):
        for s0 in (0.30, 0.45, 0.70):
            out.append((f"entry b={b} s0={s0}", {"day": S.ENTRY, "for_day": S.FOR, "book": _book(),
                                                 "entry_expiry": EXP, "mkt": S.market(b=b, s0=s0)}))
    for px in (60.0, 120.0, 250.0, 420.0):
        c = S.closes(start_px=px)
        out.append((f"entry from {px}", {"day": S.ENTRY, "for_day": S.FOR, "book": _book(), "entry_expiry": EXP,
                                         "mkt": S.market(c=c)}))
    out.append(("entry with leftover shares", {"day": S.ENTRY, "for_day": S.FOR, "book": _book(shares=-350),
                                               "entry_expiry": EXP, "mkt": S.market()}))
    day = dt.date(2026, 12, 1)
    for last in (100.0, 150.0, 172.0, 175.0, 190.0, 205.0, 240.0, 400.0):
        c = S.closes(end=day, last=last)
        hist = {("P", EXP, 145.0): S.leg_history(c, "P", EXP, 145.0, 0.5),
                ("C", EXP, 220.0): S.leg_history(c, "C", EXP, 220.0, 0.45)}
        for shares in (0, 300, -300, 1200):
            out.append((f"band last={last} shares={shares}",
                        {"day": day, "for_day": FW.next_session(day),
                         "book": _book([("P", 145, -41), ("C", 220, -59)], shares),
                         "mkt": S.market(day, c=c, chain=False), "leg_history": hist}))
    c = S.closes(end=day, last=180.0)
    missing = set(c.index[-3:])
    hist = {("P", EXP, 145.0): S.leg_history(c, "P", EXP, 145.0, 0.55, missing=missing),
            ("C", EXP, 220.0): S.leg_history(c, "C", EXP, 220.0, 0.40)}
    out.append(("band with a missing mid", {"day": day, "for_day": FW.next_session(day),
                                            "book": _book([("P", 145, -41), ("C", 220, -59)], 0),
                                            "mkt": S.market(day, c=c, chain=False), "leg_history": hist}))
    out.append(("one wing assigned", {"day": day, "for_day": FW.next_session(day),
                                      "book": _book([("C", 220, -59)], -4100),
                                      "mkt": S.market(day, c=S.closes(end=day, last=140.0), chain=False),
                                      "leg_history": {("C", EXP, 220.0): S.leg_history(
                                          S.closes(end=day, last=140.0), "C", EXP, 220.0, 0.45)}}))
    out.append(("expiry night shares", {"day": EXP, "for_day": FW.next_session(EXP), "book": _book(shares=-250),
                                        "mkt": S.market(EXP, chain=False)}))
    out.append(("quiet night", {"day": dt.date(2026, 12, 2), "for_day": dt.date(2026, 12, 3), "book": _book()}))
    return out
