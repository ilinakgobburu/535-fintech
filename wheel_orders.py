#!/usr/bin/env python3
"""
Tonight's wheel order for the course server (Sanji), from LSEG.

Run by .github/workflows/wheel-orders.yml every weeknight at 21:00 New York time
and from its Run workflow button:
  1. read config/wheel_live.yaml and the book: mkt/positions.json and
     mkt/fills.json, which Sanji writes (cash only while they do not exist);
  2. log in to LSEG (lseg_login.py) and get the stock's last close, the
     3-month T-bill and the closing bid/ask of the candidate options;
  3. pick the strike with the backtest engine's own select() (lib/wheel.py):
     implied vol from each option's own mid, delta from that vol;
  4. write mkt/open_orders.json: one order for the next session, or [].

The decision rules are in trading_app/trading_app/lib/wheel_live.py, which
does no I/O and is tested in trading_app/tests/wheel/test_wheel_live.py.

Run it locally (LSEG login from the macOS Keychain):
    python3 wheel_orders.py
"""

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import lseg.data as ld
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "trading_app"))
from lseg_login import open_session  # noqa: E402
from trading_app.lib import wheel_live as L  # noqa: E402
from trading_app.lib.ric import build_option_ric  # noqa: E402
from trading_app.lib.wheel import CALL, PUT, select  # noqa: E402

CONFIG = ROOT / "config" / "wheel_live.yaml"
MKT = ROOT / "mkt"
NY = ZoneInfo("America/New_York")
BATCH = 10          # option RICs per get_history call
PAUSE = 1.0         # seconds between calls: the course guideline is at most 3 requests a second


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def last_bar(ric: str, field: str, decision_day: dt.date) -> tuple[dt.date, float]:
    """The last daily value of `field` on or before the decision day, dropping repeated dates."""
    df = ld.get_history(universe=ric, fields=[field], interval="daily", count=10)
    s = pd.to_numeric(df[field], errors="coerce")
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="first")].dropna()
    s = s[s.index.date <= decision_day]
    if s.empty:
        sys.exit(f"LSEG returned no {field} for {ric} up to {decision_day}")
    return s.index[-1].date(), float(s.iloc[-1])


def closing_quotes(root: str, expiry: dt.date, cp: str, strikes: list[float], day: dt.date) -> pd.DataFrame:
    """Closing bid, ask and mid on `day` for each strike that has a two-sided quote."""
    rics = {build_option_ric(root, expiry, k, cp, expired=False): k for k in strikes}
    rows, names = [], list(rics)
    for i in range(0, len(names), BATCH):
        batch = names[i:i + BATCH]
        try:
            df = ld.get_history(universe=batch, fields=["BID", "ASK"], interval="daily", count=5)
        except Exception as exc:   # every RIC in the batch unlisted: the library raises
            print(f"  no quotes for {batch[0]}..{batch[-1]} ({str(exc).splitlines()[0][:80]})")
            df = None
        if df is not None and not df.empty:
            df.index = pd.to_datetime(df.index)
            df = df[~df.index.duplicated(keep="first")]
            if not isinstance(df.columns, pd.MultiIndex):   # one RIC comes back with flat columns
                df.columns = pd.MultiIndex.from_product([[batch[0]], df.columns])
            at = df[df.index.date == day]
            for ric in batch:
                if at.empty or ric not in at.columns.get_level_values(0):
                    continue
                bid = pd.to_numeric(at[ric]["BID"], errors="coerce").iloc[0]
                ask = pd.to_numeric(at[ric]["ASK"], errors="coerce").iloc[0]
                if pd.notna(bid) and pd.notna(ask) and 0 < bid <= ask:
                    rows.append({"ric": ric, "cp": cp, "expiry": pd.Timestamp(expiry),
                                 "strike": rics[ric], "bid": bid, "ask": ask, "mid": (bid + ask) / 2})
        time.sleep(PAUSE)
    return pd.DataFrame(rows)


def write_orders(orders: list[dict], for_day: dt.date) -> None:
    MKT.mkdir(exist_ok=True)
    (MKT / "open_orders.json").write_text(json.dumps(orders, indent=2) + "\n")
    print(f"Wrote mkt/open_orders.json: {len(orders)} order(s) for {for_day}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"for_day={for_day.isoformat()}\n")


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    u, s, x = cfg["underlying"], cfg["setting"], cfg["expiry"]
    now = dt.datetime.now(NY).replace(tzinfo=None)
    day, for_day = L.decision_dates(now)
    print(f"Run at {now:%Y-%m-%d %H:%M} New York: quotes from the close of {day}, order for {for_day}")

    book = L.read_book(read_json(MKT / "positions.json"), read_json(MKT / "fills.json"),
                       u["conid"], cfg["start_cash"])
    print(f"Book: cash {book.cash:,.2f}, {book.shares} shares, open options {len(book.open_options)}, "
          f"first entry {book.first_entry} -> phase {book.phase}")
    if book.phase == "hold":
        print("A short option is open and is held to expiry: no order.")
        return write_orders([], for_day)

    expiry = L.target_expiry(for_day, x["min_dte"], x["max_dte"],
                             first_entry=book.first_entry and x["first_entry_any_dte"])
    if expiry is None:
        print(f"No monthly expiry {x['min_dte']}-{x['max_dte']} days after {for_day}: wait for the next cycle.")
        return write_orders([], for_day)
    print(f"Target expiry {expiry} ({(expiry - for_day).days} days after {for_day})")

    floor = None
    if book.phase == "call" and s["floor"]:
        if book.cost_basis is None:
            sys.exit("Shares are held but mkt/positions.json gives no cost_basis, so the floor can't be applied.")
        floor = book.cost_basis

    session = open_session()
    try:
        bar_day, spot = last_bar(u["ric"], "TRDPRC_1", day)
        if bar_day != day:
            sys.exit(f"LSEG's last close for {u['ric']} is {bar_day}, not {day}: data not final yet, no order written.")
        _, rate_pct = last_bar(cfg["rate_ric"], "MID_YLD_1", day)
        step, band = cfg["strikes"]["step"], cfg["strikes"]["range_pct"] / 100
        lo, hi = spot * (1 - band), spot * (1 + band)
        strikes = [k * step for k in range(int(lo // step) + 1, int(hi // step) + 1)]
        cp = PUT if book.phase == "put" else CALL
        print(f"{u['ric']} closed at {spot:.2f} on {day}; T-bill {rate_pct:.2f}%; "
              f"asking LSEG for {len(strikes)} {'puts' if cp == PUT else 'calls'} {strikes[0]:g}-{strikes[-1]:g}")
        quotes = closing_quotes(u["root"], expiry, cp, strikes, day)
    finally:
        session.close()
        print("LSEG session closed")

    if quotes.empty:
        sys.exit(f"No two-sided closing quotes for the {expiry} {'puts' if cp == PUT else 'calls'}: no order written.")
    print(f"{len(quotes)} strikes quoted on {day}")

    target = s["put_delta"] if cp == PUT else s["call_delta"]
    pick = select(quotes, cp, pd.Timestamp(expiry), spot, pd.Timestamp(day),
                  rate_pct / 100, cfg["dividend_yield"], target, floor)
    if pick is None:
        why = f" at or above the cost basis {floor:.2f}" if floor is not None else ""
        print(f"No quoted strike{why}: the shares are held with no call this cycle.")
        return write_orders([], for_day)

    strike, mid = float(pick["strike"]), float(pick["mid"])
    qty = L.put_contracts(book.cash, strike) if cp == PUT else book.shares // L.CONTRACT
    print(f"Pick: {expiry} {strike:g} {'put' if cp == PUT else 'call'} ({pick['ric']}), delta {pick['delta']:+.3f}, "
          f"IV {pick['iv']:.1%}, bid {pick['bid']:.2f} / ask {pick['ask']:.2f}, mid {mid:.2f}")
    if qty < 1:
        print(f"Cash {book.cash:,.2f} does not secure one contract at {strike:g}: no order.")
        return write_orders([], for_day)

    order = L.build_order(root=u["root"], conid=u["conid"], for_day=for_day, cp=cp, qty=qty,
                          expiry=expiry, strike=strike, limit=mid)
    if cp == PUT:
        print(f"Sell {qty} to open at {order['lmt_prc']} or better: premium about {qty * 100 * mid:,.0f}, "
              f"collateral {qty * 100 * strike:,.0f}")
    else:
        print(f"Sell {qty} covered call(s) at {order['lmt_prc']} or better against {book.shares} shares")
    write_orders([order], for_day)


if __name__ == "__main__":
    main()
