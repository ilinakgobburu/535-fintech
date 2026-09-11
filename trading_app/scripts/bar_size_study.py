"""
Compare the hourly panel against a 1-minute pull of the same contracts and
write the result as a small committed JSON.

The minute cache is ~390 MB, which is far past anything worth putting in a
repository, so it is NOT committed -- see .gitignore. What IS committed is the
few kilobytes of findings this produces, so the page builds without it and
anyone with an LSEG session can regenerate the cache and re-run this to check
the numbers.

    python3 scripts/fetch_hw2.py --interval 1min --band 8 --lookback-days 7 \\
        --batch-size 10 --out trading_app/data/covered_call_AAPL_1min.pkl
    python3 scripts/bar_size_study.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading_app.lib.cc_analysis import bar_size_study  # noqa: E402
from trading_app.lib.covered_call import load_cache, option_panel  # noqa: E402

DEFAULT_OUT = ROOT / "trading_app" / "data" / "bar_size_study.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hourly", type=Path,
                    default=ROOT / "trading_app" / "data" / "covered_call_AAPL.pkl")
    ap.add_argument("--minute", type=Path,
                    default=ROOT / "trading_app" / "data" / "covered_call_AAPL_1min.pkl")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.minute.exists():
        print(f"no minute cache at {args.minute}; nothing to do")
        return 1

    print("loading (the minute cache is large; this takes a moment)...")
    oh = option_panel(load_cache(args.hourly))
    om = option_panel(load_cache(args.minute))
    out = bar_size_study(oh, om)

    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    h, m, s = out["hourly"], out["minute"], out["snapshot"]
    print(f"\nwrote {args.out}")
    print(f"  matched cells                {out['matched_cells']:,} (contract, day)")
    if s:
        print(f"  hourly BID == last minute    {s['bid_is_last_pct']:.1f}% of "
              f"{s['matched_hours']:,} contract-hours")
        print(f"  hourly ASK == last minute    {s['ask_is_last_pct']:.1f}%")
    print(f"  outside the quote   hourly {h['outside_pct']:.1f}%  -> minute {m['outside_pct']:.1f}%")
    print(f"  median spread, ALL bars    {h['median_spread_all']:.3f} / {m['median_spread_all']:.3f}")
    print(f"  median spread, PRINTED     {h['median_spread_printed']:.3f} / "
          f"{m['median_spread_printed']:.3f}   <- selection, not bar size")
    print(f"  share of bars that printed {h['printed_share']:.0f}% / {m['printed_share']:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
