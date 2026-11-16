#!/usr/bin/env python3
"""
Export the development panel the live model is fit on, from the development run's output.

    python3 export_vol_arb_panel.py --dev ../vol-arb-ml/cache/vol_arb_ml_dev.json --out data/vol_arb_dev_panel.json

The panel (format vol_arb_dev_panel/1, lib/vol_arb_live.export_panel) is one row per measured 2019-23
stock-cycle: id, stock, entry, expiry, the nine features and the label (null with no trade), plus the output's
sha256 and commit and the rung the development run chose for the test. Nothing is recomputed: it is the
output's own rows, through vol_arb_wf.panel(). The 2024-26 final test is not read.

After exporting, set config/vol_arb_live.yaml's `rung` to the choice this prints; vol_arb_orders.py refuses to
enter while the two disagree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "trading_app"))
from trading_app.lib import vol_arb_live as VL  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="the live model's development panel, from the development output")
    ap.add_argument("--dev", type=Path, required=True, help="the development output (cache/vol_arb_ml_dev.json)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "vol_arb_dev_panel.json")
    args = ap.parse_args(argv)
    raw = args.dev.read_bytes()
    try:
        panel = VL.export_panel(json.loads(raw), source=args.dev.name, sha256=VL.sha256_hex(raw))
    except VL.PanelError as exc:
        sys.exit(f"{args.dev}: {exc}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(panel, indent=1) + "\n")
    labelled = sum(r["label"] is not None for r in panel["rows"])
    print(f"wrote {args.out}: {len(panel['rows'])} rows ({labelled} labelled) from {args.dev.name} "
          f"(commit {panel['made_from']['commit']}); the development run chose {panel['choice_for_test']['chosen']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
