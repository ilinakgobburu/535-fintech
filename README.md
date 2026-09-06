# Algorithmic Trading II — semester app

MEng FinTech · Algorithmic Trading II. One app, grown one homework at a time.

**Published site:** _(GitHub Pages URL goes here once the repo is up)_

---

## Assignment 1.1 — Option Surface Lab

Listed options are not a filled sheet. They are a sparse cloud with large,
structured holes. The site plots the two prices that are easy to confuse —
`MID_PRICE` (the closing NBBO midpoint, used as the mark) and `TRDPRC_1`
(the last trade) — and then shows exactly where each one stops existing.

Assignment text: [`options_surface_lab/README.md`](options_surface_lab/README.md)

### Layout

```
trading_app/
  trading_app/
    theme.py            graphical identity — palette, Plotly layout helpers
    lib/ric.py          OPRA RIC parse + build
    lib/loaders.py      LSEG pickle -> tidy long table -> wide (mark | print)
    lib/metrics.py      the two required statistics, interpolation, occupancy
    lib/plots.py        Plotly figures
    pages/              future homeworks land here
    data/               the cached LSEG pickle
  scripts/fetch_lseg.py     re-pull from LSEG and write the pickle
  scripts/build_static.py   build the published page
  scripts/page_template.html
docs/index.html         <- what GitHub Pages serves
```

### Rebuild the site

```bash
pip install -r requirements.txt
cd trading_app
python3 scripts/build_static.py      # writes ../docs/index.html
```

No credentials needed — the build reads the committed pickle.

### Re-pull the data (needs LSEG Workspace running)

```bash
cd trading_app
python3 scripts/fetch_lseg.py --dry-run   # count candidates, no session
python3 scripts/fetch_lseg.py             # real pull
python3 scripts/build_static.py
```

---

## Two notes on the data

**`SETTLE` does not exist for this RIC space.** LSEG exposes no true exchange
settlement price for expired US equity options — the field returns "universe
does not support," and `TR.SettlementPrice` comes back empty. `MID_PRICE` is
the closest mark-of-the-close available, and it is still a midpoint of a
possibly-stale quote.

**Fields are pulled one at a time, on purpose.** When several fields are
requested together and only one survives, LSEG returns *flat* columns of bare
RICs with the field name parked on `columns.name` — a shape indistinguishable
from a single-field pull. Requesting each field separately and assembling the
`(RIC, field)` MultiIndex locally makes it impossible to silently mislabel or
lose a field.
