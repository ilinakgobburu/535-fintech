# Algorithmic Trading II — semester app

MEng FinTech · Algorithmic Trading II. One app, grown one homework at a time.

**Published site:** <https://ilinakgobburu.github.io/535-fintech/>

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
    lib/ric.py          OPRA RIC parse + build (see "the put wing" below)
    lib/loaders.py      LSEG pickle -> tidy long table -> wide (mark | print)
    lib/metrics.py      the two required statistics, interpolation, occupancy
    lib/vol.py          put-call parity, the implied forward, Black-76 and the
                        price-space vs vol-space comparison
    lib/plots.py        Plotly figures
    pages/              future homeworks land here
    data/               the cached LSEG pickles + the RIC-suffix probe
  scripts/fetch_lseg.py       re-pull from LSEG and write the pickle
  scripts/probe_ric_suffix.py reproduce the put-suffix finding against LSEG
  scripts/build_static.py     build the published page
  scripts/page_template.html
  tests/                      pytest — RIC round-trips, pricing math, loader shapes
docs/index.html         <- what GitHub Pages serves
docs/ccj.html           <- the control name
```

### Tests

```bash
cd trading_app
python3 -m pytest tests -q      # 108 tests, no LSEG session needed
```

Every bug these pin produced **zero rows and no error message**. That is the
failure mode worth testing here: an exception is visible, an empty panel looks
like a quiet day.

### Rebuild the site

```bash
pip install -r requirements.txt
cd trading_app
python3 scripts/build_static.py      # writes ../docs/index.html
```

No credentials needed — the build reads the committed pickle.

### Run the interactive version (optional)

```bash
cd trading_app
reflex run
```

The Reflex page is the interactive twin of the published one. Both call the same
`lib/` modules, so a number shown in one cannot drift from the other. Pages
serves the static build; Reflex is for local exploration and the live demo.

---

## The put wing was never missing

The first version of this project reported that the pull contained no puts and
called it an unexplained hole. It was not a hole in the data. The scheme in the
assignment says the expired-contract suffix `^{M}{YY}` "repeats the month
letter." **That is true for calls and wrong for puts.** LSEG keys the suffix off
the expiry month's *call* letter for both rights:

```
UUUUT212601200.U^H26   21-Aug-2026 PUT @ $12.00   39 observations   <- resolves
UUUUT212601200.U^T26   the documented form         no data          <- does not
```

Probed directly against LSEG on six expiry/strike pairs: the documented form
returned data on none, the corrected form on five
(`scripts/probe_ric_suffix.py`). The same assumption was wired into two places —
the RIC *generator* asked for contracts that do not exist, and `parse_option_ric`
*validated* against the same wrong rule, so the puts would have been discarded
even if they had arrived. Fixing one letter recovered **230 put series and
~20,000 observations**, roughly doubling the panel and making everything in the
next section possible.

The lesson is not that the handout has a typo. It is that "no data came back"
and "I asked the wrong question" are indistinguishable from inside a synthetic
universe, where empty responses are the *expected* case and cannot be treated as
a signal. The only defense is an independent check, and that check now lives in
`tests/test_ric.py`.

Appendix A has a second, harmless error: the worked example `UUUUA1502601250.U^A26`
carries ten digits in the body where the scheme on the same page specifies nine
(`DD`+`YY`+`SSSSS`). The self-consistent identifier is `UUUUA152601250.U^A26`.

## The forward, and the right space to interpolate in

With both rights in hand, put-call parity `C − P = D(F − K)` is an identity — no
model, no volatility. Fitting it across strikes returns the slope `−D` and the
intercept `D·F`, so **the forward and the rate come out of the option prices
themselves**; nothing on the site assumes a risk-free rate or a dividend. The fit
lands at spot `+$0.020` with `D = 1.00075`, which is what a non-dividend payer
over a few weeks should look like — a real validation, since the regression was
free to return anything.

Two things follow.

**Parity is a third, cleanest measure of the mid failing to be a price.** The
median residual is **$0.042** on an identity that should give zero, and only
**16 of 4,147** conversions survive paying the spread on all four legs. Same
verdict as the butterfly test and the fill-location histogram, reached without
volatility, interpolation, or a rate assumption.

**88.3% of one-sided holes never needed guessing.** Where one right is quoted and
the other is not, parity reconstructs the missing mark *exactly* — 1,709 of 1,936
cells. Worth asking, before interpolating anything, how many gaps were not gaps.

**And the interpolation bias has a fix.** The holdout section measures a
`+$0.015` bias from interpolating price and blames convexity. If that diagnosis
is right it names its own remedy: convexity belongs to the price, not the
contract, and implied vol is far flatter in strike. Running the identical test in
vol space — same cells, same triangulation, converted back to dollars through
Black-76 — gives:

| | median miss | bias |
|---|---|---|
| interpolating **price** | $0.0300 | **+$0.0150** |
| interpolating **implied vol** | $0.0247 | **+$0.0020** |

Read those in the right order. The median improves only ~18%, which is less than
the usual framing promises. The **bias falls by 87%**, and that asymmetry is the
result rather than a disappointment: random error comes from the holes being
wide and no change of variable can invent support that is not there, while
systematic error comes from drawing a straight line through a curve that bends.
Interpolating vol fixes the geometry and leaves the sparsity — which is the
assignment's actual subject.

The CCJ control replicates the part that matters. Its bias falls `+$0.0200` →
`+$0.0027`, an 86% collapse against UUUU's 87% — but its median only improves
8.8%, against UUUU's 17.7%. So the *systematic* gain is stable across names
while the *typical* gain is not, which is the same story told twice: changing
the space reliably removes the error that comes from geometry, and does
nothing dependable about the error that comes from sparsity.

---

## What the site shows

Beyond the required 3D surface, the two fields and the two statistics, three
panels go after the question the assignment poses and declines to answer —
*what price would you actually get filled at?*

**The spread is the error bar on the mark.** `BID`/`ASK` were pulled alongside
the required fields, so the mark's uncertainty is measurable: the median quote
is **23.8% of the mark**. Conditioned on moneyness, quotes on contracts that
never traded run consistently wider — up to 2× in the wings.

> Read the aggregate carefully. Pooled across everything, untraded spreads look
> *narrower* (22.2% vs 25.9%) — the opposite of the truth. Untraded contracts
> cluster deep in the money, where the option is expensive enough that a fat
> spread is small as a percentage. Condition on moneyness or you invert the
> conclusion. The page says this out loud.

**Where the print landed inside the quote.** Only **27.9%** of 2,927 prints
landed near the mid, with a pile-up at the ask. The mid was not an achievable
price for most of the trades that actually happened.

**How wrong interpolation really is.** Each observed cell is hidden in turn,
rebuilt from its neighbors, and compared to the truth. Median miss **$0.030** —
the same order as the entire mark-versus-trade effect — with a **+$0.015 bias**.
That bias is convexity: the chord sits above the curve, so interpolation
*overshoots*. And these are best-case cells, ringed by real data; the holes you
would actually want to fill have far less support.

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
