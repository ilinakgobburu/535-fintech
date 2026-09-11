# Algorithmic Trading II — semester app

MEng FinTech · Algorithmic Trading II. One app, grown one homework at a time.

**Published site:** <https://ilinakgobburu.github.io/535-fintech/>

---

## Assignment 2 — Covered Call Backtest

**Published page:** [`docs/hw2.html`](https://ilinakgobburu.github.io/535-fintech/hw2.html)

Long 100 AAPL shares, short one weekly call, ten cycles, no rolls. The strategy
is small enough to state in a sentence, so the whole result lives in the
decisions around it — which strike, at what price, at which moment — and the
page books all three explicitly and then measures what each was worth.

```
trading_app/
  trading_app/lib/covered_call.py   calendar, strike rules, blotter, ledger, Reg T
  trading_app/lib/cc_analysis.py    mid-vs-print fits, the sweeps, OHLC integrity
  scripts/fetch_hw2.py              hourly LSEG pull, banded chains
  scripts/build_hw2.py              -> docs/hw2.html
  scripts/hw2_app.js                the page's figures and computed prose
  scripts/bar_size_study.py         hourly vs a 1-minute re-pull -> a <1 KB committed JSON
  scripts/mutation_check.py         re-introduces 16 bugs, asserts the suite catches each
  tests/test_covered_call.py        47 tests
docs/hw2.html                       <- the graded artefact
```

```bash
cd trading_app
python3 -m pytest tests -q             # 171 tests (124 from 1.1, 47 here)
python3 scripts/mutation_check.py      # 16 mutations, all caught
python3 scripts/build_hw2.py           # rebuild the page from the cached pull
```

### The result

AAPL ran **$275.73 → $319.76 (+16.0%)** over the ten weeks. The book wrote 10
calls, collected **$3,567** of premium, and was **assigned 8 times**. It finished
**+$682** against **+$3,826** for the same 100 shares simply held.

That gap is the entire subject. Nearest-OTM sells a cap a median of **0.31%**
above spot, and AAPL's weekly range is far wider than 0.31%, so assignment is
close to the default outcome rather than the exception. On the 8 assigned weeks
the stock closed **$7,289** above the strikes sold — $3,567 of certain income did
not pay for $7,289 of surrendered upside. Every counterfactual strike rule beat
the booked one, and **none of them beat buy-and-hold**, monotonically in distance
from spot. None of that is a discovery about covered calls; it is a description
of what a cap does to a stock that rose 16% through the window, and a flat or
falling tape would invert the ordering.

### Two things the handout gets wrong

**The expiry day is zero-padded.** The scheme says *"DAY not zero-padded"*. Two
of the three AAPL sample RICs it prints do not resolve:

```
AAPLF52619000.U^F26    handout form    LDError
AAPLF052619000.U^F26   zero-padded     400 observations
AAPLH72620500.U^H26    handout form    LDError
AAPLH072620500.U^H26   zero-padded     478 observations
```

The third example expires on the 17th, so the rule never bites. Every testable
case fails; every padded correction works. **Aug 7 and Sep 4 are single-digit
Fridays in this very window**, so a literal reading silently drops 2 of 10
cycles — the strikes come back empty and the weeks look quiet. Same shape as the
1.1 put-wing finding: "no data came back" and "I asked the wrong question" are
indistinguishable from the outside.

**"Buy Monday, expire Friday" is a description, not a rule.** Jun 19 (Juneteenth)
and Jul 3 (Jul 4 observed) are closed, and those weeks expire on the *Thursday* —
the Thursday RIC resolves and the Friday one does not. The loop reads the
underlying's own session calendar and takes the first and last session of each
ISO week, so a holiday shifts the cycle instead of deleting it. It fires once
here, on **2026-W27 → Thu Jul 2**.

### A flat LSEG response means the opposite thing depending on what you asked for

1.1 documented that a multi-field request losing all but one field returns flat
columns of bare RICs. Probing again turned up the sharper rule — flat columns
carry whichever axis has more than one member, and **when both are singletons the
columns are fields**:

```
1 RIC,  3 fields -> columns ['BID','ASK','TRDPRC_1'], columns.name = the RIC
2 RICs, 1 field  -> columns [ric, ric],               columns.name = 'BID'
```

That is a trap for the bisection the fetcher uses to skip strikes that never
existed, because bisection drives batches to size one and flips the meaning of
the response underneath itself. The first version labelled field names as RICs
and reported 26 live series out of 20 requested — the only reason it was caught.
Labels are now resolved by *membership* in the known batch and known field list,
never by position, and the fetcher refuses to write a cache containing any label
it did not ask for.

### The bar extremes carry bad prints; the last-trade series does not

`HIGH_1` runs more than 1% above the bar's own open/close body on **12.8%** of
the 400 hourly bars and `LOW_1` more than 1% below on **21.2%**, reaching +10.6%
and −18.0% — one hour that opened and closed near $301 reports a high of $333.
`TRDPRC_1` shows nothing of the kind (median hourly move 0.26%, p99 2.05%).

So entry and settlement read **TRDPRC_1 and never HIGH_1/LOW_1**. Any rule phrased
as *"did the stock touch the strike"* would have booked assignments against trades
that never happened, and would have looked entirely reasonable doing it. A test
pins it: a bar whose `HIGH_1` is far through the strike but whose closing print is
below it must expire.

### The mid tracks the print. That is not the same as being fillable.

Pooled R² is **0.9992**, which alone proves little on a chain spanning $0.01 to
$99. I expected conditioning to collapse it the way pooling inverted 1.1's spread
conclusion. **It did not** — R² holds between **0.952 and 0.998** inside narrow
price bands, and 0.9965 in the band the book actually wrote in. The mid really
does track the print, and that is reported here because it contradicted the
expectation rather than because it flattered it.

What it does not establish is fillability, and the spread is what separates the
two claims. The median print missed the mid by **$0.035** on a median spread of
**$0.20** — 30% of the spread — only **27.4%** landed within a quarter-spread of
the mid, and **16.2% landed outside the quote entirely**. That last number is the
hourly bar, not an arbitrage: BID/ASK is the quote at the end of the hour while
TRDPRC_1 is the last trade inside it. R² near 0.999 and a mid that is the actual
trade price about a third of the time are both true at once, because R² is
answering "how big is this option" and the fill question is "who paid the spread".

### The parameter nobody declares

Writing at a different hour of the same entry session moves final P&L from
**+$682 to +$1,390** — a $708 spread around a booked result of $682. The
strategy is identical in every row; only the clock moves. The hour actually
booked, 15:00 UTC, turned out to be **the worst of the seven**, and it is left
standing because it was fixed before any of these numbers existed.

It did *not* outrank the strike rule, which spans $2,631. I expected the reverse
after watching one Monday's mid move 2× intraday, and one vivid observation
turned out to be a poor guide to the aggregate.

### Selling the cap the market prices at a 25% chance of being breached

A fixed-distance rule sells the same cap in a calm week and a violent one. Two
counterfactual rules instead back the week's at-the-money implied vol out of the
chain — reusing the Black-76 inversion from 1.1, with `F = spot` and `D = 1`,
which 1.1's own parity fit justifies at this horizon — and solve

```
K* = S · exp( σ√T · N⁻¹(1 − p)  −  σ²T/2 )
```

for a stated breach probability `p`. Implied vol ran **24.3%–47.1%**, and the
rule did widen when the week was priced to move: in the 47.1% week it pushed the
strike to its furthest, 3.83% out.

**It did not beat a fixed 2% rule** — $53,170 against $53,313, which over ten
weeks is noise. The valuable output was the calibration:

| target breach probability | realised assignment |
|---|---|
| 25% | **40%** |
| 15% | **20%** |

Both under-predicted, and they were supposed to. **The probability an option
price implies is risk-neutral, and the risk-neutral measure has zero drift by
construction.** This tape had +16%. A cap 25% likely to be breached by a
driftless stock is a good deal more likely to be breached by one marching
upward. Anyone reading an option-implied probability as a forecast should read
those two rows first.

### Checking the bar itself against a 1-minute re-pull

Two claims rested on what an hourly bar *is*. Both were assumptions, so the same
contracts were re-pulled at one minute (589,081 two-sided quoted bars) and both
were measured.

**An hourly BID/ASK is exactly the last minute's quote** — 100.0% of 10,571
matched contract-hours, against 23.8% for the hour's lowest bid. It is a
snapshot at the close of the bar, not an aggregated envelope over it. Had it
been an envelope, every "mid" in this backtest would have been the midpoint of
an hour of quote range rather than a price anyone could trade against, and the
fill assumption would not have survived. The convention is now measured.

**Two-thirds of the impossible prints were the bar.** On identical
(contract, day) cells, prints landing outside their own bar's quote fall from
**14.0% hourly to 5.1% at one minute**. The remainder is the honest rate.

**And a trap that nearly produced a false claim.** Conditioned on bars that also
printed, the median spread is $0.200 hourly against $0.050 at one minute — which
reads as "minute data is four times cleaner". Measured *unconditionally* on the
same contracts and days the two agree exactly at **$0.300**. The difference is
entirely selection: 78% of hourly bars contain a trade against 37% of minute
bars, so "this bar printed" is a far more demanding filter at one minute and it
selects the liquid, tight-spread moments. Third time this hazard has appeared
across the two assignments.

The minute cache is ~390 MB and is **not** committed; `scripts/bar_size_study.py`
distils it to a <1 KB JSON that is, so the page builds without it and the numbers
stay checkable.

### The tests, and a bug in the thing that checks the tests

47 tests, split by failure mode: the RIC and calendar tests pin bugs that produce
*silence*, the blotter/ledger/Reg T tests pin bugs that produce a *plausible wrong
number*. `scripts/mutation_check.py` re-introduces 16 specific bugs one at a time
and asserts the suite fails on each. One of them was not caught on the first run
— the bar-size study could have compared unmatched contracts and no test would
have noticed — which is the entire reason the harness exists.

That harness had a bug worth recording. CPython validates a `.pyc` against the
source's *(mtime, size)*. Every mutation here is a same-length edit (`< 2` →
`< 0`) and mutate-then-restore happens within one second, so **both** fields
match — Python accepted bytecode compiled from the *mutated* source as valid for
the *restored* source. The mutation survived the restore, in bytecode, with
correct code on disk. It surfaced as a build reporting 11 trading weeks instead of
10. The harness now runs under `PYTHONDONTWRITEBYTECODE`, deletes the bytecode
regardless, and finishes by asserting the suite still passes clean.

### What this is evidence for

Ten weekly cycles on one name in one quarter, all sharing a single price path —
closer to one observation than to ten. Nothing here supports a claim about
covered calls in general. It supports something narrower and still worth having:
given this tape, these are exactly the trades the stated rules produce, this is
what they cost, and this is the order in which the decisions mattered — strike
distance first, order hour second, fill convention a distant third. The one
finding that travels beyond this window is the calibration gap, because it is a
statement about what a risk-neutral probability *is* rather than about AAPL.

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
python3 -m pytest tests -q      # 124 tests, no LSEG session needed
```

Two different failure modes are pinned here.

The RIC and loader tests pin bugs that produced **zero rows and no error
message** — the failure mode worth testing in a synthetic RIC universe, where
an exception is visible but an empty panel looks like a quiet day.

The metrics tests pin bugs that produce a **plausible wrong number**, which is
worse, because nothing about the page looks broken. The two required
statistics, the arbitrage rules and their put sign flip, the interpolation
bias, and the fill-location denominator each have a case whose answer is known
by construction. Every one of them was checked by re-introducing the bug and
confirming the test fails.

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
intercept `D·F`, so **the forward comes out of the option prices themselves**;
nothing on the site assumes a risk-free rate or a dividend.

The forward is well identified: the fit lands at spot `+$0.020`, which is what a
non-dividend payer over a few weeks should look like. **The discount factor is
not.** At these horizons `D` is within a whisker of 1 by construction, so the
slope carries almost no information about a rate: raw fits imply annualized rates
from `−2567%` to `+394%` for `dte ≤ 15`. The code pins `D = 1.0` whenever the raw
slope leaves `0.90 < D < 1.02`, which fires on **23 of 246 fits (9.3%)** — always
on the high side; the 0.90 floor never binds on this data — and that pinning drags
the median `D` from `1.00332` to `1.00075`. So `D = 1.00075` is not independent
validation; it is partly the guard rail. (A `D` above 1 implies a slightly
*negative* rate, which is itself a sign the slope is fitting noise.)

The defensible claim is the sharper one: **parity pins the forward tightly and has
no power to identify the discount factor at these maturities.**

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
is **20.3% of the mark**. Conditioned on moneyness, quotes on contracts that
never traded run consistently wider — up to 2× in the wings (calls; 1.4× puts).

> Read the aggregate carefully. Pooled across everything, untraded spreads look
> *narrower* (19.4% vs 21.0%) — the opposite of the truth. Untraded contracts
> cluster deep in the money, where the option is expensive enough that a fat
> spread is small as a percentage. Condition on moneyness or you invert the
> conclusion. The page says this out loud.

**Where the print landed inside the quote.** Only **29.3%** of 6,014 prints
landed near the mid, with bumps at both edges of the quote — 497 at the bid,
522 at the ask. The mid was not an achievable price for most of the trades that
actually happened, and the ones that missed it missed in both directions.

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
