# MENG FinTech · Algorithmic Trading II

# Assignment 1.2 — Covered Call Backtest (FINTECH 535)

> Assignment text as posted, kept here beside the code it specifies, as
> `options_surface_lab/README.md` is for 1.1. This is the summary version; the
> course's longer `assignment.html` is not saved in this repo.

**Turn in:** your GitHub Pages URL. The published site is 100% of the grade. I
will not grade a data fetcher.

Non-enabling example (illustrative numbers, not a live LSEG pull):
<https://jakevestal.github.io/535_fintech/>
Full write-up: `assignment.html`

Graded on logical consistency (no impossible trades; reasonable fills; the
strategy does what you say), clarity, and the requirements below. A basic covered
call is enough this week.

## What you are building

A covered call backtest: long 100 shares, short 1 call, one name. Prefer liquid
weeklies (AAPL, MSFT, NVDA, SPY, QQQ). You must have a clear, algorithmic rule for
choosing the strike.

A full-credit baseline is a loop over history that records:

- **Monday:** if flat, buy 100 shares at the stock print (decrease cash).
- **Same Monday:** write 1 call expiring that Friday, nearest OTM (or ATM if spot
  sits on a strike). Increase cash by 100 × mid. Limit at mid = (BID+ASK)/2. No
  bid/ask → skip the week.
- **Friday expiry.** OTM: expire, keep shares, keep premium; next Monday just write
  the next Friday call. ITM: you are assigned — you own stock, you are assigned a
  short, you are flat. Increase cash by 100 × strike. Next Monday start the combo
  again.

Wait through expiry. No rolls, no buy-to-close. The point of the assignment is the
small decisions (especially strike) and communicating them on the site.

## Fills

Default: option fill at mid = (BID+ASK)/2 at the order timestamp. Stock at the
print. That prices the combo any time of day.

Optional: fit a vol curve to prints and assume that fill — show the fit.

No bid/ask → no fill. Do not invent a print.

## LSEG expired options

BID and ASK down to 1-minute even when expired. When a trade happens: `TRDPRC_1`,
`OPEN_PRC`, `HIGH_1`, `LOW_1`, `ACVOL_UNS`, `NUM_MOVES`. Same bar size for stock and
options (hourly default, ~10 weeks).

Fetch near-the-money calls along the chain. Graph `TRDPRC_1` (Y) vs mid (X), fit a
line, report R².

## Blotter (non-negotiable)

Trades you actually booked: time, instrument, side (BUY / SELL / EXPIRE / ASSIGN),
qty, limit/fill, cash delta, note pointing at the rule. Working orders do not
belong here. A signal chart is not a blotter.

## Ledger and Reg T

Ledger = shares, short calls (strike + expiry), cash, marks. Use Reg T, not
portfolio margin (computable without a broker PM engine; a book that lives in
Reg T lives in PM, not the reverse).

```
NAV         = cash + stock MV + option MV   (short call is negative)
LMV         = shares × stock mark
Initial     = 50% of stock LMV.  Covered short call adds $0
Maintenance = 25% of stock LMV (FINRA)
Available funds = NAV − initial
Excess          = NAV − maintenance
```

Cash moves only on blotter events: buy stock, collect premium, expire at 0,
assignment at strike.

Plot NAV and margin with mouseover. If available funds go negative, you could not
have put the trade on — say so.

## Option RICs

```
{ROOT}{MONTH}{DAY}{YY}{STRIKE}.U{^MONTHYY if expired}
```

|      | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Call | A   | B   | C   | D   | E   | F   | G   | H   | I   | J   | K   | L   |
| Put  | M   | N   | O   | P   | Q   | R   | S   | T   | U   | V   | W   | X   |

- `ROOT` — RIC before the dot (`AAPL.O` → `AAPL`)
- `DAY` not zero-padded; `STRIKE` = strike × 100, 5 digits; `.U` = OPRA
- Expired only: `^{MONTH}{YY}` with the same letter

Example: `UUUUH212601450.U^H26` (UUUU 14.5C 21 Aug 2026, expired).
AAPL sample: `AAPLF52619000.U^F26`, `AAPLG172620000.U^G26`, `AAPLH72620500.U^H26`.

## Pages vs local Data

Pages is graded. Local Data is a tool. On github.io, Data must show **Data
connection required**.

```bash
python3 helios/python/local_server.py
# http://127.0.0.1:8765/helios/data.html
```

Bake JSON into `book.js`. HTML/CSS/JS. Node not required. You may use AI; the book
and the reasoning are yours.

## Published site must contain

- Blotter — every entry and exit
- Ledger — stock, short calls, cash
- Reg T accounts, used correctly
- NAV path (and margin) with mouseover
- Mid vs trade scatter + R²
- Write-up: strike choice, wait-through-expiry (OTM expire / ITM assigned → flat),
  fill at mid, why Reg T

## Rubric (100 points — published site only)

| Criterion | Points |
|---|---|
| Algorithmic entry + strike rule; wait through expiry | 20 |
| Clean blotter + ledger that implement those rules; simulated limit fills | 25 |
| Reg T NAV / IM / MM / available funds / cash, used like an account | 20 |
| Mid vs `TRDPRC_1` scatter + R² (justify the fill assumption) | 15 |
| Analysis: what happened, where theory met tape, what you'd change | 20 |
