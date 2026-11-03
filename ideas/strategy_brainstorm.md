# Strategy brainstorm: implied vs realized vol, and the smile over time

Started 2026-11-03. **Notes and questions only.** Nothing here has been run, pulled or registered,
and no rule below is fixed. Anything that becomes a test is registered in `config/` before it runs,
as with every screen so far.

## The ideas, as first written

> - scroll thru refinitiv in a for loop
> - realized vol vs implied vol tracker. each point is an asset of consideration.
> - parabola (second-order equation) fit to the vol smile of an asset every day over time; report
>   the fit coefficients to understand how the curve changes over time.

They fit together as one pipeline:

```
for each asset in a universe          (1) loop over LSEG
    daily: implied vol, realized vol  (2) one point per asset on an IV-vs-RV chart
    daily: smile -> a, b, c           (3) parabola coefficients as time series
-> which assets look rich or cheap, and is that a trade?
```

---

## 1. Looping over LSEG (Refinitiv)

The loop itself is easy. The constraints are the hard part.

- **Cost differs a lot by what is pulled.** One `{ticker}ATMIV.U` RIC returns years of daily 30-day
  ATM implied vol in a single request, so an IV-vs-RV tracker on hundreds of stocks is cheap
  (`vol_screen_2023.py` already loops over 503 S&P 500 RICs this way). A smile needs **every option
  contract**, one RIC each, so it costs hundreds of requests per stock-year of monthlies. The daily
  cap is about 10,000 and is shared with everything else.
- **So:** the tracker can cover a wide universe, while smiles only fit a short list (or the stocks
  whose chains are already cached from the wheel pulls).
- **Count before pulling.** `vol_screen_2023.py --plan` reports what is cached and what is left
  without opening a session. Copy that pattern.
- **Resumable and cached.** Write every response to `cache/` as it arrives, and make the loop skip
  what is already there, so a cap hit or a dropped session only costs a restart.
- **RIC quirks.** A ticker may need `.N`, `.O` or `.OQ` (`premium_screen.history` handles the
  fallback), delisted names carry a `^` suffix, and some option roots differ from the ticker (BNY's
  options are under BK).
- **Survivorship.** Take the universe as it stood on a date, including names since removed (as
  `mr_screen.py` did), not today's index list.
- **Session.** `fetch_wheel_raw.open_session()` scans proxy ports, since Workspace sometimes comes
  up on 9001 rather than 9000.
- **Worth checking:**
  - which other fields `ATMIV.U` carries beyond the 30-day and 90-day call and put series (skew or
    other tenors would save chain pulls);
  - whether the Workspace session can call LSEG's volatility-surface analytics. That would give a
    fitted surface without pulling every contract, but the fit would be LSEG's model, not quotes.
- **Pages can't call LSEG.** Any tracker page is built from a cached pull, like every page so far
  (`docs/data.html`).

## 2. Implied vs realized vol tracker (one point per asset)

**The picture:** x = realized vol, y = implied vol, one dot per stock, with the 45° line drawn in.
Dots above the line have options priced above the moves the stock has been making. A slider or
animation over dates shows the dots drift.

**Decisions that change what the chart means:**

| Choice | Options | Note |
|---|---|---|
| Which realized vol | trailing 21 days (known today) / forward 21 days (known a month later) | Trailing is the **signal**; forward is the **outcome** a vol seller is paid on. Never put forward RV into an entry rule (look-ahead). `premium_screen.py` uses forward RV, because it scores the past. |
| Horizon match | iv30 (30 calendar days) ↔ 21 trading days of returns | Keep the two the same length, or the gap is partly a term-structure effect. |
| Estimator | close-to-close / Parkinson / Garman-Klass / Yang-Zhang | Close-to-close is what every screen here uses. The high-low estimators are less noisy but are biased by overnight gaps. |
| Gap measure | IV − RV (vol points) / (IV − RV) / IV / IV ÷ RV | The difference grows with the vol level, so high-vol names dominate. A ratio is comparable across stocks. `premium_screen` uses (IV − RV) / IV. |
| Which IV | LSEG's ATM iv30 / IV at the strike we would actually sell | The wheel sells 0.15–0.35 delta puts, below ATM, where the skew makes IV higher. This is where idea 3 comes in. |

**Things to know:**

- **Noise floor.** With 21 daily returns the realized-vol estimate has a standard error of about
  σ/√(2·21) ≈ 15% of its value: ±4–5 vol points on a 30-vol stock. A 2-point gap on one stock on one
  day is within the noise. Averages over many months or many stocks are not.
- **Earnings.** IV climbs into an earnings date because it prices the jump, while trailing RV has not
  seen the jump yet. Dots with earnings inside the next 30 days will sit above the line for a known
  reason. Mark them or leave them out. `term_structure_screen.py` already pulls earnings dates
  (`TR.EventStartDate`, `EventType` `RES`).
- **What the chart is dominated by.** Mostly by each stock's vol level: utilities bottom left,
  biotech top right. The information is the **distance from the line**. A second view may be
  clearer: the gap on y against the stock's own IV percentile (where today's IV sits in its past
  year) on x.
- **Prior evidence in this repo:**
  - The volatility-premium screen (`premium_screen.py`) ranks stocks by their history of IV above
    the RV that followed.
  - Wheel choice 11b (sell a put only when its IV is above 63-day RV) passed its bar on 5 of 9 test
    windows, about what a coin flip gives.
  - The 21-day version did not pass training.
  - So the IV > RV filter has been a weak signal for single stocks so far. That matches the
    literature: the premium is large and reliable on indexes, and smaller and less reliable on
    single names (Carr & Wu 2009; Driessen, Maenhout & Vilkov 2009).

## 3. Parabola fit to the smile, every day

**Model.** For each asset, each day, fit

```
σ(x) = a + b·x + c·x²
```

on the out-of-the-money options of one expiry.

| Coefficient | Reads as | Typical equity sign |
|---|---|---|
| `a` | vol at x = 0, i.e. ATM level | + (should be close to iv30, a free sanity check) |
| `b` | slope at ATM, i.e. skew | − (downside puts richer than upside calls); more negative = more paid for crash protection |
| `c` | curvature, i.e. how much both wings are bid over ATM | +; rises before events with jump risk (earnings, FDA, deals) |
| `−b / 2c` | where the smile bottoms out | usually right of ATM for equities; unstable when `c` ≈ 0 |

**The x-axis choice decides whether the coefficients mean anything over time:**

- **Raw strike K:** no. The units change with the stock price, so `b` and `c` drift every time the
  stock moves, and they can't be compared across stocks.
- **log(K/F)**, with F the forward from put-call parity (`vol.implied_forward`), centred at ATM-forward
  so that `a` is the ATM vol and `a` and `b` don't trade off. **Start here.**
- **log(K/F) / √T:** also removes most of the tenor effect (below). Probably the right default.
- **Delta:** the market's own axis (it is how desks quote skew), but delta itself depends on vol, so
  the x-values move with `a`.

**Tenor is the biggest trap.**
- `wheel.smile()` takes the monthly expiry nearest 30 days out, so day to day the expiry runs from
  about 45 days down to 7, then jumps to the next one.
- Short-dated smiles are steeper and more curved, so raw `b` and `c` will saw-tooth at every roll for
  reasons that have nothing to do with the market.
- Options, in order of effort:
  1. store days-to-expiry with each fit and compare like with like;
  2. scale x by √T;
  3. build a constant 30-day smile by interpolating total variance (σ²T) between the two expiries
     that bracket 30 days.
- The wheel cache keeps each monthly for about 50 days before expiry, so two bracketing expiries are
  not always both present. Check before relying on option 3.

**Which points go in:**

- Out-of-the-money only (puts below the forward, calls above), two-sided quotes, mid > 0, solvable IV.
  `wheel.smile()` already does all of this, so the fit can be written on top of its output.
- **A fixed x-window, held the same every day** (e.g. |delta| from 0.10 to 0.50 on each side). A
  parabola is only a local approximation: real smiles flatten into straight lines in the wings. If
  the window changes, the coefficients change for that reason alone.
- **Weights.** Wing quotes are a few cents wide on options worth a few cents, so their IVs are mostly
  noise. Use weighted least squares, by vega or by 1/(spread expressed in vol).
- **Check what is cached.** The wheel pulls kept puts from 0.70× to 1.02× and calls from 0.97× to
  1.40× of the spot around the previous expiry, so the far wings are missing, and after a big move
  the band is off-centre.
- **Skip rule.** Skip the day if either side has fewer than N points: with no puts there is no skew.

**What to save per asset per day.** One row: `date, asset, expiry, dte, spot, F, a, b, c, n_points,
x_min, x_max, rmse (vol pts), r2`. That table is the "report the fit coefficients" deliverable. Plot
`a`, `b` and `c` over time with earnings dates marked, and `a` against iv30 and RV.

**Reading coefficient changes:**

- **Sticky strike vs sticky moneyness.** If the smile stays put in strike while the stock drops, the
  ATM point slides up the skew and `a` rises by about |b|·Δx with no change in sentiment. Part of the
  movement in `a` is the stock moving along its own skew.
- **Size of the day-to-day noise.** Measure it before reading changes: refit on resampled points, or
  look at how much `a`, `b` and `c` move on quiet days.
- **Cross-asset comparison.** Standardise each coefficient within its own asset (z-score against its
  own history).
- **Next model if the parabola isn't enough:** SVI (Gatheral 2004). It has five parameters and
  straight-line wings, and it is the standard next step. A quadratic in vol can also turn negative far
  out, which is another reason to keep the window narrow.

**Link back to idea 2:** the parabola gives the IV **at the strike we would sell**. Comparing that to
RV is closer to the actual trade than comparing ATM iv30 to RV.

## 4. How these could turn into a strategy

Sketches only. Each is a different bet.

| Bet | Signal | Trade | Comment |
|---|---|---|---|
| Vol level is rich | IV well above trailing RV, outside the noise band, no earnings in window | sell puts / strangles, or a delta-hedged short straddle | Delta-hedging makes it a pure vol bet. An unhedged short put is mostly a stock bet. |
| Skew is rich | `b` unusually negative for this asset (its own z-score) | sell the put, buy the call (risk reversal), or sell put spreads | A bet on the skew flattening. Needs both sides liquid. |
| Wings are rich | `c` high with no event to explain it | sell the wings (iron condor) | Tail risk lives here. |
| Cross-section | rank all dots by IV − RV gap | sell the richest, buy the cheapest | Hedges out part of the market move. |
| Better wheel entry | put-strike IV vs RV, or `b`, at entry | the existing wheel, with a filter | Easiest to test: the engine, data and protocol already exist. |

Why IV vs RV is the right quantity for a vol trade: a delta-hedged **long** option earns about
½·Γ·S²·(σ_realized² − σ_implied²)·dt over each short step. The seller earns the negative, so the
seller wins when realized comes in under implied. The gains are weighted by gamma, so it matters when
the moves happen and where the stock is relative to the strike at the time.

## 5. What we'd want to know before calling it a strategy

**The edge**
- [ ] What exactly is the bet, in one sentence?
- [ ] Why should it exist? Who is on the other side, and why do they keep paying? (For vol selling:
      hedgers buying insurance.)
- [ ] What result would show it isn't there? Write that down before looking at any result.

**The signal**
- [ ] Is it known at the time of the trade (no forward RV, no future earnings surprises)?
- [ ] Is it bigger than its own measurement noise (the ±4–5 vol points above)?
- [ ] Does it survive leaving out earnings windows?
- [ ] Does it persist? The mean-reversion screen's edge didn't carry into the run window, and the
      random-stock comparison found no detectable advantage from the screen (p = 0.22).

**The trade**
- [ ] Which instrument and tenor? Delta-hedged or not, and if hedged, how often?
- [ ] Entry, exit, roll and early-close rules, and what happens at assignment?
- [ ] Monthlies only (cheaper to pull), or weeklies too?

**Execution and costs**
- [ ] Mid or bid/ask? Keep reporting both, as every result so far has. Option spreads are wide
      relative to the edge being chased.
- [ ] How liquid are the strikes we'd trade? How many contracts can we trade before we move the price?
- [ ] Whole contracts and capital per contract: an expensive stock makes results lumpy.

**Risk**
- [ ] Short vol earns small amounts often and loses big rarely. Does the backtest window include a
      crash (Feb 2018, Mar 2020, 2022)? If not, it overstates.
- [ ] One dot per stock, but in a selloff every dot moves together: are 20 short puts really one big
      bet on the market?
- [ ] Position sizing: equal notional, equal vega, or scaled by vol? What maximum drawdown is
      acceptable?
- [ ] Event risk per name: earnings, FDA, M&A gaps.
- [ ] Margin: cash-secured, or Reg T?

**Testing**
- [ ] Register, then run: train/test split, grid and pick rule fixed before any result (same habit as
      every screen here).
- [ ] How many variants have been tried in total? With enough tries, one passes by luck.
- [ ] Benchmarks: buy and hold, T-bills, the plain wheel, the covered-call benchmark, and a random
      draw of stocks (as in `random_wheel_2026`).
- [ ] Universe fixed as of a date (survivorship).
- [ ] Does it work in both low- and high-vol regimes?

**Practical**
- [ ] What does the assignment actually ask for: deliverable, page, due date, team or solo?
- [ ] How many LSEG requests does the first version need, and on how many days of the cap?

## 6. Already in the repo (on `main`), to reuse rather than rewrite

| Need | Where |
|---|---|
| IV from a price, greeks, forward from put-call parity | `trading_app/trading_app/lib/vol.py`: `implied_vol`, `bs_greeks`, `implied_forward` |
| One day's OTM smile with IV and delta per point | `trading_app/trading_app/lib/wheel.py`: `smile()` |
| iv30 vs forward RV per month, RIC fallback | `trading_app/scripts/premium_screen.py`: `measure`, `history` |
| Loop over 503 tickers, resumable, `--plan` request count | `trading_app/scripts/vol_screen_2023.py` |
| iv30 − iv90 and earnings dates | `trading_app/scripts/term_structure_screen.py` |
| LSEG session that finds the proxy port | `trading_app/scripts/fetch_wheel_raw.py`: `open_session` |
| Smile on any date, surface over time, implied vs realized (on the page) | `docs/hw4.html`, Volatility section |
| IV > RV as a wheel filter, already tested (11b) | `config/wheel_choices.yaml` |

## Possible first steps (cheapest first)

1. **Parabola on cached chains.**
   - Write the fit on top of `wheel.smile()` and run it on stocks whose chains are already in
     `cache/`. No new LSEG requests.
   - Plot `a`, `b`, `c` over time for one stock and check: does `a` track iv30, and does the
     saw-tooth at each roll show up?
2. **Tracker on ATMIV.U.**
   - A wide universe for little cap.
   - Trailing 21-day RV vs iv30, one dot per stock, with earnings flagged and the noise band drawn.
3. **Decide which bet in section 4 to pursue**, then register it before any result is seen.
