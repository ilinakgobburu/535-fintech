// The words on docs/hw4.html. Everything under a "## name" line is that
// section's text. Lines starting with // are notes and never reach the page.
// Blank line = new paragraph. A line starting with "- " = bullet. **bold** works.
// {{name}} is replaced by a computed figure; run
//     python3 scripts/build_hw4.py --facts
// for the full list with current values. These are first drafts: rewrite any
// of them, as long as you can explain every claim.

## thesis
// The assignment: "State your thesis for why this should make money/beat a
// benchmark/improve sharpe, then show whether it actually did."
The wheel was expected to **improve the Sharpe ratio** relative to holding Mastercard, for two reasons. First, the strategy is short options, and options are typically priced above the moves that follow, so the premium collected should exceed what the options cost on average. Second, the wheel buys after a fall (assignment) and sells after a rise (called away); hence, it is a long-only mean-reversion position and should do best on a stock whose monthly moves tend to reverse. Though the short call caps the upside, the return was expected to be close to the stock's with considerably less risk.

## verdict
// Under the table of claims. The facts: {{test_iv_over_rv}} of {{test_premium_cycles}} test cycles
// had implied above realised; variance ratio {{vr_run}} in 2024-2026 against {{vr_screen}} in
// 2014-2023, and {{rw_vr}} for a random walk.
**The claim held, though only one of its reasons did.** The options were priced above the moves that followed, but Mastercard's monthly moves stopped reversing (variance ratio **{{vr_screen}}** in 2014-2023, **{{vr_run}}** in 2024-2026, against **{{rw_vr}}** for a random walk). Hence, the higher Sharpe ratio came from the premium and the cash earning T-bills rather than from mean reversion. The return and risk against each benchmark are in the Benchmarks table.

## stock
// Why Mastercard. The Stock choice block.
Mastercard was chosen by a rule rather than by hand. The S&P 500 as it stood on 29 December 2023 ({{universe}} members, including those since removed, so the list is not today's survivors) was ranked by how strongly each stock's monthly moves reversed over 2014-2023, using the variance ratio over three months and the monthly autocorrelation, and a stock was ranked only if both indicated reversal. Mastercard ranked **first**, with a variance ratio of **{{vr_screen}}** against a median of {{vr_median}}. The backtest uses only the years after that window, because testing a stock on the years that selected it would grade it on its own selection.

## setting
// How the put delta, call delta and floor were chosen.
The put delta, call delta and cost-basis floor were chosen on 2024-2025 by training Sharpe ratio, out of 18 settings, and committed ({{pick_commit}}) before 2026 was run; 2026 was then run once. The pick ranked **{{rank}} of 18** in 2026, and its Sharpe ratio ({{test_wheel_sharpe}}) was close to that of the default setting fixed before any data was seen ({{test_baseline_sharpe}}). The choice was then walked forward: made again on 2024 alone and judged on 2025, so that each year is traded with a setting picked only from the years before it. That pick ranked {{wf1_rank}} of 18 in 2025 (Sharpe ratio {{wf1_sharpe}}, against {{wf1_baseline_sharpe}} for the default), and across 2025-2026 the walk-forward picks earned a Sharpe ratio of **{{wf_sharpe}}**, against {{wf_baseline_sharpe}} for the default setting and {{wf_hold_sharpe}} for holding Mastercard. Hence, choosing the setting from past Sharpe ratios did not beat the default, and the result depends far more on the stock and the period than on the setting.

## fills
// The assignment: "Any fill-price assumptions must be stated and justified with data."
Each option is sold at the **mid** of the closing bid and ask, and the same book is also run selling at the **bid** as a stress test. The mid is supported by LSEG's daily data: on the **{{band_n}}** days on which a contract like those the wheel sells traded (a monthly, out of the money, |delta| 0.10-0.40, 20-36 days to expiry), the last trade printed at a median of **{{band_location}}** of the way from bid to ask, where 0.5 is the mid. A regression of the print on the mid has a slope of {{band_slope}} and an R² of {{band_r2}}. Though the median is at the mid, individual prints vary: {{band_within}} fell within a quarter-spread of the mid, and {{band_at_bid}} printed at or below the bid. Hence, the result at the bid ({{test_bid_return}}) is shown beside the result at the mid ({{test_wheel_return}}) as a lower bound.

## fills-note
**Timing.** The last trade of the day can occur hours before the closing quote it is compared with, so part of the {{band_outside}} of prints outside the quote reflects the stock moving in between rather than trades outside the market. **Size.** The wheel sells one or two contracts at a time, against a median daily volume of {{band_volume}} contracts and open interest of {{band_oi}} in the band, so it is unlikely to move the price. **Assignment.** Shares are bought or delivered at the strike, by the terms of the contract, so assignment needs no fill assumption.

## portfolio
// Under the portfolio viewer.
Pick any session to see the account as it stood at that day's close. The open option is marked at that day's mid, or at its last mid when no two-sided quote exists.

## blotter
// Above the blotter table.
Every option is held to expiry by rule, so the blotter contains **no rolls and no early buy-backs**: each position closes by expiring, by assignment, or by being called away.

## accuracy
The code behind this page is checked by **{{tests}} tests**. The engine's tests run on a market small enough to verify by hand, including that no decision reads a later row than the one it trades on. The blotter and ledger are also audited independently: on {{audit_markets}} random markets, the cash and shares are rebuilt from the blotter and every fill, expiry, strike and dividend is checked; and for Mastercard, every trade, every session's cash and a random sample of sessions' account values are checked against the raw LSEG files. Before this page was built, the committed setting was re-run and reproduced all **{{reproduced}}** recorded figures exactly.

## honest
Each decision uses **only that day's closing quotes**: the strike is chosen from contracts quoted that day, and expiry settles on that day's close. The stock ({{screen_commit}}), the setting ({{pick_commit}}) and the test window were each committed before the 2026 test was run.

## limitations
Some limits to these findings include: one stock over a short test of {{test_cycles}} monthly cycles with a single assignment; daily closing quotes rather than executable prices; no commissions; early assignment is not modelled in the main run, though a re-run that models it (a put with no time value left, or a call before an ex-dividend date) gave exactly the same trades; and a final month (1-16 October 2026) that is partial but is charged a full month of T-bill yield in the Sharpe ratio, which lowers every book's Sharpe slightly.


// ---------------------------------------------------------------------------
// PART II: the same wheel on the five stocks the screen picked. Kept apart from
// Part I, which is what the assignment asks for.
// ---------------------------------------------------------------------------

## part2-motivation
The largest losses in this project came from a single stock falling while the wheel held it: Fiserv's wheel lost **{{p2_fisv_train}}** in 2024-2025. Hence, the same wheel was run on all five stocks the mean-reversion screen picked, side by side, to see whether spreading it across names reduces that risk. The setup was committed ({{p2_commit}}) before it was run, and nothing in Part I was chosen from its result.

## part2-setup
Five plain wheels, each with $100,000 at its stock's committed setting and run exactly as in Part I, with their daily account values added into one $500,000 portfolio. The benchmarks are the same five stocks bought and held, $100,000 each, and T-bills; the Cboe PutWrite index was added afterwards, as in Part I.

## part2-findings
- **Less risk for the same return:** The portfolio's return is the average of the five by construction ({{p2_train_return}} in 2024-2025), but its volatility was **{{p2_train_vol}}** against {{p2_train_avg_vol}} for a single wheel on average, and its maximum drawdown **{{p2_train_dd}}** against {{p2_train_avg_dd}}.
- **Why:** The five wheels' monthly results barely moved together, with a median correlation of {{p2_corr_train}} in 2024-2025 and {{p2_corr_test}} in 2026. Specifically, a wheel loses when its own stock falls through the strike, and that happened at different times for different stocks.
- **One collapse, absorbed:** Fiserv's wheel lost {{p2_fisv_train}} in 2024-2025, which cost the portfolio about a fifth as much.
- **Sharpe ratio:** {{p2_train_sharpe}} in 2024-2025, against {{p2_train_together}} had the five moved in lockstep ({{p2_test_sharpe}} against {{p2_test_together}} in 2026). Though the average of the five single-stock Sharpe ratios ({{p2_train_avg_sharpe}}) is higher, averaging Sharpe ratios gives a calm book such as Mastercard's the same weight as a volatile one, so it is not the right comparison.
- **Against holding the stocks:** In 2024-2025 the portfolio returned {{p2_train_return}} against {{p2_train_hold_return}} for holding the five (Sharpe {{p2_train_sharpe}} against {{p2_train_hold_sharpe}}). In 2026 it returned {{p2_test_return}} against {{p2_test_hold_return}} (Sharpe {{p2_test_sharpe}} against {{p2_test_hold_sharpe}}), because the covered calls capped a strong rally.

// ---------------------------------------------------------------------------
// HOVER NOTES. Text under "## tip: <label>" appears on the small (i) beside
// that label. Leave one empty and no (i) is shown.
// ---------------------------------------------------------------------------

## tip: Sharpe ratio
Mean monthly return above the T-bill, divided by its standard deviation, times √12. T-bills are the risk-free rate here, so their own Sharpe ratio is zero by construction and is not shown.

## tip: Max drawdown
The largest fall in account value from a previous high, on daily closes.

## tip: Premium collected
Every option sold, at the fill price, times the shares it covers.

## tip: T-bill interest
Cash in the account, including the collateral behind each put, earns the 3-month T-bill yield.

## tip: Portfolio on any date

## tip: Fill prices
The location of a print is (trade − bid) / (ask − bid): 0 is the bid, 0.5 the mid, 1 the ask.
