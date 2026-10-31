// The words on docs/hw4.html. Everything under a "## name" line is that
// section's text. Lines starting with // are notes and never reach the page.
// Blank line = new paragraph. A line starting with "- " = bullet. bold works.
// {{name}} is replaced by a computed figure; run
//     python3 scripts/build_hw4.py --facts
// for the full list with current values. Rewrite any of it, as long as you
// can explain every claim.

## thesis
// The assignment: "State your thesis for why this should make money/beat a
// benchmark/improve sharpe, then show whether it actually did."
I expected the wheel to improve the Sharpe ratio compared to holding Mastercard, for two reasons. First, the wheel is short volatility. Each option is sold at its implied volatility, but what it actually costs the seller depends on the volatility Mastercard realizes afterwards. Implied volatility is usually higher than realized volatility (the volatility risk premium), so the premium collected should be more than the options end up costing. Second, the wheel buys after a fall, when a put is assigned, and sells after a rise, when the shares are called away. Hence, it works like a long-only mean-reversion position and should do best on a stock whose monthly moves tend to reverse. Though the short call caps the upside, I expected a return close to the stock's with much less risk.

## verdict
// Under the table of claims.
Implied volatility was higher than realized in {{test_iv_over_rv}} of {{test_premium_cycles}} test cycles, {{test_iv}} against {{test_rv}} on average. The options the wheel actually sold show the same thing: {{sold_over}} of {{sold_n}} were sold above the volatility that followed, by {{sold_gap}} points on average (see Implied vs realized under Volatility).

Mean reversion did not hold up. Mastercard's variance ratio was {{vr_screen}} in 2014-2023 but {{vr_run}} in 2024-2026, which is about what a random walk gives ({{rw_vr}}). Hence, the higher Sharpe ratio came from selling volatility and from the cash earning T-bills, not from mean reversion. The Benchmarks table has the return and risk against each benchmark.

## stock
// Why Mastercard. The Stock choice block.
Mastercard was chosen by a screen, not by hand. The screen ranked the S&P 500 as it stood on December 29, 2023 ({{universe}} stocks, including ones that have since been removed, to avoid survivorship bias) by how strongly each stock's monthly moves reversed over 2014-2023. It used the three-month variance ratio and the monthly autocorrelation, and a stock was only ranked if both pointed to reversal. Mastercard came first, with a variance ratio of {{vr_screen}} against a median of {{vr_median}}.

The backtest only uses the years after that window, because testing a stock on the same years that picked it would be grading it on its own selection.

## setting
// How the put delta, call delta and floor were chosen.
The put delta, call delta and cost-basis floor were picked out of 18 settings by their Sharpe ratio in 2024-2025. The pick was committed ({{pick_commit}}) before 2026 was run, and 2026 was only run once. It ranked {{rank}} of 18 in 2026, with a Sharpe ratio of {{test_wheel_sharpe}}, close to the {{test_baseline_sharpe}} of the default setting, which was fixed before looking at any data.

I then walked the choice forward, so that each year is traded with a setting picked only from the years before it: picked on 2024 and tested on 2025, then picked on 2024-2025 and tested on 2026. The 2025 pick ranked {{wf1_rank}} of 18, with a Sharpe ratio of {{wf1_sharpe}} against {{wf1_baseline_sharpe}} for the default. Over 2025-2026 together, the walk-forward picks had a Sharpe ratio of {{wf_sharpe}}, against {{wf_baseline_sharpe}} for the default and {{wf_hold_sharpe}} for holding Mastercard.

Hence, picking the setting from past Sharpe ratios did not beat the default. The result depends much more on the stock and the period than on the setting.

## fills
// The assignment: "Any fill-price assumptions must be stated and justified with data."
Each option is sold at the mid of the closing bid and ask. As a stress test, the same wheel is also run selling at the bid.

LSEG's daily data supports the mid. On the {{band_n}} days when a contract like the ones the wheel sells actually traded (monthly, out of the money, delta between 0.10 and 0.40, 20-36 days to expiry), the median last trade printed {{band_location}} of the way from bid to ask, where 0.5 is the mid. A regression of the trade price on the mid has a slope of {{band_slope}} and an R² of {{band_r2}}. Though the median sits at the mid, single trades vary. Specifically, {{band_within}} were within a quarter of the spread from the mid, and {{band_at_bid}} printed at or below the bid. Hence, the result at the bid ({{test_bid_return}}) is shown next to the result at the mid ({{test_wheel_return}}) as a lower bound.

## fills-note
**Timing.** The last trade of the day can happen hours before the closing quote it is compared to. Some of the {{band_outside}} of trades outside the quote are the stock moving in between, not trades outside the market. **Size.** The wheel sells one or two contracts at a time, while the median contract in this band trades {{band_volume}} a day and has {{band_oi}} open, so it is unlikely to move the price. **Assignment.** Shares are bought or delivered at the strike, as the contract says, so assignment needs no fill assumption.

## portfolio
// Under the portfolio viewer.
Pick any day to see the account as it stood at that day's close. The open option is marked at that day's mid, or at its last mid if there was no two-sided quote.

## blotter
// Above the blotter table.
Every option is held to expiry by rule, so the blotter has no rolls and no early buy-backs. Each position ends by expiring, by assignment, or by being called away.

## accuracy
// Not on the page (panel removed).
The code behind this page is checked by {{tests}} tests. The engine's tests run on a market small enough to verify by hand, including that no decision reads a later row than the one it trades on. The blotter and ledger are also audited independently: on {{audit_markets}} random markets, the cash and shares are rebuilt from the blotter and every fill, expiry, strike and dividend is checked; and for Mastercard, every trade, every session's cash and a random sample of sessions' account values are checked against the raw LSEG files. Before this page was built, the committed setting was re-run and reproduced all {{reproduced}} recorded figures exactly.

## honest
// Not on the page (panel removed).
Each decision uses only that day's closing quotes: the strike is chosen from contracts quoted that day, and expiry settles on that day's close. The stock ({{screen_commit}}), the setting ({{pick_commit}}) and the test window were each committed before the 2026 test was run.

## limitations
// Not on the page (panel removed).
Some limits to these findings include: one stock over a short test of {{test_cycles}} monthly cycles with a single assignment; daily closing quotes rather than executable prices; no commissions; early assignment is not modeled in the main run, though a re-run that models it (a put with no time value left, or a call before an ex-dividend date) gave exactly the same trades; and a final month (October 1-16, 2026) that is partial but is charged a full month of T-bill yield in the Sharpe ratio, which lowers every book's Sharpe slightly.


// ---------------------------------------------------------------------------
// PART II: the same wheel on the five stocks the screen picked. Kept apart from
// Part I, which is what the assignment asks for.
// ---------------------------------------------------------------------------

## part2-motivation
The biggest losses in this project came from one stock falling while the wheel held it. Fiserv's wheel lost {{p2_fisv_train}} in 2024-2025. Hence, I ran the same wheel on all five stocks the mean-reversion screen picked, side by side, to see if spreading it across stocks reduces that risk. The setup was committed ({{p2_commit}}) before it was run, and nothing in Part I was chosen based on it.

## part2-setup
Each of the five wheels starts with $100,000 at its stock's committed setting and runs exactly as in Part I. Their daily account values are added together into one $500,000 portfolio. The benchmarks are the same five stocks bought and held, $100,000 each, and T-bills. The Cboe PutWrite index was added afterwards, as in Part I.

## part2-findings
- **Less risk for the same return:** The portfolio's return is just the average of the five ({{p2_train_return}} in 2024-2025). However, its volatility was {{p2_train_vol}} against {{p2_train_avg_vol}} for one wheel on average, and its maximum drawdown was {{p2_train_dd}} against {{p2_train_avg_dd}}.
- **Why:** The five wheels' monthly results barely moved together, with a median correlation of {{p2_corr_train}} in 2024-2025 and {{p2_corr_test}} in 2026. A wheel loses when its own stock falls through the strike, and that happened at different times for different stocks.
- **One collapse, absorbed:** Fiserv's wheel lost {{p2_fisv_train}} in 2024-2025, but it only cost the portfolio about a fifth of that.
- **Sharpe ratio:** {{p2_train_sharpe}} in 2024-2025, against {{p2_train_together}} if the five had moved together ({{p2_test_sharpe}} against {{p2_test_together}} in 2026). The average of the five single-stock Sharpe ratios ({{p2_train_avg_sharpe}}) is higher, but averaging Sharpe ratios gives a calm wheel like Mastercard's the same weight as a volatile one, so it is misleading here.
- **Against holding the stocks:** In 2024-2025 the portfolio made {{p2_train_return}} against {{p2_train_hold_return}} for holding the five (Sharpe {{p2_train_sharpe}} against {{p2_train_hold_sharpe}}). In 2026 holding won on return, {{p2_test_hold_return}} against {{p2_test_return}}, while the Sharpe ratios were about the same ({{p2_test_hold_sharpe}} against {{p2_test_sharpe}}).
- **Why holding won in 2026:** The wheel only owns a stock after a put is assigned, and once it does, the covered call caps the gain. Hence, it misses most of a big rally. The {{p2_test_gap}} gap comes almost entirely from APA and ADI. Held, APA returned {{p2_test_apa_hold}} while its wheel made {{p2_test_apa_wheel}}, a {{p2_test_apa_gap}} difference on its own, and ADI returned {{p2_test_adi_hold}} against {{p2_test_adi_wheel}}. Fiserv went the other way: held, it returned {{p2_test_fisv_hold}}, while its wheel returned {{p2_test_fisv_wheel}}. This is the tradeoff of the strategy. It gives up big rallies in exchange for smaller losses, which is also why it beat holding in 2024-2025.

// ---------------------------------------------------------------------------
// HOVER NOTES. Text under "## tip: <label>" appears on the small (i) beside
// that label. Leave one empty and no (i) is shown.
// ---------------------------------------------------------------------------

## tip: Sharpe ratio
The average monthly return above the T-bill, divided by its standard deviation, times √12. T-bills are the risk-free rate here, so their own Sharpe ratio is zero and is not shown.

## tip: Max drawdown
The largest fall in account value from a previous high, on daily closes.

## tip: Premium collected
Every option sold, at its fill price, times the shares it covers.

## tip: T-bill interest
Cash in the account, including the cash set aside for each put, earns the 3-month T-bill yield.

## tip: Portfolio on any date

## tip: Fill prices
Where a trade printed in the spread is (trade − bid) / (ask − bid): 0 is the bid, 0.5 the mid, 1 the ask.
