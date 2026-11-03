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
The wheel was expected to improve the Sharpe ratio compared to holding Mastercard, because it is short volatility. Selling an option works like selling insurance: the seller is paid a premium up front for taking on the risk of a big move. Per lecture, the wheel collects theta and is short gamma and vega, so it does well when moves are small. Each option is sold at its implied volatility, but what it actually costs the seller depends on the volatility Mastercard realizes afterwards. Implied volatility is usually higher than realized volatility (the volatility risk premium), so the premium collected should be more than the options end up costing. Though the short call caps the upside, the return was expected to be close to the stock's with much less risk.

## verdict
// Under the table of claims.
Implied volatility was higher than realized in {{test_iv_over_rv}} of {{test_premium_cycles}} test cycles, {{test_iv}} against {{test_rv}} on average. The options the wheel actually sold show the same thing: {{sold_over}} of {{sold_n}} were sold above the volatility that followed, by {{sold_gap}} points on average (see Implied vs realized under Volatility).

Most of the profit came from selling volatility and from the cash earning T-bills: the premium was {{test_premium_share}} of it and the T-bill interest {{test_interest_share}}. The rest came from the one assignment, where the shares were bought at {{test_assigned_strike}} after a fall in September and called away at {{test_called_strike}} after a rebound in October, for {{test_shares_pnl}} plus {{test_div}} in dividends, or {{test_shares_div_share}} of the profit. The Benchmarks table has the return and risk against each benchmark.

The cost of selling volatility is a payoff of many small wins and a rare large loss. Ordinary moves tend to be overpriced, but extreme moves may be underpriced (Taleb's argument for buying cheap far out-of-the-money options as crash insurance), and those are exactly the moves the wheel is short. Mastercard's worst drop in 2026 was {{test_hold_dd}}, against {{test_wheel_dd}} for the wheel, so one stock over about ten months does not show how the wheel would handle a real crash.

## stock
// Why Mastercard. The Stock choice block.
Mastercard was chosen by a screen. The screen ranked the S&P 500 as it stood on December 29, 2023 ({{universe}} stocks, including ones that have since been removed, to avoid survivorship bias) by how strongly each stock's monthly moves reversed over 2014-2023. It used the three-month variance ratio and the monthly autocorrelation: a stock was only ranked if its autocorrelation was negative, meaning a down month tended to be followed by an up month, and the ranked stocks were then sorted by variance ratio, lowest first. Mastercard came first, with a variance ratio of {{vr_screen}} against a median of {{vr_median}}. However, coming first out of the {{eligible}} stocks with enough history to score partly reflects chance. In screens of {{eligible}} random walks ranked the same way, first place usually has a variance ratio between {{rw_first}}, and Mastercard's {{vr_screen}} is inside that range. The reversal the screen found also did not continue: Mastercard's variance ratio was {{vr_run}} in 2024-2026, about what a random walk gives ({{rw_vr}}). Hence, the screen is reported here only as how the stock was found. Selling volatility needs options liquid enough to sell near the mid, and implied volatility above realized. As it turned out, Mastercard had both after it was picked: its options had two-sided quotes from 0.15 to 0.35 delta in all {{ma_cycles}} monthly cycles (see Fill prices for the evidence on the mid), and the options the wheel sold were priced above the volatility that followed {{train_sold_over}} times out of {{train_sold_n}} in 2024-2025 and {{sold_over}} out of {{sold_n}} in 2026. Before the pick it did not stand out: over 2014-2023 its 30-day implied volatility averaged {{vs_ma_points}} points {{vs_ma_side}} realized, which ranked it {{vs_ma_rank}} of {{vs_eligible}} on a screen of implied minus realized volatility, run afterwards out of curiosity. That screen also shows why liquidity matters: its top three picks, Incyte, Jack Henry and Copart, had options too thin to backtest, with {{vs_short_lo}} to {{vs_short_hi}} of cycles lacking two-sided quotes against a limit of 10%.

The backtest only uses the years after that window, because testing a stock on the same years that picked it would be grading it on its own selection.

## setting
// How the put delta, call delta and floor were chosen.
The put delta, call delta and cost-basis floor were picked out of 18 settings by their Sharpe ratio in 2024-2025. The pick was committed ({{pick_commit}}) before 2026 was run, and 2026 was only run once. It ranked {{rank}} of 18 in 2026, with a Sharpe ratio of {{test_wheel_sharpe}}, close to the {{test_baseline_sharpe}} of the default setting, which was fixed before looking at any data.

Walk-forward testing was then used, so that each year is traded with a setting picked only from the years before it: picked on 2024 and tested on 2025, then picked on 2024-2025 and tested on 2026. The 2025 pick ranked {{wf1_rank}} of 18, with a Sharpe ratio of {{wf1_sharpe}} against {{wf1_baseline_sharpe}} for the default. Over 2025-2026 together, the walk-forward picks had a Sharpe ratio of {{wf_sharpe}}, against {{wf_baseline_sharpe}} for the default and {{wf_hold_sharpe}} for holding Mastercard.

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
The wheel's return comes from selling volatility, and the risk that comes with it is one stock falling hard while the wheel holds it: Fiserv's wheel returned {{p2_fisv_train}} in 2024-2025. Hence, the same wheel was run on all five stocks the screen in Part I picked, side by side, to see whether the volatility premium showed up across them and whether spreading the wheel across stocks reduces that risk.

## part2-setup
Each of the five wheels starts with $100,000 at its stock's committed setting and runs exactly as in Part I. Their daily account values are added together into one $500,000 portfolio. The benchmarks are the same five stocks bought and held, $100,000 each, and T-bills. The Cboe PutWrite index was added afterwards, as in Part I.

## part2-findings
- **Selling volatility:** In 2026, {{p2_test_vol_above}} of the {{p2_test_vol_sold}} options the five wheels sold were priced above the volatility that followed, with an average gap of {{p2_test_vol_gap}} points and each stock's average from {{p2_test_vol_range}}. In 2024-2025 it was {{p2_train_vol_above}} of {{p2_train_vol_sold}}, and each stock's median gap was positive ({{p2_train_vol_median_range}}), but the average gap was {{p2_train_vol_gap}} points, pulled down by a few large misses: the selloff of {{p2_train_shock_month}}, which hit all five stocks' options at once ({{p2_train_shock_range}} points), and Fiserv's collapse in {{p2_train_worst_option_month}} ({{p2_train_worst_option}} points on one option). This is the payoff the thesis describes: many small wins and a rare large loss.
- **Less risk for the same return:** The portfolio's return is just the average of the five ({{p2_train_return}} in 2024-2025). However, its volatility was {{p2_train_vol}} against {{p2_train_avg_vol}} for one wheel on average, and its maximum drawdown was {{p2_train_dd}} against {{p2_train_avg_dd}}.
- **Why:** The five wheels' monthly results barely moved together, with a median correlation of {{p2_corr_train}} in 2024-2025 and {{p2_corr_test}} in 2026. A wheel loses when its own stock falls through the strike, and most of the time that happened at different times for different stocks. The exception was the selloff of {{p2_train_dd_month}}, when {{p2_train_dd_losers}} of the five lost in the same month and the portfolio had its largest drawdown ({{p2_train_dd}}). In 2026 the Mastercard and Visa wheels moved almost together (correlation {{p2_corr_ma_v_test}}).
- **One collapse, absorbed:** Fiserv's wheel returned {{p2_fisv_train}} in 2024-2025, a loss of {{p2_fisv_train_loss}}. In the $500,000 portfolio that is {{p2_fisv_train_of_portfolio}}, and the other four wheels' gains more than covered it.
- **Sharpe ratio:** {{p2_train_sharpe}} in 2024-2025, against {{p2_train_together}} if the five had moved together ({{p2_test_sharpe}} against {{p2_test_together}} in 2026). The average of the five single-stock Sharpe ratios ({{p2_train_avg_sharpe}}) is higher, but averaging Sharpe ratios gives a calm wheel like Mastercard's the same weight as a volatile one, so it is misleading here.
- **Against holding the stocks:** In 2024-2025 the portfolio made {{p2_train_return}} against {{p2_train_hold_return}} for holding the five (Sharpe {{p2_train_sharpe}} against {{p2_train_hold_sharpe}}). In 2026 holding won on return, {{p2_test_hold_return}} against {{p2_test_return}}, while the Sharpe ratios were about the same ({{p2_test_hold_sharpe}} against {{p2_test_sharpe}}).
- **Why holding won in 2026:** The wheel only owns a stock after a put is assigned, and once it does, the covered call caps the gain. Hence, it misses most of a big rally. APA and ADI alone account for {{p2_test_apa_adi_gap}} of the {{p2_test_gap}} gap, more than the whole of it, which Fiserv partly offsets. Held, APA returned {{p2_test_apa_hold}} while its wheel made {{p2_test_apa_wheel}}, a {{p2_test_apa_gap}} difference on its own, and ADI returned {{p2_test_adi_hold}} against {{p2_test_adi_wheel}}. Fiserv went the other way: held, it returned {{p2_test_fisv_hold}}, while its wheel returned {{p2_test_fisv_wheel}}. This is the tradeoff of the strategy. It gives up big rallies in exchange for smaller losses, which is also most of why it beat holding in 2024-2025: almost all of the {{p2_train_gap_abs}} came from APA ({{p2_train_apa_gap_abs}}), where the wheel held cash while the stock fell. It did not soften Fiserv's collapse ({{p2_train_fisv_wheel}} against {{p2_train_fisv_hold}} held).

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
