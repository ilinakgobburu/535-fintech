// The words on docs/hw3.html. Everything under a "## name" line is that
// section's text. Lines starting with // are notes and never reach the page.
// Blank line = new paragraph. A line starting with "- " = bullet. **bold** works.
// {{name}} is replaced by a computed figure; run
//     python3 scripts/build_hw3.py --facts
// for the full list with current values. Preview with
//     python3 scripts/build_hw3.py --draft --out /tmp/hw3_draft.html

## summary
// The Summary box beside the main chart. Bullets: start a line with "- ", and
// put the label in **bold** to get the purple starter.
- **Strategy:** A LEAP with a delta of approximately **0.80** is held, and a **one-month call** is sold against it each month.
- **Why monthly:** Though weekly options are typically used in an options backtest, a monthly short call was chosen because the LEAP lasts about a year, so the test required **several years of daily data**.
- **Context:** The covered call and buy-and-hold lost largely because Delta fell from **{{stock_first}}** to **{{stock_last}}** over the period.
- **Capital:** It was **cheaper to open** (**{{pmcc_open_cash}}** against **{{cc_open_cash}}**) but **not cheaper to run**: staying within margin rules took **{{pmcc_regt_cash}}**.

// ---------------------------------------------------------------------------
// The four things the assignment says the boss cares about, one card each, in
// the assignment's own words. These are first drafts from the facts; rewrite
// any of them.
// ---------------------------------------------------------------------------

## purpose
The objective of the strategy is to replicate a covered call with less capital. This approach is called a Poor Man's Covered Call, in which one long-dated call, a LEAP, replaces the 100 shares. The tradeoff is that it costs far less to open, but the LEAP loses time value every day and shares do not.

## performance
The strategy returned **{{pmcc_pnl}}** on **{{pmcc_open_cash}}** to open, versus **{{cc_pnl}}** for the covered call and **{{hold_pnl}}** for buy-and-hold. Its maximum drawdown was **{{pmcc_drawdown}}**, and at bid and ask prices it lost **{{pmcc_cross_pnl_abs}}**.

## accuracy
Wrote **{{tests}} tests** to check the engine, specifically for look-ahead bias, expiry, the buy-back after an assignment, option selection and margin, to verify its answers against ones worked out by hand and ensure it is accurate.

## honest
Each decision uses **only that day's closing quotes**, so there is **no look-ahead bias**. Specifically, the strike is chosen from contracts quoted that day, expiry settles on that day's close, and an assignment is bought back at the next morning's open.

## rules-short
// Your short wording for each rule line in the Strategy panel. Write the text
// after the colon. Leave one blank and the registered wording is shown.
Long leg: Buy the January call closest to a year out, at the strike with delta nearest 0.80. Replace it when under 90 days remain or its delta falls below 0.50.
Short leg: Each month, sell the next monthly call at the strike with delta nearest 0.25, never below the LEAP's break-even. If no strike qualifies, skip the month.
Earnings: Sell through earnings.
Fills: At the mid of the closing bid and ask. Stress test: sell at the bid, buy at the ask.
Assigned: The book is short 100 shares at the strike, buys them back at the next open plus $0.01, and keeps the LEAP.

## strategy-note
// A note at the bottom of the Strategy panel, under the rules and the payoff chart.
**Why 0.80 and 0.50:** A delta of 0.80 was chosen, per convention, because the LEAP then behaves similarly to about 80 shares while carrying little time value. Below 0.50, the LEAP is out of the money and no longer serves as a substitute for the shares; hence, it is replaced.

## strategy
// Why this stock. The Stock choice block, under the four cards.
// There is no separate method section any more. If you want the no-look-ahead
// point on the page, one sentence here is the place for it. The facts:
// No look-ahead: decisions and fills use the same day's closing quote; expiry
// uses that day's close; an assignment is covered at the next open plus $0.01;
// strikes are chosen only from contracts with a two-sided quote that day.
// Data: {{requests}} contracts requested, {{returned}} returned, {{never_listed}} never listed,
// {{empty}} empty, {{failed}} failed; {{quote_rows}} quote rows, {{crossed_rows}} crossed.
// Unadjusted prices; delta computed from mid with one formula for all years.
// {{tests}} tests; three books through one engine.
//
// The purpose of the strategy: what replaces the shares, what you give up, why
// someone would want it. Then why Biogen (your four reasons), and that the rules
// below were committed ({{rules_commit}}) before the backtest was run.
Delta Air Lines was chosen as the instrument because of the diagonal structure of the strategy. A long-dated call is held while a short-dated call is sold against it each month. The short call is priced on the next month alone, while the LEAP is priced on the whole year, so the strategy does best when a single month is priced at a premium relative to the year as a whole. Airline demand is seasonal, and I expected Delta's short-dated option prices to reflect that.

## trades
// OPTIONAL. Shown above the table in the Cycles tab.
// How to read the cycle table. Facts:
//   {{pmcc_written}} calls written, {{pmcc_expired}} expired, {{pmcc_assigned}} assigned, {{pmcc_skipped}} skipped
//   premium {{pmcc_premium}} vs cost of covering assignments {{pmcc_cover_cost}}
//   short calls {{pmcc_short_pnl}}, stock bought and sold around assignments {{pmcc_stock_pnl}}

## greeks
// One sentence shown under the delta and vega chart in the Exposure tab.
Net delta and vega are shown because they are the two exposures the position carries. Specifically, delta measures how closely it follows the stock, and vega shows that the LEAP leaves the position long volatility even though the short call is short it.

## limitations
// Candidates: one hand-picked stock, one five-year window; daily closing quotes,
// which are wide on Biogen, not executable prices; no commissions, interest on
// cash, or borrow fees; early exercise not modelled; the earnings split is
// exploratory.
Some limits to these findings include: one stock over one five-year period, daily closing quotes rather than guaranteed execution prices, and no commissions or interest on cash.


// ---------------------------------------------------------------------------
// HOVER NOTES. Text under a "## tip: <label>" heading appears when the reader
// hovers the small (i) beside that label. Leave one empty and no (i) is shown.
// Keep each to one or two sentences. {{placeholders}} work here too.
// ---------------------------------------------------------------------------

## tip: Cycles

## tip: Exposure

## tip: Rules

## tip: PMCC P&L

## tip: Covered call P&L

## tip: Cash to open

## tip: Reg T cash needed
The smallest starting cash that keeps the account inside Reg T margin rules on every day of the test. It is more than the cash to open because of assignment nights and the cost of replacing a LEAP.

## tip: Worst fall

## tip: Worst month

## tip: Calls assigned

## tip: Premium vs cover cost

## tip: Profit and loss

## tip: Summary

## tip: Strategy

## tip: Books

## tip: NAV and Reg T available funds
NAV is what the account is worth. Available funds is what Reg T lets it use. The sharp drops are the {{pmcc_assigned}} nights after an assignment, when the account was short 100 shares and had to post half their value. A LEAP cannot be borrowed against, so its value does not help.

## tip: LEAP value and the stock

## tip: Limitations

## tip: Purpose of the strategy

## tip: Performance

## tip: Accuracy

## tip: Reporting

## tip: Stock choice
