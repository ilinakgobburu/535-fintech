// The words on docs/hw3.html. Everything under a "## name" line is that
// section's text. Lines starting with // are notes and never reach the page.
// Blank line = new paragraph. A line starting with "- " = bullet. **bold** works.
// {{name}} is replaced by a computed figure; run
//     python3 scripts/build_hw3.py --facts
// for the full list with current values. Preview with
//     python3 scripts/build_hw3.py --draft --out /tmp/hw3_draft.html

## summary
// Two or three sentences, the first thing the boss reads: what a PMCC is, what
// was tested, what happened. Figures you may want:
//   {{pmcc_pnl}} on {{pmcc_open_cash}} to open ({{pmcc_return_open}})
//   covered call {{cc_pnl}}, buy and hold {{hold_pnl}}; stock {{stock_first}} -> {{stock_last}}
//   Reg T cash needed: PMCC {{pmcc_regt_cash}} vs covered call {{cc_regt_cash}}

## result
// What the table and chart show, and which row is the headline (the primary book
// is bid/ask fills, selling through earnings). Worth one sentence each:
//   - the two capital bases and why both are shown
//   - headline is mid fills; the bid/ask stress test: {{pmcc_cross_pnl}} against {{pmcc_pnl}}
//   - the first version of the rules, before revision: {{pmcc_v1_pnl}}
//   - skip-earnings comparison: {{pmcc_skip_pnl}}, wrote {{pmcc_skip_written}} of {{pmcc_skip_cycles}} cycles
//   - worst month {{pmcc_worst_month}} in {{pmcc_worst_month_name}}; worst fall {{pmcc_drawdown}}

## strategy
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

## capital
// The boss's question was "takes less capital". Facts:
//   to open: {{pmcc_open_cash}} vs {{cc_open_cash}}
//   to stay inside Reg T all five years: {{pmcc_regt_cash}} vs {{cc_regt_cash}}
//   worst available funds {{pmcc_worst_available}} on {{pmcc_worst_available_date}} (an assignment night)
//   cash below zero on {{pmcc_days_cash_negative}} sessions with only the opening cash
//   LEAP leg total {{pmcc_leap_pnl}}; see the LEAP trades table for the 2020 roll
// Explain the two causes: short-stock margin on assignment nights, and paying
// for a new LEAP after the old one collapsed.

## trades
// OPTIONAL. Shown above the table in the Cycles tab.
// How to read the cycle table. Facts:
//   {{pmcc_written}} calls written, {{pmcc_expired}} expired, {{pmcc_assigned}} assigned, {{pmcc_skipped}} skipped
//   premium {{pmcc_premium}} vs cost of covering assignments {{pmcc_cover_cost}}
//   short calls {{pmcc_short_pnl}}, stock bought and sold around assignments {{pmcc_stock_pnl}}

## earnings
// OPTIONAL. Shown above the table in the Earnings cycles tab.
// Short-call result by whether the cycle contained an earnings date:
//   with an earnings date: {{earn_cycles}} cycles, {{earn_assigned}} assigned, mean {{earn_mean_pnl}}, worst {{earn_worst_pnl}}
//   without:               {{no_earn_cycles}} cycles, {{no_earn_assigned}} assigned, mean {{no_earn_mean_pnl}}, worst {{no_earn_worst_pnl}}
// This split was not fixed in advance; say so if you quote it. The registered
// comparison is the skip-earnings book: {{pmcc_skip_pnl}} against {{pmcc_pnl}}.

## greeks
// OPTIONAL. Shown above the table or chart in its tab. Leave it empty
// and the tab shows the table or chart on its own.
// One short paragraph: net delta stays positive but well under 100 shares; the
// LEAP makes the book long vega while the short call is short it.

## limitations
// Candidates: one hand-picked stock, one five-year window; daily closing quotes,
// which are wide on Biogen, not executable prices; no commissions, interest on
// cash, or borrow fees; early exercise not modelled; the earnings split is
// exploratory.
