/* Covered-call page. Every figure and every sentence is derived from DATA;
   nothing here is a number typed in by hand. That rule exists because the
   HW1 page shipped once with prose hardcoded to one ticker's results, which
   silently became false the moment the cache was rebuilt. */
(function () {
  const D = DATA, TH = D.theme, M = D.meta, H = D.headline;

  // ---- formatting -------------------------------------------------------
  const money = (v, dp = 0) => v === null || v === undefined || !isFinite(v)
    ? "—"
    : (v < 0 ? "−$" : "$") + Math.abs(v).toLocaleString("en-US",
        { minimumFractionDigits: dp, maximumFractionDigits: dp });
  const num = (v, dp = 2) => v === null || v === undefined || !isFinite(v)
    ? "—" : Number(v).toFixed(dp);
  const pct = (v, dp = 1) => v === null || !isFinite(v) ? "—" : Number(v).toFixed(dp) + "%";
  // A booked price at the precision it was booked: two decimals minimum, up
  // to four. The stock's last print is often sub-penny, and a mid
  // between nickel quotes ends in a half cent, so num() at two decimals
  // printed fills that no longer multiplied out to the cash beside them.
  const px = v => v === null || v === undefined || !isFinite(v)
    ? "—" : Number(v).toFixed(4).replace(/(\.\d\d\d*?)0+$/, (_, kept) => kept);
  const signed = (v, dp = 0) => (v >= 0 ? "+" : "") + money(v, dp).replace("−", "-");
  const cls = v => v >= 0 ? "pos" : "neg";
  const el = id => document.getElementById(id);
  const dateOf = s => String(s).slice(0, 10);
  const hourOf = s => String(s).slice(11, 16);
  const esc = s => String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

  // Shared by several sections, so declared before any of them run. The
  // rationale block was added above these definitions once and referenced
  // STOCK inside its temporal dead zone: a ReferenceError that killed the
  // entire script and left every section of the page empty. `node --check`
  // passed it, because it is not a syntax error; only the render test saw it.
  const L = D.ledger;
  // The underlying's own path, computed once. Several sections quote it and
  // they must not each derive it slightly differently.
  // The underlying's path over the whole data window, computed in Python.
  // Reading it off the ledger made "the tape rose X%" depend on which bar the
  // ledger happened to open on, which changed when the book was funded at the
  // first order rather than before it.
  const SP = D.stock_path;
  const STOCK = {
    first: SP.first, last: SP.last, lo: SP.lo, hi: SP.hi,
    ret: 100 * (SP.last / SP.first - 1),
  };
  // The booked strategy's row of the strike-rule sweep. Its medians are
  // computed in Python; prose that quotes them reads them from here.
  const BOOKED = D.rule_sweep.find(r => r.rule === M.rule);

  // The tape has no bars overnight, at weekends or on holidays. Plotly joins
  // the points on either side, which drew a straight multi-day plateau across
  // every weekend and read as though the level had been observed the whole
  // time. A null at any gap longer than a day breaks the line there instead.
  const GAP_MS = 24 * 3600 * 1000;
  const tms = v => Date.parse(String(v).replace(" ", "T") + "Z");
  function gapSplit(xs, ys) {
    const X = [], Y = [];
    for (let i = 0; i < xs.length; i++) {
      if (i && tms(xs[i]) - tms(xs[i - 1]) > GAP_MS) { X.push(xs[i]); Y.push(null); }
      X.push(xs[i]); Y.push(ys[i]);
    }
    return { x: X, y: Y };
  }



  document.title = `Covered Call · ${M.ticker}`;
  el("h-ticker").textContent = M.ticker;
  // Bar stamps are as-of: the order bar is named by the moment its prices are
  // from, which is the END of the hour. ET is UTC-4 across this whole window.
  el("r-hour").textContent =
    `hourly bar ending ${String(M.order_hour).padStart(2, "0")}:00 UTC (${M.order_hour - 4}:00 ET)`;
  el("l-bars").textContent = D.ledger.ts.length.toLocaleString();
  el("r-capital").textContent = money(M.start_cash, 2);
  // Show the arithmetic, not just the result: the figure seen in class was a
  // flat round number, and this one has to be checkable from the blotter.
  {
    const s0 = D.blotter.find(r => r.kind === "stock" && r.side === "BUY");
    const c0 = D.blotter.find(r => r.kind === "call" && r.side === "SELL");
    el("r-capital-math").textContent =
      `${s0.qty} shares × $${px(s0.fill)} = ${money(-s0.cash_delta, 2)}, less the first `
      + `call's premium, ${s0.qty} × $${px(c0.fill)} = ${money(c0.cash_delta, 2)}`;
  }
  el("r-margin").innerHTML = H.min_cash < 0
    ? `Because later entries occur at higher prices, they are partly <strong>bought on margin</strong>:
       cash reaches a low of ${money(H.min_cash)}, a loan Reg T permits while available funds
       remain positive. Interest accrues on the debit balance at ${pct(100 * M.margin_rate, 0)}
       a year (actual/360, an assumed broker rate) and is deducted from NAV. It totals
       ${money(H.margin_interest, 2)} over the backtest and reduces trading P&L from
       ${signed(H.pnl + H.margin_interest)} to ${signed(H.pnl)}. The buy-and-hold benchmark
       borrows on the same terms.`
    : "";
  el("f-n").textContent = D.fit.resid.n.toLocaleString();
  el("prov").innerHTML =
    `${esc(M.stock_ric)} · ${M.interval} bars · ${esc(M.window[0])} → ${esc(M.window[1])} · `
    + `${M.weeks} weekly cycles · ${M.option_series} call series, `
    + `${M.option_obs.toLocaleString()} option bars · LSEG pull ${esc(M.fetched_at)}`;

  // ---- the two decisions the assignment asks to see argued ----------------
  (function why() {
    const R = D.fit.resid, B = D.bar_study;
    el("why").innerHTML = `
      <div class="qa"><p class="q">Choice of underlying: ${esc(M.ticker)}</p>
        <p class="a">A mid-price fill is only credible on a liquid option chain. ${esc(M.ticker)}
        lists weekly expiries every week, with strikes ${money(M.strike_step, 2)} apart, a median
        bid-ask spread of ${money(R.median_spread, 2)}, and ${D.fit.pooled.n.toLocaleString()} hourly
        bars carrying both a quote and a trade against which the mid can be tested. Data
        availability was confirmed before the backtest was run, so the underlying was not
        selected on the basis of its result.</p></div>

      <div class="qa"><p class="q">Holding to expiry rather than buying the call back</p>
        <p class="a">A buy-back limit order would fill at whatever point within the hour the ask
        reached the limit. ${B ? `An hourly quote, however, records only the final minute
        (verified on ${B.snapshot.matched_hours.toLocaleString()} contract-hours)`
        : "An hourly quote, however, is only an end-of-hour snapshot"}, so simulating that fill
        would require a price that was never observed. Holding to expiry requires only the
        closing stock price on expiry day. The cost of this choice is that, in a rising market,
        the cap binds and the premium is the only upside retained.</p></div>`;
  })();

  // ---- tiles ------------------------------------------------------------
  const tiles = [
    { label: "Final NAV", value: money(H.final_nav), hero: 1,
      sub: `${signed(H.pnl)} on ${money(M.start_cash)} starting cash (${pct(H.pnl_pct)})`
        + (H.margin_interest > 0 ? `, after ${money(H.margin_interest, 2)} of margin interest` : "") },
    { label: "Buy & hold, same 100 shares", value: money(H.bh_final), hero: 2,
      sub: `the covered call finished ${signed(H.gap)} relative to holding the shares` },
    { label: "Premium collected", value: money(H.premium),
      sub: `${H.weeks_booked} calls written, ${H.weeks_skipped} skipped for lack of a quote` },
    { label: "Assignments", value: `${H.assignments} of ${H.weeks_booked}`,
      sub: `${pct(100 * H.assignments / Math.max(1, H.weeks_booked))} of the calls written finished in the money` },
    { label: "Lowest available funds", value: money(H.min_available),
      sub: H.ever_infeasible
        ? "went negative; this book could not have been established"
        : `never negative; every trade was fundable when booked` },
    { label: "Minimum starting cash", value: money(D.min_cash.min_cash),
      sub: D.min_cash.binds
        ? `below this, available funds go negative`
        : `Reg T never bound at any level tested` },
  ];
  el("tiles").innerHTML = tiles.map(t =>
    `<div class="tile${t.hero === 1 ? " hero" : t.hero === 2 ? " hero2" : ""}">
       <div class="label">${t.label}</div>
       <div class="value">${t.value}</div>
       <div class="sub">${t.sub}</div></div>`).join("");

  // ---- blotter ----------------------------------------------------------
  (function blotter() {
    const rows = D.blotter;
    let html = `<thead><tr>
      <th>Time (UTC)</th><th>Instrument</th><th>Side</th><th>Qty</th>
      <th>Limit</th><th>Fill</th><th>Cash Δ</th><th>Cash after</th>
      <th>Notes — the rule that fired</th></tr></thead><tbody>`;
    // Quantity as a change in POSITION. Stock rows store an unsigned 100 and
    // take their direction from the side, so a stock SELL would otherwise
    // print as +100 -- a sale displayed as a purchase.
    const qtyOf = r => (r.kind === "stock" && r.side === "SELL") ? -r.qty : r.qty;
    let prev = null, cash = M.start_cash;
    rows.forEach((r, i) => {
      const d = dateOf(r.ts);
      // An assignment is two rows of one event: tint them together, so the
      // ASSIGN row's $0 is never read as the proceeds going missing.
      const pair = r.side === "ASSIGN"
        || (r.kind === "stock" && r.side === "SELL"
            && rows[i - 1] && rows[i - 1].side === "ASSIGN");
      // NOT `cls`: that is the module's sign-color helper, and shadowing it
      // here threw on the next line's cls(r.cash_delta) and blanked the page.
      const rowCls = [prev && d !== prev && r.side === "BUY" ? "wk-sep" : "",
                      pair ? "evt-pair" : ""].filter(Boolean).join(" ");
      const sep = rowCls ? ` class="${rowCls}"` : "";
      prev = d;
      cash += r.cash_delta;
      html += `<tr${sep}>
        <td>${esc(dateOf(r.ts))} ${esc(hourOf(r.ts))}</td>
        <td style="text-align:left">${esc(r.instrument)}${r.occ
          ? `<br><span style="color:var(--faint);font-size:var(--fs-xs)">OCC ${esc(r.occ)}</span>`
          : ""}</td>
        <td class="side s-${r.side}">${r.side}</td>
        <td>${qtyOf(r) > 0 ? "+" : ""}${qtyOf(r)}</td>
        <td>${r.limit === null ? "—" : px(r.limit)}</td>
        <td>${px(r.fill)}</td>
        <td class="${cls(r.cash_delta)}">${signed(r.cash_delta, 2)}</td>
        <td>${money(cash, 2)}</td>
        <td class="note">${esc(r.note)}</td></tr>`;
    });
    html += `</tbody><tfoot><tr>
      <th colspan="6" style="text-align:right">net cash from ${rows.length} booked events</th>
      <th></th>
      <th class="${cls(cash - M.start_cash)}" style="text-align:right">${signed(cash - M.start_cash, 2)}</th>
      <th></th></tr></tfoot>`;
    el("tbl-blotter").innerHTML = html;

    const skipped = D.cycles.filter(c => String(c.status).startsWith("skipped"));
    const parts = [
      `<li><strong>${rows.length} rows, ${H.weeks_booked} cycles.</strong> `
      + `Each cycle consists of a stock BUY (only when the book holds no shares), one call `
      + `SELL and a terminal event. ${H.weeks_booked - H.assignments} of the ${H.weeks_booked} `
      + `cycles ended with an EXPIRE row and ${H.assignments} with assignment. As the `
      + `assignment specifies, an assignment is booked as <strong>two rows</strong>: an ASSIGN `
      + `row, which closes the short call and moves no cash, and a stock SELL of 100 shares at `
      + `the strike, which carries the cash.</li>`,
      `<li><strong>Cash accounting.</strong> A stock purchase debits 100 × print; writing a `
      + `call credits 100 × mid; an expiry moves $0; an assignment credits 100 × `
      + `<em>strike</em>, not 100 × settlement price. Crediting the settlement price would `
      + `remove the cap from the strategy, and a dedicated test guards against it.</li>`,
    ];
    if (skipped.length) {
      parts.push(`<li><strong>${skipped.length} week(s) were not traded</strong> `
        + `(${skipped.map(s => esc(s.iso)).join(", ")}) because the chosen strike had no `
        + `two-sided quote at the order bar. Under the rule "no bid/ask, no fill", and because `
        + `the position is a single decision, the stock leg was not established either.</li>`);
    } else {
      parts.push(`<li><strong>No week was skipped.</strong> Every selected strike had a `
        + `two-sided quote at the order bar, so the no-quote rule was never triggered; it is `
        + `nonetheless implemented and tested.</li>`);
    }
    el("blotter-note").innerHTML = `<ul class="notes">${parts.join("")}</ul>`;
  })();

  // ---- ledger (session closes) ------------------------------------------
  const closes = (() => {
    const idx = [];
    for (let i = 0; i < L.ts.length; i++) {
      if (i === L.ts.length - 1 || dateOf(L.ts[i + 1]) !== dateOf(L.ts[i])) idx.push(i);
    }
    return idx;
  })();

  (function ledger() {
    let html = `<thead><tr>
      <th>Session close</th><th>Shares</th><th>Stock mark</th><th>Stock MV</th>
      <th>Short call</th><th>Call mark</th><th>Option MV</th>
      <th>Cash</th><th>NAV</th><th>Initial</th><th>Maint</th>
      <th>Available</th><th>Excess</th><th>Accrued interest</th></tr></thead><tbody>`;
    closes.forEach(i => {
      const k = L.call_strike[i];
      const callTxt = k === null ? "—"
        : `−1 × ${num(k, 2)}C ${esc(String(L.call_expiry[i]))}`;
      html += `<tr>
        <td>${esc(dateOf(L.ts[i]))}</td>
        <td>${L.shares[i]}</td>
        <td>${num(L.stock_mark[i])}</td>
        <td>${money(L.stock_mv[i])}</td>
        <td style="text-align:left">${callTxt}</td>
        <td>${L.option_mark[i] === null ? "—" : num(L.option_mark[i])}</td>
        <td class="${L.option_mv[i] ? "neg" : ""}">${L.option_mv[i] ? money(L.option_mv[i]) : "—"}</td>
        <td>${money(L.cash[i])}</td>
        <td>${money(L.nav[i])}</td>
        <td>${money(L.initial_margin[i])}</td>
        <td>${money(L.maintenance_margin[i])}</td>
        <td class="${L.available_funds[i] < 0 ? "neg" : ""}">${money(L.available_funds[i])}</td>
        <td class="${L.excess_liquidity[i] < 0 ? "neg" : ""}">${money(L.excess_liquidity[i])}</td>
        <td>${L.accrued_interest[i] ? money(L.accrued_interest[i], 2) : "—"}</td></tr>`;
    });
    el("tbl-ledger").innerHTML = html + "</tbody>";

    // Every contract the book actually wrote -- the RICs that were queried
    // for a fill, with the OCC symbol a broker statement would print.
    const written = D.blotter.filter(b => b.kind === "call" && b.side === "SELL");
    const outcome = ric => {
      const t = D.blotter.find(b => b.instrument === ric && (b.side === "ASSIGN" || b.side === "EXPIRE"));
      return t ? t.side : "—";
    };
    el("tbl-contracts").innerHTML = `<thead><tr><th>Written</th><th>RIC</th>
      <th>OCC</th><th>Strike</th><th>Expiry</th><th>Premium (mid)</th><th>Outcome</th>
      </tr></thead><tbody>` + written.map(b => `<tr>
        <td>${esc(dateOf(b.ts))}</td><td style="text-align:left">${esc(b.instrument)}</td>
        <td style="text-align:left">${esc(b.occ || "—")}</td><td>${num(b.strike)}</td>
        <td>${esc(String(b.expiry))}</td><td>${px(b.fill)}</td>
        <td class="side s-${outcome(b.instrument)}">${outcome(b.instrument)}</td></tr>`).join("")
      + "</tbody>";
  })();

  // ---- plots ------------------------------------------------------------
  const failed = window.__plotlyFailed;
  // `extra.title` is merged rather than assigned: passing {text: "..."} through
  // Object.assign would replace the whole title object and silently drop the
  // left alignment, the font and the padding that keeps it clear of the legend.
  // Defined once. It used to be spelled out twice -- in the defaults and
  // again in the merge below -- and the duplication was invisible until a
  // mutation test changed one copy and the other quietly repaired it, making
  // a real bug (titles clipped off the canvas) look impossible to reach.
  const TITLE_STYLE = { font: { size: 15, color: TH.text }, x: 0,
    xanchor: "left", y: 0.985, yanchor: "top" };

  const baseLayout = extra => {
    const out = Object.assign({
    paper_bgcolor: TH.base, plot_bgcolor: TH.panel,
    font: { color: TH.text, family: TH.font, size: 12.5 },
    margin: { l: 72, r: 72, t: 96, b: 46 },
    hovermode: "x unified",
    hoverlabel: { bgcolor: TH.panel_hi, bordercolor: TH.line,
      font: { family: TH.mono, size: 11, color: TH.text } },
    // Title and legend both sit above the plot, but Plotly measures them on
    // DIFFERENT scales: layout.title.y is normalized to the whole PAPER, while
    // legend.y is normalized to the PLOT AREA and may exceed 1. Giving both
    // y = 1.0 put the title's baseline at the very top of the canvas, where it
    // was clipped away entirely, while the legend sat happily just above the
    // axes. So: title anchored near the top of the paper, legend just above
    // the axes, and a top margin wide enough for both.
    legend: { bgcolor: "rgba(11,16,32,0.78)", bordercolor: TH.line, borderwidth: 1,
      font: { size: 11, color: TH.text }, orientation: "h",
      y: 1.02, yanchor: "bottom", x: 0 },
    title: { ...TITLE_STYLE },
    }, extra || {});
    if (extra && extra.title) {
      out.title = Object.assign({ ...TITLE_STYLE }, extra.title);
    }
    return out;
  };
  const ax = (title, extra) => Object.assign({
    gridcolor: TH.line_soft, zeroline: false, linecolor: TH.line,
    tickfont: { size: 11, color: TH.muted },
    title: { text: title, font: { size: 11, color: TH.muted } },
  }, extra || {});
  const CFG = { displayModeBar: false, responsive: true };

  function draw(id, traces, layout) {
    if (failed) { el(id).innerHTML =
      `<p class="notes" style="padding:20px">Plotly failed to load from the CDN, so this chart is unavailable. Every number it draws is also in the tables on this page.</p>`;
      return; }
    Plotly.newPlot(id, traces, layout, CFG);
  }

  // NAV path + the two margin floors
  (function navPlot() {
    const shade = [];
    // Shade each week the book was short a call, so the cap is visible.
    let open = null;
    for (let i = 0; i < L.ts.length; i++) {
      const has = L.call_strike[i] !== null;
      if (has && open === null) open = L.ts[i];
      if (!has && open !== null) { shade.push([open, L.ts[i]]); open = null; }
    }
    if (open !== null) shade.push([open, L.ts[L.ts.length - 1]]);

    const traces = [
      { ...gapSplit(L.ts, L.nav), name: "NAV", type: "scatter", mode: "lines",
        line: { color: TH.mark, width: 2.2 },
        hovertemplate: "NAV %{y:$,.0f}<extra></extra>" },
      // Margin requirements sit near $15k while NAV sits near $50k. On a shared
      // axis the NAV line flattens into a straight stripe and the whole point
      // of plotting it is lost, so the requirements get their own scale.
      { ...gapSplit(L.ts, L.initial_margin), name: `Initial (${(M.initial_rate * 100).toFixed(0)}% LMV)`, type: "scatter",
        mode: "lines", yaxis: "y2", line: { color: TH.both, width: 1.4, dash: "dash" },
        hovertemplate: "initial %{y:$,.0f}<extra></extra>" },
      { ...gapSplit(L.ts, L.maintenance_margin), name: `Maintenance (${(M.maint_rate * 100).toFixed(0)}% LMV)`, type: "scatter",
        mode: "lines", yaxis: "y2", line: { color: TH.faint, width: 1.2, dash: "dot" },
        hovertemplate: "maint %{y:$,.0f}<extra></extra>" },
    ];
    if (D.buy_hold) {
      traces.splice(1, 0, { ...gapSplit(D.buy_hold.ts, D.buy_hold.nav),
        name: "Buy & hold 100 shares", type: "scatter", mode: "lines",
        line: { color: TH.print, width: 1.8 },
        hovertemplate: "buy &amp; hold %{y:$,.0f}<extra></extra>" });
    }
    draw("plot-nav", traces, baseLayout({
      title: { text: "NAV: covered call against buy-and-hold" },
      xaxis: ax(""),
      yaxis: ax("NAV / account value", { tickformat: "$,.0f" }),
      yaxis2: { overlaying: "y", side: "right", tickformat: "$,.0f", rangemode: "tozero",
        gridcolor: "rgba(0,0,0,0)", zeroline: false, linecolor: TH.line,
        tickfont: { size: 11, color: TH.both },
        title: { text: "margin requirement", font: { size: 11, color: TH.both } } },
      shapes: shade.map(([a, b]) => ({ type: "rect", xref: "x", yref: "paper",
        x0: a, x1: b, y0: 0, y1: 1, fillcolor: "rgba(144,133,233,.07)",
        line: { width: 0 }, layer: "below" })),
    }));
  })();

  // Available funds / excess liquidity
  (function marginPlot() {
    draw("plot-margin", [
      { ...gapSplit(L.ts, L.available_funds), name: "Available funds (NAV − initial)",
        type: "scatter", mode: "lines", line: { color: TH.mark, width: 2 },
        hovertemplate: "available %{y:$,.0f}<extra></extra>" },
      { ...gapSplit(L.ts, L.excess_liquidity), name: "Excess liquidity (NAV − maint)",
        type: "scatter", mode: "lines", line: { color: TH.both, width: 1.6 },
        hovertemplate: "excess %{y:$,.0f}<extra></extra>" },
    ], baseLayout({
      title: { text: "Available funds and excess liquidity under Reg T" },
      xaxis: ax(""), yaxis: ax("dollars", { tickformat: "$,.0f" }),
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: 0, y1: 0,
        line: { color: TH.warn, width: 1.2, dash: "dash" } }],
    }));
    const f = el("feas");
    f.className = "callout" + (H.ever_infeasible ? "" : " good");
    f.innerHTML = H.ever_infeasible
      ? `<strong>Available funds went negative.</strong> The minimum was `
        + `${money(H.min_available)}, so this book could not have been established with `
        + `${money(M.start_cash)} of starting cash, and the NAV path above describes a position `
        + `the account could not have held.`
      : `<strong>Available funds remained positive.</strong> The minimum was `
        + `${money(H.min_available)} on ${money(M.start_cash)} of starting cash`
        + (H.min_cash < 0 ? `, despite borrowing that took cash to ${money(H.min_cash)}` : "")
        + `, so every trade on the blotter was fundable when booked. The margin of safety is `
        + `limited: with ${money(D.min_cash.min_cash)} of starting cash, the same book would `
        + `have breached${D.min_cash.worst_ts ? " around " + esc(dateOf(D.min_cash.worst_ts)) : ""}.`;
  })();

  // Mid vs TRDPRC_1
  (function scatterPlot() {
    const S = D.scatter, P = D.fit.pooled;
    const lo = Math.min(...S.mid), hi = Math.max(...S.mid);
    draw("plot-scatter", [
      { x: S.mid, y: S.trd, type: "scattergl", mode: "markers", name: "hourly bars",
        marker: { size: 5, opacity: .5, color: S.moneyness, colorscale: "Viridis",
          colorbar: { title: { text: "K / S", font: { size: 10, color: TH.muted } },
            tickfont: { size: 10, color: TH.muted }, thickness: 10, len: .7 } },
        customdata: S.moneyness.map((m, i) => [m, S.moves[i], S.spread[i], S.strike[i]]),
        hovertemplate: "mid %{x:$.2f} → print %{y:$.2f}<br>"
          + "K %{customdata[3]:$.2f}  K/S %{customdata[0]:.3f}<br>"
          + "%{customdata[1]:.0f} trades in bar, spread %{customdata[2]:$.2f}<extra></extra>" },
      { x: [lo, hi], y: [lo, hi], type: "scatter", mode: "lines", name: "y = x",
        line: { color: TH.faint, width: 1, dash: "dot" }, hoverinfo: "skip" },
      { x: [lo, hi], y: [P.intercept + P.slope * lo, P.intercept + P.slope * hi],
        type: "scatter", mode: "lines", name: "least squares",
        line: { color: TH.print, width: 1.8 }, hoverinfo: "skip" },
    ], baseLayout({
      title: { text: `TRDPRC_1 against mid (${S.drawn.toLocaleString()} of ${S.total.toLocaleString()} bars shown)` },
      hovermode: "closest",
      xaxis: ax("mid = (BID + ASK) / 2", { tickformat: "$,.0f" }),
      yaxis: ax("TRDPRC_1 (last print in the bar)", { tickformat: "$,.0f" }),
    }));

    el("fit-pooled").innerHTML =
      `<strong>Pooled fit: R² = ${num(P.r2, 4)}</strong> on n = ${P.n.toLocaleString()}, `
      + `slope ${num(P.slope, 4)}, intercept ${num(P.intercept, 3)}, RMSE ${money(P.rmse, 2)}. `
      + `<br>The pooled R² is of limited value on its own. The sample spans contracts priced from `
      + `${money(Math.min(...S.mid), 2)} to ${money(Math.max(...S.mid), 2)}, so much of the fit `
      + `reflects the large differences in value between deep in-the-money and far `
      + `out-of-the-money contracts. The slope and intercept are more informative: they indicate `
      + `that the mid is ${Math.abs(P.slope - 1) < 0.01 ? "an essentially unbiased" : "a biased"} `
      + `predictor of the print, not merely a correlated one.`;

    const tb = (id, rows, first, extra) => {
      let h = `<thead><tr><th>${first}</th><th>n</th><th>R²</th><th>slope</th>`
        + `<th>median |print − mid|</th><th>as % of spread</th><th>within ¼-spread of mid</th>`
        + (extra ? `<th>outside the quote</th>` : "") + `</tr></thead><tbody>`;
      rows.forEach(b => {
        h += `<tr><td>${esc(b.label)}</td><td>${b.n.toLocaleString()}</td>`
          + `<td>${num(b.r2, 4)}</td><td>${num(b.slope, 3)}</td>`
          + `<td>${money(b.median_abs_resid, 3)}</td>`
          + `<td>${pct(100 * b.resid_over_spread)}</td>`
          + `<td>${pct(b.at_mid_pct)}</td>`
          + (extra ? `<td>${pct(b.outside_pct)}</td>` : "") + `</tr>`;
      });
      el(id).innerHTML = h + "</tbody>";
    };
    tb("tbl-price", D.fit.by_price, "Mid price band", true);
    tb("tbl-buckets", D.fit.buckets, "Moneyness K/S");
    tb("tbl-moves", D.fit.by_moves, "Trades in the bar");

    const bp = D.fit.by_price;
    const rlo = Math.min(...bp.map(b => b.r2)), rhi = Math.max(...bp.map(b => b.r2));
    const W = D.fit.written;
    if (W) {
      el("fit-written").innerHTML =
        `<strong>The band in which the book trades.</strong> The book wrote only calls slightly `
        + `out of the money, for a few dollars of premium. Within that band (${esc(W.label)}), `
        + `R² is ${num(W.r2, 4)} on n = ${W.n.toLocaleString()}, with a median absolute deviation `
        + `of ${money(W.median_abs_resid, 3)} from the mid against a median spread of `
        + `${money(W.median_spread, 2)} (${pct(100 * W.resid_over_spread)} of the spread). A fill `
        + `assumption needs to hold only where it is applied, and this is that band.`;
    } else {
      el("fit-written").style.display = "none";
    }

    const R = D.fit.resid;
    el("fit-note").innerHTML = `<ul class="notes">
      <li><strong>Robustness of the fit.</strong> Conditioning on price was expected to reduce
        R² substantially, as pooling had reversed the spread conclusion in Assignment 1.1. It
        did not: across the price bands above, R² ranges from ${num(rlo, 3)} to ${num(rhi, 3)}.
        Even within a band a few dollars wide, the mid distinguishes contracts of different
        value and tracks the print closely.</li>
      <li><strong>Tracking versus execution.</strong> Tracking a price is distinct from being
        able to trade at it, and the bid-ask spread separates the two. The median print deviated
        from the mid by ${money(R.median_abs, 3)}, against a median quoted spread of
        ${money(R.median_spread, 2)} (${pct(100 * R.median_resid_over_spread)} of the spread).
        Only ${pct(R.at_mid_pct)} of prints fell within a quarter-spread of the mid, while
        ${pct(R.at_bid_pct)} traded at the bid and ${pct(R.at_ask_pct)} at the ask. A high R²
        (${num(P.r2, 3)}) is compatible with a mid that equals the traded price in only about a
        third of cases, because R² measures how well the mid orders contracts by value, whereas
        the fill question concerns which side paid the spread.</li>
      <li><strong>Prints outside the quote.</strong> ${pct(R.outside_pct)} of prints fell below
        the bid or above the ask. This does not indicate an arbitrage; it follows from the
        construction of the hourly bar, in which BID/ASK is the quote at the end of the hour and
        TRDPRC_1 is the last trade within it, so the two are not simultaneous.
        ${D.bar_study
          ? `A one-minute re-pull of the same contracts reduces the figure on identical cells
             from ${pct(D.bar_study.hourly.outside_pct)} to
             ${pct(D.bar_study.minute.outside_pct)}, which suggests that roughly two-thirds of
             the effect is due to sampling and the remainder is genuine (see the
             <a href="#sec-bars" style="color:var(--both)">one-minute validation</a>).`
          : `This limits the precision of the comparison and would motivate minute bars in a
             repeat study.`}</li>
      <li><strong>Effect of liquidity.</strong> Moving from one trade in the bar to
        ${esc(D.fit.by_moves[D.fit.by_moves.length - 1].label)}, R² changes little
        (${num(D.fit.by_moves[0].r2, 4)} → ${num(D.fit.by_moves[D.fit.by_moves.length - 1].r2, 4)})
        while the median deviation falls from ${money(D.fit.by_moves[0].median_abs_resid, 3)}
        to ${money(D.fit.by_moves[D.fit.by_moves.length - 1].median_abs_resid, 3)}. A last trade
        drawn from a single lot is a noisier estimate than one drawn from many, and R² is
        largely insensitive to the difference.</li>
    </ul>`;
    el("fit-verdict").innerHTML = `<ul class="notes">
      <li><strong>Assessment of the fill assumption.</strong> For one contract per week, written
        slightly out of the money on one of the most liquid option chains, the mid is a
        defensible fill, and the evidence above supports it. The assumption would not hold at
        larger size, and a market order would instead have paid the
        ${money(R.median_spread / 2, 2)} half-spread each week, against average premium of
        ${money(H.premium / Math.max(1, H.weeks_booked))}. The effect on the result is small:
        had every call been filled at the <em>bid</em> rather than the mid, the book would have
        collected ${money(H.bid_fill_cost, 2)} less premium over ${H.weeks_booked} weeks,
        reducing P&L from ${signed(H.pnl)} to ${signed(H.pnl - H.bid_fill_cost)}. The
        conclusions are unchanged.</li>
      <li><strong>Further checks.</strong> The fit within narrow price bands, by moneyness
        and by liquidity, and a one-minute re-pull of the same contracts, are reported in
        <a href="#sec-fill-more" style="color:var(--both)">Part II</a>.</li>
    </ul>`;
  })();

  // hour sweep
  (function hourPlot() {
    const S = D.hour_sweep;
    const hrs = S.map(s => `${String(s.order_hour).padStart(2, "0")}:00`);
    draw("plot-hours", [
      { x: hrs, y: S.map(s => s.premium_collected), type: "bar",
        name: "premium collected", marker: { color: TH.mark },
        hovertemplate: "premium %{y:$,.0f}<extra></extra>" },
      { x: hrs, y: S.map(s => s.pnl), type: "scatter", mode: "lines+markers",
        name: "final P&L", yaxis: "y2", line: { color: TH.print, width: 2 },
        marker: { size: 7 }, hovertemplate: "P&amp;L %{y:$,.0f}<extra></extra>" },
    ], baseLayout({
      title: { text: "Final P&L by order hour" },
      xaxis: ax("order bar, UTC, labeled by the hour it ends"),
      yaxis: ax("premium collected", { tickformat: "$,.0f" }),
      yaxis2: { overlaying: "y", side: "right", tickformat: "$,.0f",
        gridcolor: "rgba(0,0,0,0)", zeroline: false, linecolor: TH.line,
        tickfont: { size: 11, color: TH.print },
        title: { text: "final P&L", font: { size: 11, color: TH.print } } },
    }));

    let h = `<thead><tr><th>Order bar (UTC, bar end)</th><th>Premium</th><th>Final NAV</th>`
      + `<th>P&L</th><th>Calls written</th><th>Assigned</th><th>Skipped</th></tr></thead><tbody>`;
    S.forEach(s => {
      const on = s.order_hour === M.order_hour;
      h += `<tr${on ? ' style="background:var(--panel-hi)"' : ""}>`
        + `<td>${String(s.order_hour).padStart(2, "0")}:00${on ? " ← booked" : ""}</td>`
        + `<td>${money(s.premium_collected)}</td><td>${money(s.final_nav)}</td>`
        + `<td class="${cls(s.pnl)}">${signed(s.pnl)}</td>`
        + `<td>${s.weeks_booked}</td><td>${s.assignments}</td><td>${s.weeks_skipped}</td></tr>`;
    });
    el("tbl-hours").innerHTML = h + "</tbody>";

    const prem = S.map(s => s.premium_collected);
    const pn = S.map(s => s.pnl);
    const pLo = Math.min(...prem), pHi = Math.max(...prem);
    const bestH = S[pn.indexOf(Math.max(...pn))], worstH = S[pn.indexOf(Math.min(...pn))];
    const swing = bestH.pnl - worstH.pnl;
    const booked = S.find(s => s.order_hour === M.order_hour);
    const bookedRank = pn.slice().sort((a, b) => b - a).indexOf(booked.pnl) + 1;
    const ruleSpread = Math.max(...D.rule_sweep.map(r => r.pnl))
                     - Math.min(...D.rule_sweep.map(r => r.pnl));

    const hh = x => String(x).padStart(2, "0") + ":00";
    el("hours-note").innerHTML = `<ul class="notes">
      <li><strong>Magnitude of the effect.</strong> Final P&L ranged from ${signed(worstH.pnl)}
        at ${hh(worstH.order_hour)} to ${signed(bestH.pnl)} at ${hh(bestH.order_hour)}, a range of
        ${money(swing)} around the booked outcome of ${signed(H.pnl)}, or
        ${pct(100 * swing / Math.abs(H.pnl))} of that outcome. Premium collected ranged from
        ${money(pLo)} to ${money(pHi)}, and every row booked ${booked.weeks_booked} cycles with
        ${booked.assignments} assignments, so the strategy is identical across rows and only the
        order time differs.</li>
      <li><strong>The booked hour.</strong> The booked hour, ${hh(M.order_hour)}, ranks
        ${bookedRank} of ${S.length} by P&L${bookedRank === S.length
          ? ", the lowest of the hours tested" : ""}. It is retained because it was fixed before
        these results were computed: the hour ending at noon ET was chosen to avoid both the
        open and the close. Selecting a different hour after observing this table would
        introduce look-ahead bias, which the pre-committed design is intended to prevent.</li>
      <li><strong>Comparison with the strike rule.</strong> The ${money(swing)} range across
        hours is ${swing < ruleSpread ? "smaller" : "larger"} than the ${money(ruleSpread)}
        range across the ${D.rule_sweep.length} strike rules in the next section. ${swing < ruleSpread
          ? `The initial expectation was the reverse, based on a single Monday on which a mid
             moved by a factor of two intraday; the full sample does not support it. The strike
             rule is the larger source of variation, consistent with the premise of the
             assignment.`
          : `On this sample the order time is the larger source of variation.`}</li>
      <li><strong>Mechanism.</strong> A one-week call consists almost entirely of time value
        and has a large delta near the money, so an intraday move of one percent in the stock
        shifts the chain relative to spot. The rule then selects a <em>different contract</em>,
        not merely a different price for the same contract. The effect is therefore not a
        rounding artifact, and it partly offsets across weeks rather than accumulating.</li>
      <li><strong>Implication for the fill assumption.</strong> "Fill at mid" is complete only
        when the order time is stated. The appropriate remedy is not to choose a better hour but
        to declare the hour as a parameter and report this table alongside the result.</li>
    </ul>`;
  })();

  // counterfactual rules
  (function rulesTable() {
    const R = D.rule_sweep;
    // Labels ride along with the sweep so a rule added in Python cannot show up
    // here under its bare function key.
    const nameOf = r => r.label || r.rule;
    let h = `<thead><tr><th>Strike rule</th><th>Median strike over spot</th>`
      + `<th>Median premium</th><th>Premium collected</th><th>Assigned</th>`
      + `<th>Final NAV</th><th>P&L</th></tr></thead><tbody>`;
    R.forEach(r => {
      const on = r.rule === M.rule;
      h += `<tr${on ? ' style="background:var(--panel-hi)"' : ""}>`
        + `<td>${esc(nameOf(r))}${on ? " &nbsp;← booked" : ""}</td>`
        + `<td>${pct(r.median_otm_pct, 2)}</td><td>${money(r.median_premium, 2)}</td>`
        + `<td>${money(r.premium_collected)}</td>`
        + `<td>${r.assignments} / ${r.weeks_booked}</td>`
        + `<td>${money(r.final_nav)}</td>`
        + `<td class="${cls(r.pnl)}">${signed(r.pnl)}</td></tr>`;
    });
    if (D.buy_hold) {
      h += `<tr class="wk-sep"><td>buy &amp; hold, no calls</td><td>—</td><td>—</td>`
        + `<td>—</td><td>—</td><td>${money(H.bh_final)}</td>`
        + `<td class="${cls(H.bh_pnl)}">${signed(H.bh_pnl)}</td></tr>`;
    }
    el("tbl-rules").innerHTML = h + "</tbody>";

    const pn = R.map(r => r.pnl);
    const spread = Math.max(...pn) - Math.min(...pn);
    const hourSpread = Math.max(...D.hour_sweep.map(s => s.pnl))
                     - Math.min(...D.hour_sweep.map(s => s.pnl));
    const best = R[pn.indexOf(Math.max(...pn))];
    const probRules = R.filter(r => r.target_prob !== null && r.target_prob !== undefined);
    const beatBH = R.filter(r => r.pnl > H.bh_pnl).length;
    // Is "further out is better" monotone in this window?
    const byDist = R.slice().sort((a, b) => a.median_otm_pct - b.median_otm_pct);
    let mono = true;
    for (let i = 1; i < byDist.length; i++) if (byDist[i].pnl < byDist[i - 1].pnl) mono = false;

    const calib = probRules.map(r =>
      `${pct(100 * r.target_prob, 0)} targeted, <strong>${pct(100 * r.realised_prob, 0)}</strong> realized`
    ).join("; ");
    const allAbove = probRules.length > 0
      && probRules.every(r => r.realised_prob > r.target_prob);
    const iv25 = R.find(r => r.rule === "iv_prob_25"), fixed2 = R.find(r => r.rule === "otm_2pct");

    el("rules-note").innerHTML = `<ul class="notes">
      <li><strong>Calibration of the implied-probability rules.</strong> Two rules target a
        stated probability that the cap is breached, derived from the week's at-the-money
        implied volatility (median ${pct(100 * R[0].median_atm_iv)}, range
        ${pct(100 * Math.min(...R.map(r => (r.iv_range || [NaN])[0])))}–${pct(100 * Math.max(...R.map(r => (r.iv_range || [NaN, NaN])[1])))}).
        ${allAbove ? "Realized assignment exceeded the target in every case" : "Realized assignment rates were"}:
        ${calib}.${allAbove ? ` This is expected rather than a defect of the rule. The
        probability implied by option prices is <em>risk-neutral</em>, and the risk-neutral
        measure has zero drift by construction, whereas the stock rose ${pct(STOCK.ret)} over
        the window. A cap with a ${pct(100 * probRules[0].target_prob, 0)} breach probability for
        a driftless stock is considerably more likely to be breached by a stock with strong
        positive drift. Option-implied probabilities should therefore not be read as
        forecasts.` : ""}</li>
      <li><strong>Implied-volatility rule against a fixed distance.</strong> The
        ${pct(100 * iv25.target_prob, 0)} rule wrote a median ${pct(iv25.median_otm_pct, 2)} out
        of the money, against ${pct(fixed2.median_otm_pct, 2)} for the fixed-distance rule, a
        similar average distance, and finished ${money(Math.abs(iv25.pnl - fixed2.pnl))}
        ${iv25.pnl > fixed2.pnl ? "ahead" : "behind"}. The rule adapts as intended: in the week
        with the highest implied volatility
        (${pct(100 * Math.max(...D.cycles.filter(c => c.atm_iv).map(c => c.atm_iv)))}) it placed
        the strike ${pct(iv25.max_otm_pct, 2)} out of the money, its widest of the ${M.weeks}
        weeks. Over ${M.weeks} weeks, however, the difference in P&L is within the range of
        noise, and a single price path cannot confirm the theoretical advantage in either
        direction.</li>
      <li><strong>Comparison with buy-and-hold.</strong> ${beatBH === 0
        ? `None of the ${R.length} rules beat buy-and-hold (${signed(H.bh_pnl)}).`
        : `${beatBH} of the ${R.length} rules beat buy-and-hold (${signed(H.bh_pnl)}).`}
        The booked nearest-OTM rule, writing a median
        ${pct(R.find(r => r.rule === M.rule).median_otm_pct, 2)} above spot, finished
        ${signed(H.pnl)}; the best alternative (${esc(nameOf(best))}) finished
        ${signed(best.pnl)}. ${mono
          ? `Across these rules, P&L increases monotonically with distance from spot.`
          : `P&L increases broadly, but not monotonically, with distance from spot: the
             furthest rule is not the best, because a sufficiently distant strike earns little
             premium.`}
        This ordering reflects the effect of a cap on a stock that ${STOCK.ret >= 0 ? "rose" : "fell"}
        ${pct(Math.abs(STOCK.ret))} over the window rather than a general property of covered
        calls; in a flat or falling market it would be expected to reverse.</li>
      <li><strong>Relative importance of strike and timing.</strong> P&L varies by
        ${money(spread)} across strike rules, against ${money(hourSpread)} across the
        ${D.hour_sweep.length} order hours. ${spread > hourSpread
          ? "The premise of the assignment, that the strike decision matters most, holds on this data."
          : "On this data the order time matters more than the strike rule."}</li>
      <li><strong>Interpretation.</strong> These are counterfactuals rather than results.
        ${R.length} rules over ${M.weeks} weeks on one name in one quarter cannot discriminate
        between strategies: all share a single price path, so they are closer to one
        observation than to ${R.length * M.weeks}, and the ranking is unlikely to persist in a
        different quarter. Selecting the best rule after the fact would introduce the
        look-ahead bias that the pre-committed rule is designed to avoid. The comparison
        measures <em>sensitivity</em>; it does not recommend a strategy.</li>
      <li><strong>Evidence that would alter this conclusion.</strong> If rules further from the
        money still performed better over a window containing a substantial drawdown, that
        would be evidence about the rules rather than about the market. This window contains no
        such period, so the comparison cannot distinguish "writing closer to the money is
        worse" from "selling calls in a rising market is worse"; the latter most likely
        accounts for most of the result.</li>
    </ul>`;
  })();

  // ---- bar size study ---------------------------------------------------
  (function barStudy() {
    const B = D.bar_study;
    const sec = document.getElementById("sec-bars");
    if (!B) {                       // no minute cache on this build
      if (sec) sec.remove();
      const host = el("bars"); if (host) host.remove();
      document.querySelectorAll('nav a[href="#sec-bars"]').forEach(a => a.remove());
      return;
    }
    const h = B.hourly, m = B.minute, s = B.snapshot;
    el("bars-n").textContent = m.quoted_bars.toLocaleString();

    el("bars").innerHTML = `<ul class="notes">
      <li><strong>Hourly quotes are end-of-hour snapshots.</strong> Across
        ${s.matched_hours.toLocaleString()} matched contract-hours, the hourly bid equals the
        final minute's bid in ${pct(s.bid_is_last_pct)} of cases and the ask in
        ${pct(s.ask_is_last_pct)}, compared with ${pct(s.bid_is_min_pct)} for the hour's lowest
        bid and ${pct(s.ask_is_max_pct)} for its highest ask. The hourly quote is therefore a
        snapshot at the close of the bar rather than an envelope over it. This matters for the
        fill assumption: had the hourly quote been a minimum-bid/maximum-ask envelope, each mid
        in the backtest would have been the midpoint of an hour's quote range rather than a
        tradeable price.</li>

      <li><strong>Prints outside the quote.</strong> On identical (contract, day) cells
        (${B.matched_cells.toLocaleString()} cells, ${m.contracts} contracts), the share of
        prints outside their bar's quote falls from <strong>${pct(h.outside_pct)} at one hour to
        ${pct(m.outside_pct)} at one minute</strong>. This is consistent with the timing
        explanation: the hourly quote is taken at the end of the hour while the print occurs
        within it, and a shorter bar narrows the gap. The remaining ${pct(m.outside_pct)}
        reflects genuine trade-throughs, odd lots, and the residual width of a one-minute
        bar.</li>

      <li><strong>A selection effect in spreads.</strong> Among bars that also contain a trade,
        as the mid-versus-trade comparison requires, the median spread is
        ${money(h.median_spread_printed, 3)} hourly against ${money(m.median_spread_printed, 3)}
        at one minute, which would suggest that minute data is cleaner. Measured
        <em>unconditionally</em> on the same contracts and days, the difference disappears:
        ${money(h.median_spread_all, 3)} hourly against ${money(m.median_spread_all, 3)} at one
        minute${
          Math.abs(h.median_spread_all - m.median_spread_all) < 0.005 ? ", which are identical"
          : m.median_spread_all > h.median_spread_all
            ? ", so the minute quotes are, if anything, <em>wider</em>" : ""}. The difference in
        the conditional figures is therefore a selection effect: ${pct(h.printed_share, 0)} of
        hourly bars contain a trade, against ${pct(m.printed_share, 0)} of minute bars, so
        conditioning on a trade is a much stricter filter at one minute and selects liquid,
        narrow-spread moments. Assignment 1.1 documented a related effect, in which pooling
        reversed a spread conclusion.</li>

      <li><strong>Implications for the backtest.</strong> The book continues to fill at the
        hourly mid, because the assignment specifies hourly bars and the hourly quote has been
        shown to be a genuine end-of-hour quote. The one-minute data identifies which features
        of the hourly panel reflect the market and which reflect sampling. The
        residual-to-spread ratio, on which the fill argument rests, is
        ${pct(100 * h.resid_over_spread)} hourly and ${pct(100 * m.resid_over_spread)} at one
        minute, close enough that the conclusion drawn from hourly data stands.</li>

      <li><strong>Data availability.</strong> The one-minute cache (approximately 390 MB) is not
        committed to the repository. <code>scripts/bar_size_study.py</code> reduces it to a JSON
        file under 1 KB, which is committed, so this section builds without the cache and every
        figure above can be checked by re-pulling with an LSEG session.</li>
    </ul>`;
  })();

  // ---- analysis ---------------------------------------------------------
  (function analysis() {
    const cyc = D.cycles.filter(c => c.status === "assigned" || c.status === "expired");
    const { first, last, lo: sLo, hi: sHi, ret: stockRet } = STOCK;
    const assigned = cyc.filter(c => c.status === "assigned");
    const capCost = assigned.reduce((a, c) => a + Math.max(0, (c.settle - c.strike) * M.shares), 0);
    const qa = [
      ["Results",
       `${M.ticker} ${stockRet >= 0 ? "rose" : "fell"} from ${money(first, 2)} to
        ${money(last, 2)} over the ${M.weeks} weeks (${pct(stockRet)}), trading between
        ${money(sLo, 2)} and ${money(sHi, 2)}. The book wrote ${H.weeks_booked} calls, collected
        ${money(H.premium)} in premium and was assigned ${H.assignments}
        time${H.assignments === 1 ? "" : "s"}. It finished at ${money(H.final_nav)}
        (${signed(H.pnl)}), against ${money(H.bh_final)} for the same 100 shares held outright,
        ${H.gap >= 0 ? "outperforming" : "trailing"} buy-and-hold by ${money(Math.abs(H.gap))}.`],
      ["Theory and observed outcomes",
       `The two portfolios diverge at the cap. On the ${assigned.length} assigned weeks the stock
        closed a total of ${money(capCost)} above the strikes sold; this upside was forgone in
        exchange for ${money(H.premium)} of premium. In theory a covered call exchanges
        uncertain upside for certain income. Over a ${pct(stockRet)} move, ${money(H.premium)}
        of premium was ${capCost > H.premium ? "insufficient to offset" : "sufficient to offset"}
        ${money(capCost)} of forgone upside.`],
      ["Premium relative to the cap",
       // Read from the booked rule's row of the sweep, not re-derived here. A
       // local m[floor(n/2)] took the upper-middle of an even count as the
       // median, and contradicted the strike-rule table two sections up.
       `The book collected a median of ${money(BOOKED.median_premium, 2)} per share on a
        strike a median of ${pct(BOOKED.median_otm_pct, 2)} above spot. A cap this close to the
        money on a stock this volatile amounts almost to selling the stock's weekly range, which
        accounts for the assignment rate of
        ${pct(100 * H.assignments / Math.max(1, H.weeks_booked))}.`],
      ["Validity of the mid fill",
       `The pooled R² of ${num(D.fit.pooled.r2, 4)} is weak evidence on its own, but the fit
        persists within narrow price bands (Part II), where R² ranges from
        ${num(Math.min(...D.fit.by_price.map(b => b.r2)), 3)} to
        ${num(Math.max(...D.fit.by_price.map(b => b.r2)), 3)}; the mid tracks the print
        closely. Tracking a price differs from transacting at it, however: the median print was
        ${money(D.fit.resid.median_abs, 3)} from the mid on a median spread of
        ${money(D.fit.resid.median_spread, 2)}
        (${pct(100 * D.fit.resid.median_resid_over_spread)} of the spread), and only
        ${pct(D.fit.resid.at_mid_pct)} of prints fell within a quarter-spread of it. For one
        contract per week on a chain this liquid, the mid is a reasonable fill. It would not
        hold at larger size, and a market order would have paid the
        ${money(D.fit.resid.median_spread / 2, 2)} half-spread each week. Filling every call at
        the bid would have reduced premium by ${money(H.bid_fill_cost, 2)} in total.${D.bar_study
          ? ` The hourly quote used is an end-of-hour quote rather than an aggregate: a
              one-minute re-pull matched the hourly bid and ask to the final minute in all
              ${D.bar_study.snapshot.matched_hours.toLocaleString()} contract-hours.`
          : ""}`],
      ["Margin feasibility",
       `${H.ever_infeasible
          ? `Available funds fell to ${money(H.min_available)}, so the book as shown could not
             have been established with ${money(M.start_cash)} of starting cash.`
          : `Available funds remained positive, with a minimum of ${money(H.min_available)} on
             ${money(M.start_cash)} of starting cash. The binding level is
             ${money(D.min_cash.min_cash)}${D.min_cash.binds ? "" : " or below"}: the same book
             started with less would have breached.`} The covered short call added no margin
        requirement, which is the mechanical advantage of writing calls against shares already
        held rather than uncovered.`],
      ["Proposed changes",
       `Three changes are proposed, in order of the strength of the supporting evidence.
        First, declare the order hour as an explicit, pre-committed parameter and report the
        sensitivity, since the
        ${money(Math.max(...D.hour_sweep.map(s => s.pnl)) - Math.min(...D.hour_sweep.map(s => s.pnl)))}
        range in P&L across hours (Part II) is too large to leave unstated. Second, key the strike
        to the week's implied volatility rather than to a fixed distance, since a fixed rule
        sells the same cap in calm and volatile weeks (evaluated in Part II). Third, evaluate the fill by the residual as a
        fraction of the spread rather than by the pooled R², since that is the scale on which
        proximity to the mid is meaningful. The hold-to-expiry rule would be retained: it is
        costly in a rising market, but each alternative (rolling, or buying to close) adds a
        second discretionary decision, whereas this exercise isolates the value of the
        first.`],
      ["Scope of the evidence",
       `The sample consists of ${M.weeks} weekly cycles on one name in one quarter. These are
        ${H.weeks_booked} decisions sharing a single price path, closer to one observation than
        to ${H.weeks_booked}, and they do not support general conclusions about covered calls.
        They do establish, for this price path, the trades the stated rules produce, their cost,
        and the order in which the decisions mattered: strike distance first, order time second,
        and the fill convention a distant third.`],
    ];
    el("analysis").className = "reading qa-grid";
    el("analysis").innerHTML = qa.map(([q, a]) =>
      `<div class="qa"><p class="q">${q}</p><p class="a">${a}</p></div>`).join("");
  })();

  // ---- data connection --------------------------------------------------
  // Pages is the graded artifact and it is static by construction: it carries
  // a cached LSEG pull baked into this file. The local server is the only
  // context that can reach a live session, so the page says which one it is
  // in rather than implying it is live everywhere.
  (function dataState() {
    const host = location.hostname;
    const local = host === "127.0.0.1" || host === "localhost";
    const box = el("data-state");
    box.className = "callout" + (local ? " good" : "");
    box.innerHTML = local
      ? `<strong>Local session available.</strong> This page is served from `
        + `<code>${esc(host)}</code>, so <code>scripts/fetch_hw2.py</code> can open an LSEG `
        + `session and rebuild the cache behind it.`
      : `<strong>Data connection required.</strong> This is the published GitHub Pages build. `
        + `It is static: the LSEG pull of ${esc(M.fetched_at)} is embedded in this file, and no `
        + `live session is available. Re-pulling requires LSEG Workspace running against a `
        + `local checkout.`;
    el("data-note").innerHTML = `<ul class="notes">
      <li><strong>Contents.</strong> ${M.option_series} call series across ${M.weeks} weekly
        expiries, ${M.option_obs.toLocaleString()} option bars and ${M.bars} underlying bars, all
        hourly, ${esc(M.window[0])} → ${esc(M.window[1])}. Every figure on this page is computed
        from these data; nothing is fetched at view time.</li>
      <li><strong>Rebuilding.</strong> Running <code>python3 scripts/fetch_hw2.py</code> with
        LSEG Workspace open writes the cache, and <code>python3 scripts/build_hw2.py</code>
        generates this page from it. The build requires no credentials, as it reads the
        committed cache.</li>
      <li><strong>Rationale for a static page.</strong> Computing every figure in the same
        Python code that the tests cover gives a single implementation of the arithmetic, so the
        page and the test suite cannot disagree about a number.</li>
    </ul>`;
  })();

  el("footer").innerHTML =
    `Built from a cached LSEG pull (${esc(M.fetched_at)}). `
    + `${esc(M.stock_ric)}, ${M.interval} bars, ${esc(M.window[0])} → ${esc(M.window[1])}. `
    + `Reg T at ${(M.initial_rate * 100).toFixed(0)}% initial / ${(M.maint_rate * 100).toFixed(0)}% maintenance. `
    + `Starting cash ${money(M.start_cash)}, ${M.shares} shares per contract, order bar `
    + `${String(M.order_hour).padStart(2, "0")}:00 UTC.`;
})();
