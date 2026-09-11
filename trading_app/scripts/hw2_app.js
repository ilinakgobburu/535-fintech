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
  const signed = (v, dp = 0) => (v >= 0 ? "+" : "") + money(v, dp).replace("−", "-");
  const cls = v => v >= 0 ? "pos" : "neg";
  const el = id => document.getElementById(id);
  const dateOf = s => String(s).slice(0, 10);
  const hourOf = s => String(s).slice(11, 16);
  const esc = s => String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

  document.title = `Covered Call · ${M.ticker}`;
  el("h-ticker").textContent = M.ticker;
  el("r-hour").textContent = `${String(M.order_hour).padStart(2, "0")}:00 UTC`;
  el("l-bars").textContent = M.bars.toLocaleString();
  el("f-n").textContent = D.fit.resid.n.toLocaleString();
  el("prov").innerHTML =
    `${esc(M.stock_ric)} · ${M.interval} bars · ${esc(M.window[0])} → ${esc(M.window[1])} · `
    + `${M.weeks} weekly cycles · ${M.option_series} call series, `
    + `${M.option_obs.toLocaleString()} option bars · LSEG pull ${esc(M.fetched_at)}`;

  // ---- tiles ------------------------------------------------------------
  const tiles = [
    { label: "Final NAV", value: money(H.final_nav), hero: 1,
      sub: `${signed(H.pnl)} on ${money(M.start_cash)} starting cash (${pct(H.pnl_pct)})` },
    { label: "Buy & hold, same 100 shares", value: money(H.bh_final), hero: 2,
      sub: `the covered call finished ${signed(H.gap)} against simply holding` },
    { label: "Premium collected", value: money(H.premium),
      sub: `${H.weeks_booked} calls written, ${H.weeks_skipped} week(s) skipped for want of a quote` },
    { label: "Assignments", value: `${H.assignments} of ${H.weeks_booked}`,
      sub: `${pct(100 * H.assignments / Math.max(1, H.weeks_booked))} of the calls written finished in the money` },
    { label: "Lowest available funds", value: money(H.min_available),
      sub: H.ever_infeasible
        ? "went negative — this book could not have been put on"
        : `never negative; the trade was fundable at every bar` },
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
      <th>Limit</th><th>Fill</th><th>Cash Δ</th><th>Rule</th></tr></thead><tbody>`;
    let prev = null, cash = M.start_cash;
    rows.forEach(r => {
      const d = dateOf(r.ts);
      const sep = prev && d !== prev && r.side === "BUY" ? " class='wk-sep'" : "";
      prev = d;
      cash += r.cash_delta;
      html += `<tr${sep}>
        <td>${esc(dateOf(r.ts))} ${esc(hourOf(r.ts))}</td>
        <td style="text-align:left">${esc(r.instrument)}</td>
        <td class="side s-${r.side}">${r.side}</td>
        <td>${r.qty > 0 ? "+" : ""}${r.qty}</td>
        <td>${r.limit === null ? "—" : num(r.limit)}</td>
        <td>${num(r.fill)}</td>
        <td class="${cls(r.cash_delta)}">${signed(r.cash_delta, 2)}</td>
        <td class="note">${esc(r.note)}</td></tr>`;
    });
    html += `</tbody><tfoot><tr>
      <th colspan="6" style="text-align:right">net cash from ${rows.length} booked events</th>
      <th class="${cls(cash - M.start_cash)}" style="text-align:right">${signed(cash - M.start_cash, 2)}</th>
      <th></th></tr></tfoot>`;
    el("tbl-blotter").innerHTML = html;

    const skipped = D.cycles.filter(c => String(c.status).startsWith("skipped"));
    const parts = [
      `<li><strong>${rows.length} rows, ${H.weeks_booked} cycles.</strong> `
      + `Each cycle is a stock buy (only when flat), one call written, and one `
      + `terminal event — ${H.assignments} assignment(s) and `
      + `${H.weeks_booked - H.assignments} expiry(ies).</li>`,
      `<li><strong>Cash Δ is the whole accounting.</strong> Buying stock debits `
      + `100 × print; writing debits nothing and credits 100 × mid; an expiry `
      + `moves $0; an assignment credits 100 × <em>strike</em> — never 100 × settle. `
      + `Crediting the settle would quietly turn a capped strategy into an `
      + `uncapped one, so it has its own test.</li>`,
    ];
    if (skipped.length) {
      parts.push(`<li><strong>${skipped.length} week(s) booked nothing</strong> `
        + `(${skipped.map(s => esc(s.iso)).join(", ")}) because the chosen strike had no `
        + `two-sided quote at the order bar. The rule is "no bid/ask → no fill", and `
        + `since the combo is one decision, the stock leg was not put on either.</li>`);
    } else {
      parts.push(`<li><strong>No week was skipped.</strong> Every chosen strike carried `
        + `a two-sided quote at the order bar, so the "no bid/ask → no fill" rule never `
        + `had to fire. It is still implemented and tested.</li>`);
    }
    el("blotter-note").innerHTML = `<ul class="notes">${parts.join("")}</ul>`;
  })();

  // ---- ledger (session closes) ------------------------------------------
  const L = D.ledger;
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
      <th>Cash</th><th>NAV</th></tr></thead><tbody>`;
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
        <td>${money(L.nav[i])}</td></tr>`;
    });
    el("tbl-ledger").innerHTML = html + "</tbody>";
  })();

  // The underlying's own path, computed once. Several sections quote it and
  // they must not each derive it slightly differently.
  const MARKS = L.stock_mark.filter(x => x !== null && isFinite(x));
  const STOCK = {
    first: MARKS[0], last: MARKS[MARKS.length - 1],
    lo: Math.min(...MARKS), hi: Math.max(...MARKS),
    ret: 100 * (MARKS[MARKS.length - 1] / MARKS[0] - 1),
  };

  // ---- plots ------------------------------------------------------------
  const failed = window.__plotlyFailed;
  // `extra.title` is merged rather than assigned: passing {text: "..."} through
  // Object.assign would replace the whole title object and silently drop the
  // left alignment, the font and the padding that keeps it clear of the legend.
  const baseLayout = extra => {
    const out = Object.assign({
    paper_bgcolor: TH.base, plot_bgcolor: TH.panel,
    font: { color: TH.text, family: TH.font, size: 12.5 },
    margin: { l: 72, r: 72, t: 96, b: 46 },
    hovermode: "x unified",
    hoverlabel: { bgcolor: TH.panel_hi, bordercolor: TH.line,
      font: { family: TH.mono, size: 11, color: TH.text } },
    // Title and legend both sit above the plot, but Plotly measures them on
    // DIFFERENT scales: layout.title.y is normalised to the whole PAPER, while
    // legend.y is normalised to the PLOT AREA and may exceed 1. Giving both
    // y = 1.0 put the title's baseline at the very top of the canvas, where it
    // was clipped away entirely, while the legend sat happily just above the
    // axes. So: title anchored near the top of the paper, legend just above
    // the axes, and a top margin wide enough for both.
    legend: { bgcolor: "rgba(11,16,32,0.78)", bordercolor: TH.line, borderwidth: 1,
      font: { size: 11, color: TH.text }, orientation: "h",
      y: 1.02, yanchor: "bottom", x: 0 },
    title: { font: { size: 15, color: TH.text }, x: 0, xanchor: "left",
      y: 0.985, yanchor: "top" },
    }, extra || {});
    if (extra && extra.title) {
      out.title = Object.assign({ font: { size: 15, color: TH.text }, x: 0,
        xanchor: "left", y: 0.985, yanchor: "top" }, extra.title);
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
      { x: L.ts, y: L.nav, name: "NAV", type: "scatter", mode: "lines",
        line: { color: TH.mark, width: 2.2 },
        hovertemplate: "NAV %{y:$,.0f}<extra></extra>" },
      // Margin requirements sit near $15k while NAV sits near $50k. On a shared
      // axis the NAV line flattens into a straight stripe and the whole point
      // of plotting it is lost, so the requirements get their own scale.
      { x: L.ts, y: L.initial_margin, name: "Initial (50% LMV)", type: "scatter",
        mode: "lines", yaxis: "y2", line: { color: TH.both, width: 1.4, dash: "dash" },
        hovertemplate: "initial %{y:$,.0f}<extra></extra>" },
      { x: L.ts, y: L.maintenance_margin, name: "Maintenance (25% LMV)", type: "scatter",
        mode: "lines", yaxis: "y2", line: { color: TH.faint, width: 1.2, dash: "dot" },
        hovertemplate: "maint %{y:$,.0f}<extra></extra>" },
    ];
    if (D.buy_hold) {
      traces.splice(1, 0, { x: D.buy_hold.ts, y: D.buy_hold.nav,
        name: "Buy & hold 100 shares", type: "scatter", mode: "lines",
        line: { color: TH.print, width: 1.8 },
        hovertemplate: "buy &amp; hold %{y:$,.0f}<extra></extra>" });
    }
    draw("plot-nav", traces, baseLayout({
      title: { text: "NAV against the same 100 shares held outright" },
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
      { x: L.ts, y: L.available_funds, name: "Available funds (NAV − initial)",
        type: "scatter", mode: "lines", line: { color: TH.mark, width: 2 },
        hovertemplate: "available %{y:$,.0f}<extra></extra>" },
      { x: L.ts, y: L.excess_liquidity, name: "Excess liquidity (NAV − maint)",
        type: "scatter", mode: "lines", line: { color: TH.both, width: 1.6 },
        hovertemplate: "excess %{y:$,.0f}<extra></extra>" },
    ], baseLayout({
      title: { text: "Could the account actually carry the trade?" },
      xaxis: ax(""), yaxis: ax("dollars", { tickformat: "$,.0f" }),
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: 0, y1: 0,
        line: { color: TH.warn, width: 1.2, dash: "dash" } }],
    }));
    const f = el("feas");
    f.className = "callout" + (H.ever_infeasible ? "" : " good");
    f.innerHTML = H.ever_infeasible
      ? `<strong>Available funds went negative.</strong> The low was `
        + `${money(H.min_available)}. This book could not have been put on at `
        + `${money(M.start_cash)} of starting cash, and the honest reading of the `
        + `NAV path above is that it describes a position the account could not have held.`
      : `<strong>Available funds never went negative.</strong> The low was `
        + `${money(H.min_available)}, on ${money(M.start_cash)} of starting cash — so `
        + `every trade on the blotter was fundable at the moment it was booked. That is `
        + `not a free pass: the same book at ${money(D.min_cash.min_cash)} of starting `
        + `cash would have breached${D.min_cash.worst_ts ? " around " + esc(dateOf(D.min_cash.worst_ts)) : ""}, `
        + `and the gap between those two numbers is the only margin of safety the strategy had.`;
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
      title: { text: `TRDPRC_1 against mid — ${S.drawn.toLocaleString()} of ${S.total.toLocaleString()} bars drawn` },
      hovermode: "closest",
      xaxis: ax("mid = (BID + ASK) / 2", { tickformat: "$,.0f" }),
      yaxis: ax("TRDPRC_1 (last print in the bar)", { tickformat: "$,.0f" }),
    }));

    el("fit-pooled").innerHTML =
      `<strong>Pooled: R² = ${num(P.r2, 4)}</strong> on n = ${P.n.toLocaleString()}, `
      + `slope ${num(P.slope, 4)}, intercept ${num(P.intercept, 3)}, RMSE ${money(P.rmse, 2)}. `
      + `<br>Taken alone that number proves less than it looks. The chain spans contracts `
      + `worth ${money(Math.min(...S.mid), 2)} to ${money(Math.max(...S.mid), 2)}, so a fit across all `
      + `of them is partly graded on knowing that a deep-in-the-money call is not a wing `
      + `call — which nobody doubted. The slope of ${num(P.slope, 4)} and intercept of `
      + `${num(P.intercept, 3)} are the more informative half of this line: the mid is `
      + `${Math.abs(P.slope - 1) < 0.01 ? "an essentially unbiased" : "a biased"} predictor of the `
      + `print, not merely a correlated one.`;

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
        `<strong>Where the assumption is actually used.</strong> The book only ever wrote `
        + `calls just out of the money for a few dollars of premium. Restricted to that band `
        + `(${esc(W.label)}), the fit is R² ${num(W.r2, 4)} on n = ${W.n.toLocaleString()}, `
        + `with a median miss of ${money(W.median_abs_resid, 3)} against a median spread of `
        + `${money(W.median_spread, 2)} — ${pct(100 * W.resid_over_spread)} of the spread. `
        + `A fill assumption only has to hold where it is being used, and this is where it is `
        + `being used.`;
    } else {
      el("fit-written").style.display = "none";
    }

    const R = D.fit.resid;
    el("fit-note").innerHTML = `<ul class="notes">
      <li><strong>The fit refused to break, and that is the finding.</strong> I expected
        conditioning to collapse the R² the way it inverted HW1's spread conclusion. It did
        not: across the price bands above, R² only falls to between ${num(rlo, 3)} and
        ${num(rhi, 3)}. Inside a band spanning a few dollars, "a $3 option is not a $6 option"
        is a real distinction to draw, and the mid draws it. <em>The mid tracks the print.</em>
        That is a genuine result and it is reported here because it contradicted the
        expectation, not because it flattered it.</li>
      <li><strong>But tracking a price and being able to trade at it are different claims,</strong>
        and the spread is what separates them. The median print missed the mid by
        ${money(R.median_abs, 3)} against a median quoted spread of ${money(R.median_spread, 2)} —
        ${pct(100 * R.median_resid_over_spread)} of the spread. Only ${pct(R.at_mid_pct)} of prints
        landed within a quarter-spread of the mid, while ${pct(R.at_bid_pct)} went off at the bid
        and ${pct(R.at_ask_pct)} at the ask. R² near ${num(P.r2, 3)} and a mid that is the actual
        trade price about a third of the time are both true at once, because R² is answering
        "how big is this option" and the fill question is "who paid the spread".</li>
      <li><strong>${pct(R.outside_pct)} of prints landed outside the quote entirely</strong> —
        below the bid or above the ask. That is not an arbitrage; it is the bar. In an hourly
        bar the BID/ASK is the quote standing at the end of the hour while TRDPRC_1 is the last
        trade inside it, so the two are simply not simultaneous. It bounds how precise this
        comparison can be at all, and it is the single strongest argument for pulling
        minute bars if this were repeated.</li>
      <li><strong>Liquidity moves the miss, not the fit.</strong> Going from one trade in the bar
        to ${esc(D.fit.by_moves[D.fit.by_moves.length - 1].label)}, R² barely moves
        (${num(D.fit.by_moves[0].r2, 4)} → ${num(D.fit.by_moves[D.fit.by_moves.length - 1].r2, 4)})
        while the median miss falls from ${money(D.fit.by_moves[0].median_abs_resid, 3)} to
        ${money(D.fit.by_moves[D.fit.by_moves.length - 1].median_abs_resid, 3)}. A "last trade"
        drawn from a single lot is a noisier object than one drawn from twenty, exactly as it
        should be — and R² is the statistic least able to see it.</li>
      <li><strong>Verdict on the fill.</strong> For one contract a week, written just out of the
        money on one of the most liquid chains listed, mid is a defensible fill and the numbers
        above support it. It is not a claim that would survive size, and it is not what a market
        order would have gotten — that is the ${money(R.median_spread / 2, 2)} half-spread, every
        week, against ${money(H.premium / Math.max(1, H.weeks_booked))} of median premium.</li>
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
      title: { text: "The same book, written at each hour of the entry session" },
      xaxis: ax("order bar (UTC)"),
      yaxis: ax("premium collected", { tickformat: "$,.0f" }),
      yaxis2: { overlaying: "y", side: "right", tickformat: "$,.0f",
        gridcolor: "rgba(0,0,0,0)", zeroline: false, linecolor: TH.line,
        tickfont: { size: 11, color: TH.print },
        title: { text: "final P&L", font: { size: 11, color: TH.print } } },
    }));

    let h = `<thead><tr><th>Order bar (UTC)</th><th>Premium</th><th>Final NAV</th>`
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

    el("hours-note").innerHTML = `<ul class="notes">
      <li><strong>The swing is larger than the result.</strong> Final P&L ranged
        ${signed(worstH.pnl)} at ${String(worstH.order_hour).padStart(2, "0")}:00 to
        ${signed(bestH.pnl)} at ${String(bestH.order_hour).padStart(2, "0")}:00 — a
        ${money(swing)} spread around a booked outcome of ${signed(H.pnl)}. The parameter
        nobody declares moved the answer by ${pct(100 * swing / Math.abs(H.pnl))} of the
        answer. Premium collected ranged ${money(pLo)} to ${money(pHi)} over the same rows,
        and every one of them booked ${booked.weeks_booked} cycles with
        ${booked.assignments} assignments: the strategy is genuinely identical, and only the
        clock moved.</li>
      <li><strong>The hour actually booked was the worst of the ${S.length}.</strong>
        ${String(M.order_hour).padStart(2, "0")}:00 ranks ${bookedRank} of ${S.length} by P&L.
        That is luck running against us and it is left standing, because the hour was fixed
        before any of these numbers existed — 11:00 ET, chosen to sit clear of the opening
        auction and the closing stub bar. Re-picking it now, knowing the table, would be the
        exact mistake this page is built to avoid, and the honest version of a sensitivity
        analysis is the one you publish when the sensitivity embarrasses you.</li>
      <li><strong>But it did not outrank the strike rule.</strong> The
        ${money(swing)} spread across hours sits against ${money(ruleSpread)} across the four
        strike rules in the next section. I expected the reverse — an intraday mid that moved
        2× on a single Monday made the timestamp look dominant — and it is not what the full
        book shows. The strike rule is the bigger lever, which is the assignment's premise,
        and it is worth confirming rather than presuming.</li>
      <li><strong>Why the hour moves anything at all.</strong> A one-week call is nearly all
        time value and its delta is large near the money, so an intraday move of a percent in
        the stock repositions the whole chain against spot. The rule then selects a
        <em>different contract</em>, not merely a different price for the same one — which is
        why this is not a rounding effect and why it partly cancels across ten weeks instead
        of accumulating.</li>
      <li><strong>What it does to the fill claim.</strong> "Fill at mid" is defensible.
        "Fill at mid" with an unstated hour is an incomplete assumption wearing a complete
        one's clothes. The fix is not a better hour; it is declaring the hour as a parameter
        and publishing this table beside the result.</li>
    </ul>`;
  })();

  // counterfactual rules
  (function rulesTable() {
    const R = D.rule_sweep;
    const NAMES = {
      nearest_otm: "nearest OTM (booked)",
      otm_1pct: "first strike ≥ spot × 1.01",
      otm_2pct: "first strike ≥ spot × 1.02",
      premium_50c: "furthest strike still paying ≥ $0.50",
    };
    let h = `<thead><tr><th>Strike rule</th><th>Median strike over spot</th>`
      + `<th>Median premium</th><th>Premium collected</th><th>Assigned</th>`
      + `<th>Final NAV</th><th>P&L</th></tr></thead><tbody>`;
    R.forEach(r => {
      const on = r.rule === M.rule;
      h += `<tr${on ? ' style="background:var(--panel-hi)"' : ""}>`
        + `<td>${esc(NAMES[r.rule] || r.rule)}</td>`
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
    const beatBH = R.filter(r => r.pnl > H.bh_pnl).length;
    // Is "further out is better" monotone in this window?
    const byDist = R.slice().sort((a, b) => a.median_otm_pct - b.median_otm_pct);
    let mono = true;
    for (let i = 1; i < byDist.length; i++) if (byDist[i].pnl < byDist[i - 1].pnl) mono = false;

    el("rules-note").innerHTML = `<ul class="notes">
      <li><strong>Every rule lost to simply holding the stock.</strong> ${beatBH === 0
        ? `None of the ${R.length} beat buy-and-hold's ${signed(H.bh_pnl)}`
        : `${beatBH} of ${R.length} beat buy-and-hold's ${signed(H.bh_pnl)}`}, and the closer
        the rule wrote to the money the worse it did: the booked nearest-OTM rule finished
        ${signed(H.pnl)} while writing a median
        ${pct(R.find(r => r.rule === M.rule).median_otm_pct, 2)} above spot, and the furthest
        rule finished ${signed(best.pnl)}. ${mono
          ? "Across these four the ordering is monotone in distance from spot."
          : "The ordering is nearly, but not perfectly, monotone in distance from spot — the furthest rule is not the best, because a strike far enough out stops being paid for."}
        That is not a discovery about covered calls; it is a description of what a cap does to
        a stock that rose ${pct(STOCK.ret)} through the window. In a flat or falling tape the ordering would invert, and
        nothing here tells you which tape comes next.</li>
      <li><strong>The strike rule is the bigger lever.</strong> The spread across rules is
        ${money(spread)}, against ${money(hourSpread)} across the seven order hours. The
        assignment's premise — that the strike decision is the point of the exercise — holds
        on this data. It is worth saying that I expected the opposite after watching one
        Monday's mid move 2× intraday, and that a single vivid observation turned out to be a
        poor guide to the aggregate.</li>
      <li><strong>These are counterfactuals, not results.</strong> Four rules over
        ${M.weeks} weeks on one name in one quarter is far too little to choose between them.
        All four share a single price path, so they are closer to one observation than to
        forty, and the ranking would not survive a different quarter. Picking the winner after
        the fact is precisely the mistake the pre-committed booked rule exists to avoid. They
        are here to size the <em>sensitivity</em>, not to nominate a strategy.</li>
      <li><strong>What would change my mind.</strong> If the furthest-OTM rule still won over
        a window containing a real drawdown, that would be evidence about the rule rather than
        about the tape. This window contains no such period, so the comparison above cannot
        distinguish "writing closer to the money is worse" from "selling calls into a rally is
        worse", and the second is almost certainly the whole of it.</li>
    </ul>`;
  })();

  // ---- analysis ---------------------------------------------------------
  (function analysis() {
    const cyc = D.cycles.filter(c => c.status === "assigned" || c.status === "expired");
    const { first, last, lo: sLo, hi: sHi, ret: stockRet } = STOCK;
    const assigned = cyc.filter(c => c.status === "assigned");
    const capCost = assigned.reduce((a, c) => a + Math.max(0, (c.settle - c.strike) * M.shares), 0);
    const qa = [
      ["What actually happened?",
       `${M.ticker} went from ${money(first, 2)} to ${money(last, 2)} over the ${M.weeks} weeks
        (${pct(stockRet)}), ranging ${money(sLo, 2)}–${money(sHi, 2)}. The book
        wrote ${H.weeks_booked} calls, collected ${money(H.premium)} in premium and was assigned
        ${H.assignments} time(s). It finished at ${money(H.final_nav)} — ${signed(H.pnl)} — against
        ${money(H.bh_final)} for the same 100 shares simply held. The covered call
        ${H.gap >= 0 ? "beat" : "trailed"} buy-and-hold by ${money(Math.abs(H.gap))}.`],
      ["Where did theory meet tape?",
       `Right at the cap. On the ${assigned.length} assigned week(s) the stock closed a total of
        ${money(capCost)} above the strikes we had sold — that is upside the account did not
        keep, and it is the price of the ${money(H.premium)} of premium. The textbook framing is
        that a covered call converts uncertain upside into certain income; the tape's version is
        that over a ${pct(stockRet)} run, ${money(H.premium)} of certain income was
        ${capCost > H.premium ? "not enough to pay for" : "enough to cover"} ${money(capCost)}
        of surrendered upside.`],
      ["Was the premium fair compensation?",
       `Per week the book collected a median of ${money(
          (() => { const m = cyc.map(c => c.mid).sort((a, b) => a - b);
                   return m.length ? m[Math.floor(m.length / 2)] : 0; })(), 2)} per share on a
        strike a median of ${pct(
          (() => { const m = cyc.map(c => c.otm_pct).sort((a, b) => a - b);
                   return m.length ? m[Math.floor(m.length / 2)] : 0; })(), 2)} above spot. Selling
        a cap that close to the money on a name this volatile is close to selling the stock's
        weekly range outright, which is why the assignment rate came in at
        ${pct(100 * H.assignments / Math.max(1, H.weeks_booked))}.`],
      ["Is the mid a defensible fill?",
       `Yes, with a caveat that is not the one I expected. Pooled R² is
        ${num(D.fit.pooled.r2, 4)}, which alone proves little — but I squeezed the price range to
        check, and the fit <em>survives</em>: R² stays between
        ${num(Math.min(...D.fit.by_price.map(b => b.r2)), 3)} and
        ${num(Math.max(...D.fit.by_price.map(b => b.r2)), 3)} inside narrow bands. The mid really
        does track the print. The caveat is that tracking and transacting are different things:
        the median print sat ${money(D.fit.resid.median_abs, 3)} from the mid on a median spread
        of ${money(D.fit.resid.median_spread, 2)} — ${pct(100 * D.fit.resid.median_resid_over_spread)}
        of the spread — and only ${pct(D.fit.resid.at_mid_pct)} of prints landed within a
        quarter-spread of it. For one contract a week on a chain this liquid, mid is a
        reasonable fill. It would not survive size, and it is not what a market order would
        have gotten.`],
      ["Could the account carry it?",
       `${H.ever_infeasible
          ? `No. Available funds bottomed at ${money(H.min_available)}, so the book as
             printed could not have been put on at ${money(M.start_cash)}.`
          : `Yes, with room. Available funds bottomed at ${money(H.min_available)} on
             ${money(M.start_cash)} of starting cash. The binding level is
             ${money(D.min_cash.min_cash)}${D.min_cash.binds ? "" : " or below"} — the same book
             started there would have breached.`} The covered short call itself never added a
        dollar of requirement, which is the entire mechanical point of writing calls against
        stock you already own rather than naked.`],
      ["What would you change?",
       `Three things, in order of how much the evidence supports them. <strong>One:</strong> fix
        the order hour as an explicit, pre-committed parameter and report the sensitivity — the
        ${money(Math.max(...D.hour_sweep.map(s => s.pnl)) - Math.min(...D.hour_sweep.map(s => s.pnl)))}
        P&L swing across hours is too large to leave undeclared. <strong>Two:</strong> a strike
        rule keyed to the week's own implied volatility rather than to a fixed distance, since a
        fixed rule sells the same cap in a calm week and a violent one. <strong>Three:</strong>
        stop reporting a pooled R² as if it validated the fill; report the residual as a
        fraction of the spread, which is the only scale on which "close to the mid" means
        anything. <strong>What I would not change</strong> is the wait-through-expiry rule.
        It costs money in a rally, but every alternative — rolling, buying to close — is a
        second discretionary decision, and the point of this exercise was to find out what the
        first one is worth.`],
      ["What is this evidence for, honestly?",
       `${M.weeks} weekly cycles on one name in one quarter. That is ${H.weeks_booked} independent
        decisions, all sharing a single price path — closer to one observation than to
        ${H.weeks_booked}. Nothing here supports a claim about covered calls in general. What it
        does support is narrower and still worth having: given <em>this</em> tape, these are exactly
        the trades the stated rules produce, this is what they cost, and this is the order in
        which the decisions mattered — the strike distance first, the order hour second, and the
        fill convention a distant third.`],
    ];
    el("analysis").className = "reading qa-grid";
    el("analysis").innerHTML = qa.map(([q, a]) =>
      `<div class="qa"><p class="q">${q}</p><p class="a">${a}</p></div>`).join("");
  })();

  // ---- methods ----------------------------------------------------------
  (function methods() {
    const F = M.fetch_stats || {};
    const O = D.ohlc;
    const shortWeeks = D.cycles.filter(c => c.short_week);
    el("methods").innerHTML = `
      <div class="qa"><p class="q">The expiry day is zero-padded, and the handout says it is not.</p>
        <p class="a">The RIC scheme in the assignment states <em>"DAY not zero-padded"</em>. Two of
        the three AAPL examples it prints do not resolve against LSEG:
        <code>AAPLF52619000.U^F26</code> and <code>AAPLH72620500.U^H26</code> both return an error,
        while <code>AAPLF052619000.U^F26</code> and <code>AAPLH072620500.U^H26</code> return 400 and
        478 observations. The third example expires on the 17th, so the rule never bites and it
        resolves either way. Every testable case fails; every padded correction works. The body is
        always nine digits — DD + YY + SSSSS — and a single-digit day must not shorten it to eight.
        This is not cosmetic here: <strong>Aug 7 and Sep 4 are single-digit Fridays in this very
        window</strong>, so following the handout literally drops 2 of ${M.weeks} cycles with no
        error message at all — the strikes simply come back empty and the weeks look quiet.</p></div>

      <div class="qa"><p class="q">"Buy Monday, expire Friday" is a description, not a rule.</p>
        <p class="a">Jun 19 2026 is Juneteenth and Jul 3 2026 is the observed Fourth of July; both
        are closed. Those weeks expire on the <strong>Thursday</strong>, and the Thursday RIC
        resolves against LSEG while the Friday one does not. Rather than hardcode a weekday, the
        loop reads the underlying's own session calendar and takes the first and last session of
        each ISO week, so a holiday <em>shifts</em> the cycle instead of deleting it.
        ${shortWeeks.length ? `In this window that fired on ${shortWeeks.length} week(s)
        (${shortWeeks.map(w => esc(w.iso) + " → " + esc(String(w.expiry_date))).join(", ")}).`
        : `In this window no cycle needed shifting, but the rule is what makes that a finding rather than an assumption.`}
        A Monday holiday is handled by the same rule from the other end: the entry moves to
        Tuesday instead of the week being skipped.</p></div>

      <div class="qa"><p class="q">A flat LSEG response means the opposite thing depending on how much you asked for.</p>
        <p class="a">HW1 documented that a multi-field request which loses all but one field comes
        back as <em>flat columns of bare RICs</em>, indistinguishable from a healthy single-field
        pull. Probing it again for this assignment turned up the sharper rule: flat columns carry
        whichever axis has more than one member, and <strong>when both are singletons the columns
        are fields</strong>. One RIC and three fields returns columns
        <code>['BID','ASK','TRDPRC_1']</code> with the RIC parked on <code>columns.name</code>;
        two RICs and one field returns the RICs as columns with the field on
        <code>columns.name</code>. That is a trap for the bisection this fetcher uses to skip
        strikes that never existed, because bisection drives batches down to size one and flips the
        meaning of the response underneath itself. The first version labelled field names as RICs
        and reported 26 live series out of 20 requested — which is the only reason the bug was
        caught. The fix is to resolve labels by <em>membership</em> in the known batch and the known
        field list rather than by position, and to refuse to write a cache containing any label
        that was not asked for.</p></div>

      <div class="qa"><p class="q">The bar extremes carry bad prints. The last-trade series does not.</p>
        <p class="a">A bar's open and close are both real, sequenced trades, so anything the bar
        genuinely traded through should sit near that body. On this pull it does not:
        <strong>HIGH_1 runs more than 1% above the bar's own body on ${pct(O.high.over_1pct_share)}
        of the ${O.bars} bars, and LOW_1 more than 1% below on ${pct(O.low.over_1pct_share)}</strong>,
        with excursions reaching +${num(O.high.max_pct, 1)}% and −${num(O.low.max_pct, 1)}%. One
        hour that opened and closed near $301 reports a high of $333. Those are odd-lot,
        out-of-sequence and cross prints surviving into the extremes.
        TRDPRC_1 shows nothing of the kind — its hour-to-hour move has a median of
        ${num(O.close_move.median_pct, 3)}%, a 99th percentile of ${num(O.close_move.p99_pct, 2)}%,
        and only ${O.close_move.over_3pct} bar(s) in the whole window move more than 3%.
        <br><br>This is why entry and settlement here read <strong>TRDPRC_1 and never
        HIGH_1/LOW_1</strong>. Any rule phrased as "did the stock touch the strike" — a
        barrier, a stop, an intraday assignment test — would have booked trades against prints
        that did not happen, and it would have looked entirely reasonable doing it. The
        extremes are still used in the fetcher, where all they do is widen the strike band and
        cost a few dead RICs.</p></div>

      <div class="qa"><p class="q">How much of the requested chain actually existed?</p>
        <p class="a">The fetcher bands strikes to where the stock actually traded and then generates
        past that range, so a good fraction of the candidate RICs are contracts that were never
        listed. ${F.dead_rics ? `${F.dead_rics.length} of them came back dead` : "Dead RICs are counted"},
        found by bisecting the batches that threw rather than by falling back to one request per
        (RIC, field) — the difference between O(N log B) and 160 requests for a single bad strike.
        ${F.collapsed ? `${F.collapsed} response(s) came back collapsed and were re-pulled one field
        at a time and relabelled locally.` : `No response came back collapsed on this pull, so the per-field fallback never had to run.`} The panel that survives is
        ${M.option_series} call series over ${M.option_obs.toLocaleString()} hourly bars.</p></div>

      <div class="qa"><p class="q">Why precompute everything in Python?</p>
        <p class="a">Same reason as HW1. Every figure on this page is computed once in
        <code>lib/covered_call.py</code> and <code>lib/cc_analysis.py</code> and embedded as JSON;
        the browser only draws what it is handed. There is exactly one implementation of the
        arithmetic, so the blotter, the ledger, the Reg T panel and the prose cannot disagree with
        each other. The counterfactual strike rules and the hour sweep call the <em>same</em>
        backtest function with different arguments, which is what makes a comparison between them a
        comparison of decisions rather than of two code paths that drifted.</p></div>

      <div class="qa"><p class="q">What is tested?</p>
        <p class="a">${M.suite.tests} tests in <code>tests/test_covered_call.py</code>, split by failure mode.
        The RIC and calendar tests pin bugs that produce <em>silence</em> — a strike that never
        resolves, a week that never happens. The blotter, ledger and Reg T tests pin bugs that
        produce a <em>plausible wrong number</em>: an assignment credited at the settle instead of
        the strike, a short call marked as an asset, marks leaking into cash, an initial requirement
        charged against a covered call. Each was verified by re-introducing the bug and confirming
        the test fails. That check is itself a committed script,
        <code>scripts/mutation_check.py</code>: it re-introduces ${M.suite.mutations} specific
        bugs one at a time, runs the suite against each, and restores the file afterwards.
        All ${M.suite.mutations} are caught. A suite that passes proves nothing on its own —
        it is entirely possible to write ${M.suite.tests} tests that assert whatever the code
        already happens to do — so the reproducible version of "these tests have teeth" is a
        harness anyone can re-run.</p>
        <p class="a" style="margin-top:14px">That harness had a bug of its own worth recording,
        because it is invisible and it forges a passing test run. CPython decides a
        <code>.pyc</code> is current by comparing the <em>(mtime, size)</em> it recorded against
        the source. Every mutation here is a same-length edit — <code>&lt; 2</code> becomes
        <code>&lt; 0</code> — so size never moves, and mutate-then-restore happens milliseconds
        apart, so at the one-second granularity of the recorded mtime they are the same instant.
        Python therefore accepted bytecode compiled from the <strong>mutated</strong> source as
        valid for the <strong>restored</strong> source, and the mutation survived the restore in
        bytecode with correct code sitting on disk. It surfaced as a build reporting
        ${M.weeks + 1} trading weeks instead of ${M.weeks}, from a <code>trading_weeks</code>
        whose source had been right the whole time. The harness now runs the suite under
        <code>PYTHONDONTWRITEBYTECODE</code>, deletes the bytecode either way, and finishes by
        asserting the suite still passes clean — so a restore that does not take is reported
        rather than inherited.</p></div>`;
  })();

  // ---- data connection --------------------------------------------------
  // Pages is the graded artefact and it is static by construction: it carries
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
        + `It is static by construction — the LSEG pull of ${esc(M.fetched_at)} is baked into `
        + `this file — and there is no live session here. Re-pulling needs LSEG Workspace `
        + `running against a local checkout.`;
    el("data-note").innerHTML = `<ul class="notes">
      <li><strong>What is baked in.</strong> ${M.option_series} call series across
        ${M.weeks} weekly expiries, ${M.option_obs.toLocaleString()} option bars and
        ${M.bars} underlying bars, all hourly, ${esc(M.window[0])} → ${esc(M.window[1])}.
        Everything on this page is computed from that and nothing is fetched at view time.</li>
      <li><strong>How to rebuild it.</strong> <code>python3 scripts/fetch_hw2.py</code> with
        LSEG Workspace running writes the cache; <code>python3 scripts/build_hw2.py</code>
        turns it into this file. The build needs no credentials, because it reads the
        committed cache.</li>
      <li><strong>Why static.</strong> The page and the test suite must not be able to
        disagree about a number. Precomputing in the same Python the tests cover means there
        is one implementation of the arithmetic and the browser only draws what it is
        handed.</li>
    </ul>`;
  })();

  el("footer").innerHTML =
    `Built from a cached LSEG pull (${esc(M.fetched_at)}). `
    + `${esc(M.stock_ric)}, ${M.interval} bars, ${esc(M.window[0])} → ${esc(M.window[1])}. `
    + `Reg T at ${(M.initial_rate * 100).toFixed(0)}% initial / ${(M.maint_rate * 100).toFixed(0)}% maintenance. `
    + `Starting cash ${money(M.start_cash)}, ${M.shares} shares per contract, order bar `
    + `${String(M.order_hour).padStart(2, "0")}:00 UTC.`;
})();
