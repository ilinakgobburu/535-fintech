/*
 * The nightly NVDA marks, shown on the strategy pages. No libraries; plain ES5 plus fetch and Promise (Safari 15+).
 *
 * Every element with data-nvda-mark="<strategy>" holds a baked fallback (the figures as of data-baked="YYYY-MM-DD").
 * This script fetches that strategy's JSON (docs/data/, written each weeknight by .github/workflows/nvda-marks.yml),
 * relative to the page and never from the browser's cache, checks it, and only then replaces the element's content.
 * If the fetch fails or the JSON is not what it should be, the baked content stays and one muted line says so.
 *
 * A strategy is one entry in RENDERERS: the file it reads, a check that throws on a malformed file, and a render
 * function that returns the new content. Another strategy is added by adding its entry and its two functions.
 *
 *   <div data-nvda-mark="vol_arb" data-baked="2026-11-16"> ...the baked figures... </div>
 *   <script src="assets/nvda_marks.js" defer></script>
 */
(function () {
  "use strict";

  var STALE_DAYS = 4;            // an open cycle's mark older than this many calendar days is called stale
  var MINUS = "−", JOIN = "⁠";
  var MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                "November", "December"];
  var SVG = "http://www.w3.org/2000/svg";
  var uid = 0;                   // unique ids for each chart's title and description

  var RENDERERS = {
    vol_arb: {src: "data/vol_arb_nvda_mark.json", check: checkVolArb, render: renderVolArb}
  };

  var CSS = [
    ".nm{font-family:var(--font,system-ui,-apple-system,sans-serif);color:var(--text,#1C2430);min-width:0}",
    ".nm *{box-sizing:border-box}",
    ".nm-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 12px;margin:0 0 14px}",
    ".nm-title{margin:0;font:600 18px/1.3 var(--serif,Georgia,serif);color:var(--heading,#16263F)}",
    ".nm-asof{font-size:13.5px;color:var(--muted,#4E5868)}",
    ".nm-badge{display:inline-flex;align-items:center;padding:2px 9px;border-radius:999px;font:600 11px/1.6 " +
      "var(--font,system-ui,sans-serif);letter-spacing:.04em;text-transform:uppercase;border:1px solid}",
    ".nm-badge.nm-settled{color:var(--accent,#2F5D8A);background:var(--tint,#E8EEF5);border-color:var(--tint-line,#B9CBDD)}",
    ".nm-badge.nm-stale{color:var(--down,#B42318);background:rgba(180,35,24,.06);border-color:rgba(180,35,24,.32)}",
    ".nm-stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:0 0 16px}",
    ".nm-stat{min-width:0;padding:11px 14px;border:1px solid var(--line,#E1E5EB);border-radius:8px;" +
      "background:var(--panel,#fff)}",
    ".nm-k{font-size:11.5px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:var(--faint,#5F6878)}",
    ".nm-v{font:600 22px/1.2 var(--font,system-ui,sans-serif);color:var(--heading,#16263F);margin-top:5px;" +
      "font-variant-numeric:tabular-nums}",
    ".nm-d{font-size:13px;line-height:1.45;color:var(--muted,#4E5868);margin-top:4px}",
    ".nm-tblwrap{overflow-x:auto;max-width:100%;-webkit-overflow-scrolling:touch;border:1px solid " +
      "var(--line-soft,#EBEEF2);border-radius:8px;margin:0 0 16px}",
    ".nm-tbl{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}",
    ".nm-tbl caption{text-align:left;font-size:13px;color:var(--muted,#4E5868);padding:9px 12px 0;caption-side:top}",
    ".nm-tbl th,.nm-tbl td{padding:8px 12px;border-bottom:1px solid var(--line-soft,#EBEEF2);text-align:right;" +
      "vertical-align:top}",
    ".nm-tbl thead th{font-size:11.5px;font-weight:600;letter-spacing:.03em;text-transform:uppercase;" +
      "color:var(--faint,#5F6878);background:var(--panel-hi,#F0F2F5);white-space:nowrap}",
    ".nm-tbl tbody tr:last-child th,.nm-tbl tbody tr:last-child td{border-bottom:none}",
    ".nm-tbl .l{text-align:left}",
    ".nm-tbl tbody th{font-weight:600;color:var(--heading,#16263F);white-space:nowrap}",
    ".nm-tbl .nm-sub{display:block;font-weight:400;font-size:12.5px;color:var(--faint,#5F6878)}",
    ".nm-tbl .nm-who{font-size:13px;color:var(--text,#1C2430);min-width:9.5em}",
    ".nm-tbl .nm-who span{display:block;white-space:nowrap}",
    ".nm-tbl .nm-none{color:var(--faint,#5F6878)}",
    ".nm-pos{color:var(--up,#1E7A46)} .nm-neg{color:var(--down,#B42318)}",
    ".nm-chart{position:relative;margin:0 0 6px;min-width:0}",
    ".nm-legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:13px;color:var(--muted,#4E5868);margin:0 0 4px}",
    ".nm-legend i{display:inline-block;width:14px;height:2px;border-radius:1px;vertical-align:middle;margin-right:6px}",
    ".nm-chart svg{display:block;width:100%;overflow:visible}",
    ".nm-chart text{font-family:var(--font,system-ui,sans-serif)}",
    ".nm-chart text.nm-halo{paint-order:stroke;stroke:var(--panel,#fff);stroke-width:4px;stroke-linejoin:round}",
    ".nm-sr{position:absolute;width:1px;height:1px;margin:-1px;padding:0;overflow:hidden;clip:rect(0 0 0 0);border:0}",
    ".nm-tip{position:absolute;top:0;pointer-events:none;background:var(--panel,#fff);border:1px solid " +
      "var(--line,#E1E5EB);border-radius:6px;box-shadow:0 2px 8px rgba(22,38,63,.12);padding:6px 9px;" +
      "font-size:12.5px;line-height:1.45;color:var(--text,#1C2430);white-space:nowrap;display:none;z-index:2}",
    ".nm-tip b{color:var(--heading,#16263F);font-weight:600}",
    ".nm-cap{font-size:13px;color:var(--muted,#4E5868);line-height:1.5;margin:0 0 12px}",
    ".nm-notes{margin:0;padding:0 0 0 18px;font-size:13.5px;line-height:1.55;color:var(--muted,#4E5868)}",
    ".nm-notes li{margin:0 0 6px}",
    ".nm-sub-h{font-size:11.5px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;" +
      "color:var(--faint,#5F6878);margin:14px 0 6px}",
    ".nm-fail{font-family:var(--font,system-ui,sans-serif);font-size:13px;color:var(--faint,#5F6878);margin:10px 0 0}",
    "@media (max-width:640px){.nm-stats{grid-template-columns:minmax(0,1fr)}.nm-v{font-size:20px}" +
      ".nm-tbl th,.nm-tbl td{padding:7px 9px;font-size:13px}" +
      ".nm-tbl tbody th{white-space:normal;min-width:8.5em}.nm-tbl thead th{white-space:normal}}"
  ].join("\n");

  // ------------------------------------------------------------------ formatting
  function money(x) {
    var n = Math.round(Math.abs(x));
    var s = String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    if (n === 0) { return "$0"; }
    return (x < 0 ? MINUS : "+") + JOIN + "$" + s;
  }
  function price(x) { return "$" + x.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
  function strike(x) { return "$" + (Math.round(x) === x ? String(x) : x.toFixed(2)); }
  function pct(x, digits) { return (x < 0 ? MINUS : "+") + Math.abs(x).toFixed(digits) + "%"; }
  function vol(x) { return (100 * x).toFixed(1) + "%"; }
  // a model's p: three decimals, or five when three would land on the band's edge (0.44996 is not 0.450)
  function prob(p) {
    var t = p.toFixed(3);
    return t === "0.450" || t === "0.550" ? p.toFixed(5) : t;
  }
  function cls(x) { return Math.round(x) > 0 ? "nm-pos" : (Math.round(x) < 0 ? "nm-neg" : ""); }
  function parts(iso) { var p = iso.split("-"); return [+p[0], +p[1], +p[2]]; }
  function dayName(iso) { var p = parts(iso); return MONTHS[p[1] - 1] + " " + p[2]; }
  function shortDay(iso) { var p = parts(iso); return MONTHS[p[1] - 1].slice(0, 3) + " " + p[2]; }
  function dayNumber(iso) { var p = parts(iso); return Date.UTC(p[0], p[1] - 1, p[2]) / 864e5; }
  function todayNumber(now) { return Date.UTC(now.getFullYear(), now.getMonth(), now.getDate()) / 864e5; }
  function isDate(s) { return typeof s === "string" && /^\d{4}-\d{2}-\d{2}$/.test(s); }
  function isNum(x) { return typeof x === "number" && isFinite(x); }

  function h(tag, attrs, kids) {
    var e = document.createElement(tag);
    var k;
    for (k in (attrs || {})) {
      if (Object.prototype.hasOwnProperty.call(attrs, k)) { e.setAttribute(k, attrs[k]); }
    }
    add(e, kids);
    return e;
  }
  function s(tag, attrs, kids) {
    var e = document.createElementNS(SVG, tag);
    var k;
    for (k in (attrs || {})) {
      if (Object.prototype.hasOwnProperty.call(attrs, k)) { e.setAttribute(k, attrs[k]); }
    }
    add(e, kids);
    return e;
  }
  function add(e, kids) {
    if (kids === undefined || kids === null) { return; }
    if (!(kids instanceof Array)) { kids = [kids]; }
    for (var i = 0; i < kids.length; i++) {
      if (kids[i] === null || kids[i] === undefined) { continue; }
      e.appendChild(typeof kids[i] === "string" ? document.createTextNode(kids[i]) : kids[i]);
    }
  }

  // ------------------------------------------------------------------ the vol-arb's file
  var SIDES = ["put_rich", "call_rich"];
  var FILLS = ["mid", "stress"];
  var MODELS = ["logistic_refit", "logistic_fit_once", "forest_refit", "rule_1", "rule_0"];
  var SIDE_NAMES = {put_rich: "Put side rich", call_rich: "Call side rich"};

  function fail(why) { throw new Error("malformed mark: " + why); }

  function checkVolArb(d) {
    if (!d || typeof d !== "object") { fail("not an object"); }
    if (d.schema_version !== 1) { fail("schema_version"); }
    if (d.strategy !== "vol_arb") { fail("strategy"); }
    if (d.status !== "open" && d.status !== "settled") { fail("status"); }
    if (!isDate(d.as_of) || !d.cycle || !isDate(d.cycle.entry) || !isDate(d.cycle.expiry)) { fail("dates"); }
    if (!d.nvda || !isNum(d.nvda.close) || !isNum(d.nvda.entry_close) || !isNum(d.nvda.change_pct)) { fail("nvda"); }
    if (!d.vol || !isNum(d.vol.atm_iv_at_entry) ||
        !(isNum(d.vol.realized_since_entry) || (d.vol.realized_since_entry === null && !d.vol.returns))) {
      fail("vol");
    }
    if (!d.cycle.legs || !d.cycle.legs.put || !d.cycle.legs.call) { fail("legs"); }
    if (!d.cycle.decisions || !d.cycle.decisions.f7_0 || !d.cycle.decisions.f7_1) { fail("decisions"); }
    for (var i = 0; i < SIDES.length; i++) {
      for (var j = 0; j < FILLS.length; j++) {
        var m = d.marks && d.marks[SIDES[i]] && d.marks[SIDES[i]][FILLS[j]];
        if (!m || !isNum(m.pnl) || !(m.daily instanceof Array) || !m.daily.length) { fail("marks"); }
        for (var k = 0; k < m.daily.length; k++) {
          if (!isDate(m.daily[k].date) || !isNum(m.daily[k].pnl_to_date)) { fail("daily row"); }
        }
        if (m.daily[m.daily.length - 1].date !== d.as_of) { fail("the last row is not the mark's"); }
      }
    }
    if (!(d.notes instanceof Array)) { fail("notes"); }
    if (d.status === "settled" && (!d.settlement || !isDate(d.settlement.date))) { fail("settlement"); }
  }

  function chooserList(dec, side) {
    var names = [];
    for (var i = 0; i < MODELS.length; i++) {
      var m = dec[MODELS[i]];
      if (m && m.direction === side) { names.push(MODELS[i]); }
    }
    return names;
  }

  function whoCell(doc, reading, side) {
    var names = chooserList(doc.cycle.decisions[reading], side);
    var td = h("td", {"class": "l nm-who"});
    if (!names.length) { add(td, h("span", {"class": "nm-none"}, "none")); return td; }
    for (var i = 0; i < names.length; i++) {
      var m = doc.cycle.decisions[reading][names[i]];
      var label = (doc.cycle.models && doc.cycle.models[names[i]]) || names[i];
      add(td, h("span", null, label + (isNum(m.p) ? " (p " + prob(m.p) + ")" : "")));
    }
    return td;
  }

  function sideRow(doc, side) {
    var leg = doc.cycle.legs;
    var sub = side === "put_rich"
      ? "sell " + leg.put.contracts + " " + strike(leg.put.strike) + " puts, buy " + leg.call.contracts + " " +
        strike(leg.call.strike) + " calls"
      : "buy " + leg.put.contracts + " " + strike(leg.put.strike) + " puts, sell " + leg.call.contracts + " " +
        strike(leg.call.strike) + " calls";
    var cells = [h("th", {scope: "row", "class": "l"}, [SIDE_NAMES[side], h("span", {"class": "nm-sub"}, sub)])];
    for (var j = 0; j < FILLS.length; j++) {
      var x = doc.marks[side][FILLS[j]].pnl;
      cells.push(h("td", {"class": cls(x)}, money(x)));
    }
    cells.push(whoCell(doc, "f7_0", side));
    cells.push(whoCell(doc, "f7_1", side));
    return h("tr", null, cells);
  }

  function table(doc) {
    var rows = [sideRow(doc, "put_rich"), sideRow(doc, "call_rich")];
    var idle = chooserList(doc.cycle.decisions.f7_0, null).length + chooserList(doc.cycle.decisions.f7_1, null).length;
    if (idle) {
      rows.push(h("tr", null, [h("th", {scope: "row", "class": "l"}, ["No trade",
                                 h("span", {"class": "nm-sub"}, "p inside the 0.45 to 0.55 band")]),
                               h("td", null, "$0"), h("td", null, "$0"),
                               whoCell(doc, "f7_0", null), whoCell(doc, "f7_1", null)]));
    }
    return h("div", null, [h("div", {"class": "nm-sub-h"}, "P&L to date, by the side traded"),
      h("div", {"class": "nm-tblwrap"}, h("table", {"class": "nm-tbl"}, [
      h("caption", {"class": "nm-sr"}, "P&L to date in dollars, by the side traded, and the models that chose each side"),
      h("thead", null, h("tr", null, [h("th", {scope: "col", "class": "l"}, "Side"),
                                      h("th", {scope: "col"}, "Mid"), h("th", {scope: "col"}, "Stress"),
                                      h("th", {scope: "col", "class": "l"}, "Chosen, f7 = 0"),
                                      h("th", {scope: "col", "class": "l"}, "Chosen, f7 = 1")])),
      h("tbody", null, rows)]))]);
  }

  function stat(k, v, d) {
    return h("div", {"class": "nm-stat"}, [h("div", {"class": "nm-k"}, k), h("div", {"class": "nm-v"}, v),
                                           h("div", {"class": "nm-d"}, d)]);
  }

  function stats(doc) {
    var n = doc.nvda, c = doc.cycle;
    var third;
    if (doc.status === "settled") {
      var st = doc.settlement;
      third = stat("Settled", dayName(st.date),
                   "at NVDA's " + price(st.close) + " close: the put worth " + price(st.intrinsic.put) +
                   ", the call " + price(st.intrinsic.call) + "; the shares closed");
    } else {
      var left = dayNumber(c.expiry) - dayNumber(doc.as_of);
      third = stat("Cycle", shortDay(c.entry) + " to " + shortDay(c.expiry),
                   c.legs.put.contracts + " " + strike(c.legs.put.strike) + " puts and " + c.legs.call.contracts +
                   " " + strike(c.legs.call.strike) + " calls; " + left + " day" + (left === 1 ? "" : "s") +
                   " to the expiry");
    }
    var changeDollars = n.close - n.entry_close;
    return h("div", {"class": "nm-stats"}, [
      stat("NVDA close", price(n.close),
           pct(n.change_pct, 1) + " (" + (changeDollars < 0 ? MINUS : "+") + JOIN + price(Math.abs(changeDollars)) +
           ") since the " + dayName(c.entry) + " entry at " + price(n.entry_close)),
      isNum(doc.vol.realized_since_entry)
        ? stat("Realized vol since entry", vol(doc.vol.realized_since_entry),
               "against " + vol(doc.vol.atm_iv_at_entry) + " implied at entry (at the money)")
        : stat("Implied vol at entry", vol(doc.vol.atm_iv_at_entry),
               "at the money; realized vol starts with the next close"),
      third]);
  }

  // ---- the chart: daily P&L to date at mid, both sides, a zero line
  function niceStep(span, n) {
    var raw = span / n, mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10)), f = raw / mag;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * mag;
  }
  function kLabel(v) {
    if (v === 0) { return "$0"; }
    var a = Math.abs(v), t = a >= 1000 ? (a / 1000).toFixed(a % 1000 ? 1 : 0) + "k" : String(a);
    return (v < 0 ? MINUS : "+") + "$" + t;
  }

  function drawChart(box, doc) {
    var width = Math.max(260, Math.floor(box.getBoundingClientRect().width || 600));
    var height = width < 480 ? 210 : 240;
    var pad = {l: 50, r: 14, t: 22, b: 26};
    var put = doc.marks.put_rich.mid.daily, call = doc.marks.call_rich.mid.daily;
    var x0 = dayNumber(doc.cycle.entry), x1 = dayNumber(doc.cycle.expiry);
    var lo = 0, hi = 0, i;
    for (i = 0; i < put.length; i++) {
      lo = Math.min(lo, put[i].pnl_to_date, call[i].pnl_to_date);
      hi = Math.max(hi, put[i].pnl_to_date, call[i].pnl_to_date);
    }
    if (hi - lo < 1) { hi += 100; lo -= 100; }
    var step = niceStep(hi - lo, 4);
    lo = Math.floor(lo / step) * step;
    hi = Math.ceil(hi / step) * step;
    function X(iso) { return pad.l + (dayNumber(iso) - x0) / (x1 - x0) * (width - pad.l - pad.r); }
    function Y(v) { return pad.t + (hi - v) / (hi - lo) * (height - pad.t - pad.b); }

    var id = "nm-svg-" + (++uid);
    var svg = s("svg", {viewBox: "0 0 " + width + " " + height, width: width, height: height, role: "img",
                        "aria-labelledby": id + "-t " + id + "-d"});
    add(svg, s("title", {id: id + "-t"}, "Daily P&L to date at mid, both sides"));
    add(svg, s("desc", {id: id + "-d"},
               "From the " + dayName(doc.cycle.entry) + " entry to the " + dayName(doc.as_of) + " close: put side " +
               "rich " + money(doc.marks.put_rich.mid.pnl) + ", call side rich " + money(doc.marks.call_rich.mid.pnl) +
               ". The table above has the figures."));
    for (var v = lo; v <= hi + step / 2; v += step) {
      var y = Y(v);
      add(svg, s("line", {x1: pad.l, x2: width - pad.r, y1: y, y2: y,
                          style: "stroke:" + (v === 0 ? "var(--zero,#A4ABB7)" : "var(--line-soft,#EBEEF2)"),
                          "stroke-width": v === 0 ? 1.25 : 1}));
      add(svg, s("text", {x: pad.l - 8, y: y + 4, "text-anchor": "end", "font-size": 11,
                          style: "fill:var(--faint,#5F6878)"}, kLabel(Math.round(v))));
    }
    // the expiry and any event inside the cycle, as thin dashed rules
    var marks = [{date: doc.cycle.expiry, label: "expiry"}];
    for (i = 0; i < (doc.events || []).length; i++) {
      var e = doc.events[i];
      if (isDate(e.date) && e.date > doc.cycle.entry && e.date < doc.cycle.expiry) {
        marks.push({date: e.date, label: "report", after: e.timing === "after_close"});
      }
    }
    var prevX = null;
    for (i = 0; i < marks.length; i++) {
      var mx = X(marks[i].date) + (marks[i].after ? 0.5 * (width - pad.l - pad.r) / (x1 - x0) : 0);
      add(svg, s("line", {x1: mx, x2: mx, y1: pad.t - 4, y2: height - pad.b, style: "stroke:var(--zero,#A4ABB7)",
                          "stroke-width": 1, "stroke-dasharray": "3 3"}));
      // the expiry's label ends at its rule; an event's sits left of its rule, one line lower when it is close
      var near = prevX !== null && prevX - mx < 70;
      add(svg, s("text", {x: i === 0 ? mx : mx - 4, y: near ? pad.t + 8 : pad.t - 8, "text-anchor": "end",
                          "font-size": 11, "class": "nm-halo", style: "fill:var(--faint,#5F6878)"}, marks[i].label));
      if (i === 0) { prevX = mx; }
    }
    // x labels: the entry, the mark, the expiry (the mark's only when it has room)
    var base = height - pad.b + 17;
    add(svg, s("text", {x: pad.l, y: base, "font-size": 11, style: "fill:var(--faint,#5F6878)"}, shortDay(doc.cycle.entry)));
    add(svg, s("text", {x: width - pad.r, y: base, "text-anchor": "end", "font-size": 11,
                        style: "fill:var(--faint,#5F6878)"}, shortDay(doc.cycle.expiry)));
    var ax = X(doc.as_of);
    if (ax - pad.l > 56 && width - pad.r - ax > 56) {
      add(svg, s("text", {x: ax, y: base, "text-anchor": "middle", "font-size": 11, style: "fill:var(--faint,#5F6878)"},
                 shortDay(doc.as_of)));
    }
    var series = [{rows: put, color: "var(--series-1,#2A5FA5)", name: "Put side rich", key: "put_rich"},
                  {rows: call, color: "var(--series-2,#C26B1D)", name: "Call side rich", key: "call_rich"}];
    for (i = 0; i < series.length; i++) {
      var pts = [];
      for (var k = 0; k < series[i].rows.length; k++) {
        pts.push(X(series[i].rows[k].date).toFixed(1) + "," + Y(series[i].rows[k].pnl_to_date).toFixed(1));
      }
      add(svg, s("polyline", {points: pts.join(" "), fill: "none", style: "stroke:" + series[i].color, "stroke-width": 2,
                              "stroke-linejoin": "round", "stroke-linecap": "round"}));
    }
    // the end points, ringed in the surface color, each labeled with its side and value
    var ends = [];
    for (i = 0; i < series.length; i++) {
      var last = series[i].rows[series[i].rows.length - 1];
      ends.push({x: X(last.date), y: Y(last.pnl_to_date), v: last.pnl_to_date, s: series[i]});
    }
    ends.sort(function (a, b) { return a.y - b.y; });
    for (i = 0; i < ends.length; i++) {
      add(svg, s("circle", {cx: ends[i].x, cy: ends[i].y, r: 4, style: "fill:" + ends[i].s.color + ";stroke:var(--panel,#fff)", "stroke-width": 2}));
      var above = i === 0;
      add(svg, s("text", {x: ends[i].x - 8, y: ends[i].y + (above ? -9 : 17), "text-anchor": "end", "font-size": 12,
                          "font-weight": 600, "class": "nm-halo", style: "fill:var(--text,#1C2430)"},
                 ends[i].s.name + " " + money(ends[i].v)));
    }
    // hover: the nearest close's figures
    var tip = h("div", {"class": "nm-tip", role: "presentation"});
    var cross = s("line", {x1: 0, x2: 0, y1: pad.t, y2: height - pad.b, style: "stroke:var(--faint,#5F6878)",
                           "stroke-width": 1, visibility: "hidden"});
    add(svg, cross);
    var hit = s("rect", {x: pad.l, y: 0, width: width - pad.l - pad.r, height: height, fill: "transparent"});
    add(svg, hit);
    function show(evt) {
      var r = svg.getBoundingClientRect(), px = (evt.clientX - r.left) * width / r.width, best = 0, bd = 1e9;
      for (var q = 0; q < put.length; q++) {
        var dd = Math.abs(X(put[q].date) - px);
        if (dd < bd) { bd = dd; best = q; }
      }
      var cx = X(put[best].date);
      cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.setAttribute("visibility", "visible");
      tip.innerHTML = "";
      add(tip, [h("b", null, dayName(put[best].date)), h("br"),
                "Put side rich " + money(put[best].pnl_to_date), h("br"),
                "Call side rich " + money(call[best].pnl_to_date), h("br"),
                "NVDA " + price(put[best].close)]);
      tip.style.display = "block";
      var left = cx * r.width / width + 12;
      if (left + tip.offsetWidth > r.width) { left = cx * r.width / width - tip.offsetWidth - 12; }
      tip.style.left = Math.max(0, left) + "px";
      tip.style.top = (pad.t + 4) + "px";
    }
    function hide() { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); }
    hit.addEventListener("mousemove", show);
    hit.addEventListener("mouseleave", hide);
    hit.addEventListener("touchstart", function (e) { if (e.touches && e.touches[0]) { show(e.touches[0]); } },
                         {passive: true});
    box.innerHTML = "";
    add(box, [svg, tip]);
    box.setAttribute("data-width", String(width));
  }

  function chart(doc) {
    var legend = h("div", {"class": "nm-legend", "aria-hidden": "true"}, [
      h("span", null, [h("i", {style: "background:var(--series-1,#2A5FA5)"}), "Put side rich"]),
      h("span", null, [h("i", {style: "background:var(--series-2,#C26B1D)"}), "Call side rich"])]);
    var box = h("div", {"class": "nm-chart"});
    var wrap = h("div", null, [h("div", {"class": "nm-sub-h"}, "P&L to date at mid, each close"), legend, box]);
    wrap.nmDraw = function () {
      try {
        drawChart(box, doc);
      } catch (e) {                  // the table above has every figure: leave the chart out rather than break
        box.innerHTML = "";
      }
    };
    wrap.nmBox = box;
    return wrap;
  }

  function notes(doc) {
    if (!doc.notes.length) { return null; }
    var ul = h("ul", {"class": "nm-notes"});
    for (var i = 0; i < doc.notes.length; i++) { add(ul, h("li", null, String(doc.notes[i]))); }
    return h("div", null, [h("div", {"class": "nm-sub-h"}, "Data notes"), ul]);
  }

  function renderVolArb(doc, now) {
    var badges = [];
    if (doc.status === "settled") {
      badges.push(h("span", {"class": "nm-badge nm-settled"}, "Settled at expiry"));
    } else if (todayNumber(now) - dayNumber(doc.as_of) > STALE_DAYS) {
      badges.push(h("span", {"class": "nm-badge nm-stale"}, "stale: last mark " + dayName(doc.as_of)));
    }
    var c = chart(doc);
    var root = h("div", {"class": "nm", "data-as-of": doc.as_of, "data-status": doc.status}, [
      h("div", {"class": "nm-head"}, [h("h3", {"class": "nm-title"}, "Latest mark, updated nightly"),
                                      h("span", {"class": "nm-asof"}, "as of the " + dayName(doc.as_of) + " close")]
        .concat(badges)),
      stats(doc),
      table(doc),
      c,
      h("p", {"class": "nm-cap"},
        "Both sides are marked whichever a model chose. Mid: " + doc.fills.mid + ". Stress: " + doc.fills.stress +
        ". The hedge is rebalanced at each open to the previous close's delta; options are marked at each close's " +
        "mid."),
      notes(doc)]);
    root.nmChart = c;
    return root;
  }

  // ------------------------------------------------------------------ loading
  function injectStyle() {
    if (document.getElementById("nm-style")) { return; }
    var st = document.createElement("style");
    st.id = "nm-style";
    st.appendChild(document.createTextNode(CSS));
    (document.head || document.documentElement).appendChild(st);
  }

  function fallback(el) {
    if (el.querySelector(".nm-fail")) { return; }
    var baked = el.getAttribute("data-baked");
    add(el, h("p", {"class": "nm-fail"},
              "Couldn't load tonight's mark; showing the figures as of " +
              (isDate(baked) ? dayName(baked) : "the last build") + "."));
    el.setAttribute("data-nm-state", "fallback");
  }

  function load(el) {
    var r = RENDERERS[el.getAttribute("data-nvda-mark")];
    if (!r) { return; }
    var done = function (doc) {
      r.check(doc);
      var node = r.render(doc, new Date());     // built whole before the baked content is touched
      el.innerHTML = "";
      el.appendChild(node);
      el.setAttribute("data-nm-state", "live");
      if (node.nmChart) {
        node.nmChart.nmDraw();
        var timer = null;
        window.addEventListener("resize", function () {
          clearTimeout(timer);
          timer = setTimeout(function () {
            var w = String(Math.max(260, Math.floor(node.nmChart.nmBox.getBoundingClientRect().width || 600)));
            if (node.nmChart.nmBox.getAttribute("data-width") !== w) { node.nmChart.nmDraw(); }
          }, 150);
        });
      }
    };
    try {
      fetch(r.src, {cache: "no-store"})
        .then(function (res) { if (!res.ok) { throw new Error("HTTP " + res.status); } return res.json(); })
        .then(done)
        .then(null, function () { fallback(el); });
    } catch (e) {
      fallback(el);
    }
  }

  function init() {
    injectStyle();
    var els = document.querySelectorAll("[data-nvda-mark]");
    for (var i = 0; i < els.length; i++) { load(els[i]); }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
