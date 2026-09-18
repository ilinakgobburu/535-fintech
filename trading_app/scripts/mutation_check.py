"""
Re-introduce each bug the covered-call tests exist to catch, and confirm the
suite fails.

A passing test suite proves nothing on its own. It is entirely possible to
write 35 tests that assert whatever the code already does, and the failure
mode worth worrying about here is not a crash but a PLAUSIBLE WRONG NUMBER --
an assignment credited at the settle instead of the strike, a short call
marked as an asset, an initial requirement charged against a covered call.
Nothing about the page looks broken when one of those is wrong.

So each mutation below is a bug that would produce a believable result, and
the check is that at least one test notices. Run it after touching anything
in lib/covered_call.py:

    python3 scripts/mutation_check.py

The file is restored after every mutation, including on failure.

A WARNING ABOUT BYTECODE, learned the hard way
----------------------------------------------
CPython decides a .pyc is current by comparing the (mtime, size) it recorded
against the source file. Both of this harness's conditions defeat that:

  * every mutation is a SAME-LENGTH edit ("< 2" -> "< 0"), so size never moves;
  * mutate and restore happen milliseconds apart, so at the one-second
    granularity of the recorded mtime they are the same instant.

The result is a .pyc compiled from the MUTATED source that Python considers
valid for the RESTORED source -- so the mutation survives the restore, in
bytecode, invisibly. It cost a debugging session: a build after this script
reported eleven trading weeks instead of ten because `trading_weeks` was still
running a mutant whose source had been correct on disk the whole time.

Three defences, all of them cheap:
  1. the subprocess runs with PYTHONDONTWRITEBYTECODE, so mutated source never
     produces a .pyc at all;
  2. any .pyc for a mutated file is deleted after the restore anyway;
  3. the run ends by asserting the suite passes CLEAN, which catches any
     restore that did not take.

DO NOT RUN THIS CONCURRENTLY WITH THE TEST SUITE
------------------------------------------------
It edits source files in place. Running `pytest` beside it sees whatever
mutation happens to be installed at that instant and fails for reasons that
have nothing to do with the working tree -- which happened, and cost a few
minutes of confusion over two tests insisting a count was a float. A lock file
now refuses the second concurrent run outright.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CC = ROOT / "trading_app" / "lib" / "covered_call.py"
RIC = ROOT / "trading_app" / "lib" / "ric.py"
AN = ROOT / "trading_app" / "lib" / "cc_analysis.py"
F = ROOT / "scripts" / "fetch_hw2.py"
TESTS = ROOT / "tests" / "test_covered_call.py"
TESTS_FETCH = ROOT / "tests" / "test_fetch_shapes.py"
TESTS_PAGE = ROOT / "tests" / "test_page.py"
APP_JS = ROOT / "scripts" / "hw2_app.js"
BUILD = ROOT / "scripts" / "build_hw2.py"
TEMPLATE = ROOT / "scripts" / "hw2_template.html"
README = ROOT.parent / "README.md"

# (file, find, replace, description, which test file must notice)
#
# Naming the test file keeps the run fast and makes the check sharper: it
# asserts not merely that SOMETHING failed, but that the test written for this
# bug is the one that failed.
MUTATIONS = [
    (CC, '"cash_delta": qty * float(strike),',
         '"cash_delta": qty * float(s_t),',
     "assignment credits the settle instead of the strike "
     "(turns a capped strategy into an uncapped one)", TESTS),
    (CC, 'hit = ks[ks >= spot - 1e-9]',
         'hit = ks[ks > spot + 1e-9]',
     "strike rule uses strictly-greater, silently skipping the ATM strike", TESTS),
    (CC, 'itm = s_t > strike',
         'itm = s_t >= strike',
     "the S_T == K tie assigns instead of expiring", TESTS),
    (CC, 'INITIAL_RATE = 0.50',
         'INITIAL_RATE = 0.25',
     "Reg T initial computed at the maintenance rate", TESTS),
    (CC, 'im = INITIAL_RATE * lmv',
         'im = INITIAL_RATE * lmv + 100.0',
     "a covered short call wrongly adds an initial requirement", TESTS),
    (CC, 'option_mv = -SHARES_PER_CONTRACT * opt_mark if call is not None else 0.0',
         'option_mv = SHARES_PER_CONTRACT * opt_mark if call is not None else 0.0',
     "the short call is marked as an asset instead of a liability", TESTS),
    (CC, 'cash += ev["cash_delta"]',
         'cash += ev["cash_delta"] * 1.0001',
     "marks leak into cash, so the ledger drifts from the blotter", TESTS),
    (RIC, 'f"{expiry.strftime(\'%d\')}{expiry.strftime(\'%y\')}"',
          'f"{expiry.day}{expiry.strftime(\'%y\')}"',
     "the handout's unpadded expiry day (loses single-digit Fridays silently)", TESTS),
    (CC, '    return ts, float(stock.at[ts, "TRDPRC_1"]), "last trade"',
         '    return ts, float(stock.at[ts, "HIGH_1"]), "last trade"',
     "settlement read off HIGH_1, which carries bad prints, instead of the last trade", TESTS),
    (CC, 'spot = stock.at[ts, "TRDPRC_1"] if "TRDPRC_1" in stock.columns else np.nan',
         'spot = stock.at[ts, "LOW_1"] if "LOW_1" in stock.columns else np.nan',
     "the stock leg fills at the bar low rather than the print", TESTS),
    (CC, 'sess = list(grp)\n        if len(sess) < 2:',
         'sess = list(grp)\n        if len(sess) < 0:',
     "a one-session week is accepted as a full cycle", TESTS),
    (AN, 'b, a = np.polyfit(x, y, 1)',
         'b, a = np.polyfit(x, y, 1); b = b * 1.05',
     "the OLS slope behind every reported R-squared is biased", TESTS),
    (CC, 'k_star = spot * np.exp(sigma * np.sqrt(T) * ndtri(1.0 - target)',
         'k_star = spot * np.exp(sigma * T * ndtri(1.0 - target)',
     "the implied-vol rule scales by T instead of sqrt(T)", TESTS),
    (CC, 'ndtri(1.0 - target)', 'ndtri(target)',
     "the breach probability is inverted, selling caps the wrong side of spot", TESTS),
    (CC, 'vols = [implied_vol(float(r.mid), float(spot), float(r.strike), T, 1.0, "C")',
         'vols = [implied_vol(float(r.mid), float(spot), float(r.strike), T * 2, 1.0, "C")',
     "implied vol inverted against the wrong horizon", TESTS),
    (AN, 'mask = pd.Series([(r, d) in keys for r, d in zip(opt_h["ric"], opt_h["date"])],',
         'mask = pd.Series([True for r, d in zip(opt_h["ric"], opt_h["date"])],',
     "the bar-size study compares unmatched contracts", TESTS),

    # --- the loaders: bugs that yield zero rows and no error message -------
    (CC, 'lvl = int(np.argmax(scores))', 'lvl = df.columns.nlevels - 1',
     "stock_panel picks the MultiIndex level by position instead of by "
     "membership, dying far from the cause on a (Field, RIC) frame", TESTS),
    (CC, 'ok = panel["bid"].notna() & panel["ask"].notna() & (panel["ask"] >= panel["bid"])',
         'ok = panel["bid"].notna() & panel["ask"].notna()',
     "a crossed quote gets a midpoint, inventing a price from bad data", TESTS),
    (CC, 'if meta is None or meta["cp"] != "C":', 'if meta is None:',
     "puts leak into a calls-only book", TESTS),
    (AN, 'px = px[~px.index.duplicated(keep="last")]', 'px = px',
     "mid_vs_print dies on a stock frame with a repeated timestamp", TESTS),

    # --- the fetcher's response-shape resolution --------------------------
    (F, 'got.columns = pd.MultiIndex.from_tuples([(name, l) for l in labels],',
        'got.columns = pd.MultiIndex.from_tuples([(l, name) for l in labels],',
     "a flat one-RIC response is labelled RIC-side-out, putting field names on "
     "the RIC level -- the bug that reported 26 series out of 20", TESTS_FETCH),
    (F, 'if len(batch) == 1 and set(labels) <= fset:',
        'if len(batch) == 1:',
     "a single-RIC response of unknown shape is relabelled anyway instead of "
     "falling back", TESTS_FETCH),
    (F, 'if lvl0 <= fset and not (lvl0 <= bset):   # (Field, RIC) ordering',
        'if False:   # (Field, RIC) ordering',
     "a (Field, RIC) MultiIndex is never swapped", TESTS_FETCH),

    # --- the formatters the build log and the page share ------------------
    (BUILD, 'return f"{Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP):,}"',
            'return f"{x:,.0f}"',
     "money() rounds halves to even, so the build log disagrees with the "
     "page it just wrote", TESTS_PAGE),
    (BUILD, 'return int(x)          # counts must stay ints; "n = 26361.0" reads as a bug',
            'return round(float(x), 6)',
     "counts serialise as floats and the page reports n = 26361.0", TESTS_PAGE),

    # --- the page itself ---------------------------------------------------
    (APP_JS, 'xanchor: "left", y: 0.985, yanchor: "top" };',
             'xanchor: "left", y: 1.0, yanchor: "bottom" };',
     "chart titles are clipped off the top of the canvas; the charts still "
     "draw, silently unlabelled", TESTS_PAGE),
    (APP_JS, 'name: `Initial (${(M.initial_rate * 100).toFixed(0)}% LMV)`',
             'name: "Initial (50% LMV)"',
     "the margin legend asserts 50% rather than reading the rate it was given",
     TESTS_PAGE),
    (TEMPLATE, '.grid2 > *{min-width:0}', '.grid2 > *{min-width:auto}',
     "tables inside a grid push the whole page sideways below ~420px", TESTS_PAGE),
    (README, 'more than 1% below on **21.4%**', 'more than 1% below on **21.3%**',
     "a README figure drifts from the page by one rounding step", TESTS_PAGE),

    # --- the bisection, and the bar the order is sent on ------------------
    (F, '        h = len(batch) // 2', '        return None',
     "a batch containing one dead RIC discards every live RIC with it",
     TESTS_FETCH),
    (CC, '# -100 shares on every assignment.\n                call = None',
         '# -100 shares on every assignment.\n                shares -= (SHARES_PER_CONTRACT if ev["side"] == ASSIGN else 0)\n                call = None',
     "assignment removes the shares on the ASSIGN row as well as the stock "
     "SELL, taking the book to -100", TESTS),
    (APP_JS, 'r.side === "SELL") ? -r.qty : r.qty;',
             'r.side === "SELL") ? r.qty : r.qty;',
     "the stock delivered against assignment prints as +100, a sale shown as "
     "a purchase", TESTS_PAGE),
    (CC, '"side": ASSIGN, "qty": contracts, "limit": None,\n                "fill": float(strike), "cash_delta": 0.0,',
         '"side": ASSIGN, "qty": contracts, "limit": None,\n                "fill": float(strike), "cash_delta": qty * float(strike),',
     "assignment cash is booked on BOTH rows, double-counting the strike", TESTS),
    (BUILD, '    start_cash = first_combo_cost(stock, options, weeks)',
            '    start_cash = 50_000.0',
     "starting cash goes back to a flat $50,000 and Jun 29's NAV to about $50k",
     TESTS_PAGE),
    (CC, '                        f"{qty} shares are delivered on the next row, "',
         '                        f"{qty} shares are settled, "',
     "the ASSIGN row stops pointing at the stock leg that carries the cash",
     TESTS),
    # --- margin interest and the fill sensitivity --------------------------
    (CC, '        if rate and cash < 0 and ts in accrual_days:',
         '        if rate and ts in accrual_days:',
     "interest is charged on a positive cash balance too", TESTS),
    (CC, '            accrued += -cash * rate * accrual_days[ts] / MARGIN_DAY_COUNT',
         '            accrued += -cash * rate / MARGIN_DAY_COUNT',
     "the weekend accrues one day instead of three", TESTS),
    (CC, '        nav = cash + stock_mv + option_mv - accrued',
         '        nav = cash + stock_mv + option_mv',
     "accrued interest is computed but never reaches NAV", TESTS),
    (AN, '        if margin_rate and cash < 0 and t in accrual_days:',
         '        if False:',
     "the buy-and-hold benchmark borrows interest-free while the book pays", TESTS),
    (BUILD, '                (c["mid"] - c["bid"]) * SHARES_PER_CONTRACT for c in run["cycles"]',
            '                (c["ask"] - c["mid"]) * SHARES_PER_CONTRACT * 0 for c in run["cycles"]',
     "the bid-fill cost reported on the page is not the half-spread given up",
     TESTS_PAGE),
    # --- the off-by-one found in class: bar stamps and the close -----------
    (CC, '    return keep, end\n', '    return keep, start\n',
     "bars keep LSEG's START stamps, so every price is labelled one period "
     "early and no bar is the close", TESTS),
    (CC, 'keep = np.asarray((end > open_) & (end <= close))',
         'keep = np.asarray((end > open_) & (end <= close + pd.Timedelta(hours=1)))',
     "the after-hours bar is admitted back into the session", TESTS),
    (CC, '    return px.where(~use, official)', '    return px',
     "end-of-day NAV is marked at the last trade instead of the official close "
     "(Jun 29 reads 50,025, not 50,036)", TESTS),
    (CC, '        if pd.notna(oc) and np.isfinite(float(oc)):',
         '        if False:',
     "expiries settle on the last trade before the bell rather than the "
     "closing auction", TESTS),
    (CC, 'return upto[-1] if len(upto) else same_day[0]',
         'return same_day[0]',
     "the order always fills on the first bar of the day regardless of the "
     "hour asked for, silently ignoring the fill-hour parameter", TESTS),
    (APP_JS, '"\u2014" : Number(v).toFixed(4).replace(',
             '"\u2014" : Number(v).toFixed(2).replace(',
     "blotter fills round to the cent, so 100 x fill stops matching the cash column",
     TESTS_PAGE),
]


LOCK = ROOT / ".mutation_check.lock"


class _Lock:
    """
    Refuse to start if another run is live. The harness leaves mutated source
    on disk for the duration of each test run, so two of them interleaved, or
    one of them beside a plain pytest, produce failures that describe nothing.
    """

    def __enter__(self):
        if LOCK.exists():
            raise SystemExit(
                f"another mutation run appears to be live ({LOCK}).\n"
                f"This harness edits source files in place; two at once will "
                f"corrupt each other.\nIf you are sure nothing is running, "
                f"delete that file.")
        LOCK.write_text(str(os.getpid()), encoding="utf-8")
        return self

    def __exit__(self, *exc):
        LOCK.unlink(missing_ok=True)
        return False


def _purge_pyc(path: pathlib.Path) -> None:
    """Drop any cached bytecode for `path`. See the docstring warning."""
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(f"{path.stem}.*.pyc"):
            pyc.unlink(missing_ok=True)


# A mutation to the page's script or template only becomes visible once the
# page is rebuilt, because the tests read docs/hw2.html -- the artefact, not
# the source. Without this, three of the mutations below would be judged
# against a stale page and would look "caught" or "missed" for the wrong
# reason.
NEEDS_REBUILD = {APP_JS, TEMPLATE}


def _rebuild_page() -> None:
    subprocess.run([sys.executable, str(BUILD)], capture_output=True, text=True,
                   cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def _run_suite(target: pathlib.Path | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    which = [str(target)] if target is not None else [str(ROOT / "tests")]
    return subprocess.run(
        [sys.executable, "-m", "pytest", *which, "-q", "--no-header", "-x"],
        capture_output=True, text=True, cwd=ROOT, env=env)


def main() -> int:
  with _Lock():
      _rebuild_page()                   # start from an artefact that matches source
      baseline = _run_suite()
      if baseline.returncode != 0:
          print("the suite does not pass before any mutation -- fix that first\n")
          print(baseline.stdout[-2000:])
          return 2

      missed = []
      print(f"{len(MUTATIONS)} mutations against the test suite\n")
      for path, old, new, desc, target in MUTATIONS:
          src = path.read_text()
          if old not in src:
              print(f"  STALE   {desc}\n          (pattern no longer in {path.name})")
              missed.append(desc)
              continue
          try:
              path.write_text(src.replace(old, new, 1))
              _purge_pyc(path)
              if path in NEEDS_REBUILD:
                  _rebuild_page()
              r = _run_suite(target)
          finally:
              path.write_text(src)          # always restore, even on Ctrl-C
              _purge_pyc(path)              # and never leave a mutant in bytecode
              if path in NEEDS_REBUILD:
                  _rebuild_page()
          caught = r.returncode != 0
          if not caught:
              missed.append(desc)
          print(f"  {'CAUGHT' if caught else 'MISSED'}  [{target.name}] {desc}")

      print()
      if missed:
          print(f"{len(missed)} mutation(s) NOT caught -- the suite has a hole:")
          for d in missed:
              print(f"  - {d}")
          return 1
      _rebuild_page()
      after = _run_suite()
      if after.returncode != 0:
          print("MUTATIONS RESTORED BADLY -- the suite no longer passes clean:\n")
          print(after.stdout[-2000:])
          return 3

      print(f"all {len(MUTATIONS)} mutations caught, and the suite passes clean afterwards")
      return 0


if __name__ == "__main__":
    raise SystemExit(main())
