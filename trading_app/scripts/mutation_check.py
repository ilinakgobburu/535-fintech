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
TESTS = ROOT / "tests" / "test_covered_call.py"

# (file, find, replace, what the bug would look like in production)
MUTATIONS = [
    (CC, '"cash_delta": qty * float(strike),',
         '"cash_delta": qty * float(s_t),',
     "assignment credits the settle instead of the strike "
     "(turns a capped strategy into an uncapped one)"),
    (CC, 'hit = ks[ks >= spot - 1e-9]',
         'hit = ks[ks > spot + 1e-9]',
     "strike rule uses strictly-greater, silently skipping the ATM strike"),
    (CC, 'itm = s_t > strike',
         'itm = s_t >= strike',
     "the S_T == K tie assigns instead of expiring"),
    (CC, 'INITIAL_RATE = 0.50',
         'INITIAL_RATE = 0.25',
     "Reg T initial computed at the maintenance rate"),
    (CC, 'im = INITIAL_RATE * lmv',
         'im = INITIAL_RATE * lmv + 100.0',
     "a covered short call wrongly adds an initial requirement"),
    (CC, 'option_mv = -SHARES_PER_CONTRACT * opt_mark if call is not None else 0.0',
         'option_mv = SHARES_PER_CONTRACT * opt_mark if call is not None else 0.0',
     "the short call is marked as an asset instead of a liability"),
    (CC, 'cash += ev["cash_delta"]',
         'cash += ev["cash_delta"] * 1.0001',
     "marks leak into cash, so the ledger drifts from the blotter"),
    (RIC, 'f"{expiry.strftime(\'%d\')}{expiry.strftime(\'%y\')}"',
          'f"{expiry.day}{expiry.strftime(\'%y\')}"',
     "the handout's unpadded expiry day (loses single-digit Fridays silently)"),
    (CC, 's_t = float(stock.at[ets, "TRDPRC_1"]) if ets is not None else np.nan',
         's_t = float(stock.at[ets, "HIGH_1"]) if ets is not None else np.nan',
     "settlement read off HIGH_1, which carries bad prints, instead of the last trade"),
    (CC, 'spot = stock.at[ts, "TRDPRC_1"] if "TRDPRC_1" in stock.columns else np.nan',
         'spot = stock.at[ts, "LOW_1"] if "LOW_1" in stock.columns else np.nan',
     "the stock leg fills at the bar low rather than the print"),
    (CC, 'sess = list(grp)\n        if len(sess) < 2:',
         'sess = list(grp)\n        if len(sess) < 0:',
     "a one-session week is accepted as a full cycle"),
    (AN, 'b, a = np.polyfit(x, y, 1)',
         'b, a = np.polyfit(x, y, 1); b = b * 1.05',
     "the OLS slope behind every reported R-squared is biased"),
]


def _purge_pyc(path: pathlib.Path) -> None:
    """Drop any cached bytecode for `path`. See the docstring warning."""
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(f"{path.stem}.*.pyc"):
            pyc.unlink(missing_ok=True)


def _run_suite() -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(TESTS), "-q", "--no-header", "-x"],
        capture_output=True, text=True, cwd=ROOT, env=env)


def main() -> int:
    baseline = _run_suite()
    if baseline.returncode != 0:
        print("the suite does not pass before any mutation -- fix that first\n")
        print(baseline.stdout[-2000:])
        return 2

    missed = []
    print(f"{len(MUTATIONS)} mutations against {TESTS.name}\n")
    for path, old, new, desc in MUTATIONS:
        src = path.read_text()
        if old not in src:
            print(f"  STALE   {desc}\n          (pattern no longer in {path.name})")
            missed.append(desc)
            continue
        try:
            path.write_text(src.replace(old, new, 1))
            _purge_pyc(path)
            r = _run_suite()
        finally:
            path.write_text(src)          # always restore, even on Ctrl-C
            _purge_pyc(path)              # and never leave a mutant in bytecode
        caught = r.returncode != 0
        if not caught:
            missed.append(desc)
        print(f"  {'CAUGHT' if caught else 'MISSED'}  {desc}")

    print()
    if missed:
        print(f"{len(missed)} mutation(s) NOT caught -- the suite has a hole:")
        for d in missed:
            print(f"  - {d}")
        return 1
    after = _run_suite()
    if after.returncode != 0:
        print("MUTATIONS RESTORED BADLY -- the suite no longer passes clean:\n")
        print(after.stdout[-2000:])
        return 3

    print(f"all {len(MUTATIONS)} mutations caught, and the suite passes clean afterwards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
