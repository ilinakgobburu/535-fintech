"""
nvda_marks.py end to end with LSEG faked (nm_synth.FakeLSEG answers from made-up data as LSEG would after a given
close; no network): the first run writes the same file the library builds from the full data; when D's close, open
or a leg's D row is not there yet it stops with the old file untouched, and the retry marks with what there is and
says so; the retry does nothing once D is marked; a settled cycle is left alone; after the expiry the legs are read
under their expired names; the write is atomic; the same D gives the same file; every request is paced; the readers
ask LSEG what vol_arb_orders.py's readers ask and give the same market.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

import nm_synth as S
from trading_app.lib import nvda_marks as NM
from trading_app.lib import vol_arb_live as VL


def _load(name, path):
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path[:] = saved


O = _load("nvda_marks_script", S.REPO / "nvda_marks.py")
GEN = "2026-11-17T03:30:00Z"
D = pd.Timestamp("2026-11-16")


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg = S.config()
    cfg_path = S.write_config(tmp_path / "nvda_marks.yaml", cfg)
    out = tmp_path / "docs" / "data" / "vol_arb_nvda_mark.json"
    monkeypatch.setattr(O, "ROOT", tmp_path)
    monkeypatch.setattr(O, "CONFIG", cfg_path)
    monkeypatch.setattr(O, "OUT", out)
    monkeypatch.setattr(O, "now_utc", lambda: GEN)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    state = SimpleNamespace(cfg=cfg, out=out, sleeps=[], sessions=[], fake=None, tmp=tmp_path)
    monkeypatch.setattr(O.time, "sleep", lambda s: state.sleeps.append(s))

    def open_lseg():
        s = S.Session()
        state.sessions.append(s)
        return s
    monkeypatch.setattr(O, "open_lseg", open_lseg)

    def go(argv=(), fake=None, now=dt.datetime(2026, 11, 16, 22, 30)):
        state.fake = fake
        monkeypatch.setattr(O, "ld", fake)
        monkeypatch.setattr(O, "now_ny", lambda: now)
        return O.main(list(argv))
    state.go = go
    return state


def night(day: str, hour: int = 22, minute: int = 30) -> dt.datetime:
    return dt.datetime.fromisoformat(day).replace(hour=hour, minute=minute)


def library_doc(as_of, cfg) -> str:
    mk, _ = S.market()
    spec = NM.cycle_spec(cfg)
    return S.dumps(NM.document(NM.mark_open_cycle(mk, spec, as_of), spec, cfg, generated_at=GEN))


def old_file(env) -> bytes:
    env.out.parent.mkdir(parents=True, exist_ok=True)
    old = {"schema_version": 1, "strategy": "vol_arb", "as_of": "2026-11-13", "status": "open",
           "cycle": {"entry": "2026-10-19", "expiry": "2026-11-20"}}
    env.out.write_text(json.dumps(old) + "\n")
    return env.out.read_bytes()


def leftovers(env) -> list:
    return sorted(p.name for p in env.out.parent.iterdir() if p.name != env.out.name) if env.out.parent.exists() else []


# --------------------------------------------------------------------------- the night
def test_the_first_run_writes_the_librarys_mark_of_ds_close(env, monkeypatch):
    gh = env.tmp / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(gh))
    fake = S.FakeLSEG(published=D)
    assert env.go(fake=fake) == 0
    text = env.out.read_text()
    doc = json.loads(text)
    assert S.schema_problems(doc) == [] and doc["as_of"] == "2026-11-16" and doc["status"] == "open"
    assert text == library_doc(D, env.cfg)          # the script's pulls give the same mark as the whole data
    assert gh.read_text() == "as_of=2026-11-16\n"
    assert len(env.sessions) == 1 and env.sessions[0].closed
    # every request paced, nothing asked past D, the stock unadjusted as the live scripts ask
    assert len(fake.calls) == 5 and env.sleeps == [O.PAUSE] * 5 and O.PAUSE >= 0.4
    hist = [c for c in fake.calls if c[0] == "get_history"]
    assert all(c[4] == "2026-11-16" for c in hist)
    stock = next(c for c in hist if c[1] == "NVDA.O")
    assert stock[2] == ("TRDPRC_1", "OPEN_PRC") and stock[5] == ("exchangeCorrection",) and stock[3] == "2026-10-12"
    legs = {c[1] for c in hist if c[2] == ("BID", "ASK")}
    assert legs == {env.cfg["cycle"]["legs"]["P"]["ric"], env.cfg["cycle"]["legs"]["C"]["ric"]}
    assert leftovers(env) == []
    assert oct(env.out.stat().st_mode & 0o777) == "0o644"


def test_the_default_d_is_the_last_close_that_has_passed(env):
    fake = S.FakeLSEG(published="2026-11-13")
    assert env.go(fake=fake, now=dt.datetime(2026, 11, 16, 9, 0)) == 0     # Monday morning: Friday's close
    assert json.loads(env.out.read_text())["as_of"] == "2026-11-13"


@pytest.mark.parametrize("hole", ["close", "open", "put_row", "call_row"])
def test_data_not_final_stops_with_the_old_file_untouched(env, hole):
    before = old_file(env)
    legs = env.cfg["cycle"]["legs"]
    fake = S.FakeLSEG(published=D, no_close=[D] if hole == "close" else (), no_open=[D] if hole == "open" else (),
                      no_row=[(legs["P" if hole == "put_row" else "C"]["ric"], D)] if hole.endswith("row") else ())
    with pytest.raises(SystemExit) as e:
        env.go(fake=fake)
    assert "data not final, nothing marked" in str(e.value.code)
    assert env.out.read_bytes() == before and leftovers(env) == []


def test_lseg_not_having_d_at_all_is_not_final(env):
    before = old_file(env)
    with pytest.raises(SystemExit) as e:
        env.go(fake=S.FakeLSEG(published="2026-11-13"))
    assert "no NVDA close for 2026-11-16" in str(e.value.code) and env.out.read_bytes() == before


def test_the_retry_marks_with_a_missing_leg_quote_carried_and_says_so(env):
    old_file(env)
    put = env.cfg["cycle"]["legs"]["P"]["ric"]
    assert env.go(["--retry"], fake=S.FakeLSEG(published=D, no_row=[(put, D)])) == 0
    doc = json.loads(env.out.read_text())
    assert doc["as_of"] == "2026-11-16" and S.schema_problems(doc) == []
    assert any(n.startswith("Marked by the retry with LSEG's data as it stood: LSEG has no 2026-11-16 quote for the "
                            "put") for n in doc["notes"])
    assert any(n.startswith("The put had no two-sided closing quote on November 16") for n in doc["notes"])
    assert doc["data_complete"] is False
    rows = doc["marks"]["put_rich"]["mid"]["daily"]
    assert rows[-1]["put_mark"] == rows[-2]["put_mark"] and rows[-1]["put_iv"] == rows[-2]["put_iv"]
    # only D's quote is carried: every earlier close is the full data's
    full = json.loads(library_doc(D, env.cfg))["marks"]["put_rich"]["mid"]["daily"]
    assert rows[:-1] == full[:-1] and rows[-1] != full[-1]


def test_the_retry_with_ds_close_still_missing_marks_the_last_complete_close(env, monkeypatch):
    gh = env.tmp / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(gh))
    assert env.go(["--retry"], fake=S.FakeLSEG(published=D, no_close=[D])) == 0
    doc = json.loads(env.out.read_text())
    assert doc["as_of"] == "2026-11-13"
    assert gh.read_text() == "as_of=2026-11-13\n"                      # the commit message names the close marked
    assert "Marked by the retry at the November 13 close: LSEG had no complete November 16 bar for NVDA at the " \
           "retry." in doc["notes"]
    cfg = env.cfg
    assert {k: v for k, v in doc.items() if k != "notes"} == \
        {k: v for k, v in json.loads(library_doc("2026-11-13", cfg)).items() if k != "notes"}


def test_the_retry_does_nothing_once_d_is_marked(env):
    assert env.go(fake=S.FakeLSEG(published=D)) == 0
    before = env.out.read_bytes()
    calls = len(env.sessions)

    def no_lseg():
        raise AssertionError("the retry must not log in")
    O.open_lseg, saved = no_lseg, O.open_lseg
    try:
        assert env.go(["--retry"], fake=None, now=dt.datetime(2026, 11, 17, 2, 30)) == 0
    finally:
        O.open_lseg = saved
    assert env.out.read_bytes() == before and len(env.sessions) == calls


def test_the_retry_marks_when_the_file_is_from_an_earlier_close(env):
    old_file(env)                                                          # as of Nov 13
    assert env.go(["--retry"], fake=S.FakeLSEG(published=D)) == 0
    assert env.out.read_text() == library_doc(D, env.cfg)


def test_a_leg_with_no_bid_is_final_data_carried_with_a_note(env):
    d = pd.Timestamp("2026-11-02")                                         # the put has a row but no bid
    assert env.go(fake=S.FakeLSEG(published=d), now=night("2026-11-02")) == 0
    doc = json.loads(env.out.read_text())
    assert doc["as_of"] == "2026-11-02"
    assert any(n.startswith("The put had no two-sided closing quote on November 2") for n in doc["notes"])
    assert not any(n.startswith("Marked by the retry") for n in doc["notes"])


def test_after_the_expiry_the_cycle_settles_under_the_expired_rics_and_is_then_left_alone(env):
    legs = env.cfg["cycle"]["legs"]
    fake = S.FakeLSEG(published="2026-11-23", renamed=[legs["P"]["ric"], legs["C"]["ric"]])
    assert env.go(fake=fake, now=night("2026-11-23")) == 0
    doc = json.loads(env.out.read_text())
    assert doc["status"] == "settled" and doc["as_of"] == "2026-11-20" and S.schema_problems(doc) == []
    assert env.out.read_text() == library_doc("2026-11-20", env.cfg)
    asked = [c[1] for c in fake.calls if c[0] == "get_history" and c[2] == ("BID", "ASK")]
    assert asked == [legs["P"]["ric"], legs["P"]["ric"] + "^K26", legs["C"]["ric"], legs["C"]["ric"] + "^K26"]
    assert env.sleeps == [O.PAUSE] * len(fake.calls)                     # paced after the refused requests too
    assert all(c[4] == "2026-11-20" for c in fake.calls if c[0] == "get_history")
    before, n = env.out.read_bytes(), len(env.sessions)
    assert env.go(fake=None, now=night("2026-11-24")) == 0                 # settled: no LSEG, nothing changed
    assert env.out.read_bytes() == before and len(env.sessions) == n
    assert env.go(["--dry-run"], fake=S.FakeLSEG(published="2026-11-24", renamed=fake.renamed),
                  now=night("2026-11-24")) == 0                            # a dry run still marks
    assert len(env.sessions) == n + 1 and env.out.read_bytes() == before


def open_file(env, as_of="2026-11-19") -> bytes:
    """The open cycle's file as the night before the expiry wrote it."""
    env.out.parent.mkdir(parents=True, exist_ok=True)
    env.out.write_text(library_doc(as_of, env.cfg))
    return env.out.read_bytes()


@pytest.mark.parametrize("empty_as", ["frame", None])
@pytest.mark.parametrize("retry", [False, True])
def test_on_the_expiry_night_an_empty_answer_for_a_leg_settles_nothing(env, empty_as, retry):
    """The review's case: LSEG lists the put's live RIC but answers with no rows, and knows no expired name."""
    before = open_file(env)
    put = env.cfg["cycle"]["legs"]["P"]["ric"]
    fake = S.FakeLSEG(published="2026-11-20", empty=[put], empty_as=empty_as)
    with pytest.raises(SystemExit) as e:
        env.go(["--retry"] if retry else [], fake=fake, now=night("2026-11-20"))
    assert str(e.value.code) == (f"LSEG gave no quotes for the put ({put}) from 2026-10-19 to 2026-11-19: the legs' "
                                 f"quote history is not complete, so nothing was marked (the retry does not mark "
                                 f"over this).")
    assert env.out.read_bytes() == before and leftovers(env) == []
    asked = [c[1] for c in fake.calls if c[0] == "get_history" and c[2] == ("BID", "ASK")]
    assert asked[:2] == [put, put + "^K26"]                    # the empty answer after the expiry: the expired name


@pytest.mark.parametrize("empty_as", ["frame", None])
def test_after_the_expiry_an_empty_live_answer_reads_the_expired_name(env, empty_as):
    legs = env.cfg["cycle"]["legs"]
    fake = S.FakeLSEG(published="2026-11-23", renamed=[legs["P"]["ric"], legs["C"]["ric"]],
                      empty=[legs["P"]["ric"], legs["C"]["ric"]], empty_as=empty_as)
    assert env.go(fake=fake, now=night("2026-11-23")) == 0
    assert env.out.read_text() == library_doc("2026-11-20", env.cfg)
    asked = [c[1] for c in fake.calls if c[0] == "get_history" and c[2] == ("BID", "ASK")]
    assert asked == [legs["P"]["ric"], legs["P"]["ric"] + "^K26", legs["C"]["ric"], legs["C"]["ric"] + "^K26"]


def test_before_the_expiry_an_empty_answer_is_not_taken_for_the_rename(env):
    before = old_file(env)
    call = env.cfg["cycle"]["legs"]["C"]["ric"]
    fake = S.FakeLSEG(published=D, empty=[call])
    with pytest.raises(SystemExit) as e:
        env.go(fake=fake)
    assert str(e.value.code).startswith(f"LSEG gave no quotes for the call ({call}) from 2026-10-19 to 2026-11-16")
    assert [c[1] for c in fake.calls if c[0] == "get_history" and c[2] == ("BID", "ASK")][1:] == [call]
    assert env.out.read_bytes() == before


@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("cut, why", [
    ((None, "2026-11-12"), "LSEG's last quote for the put ({ric}) is from 2026-11-12, before both 2026-11-18 and "
                           "2026-11-19"),
    (("2026-11-02", None), "LSEG has no quote for the put ({ric}) on the entry close, 2026-10-19: its history starts "
                           "2026-11-02")])
def test_on_the_expiry_night_a_cut_history_settles_nothing(env, retry, cut, why):
    before = open_file(env)
    legs = env.cfg["cycle"]["legs"]
    fake = S.FakeLSEG(published="2026-11-23", renamed=[legs["P"]["ric"], legs["C"]["ric"]],
                      cut={legs["P"]["ric"]: cut})
    with pytest.raises(SystemExit) as e:
        env.go(["--retry"] if retry else [], fake=fake, now=night("2026-11-23"))
    assert str(e.value.code).startswith(why.format(ric=legs["P"]["ric"]) + ": the legs' quote history is not complete")
    assert env.out.read_bytes() == before and leftovers(env) == []


def test_the_retry_does_not_mark_over_an_empty_leg(env):
    """The review's case: the call's live RIC answers with no rows at the 02:30 retry; the old file stays."""
    before = old_file(env)
    call = env.cfg["cycle"]["legs"]["C"]["ric"]
    with pytest.raises(SystemExit) as e:
        env.go(["--retry"], fake=S.FakeLSEG(published=D, empty=[call]), now=dt.datetime(2026, 11, 17, 2, 30))
    assert "LSEG gave no quotes for the call" in str(e.value.code) and "the retry does not mark over this" in \
        str(e.value.code)
    assert env.out.read_bytes() == before and leftovers(env) == []


@pytest.mark.parametrize("retry", [False, True])
def test_a_leg_whose_quotes_stop_two_closes_before_d_is_never_marked(env, retry):
    before = old_file(env)
    put = env.cfg["cycle"]["legs"]["P"]["ric"]
    fake = S.FakeLSEG(published=D, cut={put: (None, "2026-11-12")})
    with pytest.raises(SystemExit) as e:
        env.go(["--retry"] if retry else [], fake=fake, now=dt.datetime(2026, 11, 17, 2, 30))
    assert str(e.value.code).startswith(f"LSEG's last quote for the put ({put}) is from 2026-11-12, before both "
                                        f"2026-11-13 and 2026-11-16")
    assert env.out.read_bytes() == before


def test_the_retry_carries_one_missing_close_and_no_more(env):
    put = env.cfg["cycle"]["legs"]["P"]["ric"]
    before = old_file(env)
    with pytest.raises(SystemExit):                                         # Nov 13 and 16 both missing
        env.go(["--retry"], fake=S.FakeLSEG(published=D, no_row=[(put, D), (put, "2026-11-13")]))
    assert env.out.read_bytes() == before
    assert env.go(["--retry"], fake=S.FakeLSEG(published=D, no_row=[(put, D)])) == 0     # Nov 16 alone
    doc = json.loads(env.out.read_text())
    assert doc["as_of"] == "2026-11-16" and doc["data_complete"] is False


def test_the_retrys_settlement_is_settled_again_by_the_next_run(env):
    """The put's Nov 19 row is missing on the expiry night: the first run stops, the retry settles with it carried
    and data_complete false, the next night's run settles from complete data, and only that file is left alone."""
    before = open_file(env)
    legs = env.cfg["cycle"]["legs"]
    put = legs["P"]["ric"]
    hole = [(put, "2026-11-19")]
    with pytest.raises(SystemExit) as e:
        env.go(fake=S.FakeLSEG(published="2026-11-20", no_row=hole), now=night("2026-11-20"))
    assert str(e.value.code) == (f"LSEG has no 2026-11-19 quote for the put ({put}), the last close before the "
                                 f"expiry: data not final, nothing marked (the retry asks again).")
    assert env.out.read_bytes() == before
    assert env.go(["--retry"], fake=S.FakeLSEG(published="2026-11-20", no_row=hole),
                  now=dt.datetime(2026, 11, 21, 2, 30)) == 0
    doc = json.loads(env.out.read_text())
    assert doc["status"] == "settled" and doc["data_complete"] is False and S.schema_problems(doc) == []
    assert f"Marked by the retry with LSEG's data as it stood: LSEG has no 2026-11-19 quote for the put ({put}), the " \
           f"last close before the expiry (the leg's last mark is carried, as the backtest carries it; a later run " \
           f"settles the cycle again once LSEG has it)." in doc["notes"]
    retried = env.out.read_bytes()
    # the next retry has nothing new to do; the next first run, with the row there now, settles it again
    assert env.go(["--retry"], fake=None, now=dt.datetime(2026, 11, 24, 2, 30)) == 0
    assert env.out.read_bytes() == retried
    fake = S.FakeLSEG(published="2026-11-23", renamed=[put, legs["C"]["ric"]])
    assert env.go(fake=fake, now=night("2026-11-23")) == 0
    assert env.out.read_text() == library_doc("2026-11-20", env.cfg)
    assert json.loads(env.out.read_text())["data_complete"] is True
    final, n = env.out.read_bytes(), len(env.sessions)
    assert env.go(fake=None, now=night("2026-11-24")) == 0                 # complete and settled: left alone
    assert env.go(["--retry"], fake=None, now=dt.datetime(2026, 11, 25, 2, 30)) == 0
    assert env.out.read_bytes() == final and len(env.sessions) == n


def test_a_settled_file_without_the_completeness_flag_is_settled_again(env):
    legs = env.cfg["cycle"]["legs"]
    env.out.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(library_doc("2026-11-20", env.cfg))
    del doc["data_complete"]
    env.out.write_text(json.dumps(doc))
    assert env.go(fake=S.FakeLSEG(published="2026-11-23", renamed=[legs["P"]["ric"], legs["C"]["ric"]]),
                  now=night("2026-11-23")) == 0
    assert env.out.read_text() == library_doc("2026-11-20", env.cfg)


def test_a_leg_lseg_does_not_know_by_either_name_stops(env):
    before = old_file(env)
    put = env.cfg["cycle"]["legs"]["P"]["ric"]
    fake = S.FakeLSEG(published=D)
    fake.frames.pop(put)
    with pytest.raises(SystemExit) as e:
        env.go(fake=fake)
    assert "LSEG knows neither" in str(e.value.code) and env.out.read_bytes() == before


def test_an_lseg_error_stops_with_the_old_file_untouched(env):
    before = old_file(env)
    with pytest.raises(SystemExit) as e:
        env.go(fake=S.FakeLSEG(published=D, broken=[env.cfg["cycle"]["legs"]["C"]["ric"]]))
    assert "LSEG error on" in str(e.value.code) and env.out.read_bytes() == before
    assert env.sessions[-1].closed


# --------------------------------------------------------------------------- the file
def test_a_dry_run_writes_nothing(env, capsys):
    assert env.go(["--dry-run"], fake=S.FakeLSEG(published=D)) == 0
    assert not env.out.exists()
    printed = capsys.readouterr().out
    assert printed.split("would be:\n", 1)[1] == library_doc(D, env.cfg)


@pytest.mark.parametrize("where", ["replace", "chmod", "write"])
def test_a_failed_write_leaves_the_old_file_and_no_temporary_file(env, monkeypatch, where):
    before = old_file(env)

    def boom(*a, **k):
        raise OSError("disk full")
    if where == "write":
        real = O.os.fdopen

        class Bad:
            def __init__(self, f):
                self.f = f

            def __enter__(self):
                return self

            def __exit__(self, *a):
                self.f.close()

            def write(self, text):
                self.f.write(text[:100])
                raise OSError("disk full")
        monkeypatch.setattr(O.os, "fdopen", lambda fd, mode: Bad(real(fd, mode)))
    else:
        monkeypatch.setattr(O.os, where, boom)
    with pytest.raises(OSError):
        env.go(fake=S.FakeLSEG(published=D))
    assert env.out.read_bytes() == before and leftovers(env) == []


def test_the_same_close_gives_the_same_file(env, monkeypatch):
    assert env.go(fake=S.FakeLSEG(published=D)) == 0
    first = env.out.read_text()
    monkeypatch.setattr(O, "now_utc", lambda: "2026-11-17T07:30:00Z")
    assert env.go(fake=S.FakeLSEG(published=D), now=dt.datetime(2026, 11, 17, 2, 30)) == 0
    second = env.out.read_text()
    assert second != first
    assert second.replace("2026-11-17T07:30:00Z", GEN) == first
    assert [a for a, b in zip(first.splitlines(), second.splitlines()) if a != b] == [f' "generated_at": "{GEN}",']
    monkeypatch.setattr(O, "now_utc", lambda: GEN)
    assert env.go(fake=S.FakeLSEG(published=D)) == 0
    assert env.out.read_text() == first


def test_as_of_and_the_calendar(env):
    with pytest.raises(SystemExit) as e:
        env.go(["--as-of", "2026-11-14"], fake=S.FakeLSEG(published=D))
    assert "not an NYSE session" in str(e.value.code)
    with pytest.raises(SystemExit) as e:
        env.go(["--as-of", "2026-10-16"], fake=S.FakeLSEG(published=D))
    assert "before the cycle's entry" in str(e.value.code)
    assert env.go(["--as-of", "2026-11-10"], fake=S.FakeLSEG(published=D)) == 0
    assert json.loads(env.out.read_text())["as_of"] == "2026-11-10"
    assert not env.sessions or all(s.closed for s in env.sessions)


def test_a_broken_config_or_changed_backtest_code_stops_before_lseg(env, monkeypatch):
    bad = S.copy_cfg(env.cfg)
    bad["cycle"]["decisions"]["f7_0"]["logistic_refit"]["direction"] = "put_rich"
    S.write_config(O.CONFIG, bad)
    with pytest.raises(SystemExit) as e:
        env.go(fake=S.FakeLSEG(published=D))
    assert str(e.value.code).startswith("config/nvda_marks.yaml: decisions.f7_0.logistic_refit") and not env.sessions
    S.write_config(O.CONFIG, env.cfg)
    pins = yaml.safe_load(O.PINS.read_text())
    pins["lib"]["sha256"]["trading_app/trading_app/lib/vol_arb.py"] = "0" * 64
    monkeypatch.setattr(O, "PINS", S.write_config(env.tmp / "pins.yaml", pins))
    with pytest.raises(SystemExit) as e:
        env.go(fake=S.FakeLSEG(published=D))
    assert "is not the frozen backtest code" in str(e.value.code) and not env.sessions
    pins = yaml.safe_load((S.REPO / "config" / "vol_arb_live.yaml").read_text())
    name = next(iter(pins["lib"]["functions"]))
    pins["lib"]["functions"][name] = "0" * 64
    S.write_config(O.PINS, pins)
    with pytest.raises(SystemExit) as e:
        env.go(fake=S.FakeLSEG(published=D))
    assert "lib functions are not the recorded ones" in str(e.value.code) and not env.sessions


# --------------------------------------------------------------------------- the readers
def test_the_readers_ask_and_answer_as_vol_arb_orders_readers_do(monkeypatch):
    VO = _load("vol_arb_orders_for_marks", S.REPO / "vol_arb_orders.py")
    fake = S.FakeLSEG(published=D)
    monkeypatch.setattr(O, "ld", fake)
    monkeypatch.setattr(VO, "ld", fake)
    monkeypatch.setattr(O.time, "sleep", lambda s: None)
    day, days = D.date(), 40
    start = day - dt.timedelta(days=days)
    c1, o1 = O.stock_bars("NVDA.O", start, day)
    c2, o2 = VO.stock_bars("NVDA.O", day, days)
    pd.testing.assert_series_equal(c1, c2, check_names=False)
    pd.testing.assert_series_equal(o1, o2, check_names=False)
    pd.testing.assert_series_equal(O.tbill("US3MT=RR", day - dt.timedelta(days=30), day),
                                   VO.tbill("US3MT=RR", day))
    pd.testing.assert_series_equal(O.dividends("NVDA.O", start, day), VO.dividends("NVDA.O", day, days))
    a, b = fake.calls[-2], fake.calls[-1]
    assert a == b                                    # the same dividend request, window and fields
    cfg = S.config()
    leg = NM.cycle_spec(cfg)["legs"]["P"]
    mine = O.leg_quotes("NVDA", leg, "P", S.EXPIRY.date(), start, day)
    theirs = VO.option_quotes(leg["ric"], start, day)
    assert list(mine["date"]) == list(theirs.index)
    assert mine["bid"].tolist() == pytest.approx(theirs["bid"].tolist(), nan_ok=True)
    assert mine["ask"].tolist() == theirs["ask"].tolist()
    m1 = VL.market(c1, o1, O.tbill("US3MT=RR", day - dt.timedelta(days=30), day), O.dividends("NVDA.O", start, day),
                   (), mine)
    m2 = VL.market(c2, o2, VO.tbill("US3MT=RR", day), VO.dividends("NVDA.O", day, days), (), mine)
    pd.testing.assert_frame_equal(m1["stock"], m2["stock"])
    pd.testing.assert_series_equal(m1["tbill"], m2["tbill"])


def test_a_revised_entry_close_is_noted_in_the_file(env):
    fake = S.FakeLSEG(published=D)
    fake.close[S.ENTRY] += 0.25
    assert env.go(fake=fake) == 0
    notes = json.loads(env.out.read_text())["notes"]
    assert sum(n.startswith("LSEG's October 19 NVDA close is now $") for n in notes) == 1


def test_generated_at_is_utc_to_the_second():
    import re
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", O.now_utc())
    assert O.now_ny().tzinfo is None
