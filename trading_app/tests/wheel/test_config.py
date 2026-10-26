"""config/wheel.yaml loads, and says what the run script reads from it."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from trading_app.lib import wheel as W

CFG = yaml.safe_load((Path(__file__).resolve().parents[3] / "config" / "wheel.yaml").read_text())


def test_the_grid_is_eighteen_settings_and_contains_the_baseline():
    g = CFG["grid"]
    cells = W.grid_cells(g["put_delta"], g["call_delta"], g["floor"])
    assert len(cells) == 18
    assert CFG["baseline"]["setting"] in cells


def test_the_test_window_starts_after_the_training_window_ends():
    train, test = CFG["windows"]["train"], CFG["windows"]["test"]
    assert pd.Timestamp(str(train["end"])) < pd.Timestamp(str(test["start"]))


def test_the_score_is_a_figure_the_summary_reports():
    assert CFG["selection"]["score"].split()[0].lower() == "sharpe"
