"""Tests that evaluation windows are strictly time-ordered and never overlap."""

from __future__ import annotations

import pandas as pd
import pytest

from automl.split import make_windows, slice_window, weeks_index, windows_table

from .conftest import make_weekly_panel


def _weeks(n_weeks: int = 160) -> pd.DatetimeIndex:
    return weeks_index(make_weekly_panel(n_weeks=n_weeks, markets=("A",)))


def test_within_each_window_train_precedes_valid_precedes_test(experiment_config):
    for window in make_windows(_weeks(), experiment_config):
        assert window.train_start <= window.train_end
        assert window.train_end < window.valid_start
        assert window.valid_start <= window.valid_end
        assert window.valid_end < window.test_start
        assert window.test_start <= window.test_end


def test_test_blocks_are_disjoint_and_move_forward(experiment_config):
    windows = make_windows(_weeks(), experiment_config)
    for earlier, later in zip(windows[:-1], windows[1:], strict=True):
        assert earlier.test_end < later.test_start
        assert earlier.window_id < later.window_id


def test_training_data_never_includes_a_future_test_week(experiment_config):
    windows = make_windows(_weeks(), experiment_config)
    for window in windows:
        for other in windows:
            if other.window_id >= window.window_id:
                assert window.train_end < other.test_start


def test_blocks_have_the_configured_length(experiment_config):
    weeks = _weeks()
    splits = experiment_config["splits"]
    for window in make_windows(weeks, experiment_config):
        valid = weeks[(weeks >= window.valid_start) & (weeks <= window.valid_end)]
        test = weeks[(weeks >= window.test_start) & (weeks <= window.test_end)]
        assert len(valid) == splits["valid_weeks"]
        assert len(test) == splits["test_weeks"]


def test_expanding_scheme_always_starts_at_the_first_week(experiment_config):
    weeks = _weeks()
    for window in make_windows(weeks, experiment_config):
        assert window.train_start == weeks[0]


def test_rolling_scheme_caps_the_training_length(experiment_config):
    experiment_config["splits"]["scheme"] = "rolling"
    experiment_config["splits"]["train_weeks"] = 60
    for window in make_windows(_weeks(), experiment_config):
        assert window.n_train_weeks <= 60


def test_rolling_scheme_requires_train_weeks(experiment_config):
    experiment_config["splits"]["scheme"] = "rolling"
    experiment_config["splits"]["train_weeks"] = None
    with pytest.raises(ValueError, match="requires splits.train_weeks"):
        make_windows(_weeks(), experiment_config)


def test_short_timeline_is_rejected(experiment_config):
    with pytest.raises(ValueError, match="needs at least"):
        make_windows(_weeks(n_weeks=40), experiment_config)


def test_slice_window_returns_only_in_range_weeks(experiment_config):
    panel = make_weekly_panel(n_weeks=160, markets=("A", "B"))
    window = make_windows(weeks_index(panel), experiment_config)[2]
    for part, (start, end) in {
        "train": (window.train_start, window.train_end),
        "valid": (window.valid_start, window.valid_end),
        "test": (window.test_start, window.test_end),
    }.items():
        block = slice_window(panel, window, part)
        assert not block.empty
        assert block["week_start"].min() >= start
        assert block["week_start"].max() <= end


def test_slice_window_rejects_an_unknown_part(experiment_config):
    panel = make_weekly_panel(n_weeks=160, markets=("A",))
    window = make_windows(weeks_index(panel), experiment_config)[0]
    with pytest.raises(ValueError, match="unknown window part"):
        slice_window(panel, window, "holdout")


def test_windows_table_has_one_row_per_window(experiment_config):
    windows = make_windows(_weeks(), experiment_config)
    table = windows_table(windows)
    assert len(table) == len(windows)
    assert list(table["window_id"]) == [w.window_id for w in windows]
