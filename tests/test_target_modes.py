"""Tests for the level and delta target representations."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from automl.features import (
    ANCHOR_COLUMN,
    MODEL_TARGET_COLUMN,
    TARGET_COLUMN,
    build_features,
    feature_columns,
    is_delta_frame,
    split_xy,
    to_price_level,
)

from .conftest import make_weekly_panel

BASE = {
    "target": {"column": "modal_price", "horizon_weeks": 1, "mode": "level"},
    "lags_weeks": [1, 2, 4],
    "rolling_mean_windows_weeks": [4],
    "calendar": {"month": True, "season": True},
    "categorical": ["market"],
}


def _config(mode: str) -> dict:
    config = copy.deepcopy(BASE)
    config["target"]["mode"] = mode
    return config


def test_level_mode_fits_on_the_price_itself():
    frame = build_features(make_weekly_panel(n_weeks=120, markets=("A",)), _config("level"))
    assert np.allclose(frame[MODEL_TARGET_COLUMN], frame[TARGET_COLUMN])
    assert not is_delta_frame(frame)


def test_delta_mode_fits_on_the_week_on_week_change():
    frame = build_features(make_weekly_panel(n_weeks=120, markets=("A",)), _config("delta"))
    expected = frame[TARGET_COLUMN] - frame[ANCHOR_COLUMN]
    assert np.allclose(frame[MODEL_TARGET_COLUMN], expected)
    assert is_delta_frame(frame)


def test_delta_predictions_reconstruct_to_a_price():
    frame = build_features(make_weekly_panel(n_weeks=120, markets=("A",)), _config("delta"))
    perfect = frame[MODEL_TARGET_COLUMN].to_numpy(dtype=float)
    assert np.allclose(to_price_level(frame, perfect), frame[TARGET_COLUMN])


def test_level_reconstruction_is_the_identity():
    frame = build_features(make_weekly_panel(n_weeks=120, markets=("A",)), _config("level"))
    raw = np.arange(len(frame), dtype=float)
    assert np.allclose(to_price_level(frame, raw), raw)


def test_a_zero_delta_prediction_reproduces_the_naive_forecast():
    frame = build_features(make_weekly_panel(n_weeks=120, markets=("A",)), _config("delta"))
    assert np.allclose(to_price_level(frame, np.zeros(len(frame))), frame[ANCHOR_COLUMN])


def test_delta_mode_drops_rows_with_no_anchor():
    frame = build_features(make_weekly_panel(n_weeks=120, markets=("A",)), _config("delta"))
    assert frame[ANCHOR_COLUMN].notna().all()
    assert frame[MODEL_TARGET_COLUMN].notna().all()


def test_neither_target_column_leaks_into_the_features():
    for mode in ("level", "delta"):
        frame = build_features(make_weekly_panel(n_weeks=120, markets=("A",)), _config(mode))
        columns = feature_columns(frame)
        assert TARGET_COLUMN not in columns
        assert MODEL_TARGET_COLUMN not in columns


def test_split_xy_returns_the_fitting_label():
    frame = build_features(make_weekly_panel(n_weeks=120, markets=("A",)), _config("delta"))
    _, y = split_xy(frame)
    assert y.name == MODEL_TARGET_COLUMN


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown target.mode"):
        build_features(make_weekly_panel(n_weeks=60, markets=("A",)), _config("logarithmic"))


def test_delta_mode_needs_a_lag_one_feature():
    config = _config("delta")
    config["lags_weeks"] = [2, 4]
    with pytest.raises(ValueError, match="requires a lag_1"):
        build_features(make_weekly_panel(n_weeks=60, markets=("A",)), config)


def test_forecast_rows_are_dated_past_the_last_observation():
    from automl.features import build_forecast_rows

    weekly = make_weekly_panel(n_weeks=120, markets=("A", "B"))
    rows = build_forecast_rows(weekly, _config("delta"))
    assert len(rows) == 2
    assert (rows["week_start"] > weekly["week_start"].max()).all()


def test_forecast_rows_anchor_on_the_last_observed_price():
    from automl.features import ANCHOR_COLUMN, build_forecast_rows

    weekly = make_weekly_panel(n_weeks=120, markets=("A",))
    rows = build_forecast_rows(weekly, _config("delta"))
    last_price = weekly.sort_values("week_start")["modal_price"].iloc[-1]
    assert rows[ANCHOR_COLUMN].iloc[0] == pytest.approx(last_price)


def test_forecast_rows_reconstruct_through_the_recorded_mode():
    from automl.features import ANCHOR_COLUMN, build_forecast_rows

    weekly = make_weekly_panel(n_weeks=120, markets=("A",))
    rows = build_forecast_rows(weekly, _config("delta"))
    # A zero delta must come back as last week's price, even though these rows carry
    # no target column for is_delta_frame to inspect.
    assert to_price_level(rows, np.zeros(len(rows))) == pytest.approx(rows[ANCHOR_COLUMN])


def test_forecast_rows_in_level_mode_are_passed_through():
    from automl.features import build_forecast_rows

    weekly = make_weekly_panel(n_weeks=120, markets=("A",))
    rows = build_forecast_rows(weekly, _config("level"))
    assert to_price_level(rows, np.array([1500.0])) == pytest.approx(np.array([1500.0]))


def test_mode_stamp_is_not_a_model_feature():
    from automl.features import MODE_COLUMN, build_forecast_rows

    weekly = make_weekly_panel(n_weeks=120, markets=("A",))
    for mode in ("level", "delta"):
        rows = build_forecast_rows(weekly, _config(mode))
        assert MODE_COLUMN not in feature_columns(rows)
