"""Tests for saving and loading the production model and its card."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from automl.features import (
    MODE_COLUMN,
    build_features,
    build_forecast_rows,
    feature_columns,
    split_xy,
)
from automl.models.lgbm import default_params, fit
from automl.registry import (
    SNAPSHOT_PATH,
    ModelCard,
    git_commit,
    load_card,
    load_model,
    load_serving_features,
    save_model,
    save_serving_features,
)

from .conftest import make_weekly_panel

FEATURES_CONFIG = {
    "target": {"column": "modal_price", "horizon_weeks": 1, "mode": "delta"},
    "lags_weeks": [1, 2, 4],
    "rolling_mean_windows_weeks": [4],
    "calendar": {"month": True, "season": True},
    "categorical": ["market"],
}


@pytest.fixture
def weekly():
    return make_weekly_panel(n_weeks=140, markets=("A", "B"))


@pytest.fixture
def features(weekly):
    return build_features(weekly, FEATURES_CONFIG)


@pytest.fixture
def forecast_rows(weekly):
    return build_forecast_rows(weekly, FEATURES_CONFIG)


@pytest.fixture
def card():
    return ModelCard(
        version="0.0.1",
        created_utc="2026-01-01T00:00:00+00:00",
        git_commit="deadbee",
        crop="Onion",
        crop_slug="onion",
        entity_level="state",
        markets=["A", "B"],
        train_start="2019-01-07",
        train_end="2021-01-04",
        test_start="2021-01-11",
        test_end="2021-03-29",
        horizon_weeks=1,
        hyperparameters=default_params(),
        metrics={"mae": 100.0, "rmse": 150.0, "mape": 8.0},
        feature_names=["lag_1"],
        n_train_rows=200,
    )


def test_model_round_trips_through_disk(tmp_path, features, card):
    x, y = split_xy(features)
    model = fit(default_params(), x, y, seed=0)
    model_path, card_path = save_model(model, card, tmp_path / "model.pkl", tmp_path / "card.json")
    assert model_path.is_file() and card_path.is_file()

    reloaded = load_model(model_path)
    assert list(reloaded.predict(x.head(5))) == list(model.predict(x.head(5)))


def test_card_is_valid_json_with_the_expected_keys(tmp_path, features, card):
    x, y = split_xy(features)
    model = fit(default_params(), x, y, seed=0)
    _, saved_card = save_model(model, card, tmp_path / "m.pkl", tmp_path / "c.json")
    payload = json.loads(saved_card.read_text())
    for key in ("version", "git_commit", "created_utc", "hyperparameters", "metrics"):
        assert key in payload
    assert load_card(saved_card)["version"] == "0.0.1"


def test_serving_rows_cover_every_market_once(tmp_path, forecast_rows):
    path = save_serving_features(forecast_rows, tmp_path / "serving.pkl")
    reloaded = load_serving_features(path)
    assert len(reloaded) == 2
    assert set(reloaded["market"].astype(str)) == {"A", "B"}


def test_serving_rows_are_for_a_week_that_has_not_happened_yet(
    tmp_path, weekly, features, forecast_rows
):
    """The decisive serving test: forecasting an already-observed week is not a forecast."""
    path = save_serving_features(forecast_rows, tmp_path / "serving.pkl")
    reloaded = load_serving_features(path)
    for market, group in weekly.groupby("market", observed=True):
        last_observed = group["week_start"].max()
        row = reloaded[reloaded["market"].astype(str) == str(market)]
        assert row["week_start"].iloc[0] > last_observed
        assert row["week_start"].iloc[0] > features["week_start"].max()


def test_serving_rows_record_the_target_mode(tmp_path, forecast_rows):
    path = save_serving_features(forecast_rows, tmp_path / "serving.pkl")
    reloaded = load_serving_features(path)
    assert reloaded[MODE_COLUMN].iloc[0] == FEATURES_CONFIG["target"]["mode"]


def test_serving_rows_preserve_the_model_input_columns(tmp_path, features, forecast_rows):
    path = save_serving_features(forecast_rows, tmp_path / "serving.pkl")
    reloaded = load_serving_features(path)
    assert feature_columns(reloaded) == feature_columns(features)


def test_saved_serving_rows_still_feed_the_model(tmp_path, features, forecast_rows):
    """Dtypes are rewritten on save; what matters is that the model still predicts."""
    x, y = split_xy(features)
    model = fit(default_params(), x, y, seed=0)
    path = save_serving_features(forecast_rows, tmp_path / "serving.pkl")
    reloaded = load_serving_features(path)
    predictions = model.predict(reloaded[feature_columns(reloaded)])
    assert len(predictions) == len(reloaded)
    assert np.isfinite(predictions).all()


def test_saved_artifact_carries_no_pyarrow_reference(tmp_path, forecast_rows):
    """The runtime image omits pyarrow, so the pickle must not mention it.

    pandas 3 stores strings PyArrow-backed by default, including in the column Index.
    A pickle that references pyarrow loads fine in development and crashes the
    container, which is exactly the failure this test exists to prevent.
    """
    path = save_serving_features(forecast_rows, tmp_path / "serving.pkl")
    assert b"pyarrow" not in path.read_bytes()


def test_missing_artifacts_raise_a_useful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="automl.registry"):
        load_model(tmp_path / "absent.pkl")
    with pytest.raises(FileNotFoundError, match="automl.registry"):
        load_card(tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError, match="automl.registry"):
        load_serving_features(tmp_path / "absent.pkl")


def test_git_commit_returns_a_string():
    commit = git_commit()
    assert isinstance(commit, str) and commit


def test_crop_paths_are_namespaced_by_slug():
    from automl.registry import card_path, crop_dir, model_path, serving_path

    assert crop_dir("tomato").name == "tomato"
    assert model_path("tomato").parent.name == "tomato"
    assert card_path("tomato").name == "model_card.json"
    assert serving_path("tomato").name == "serving_features.pkl"
    assert model_path("onion") != model_path("tomato")


def test_weekly_snapshot_is_csv_and_pyarrow_free(tmp_path):
    """The container reads this snapshot and has no parquet engine installed."""
    from automl.registry import write_weekly_snapshot

    weekly = make_weekly_panel(n_weeks=30, markets=("A", "B")).assign(crop="Onion")
    path = write_weekly_snapshot(weekly, tmp_path / "snap.csv")
    assert path.suffix == ".csv"
    text = path.read_text()
    assert "pyarrow" not in text
    reloaded = pd.read_csv(path, parse_dates=["week_start"])
    assert set(reloaded.columns) == {"crop", "market", "week_start", "modal_price", "n_days"}
    assert len(reloaded) == len(weekly)


def test_snapshot_default_path_sits_with_the_models():
    assert SNAPSHOT_PATH.startswith("models/")
