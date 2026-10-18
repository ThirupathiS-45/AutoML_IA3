"""Tests that the committed YAML configs are valid and mutually consistent."""

from __future__ import annotations

import pytest

from automl.config import load_config, resolve_path
from automl.features import MONTH_TO_SEASON
from automl.tuner.space import SearchSpace


@pytest.fixture
def data_config():
    return load_config("configs/data.yaml")


@pytest.fixture
def features_config():
    return load_config("configs/features.yaml")


@pytest.fixture
def experiment_config():
    return load_config("configs/experiment.yaml")


@pytest.fixture
def search_space_config():
    return load_config("configs/search_space.yaml")


def test_every_config_file_exists():
    for name in ("data", "features", "search_space", "experiment", "generality"):
        assert resolve_path(f"configs/{name}.yaml").is_file()


def test_lags_are_never_shorter_than_the_horizon(features_config):
    horizon = int(features_config["target"]["horizon_weeks"])
    assert min(features_config["lags_weeks"]) >= horizon


def test_season_map_covers_the_whole_year():
    assert sorted(MONTH_TO_SEASON) == list(range(1, 13))


def test_search_space_builds_and_round_trips(search_space_config):
    space = SearchSpace.from_config(search_space_config)
    assert len(space) == len(search_space_config["hyperparameters"])
    midpoint = {d.name: d.decode(0.5) for d in space.dimensions}
    assert space.from_vector(space.to_vector(midpoint)) == midpoint


def test_log_scaled_dimensions_have_positive_lower_bounds(search_space_config):
    for name, spec in search_space_config["hyperparameters"].items():
        if spec.get("log"):
            assert spec["low"] > 0, f"{name} is log-scaled but allows a non-positive low"


def test_experiment_uses_at_least_five_seeds(experiment_config):
    assert len(experiment_config["seeds"]) >= 5


def test_every_strategy_shares_one_trial_budget(experiment_config):
    budget = experiment_config["budget"]
    assert budget["trials"] > 0
    assert 0 < budget["n_initial_random"] <= budget["trials"]


def test_split_layout_is_internally_consistent(experiment_config):
    splits = experiment_config["splits"]
    assert splits["scheme"] in {"expanding", "rolling"}
    assert splits["n_windows"] >= 4
    assert splits["valid_weeks"] > 0 and splits["test_weeks"] > 0
    if splits["scheme"] == "rolling":
        assert splits["train_weeks"] is not None


def test_stop_rule_threshold_is_a_fraction(data_config):
    assert 0 < data_config["max_missing_week_fraction"] < 1


def test_price_bounds_are_ordered(data_config):
    clean = data_config["clean"]
    assert 0 < clean["min_price"] < clean["max_price"]


def test_weekly_frequency_starts_on_monday(data_config):
    assert data_config["aggregate"]["frequency"] == "W-MON"


def test_drift_method_is_one_we_implement(experiment_config):
    assert experiment_config["drift"]["method"] in {"ks", "evidently"}
