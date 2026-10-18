"""Tests that the evaluation harness keeps test data out of the tuning loop."""

from __future__ import annotations

import pandas as pd

from automl.evaluate import evaluate_frozen_model, evaluate_on_test, make_objective
from automl.features import build_features
from automl.models.lgbm import default_params
from automl.split import make_windows, slice_window, weeks_index

from .conftest import make_weekly_panel

FEATURES_CONFIG = {
    "target": {"column": "modal_price", "horizon_weeks": 1},
    "lags_weeks": [1, 2, 4],
    "rolling_mean_windows_weeks": [4],
    "calendar": {"month": True, "season": True},
    "categorical": ["market"],
}


def _setup():
    features = build_features(make_weekly_panel(n_weeks=200, markets=("A", "B")), FEATURES_CONFIG)
    config = {
        "splits": {
            "scheme": "expanding",
            "train_weeks": None,
            "n_windows": 3,
            "valid_weeks": 10,
            "test_weeks": 10,
            "min_train_weeks": 52,
        }
    }
    return features, make_windows(weeks_index(features), config)


def test_objective_returns_a_positive_error():
    features, windows = _setup()
    objective = make_objective(features, windows[0], seed=0)
    score = objective(default_params())
    assert score > 0 and pd.notna(score)


def test_objective_is_deterministic_for_a_fixed_seed():
    features, windows = _setup()
    objective = make_objective(features, windows[0], seed=0)
    assert objective(default_params()) == objective(default_params())


def test_test_metrics_are_reported_for_every_metric():
    features, windows = _setup()
    metrics = evaluate_on_test(features, windows[0], default_params(), seed=0)
    assert set(metrics) == {"mae", "rmse", "mape"}
    assert metrics["rmse"] >= metrics["mae"]


def test_frozen_model_scores_a_later_window():
    features, windows = _setup()
    metrics = evaluate_frozen_model(features, windows[0], windows[2], default_params(), seed=0)
    assert metrics["mae"] > 0


def test_objective_never_sees_the_test_block():
    """Changing only the test block must not move the validation score."""
    features, windows = _setup()
    window = windows[0]
    before = make_objective(features, window, seed=0)(default_params())

    tampered = features.copy()
    mask = (tampered["week_start"] >= window.test_start) & (
        tampered["week_start"] <= window.test_end
    )
    assert mask.any()
    tampered.loc[mask, "target_price"] += 50_000.0
    after = make_objective(tampered, window, seed=0)(default_params())

    assert before == after


def test_test_block_rows_are_disjoint_from_training_rows():
    features, windows = _setup()
    window = windows[1]
    train_weeks = set(slice_window(features, window, "train")["week_start"])
    test_weeks = set(slice_window(features, window, "test")["week_start"])
    assert train_weeks.isdisjoint(test_weeks)
