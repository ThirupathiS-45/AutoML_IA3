"""Build, fit and predict with the LightGBM regressor that the tuner searches over."""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

#: Parameters held fixed for every trial, so the search only varies what we intend.
FIXED_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "l1",
    "verbosity": -1,
    "n_jobs": 1,
    "deterministic": True,
    "force_row_wise": True,
}


def build_model(params: dict[str, Any], seed: int) -> lgb.LGBMRegressor:
    """Instantiate a LightGBM regressor from a hyperparameter dictionary.

    Args:
        params: Tunable hyperparameters, as produced by :mod:`automl.tuner.space`.
        seed: Random seed, fixed so a repeated run gives an identical model.

    Returns:
        An unfitted regressor.
    """
    merged = {**FIXED_PARAMS, **params, "random_state": seed}
    return lgb.LGBMRegressor(**merged)


def fit_predict(
    params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_eval: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    """Fit on the training block and predict on an evaluation block.

    Args:
        params: Tunable hyperparameters.
        x_train: Training features.
        y_train: Training labels.
        x_eval: Features to predict on.
        seed: Random seed.

    Returns:
        Predicted prices for ``x_eval``.
    """
    model = build_model(params, seed)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_eval), dtype=float)


def fit(
    params: dict[str, Any], x_train: pd.DataFrame, y_train: pd.Series, seed: int
) -> lgb.LGBMRegressor:
    """Fit and return the model itself, for the registry and the web app.

    Args:
        params: Tunable hyperparameters.
        x_train: Training features.
        y_train: Training labels.
        seed: Random seed.

    Returns:
        The fitted regressor.
    """
    model = build_model(params, seed)
    model.fit(x_train, y_train)
    return model


def default_params() -> dict[str, Any]:
    """Return LightGBM's own defaults for the parameters we tune.

    Returns:
        The untuned baseline configuration used in Phase 3.
    """
    return {
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 0.0,
        "n_estimators": 100,
    }
