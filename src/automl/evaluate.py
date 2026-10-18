"""Turn a window plus a hyperparameter configuration into a number the tuner can use.

Two different questions get asked of every window, and keeping them apart is what
makes the comparison fair:

Both are always reported as a price in rupees per quintal, even when the model is
fitted on week-on-week changes, so the numbers stay comparable across target modes.

*Validation error* is what the tuner optimises. The model is fitted on the training
weeks only and scored on the validation weeks. The tuner is allowed to look at this
number as often as its budget permits.

*Test error* is what we report. Once the search is over, the winning configuration is
refitted on training **and** validation weeks together -- the data a real deployment
would have by then -- and scored once on the test weeks, which no part of the search
ever saw.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

from automl.features import TARGET_COLUMN, split_xy, to_price_level
from automl.metrics import all_metrics
from automl.models.lgbm import fit_predict
from automl.split import Window, slice_window

LOGGER = logging.getLogger(__name__)


def make_objective(
    features: pd.DataFrame, window: Window, seed: int
) -> Callable[[dict[str, Any]], float]:
    """Build the validation-error function the tuner minimises.

    Args:
        features: The full feature frame for every market and week.
        window: The window being tuned.
        seed: Random seed passed to the model.

    Returns:
        A callable mapping hyperparameters to validation MAE in rupees per quintal.
    """
    train = slice_window(features, window, "train")
    valid = slice_window(features, window, "valid")
    x_train, y_train = split_xy(train)
    x_valid, y_valid = split_xy(valid)

    def objective(params: dict[str, Any]) -> float:
        raw = fit_predict(params, x_train, y_train, x_valid, seed)
        prices = to_price_level(valid, raw)
        return all_metrics(valid[TARGET_COLUMN], prices)["mae"]

    return objective


def evaluate_on_test(
    features: pd.DataFrame, window: Window, params: dict[str, Any], seed: int
) -> dict[str, float]:
    """Refit on train plus validation and score once on the untouched test weeks.

    Args:
        features: The full feature frame.
        window: The window being evaluated.
        params: The chosen hyperparameters.
        seed: Random seed passed to the model.

    Returns:
        A mapping with ``mae``, ``rmse`` and ``mape`` on the test block.
    """
    train = slice_window(features, window, "train")
    valid = slice_window(features, window, "valid")
    test = slice_window(features, window, "test")
    fitted = pd.concat([train, valid], ignore_index=True)
    x_fit, y_fit = split_xy(fitted)
    x_test, _ = split_xy(test)
    raw = fit_predict(params, x_fit, y_fit, x_test, seed)
    return all_metrics(test[TARGET_COLUMN], to_price_level(test, raw))


def evaluate_frozen_model(
    features: pd.DataFrame,
    train_window: Window,
    test_window: Window,
    params: dict[str, Any],
    seed: int,
) -> dict[str, float]:
    """Score a model fitted on an *older* window against a newer window's test weeks.

    This is the "no retrain" strategy: the model that was deployed after
    ``train_window`` is left exactly as it was and simply keeps predicting.

    Args:
        features: The full feature frame.
        train_window: The window whose data the deployed model was fitted on.
        test_window: The newer window supplying the test weeks.
        params: The hyperparameters the deployed model was built with.
        seed: Random seed passed to the model.

    Returns:
        A mapping with ``mae``, ``rmse`` and ``mape`` on ``test_window``'s test block.
    """
    train = slice_window(features, train_window, "train")
    valid = slice_window(features, train_window, "valid")
    fitted = pd.concat([train, valid], ignore_index=True)
    x_fit, y_fit = split_xy(fitted)
    test = slice_window(features, test_window, "test")
    x_test, _ = split_xy(test)
    raw = fit_predict(params, x_fit, y_fit, x_test, seed)
    return all_metrics(test[TARGET_COLUMN], to_price_level(test, raw))
