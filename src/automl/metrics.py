"""Forecast error metrics, all in rupees per quintal so they stay interpretable."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean absolute error.

    Args:
        y_true: Observed prices.
        y_pred: Predicted prices.

    Returns:
        The average absolute gap, in the same unit as the inputs.
    """
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Root mean squared error.

    Args:
        y_true: Observed prices.
        y_pred: Predicted prices.

    Returns:
        The square root of the average squared gap. Punishes large misses harder
        than :func:`mae` does.
    """
    difference = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(difference**2)))


def mape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean absolute percentage error, ignoring non-positive observations.

    Args:
        y_true: Observed prices.
        y_pred: Predicted prices.

    Returns:
        The average absolute error as a percentage of the observed price, or
        ``nan`` if no observation is positive.
    """
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    usable = truth > 0
    if not usable.any():
        return float("nan")
    return float(np.mean(np.abs((truth[usable] - prediction[usable]) / truth[usable])) * 100.0)


def all_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    """Compute every metric at once.

    Args:
        y_true: Observed prices.
        y_pred: Predicted prices.

    Returns:
        A mapping with ``mae``, ``rmse`` and ``mape`` keys.
    """
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "mape": mape(y_true, y_pred)}
