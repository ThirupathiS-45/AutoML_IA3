"""Naive forecasters. If the tuned model cannot beat these, nothing else matters.

Both are computed from columns the feature builder already produced, so they are
scored on exactly the same rows as every other method.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def predict_last_week(frame: pd.DataFrame) -> np.ndarray:
    """Predict that next week's price equals last week's price.

    Args:
        frame: A feature frame containing a ``lag_1`` column.

    Returns:
        The predicted prices.

    Raises:
        KeyError: If ``lag_1`` is absent.
    """
    if "lag_1" not in frame.columns:
        raise KeyError("predict_last_week needs a lag_1 column")
    return np.asarray(frame["lag_1"], dtype=float)


def predict_rolling_mean(frame: pd.DataFrame, window: int = 4) -> np.ndarray:
    """Predict next week's price as the mean of the previous ``window`` weeks.

    Args:
        frame: A feature frame containing the matching ``roll_mean_{window}`` column.
        window: The rolling window length in weeks.

    Returns:
        The predicted prices.

    Raises:
        KeyError: If the rolling-mean column is absent.
    """
    column = f"roll_mean_{window}"
    if column not in frame.columns:
        raise KeyError(f"predict_rolling_mean needs a {column} column")
    return np.asarray(frame[column], dtype=float)
